"""
Grad-CAM Heatmap Generator — tuned for chest X-rays (DenseNet-121).

Improvements over the previous version:
  1. Explicit Grad-CAM target = ``model.features.norm5`` (the canonical
     DenseNet hook point: 7×7×1024 feature map just before the classifier).
     Falls back to the last ``Conv2d`` if that sub-module is absent.
  2. Gaussian smoothing after upsampling for cleaner overlays.
  3. **Lung-region mask** — zeros out activations on image borders so the
     heatmap can no longer "stick" to corners or burned-in text.
  4. Bbox extraction scores contours by **mean heat × area**, discards
     regions below 5 % of image area, and enforces overlap with the
     top-30 % activation pixels.

Pseudo-localization is approximate — NOT a substitute for anatomical
segmentation. The ``note`` field in the output is shown to the user as
a disclaimer.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------
# Heatmap pixels above this value are considered part of the attention region.
_BBOX_THRESHOLD: float = 0.6
# Gaussian blur kernel (odd number). Higher = smoother heatmap.
_GAUSSIAN_KSIZE: int = 31
# Heatmap sharpening cutoff — zero pixels below this after smoothing,
# so only the concentrated attention region remains. Keeps colouring
# tight on the actual finding instead of diffusing across the lungs.
_SHARPEN_CUTOFF: float = 0.55
# Shrink the final bbox inward by this fraction of (w, h) to avoid
# presenting an overly-large region as "the" pathology location.
_BBOX_SHRINK: float = 0.10
# Lung-region bounding fractions (relative to original image size).
# Anything outside this rectangle is zeroed in the heatmap.
_LUNG_MASK_TOP: float = 0.08
_LUNG_MASK_BOTTOM: float = 0.92
_LUNG_MASK_LEFT: float = 0.06
_LUNG_MASK_RIGHT: float = 0.94


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def generate_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.4,
) -> Dict:
    """Produce a Grad-CAM heatmap, overlay, and pseudo-localization bbox.

    Args:
        model: TorchXRayVision / torchvision DenseNet (already on device).
        image_tensor: Preprocessed input tensor, shape ``(1, C, H, W)``.
        original_image: Source image as NumPy array (grayscale or RGB).
        target_class_idx: Index of the pathology class to visualise.
            If ``None``, the class with the highest predicted score is used.
            **Pass the same class the UI shows as "top finding"** so the
            heatmap and the textual ranking agree.
        alpha: Opacity of the heatmap overlay (0 = transparent, 1 = opaque).

    Returns:
        ``{"overlay_image", "raw_heatmap", "bbox", "note"}``
            - ``overlay_image``: RGB uint8 ndarray (heatmap + bbox + label).
            - ``raw_heatmap``: float32 [0, 1] at original resolution.
            - ``bbox``: ``[x, y, w, h]`` or ``None`` if no region passes
              the threshold / minimum-area filter.
            - ``note``: safety disclaimer.
    """
    # 1) Raw 7×7 (or similar) CAM at feature-map resolution
    raw_cam = _compute_gradcam(model, image_tensor, target_class_idx)

    orig_h, orig_w = original_image.shape[:2]

    # 2) Upsample to original resolution
    heatmap = cv2.resize(raw_cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 3) Gaussian smoothing — removes upsampling checkerboard artefacts
    heatmap = cv2.GaussianBlur(heatmap, (_GAUSSIAN_KSIZE, _GAUSSIAN_KSIZE), 0)

    # 4) Lung-region mask — kills border/corner activations
    heatmap = _apply_lung_mask(heatmap)

    # 5) Re-normalise after masking
    hmax = heatmap.max()
    if hmax > 1e-8:
        heatmap = heatmap / hmax
    else:
        heatmap = np.zeros_like(heatmap)

    # 6) **Sharpen** — zero out weak/diffuse activations so only the
    #    concentrated region remains visible in the overlay.
    heatmap = heatmap.copy()
    heatmap[heatmap < _SHARPEN_CUTOFF] = 0.0

    # 7) Build colored overlay from the sharpened map
    overlay = _build_overlay(original_image, heatmap, alpha)

    # 8) Bounding box — pick the contour with the **highest (mean heat × area)**
    #    score, enforce overlap with the top-30 % activation region, then shrink.
    bbox = extract_bbox_from_heatmap(heatmap, threshold=_BBOX_THRESHOLD)

    # 9) Draw bbox + safety label
    if bbox is not None:
        _draw_bbox_with_label(overlay, bbox, orig_w)

    return {
        "overlay_image": overlay.astype(np.uint8),
        "raw_heatmap": heatmap.astype(np.float32),
        "bbox": bbox,
        "note": "Localization is approximate based on model attention",
    }


def extract_bbox_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = _BBOX_THRESHOLD,
) -> Optional[List[int]]:
    """Return ``[x, y, w, h]`` for the tightest high-activation region.

    Selection policy:
        1. Binarise at ``threshold``.
        2. Drop contours whose bounding-box area is below 5 % of the
           total image area.
        3. Score each remaining contour by **mean heatmap value × bbox
           area** — rewards both focused activation and spatial extent.
        4. Ensure the final bbox overlaps the top-30 % highest-
           activation region of the heatmap.
        5. Shrink the winning bbox inward by ``_BBOX_SHRINK`` so we
           don't oversell the localization precision.
    """
    mask = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_h, img_w = heatmap.shape[:2]
    img_area = img_h * img_w
    min_bbox_area = img_area * 0.05

    valid = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_bbox_area:
            valid.append(c)

    if not valid:
        return None

    # Score each contour by mean heat × bbox area
    def _score(c) -> float:
        x, y, w, h = cv2.boundingRect(c)
        region = heatmap[y:y + h, x:x + w]
        mean_heat = float(region.mean()) if region.size else 0.0
        return mean_heat * (w * h)

    best = max(valid, key=_score)
    x, y, w, h = cv2.boundingRect(best)

    # 4) Overlap check with top-30 % activation region
    heatmap_nonzero = heatmap[heatmap > 0]
    if heatmap_nonzero.size > 0:
        top_30_threshold = float(np.percentile(heatmap_nonzero, 70))
        top_30_mask = heatmap >= top_30_threshold
        if not top_30_mask[y:y + h, x:x + w].any():
            return None

    # 5) Shrink inward to avoid overclaiming precision
    dx = int(w * _BBOX_SHRINK / 2)
    dy = int(h * _BBOX_SHRINK / 2)
    x_s, y_s = x + dx, y + dy
    w_s, h_s = max(1, w - 2 * dx), max(1, h - 2 * dy)

    # If shrinking removes all top-activation overlap, revert to original
    if heatmap_nonzero.size > 0:
        if not top_30_mask[y_s:y_s + h_s, x_s:x_s + w_s].any():
            return [int(x), int(y), int(w), int(h)]

    return [int(x_s), int(y_s), int(w_s), int(h_s)]


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------

def _compute_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class_idx: Optional[int],
) -> np.ndarray:
    """Run Grad-CAM on the canonical DenseNet target layer.

    Target = ``model.features.denseblock4`` — its output is the 7×7×1024
    feature map just before the final BatchNorm. We deliberately do NOT
    hook ``norm5`` because xrv's ``DenseNet.forward`` applies
    ``F.relu(..., inplace=True)`` on norm5's output, which breaks
    backward hooks on that layer (PyTorch raises a "custom Function
    gradient override" error). Hooking denseblock4 avoids the in-place
    edge while giving an equivalent 7×7×1024 spatial map.

    Falls back to the last ``Conv2d`` in ``model.features`` if
    denseblock4 is absent (non-DenseNet architectures).
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    target_layer = _get_gradcam_target(model)
    if target_layer is None:
        logger.warning("Could not locate Grad-CAM target layer; returning zeros")
        return np.zeros((7, 7), dtype=np.float32)

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    fwd_handle = target_layer.register_forward_hook(
        lambda _m, _i, out: activations.append(out.detach())
    )
    bwd_handle = target_layer.register_full_backward_hook(
        lambda _m, _gi, go: gradients.append(go[0].detach())
    )

    # enable_grad() because CLIP (loaded elsewhere) globally disables grads
    model.zero_grad()
    with torch.enable_grad():
        inp = image_tensor.clone().requires_grad_(True)
        output = model(inp)  # (1, num_classes)

        if target_class_idx is None:
            target_class_idx = int(output.argmax(dim=1).item())

        score = output[0, target_class_idx]
        score.backward(retain_graph=False)

    fwd_handle.remove()
    bwd_handle.remove()

    if not activations or not gradients:
        logger.warning("Grad-CAM hooks captured nothing; returning zeros")
        return np.zeros((7, 7), dtype=np.float32)

    # Global-average-pool the gradients to get per-channel weights
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
    cam = (weights * activations[0]).sum(dim=1, keepdim=True)    # (1, 1, h, w)
    cam = F.relu(cam).squeeze().cpu().numpy().astype(np.float32)

    # Normalise to [0, 1]
    cam = np.maximum(cam, 0.0)
    cmax = cam.max()
    if cmax > 1e-8:
        cam = cam / cmax
    else:
        cam = np.zeros_like(cam)

    return cam


def _get_gradcam_target(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Return the canonical Grad-CAM layer for DenseNet.

    Prefers ``model.features.denseblock4`` (safe: its output is consumed
    by a non-inplace BatchNorm, unlike ``norm5`` whose output is clobbered
    by xrv's in-place ``F.relu``). Falls back to the last ``Conv2d`` in
    the feature extractor for non-DenseNet architectures.
    """
    features = getattr(model, "features", None)
    if features is not None and hasattr(features, "denseblock4"):
        return features.denseblock4

    module_iter = (features or model).modules()
    last_conv = None
    for m in module_iter:
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    return last_conv


def _apply_lung_mask(heatmap: np.ndarray) -> np.ndarray:
    """Zero out activations outside the central lung region.

    Uses a fixed-fraction bounding box tuned for typical PA/AP chest
    X-ray framing — simple, fast, and avoids heavy segmentation models.
    """
    h, w = heatmap.shape[:2]
    mask = np.zeros_like(heatmap, dtype=np.float32)
    top = int(h * _LUNG_MASK_TOP)
    bot = int(h * _LUNG_MASK_BOTTOM)
    left = int(w * _LUNG_MASK_LEFT)
    right = int(w * _LUNG_MASK_RIGHT)
    mask[top:bot, left:right] = 1.0
    return heatmap * mask


def _build_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend the heatmap (JET colourmap) on top of the source image."""
    background = _to_rgb(original_image)

    heatmap_bgr = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(background, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay.astype(np.uint8)


def _draw_bbox_with_label(overlay: np.ndarray, bbox: List[int], image_width: int) -> None:
    """Draw a green bbox + safety label in-place on the overlay."""
    x, y, w, h = bbox
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = "Model Attention Region (Approximate)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.9, image_width / 800.0))
    thickness = max(1, int(font_scale * 2))

    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    label_y = max(y - 8, th + 4)

    cv2.rectangle(
        overlay,
        (x, label_y - th - 4),
        (x + tw + 6, label_y + 4),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        overlay, label, (x + 3, label_y),
        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA,
    )


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Force an arbitrary input to 3-channel uint8 RGB for overlay."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    return image
