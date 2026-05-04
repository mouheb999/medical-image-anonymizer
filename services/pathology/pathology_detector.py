"""
Chest X-ray Pathology Detector — TorchXRayVision DenseNet-121.

Uses ``torchxrayvision`` (xrv) — the de-facto standard library for
chest X-ray AI — with the multi-dataset ``densenet121-res224-all``
model (trained on NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest,
OpenI, RSNA, Google, …).

The model returns 18 sigmoid probabilities. Because raw scores are
tiny in absolute terms (typical max ≈ 0.05–0.20), we calibrate them
against the per-class operating thresholds shipped with the model
(``model.op_threshs``) using ``xrv.models.op_norm`` — this maps each
class so that 0.5 corresponds to the calibrated decision boundary.

Inference-only. No training, no data storage, no external API calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------
DEFAULT_MODEL_NAME = "densenet121-res224-all"
DEFAULT_CONFIDENCE_THRESHOLD = 0.6     # on calibrated scores (op_norm)
MAX_PATHOLOGIES = 2                    # keep only top-1 / top-2 for safety
CONFIDENCE_CAP = 0.95                  # never display >95% — avoids overclaiming

# Cache directory for xrv weights (auto-downloaded on first use)
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "weights" / "xrv"

# ---------------------------------------------------------------------
# Rule-based suppression of conflicting / overlapping patterns.
# Key is the "preferred" label; value is the set of labels to drop if
# the preferred one is also present. Ordered by clinical preference.
# ---------------------------------------------------------------------
_CONFLICT_SUPPRESSION: Dict[str, set] = {
    "Lung Opacity": {"Mass", "Nodule", "Lung Lesion"},
    "Infiltration": {"Fibrosis"},
    "Consolidation": {"Atelectasis"},
}


def _severity_label(confidence: float) -> str:
    """Map a calibrated confidence value to a 3-level severity bucket."""
    if confidence > 0.8:
        return "High"
    if confidence > 0.65:
        return "Moderate"
    return "Low"


def _safe_description(name: str, confidence: float) -> str:
    """Produce medically safe, hedged phrasing for a single finding.

    Example: "Pattern possibly consistent with Lung Opacity (low specificity)".
    """
    clean = name.replace("_", " ")
    return f"Pattern possibly consistent with {clean} (low specificity)"


class PathologyDetector:
    """Detect 18 chest X-ray pathologies using TorchXRayVision DenseNet-121.

    The model and its preprocessing pipeline are loaded once in the
    constructor and reused for every call to :meth:`detect`.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_results: int = MAX_PATHOLOGIES,
        device: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        """
        Args:
            confidence_threshold: Minimum **calibrated** score to report
                a pathology (0–1, after op_norm).
            max_results: Maximum number of top pathologies to return.
            device: ``'cpu'`` or ``'cuda'``. Auto-detected if ``None``.
            model_name: Any xrv weight name; defaults to the multi-dataset
                ``densenet121-res224-all`` (most robust for general use).
        """
        try:
            import torchxrayvision as xrv  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "torchxrayvision is required. Install with: "
                "pip install torchxrayvision"
            ) from e

        self.confidence_threshold = float(confidence_threshold)
        self.max_results = int(max_results)
        self.model_name = model_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Point xrv's HTTP cache at our weights/ folder
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        import os
        os.environ.setdefault("TORCH_HOME", str(_WEIGHTS_DIR.parent))

        self.model = self._build_and_load()
        self.pathology_names: List[str] = list(self.model.pathologies)
        # op_threshs is a per-class operating point baked into xrv weights
        self.op_threshs: Optional[torch.Tensor] = getattr(
            self.model, "op_threshs", None
        )

        # XRV's standard preprocessing: center-crop to square + resize 224
        self._xrv_transform = torchvision.transforms.Compose([
            self._XRayCenterCrop(),
            self._XRayResizer(224),
        ])

        logger.info(
            "PathologyDetector ready (xrv %s) on %s — %d pathology classes",
            model_name, self.device, len(self.pathology_names),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> Dict:
        """Return calibrated, filtered, and rule-suppressed predictions.

        Output keys:
            - ``pathologies``: list of dicts
              ``{"name", "confidence", "severity", "description"}``
              sorted by confidence (highest first). Already filtered by
              ``confidence_threshold``, conflict-suppressed, and capped
              at ``max_results`` (default 2).
            - ``summary``: short structured text with primary / secondary
              finding lines (for safe display in the UI / report).
            - ``top_class_idx``: index (in ``self.pathology_names``) of
              the highest-calibrated class — **pass this to Grad-CAM**
              so the heatmap matches the #1 finding shown.
            - ``disclaimer``: mandatory research-only disclaimer string.
        """
        tensor = self._preprocess(image)
        probs_calibrated = self._infer(tensor)
        top_idx = int(np.argmax(probs_calibrated))

        pathologies = self._postprocess(probs_calibrated)
        return {
            "pathologies": pathologies,
            "summary": self._build_summary(pathologies),
            "top_class_idx": top_idx,
            "disclaimer": (
                "This is an AI-assisted analysis intended for research "
                "purposes only. The highlighted region represents model "
                "attention and does not indicate precise anatomical "
                "localization."
            ),
        }

    def detect_raw(self, image: np.ndarray) -> np.ndarray:
        """Return the **raw** sigmoid probability vector (uncalibrated)."""
        tensor = self._preprocess(image)
        with torch.no_grad():
            probs = self.model(tensor).cpu().numpy().flatten()
        return probs

    def get_model(self) -> torch.nn.Module:
        """Expose the underlying torch model (for Grad-CAM)."""
        return self.model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_and_load(self) -> torch.nn.Module:
        """Instantiate xrv DenseNet (auto-downloads weights on first use)."""
        import torchxrayvision as xrv

        logger.info("Loading xrv model '%s'...", self.model_name)
        model = xrv.models.DenseNet(weights=self.model_name)
        model = model.to(self.device).eval()
        logger.info("xrv model loaded — pathologies: %d", len(model.pathologies))
        return model

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert an arbitrary input image to xrv format.

        Steps (matching xrv's training-time pipeline):
          1. RGB/RGBA → grayscale (luminance weights).
          2. Convert to float32 in the original [0, 255] range.
          3. ``xrv.datasets.normalize(image, 255)`` → HU-style [-1024, +1024].
          4. Add channel dim → ``(1, H, W)``.
          5. ``XRayCenterCrop`` + ``XRayResizer(224)``.
          6. Add batch dim → ``(1, 1, 224, 224)``.
        """
        import torchxrayvision as xrv

        # 1) Ensure 2-D grayscale uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]
            # Luminance grayscale — matches the doc's preprocessing exactly
            image = (
                0.299 * image[:, :, 0]
                + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]
            ).astype(np.uint8)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        # 2) float32 (still in 0–255)
        image = image.astype(np.float32)

        # 3) xrv normalization → [-1024, +1024]
        image = xrv.datasets.normalize(image, 255)

        # 4) (H, W) → (1, H, W)
        image = image[None, :, :]

        # 5) center-crop + resize to 224
        image = self._xrv_transform(image)

        # 6) add batch dim
        tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, 1, 224, 224)
        return tensor.to(self.device)

    @torch.no_grad()
    def _infer(self, tensor: torch.Tensor) -> np.ndarray:
        """Forward pass + xrv operating-point calibration.

        xrv's DenseNet already applies sigmoid internally, so ``model(x)``
        returns sigmoid probabilities. We then run ``op_norm`` to map
        them so 0.5 corresponds to the per-class decision boundary.
        """
        import torchxrayvision as xrv

        raw_probs = self.model(tensor)  # (1, 18) sigmoid probs

        if self.op_threshs is not None:
            calibrated = xrv.models.op_norm(raw_probs, self.op_threshs.to(self.device))
        else:
            calibrated = raw_probs

        return calibrated.cpu().numpy().flatten()

    def _postprocess(self, probs: np.ndarray) -> List[Dict]:
        """Build the safe, UI-ready list of findings.

        Pipeline:
          1. Drop empty labels and NaNs.
          2. **Cap** confidence at ``CONFIDENCE_CAP`` (0.95) so we never
             report overconfident-looking scores.
          3. **Threshold** against ``self.confidence_threshold`` (0.6
             by default on the calibrated scale).
          4. Sort descending.
          5. **Suppress conflicting / overlapping labels** using
             ``_CONFLICT_SUPPRESSION`` (e.g. drop *Mass* when
             *Lung Opacity* is also present).
          6. Cap to ``self.max_results`` (default 2).
          7. Attach ``severity`` and hedged ``description`` strings.
        """
        # 1-3: build + filter
        scored: List[Dict] = []
        for i, p in enumerate(probs):
            if i >= len(self.pathology_names):
                break
            name = self.pathology_names[i]
            if not name or np.isnan(p):
                continue
            conf = min(float(p), CONFIDENCE_CAP)
            if conf < self.confidence_threshold:
                continue
            scored.append({"name": name, "confidence": round(conf, 4)})

        # 4: sort
        scored.sort(key=lambda d: d["confidence"], reverse=True)

        # 5: rule-based conflict suppression
        scored = self._suppress_conflicts(scored)

        # 6: cap
        scored = scored[: self.max_results]

        # 7: enrich with severity + safe description
        for d in scored:
            d["severity"] = _severity_label(d["confidence"])
            d["description"] = _safe_description(d["name"], d["confidence"])

        return scored

    @staticmethod
    def _suppress_conflicts(findings: List[Dict]) -> List[Dict]:
        """Drop labels that clinically overlap a higher-ranked finding.

        Rules live in ``_CONFLICT_SUPPRESSION``. If the *preferred*
        label is present, any *conflicting* label below it is removed.
        """
        present = {d["name"] for d in findings}
        to_drop: set = set()
        for preferred, conflicts in _CONFLICT_SUPPRESSION.items():
            if preferred in present:
                to_drop |= conflicts
        return [d for d in findings if d["name"] not in to_drop]

    @staticmethod
    def _build_summary(findings: List[Dict]) -> Dict[str, Optional[str]]:
        """Produce a short primary/secondary summary for safe display."""
        def _fmt(f: Dict) -> str:
            return (
                f"{f['description']} — {f['severity']} confidence "
                f"({f['confidence'] * 100:.0f}%)"
            )

        if not findings:
            return {
                "primary_finding": None,
                "secondary_finding": None,
                "text": "No distinctive patterns detected above threshold.",
            }
        primary = _fmt(findings[0])
        secondary = _fmt(findings[1]) if len(findings) > 1 else None
        text = "Primary finding: " + primary
        if secondary:
            text += "\nSecondary finding: " + secondary
        return {
            "primary_finding": primary,
            "secondary_finding": secondary,
            "text": text,
        }

    # ------------------------------------------------------------------
    # XRV preprocessing transforms (re-exported lazily so import order
    # works even when xrv has not been imported elsewhere yet)
    # ------------------------------------------------------------------

    @property
    def _XRayCenterCrop(self):
        import torchxrayvision as xrv
        return xrv.datasets.XRayCenterCrop

    @property
    def _XRayResizer(self):
        import torchxrayvision as xrv
        return xrv.datasets.XRayResizer
