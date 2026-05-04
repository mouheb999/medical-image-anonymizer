"""
Convert a Keras DenseNet-121 CheXNet checkpoint (HDF5, brucechou1983-style)
into a PyTorch state_dict compatible with `torchvision.models.densenet121`.

Usage (as a module):
    from services.pathology._keras_to_torch import convert_keras_to_torch
    convert_keras_to_torch("weights/chexnet.pth.tar", "weights/chexnet_torch.pth")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)

# DenseNet-121 block layout (number of denselayers per denseblock)
_BLOCK_LAYERS = {1: 6, 2: 12, 3: 24, 4: 16}
# Keras convX_blockY maps to PyTorch denseblock(X-1), denselayer(Y)
_KERAS_STAGE_TO_TORCH_BLOCK = {2: 1, 3: 2, 4: 3, 5: 4}


def _get(h5, keras_layer: str, var: str) -> np.ndarray:
    """
    Read a Keras-saved weight. Keras HDF5 layout puts each layer's variables
    under `{layer_name}/{layer_name}/{var}:0`.
    """
    grp = h5[keras_layer][keras_layer]
    key = f"{var}:0"
    if key not in grp:
        # Some saves use the bare variable name
        key = var
    return np.asarray(grp[key])


def _conv(h5, keras_layer: str) -> torch.Tensor:
    # Keras Conv2D kernel: (H, W, in_C, out_C) → PyTorch: (out_C, in_C, H, W)
    w = _get(h5, keras_layer, "kernel")
    return torch.from_numpy(np.transpose(w, (3, 2, 0, 1)).copy()).float()


def _bn(h5, keras_layer: str) -> Dict[str, torch.Tensor]:
    return {
        "weight":       torch.from_numpy(_get(h5, keras_layer, "gamma").copy()).float(),
        "bias":         torch.from_numpy(_get(h5, keras_layer, "beta").copy()).float(),
        "running_mean": torch.from_numpy(_get(h5, keras_layer, "moving_mean").copy()).float(),
        "running_var":  torch.from_numpy(_get(h5, keras_layer, "moving_variance").copy()).float(),
    }


def _dense(h5, keras_layer: str) -> Dict[str, torch.Tensor]:
    # Keras Dense kernel: (in, out) → PyTorch Linear weight: (out, in)
    w = _get(h5, keras_layer, "kernel")
    b = _get(h5, keras_layer, "bias")
    return {
        "weight": torch.from_numpy(np.transpose(w, (1, 0)).copy()).float(),
        "bias":   torch.from_numpy(b.copy()).float(),
    }


def convert_keras_to_torch(keras_path: str | Path, torch_path: str | Path) -> Path:
    """
    Convert a Keras CheXNet HDF5 checkpoint to a PyTorch state_dict (.pth).

    The output state_dict matches `torchvision.models.densenet121(num_classes=14)`
    exactly — i.e., keys like ``features.conv0.weight``, ``classifier.weight``.
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is required. Run: pip install h5py") from e

    keras_path = Path(keras_path)
    torch_path = Path(torch_path)
    state: Dict[str, torch.Tensor] = {}

    with h5py.File(keras_path, "r") as h5:
        # ---- Stem ---------------------------------------------------------
        state["features.conv0.weight"] = _conv(h5, "conv1/conv")
        for k, v in _bn(h5, "conv1/bn").items():
            state[f"features.norm0.{k}"] = v

        # ---- Dense blocks -------------------------------------------------
        for keras_stage, torch_block in _KERAS_STAGE_TO_TORCH_BLOCK.items():
            n_layers = _BLOCK_LAYERS[torch_block]
            for j in range(1, n_layers + 1):
                prefix_k = f"conv{keras_stage}_block{j}"
                prefix_t = f"features.denseblock{torch_block}.denselayer{j}"

                # norm1 (0_bn)
                for k, v in _bn(h5, f"{prefix_k}_0_bn").items():
                    state[f"{prefix_t}.norm1.{k}"] = v
                # conv1 (1_conv)
                state[f"{prefix_t}.conv1.weight"] = _conv(h5, f"{prefix_k}_1_conv")
                # norm2 (1_bn)
                for k, v in _bn(h5, f"{prefix_k}_1_bn").items():
                    state[f"{prefix_t}.norm2.{k}"] = v
                # conv2 (2_conv)
                state[f"{prefix_t}.conv2.weight"] = _conv(h5, f"{prefix_k}_2_conv")

        # ---- Transitions --------------------------------------------------
        # Keras pool{N} (N=2,3,4,5) lives AFTER convN-1 block's completion.
        # torchvision naming: transition{k} (k=1,2,3) = between denseblocks.
        # pool2 → transition1, pool3 → transition2, pool4 → transition3.
        for keras_idx, torch_idx in [(2, 1), (3, 2), (4, 3)]:
            for k, v in _bn(h5, f"pool{keras_idx}_bn").items():
                state[f"features.transition{torch_idx}.norm.{k}"] = v
            state[f"features.transition{torch_idx}.conv.weight"] = _conv(
                h5, f"pool{keras_idx}_conv"
            )

        # ---- Final BN + classifier ---------------------------------------
        for k, v in _bn(h5, "bn").items():
            state[f"features.norm5.{k}"] = v
        cls = _dense(h5, "predictions")
        state["classifier.weight"] = cls["weight"]
        state["classifier.bias"] = cls["bias"]

    # Sanity check against torchvision
    from torchvision.models import densenet121
    ref = densenet121(weights=None, num_classes=14)
    ref_state = ref.state_dict()

    # Fill in BN num_batches_tracked counters (zeros) that newer PyTorch expects
    for k in ref_state:
        if k.endswith("num_batches_tracked") and k not in state:
            state[k] = torch.zeros((), dtype=torch.long)

    missing = set(ref_state.keys()) - set(state.keys())
    extra = set(state.keys()) - set(ref_state.keys())
    if missing:
        raise RuntimeError(f"Converter missing keys: {sorted(missing)[:5]}...")
    if extra:
        logger.warning("Converter produced %d extra keys (ignored)", len(extra))

    # Shape check
    for k, v in ref.state_dict().items():
        if state[k].shape != v.shape:
            raise RuntimeError(
                f"Shape mismatch at {k}: got {state[k].shape}, expected {v.shape}"
            )

    torch_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, torch_path)
    logger.info("Saved PyTorch weights → %s (%d keys)", torch_path, len(state))
    return torch_path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    src = sys.argv[1] if len(sys.argv) > 1 else "weights/chexnet.pth.tar"
    dst = sys.argv[2] if len(sys.argv) > 2 else "weights/chexnet_torch.pth"
    convert_keras_to_torch(src, dst)
    print(f"OK → {dst}")
