"""
Chest X-ray Pathology Detection Service.

Uses TorchXRayVision pretrained DenseNet for inference-only
pathology detection with Grad-CAM heatmap generation.
"""

from .pathology_detector import PathologyDetector
from .heatmap import generate_heatmap, extract_bbox_from_heatmap

__all__ = ["PathologyDetector", "generate_heatmap", "extract_bbox_from_heatmap"]
