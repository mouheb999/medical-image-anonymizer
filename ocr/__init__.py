"""
ocr – lightweight OCR wrappers for medical image anonymization.

Public API
----------
TextDetector : class
    Loads the OCR model once and exposes ``detect_text`` for bounding-box
    extraction on RGB numpy arrays.
BorderPreprocessor : class
    Applies CLAHE contrast enhancement to border strips only,
    improving detection of small faint text without affecting center.
EasyTextDetector : class
    EasyOCR-based detector for small text that PaddleOCR misses.
    Used as a second pass, border-regions only.
"""

from .text_detector import TextDetector
from .preprocessor import BorderPreprocessor
from .easy_text_detector import EasyTextDetector

__all__ = ["TextDetector", "BorderPreprocessor", "EasyTextDetector"]