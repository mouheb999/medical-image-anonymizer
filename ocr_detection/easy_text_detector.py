"""
easy_text_detector.py
---------------------
EasyOCR-based detector for small text that PaddleOCR misses.
Used as a second pass, border-regions only.
"""

from __future__ import annotations
import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class EasyTextDetector:
    """EasyOCR wrapper for detecting small text in border regions.
    
    Used as a second pass after PaddleOCR to catch small text
    like KV/mA values that PaddleOCR misses at low confidence.
    Only returns detections within the border margin to prevent
    false positives on diagnostic image content.
    
    Parameters
    ----------
    conf_threshold:
        Minimum confidence score [0,1] (default 0.1)
    border_pct:
        Only return detections within this % of any edge (default 0.20)
        
    Raises
    ------
    ImportError
        If easyocr is not installed.
    """

    def __init__(
        self,
        conf_threshold: float = 0.1,
        border_pct: float = 0.20
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(
                f"conf_threshold must be in [0,1], got {conf_threshold}"
            )
        self.conf_threshold = conf_threshold
        self.border_pct = border_pct

        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "easyocr is required. Run: pip install easyocr"
            ) from exc

        logger.debug("Initializing EasyOCR reader (cpu).")
        self._reader: Any = easyocr.Reader(
            ['en'],
            gpu=False,
            verbose=False
        )
        logger.debug("EasyOCR initialized successfully.")

    def detect_text(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect small text in border regions of image.
        
        Parameters
        ----------
        image:
            RGB uint8 numpy array shape (H, W, 3)
            
        Returns
        -------
        list[numpy.ndarray]
            List of (4,2) int32 polygon arrays, border regions only.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be numpy.ndarray, got {type(image)}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"image dtype must be uint8, got {image.dtype}"
            )

        H, W = image.shape[:2]
        margin_x = int(W * self.border_pct)
        margin_y = int(H * self.border_pct)

        try:
            raw_results = self._reader.readtext(image)
        except Exception as exc:
            logger.warning("EasyOCR detection failed: %s", exc)
            return []

        boxes: list[np.ndarray] = []

        for (bbox, text, conf) in raw_results:
            if conf < self.conf_threshold:
                logger.debug(
                    "Dropping detection '%s' conf=%.3f < threshold=%.3f",
                    text, conf, self.conf_threshold
                )
                continue

            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Border filter — only keep edge detections
            is_border = (
                x_min < margin_x or
                x_max > W - margin_x or
                y_min < margin_y or
                y_max > H - margin_y
            )

            if not is_border:
                logger.debug(
                    "Dropping central detection '%s' at (%d,%d)-(%d,%d)",
                    text, x_min, y_min, x_max, y_max
                )
                continue

            polygon = np.array(
                [[x_min, y_min], [x_max, y_min],
                 [x_max, y_max], [x_min, y_max]],
                dtype=np.int32
            )
            boxes.append(polygon)
            logger.info(
                "EasyOCR detected '%s' conf=%.3f at (%d,%d)-(%d,%d)",
                text, conf, x_min, y_min, x_max, y_max
            )

        return boxes
