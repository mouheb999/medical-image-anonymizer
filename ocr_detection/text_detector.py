"""
text_detector.py
----------------
Provides :class:`TextDetector`, a thin, production-ready wrapper around
PaddleOCR that:

* Loads the OCR model **once** in the constructor.
* Accepts RGB numpy images.
* Filters detections below a configurable confidence threshold.
* Returns only bounding-box polygons (no raw text).
* Avoids every deprecated PaddleOCR constructor argument.

CRITICAL: Environment variables MUST be set before any paddle import.
"""

from __future__ import annotations

# ============================================================================
# CRITICAL: Disable oneDNN/MKLDNN and PIR API BEFORE any paddle import
# These must be set before paddle or paddleocr are imported anywhere
# ============================================================================
import os
import sys

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TextDetector:
    """OCR model wrapper for bounding-box detection on RGB images.

    The underlying PaddleOCR engine is initialised once in the constructor;
    subsequent calls to :meth:`detect_text` are therefore cheap.

    Parameters
    ----------
    lang:
        Language code forwarded to PaddleOCR (default ``"en"``).
    conf_threshold:
        Minimum confidence score ``[0, 1]`` required to retain a detection
        (default ``0.5``).

    Raises
    ------
    ImportError
        If ``paddleocr`` is not installed in the current environment.
    ValueError
        If *conf_threshold* is not in the range ``[0, 1]``.
    """

    def __init__(self, lang: str = "en", conf_threshold: float = 0.5) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(
                f"conf_threshold must be in [0, 1]; got {conf_threshold!r}"
            )

        self._conf_threshold = conf_threshold

        try:
            from paddleocr import PaddleOCR  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required but not installed.  "
                "Run: pip install paddleocr"
            ) from exc

        logger.debug("Initialising PaddleOCR (lang=%s).", lang)

        # Stable PaddleOCR 2.7.3 initialization (CPU only)
        # Note: use_gpu and show_log are not supported in this version
        self._engine: Any = PaddleOCR(
            lang=lang,
            use_angle_cls=True
        )

        logger.debug("PaddleOCR initialised successfully.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect_text(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect text regions in *image* and return their bounding polygons.

        Parameters
        ----------
        image:
            An RGB image as a ``numpy.ndarray`` with dtype ``uint8`` and
            shape ``(H, W, 3)``.

        Returns
        -------
        list[numpy.ndarray]
            A list of polygon arrays, each with shape ``(4, 2)`` and dtype
            ``int32``, representing the four corner points of one text region
            in pixel coordinates ``(x, y)``.  Regions whose confidence score
            falls below *conf_threshold* are excluded.

        Raises
        ------
        TypeError
            If *image* is not a ``numpy.ndarray``.
        ValueError
            If *image* does not have the expected shape or dtype.
        """
        self._validate_image(image)

        # PaddleOCR expects a BGR uint8 array when fed a numpy array directly.
        bgr_image = image[:, :, ::-1].copy()

        raw_results = self._engine.ocr(bgr_image)

        return self._parse_results(raw_results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Raise an informative error when *image* is not a valid RGB array."""
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a numpy.ndarray; got {type(image).__name__!r}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must have shape (H, W, 3); got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"image dtype must be uint8; got {image.dtype}"
            )

    def _parse_results(self, raw_results: Any) -> list[np.ndarray]:
        """Extract and filter bounding polygons from PaddleOCR output.

        PaddleOCR returns a nested structure::

            [  # one element per page / image
                [  # one element per detected line
                    [polygon, (text, confidence)],
                    ...
                ]
            ]

        Parameters
        ----------
        raw_results:
            The value returned by ``PaddleOCR.ocr``.

        Returns
        -------
        list[numpy.ndarray]
            Filtered list of ``(4, 2)`` int32 polygon arrays.
        """
        boxes: list[np.ndarray] = []

        if not raw_results:
            return boxes

        # Flatten across pages (we only ever pass one image).
        for page in raw_results:
            if page is None:
                continue
            for detection in page:
                if detection is None:
                    continue

                try:
                    polygon_raw, (_, confidence) = detection
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed detection record: %s – %s",
                        detection,
                        exc,
                    )
                    continue

                if confidence < self._conf_threshold:
                    logger.debug(
                        "Dropping detection with confidence %.3f < threshold %.3f.",
                        confidence,
                        self._conf_threshold,
                    )
                    continue

                polygon = np.array(polygon_raw, dtype=np.int32)
                boxes.append(polygon)

        return boxes