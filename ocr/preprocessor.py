"""
preprocessor.py
---------------
Border contrast enhancement for OCR preprocessing.
Applies CLAHE only to border strips of the image to improve
detection of small faint text without affecting diagnostic content.
"""

from __future__ import annotations
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BorderPreprocessor:
    """Enhances border regions of medical images for better OCR detection.
    
    Applies CLAHE contrast enhancement ONLY to border strips,
    leaving the diagnostic center of the image untouched.
    
    Parameters
    ----------
    border_pct:
        Percentage of image dimensions to treat as border (default 0.15)
    clip_limit:
        CLAHE clip limit (default 3.0)
    tile_size:
        CLAHE tile grid size (default (4,4))
    """

    def __init__(
        self,
        border_pct: float = 0.15,
        clip_limit: float = 3.0,
        tile_size: tuple[int, int] = (4, 4)
    ) -> None:
        self.border_pct = border_pct
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size
        )
        logger.debug(
            f"BorderPreprocessor initialized: border_pct={border_pct}, "
            f"clip_limit={clip_limit}"
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to border strips only.
        
        Parameters
        ----------
        image:
            RGB or grayscale uint8 numpy array
            
        Returns
        -------
        numpy.ndarray
            Enhanced image, same shape and dtype as input.
            Only border regions are modified.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be numpy.ndarray, got {type(image)}")
        if image.dtype != np.uint8:
            raise ValueError(f"image dtype must be uint8, got {image.dtype}")

        H, W = image.shape[:2]
        bh = int(H * self.border_pct)
        bw = int(W * self.border_pct)

        # Convert to grayscale for CLAHE
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        enhanced_gray = gray.copy()

        border_slices = [
            (slice(0, bh), slice(0, W)),         # top
            (slice(H - bh, H), slice(0, W)),     # bottom
            (slice(0, H), slice(0, bw)),         # left
            (slice(0, H), slice(W - bw, W)),     # right
        ]

        for row_slice, col_slice in border_slices:
            region = gray[row_slice, col_slice]
            enhanced_gray[row_slice, col_slice] = self._clahe.apply(region)

        # Convert back to original format
        if len(image.shape) == 3:
            return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        return enhanced_gray
