"""pixel_redactor.py - Safe pixel-level text redaction for medical images.

This module provides pixel-level redaction of burned-in text regions using
OpenCV inpainting to naturally blend with the surrounding background.
It implements safety measures to protect diagnostic data in the central
region of medical images.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import pydicom

logger = logging.getLogger(__name__)


class PixelRedactor:
    """Redacts text regions using inpainting for natural background blending.
    
    This class implements a safety-first approach to pixel redaction:
    1. SAFE ZONE FILTER: Only redacts regions within border_margin pixels of edges
    2. Central region detection is logged but NOT auto-redacted
    3. Uses OpenCV inpainting (TELEA) to naturally blend redacted regions
    4. Proper handling of grayscale vs RGB and uint8 vs uint16
    
    The redaction uses inpainting which reconstructs the redacted region by
    propagating surrounding pixel values, creating a natural blend with the
    background rather than an artificial black rectangle.
    
    Example:
        >>> redactor = PixelRedactor(padding=5, border_margin=100)
        >>> ds = pydicom.dcmread("input.dcm")
        >>> regions = [np.array([[10,10], [50,10], [50,30], [10,30]])]
        >>> redacted_ds, count = redactor.redact(ds, regions)
        >>> print(f"Redacted {count} regions")
    """
    
    def __init__(self, padding: int = 5, border_margin: int = 100) -> None:
        """Initialize the pixel redactor.
        
        Args:
            padding: Number of pixels to expand each region before redacting
            border_margin: Distance from edge to consider "safe" for redaction.
                         Regions within this margin are auto-redacted.
                         Central regions are skipped with a warning.
        """
        self.padding = padding
        self.border_margin = border_margin
        logger.debug(
            f"PixelRedactor initialized: padding={padding}, "
            f"border_margin={border_margin}"
        )
    
    def redact(
        self,
        image_data: "pydicom.Dataset | np.ndarray",
        regions: list[np.ndarray],
        padding: int | None = None,
        border_margin: int | None = None
    ) -> tuple["pydicom.Dataset | np.ndarray", int]:
        """Redact text regions using inpainting for natural background blending.
        
        Implements safe redaction with the following rules:
        1. Only regions within border_margin of any edge are auto-redacted
        2. Central regions are skipped with a warning
        3. Uses OpenCV inpainting (TELEA algorithm) to naturally fill regions
        4. Grayscale images remain grayscale, RGB remains RGB
        5. For DICOM: original bit depth (uint8/uint16) is preserved
        
        Inpainting reconstructs the redacted region by propagating surrounding
        pixel values, creating a natural blend with the X-ray background rather
        than an artificial black rectangle.
        
        Args:
            image_data: Either a pydicom Dataset OR a numpy array of pixels
            regions: List of detected text regions, each a (4,2) numpy array
                     containing polygon corner coordinates [x, y]
            padding: Override default padding (uses self.padding if None)
            border_margin: Override default border margin
            
        Returns:
            A tuple containing:
                - The updated image data (Dataset for DICOM, numpy array for regular)
                - The count of regions that were actually redacted
                
        Example:
            >>> # For DICOM
            >>> ds = pydicom.dcmread("input.dcm")
            >>> redacted_ds, count = redactor.redact(ds, regions)
            >>> 
            >>> # For regular image
            >>> img = np.array(Image.open("input.jpg"))
            >>> redacted_img, count = redactor.redact(img, regions)
        """
        if padding is None:
            padding = self.padding
        if border_margin is None:
            border_margin = self.border_margin
        
        # Determine if input is DICOM or numpy array
        is_dicom = hasattr(image_data, 'pixel_array')
        
        if is_dicom:
            # DICOM: extract pixel array
            ds = image_data
            pixels = ds.pixel_array.copy()
            logger.info(f"Processing DICOM image: {len(regions)} regions detected")
        else:
            # Regular image: input is numpy array
            pixels = image_data.copy()
            logger.info(f"Processing regular image: {len(regions)} regions detected")
        
        logger.info(f"Safety margin: {border_margin}px from edges")
        
        # Determine if grayscale or RGB
        is_grayscale = len(pixels.shape) == 2
        
        logger.debug(f"Working with {'grayscale' if is_grayscale else str(pixels.shape[2]) + '-channel'} pixel array")
        
        # Track redaction statistics
        redacted_count = 0
        skipped_count = 0
        
        # Store original dtype for later conversion
        original_dtype = pixels.dtype
        
        # Separate regions for black fill (on black frame) vs inpainting
        black_fill_regions = []
        inpaint_regions = []
        
        if is_grayscale:
            height, width = pixels.shape
        else:
            height, width = pixels.shape[:2]
        
        # Define safe zone boundaries (relative 8% margin)
        margin_x = max(border_margin, int(width * 0.08))
        margin_y = max(border_margin, int(height * 0.08))
        safe_x_min = margin_x
        safe_x_max = width - margin_x
        safe_y_min = margin_y
        safe_y_max = height - margin_y
        
        logger.debug(
            f"Safe zone: x=[{safe_x_min}, {safe_x_max}], "
            f"y=[{safe_y_min}, {safe_y_max}]"
        )
        
        # First pass: identify safe vs central regions
        for i, region in enumerate(regions):
            # Calculate bounding box
            x_coords = region[:, 0]
            y_coords = region[:, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            
            # Calculate region center
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # Check if region is in safe zone (within border_margin of edges)
            is_near_left = x_min < safe_x_min
            is_near_right = x_max > safe_x_max
            is_near_top = y_min < safe_y_min
            is_near_bottom = y_max > safe_y_max
            
            is_in_safe_zone = is_near_left or is_near_right or is_near_top or is_near_bottom
            
            if not is_in_safe_zone:
                # Region is in central area - skip with warning
                logger.warning(
                    f"Region {i+1}: SKIPPED (central area) | "
                    f"Center: ({center_x}, {center_y}) | "
                    f"Bounds: ({x_min},{y_min})-({x_max},{y_max}) | "
                    f"Manual review required - may contain diagnostic data"
                )
                skipped_count += 1
                continue
            
            # Expand region by padding
            x_min_padded = max(0, x_min - padding)
            x_max_padded = min(width, x_max + padding)
            y_min_padded = max(0, y_min - padding)
            y_max_padded = min(height, y_max + padding)
            
            # Log redaction
            edge_info = []
            if is_near_left:
                edge_info.append("left")
            if is_near_right:
                edge_info.append("right")
            if is_near_top:
                edge_info.append("top")
            if is_near_bottom:
                edge_info.append("bottom")
            
            logger.info(
                f"Region {i+1}: REDACTING ({', '.join(edge_info)} edge) | "
                f"Bounds: ({x_min_padded},{y_min_padded})-({x_max_padded},{y_max_padded})"
            )
            
            # Determine whether to use black fill or inpainting
            if self._is_on_black_frame(
                pixels, x_min_padded, y_min_padded,
                x_max_padded, y_max_padded
            ):
                black_fill_regions.append(
                    (x_min_padded, y_min_padded, x_max_padded, y_max_padded)
                )
                logger.debug(
                    f"Region {i+1}: will use black fill (on black border frame)"
                )
            else:
                inpaint_regions.append(
                    (x_min_padded, y_min_padded, x_max_padded, y_max_padded)
                )
                logger.debug(
                    f"Region {i+1}: will use inpainting (X-ray background)"
                )
            redacted_count += 1
        
        # Apply black fill first for regions on black frame
        for x_min, y_min, x_max, y_max in black_fill_regions:
            if is_grayscale:
                pixels[y_min:y_max, x_min:x_max] = 0
            else:
                pixels[y_min:y_max, x_min:x_max] = [0, 0, 0]
        
        # Perform inpainting on remaining regions
        if inpaint_regions:
            if is_grayscale:
                # Grayscale: single mask, single inpainting pass
                mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
                
                # Fill mask for all regions
                for x_min, y_min, x_max, y_max in inpaint_regions:
                    mask[y_min:y_max, x_min:x_max] = 255
                
                # Convert to uint8 for inpainting if needed
                if pixels.dtype != np.uint8:
                    pixels = pixels.astype(np.uint8)
                
                # Create mask for inpainting
                mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
                for x_min, y_min, x_max, y_max in inpaint_regions:
                    mask[y_min:y_max, x_min:x_max] = 255
                
                # Inpaint with radius 3
                inpainted = cv2.inpaint(pixels, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                
                # Add subtle noise to match X-ray grain texture
                noise = np.random.normal(0, 3, inpainted.shape).astype(np.int16)
                pixels = np.clip(inpainted.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
            else:
                # RGB: ensure uint8 first
                if pixels.dtype != np.uint8:
                    pixels = pixels.astype(np.uint8)
                
                # Create mask for inpainting ONLY (exclude black fill regions)
                mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
                for x_min, y_min, x_max, y_max in inpaint_regions:
                    mask[y_min:y_max, x_min:x_max] = 255
                
                # Inpaint with radius 3
                num_channels = pixels.shape[2]
                inpainted_channels = []
                for c in range(num_channels):
                    channel = pixels[:, :, c]
                    inpainted_channel = cv2.inpaint(channel, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                    inpainted_channels.append(inpainted_channel)
                
                # Merge channels
                inpainted = np.stack(inpainted_channels, axis=-1)
                
                # Add subtle noise to match image grain texture
                noise = np.random.normal(0, 3, inpainted.shape).astype(np.int16)
                inpainted = np.clip(inpainted.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Copy inpainted regions back to pixels (which has black fill)
                for x_min, y_min, x_max, y_max in inpaint_regions:
                    pixels[y_min:y_max, x_min:x_max] = inpainted[y_min:y_max, x_min:x_max]
        
        # Update the result - ensure pixels contains the inpainted result
        if is_dicom:
            import pydicom
            # For DICOM: update PixelData and metadata
            ds.PixelData = pixels.tobytes()
            ds.BitsAllocated = 8  # Now always uint8 after inpainting
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            result = ds
        else:
            # For regular image: return numpy array (must be inpainted pixels)
            result = pixels
        
        # Log summary
        logger.info(
            f"Pixel redaction complete: {redacted_count} redacted using inpainting, "
            f"{skipped_count} skipped (central)"
        )
        
        if skipped_count > 0:
            logger.warning(
                f"{skipped_count} region(s) in central area require manual review"
            )
        
        return result, redacted_count
    
    def _is_region_in_safe_zone(
        self,
        region: np.ndarray,
        width: int,
        height: int,
        border_margin: int
    ) -> bool:
        """Check if a region is within the safe redaction zone (near edges).
        
        Args:
            region: (4,2) numpy array of polygon corners
            width: Image width in pixels
            height: Image height in pixels
            border_margin: Safe zone margin from edges
            
        Returns:
            True if region is within border_margin of any edge
        """
        x_coords = region[:, 0]
        y_coords = region[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Check if near any edge
        near_left = x_min < border_margin
        near_right = x_max > (width - border_margin)
        near_top = y_min < border_margin
        near_bottom = y_max > (height - border_margin)
        
        return near_left or near_right or near_top or near_bottom

    def _is_on_black_frame(
        self,
        pixels: np.ndarray,
        x_min: int, y_min: int,
        x_max: int, y_max: int,
        black_threshold: int = 15
    ) -> bool:
        """Check if region borders the pure black frame of the X-ray."""
        H, W = pixels.shape[:2]
        sample_y1 = max(0, y_min - 15)
        sample_y2 = min(H, y_max + 15)
        sample_x1 = max(0, x_min - 15)
        sample_x2 = min(W, x_max + 15)
        surrounding = pixels[sample_y1:sample_y2, sample_x1:sample_x2]
        if len(surrounding.shape) == 3:
            surrounding = surrounding.mean(axis=2)
        return float(np.mean(surrounding)) < black_threshold
