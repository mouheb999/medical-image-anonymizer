"""
OCR Service

What this module does:
    Provides text detection capabilities for medical images.
    Uses PaddleOCR (primary) with fallback to basic image processing.
    Detects both printed and handwritten text that may contain
    patient-identifying information.

Why it is used:
    Text detection is critical for anonymization - names, dates,
    IDs, and other PHI often appear as overlays on medical images.
    This service provides a reliable abstraction over OCR engines.

Assumptions:
    - Medical images may contain white text on dark backgrounds
    - Text can vary in size and orientation
    - OCR may not be 100% accurate, so we use conservative bounding boxes
    - Fallback mechanisms ensure the pipeline never breaks

Author: PFE Medical Anonymizer
Date: 2025
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """Represents a detected text region with metadata."""
    bbox: np.ndarray  # 4x2 array of corner points
    text: str
    confidence: float
    region_type: str = "unknown"  # "header", "overlay", "annotation"


class OCRService:
    """
    Service for detecting text in medical images.
    
    Primary engine: PaddleOCR (if available)
    Fallback: Basic image processing for common text locations
    """
    
    def __init__(self, use_paddle: bool = True):
        """
        Initialize OCR service.
        
        Args:
            use_paddle: Whether to attempt using PaddleOCR
        """
        self.ocr_engine = None
        self.engine_name = "fallback"
        
        if use_paddle:
            self._init_paddle()
    
    def _init_paddle(self) -> None:
        """Initialize PaddleOCR with error handling."""
        try:
            logger.info("Attempting to initialize PaddleOCR...")
            
            # Import here to avoid hard dependency
            from paddleocr import PaddleOCR
            
            # Initialize with English language (medical terms)
            self.ocr_engine = PaddleOCR(
                lang='en',
                show_log=False
            )
            
            self.engine_name = "paddleocr"
            logger.info("PaddleOCR initialized successfully")
            
        except ImportError as e:
            logger.warning(f"PaddleOCR not available: {e}")
            logger.info("Will use fallback detection methods")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            logger.info("Falling back to basic detection")
    
    def detect_text(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions in the image.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            List of detected TextRegion objects
        """
        logger.info(f"Detecting text using {self.engine_name}")
        
        if self.engine_name == "paddleocr" and self.ocr_engine is not None:
            try:
                regions = self._detect_with_paddle(image)
                if len(regions) > 0:
                    return regions
                logger.info("PaddleOCR found no text, trying fallback")
            except Exception as e:
                logger.warning(f"PaddleOCR detection failed: {e}")
        
        # Fallback to basic detection
        return self._detect_with_fallback(image)
    
    def _detect_with_paddle(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text using PaddleOCR.
        
        Args:
            image: Input image
            
        Returns:
            List of TextRegion objects
        """
        regions = []
        
        try:
            # Ensure image is suitable for PaddleOCR
            if len(image.shape) == 2:
                # Convert grayscale to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image
            
            # Run OCR
            result = self.ocr_engine.ocr(image_bgr)
            
            # Handle different result formats
            if result is None or len(result) == 0:
                return regions
            
            # Parse results
            # PaddleOCR format: [[[bbox], (text, confidence)], ...]
            for line in result[0] if result[0] else []:
                if len(line) >= 2:
                    bbox = np.array(line[0]).astype(np.int32)
                    text_info = line[1]
                    
                    if isinstance(text_info, tuple) and len(text_info) >= 2:
                        text, confidence = text_info[0], text_info[1]
                    else:
                        text, confidence = str(text_info), 0.5
                    
                    region = TextRegion(
                        bbox=bbox,
                        text=text,
                        confidence=float(confidence),
                        region_type="paddle_detected"
                    )
                    regions.append(region)
            
            logger.info(f"PaddleOCR detected {len(regions)} text regions")
            
        except Exception as e:
            logger.error(f"PaddleOCR processing error: {e}")
            raise
        
        return regions
    
    def _detect_with_fallback(self, image: np.ndarray) -> List[TextRegion]:
        """
        Fallback text detection using image processing.
        
        Detects white text on dark backgrounds (common in medical images)
        and dark text on light backgrounds.
        
        Args:
            image: Input image
            
        Returns:
            List of TextRegion objects
        """
        logger.info("Using fallback text detection")
        
        regions = []
        
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect white text on dark background (inverted)
        white_text = self._detect_text_by_color(gray, invert=True)
        regions.extend(white_text)
        
        # Detect dark text on light background
        dark_text = self._detect_text_by_color(gray, invert=False)
        regions.extend(dark_text)
        
        # Add common header/footer regions for medical images
        header_regions = self._add_common_text_regions(gray.shape)
        
        # Only add header regions if we found very little text
        if len(regions) < 2:
            regions.extend(header_regions)
            logger.info(f"Added {len(header_regions)} common text regions")
        
        logger.info(f"Fallback detection found {len(regions)} regions")
        
        return regions
    
    def _detect_text_by_color(
        self, 
        gray: np.ndarray, 
        invert: bool = False
    ) -> List[TextRegion]:
        """
        Detect text of specific color (white or dark).
        
        Args:
            gray: Grayscale image
            invert: True for white text, False for dark text
            
        Returns:
            List of TextRegion objects
        """
        regions = []
        
        # Preprocess
        if invert:
            processed = cv2.bitwise_not(gray)
        else:
            processed = gray.copy()
        
        # Thresholding
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            aspect_ratio = w / max(h, 1)
            area = w * h
            
            # Text-like criteria
            if (15 < w < 400 and 8 < h < 100 and
                1.0 < aspect_ratio < 15 and
                100 < area < 30000):
                
                # Add padding
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(gray.shape[1], x + w + pad)
                y2 = min(gray.shape[0], y + h + pad)
                
                bbox = np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ])
                
                color_type = "white" if invert else "dark"
                
                region = TextRegion(
                    bbox=bbox,
                    text=f"{color_type}_text_{i}",
                    confidence=0.7,
                    region_type=f"{color_type}_text"
                )
                regions.append(region)
        
        return regions
    
    def _add_common_text_regions(self, image_shape: Tuple[int, ...]) -> List[TextRegion]:
        """
        Add common text regions for medical images.
        
        Medical images typically have text in:
        - Top header (patient info)
        - Bottom footer (dates, hospital info)
        - Corners (technical info)
        
        Args:
            image_shape: Shape of the image (H, W) or (H, W, C)
            
        Returns:
            List of TextRegion objects for common locations
        """
        regions = []
        
        h, w = image_shape[:2]
        
        # Common text locations in medical images
        common_regions = [
            # Top header - patient name/info
            {
                'coords': [[20, 20], [w-20, 20], [w-20, 80], [20, 80]],
                'name': 'header_patient_info',
                'type': 'header'
            },
            # Bottom - date/hospital info
            {
                'coords': [[20, h-80], [w-20, h-80], [w-20, h-20], [20, h-20]],
                'name': 'footer_info',
                'type': 'footer'
            },
            # Top right - technical parameters
            {
                'coords': [[w-200, 20], [w-20, 20], [w-20, 150], [w-200, 150]],
                'name': 'technical_info',
                'type': 'side'
            },
        ]
        
        for region_def in common_regions:
            region = TextRegion(
                bbox=np.array(region_def['coords']),
                text=region_def['name'],
                confidence=0.5,  # Lower confidence for manual regions
                region_type=region_def['type']
            )
            regions.append(region)
        
        return regions
    
    def merge_overlapping_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """
        Merge overlapping text regions to avoid double-processing.
        
        Args:
            regions: List of TextRegion objects
            
        Returns:
            Merged list of regions
        """
        if len(regions) <= 1:
            return regions
        
        # Sort by area (largest first)
        regions = sorted(regions, key=lambda r: self._bbox_area(r.bbox), reverse=True)
        
        merged = []
        used = set()
        
        for i, region in enumerate(regions):
            if i in used:
                continue
            
            # Find overlapping regions
            current_bbox = region.bbox.copy()
            merged_indices = {i}
            
            for j, other in enumerate(regions[i+1:], start=i+1):
                if j in used:
                    continue
                
                if self._bboxes_overlap(current_bbox, other.bbox):
                    # Merge bounding boxes
                    current_bbox = self._merge_bboxes(current_bbox, other.bbox)
                    merged_indices.add(j)
            
            # Create merged region
            merged_region = TextRegion(
                bbox=current_bbox,
                text=region.text,
                confidence=region.confidence,
                region_type=region.region_type
            )
            
            merged.append(merged_region)
            used.update(merged_indices)
        
        return merged
    
    def _bbox_area(self, bbox: np.ndarray) -> float:
        """Calculate area of bounding box."""
        x_coords = bbox[:, 0]
        y_coords = bbox[:, 1]
        return (x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min())
    
    def _bboxes_overlap(self, bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
        """Check if two bounding boxes overlap."""
        # Simple rectangle intersection check
        x1_min, x1_max = bbox1[:, 0].min(), bbox1[:, 0].max()
        y1_min, y1_max = bbox1[:, 1].min(), bbox1[:, 1].max()
        
        x2_min, x2_max = bbox2[:, 0].min(), bbox2[:, 0].max()
        y2_min, y2_max = bbox2[:, 1].min(), bbox2[:, 1].max()
        
        return not (x1_max < x2_min or x2_max < x1_min or 
                   y1_max < y2_min or y2_max < y1_min)
    
    def _merge_bboxes(self, bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
        """Merge two bounding boxes into one."""
        x1_min, x1_max = bbox1[:, 0].min(), bbox1[:, 0].max()
        y1_min, y1_max = bbox1[:, 1].min(), bbox1[:, 1].max()
        
        x2_min, x2_max = bbox2[:, 0].min(), bbox2[:, 0].max()
        y2_min, y2_max = bbox2[:, 1].min(), bbox2[:, 1].max()
        
        x_min, x_max = min(x1_min, x2_min), max(x1_max, x2_max)
        y_min, y_max = min(y1_min, y2_min), max(y1_max, y2_max)
        
        return np.array([
            [x_min, y_min], [x_max, y_min], 
            [x_max, y_max], [x_min, y_max]
        ])
