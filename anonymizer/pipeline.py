"""pipeline.py - Complete medical image anonymization pipeline orchestrator.

This module orchestrates the full anonymization workflow for both DICOM and regular images:
1. Image validation (DICOM or JPEG/PNG)
2. Metadata PHI anonymization (DICOM only)
3. Text detection (OCR)
4. Safe pixel redaction
5. Save anonymized output

The pipeline integrates all components into a single cohesive workflow
with proper error handling and logging.
"""

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Set PaddleOCR environment variables BEFORE any imports
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr.text_detector import TextDetector
from anonymizer.image_validator import ImageValidator, ImageValidationError
from anonymizer.metadata_anonymizer import MetadataAnonymizer
from anonymizer.pixel_redactor import PixelRedactor

if TYPE_CHECKING:
    import pydicom

logger = logging.getLogger(__name__)


class AnonymizationPipeline:
    """Complete medical image anonymization pipeline.
    
    This class orchestrates the full anonymization workflow for both
    DICOM files (.dcm) and regular images (.jpg, .png, etc.):
    1. Validates image file
    2. Anonymizes metadata (PHI tags) - DICOM only
    3. Extracts pixels for OCR
    4. Detects burned-in text
    5. Redacts text regions safely (edge regions only)
    6. Saves anonymized output (preserves original format)
    
    For regular images (JPEG/PNG), metadata anonymization is skipped
    since they don't contain DICOM-style PHI tags.
    
    The pipeline is designed to protect diagnostic data by only auto-redacting
    text regions near the edges of images. Central regions are skipped and
    logged for manual review.
    
    Example:
        >>> pipeline = AnonymizationPipeline(
        ...     conf_threshold=0.5,
        ...     padding=5,
        ...     border_margin=100
        ... )
        >>> result = pipeline.process("input.dcm", "output_dir/")
        >>> print(f"Anonymized {result['tags_anonymized']} tags")
        >>> print(f"Redacted {result['regions_redacted']} regions")
        >>> 
        >>> # For regular images
        >>> result = pipeline.process("input.jpg", "output_dir/")
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.5,
        padding: int = 5,
        border_margin: int = 100
    ) -> None:
        """Initialize the anonymization pipeline.
        
        Args:
            conf_threshold: Minimum confidence for OCR text detection
            padding: Pixels to expand each region before redaction
            border_margin: Distance from edge for safe auto-redaction
        """
        logger.info("Initializing AnonymizationPipeline")
        
        self.conf_threshold = conf_threshold
        self.padding = padding
        self.border_margin = border_margin
        
        # Initialize all stage components
        self.validator = ImageValidator()
        self.metadata_anonymizer = MetadataAnonymizer()
        self.pixel_redactor = PixelRedactor(padding=padding, border_margin=border_margin)
        self.text_detector = TextDetector(lang="en", conf_threshold=conf_threshold)
        
        logger.info(
            f"Pipeline initialized: conf_threshold={conf_threshold}, "
            f"padding={padding}, border_margin={border_margin}"
        )
    
    def process(self, input_path: str, output_dir: str) -> dict[str, Any]:
        """Process an image file through the full anonymization pipeline.
        
        Executes all stages in sequence:
        1. Image validation (DICOM or JPEG/PNG)
        2. Metadata anonymization (DICOM only)
        3. Pixel extraction for OCR
        4. Text detection
        5. Pixel redaction
        6. Save output (preserves original format)
        
        Args:
            input_path: Path to the input file (.dcm, .jpg, .png, etc.)
            output_dir: Directory where anonymized file will be saved
            
        Returns:
            Summary dictionary with:
                - input: Input file path
                - output: Output file path
                - is_dicom: True if DICOM format
                - tags_anonymized: Number of PHI tags anonymized (DICOM only)
                - regions_detected: Number of text regions detected
                - regions_redacted: Number of regions auto-redacted
                - regions_skipped_central: Number of central regions skipped
                - status: "success" or "failed"
                
        Raises:
            ImageValidationError: If image file is invalid
            Exception: If any pipeline stage fails (logged and re-raised)
            
        Example:
            >>> pipeline = AnonymizationPipeline()
            >>> result = pipeline.process("patient.dcm", "./output")
            >>> print(f"Output saved to: {result['output']}")
        """
        from PIL import Image
        
        input_path_obj = Path(input_path)
        output_dir_obj = Path(output_dir)
        
        # Determine output format
        is_dicom_input = ImageValidator.is_dicom_file(str(input_path_obj))
        
        # Initialize result
        result = {
            "input": str(input_path_obj.absolute()),
            "output": None,
            "is_dicom": is_dicom_input,
            "tags_anonymized": 0,
            "regions_detected": 0,
            "regions_redacted": 0,
            "regions_skipped_central": 0,
            "status": "failed",
            "error": None
        }
        
        try:
            # Ensure output directory exists
            output_dir_obj.mkdir(parents=True, exist_ok=True)
            
            # =====================================================================
            # STAGE 1: Validate Image
            # =====================================================================
            logger.info("=" * 60)
            logger.info(f"STAGE 1: {'DICOM' if is_dicom_input else 'Image'} Validation")
            logger.info("=" * 60)
            
            try:
                validation_result = self.validator.validate(str(input_path_obj))
                is_dicom = validation_result.is_dicom
                dataset = validation_result.dataset
                pixel_array_source = validation_result.pixel_array
                
                logger.info(f"✓ Validation passed: {input_path_obj.name}")
                if is_dicom:
                    logger.info(f"  Format: DICOM | Modality: {getattr(dataset, 'Modality', 'Unknown')}")
                else:
                    logger.info(f"  Format: {input_path_obj.suffix.upper()}")
            except ImageValidationError as e:
                logger.error(f"✗ Validation failed: {e.message}")
                result["error"] = f"Validation failed: {e.message}"
                raise
            
            # =====================================================================
            # STAGE 2: Anonymize Metadata (DICOM only)
            # =====================================================================
            if is_dicom:
                logger.info("\n" + "=" * 60)
                logger.info("STAGE 2: Metadata Anonymization")
                logger.info("=" * 60)
                
                dataset, tags_anonymized = self.metadata_anonymizer.anonymize(dataset)
                result["tags_anonymized"] = tags_anonymized
                logger.info(f"✓ Anonymized {tags_anonymized} PHI tags")
            else:
                logger.info("\n" + "=" * 60)
                logger.info("STAGE 2: Metadata Anonymization")
                logger.info("=" * 60)
                logger.info("⊘ Skipped - Not a DICOM file (no PHI metadata)")
                result["tags_anonymized"] = 0
            
            # =====================================================================
            # STAGE 3: Extract Pixels for OCR
            # =====================================================================
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 3: Pixel Extraction for OCR")
            logger.info("=" * 60)
            
            # Get pixel array
            if is_dicom:
                pixel_array = dataset.pixel_array.copy()
            else:
                pixel_array = pixel_array_source.copy()
            
            logger.info(f"Pixel array shape: {pixel_array.shape}, dtype: {pixel_array.dtype}")
            
            # Convert to RGB uint8 for OCR
            rgb_image = self._convert_to_rgb_uint8(pixel_array)
            logger.info(f"Converted to RGB uint8: {rgb_image.shape}")
            
            # =====================================================================
            # STAGE 4: Detect Burned-in Text
            # =====================================================================
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 4: Text Detection (OCR)")
            logger.info("=" * 60)
            
            regions = self.text_detector.detect_text(rgb_image)
            result["regions_detected"] = len(regions)
            logger.info(f"✓ Detected {len(regions)} text region(s)")
            
            if len(regions) > 0:
                for i, region in enumerate(regions):
                    x_coords = region[:, 0]
                    y_coords = region[:, 1]
                    logger.debug(
                        f"  Region {i+1}: "
                        f"x={x_coords.min()}-{x_coords.max()}, "
                        f"y={y_coords.min()}-{y_coords.max()}"
                    )
            
            # =====================================================================
            # STAGE 5: Redact Pixels
            # =====================================================================
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 5: Pixel Redaction")
            logger.info("=" * 60)
            
            if len(regions) > 0:
                # Pass appropriate data type to redactor
                if is_dicom:
                    redacted_data, regions_redacted = self.pixel_redactor.redact(
                        dataset,
                        regions,
                        padding=self.padding,
                        border_margin=self.border_margin
                    )
                    dataset = redacted_data
                else:
                    redacted_data, regions_redacted = self.pixel_redactor.redact(
                        pixel_array,
                        regions,
                        padding=self.padding,
                        border_margin=self.border_margin
                    )
                    pixel_array = redacted_data
                
                result["regions_redacted"] = regions_redacted
                result["regions_skipped_central"] = len(regions) - regions_redacted
                logger.info(f"✓ Redacted {regions_redacted} region(s)")
                if result["regions_skipped_central"] > 0:
                    logger.warning(
                        f"⚠ Skipped {result['regions_skipped_central']} "
                        f"central region(s) - manual review required"
                    )
            else:
                logger.info("No text regions to redact")
            
            # =====================================================================
            # STAGE 6: Save Output
            # =====================================================================
            logger.info("\n" + "=" * 60)
            logger.info(f"STAGE 6: Save Anonymized {'DICOM' if is_dicom else 'Image'}")
            logger.info("=" * 60)
            
            if is_dicom:
                # Save as DICOM
                output_filename = f"anonymized_{input_path_obj.name}"
                output_path = output_dir_obj / output_filename
                
                import pydicom
                pydicom.dcmwrite(str(output_path), dataset)
                result["output"] = str(output_path.absolute())
                
                logger.info(f"✓ Saved anonymized DICOM: {output_path}")
            else:
                # Save as regular image (preserve original format)
                output_filename = f"anonymized_{input_path_obj.stem}{input_path_obj.suffix}"
                output_path = output_dir_obj / output_filename
                
                # Convert back to PIL Image and save
                if len(pixel_array.shape) == 2:
                    # Grayscale
                    pil_image = Image.fromarray(pixel_array.astype(np.uint8), mode='L')
                else:
                    # RGB
                    pil_image = Image.fromarray(pixel_array.astype(np.uint8), mode='RGB')
                
                pil_image.save(str(output_path))
                result["output"] = str(output_path.absolute())
                
                logger.info(f"✓ Saved anonymized image: {output_path}")
            
            result["status"] = "success"
            
            # =====================================================================
            # Summary
            # =====================================================================
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Input:  {result['input']}")
            logger.info(f"Output: {result['output']}")
            logger.info(f"Format: {'DICOM' if result['is_dicom'] else 'Image'}")
            if result['is_dicom']:
                logger.info(f"Tags anonymized:        {result['tags_anonymized']}")
            logger.info(f"Regions detected:       {result['regions_detected']}")
            logger.info(f"Regions redacted:       {result['regions_redacted']}")
            logger.info(f"Regions skipped:        {result['regions_skipped_central']}")
            
            return result
            
        except Exception as e:
            logger.exception("Pipeline failed")
            result["error"] = str(e)
            result["status"] = "failed"
            raise
    
    def _convert_to_rgb_uint8(self, pixel_array: np.ndarray) -> np.ndarray:
        """Convert pixel array to RGB uint8 format for OCR.
        
        Handles:
        - Grayscale (2D) → RGB (3D)
        - uint16 → uint8 normalization
        - Multi-channel preservation
        
        Args:
            pixel_array: Input pixel array (2D grayscale or 3D multi-channel)
            
        Returns:
            RGB uint8 numpy array suitable for TextDetector
        """
        # Make a copy to avoid modifying original
        pixels = pixel_array.copy()
        
        # Handle grayscale → RGB conversion
        if len(pixels.shape) == 2:
            # Convert grayscale to RGB
            pixels = np.stack([pixels] * 3, axis=-1)
        
        # Normalize to uint8 if needed
        if pixels.dtype != np.uint8:
            # Normalize to 0-255 range
            if pixels.max() > 0:
                pixels = ((pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255)
            pixels = pixels.astype(np.uint8)
        
        return pixels
