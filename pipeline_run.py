"""pipeline_run.py - CLI entry point for medical image anonymization pipeline.

Usage:
    python pipeline_run.py <input_file> <output_dir>

Optional flags:
    --confidence FLOAT   OCR confidence threshold (default: 0.1)
    --padding INT        Redaction padding in pixels (default: 5)
    --margin INT         Border margin for safe redaction (default: 100)
    --verbose            Enable DEBUG logging

Supported formats:
    - DICOM: .dcm, .dicom
    - Images: .jpg, .jpeg, .png, .bmp, .tiff, .tif

The pipeline executes 7 stages:
1. Classification - Verify image is medical using CLIP
2. Validation - Validate DICOM/image file structure
3. Metadata Anonymization - Remove PHI tags (DICOM only)
4. Preprocessing - Enhance borders for OCR
5. Dual OCR Detection - PaddleOCR + EasyOCR with deduplication
6. Pixel Redaction - Inpaint detected text regions
7. Save Output - Write anonymized file
"""

import os
import sys

# =============================================================================
# CRITICAL: Set PaddleOCR environment variables BEFORE any imports
# =============================================================================
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from medical_anonymizer.improved_medical_classifier import MedicalImageClassifier
from anonymizer import ImageValidator, ImageValidationError, MetadataAnonymizer, PixelRedactor
from ocr import TextDetector, BorderPreprocessor, EasyTextDetector

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def convert_to_rgb_uint8(pixel_array: np.ndarray) -> np.ndarray:
    """Convert pixel array to RGB uint8 format for OCR."""
    pixels = pixel_array.copy()
    
    # Handle grayscale → RGB conversion
    if len(pixels.shape) == 2:
        pixels = np.stack([pixels] * 3, axis=-1)
    
    # Normalize to uint8 if needed
    if pixels.dtype != np.uint8:
        if pixels.max() > 0:
            pixels = ((pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255)
        pixels = pixels.astype(np.uint8)
    
    return pixels


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union between two polygon boxes.
    
    Args:
        box1: (4,2) polygon array
        box2: (4,2) polygon array
        
    Returns:
        IoU value between 0 and 1
    """
    # Convert polygons to bounding boxes for simpler IoU calculation
    x1_min, y1_min = box1[:, 0].min(), box1[:, 1].min()
    x1_max, y1_max = box1[:, 0].max(), box1[:, 1].max()
    x2_min, y2_min = box2[:, 0].min(), box2[:, 1].min()
    x2_max, y2_max = box2[:, 0].max(), box2[:, 1].max()
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def merge_and_deduplicate(
    paddle_regions: List[np.ndarray],
    easy_regions: List[np.ndarray],
    iou_threshold: float = 0.5
) -> List[np.ndarray]:
    """Merge OCR results and remove duplicates based on IoU.
    
    Args:
        paddle_regions: Regions from PaddleOCR
        easy_regions: Regions from EasyOCR
        iou_threshold: If IoU > threshold, consider as duplicate
        
    Returns:
        Deduplicated list of regions
    """
    # Start with all paddle regions
    merged = list(paddle_regions)
    
    # Add easy regions only if they don't overlap significantly with existing
    for easy_box in easy_regions:
        is_duplicate = False
        for existing_box in merged:
            if compute_iou(easy_box, existing_box) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(easy_box)
    
    return merged


def print_summary_box(
    category: str,
    confidence: float,
    file_format: str,
    metadata_count: int,
    is_dicom: bool,
    paddle_count: int,
    easy_count: int,
    merged_count: int,
    redacted_count: int,
    skipped_count: int,
    output_path: str
) -> None:
    """Print formatted summary box."""
    # Prepare values
    classification_str = f"{category} ({confidence:.2f})"
    metadata_str = str(metadata_count) if is_dicom else f"{metadata_count} (not DICOM)"
    
    # Calculate box width based on longest line
    lines = [
        ("Classification:", classification_str),
        ("Format:", file_format),
        ("Metadata cleaned:", metadata_str),
        ("PaddleOCR regions:", str(paddle_count)),
        ("EasyOCR regions:", str(easy_count)),
        ("After merge:", str(merged_count)),
        ("Redacted:", str(redacted_count)),
        ("Skipped (central):", str(skipped_count)),
        ("Output:", output_path),
    ]
    
    # Find max width needed
    max_label_len = max(len(label) for label, _ in lines)
    max_value_len = max(len(value) for _, value in lines)
    inner_width = max_label_len + max_value_len + 3  # 3 for spacing
    inner_width = max(inner_width, 40)  # minimum width
    
    # Print box
    print()
    print(f"  ╔{'═' * inner_width}╗")
    print(f"  ║{'ANONYMIZATION COMPLETE':^{inner_width}}║")
    print(f"  ╠{'═' * inner_width}╣")
    
    for label, value in lines:
        content = f" {label:<{max_label_len}} {value}"
        print(f"  ║{content:<{inner_width}}║")
    
    print(f"  ╚{'═' * inner_width}╝")
    print()


def main() -> int:
    """Main entry point for the medical image anonymization CLI."""
    parser = argparse.ArgumentParser(
        description="Medical Image Anonymization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pipeline_run.py patient.dcm ./output/
    python pipeline_run.py image.jpg ./output/ --verbose
    python pipeline_run.py scan.png ./anonymized/ --confidence 0.2 --padding 10
        """
    )
    
    parser.add_argument("input", help="Path to input file (.dcm, .jpg, .png, etc.)")
    parser.add_argument("output_dir", help="Directory where anonymized file will be saved")
    parser.add_argument("--confidence", type=float, default=0.1,
                        help="OCR confidence threshold (default: 0.1)")
    parser.add_argument("--padding", type=int, default=5,
                        help="Redaction padding in pixels (default: 5)")
    parser.add_argument("--margin", type=int, default=100,
                        help="Border margin for safe redaction (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track stats for summary
    paddle_count = 0
    easy_count = 0
    merged_count = 0
    redacted_count = 0
    skipped_count = 0
    tags_anonymized = 0
    
    try:
        # =====================================================================
        # STAGE 1: CLASSIFICATION
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 1: Classification")
        logger.info("=" * 60)
        
        classifier = MedicalImageClassifier()
        category, confidence, metadata = classifier.classify_image(str(input_path))
        
        if category == "non_medical":
            print(f"✗ Error: Image classified as non-medical (confidence: {confidence:.2f})")
            print("  This pipeline only processes medical images.")
            return 1
        
        print(f"✓ Classification: {category} (confidence: {confidence:.2f})")
        logger.info(f"Category: {category}, Confidence: {confidence:.2f}")
        
        # =====================================================================
        # STAGE 2: VALIDATION
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 2: Validation")
        logger.info("=" * 60)
        
        validator = ImageValidator()
        try:
            validation_result = validator.validate(str(input_path))
        except ImageValidationError as e:
            print(f"✗ Validation failed: {e.message}", file=sys.stderr)
            return 1
        
        is_dicom = validation_result.is_dicom
        dataset = validation_result.dataset
        pixel_array = validation_result.pixel_array if not is_dicom else dataset.pixel_array.copy()
        
        file_format = "DICOM" if is_dicom else input_path.suffix.upper().lstrip(".")
        print(f"✓ Validation passed: {file_format}")
        logger.info(f"Format: {file_format}, Shape: {pixel_array.shape}, Dtype: {pixel_array.dtype}")
        
        # =====================================================================
        # STAGE 3: METADATA ANONYMIZATION (DICOM only)
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 3: Metadata Anonymization")
        logger.info("=" * 60)
        
        metadata_anonymizer = MetadataAnonymizer()
        
        if is_dicom:
            dataset, tags_anonymized = metadata_anonymizer.anonymize(dataset)
            print(f"✓ Metadata tags anonymized: {tags_anonymized}")
            logger.info(f"Anonymized {tags_anonymized} PHI tags")
        else:
            tags_anonymized = 0
            print("⊘ Skipped - Not a DICOM file")
            logger.info("Skipped metadata anonymization (not DICOM)")
        
        # =====================================================================
        # STAGE 4: PREPROCESSING
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 4: Preprocessing")
        logger.info("=" * 60)
        
        # Convert to RGB uint8 for OCR
        rgb_image = convert_to_rgb_uint8(pixel_array)
        logger.info(f"Converted to RGB uint8: {rgb_image.shape}")
        
        # Enhance borders
        preprocessor = BorderPreprocessor()
        enhanced_image = preprocessor.enhance(rgb_image)
        
        print("✓ Border preprocessing complete")
        logger.info("Border CLAHE enhancement applied")
        
        # =====================================================================
        # STAGE 5: DUAL OCR DETECTION
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 5: Dual OCR Detection")
        logger.info("=" * 60)
        
        # PaddleOCR pass
        paddle_detector = TextDetector(lang="en", conf_threshold=args.confidence)
        paddle_regions = paddle_detector.detect_text(enhanced_image)
        paddle_count = len(paddle_regions)
        logger.info(f"PaddleOCR detected {paddle_count} regions")
        
        # EasyOCR pass (wrapped in try/except - never crash)
        easy_regions = []
        try:
            easy_detector = EasyTextDetector(conf_threshold=args.confidence, border_pct=0.20)
            easy_regions = easy_detector.detect_text(enhanced_image)
            easy_count = len(easy_regions)
            logger.info(f"EasyOCR detected {easy_count} regions")
        except Exception as e:
            logger.warning(f"EasyOCR failed (non-fatal): {e}")
            easy_count = 0
        
        # Merge and deduplicate
        merged_regions = merge_and_deduplicate(paddle_regions, easy_regions, iou_threshold=0.5)
        merged_count = len(merged_regions)
        
        print(f"✓ PaddleOCR: {paddle_count} | EasyOCR: {easy_count} | Total after merge: {merged_count}")
        logger.info(f"After deduplication: {merged_count} regions")
        
        # =====================================================================
        # STAGE 6: PIXEL REDACTION
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 6: Pixel Redaction")
        logger.info("=" * 60)
        
        redactor = PixelRedactor(padding=args.padding, border_margin=args.margin)
        
        if merged_count > 0:
            if is_dicom:
                redacted_data, redacted_count = redactor.redact(
                    dataset, merged_regions,
                    padding=args.padding, border_margin=args.margin
                )
                dataset = redacted_data
            else:
                redacted_data, redacted_count = redactor.redact(
                    pixel_array, merged_regions,
                    padding=args.padding, border_margin=args.margin
                )
                pixel_array = redacted_data
        else:
            redacted_count = 0
            logger.info("No regions to redact")
        
        skipped_count = merged_count - redacted_count
        print(f"✓ Regions redacted: {redacted_count}")
        if skipped_count > 0:
            print(f"⚠ Regions skipped (central): {skipped_count}")
        logger.info(f"Redacted: {redacted_count}, Skipped: {skipped_count}")
        
        # =====================================================================
        # STAGE 7: SAVE OUTPUT
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 7: Save Output")
        logger.info("=" * 60)
        
        output_filename = f"anonymized_{input_path.name}"
        output_path = output_dir / output_filename
        
        if is_dicom:
            import pydicom
            pydicom.dcmwrite(str(output_path), dataset)
        else:
            from PIL import Image
            
            # Get the redacted pixel array
            redacted_pixels = pixel_array
            
            # Ensure uint8
            if redacted_pixels.dtype != np.uint8:
                if redacted_pixels.max() > 0:
                    redacted_pixels = ((redacted_pixels - redacted_pixels.min()) / 
                                       (redacted_pixels.max() - redacted_pixels.min()) * 255)
                redacted_pixels = redacted_pixels.astype(np.uint8)
            
            # Create PIL image
            if len(redacted_pixels.shape) == 2:
                pil_image = Image.fromarray(redacted_pixels, mode='L')
            else:
                pil_image = Image.fromarray(redacted_pixels)
            
            # Save with quality=95 for JPEG
            if input_path.suffix.lower() in {'.jpg', '.jpeg'}:
                pil_image.save(str(output_path), quality=95)
            else:
                pil_image.save(str(output_path))
        
        print(f"✓ Saved: {output_path}")
        logger.info(f"Output saved to: {output_path}")
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        print_summary_box(
            category=category,
            confidence=confidence,
            file_format=file_format,
            metadata_count=tags_anonymized,
            is_dicom=is_dicom,
            paddle_count=paddle_count,
            easy_count=easy_count,
            merged_count=merged_count,
            redacted_count=redacted_count,
            skipped_count=skipped_count,
            output_path=str(output_path)
        )
        
        return 0
        
    except Exception as e:
        logger.exception("Pipeline execution failed")
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())