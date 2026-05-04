"""
test_ocr.py
-----------
Production test script for the TextDetector OCR module.

Usage:
    python -m ocr.test_ocr

Requirements:
    - A test image must exist in test_images/ directory
    - OpenCV (cv2) for image I/O and visualization
    - The TextDetector module from this package
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Set environment variables BEFORE importing paddle
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ocr.text_detector import TextDetector


def find_test_image() -> Optional[Path]:
    """Locate a test image in the project directory.
    
    Returns
    -------
    Path or None
        Path to the first valid image found, or None if no images exist.
    """
    # Try test_images directory first
    test_images_dir = Path(__file__).parent.parent / "test_images"
    if test_images_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images = list(test_images_dir.glob(ext))
            if images:
                return images[0]
    
    # Fallback to project root
    project_root = Path(__file__).parent.parent
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images = list(project_root.glob(ext))
        if images:
            return images[0]
    
    return None


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Load an image from disk and convert to RGB format.
    
    Parameters
    ----------
    image_path : Path
        Path to the image file.
    
    Returns
    -------
    np.ndarray
        RGB image as uint8 array with shape (H, W, 3).
    
    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the image cannot be loaded or is invalid.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # OpenCV loads images in BGR format
    bgr_image = cv2.imread(str(image_path))
    
    if bgr_image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR → RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    return rgb_image


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: list[np.ndarray],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw bounding box polygons on an image.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image to draw on (will be copied, not modified in-place).
    boxes : list[np.ndarray]
        List of polygon arrays, each with shape (4, 2).
    color : tuple[int, int, int]
        RGB color for the bounding boxes (default: green).
    thickness : int
        Line thickness in pixels (default: 2).
    
    Returns
    -------
    np.ndarray
        Copy of the image with bounding boxes drawn.
    """
    # Work on a copy to avoid modifying the original
    result = image.copy()
    
    # Convert RGB → BGR for OpenCV drawing
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    for box in boxes:
        # Reshape to required format for polylines
        pts = box.reshape((-1, 1, 2))
        cv2.polylines(result_bgr, [pts], isClosed=True, color=color[::-1], thickness=thickness)
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    return result_rgb


def main() -> int:
    """Main test routine.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    print("=" * 70)
    print("OCR Module Test - TextDetector")
    print("=" * 70)
    
    # 1. Find test image
    print("\n[1/5] Locating test image...")
    image_path = find_test_image()
    
    if image_path is None:
        print("❌ ERROR: No test images found in test_images/ or project root")
        print("   Please add a .jpg or .png file to test_images/ directory")
        return 1
    
    print(f"✓ Found: {image_path.name}")
    
    # 2. Load image
    print("\n[2/5] Loading image...")
    try:
        rgb_image = load_image_rgb(image_path)
        h, w = rgb_image.shape[:2]
        print(f"✓ Loaded: {w}x{h} RGB image")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1
    
    # 3. Initialize TextDetector
    print("\n[3/5] Initializing TextDetector...")
    try:
        detector = TextDetector(lang="en", conf_threshold=0.5)
        print("✓ PaddleOCR initialized successfully")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1
    
    # 4. Detect text regions
    print("\n[4/5] Detecting text regions...")
    try:
        boxes = detector.detect_text(rgb_image)
        print(f"✓ Detection complete: {len(boxes)} region(s) found")
        
        if boxes:
            print(f"   Confidence threshold: 0.5")
            print(f"   Bounding boxes shape: {boxes[0].shape}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1
    
    # 5. Draw and save results
    print("\n[5/5] Saving visualization...")
    try:
        result_image = draw_bounding_boxes(rgb_image, boxes, color=(0, 255, 0), thickness=2)
        
        # Save as BGR for OpenCV
        output_path = Path(__file__).parent / "ocr_result.jpg"
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result_bgr)
        
        print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TEST PASSED")
    print("=" * 70)
    print(f"Input:  {image_path}")
    print(f"Output: {output_path}")
    print(f"Detected regions: {len(boxes)}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
