"""
test_confidence_variation.py
-----------------------------
Demonstrates that OCR confidence threshold variations actually affect results.

This script runs the same image through TextDetector with different confidence
thresholds and shows how the number of detected regions changes.

Usage:
    python -m ocr.test_confidence_variation

Output:
    - Console table showing detection counts at different thresholds
    - Visual comparison images saved to ocr/confidence_comparison/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Set environment variables BEFORE importing paddle
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ocr.text_detector import TextDetector


def find_test_image() -> Path | None:
    """Find a test image with text."""
    test_images_dir = Path(__file__).parent.parent / "test_images"
    if test_images_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images = list(test_images_dir.glob(ext))
            if images:
                return images[0]
    
    project_root = Path(__file__).parent.parent
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images = list(project_root.glob(ext))
        if images:
            return images[0]
    
    return None


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Load image and convert to RGB."""
    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def draw_boxes_with_label(
    image: np.ndarray,
    boxes: List[np.ndarray],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draw bounding boxes and add a label to the image."""
    result = image.copy()
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Draw boxes
    for box in boxes:
        pts = box.reshape((-1, 1, 2))
        cv2.polylines(result_bgr, [pts], isClosed=True, color=color[::-1], thickness=2)
    
    # Add label at top
    label_text = f"{label} - {len(boxes)} regions detected"
    cv2.putText(
        result_bgr, label_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
    )
    
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def run_confidence_test(image: np.ndarray, thresholds: List[float]) -> List[Tuple[float, int, List[np.ndarray]]]:
    """Run OCR with different confidence thresholds and collect results.
    
    Returns:
        List of (threshold, detection_count, boxes) tuples
    """
    results = []
    
    print("\n" + "=" * 70)
    print("Running OCR with Different Confidence Thresholds")
    print("=" * 70)
    
    for threshold in thresholds:
        print(f"\n[Testing] Confidence threshold: {threshold:.2f}")
        
        # Initialize detector with this threshold
        detector = TextDetector(lang="en", conf_threshold=threshold)
        
        # Detect text
        boxes = detector.detect_text(image)
        count = len(boxes)
        
        print(f"  → Detected: {count} regions")
        
        results.append((threshold, count, boxes))
    
    return results


def create_comparison_grid(
    image: np.ndarray,
    results: List[Tuple[float, int, List[np.ndarray]]],
    output_dir: Path
) -> None:
    """Create a visual comparison grid showing all threshold results."""
    output_dir.mkdir(exist_ok=True)
    
    # Save individual images
    for threshold, count, boxes in results:
        labeled_image = draw_boxes_with_label(
            image, boxes, f"Threshold: {threshold:.2f}"
        )
        
        output_path = output_dir / f"threshold_{threshold:.2f}.jpg"
        bgr_image = cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr_image)
        print(f"  Saved: {output_path.name}")


def print_results_table(results: List[Tuple[float, int, List[np.ndarray]]]) -> None:
    """Print a formatted table of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Confidence Threshold':<25} {'Regions Detected':<20} {'Change':<15}")
    print("-" * 70)
    
    prev_count = None
    for threshold, count, _ in results:
        change_str = ""
        if prev_count is not None:
            diff = count - prev_count
            if diff > 0:
                change_str = f"+{diff} more"
            elif diff < 0:
                change_str = f"{diff} fewer"
            else:
                change_str = "same"
        
        print(f"{threshold:<25.2f} {count:<20} {change_str:<15}")
        prev_count = count
    
    print("=" * 70)


def main() -> int:
    """Main test routine."""
    print("=" * 70)
    print("OCR Confidence Threshold Variation Test")
    print("=" * 70)
    
    # Find test image
    print("\n[1/4] Locating test image...")
    image_path = find_test_image()
    
    if image_path is None:
        print("❌ ERROR: No test images found")
        print("   Add a medical image with text to test_images/ directory")
        return 1
    
    print(f"✓ Found: {image_path.name}")
    
    # Load image
    print("\n[2/4] Loading image...")
    try:
        rgb_image = load_image_rgb(image_path)
        h, w = rgb_image.shape[:2]
        print(f"✓ Loaded: {w}x{h} RGB image")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1
    
    # Run tests with different thresholds
    print("\n[3/4] Running OCR with varying confidence thresholds...")
    
    # Test thresholds from very permissive (0.0) to very strict (0.9)
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    try:
        results = run_confidence_test(rgb_image, thresholds)
    except Exception as e:
        print(f"❌ ERROR during OCR: {e}")
        return 1
    
    # Print results table
    print_results_table(results)
    
    # Create visual comparison
    print("\n[4/4] Creating visual comparison...")
    output_dir = Path(__file__).parent / "confidence_comparison"
    
    try:
        create_comparison_grid(rgb_image, results, output_dir)
        print(f"\n✓ Visual comparisons saved to: {output_dir}")
    except Exception as e:
        print(f"❌ ERROR saving visualizations: {e}")
        return 1
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    counts = [count for _, count, _ in results]
    
    if len(set(counts)) == 1:
        print("⚠️  All thresholds produced the same result.")
        print("   This could mean:")
        print("   - The image has no text")
        print("   - All detections have very high confidence")
        print("   - Try an image with more varied text")
    else:
        print("✅ Confidence threshold variation DOES affect results!")
        print(f"   Lowest threshold (0.0): {counts[0]} regions")
        print(f"   Highest threshold (0.9): {counts[-1]} regions")
        print(f"   Difference: {counts[0] - counts[-1]} regions filtered")
        print("\n   This proves the confidence threshold is actively filtering detections.")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
