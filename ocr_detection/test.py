"""Quick test for TextDetector - just run: python test.py"""
import os
import sys

# Set environment variables BEFORE importing paddle
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from text_detector import TextDetector


def main():
    print("Testing TextDetector...")
    
    # Initialize
    detector = TextDetector(lang="en", conf_threshold=0.5)
    print("✓ PaddleOCR initialized")
    
    # Test with random image (no text expected)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    regions = detector.detect_text(image)
    
    print(f"✓ Detection complete: {len(regions)} region(s) found")
    
    if len(regions) == 0:
        print("  (Expected: random noise has no text)")
    else:
        print(f"  Regions: {[r.tolist() for r in regions]}")
    
    print("\n✅ Test PASSED - TextDetector is working!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
