# OCR Module

Text detection using PaddleOCR (CPU-only, oneDNN disabled).

## Quick Test

```bash
cd C:\Users\MSI\Desktop\PFE_Test
.\venv\Scripts\python.exe ocr\test.py
```

## Usage

```python
from ocr.text_detector import TextDetector
import numpy as np

detector = TextDetector(lang='en', conf_threshold=0.5)
regions = detector.detect_text(image)  # image: (H, W, 3) uint8 RGB

if len(regions) == 0:
    print("No text detected")
else:
    for region in regions:  # each is (4, 2) polygon
        print(f"Text at: {region}")
```

## Files

- `text_detector.py` - Main TextDetector class
- `test.py` - Quick test (run this)
- `__init__.py` - Package init
