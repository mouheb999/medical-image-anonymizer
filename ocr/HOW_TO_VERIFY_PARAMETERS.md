# How to Verify OCR Parameters Are Actually Applied

This guide shows you how to **prove** that OCR parameter variations (confidence threshold, language, etc.) actually affect the model's behavior and aren't just visual settings.

---

## 🎯 Quick Demonstration

Run this command to see immediate proof:

```bash
python -m ocr.demo_parameter_effects
```

**What it does:**
- Creates a test image with text
- Runs OCR with 5 different confidence thresholds: `0.0, 0.3, 0.5, 0.7, 0.9`
- Shows how detection count changes with each threshold
- Saves visual comparisons to `ocr/demo_output/`

**Expected output:**
```
Threshold       Detections      Effect
--------------------------------------------------------------------------------
0.00            4               Baseline (accepts all detections)
0.30            4               Same as previous threshold
0.50            3               🔽 Filtered out 1 low-confidence detection(s)
0.70            2               🔽 Filtered out 1 low-confidence detection(s)
0.90            1               🔽 Filtered out 1 low-confidence detection(s)
```

---

## 📊 Comprehensive Testing

For detailed analysis with real medical images:

```bash
python -m ocr.test_confidence_variation
```

**What it does:**
- Uses actual test images from `test_images/`
- Tests 6 different thresholds: `0.0, 0.1, 0.3, 0.5, 0.7, 0.9`
- Generates comparison images showing bounding boxes
- Saves results to `ocr/confidence_comparison/`

**Output:**
- Console table showing detection counts
- Visual comparison images for each threshold
- Analysis of whether parameters affect results

---

## 🔍 How to Verify Parameters Are Applied

### 1. **Confidence Threshold Verification**

**Location in code:** `ocr/text_detector.py:188-194`

```python
if confidence < self._conf_threshold:
    logger.debug(
        "Dropping detection with confidence %.3f < threshold %.3f.",
        confidence,
        self._conf_threshold,
    )
    continue  # ← This line SKIPS the detection
```

**Proof:**
1. Run OCR with `conf_threshold=0.1` → Get N detections
2. Run OCR with `conf_threshold=0.9` → Get M detections (M < N)
3. The difference proves filtering is active

**Manual test:**
```python
from ocr import TextDetector
import cv2

image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Low threshold - more detections
detector_low = TextDetector(conf_threshold=0.1)
boxes_low = detector_low.detect_text(image_rgb)
print(f"Low threshold (0.1): {len(boxes_low)} detections")

# High threshold - fewer detections
detector_high = TextDetector(conf_threshold=0.9)
boxes_high = detector_high.detect_text(image_rgb)
print(f"High threshold (0.9): {len(boxes_high)} detections")

# If len(boxes_low) > len(boxes_high), filtering is working!
```

---

### 2. **Language Parameter Verification**

**Location in code:** `ocr/text_detector.py:79-82`

```python
self._engine: Any = PaddleOCR(
    lang=lang,  # ← Passed directly to PaddleOCR
    use_angle_cls=True
)
```

**Proof:**
1. Check PaddleOCR logs during initialization
2. Different languages download different model files
3. Model files are cached in `~/.paddleocr/whl/`

**What happens:**
- `lang="en"` → Downloads `en_PP-OCRv4_rec_infer` model
- `lang="ch"` → Downloads `ch_PP-OCRv4_rec_infer` model
- Different models = different character recognition

**Manual test:**
```python
from ocr import TextDetector

# English model
detector_en = TextDetector(lang="en")
# Check console output - you'll see:
# "rec_model_dir='.../.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer'"

# Chinese model
detector_ch = TextDetector(lang="ch")
# Check console output - you'll see:
# "rec_model_dir='.../.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer'"
```

---

### 3. **Border Percentage (EasyOCR) Verification**

**Location in code:** `ocr/easy_text_detector.py`

```python
easy_detector = EasyTextDetector(conf_threshold=0.1, border_pct=0.20)
```

**Proof:**
- `border_pct=0.20` → Only scans 20% border region
- `border_pct=0.50` → Scans 50% border region
- Larger border_pct = more area scanned = more detections

**Manual test:**
```python
from ocr import EasyTextDetector
import cv2

image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Small border
detector_small = EasyTextDetector(border_pct=0.10)
boxes_small = detector_small.detect_text(image_rgb)

# Large border
detector_large = EasyTextDetector(border_pct=0.50)
boxes_large = detector_large.detect_text(image_rgb)

print(f"10% border: {len(boxes_small)} detections")
print(f"50% border: {len(boxes_large)} detections")
```

---

## 📈 Visual Proof

After running the demo scripts, check these directories:

```
ocr/
├── demo_output/                    # Quick demo results
│   ├── test_image.jpg             # Original test image
│   ├── result_threshold_0.00.jpg  # All detections
│   ├── result_threshold_0.30.jpg  # Some filtered
│   ├── result_threshold_0.50.jpg  # More filtered
│   ├── result_threshold_0.70.jpg  # Even more filtered
│   └── result_threshold_0.90.jpg  # Most filtered
│
└── confidence_comparison/          # Detailed comparison
    ├── threshold_0.00.jpg
    ├── threshold_0.10.jpg
    ├── threshold_0.30.jpg
    ├── threshold_0.50.jpg
    ├── threshold_0.70.jpg
    └── threshold_0.90.jpg
```

**What to look for:**
- Green bounding boxes around detected text
- Higher thresholds = fewer boxes
- Label at top shows threshold and detection count

---

## 🧪 Production Pipeline Verification

To see parameters in action during actual anonymization:

1. **Check the API logs** when processing an image:
   ```
   [DEBUG] PaddleOCR done: 3 regions
   [DEBUG] EasyOCR done: 4 regions
   [DEBUG] Region merging done: 4 total regions
   ```

2. **Modify parameters** in `api/main.py:345`:
   ```python
   # Change from 0.1 to 0.5
   paddle_detector = TextDetector(lang="en", conf_threshold=0.5)
   ```

3. **Process the same image again** and compare:
   - Lower threshold → More regions detected
   - Higher threshold → Fewer regions detected

---

## 🎓 Understanding the Code Flow

```
User uploads image
       ↓
API receives image (api/main.py)
       ↓
TextDetector initialized with conf_threshold=0.1
       ↓
PaddleOCR runs and returns ALL detections with confidence scores
       ↓
_parse_results() filters detections:
   - For each detection:
     - If confidence < 0.1 → SKIP (line 188)
     - If confidence >= 0.1 → KEEP (line 196)
       ↓
Only filtered detections are returned
       ↓
Pixel redaction uses only these filtered regions
```

**Key insight:** PaddleOCR always returns ALL detections. Your code actively filters them based on `conf_threshold`.

---

## ✅ Checklist: Verify Parameters Work

- [ ] Run `python -m ocr.demo_parameter_effects`
- [ ] Check `ocr/demo_output/` for visual comparisons
- [ ] Verify detection counts decrease with higher thresholds
- [ ] Run `python -m ocr.test_confidence_variation` with real images
- [ ] Check `ocr/confidence_comparison/` for detailed results
- [ ] Review code at `text_detector.py:188-194` to see filtering logic
- [ ] Test with your own images and different thresholds

---

## 🚨 Common Misconceptions

❌ **"Parameters are just UI settings"**
✅ **Reality:** Parameters are passed to the model and actively filter results

❌ **"Confidence threshold is set in PaddleOCR"**
✅ **Reality:** PaddleOCR returns all detections; YOUR code filters by confidence

❌ **"Language parameter doesn't matter"**
✅ **Reality:** Different languages load completely different OCR models

---

## 📞 Need More Proof?

If you still have doubts, try this:

```python
# Add logging to see what's being filtered
import logging
logging.basicConfig(level=logging.DEBUG)

from ocr import TextDetector
detector = TextDetector(conf_threshold=0.5)
boxes = detector.detect_text(image)

# Check console - you'll see:
# "Dropping detection with confidence 0.23 < threshold 0.50"
# "Dropping detection with confidence 0.41 < threshold 0.50"
```

This shows **exactly which detections are being filtered** in real-time.

---

## 📝 Summary

**All parameters are functionally applied:**

| Parameter | Where Applied | Effect |
|-----------|--------------|--------|
| `conf_threshold` | `text_detector.py:188` | Filters low-confidence detections |
| `lang` | `text_detector.py:80` | Loads language-specific OCR model |
| `border_pct` | `easy_text_detector.py` | Limits scan region to borders |
| `iou_threshold` | `api/main.py:365` | Merges overlapping detections |
| `padding` | `api/main.py:377` | Expands redaction area |
| `border_margin` | `api/main.py:377` | Safety boundary for redaction |

**None of these are cosmetic. They all directly affect pipeline behavior.**
