# Testing UI Parameters → Model Flow

This guide demonstrates that **UI input parameters actually affect the AI model behavior**, not just visual changes.

---

## 🎯 What We're Testing

When users adjust these UI controls in the frontend:
- **OCR Confidence Threshold** (0.0 - 1.0)
- **Redaction Padding** (0 - 20 pixels)
- **Border Safety Margin** (50 - 200 pixels)
- **Border Scan Percentage** (10% - 50%)

These values should flow through:
```
Frontend UI → Backend API → FastAPI Pipeline → AI Models
```

---

## 🔍 Complete Data Flow

### 1. **Frontend (React)** - `client/src/components/UploadZone.jsx`

User adjusts sliders in the Advanced Settings panel:

```javascript
const [settings, setSettings] = useState({
  conf_threshold: 0.1,
  padding: 5,
  border_margin: 100,
  border_pct: 0.20
})
```

When user clicks "Anonymize Image", parameters are sent:

```javascript
const formData = new FormData()
formData.append('file', selectedFile)
formData.append('conf_threshold', settings.conf_threshold.toString())
formData.append('padding', settings.padding.toString())
formData.append('border_margin', settings.border_margin.toString())
formData.append('border_pct', settings.border_pct.toString())

await API.post('/anonymize', formData)
```

**✅ Proof Point 1:** Check browser Network tab - FormData includes all parameters

---

### 2. **Backend (Node.js)** - `backend/src/controllers/anonymizeController.js`

Backend extracts parameters from request:

```javascript
const conf_threshold = parseFloat(req.body.conf_threshold) || 0.1
const padding = parseInt(req.body.padding) || 5
const border_margin = parseInt(req.body.border_margin) || 100
const border_pct = parseFloat(req.body.border_pct) || 0.20

console.log('[DEBUG] OCR Parameters from frontend:', {
  conf_threshold, padding, border_margin, border_pct
})
```

Then forwards to FastAPI:

```javascript
formData.append('conf_threshold', conf_threshold.toString())
formData.append('padding', padding.toString())
formData.append('border_margin', border_margin.toString())
formData.append('border_pct', border_pct.toString())

await axios.post(`${process.env.FASTAPI_URL}/anonymize`, formData)
```

**✅ Proof Point 2:** Check backend console logs - parameters are logged

---

### 3. **AI Pipeline (FastAPI)** - `api/main.py`

FastAPI receives parameters as Form fields:

```python
@app.post("/anonymize")
async def anonymize_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.1),
    padding: int = Form(5),
    border_margin: int = Form(100),
    border_pct: float = Form(0.20)
):
    logger.info(f"Parameters received: conf_threshold={conf_threshold}, ...")
    print(f"[DEBUG] OCR Parameters: conf_threshold={conf_threshold}, ...")
```

**✅ Proof Point 3:** Check FastAPI console logs - parameters are received

---

### 4. **OCR Models** - `api/main.py` (Stage 5)

Parameters are passed to TextDetector and EasyTextDetector:

```python
# PaddleOCR with dynamic threshold
paddle_detector = TextDetector(lang="en", conf_threshold=conf_threshold)
paddle_regions = paddle_detector.detect_text(enhanced_image)

# EasyOCR with dynamic threshold and border percentage
easy_detector = EasyTextDetector(conf_threshold=conf_threshold, border_pct=border_pct)
easy_regions = easy_detector.detect_text(enhanced_image)
```

**✅ Proof Point 4:** Check logs - detection counts vary with threshold

---

### 5. **Redaction** - `api/main.py` (Stage 6)

Parameters are passed to PixelRedactor:

```python
redactor = PixelRedactor(padding=padding, border_margin=border_margin)

redacted_data, redacted_count = redactor.redact(
    pixel_array, merged_regions,
    padding=padding,
    border_margin=border_margin,
    redact_all_regions=True
)
```

**✅ Proof Point 5:** Check logs - redaction behavior changes with parameters

---

## 🧪 Step-by-Step Testing Procedure

### Test 1: Confidence Threshold Variation

1. **Start all services:**
   ```bash
   # Terminal 1 - Backend
   cd backend
   npm start

   # Terminal 2 - FastAPI
   cd api
   uvicorn main:app --reload --port 8000

   # Terminal 3 - Frontend
   cd client
   npm run dev
   ```

2. **Open browser:** http://localhost:5173

3. **Upload a medical image with text** (e.g., chest X-ray with patient info)

4. **Expand "Advanced Settings"**

5. **Test with LOW threshold (0.1):**
   - Set confidence threshold to **0.1**
   - Click "Anonymize Image"
   - **Expected:** More detections (includes low-confidence text)
   - Check backend console: `[DEBUG] PaddleOCR done: X regions (threshold=0.1)`

6. **Test with HIGH threshold (0.9):**
   - Upload the SAME image again
   - Set confidence threshold to **0.9**
   - Click "Anonymize Image"
   - **Expected:** Fewer detections (only high-confidence text)
   - Check backend console: `[DEBUG] PaddleOCR done: Y regions (threshold=0.9)`

7. **Verify:** X > Y (more detections at lower threshold)

---

### Test 2: Padding Variation

1. **Upload an image with text**

2. **Test with SMALL padding (0px):**
   - Set padding to **0**
   - Anonymize
   - Download result
   - **Expected:** Tight redaction boxes around text

3. **Test with LARGE padding (20px):**
   - Upload SAME image
   - Set padding to **20**
   - Anonymize
   - Download result
   - **Expected:** Larger redaction boxes (20px extra around text)

4. **Compare images side-by-side:**
   - Redaction boxes should be visibly larger with padding=20

---

### Test 3: Border Margin Variation

1. **Upload an image with text in center AND borders**

2. **Test with SMALL margin (50px):**
   - Set border_margin to **50**
   - Anonymize
   - Check logs: `skipped` count
   - **Expected:** More text skipped (center text not redacted)

3. **Test with LARGE margin (200px):**
   - Upload SAME image
   - Set border_margin to **200**
   - Anonymize
   - Check logs: `skipped` count
   - **Expected:** Less text skipped (larger safe zone for redaction)

---

### Test 4: Border Scan Percentage

1. **Upload an image**

2. **Test with SMALL border (10%):**
   - Set border_pct to **0.10**
   - Anonymize
   - Check logs: `EasyOCR done: X regions (border_pct=0.1)`
   - **Expected:** Fewer EasyOCR detections (smaller scan area)

3. **Test with LARGE border (50%):**
   - Upload SAME image
   - Set border_pct to **0.50**
   - Anonymize
   - Check logs: `EasyOCR done: Y regions (border_pct=0.5)`
   - **Expected:** More EasyOCR detections (larger scan area)

---

## 📊 Expected Console Output

### Frontend (Browser Console)
```
FormData {
  file: File,
  conf_threshold: "0.5",
  padding: "10",
  border_margin: "150",
  border_pct: "0.30"
}
```

### Backend (Node.js Console)
```
[DEBUG] OCR Parameters from frontend: {
  conf_threshold: 0.5,
  padding: 10,
  border_margin: 150,
  border_pct: 0.3
}
```

### FastAPI (Python Console)
```
[DEBUG] OCR Parameters: conf_threshold=0.5, padding=10, border_margin=150, border_pct=0.3
[DEBUG] Starting PaddleOCR with conf_threshold=0.5...
[DEBUG] PaddleOCR done: 4 regions (threshold=0.5)
[DEBUG] Starting EasyOCR with conf_threshold=0.5, border_pct=0.3...
[DEBUG] EasyOCR done: 2 regions (threshold=0.5, border_pct=0.3)
[DEBUG] Starting pixel redaction with padding=10, border_margin=150...
```

---

## 🔬 Verification Checklist

- [ ] Browser Network tab shows parameters in FormData
- [ ] Backend console logs show extracted parameters
- [ ] FastAPI console logs show received parameters
- [ ] OCR detection counts change with different thresholds
- [ ] Redaction box sizes change with different padding
- [ ] Skipped regions change with different border margins
- [ ] EasyOCR detections change with different border percentages
- [ ] Downloaded images show visual differences

---

## 🎓 Understanding the Code

### Where Filtering Happens

**`ocr/text_detector.py:188-194`**
```python
if confidence < self._conf_threshold:
    logger.debug(
        "Dropping detection with confidence %.3f < threshold %.3f.",
        confidence,
        self._conf_threshold,
    )
    continue  # ← This line SKIPS the detection
```

**Key Insight:** PaddleOCR returns ALL detections. Your code actively filters them based on `conf_threshold`.

### Where Padding is Applied

**`anonymizer/pixel_redactor.py`**
```python
def redact(self, image, regions, padding=5, border_margin=100):
    for region in regions:
        # Expand region by padding
        x_min = max(0, region[:, 0].min() - padding)
        y_min = max(0, region[:, 1].min() - padding)
        x_max = min(w, region[:, 0].max() + padding)
        y_max = min(h, region[:, 1].max() + padding)
```

**Key Insight:** Padding expands the redaction box in all directions.

---

## 🚨 Common Issues

### Issue: Parameters not changing results

**Possible causes:**
1. Browser cache - Hard refresh (Ctrl+Shift+R)
2. Services not restarted after code changes
3. Wrong image uploaded (no text to detect)
4. Logs not being checked

**Solution:**
- Restart all services
- Clear browser cache
- Use an image with visible text
- Check ALL console logs (browser, backend, FastAPI)

---

## 📈 Expected Results Summary

| Parameter | Low Value | High Value | Expected Effect |
|-----------|-----------|------------|-----------------|
| **conf_threshold** | 0.1 | 0.9 | More detections → Fewer detections |
| **padding** | 0px | 20px | Tight boxes → Large boxes |
| **border_margin** | 50px | 200px | More skipped → Less skipped |
| **border_pct** | 0.10 | 0.50 | Fewer EasyOCR → More EasyOCR |

---

## ✅ Success Criteria

**The test is successful if:**

1. ✅ Different threshold values produce different detection counts
2. ✅ Different padding values produce visibly different redaction sizes
3. ✅ Different border margins affect skipped region counts
4. ✅ Different border percentages affect EasyOCR detection counts
5. ✅ Console logs show parameters at every stage
6. ✅ Downloaded images show visual differences

**If ALL criteria are met, parameters ARE being applied to the models!**

---

## 🎯 Quick Test Script

Run this in browser console after uploading an image:

```javascript
// Test 1: Low threshold
document.querySelector('#conf_threshold').value = 0.1
document.querySelector('#conf_threshold').dispatchEvent(new Event('input', { bubbles: true }))

// Test 2: High threshold
document.querySelector('#conf_threshold').value = 0.9
document.querySelector('#conf_threshold').dispatchEvent(new Event('input', { bubbles: true }))

// Compare results in console logs
```

---

## 📞 Need Help?

If parameters still don't seem to work:

1. Check browser DevTools → Network tab → Request payload
2. Check backend console for `[DEBUG] OCR Parameters from frontend:`
3. Check FastAPI console for `[DEBUG] OCR Parameters:`
4. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
5. Compare downloaded images visually

---

**Remember:** The whole point is to PROVE that UI changes → Model changes, not just visual changes!
