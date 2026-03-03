# Frontend Setup Instructions

## Quick Start

### 1. Start the FastAPI Backend

```bash
# From project root (PFE_Test/)
cd c:\Users\MSI\Desktop\PFE_Test

# Activate virtual environment
venv\Scripts\activate

# Start the API server
uvicorn api.main:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 2. Open the Frontend

Simply open `frontend/index.html` in your web browser:

**Option A: Double-click**
- Navigate to `c:\Users\MSI\Desktop\PFE_Test\frontend\`
- Double-click `index.html`

**Option B: Direct path in browser**
```
file:///c:/Users/MSI/Desktop/PFE_Test/frontend/index.html
```

### 3. Test the Pipeline

1. **Drag and drop** or **click to upload** a medical image:
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.dcm`, `.dicom`
   - Test with: `person49_virus_101.jpeg` or `person1656_virus_2862.jpeg`

2. **Click "Anonymize Image"**
   - The 7-stage pipeline will execute
   - Progress spinner shows while processing

3. **View Results**
   - Original image on left
   - Anonymized image on right
   - Stats panel shows all pipeline metrics

4. **Download**
   - Click "Download Anonymized Image" to save the result

---

## API Endpoints

### POST /anonymize
Upload and anonymize a medical image.

**Request:**
```bash
curl -X POST "http://localhost:8000/anonymize" \
  -F "file=@person49_virus_101.jpeg"
```

**Response (Success):**
```json
{
  "status": "success",
  "classification": "chest x-ray",
  "confidence": 0.93,
  "format": "JPEG",
  "tags_anonymized": 0,
  "paddle_regions": 6,
  "easy_regions": 5,
  "total_regions": 6,
  "redacted": 6,
  "skipped": 0,
  "output_filename": "anonymized_person49_virus_101.jpeg"
}
```

**Response (Non-Medical):**
```json
{
  "status": "failed",
  "error": "non_medical",
  "message": "Image classified as non-medical (confidence: 0.92)",
  "classification": "non_medical",
  "confidence": 0.92
}
```

**Response (Other Medical):**
```json
{
  "status": "failed",
  "error": "other_medical",
  "message": "Image type not supported for automatic anonymization",
  "classification": "other_medical",
  "confidence": 0.94
}
```

### GET /result/{filename}
Download or view anonymized result.

```bash
curl "http://localhost:8000/result/anonymized_person49_virus_101.jpeg" \
  --output result.jpeg
```

### GET /health
Check API health status.

```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "ok",
  "services": {
    "classifier": "CLIP ready",
    "ocr": "PaddleOCR + EasyOCR",
    "redaction": "OpenCV inpainting"
  }
}
```

---

## Project Structure

```
PFE_Test/
│
├── api/                          ← NEW: FastAPI backend
│   ├── __init__.py
│   ├── main.py                   ← Main API application
│   └── temp/                     ← Temporary uploaded files
│
├── frontend/                     ← NEW: Web interface
│   └── index.html                ← Single-page application
│
├── output/
│   └── results/                  ← Anonymized output files
│
├── ocr/                          ← Unchanged
├── anonymizer/                   ← Unchanged
├── image_classifier/             ← Unchanged
├── pipeline_run.py               ← Unchanged (used by API)
└── requirements.txt
```

---

## Troubleshooting

### Issue: "Failed to connect to the API server"

**Cause:** FastAPI server not running

**Solution:**
```bash
# Make sure server is running
uvicorn api.main:app --reload --port 8000
```

### Issue: "CORS error" in browser console

**Cause:** CORS middleware not configured

**Solution:** Already configured in `api/main.py`. If still occurs, check browser console for details.

### Issue: "ModuleNotFoundError: No module named 'fastapi'"

**Cause:** FastAPI not installed

**Solution:**
```bash
pip install fastapi uvicorn python-multipart
```

### Issue: Images not displaying in frontend

**Cause:** Result files not accessible

**Solution:** Check that `output/results/` directory exists and contains the files.

### Issue: "Non-medical image detected" for medical images

**Cause:** CLIP classifier may misclassify edge cases

**Solution:** This is expected behavior. Only chest, skull, dental, and pelvic X-rays are supported.

---

## Testing Workflow

### Test 1: Chest X-Ray (Should Succeed)
```bash
# File: person49_virus_101.jpeg
Expected: Success, 6 regions redacted
```

### Test 2: Non-Medical Image (Should Reject)
```bash
# File: cat.jpg (from dataset/non_medical/)
Expected: Error - "Image classified as non-medical"
```

### Test 3: Other Medical (Should Reject)
```bash
# File: 10025.png (hand X-ray)
Expected: Error - "Image type not supported"
```

### Test 4: DICOM File (Should Succeed)
```bash
# File: Any .dcm file
Expected: Success, metadata tags anonymized
```

---

## Performance Notes

- **First run:** 5-10 minutes (downloads CLIP, PaddleOCR, EasyOCR models ~2GB)
- **Subsequent runs:** 12-15 seconds per image
- **Models cached:** `~/.cache/huggingface/`, `~/.paddleocr/`, `~/.EasyOCR/`

---

## API Documentation

Once the server is running, visit:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Next Steps

1. ✅ Start FastAPI server
2. ✅ Open frontend in browser
3. ✅ Upload test image
4. ✅ View anonymized result
5. ✅ Download output file

**Your frontend is ready to use!** 🎉
