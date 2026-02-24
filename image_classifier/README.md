# Medical Image Anonymization API

PFE-level REST API for anonymizing medical DICOM images by detecting and removing text overlays containing patient-identifying information (PHI).

## Features

- **DICOM Support**: Load and save medical images in DICOM format
- **Text Detection**: OCR-based text detection with PaddleOCR + fallback methods
- **Anonymization**: Multiple algorithms (OpenCV TELEA, Navier-Stokes, Mean replacement)
- **Soft Validation**: Non-blocking validation and quality checks
- **FastAPI**: Production-ready REST API with auto-generated docs

## Project Structure

```
medical_anonymizer/
├── app/
│   ├── services/
│   │   ├── dicom_handler.py          # DICOM loading/saving
│   │   ├── ocr_service.py            # Text detection (PaddleOCR + fallback)
│   │   ├── anonymization_service.py  # Image anonymization
│   │   └── guard_service.py          # Soft validation
│   └── main.py                       # FastAPI application
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Navigate to project directory
```bash
cd medical_anonymizer
```

### 2. Start the API server
```bash
# Using the virtual environment from PFE_Test
..\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or if you activated the virtual environment:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Docs**: http://localhost:8000/docs
- **API Info**: http://localhost:8000/info
- **Health Check**: http://localhost:8000/health

### 4. Test Anonymization

Using curl:
```bash
curl -X POST "http://localhost:8000/anonymize" \
  -H "accept: application/dicom" \
  -F "file=@test_sample.dcm" \
  -F "method=telea" \
  --output anonymized.dcm
```

Or using the interactive docs at http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API root info |
| `/health` | GET | Service health check |
| `/info` | GET | API capabilities |
| `/anonymize` | POST | Anonymize DICOM file |

## Anonymization Methods

- **`auto`** (default): Automatically selects best available method
- **`telea`: OpenCV TELEA inpainting (recommended)
- **`ns`**: OpenCV Navier-Stokes inpainting
- **`mean`**: Aggressive mean replacement

## Architecture

### Pipeline Flow

1. **Load DICOM** → Extract pixel array
2. **Validate** → Soft validation (non-blocking)
3. **Detect Text** → OCR detection with fallback
4. **Anonymize** → Remove text regions
5. **Save DICOM** → Store anonymized result

### Services

**DICOMHandler**: Loads/saves DICOM files, handles pixel normalization (16-bit → 8-bit → 16-bit), anonymizes metadata tags.

**OCRService**: Detects text using PaddleOCR (primary) or image processing fallback (white/dark text detection + common header/footer regions).

**AnonymizationService**: Removes text via inpainting with automatic fallback chain (TELEA → NS → Mean replacement).

**GuardService**: Soft validation providing warnings and quality metrics without blocking processing.

## Dependencies

**Already Installed**:
- pydicom, numpy, opencv-python
- paddleocr + paddlepaddle (OCR engine)

**Newly Installed**:
- fastapi, uvicorn, python-multipart (API framework)

**Optional**:
- simple-lama-inpainting (better quality, heavy)

## Error Handling

All services implement fallback mechanisms:
- PaddleOCR fails → Use image processing fallback
- LaMa unavailable → Use OpenCV inpainting
- All inpainting fails → Use mean replacement
- Validation fails → Continue with warnings

## Production Notes

- Never stores non-anonymized images
- Temp files cleaned up automatically
- All errors caught and returned as HTTP responses
- Logging provides detailed pipeline tracking

## Example Response

```json
{
  "original_filename": "patient_scan.dcm",
  "anonymized": true,
  "method_used": "telea",
  "text_regions_detected": 3,
  "pixels_modified": 12543,
  "confidence_score": 0.87,
  "validation": {
    "performed": true,
    "quality_score": 0.92,
    "warnings": 0
  }
}
```

## PFE Integration

Replace the basic detection in `ocr_service.py` with your advanced model:

```python
def _detect_with_advanced_model(self, image):
    # TODO: Integrate your trained model here
    # Return list of TextRegion objects
    pass
```

For LaMa inpainting, uncomment in `requirements.txt`:
```
simple-lama-inpainting>=0.1.0
```

And set `use_lama=True` in `AnonymizationService` initialization.

## Testing

Run API tests:
```bash
pytest app/tests/ -v
```

Manual test with sample DICOM:
```bash
python -c "
import requests
with open('test_sample.dcm', 'rb') as f:
    response = requests.post('http://localhost:8000/anonymize', files={'file': f})
    print(f'Status: {response.status_code}')
    print(f'Regions: {response.headers.get(\"X-Anonymization-Metadata\")}')
"
```

## License

PFE Project - Medical Image Anonymization
