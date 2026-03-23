# Medical Image Anonymization Pipeline

A production-ready, AI-powered medical image anonymization system that combines CLIP-based classification, dual OCR detection, and intelligent pixel redaction to protect patient privacy while preserving diagnostic content.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## � Full Stack Application

This project includes a complete web application:

- **React Frontend** (`client/`) — 3-role UI with dark/light themes
- **Node.js Backend** (`backend/`) — JWT auth, MongoDB, MinIO integration
- **FastAPI Pipeline** (`api/`) — 7-stage AI anonymization
- **MinIO S3** — Secure image storage with category organization

See [MERN_SETUP.md](MERN_SETUP.md) for setup instructions.

---

## �🎯 Features

- **7-Stage Processing Pipeline**: Classification → Validation → Metadata Anonymization → Preprocessing → Dual OCR → Redaction → Save
- **AI-Powered Classification**: CLIP model detects medical vs non-medical images
- **Dual OCR Detection**: PaddleOCR + EasyOCR with IoU-based deduplication (~25% better detection)
- **Safe Redaction**: Border-only pixel redaction protects diagnostic content
- **Format Support**: DICOM (.dcm), JPEG, PNG, TIFF, BMP
- **DICOM Metadata**: Anonymizes 12 PHI tags automatically
- **Docker Ready**: Fully containerized for reproducible deployments

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start (Docker)](#quick-start-docker)
- [Local Installation](#local-installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## 🏗️ Architecture

### 7-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Classification (CLIP)                             │
│  ├─ Detect medical vs non-medical                           │
│  └─ Classify anatomy: chest, skull, dental, pelvic, other   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Validation                                         │
│  ├─ DICOM: Check required tags                              │
│  └─ Images: Validate pixel data                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Metadata Anonymization (DICOM only)               │
│  ├─ Replace 12 PHI tags with "ANONYMIZED"                   │
│  └─ Remove overlay planes                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Border Preprocessing                              │
│  ├─ Apply CLAHE to border regions only (15%)                │
│  └─ Enhance faint text visibility for OCR                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Dual OCR Detection                                │
│  ├─ PaddleOCR: Fast, accurate for standard text             │
│  ├─ EasyOCR: Small text in borders (KV, mA values)          │
│  └─ IoU deduplication: Merge and remove overlaps            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 6: Pixel Redaction                                   │
│  ├─ Border-only redaction (100px margin default)            │
│  ├─ OpenCV TELEA inpainting                                 │
│  └─ Central regions skipped → manual review                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 7: Save Output                                       │
│  ├─ DICOM: Preserve format with pydicom                     │
│  └─ Images: Save with quality=95 (JPEG)                     │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| AI Classification | OpenAI CLIP (vit-base-patch32) |
| Primary OCR | PaddleOCR (PP-OCRv4) |
| Secondary OCR | EasyOCR |
| Image Processing | OpenCV (CLAHE, inpainting) |
| Medical Imaging | pydicom |
| Deep Learning | PyTorch, Transformers |

---

## 🚀 Quick Start (Docker)

### Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (included with Docker Desktop)

### 1. Build the Docker Image

```bash
docker build -t medical-anonymizer .
```

**Build time:** ~5-10 minutes (downloads models and dependencies)

### 2. Run the Pipeline

#### Option A: Direct Docker Run

```bash
# Basic usage
docker run medical-anonymizer input.jpg output/

# With volume mounts (recommended)
docker run -v $(pwd)/input:/app/input \
           -v $(pwd)/output:/app/output \
           medical-anonymizer input/patient.dcm output/

# With custom parameters
docker run -v $(pwd)/input:/app/input \
           -v $(pwd)/output:/app/output \
           medical-anonymizer input/scan.jpg output/ --confidence 0.2 --verbose
```

#### Option B: Docker Compose

```bash
# Create input/output directories
mkdir -p input output

# Place your images in input/
cp your_image.jpg input/

# Run pipeline
docker-compose run anonymizer input/your_image.jpg output/

# With options
docker-compose run anonymizer input/patient.dcm output/ --verbose
```

### 3. View Results

```bash
ls output/
# Output: anonymized_your_image.jpg
```

---

## 💻 Local Installation

### Prerequisites

- Python 3.11
- pip

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/medical-anonymizer.git
cd medical-anonymizer
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important:** numpy must be <2.0 for EasyOCR compatibility.

### 4. Set Environment Variables (Windows)

```powershell
$env:FLAGS_use_mkldnn="0"
$env:FLAGS_use_pir_api="0"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

**Linux/Mac:**

```bash
export FLAGS_use_mkldnn=0
export FLAGS_use_pir_api=0
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 5. Run Pipeline

```bash
# Windows
venv\Scripts\python.exe pipeline_run.py input.jpg output/

# Linux/Mac
python pipeline_run.py input.jpg output/
```

---

## 📖 Usage

### Command Line Interface

```bash
python pipeline_run.py <input_file> <output_dir> [options]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input_file` | str | required | Path to input file (.dcm, .jpg, .png, etc.) |
| `output_dir` | str | required | Directory for anonymized output |
| `--confidence` | float | 0.1 | OCR confidence threshold (0.0-1.0) |
| `--padding` | int | 5 | Padding around text regions (pixels) |
| `--margin` | int | 100 | Border margin for safe redaction (pixels) |
| `--verbose` | flag | false | Enable DEBUG logging |

### Examples

#### Basic Usage

```bash
# DICOM file
docker run -v $(pwd):/app medical-anonymizer patient.dcm output/

# JPEG image
docker run -v $(pwd):/app medical-anonymizer scan.jpg output/
```

#### Custom Parameters

```bash
# Lower confidence threshold (more detections)
docker run -v $(pwd):/app medical-anonymizer image.jpg output/ --confidence 0.05

# Larger safety margin
docker run -v $(pwd):/app medical-anonymizer xray.dcm output/ --margin 150

# Verbose logging
docker run -v $(pwd):/app medical-anonymizer image.png output/ --verbose
```

#### Batch Processing

```bash
# Process all DICOM files in a directory
for file in input/*.dcm; do
    docker run -v $(pwd):/app medical-anonymizer "$file" output/
done
```

### Output Format

```
  ╔═══════════════════════════════════════════════════════════╗
  ║                  ANONYMIZATION COMPLETE                   ║
  ╠═══════════════════════════════════════════════════════════╣
  ║ Classification:    chest (1.00)                           ║
  ║ Format:            JPEG                                   ║
  ║ Metadata cleaned:  0 (not DICOM)                          ║
  ║ PaddleOCR regions: 3                                      ║
  ║ EasyOCR regions:   4                                      ║
  ║ After merge:       4                                      ║
  ║ Redacted:          4                                      ║
  ║ Skipped (central): 0                                      ║
  ║ Output:            output/anonymized_image.jpeg           ║
  ╚═══════════════════════════════════════════════════════════╝
```

---

## 🔧 Pipeline Stages

### Stage 1: Classification

- **Purpose:** Verify image is medical
- **Model:** OpenAI CLIP (vit-base-patch32)
- **Output:** Category + confidence
- **Rejection:** Non-medical images exit with code 1

### Stage 2: Validation

- **DICOM:** Checks Modality, PatientID, PixelData tags
- **Images:** Validates pixel array extraction
- **Formats:** .dcm, .dicom, .jpg, .jpeg, .png, .bmp, .tiff, .tif

### Stage 3: Metadata Anonymization

- **DICOM Only:** Replaces 12 PHI tags
- **Tags:** PatientName, PatientID, PatientBirthDate, etc.
- **Value:** "ANONYMIZED"

### Stage 4: Preprocessing

- **Algorithm:** CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Region:** Border only (15% from edges)
- **Purpose:** Enhance faint text for OCR

### Stage 5: Dual OCR

- **PaddleOCR:** Fast, accurate for standard text
- **EasyOCR:** Specialized for small border text
- **Deduplication:** IoU threshold 50%
- **Improvement:** ~25% more detections vs single OCR

### Stage 6: Redaction

- **Algorithm:** OpenCV TELEA inpainting
- **Safety:** Border-only (100px margin default)
- **Radius:** 3 pixels
- **Protection:** Central diagnostic content never auto-redacted

### Stage 7: Save

- **DICOM:** Preserves format with pydicom
- **Images:** Quality=95 for JPEG
- **Naming:** `anonymized_<original_filename>`

---

## ⚙️ Configuration

### Environment Variables

Set these before running (already configured in Docker):

```bash
FLAGS_use_mkldnn=0
FLAGS_use_pir_api=0
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
KMP_DUPLICATE_LIB_OK=TRUE
```

### Docker Resource Limits

Edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

### Model Caching

Models are downloaded on first run:

- **CLIP:** `~/.cache/huggingface/`
- **PaddleOCR:** `~/.paddleocr/`
- **EasyOCR:** `~/.EasyOCR/`

To persist models between container runs:

```bash
docker run -v ~/.cache:/root/.cache \
           -v $(pwd)/input:/app/input \
           -v $(pwd)/output:/app/output \
           medical-anonymizer input/image.jpg output/
```

---

## 🐛 Troubleshooting

### First Run is Slow

**Issue:** First execution takes 5-10 minutes

**Cause:** Downloading CLIP, PaddleOCR, and EasyOCR models (~2GB total)

**Solution:** Models are cached. Subsequent runs take ~12-15 seconds.

### EasyOCR Not Working

**Issue:** `WARNING - EasyOCR failed (non-fatal)`

**Cause:** EasyOCR not installed or numpy version conflict

**Solution:** 
- Ensure numpy<2.0: `pip install "numpy<2.0" --force-reinstall`
- Pipeline continues with PaddleOCR only (graceful degradation)

### Non-Medical Image Rejected

**Issue:** `Error: Image classified as non-medical`

**Cause:** CLIP model detected non-medical content

**Solution:** This is expected behavior. Pipeline only processes medical images.

### Permission Denied (Docker)

**Issue:** Cannot write to output directory

**Solution:** 
```bash
# Linux/Mac: Fix permissions
chmod 777 output/

# Or run with user ID
docker run -u $(id -u):$(id -g) -v $(pwd):/app medical-anonymizer input.jpg output/
```

### Out of Memory

**Issue:** Container crashes during processing

**Solution:** Increase Docker memory limit in Docker Desktop settings or docker-compose.yml

### DICOM Tags Not Anonymized

**Issue:** Metadata still contains PHI

**Cause:** Non-DICOM file or missing tags

**Solution:** Verify file is valid DICOM with `pydicom.dcmread()`

---

## 📁 Project Structure

```
medical-anonymizer/
├── medical_anonymizer/          # AI Classification
│   ├── improved_medical_classifier.py
│   ├── evaluate_accuracy.py
│   └── clip-test.py
├── anonymizer/                  # Core Anonymization
│   ├── image_validator.py
│   ├── metadata_anonymizer.py
│   ├── pixel_redactor.py
│   └── pipeline.py
├── ocr/                         # Dual OCR System
│   ├── text_detector.py         # PaddleOCR
│   ├── easy_text_detector.py    # EasyOCR
│   └── preprocessor.py          # CLAHE
├── pipeline_run.py              # Main CLI Entry Point
├── detect.py                    # Standalone OCR Test
├── Dockerfile                   # Docker build config
├── docker-compose.yml           # Docker orchestration
├── requirements.txt             # Python dependencies
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
└── README.md                   # This file
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Processing Time | ~12-15 seconds per image |
| Detection Improvement | +25% with dual OCR |
| Model Download (first run) | ~2GB, 5-10 minutes |
| Docker Image Size | ~3.5GB |
| Memory Usage | ~2-4GB |

---

## 🔒 Security & Privacy

- **No Data Retention:** Images are not stored in the container
- **Local Processing:** All computation happens locally
- **No External Calls:** Except model downloads (first run only)
- **PHI Protection:** 12 DICOM tags anonymized automatically
- **Safe Redaction:** Border-only to prevent diagnostic data loss

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Support

For issues or questions:

- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 🙏 Acknowledgments

- **OpenAI CLIP** - Image classification
- **PaddleOCR** - Primary OCR engine
- **EasyOCR** - Secondary OCR engine
- **pydicom** - DICOM handling

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{medical_anonymizer_2026,
  title={Medical Image Anonymization Pipeline},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/medical-anonymizer}
}
```

---

**Built with ❤️ for medical imaging privacy**
