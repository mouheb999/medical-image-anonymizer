# HOW TO RUN:
# 1. Start MinIO:     docker-compose up -d minio
# 2. Start API:       uvicorn api.main:app --reload --port 8000
# 3. MinIO Console:   http://localhost:9001 (minioadmin/minioadmin)
# 4. Open frontend:   frontend/index.html

"""
Medical Image Anonymization API - Full Pipeline Integration

This FastAPI application provides a REST API for the complete medical image
anonymization pipeline, supporting JPEG, PNG, and DICOM formats.
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set PaddleOCR environment variables before imports
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import pipeline components
from image_classifier.improved_medical_classifier import MedicalImageClassifier
from anonymizer import ImageValidator, ImageValidationError, MetadataAnonymizer, PixelRedactor
from ocr import TextDetector, BorderPreprocessor, EasyTextDetector
from services.pathology import PathologyDetector, generate_heatmap
import numpy as np
import base64
import io
import traceback
from PIL import Image
import pydicom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Image Anonymization API",
    description="AI-powered medical image anonymization with CLIP classification and dual OCR",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directories
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for serving results
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR.parent)), name="static")


def merge_and_deduplicate(paddle_regions, easy_regions, iou_threshold=0.5):
    """Merge OCR results and remove duplicates using IoU."""
    def compute_iou(box1, box2):
        x1_min, y1_min = box1[:, 0].min(), box1[:, 1].min()
        x1_max, y1_max = box1[:, 0].max(), box1[:, 1].max()
        x2_min, y2_min = box2[:, 0].min(), box2[:, 1].min()
        x2_max, y2_max = box2[:, 0].max(), box2[:, 1].max()
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    merged = list(paddle_regions)
    
    for easy_region in easy_regions:
        is_duplicate = False
        for paddle_region in paddle_regions:
            iou = compute_iou(easy_region, paddle_region)
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(easy_region)
    
    return merged


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "Medical Image Anonymization API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "frontend": "Open frontend/index.html in browser"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "services": {
            "classifier": "CLIP ready",
            "ocr": "PaddleOCR + EasyOCR",
            "redaction": "OpenCV inpainting",
            "pathology": "TorchXRayVision DenseNet-121 (loaded)" if _pathology_detector else "TorchXRayVision DenseNet-121 (lazy)"
        }
    }


@app.post("/anonymize")
async def anonymize_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.1),
    padding: int = Form(5),
    border_margin: int = Form(100),
    border_pct: float = Form(0.20)
):
    """
    Anonymize a medical image using the full 7-stage pipeline.
    
    Accepts: JPEG, PNG, DICOM
    Parameters:
        - conf_threshold: OCR confidence threshold (0.0-1.0)
        - padding: Redaction padding in pixels
        - border_margin: Border safety margin in pixels
        - border_pct: Border scan percentage for EasyOCR (0.0-1.0)
    Returns: JSON with anonymization results and output filename
    """
    import time
    import traceback
    start_time = time.time()
    
    logger.info(f"Parameters received: conf_threshold={conf_threshold}, padding={padding}, border_margin={border_margin}, border_pct={border_pct}")
    print(f"[DEBUG] OCR Parameters: conf_threshold={conf_threshold}, padding={padding}, border_margin={border_margin}, border_pct={border_pct}")
    
    logger.info(f"Received anonymization request: {file.filename}")
    print(f"[DEBUG] File received: {file.filename}")
    
    # Validate file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.dcm', '.dicom', '.bmp', '.tiff', '.tif'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file to temp
    temp_input = TEMP_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    
    try:
        content = await file.read()
        with open(temp_input, 'wb') as f:
            f.write(content)
        
        logger.info(f"Saved temp file: {temp_input}")
        print(f"[DEBUG] Temp file saved: {temp_input}")
        print(f"[TIMING] File save took: {time.time() - start_time:.2f}s")
        
        # === STAGE 1: Classification ===
        logger.info("STAGE 1: Classification")
        print(f"[DEBUG] Starting classification...")
        stage_start = time.time()
        classifier = MedicalImageClassifier()
        category, confidence, metadata = classifier.classify_image(str(temp_input))
        print(f"[DEBUG] Classification done: {category} {confidence}")
        print(f"[TIMING] Stage 1 (Classification) took: {time.time() - stage_start:.2f}s")
        
        category_lower = category.lower()
        
        # Check for non-medical or unsupported
        if (category == "non_medical" or 
            "non" in category_lower or 
            "not medical" in category_lower):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "error": "non_medical",
                    "message": f"Image classified as non-medical (confidence: {confidence:.2f})",
                    "classification": category,
                    "confidence": float(confidence)
                }
            )
        
        if (category == "other_medical" or
            "rejected" in category_lower or
            "other medical" in category_lower):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "error": "other_medical",
                    "message": "Image type not supported for automatic anonymization",
                    "classification": category,
                    "confidence": float(confidence)
                }
            )
        
        # Check for classification failure
        if (category is None or 
            "error" in str(category).lower() or 
            "failed" in str(category).lower() or 
            "unable" in str(category).lower()):
            raise ValueError(
                "Classification failed — cannot process this file. "
                "If this is a compressed DICOM file, install missing "
                "dependencies: pip install pylibjpeg pylibjpeg-libjpeg"
            )
        
        logger.info(f"Classification: {category} ({confidence:.2f})")
        
        # === STAGE 2: Validation ===
        logger.info("STAGE 2: Validation")
        print(f"[DEBUG] Starting validation...")
        stage_start = time.time()
        validator = ImageValidator()
        validation_result = validator.validate(str(temp_input))
        print(f"[DEBUG] Validation done")
        print(f"[TIMING] Stage 2 (Validation) took: {time.time() - stage_start:.2f}s")
        
        is_dicom = validation_result.is_dicom
        dataset = validation_result.dataset
        print(f"[DEBUG] is_dicom={is_dicom}")
        
        # Safe pixel extraction for DICOM with compression handling
        print(f"[DEBUG] Extracting pixel array...")
        if is_dicom:
            from anonymizer.dicom_decompressor import decompress_dicom
            try:
                pixel_array = decompress_dicom(dataset)
            except Exception as e:
                raise ValueError(
                    f"Cannot read DICOM pixel data: {e}. "
                    f"Install missing codecs: "
                    f"pip install python-gdcm"
                )
        else:
            pixel_array = validation_result.pixel_array
        
        print(f"[DEBUG] Pixel array: {pixel_array.shape if pixel_array is not None else 'None'}")
        
        # Generate original preview for DICOM (before any processing)
        original_preview_filename = None
        if is_dicom:
            try:
                from anonymizer.dicom_decompressor import normalize_to_uint8
                
                orig_preview_uint8 = normalize_to_uint8(pixel_array)
                
                if len(orig_preview_uint8.shape) == 2:
                    orig_preview_image = Image.fromarray(orig_preview_uint8, mode='L')
                else:
                    orig_preview_image = Image.fromarray(orig_preview_uint8)
                
                original_preview_filename = f"original_preview_{file.filename.replace('.dcm', '').replace('.dicom', '')}.png"
                orig_preview_path = OUTPUT_DIR / original_preview_filename
                orig_preview_image.save(str(orig_preview_path))
                
                logger.info(f"Original DICOM preview saved: {orig_preview_path}")
            except Exception as e:
                logger.warning(f"Original preview generation failed (non-fatal): {e}")
                original_preview_filename = None
        
        image_format = "DICOM" if is_dicom else Path(file.filename).suffix.upper().replace('.', '')
        logger.info(f"Format: {image_format}")
        
        # === STAGE 3: Metadata Anonymization (DICOM only) ===
        tags_anonymized = 0
        if is_dicom:
            logger.info("STAGE 3: Metadata Anonymization")
            print(f"[DEBUG] Starting metadata anonymization...")
            stage_start = time.time()
            meta_anonymizer = MetadataAnonymizer()
            dataset, tags_anonymized = meta_anonymizer.anonymize(dataset)
            logger.info(f"Anonymized {tags_anonymized} PHI tags")
            print(f"[DEBUG] Metadata anonymization done: {tags_anonymized} tags")
            print(f"[TIMING] Stage 3 (Metadata) took: {time.time() - stage_start:.2f}s")
        
        # === STAGE 4: Preprocessing ===
        logger.info("STAGE 4: Preprocessing")
        print(f"[DEBUG] Starting preprocessing...")
        stage_start = time.time()
        
        # Convert to RGB uint8 for OCR
        # Check for None pixel_array before accessing shape
        if pixel_array is None:
            raise ValueError(
                "Failed to extract pixel data from DICOM file. "
                "The file uses compressed pixel data that requires "
                "additional libraries. "
                "Run: pip install pylibjpeg pylibjpeg-libjpeg"
            )
        
        if len(pixel_array.shape) == 2:
            # Grayscale image - convert to RGB
            pixel_array_rgb = np.stack([pixel_array] * 3, axis=-1)
        elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 1:
            # Single channel image - convert to RGB
            pixel_array_rgb = np.concatenate([pixel_array] * 3, axis=-1)
        else:
            # Already RGB
            pixel_array_rgb = pixel_array
        
        if pixel_array_rgb.dtype != np.uint8:
            pixel_array_rgb = ((pixel_array_rgb - pixel_array_rgb.min()) / 
                              (pixel_array_rgb.max() - pixel_array_rgb.min()) * 255).astype(np.uint8)
        
        preprocessor = BorderPreprocessor(border_pct=0.15)
        enhanced_image = preprocessor.enhance(pixel_array_rgb)
        logger.info("Border CLAHE enhancement applied")
        print(f"[DEBUG] Preprocessing done")
        print(f"[TIMING] Stage 4 (Preprocessing) took: {time.time() - stage_start:.2f}s")
        
        # === STAGE 5: Dual OCR Detection ===
        logger.info("STAGE 5: Dual OCR Detection")
        print(f"[DEBUG] Starting PaddleOCR with conf_threshold={conf_threshold}...")
        stage_start = time.time()
        paddle_detector = TextDetector(lang="en", conf_threshold=conf_threshold)
        paddle_regions = paddle_detector.detect_text(enhanced_image)
        paddle_count = len(paddle_regions)
        print(f"[DEBUG] PaddleOCR done: {paddle_count} regions (threshold={conf_threshold})")
        print(f"[TIMING] PaddleOCR took: {time.time() - stage_start:.2f}s")
        
        easy_regions = []
        easy_count = 0
        try:
            print(f"[DEBUG] Starting EasyOCR with conf_threshold={conf_threshold}, border_pct={border_pct}...")
            easy_start = time.time()
            easy_detector = EasyTextDetector(conf_threshold=conf_threshold, border_pct=border_pct)
            easy_regions = easy_detector.detect_text(enhanced_image)
            easy_count = len(easy_regions)
            print(f"[DEBUG] EasyOCR done: {easy_count} regions (threshold={conf_threshold}, border_pct={border_pct})")
            print(f"[TIMING] EasyOCR took: {time.time() - easy_start:.2f}s")
        except Exception as e:
            logger.warning(f"EasyOCR failed (non-fatal): {e}")
            print(f"[DEBUG] EasyOCR failed: {e}")
        
        merged_regions = merge_and_deduplicate(paddle_regions, easy_regions, iou_threshold=0.5)
        merged_count = len(merged_regions)
        
        logger.info(f"PaddleOCR: {paddle_count} | EasyOCR: {easy_count} | Merged: {merged_count}")
        print(f"[DEBUG] Region merging done: {merged_count} total regions")
        print(f"[TIMING] Stage 5 (OCR) total took: {time.time() - stage_start:.2f}s")
        
        # === STAGE 6: Pixel Redaction ===
        logger.info("STAGE 6: Pixel Redaction")
        print(f"[DEBUG] Starting pixel redaction with padding={padding}, border_margin={border_margin}...")
        stage_start = time.time()
        
        redactor = PixelRedactor(padding=padding, border_margin=border_margin)
        
        if merged_count > 0:
            if is_dicom:
                redacted_data, redacted_count = redactor.redact(
                    dataset, merged_regions,
                    padding=padding, border_margin=border_margin,
                    redact_all_regions=True
                )
                dataset = redacted_data
            else:
                redacted_data, redacted_count = redactor.redact(
                    pixel_array, merged_regions,
                    padding=padding, border_margin=border_margin,
                    redact_all_regions=True
                )
                pixel_array = redacted_data
        else:
            redacted_count = 0
        
        skipped_count = merged_count - redacted_count
        logger.info(f"Redacted: {redacted_count}, Skipped: {skipped_count}")
        print(f"[DEBUG] Redaction done: {redacted_count} redacted, {skipped_count} skipped")
        print(f"[TIMING] Stage 6 (Redaction) took: {time.time() - stage_start:.2f}s")
        
        # === STAGE 7: Save Output ===
        logger.info("STAGE 7: Save Output")
        print(f"[DEBUG] Starting output save...")
        stage_start = time.time()
        
        output_filename = f"anonymized_{file.filename}"
        output_path = OUTPUT_DIR / output_filename
        preview_filename = None  # Initialize for non-DICOM files
        ###### For DICOM, save the modified dataset. For images, save the redacted pixel array.
        if is_dicom:
            pydicom.dcmwrite(str(output_path), dataset)
            
            # Generate PNG preview for DICOM
            try:
                from anonymizer.dicom_decompressor import (
                    decompress_dicom, normalize_to_uint8
                )
                
                preview_pixels = decompress_dicom(dataset)
                preview_uint8 = normalize_to_uint8(preview_pixels)
                
                if len(preview_uint8.shape) == 2:
                    preview_image = Image.fromarray(preview_uint8, mode='L')
                else:
                    preview_image = Image.fromarray(preview_uint8)
                
                preview_filename = output_path.stem + "_preview.png"
                preview_path = output_path.parent / preview_filename
                preview_image.save(str(preview_path))
                
                logger.info(f"DICOM preview saved: {preview_path}")
            except Exception as e:
                logger.warning(f"Preview generation failed (non-fatal): {e}")
                preview_filename = None
        else:
            # Convert to correct mode before saving
            if len(pixel_array.shape) == 2:
                pil_image = Image.fromarray(pixel_array, mode='L')
            else:
                pil_image = Image.fromarray(pixel_array)
            
            # RGBA cannot be saved as JPEG — convert to RGB
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            
            # Grayscale cannot be saved as JPEG directly in some cases
            if pil_image.mode == 'L' and file_ext in ['.jpg', '.jpeg']:
                pil_image = pil_image.convert('RGB')
            
            if file_ext in ['.jpg', '.jpeg']:
                pil_image.save(str(output_path), 'JPEG', quality=95)
            elif file_ext == '.png':
                pil_image.save(str(output_path), 'PNG')
            else:
                pil_image.save(str(output_path))
        ##################################################################################################
        
        logger.info(f"Saved: {output_path}")
        print(f"[DEBUG] Output saved: {output_path}")
        print(f"[TIMING] Stage 7 (Save) took: {time.time() - stage_start:.2f}s")
        
        # Upload to MinIO
        minio_uri = None
        download_url = None
        print(f"[DEBUG] Starting MinIO upload...")
        minio_start = time.time()
        try:
            from api.storage import MinIOStorage
            from api.config import settings
            
            storage = MinIOStorage(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket_name=settings.minio_bucket,
                secure=settings.minio_secure
            )
            
            minio_uri = storage.upload_file(str(output_path), category=category)
            
            # Extract object name from URI for presigned URL
            # minio://localhost:9000/bucket/2026/02/25/file.jpg -> 2026/02/25/file.jpg
            object_name = minio_uri.split(f"/{settings.minio_bucket}/", 1)[1]
            download_url = storage.get_url(object_name, expires_hours=24)
            
            logger.info(f"Uploaded to MinIO: {minio_uri}")
            print(f"[DEBUG] MinIO upload done: {minio_uri}")
            print(f"[TIMING] MinIO upload took: {time.time() - minio_start:.2f}s")
        except Exception as e:
            logger.warning(f"MinIO upload failed (non-fatal): {e}")
            print(f"[DEBUG] MinIO upload failed: {e}")
            minio_uri = None
            download_url = None
        
        # Clean up temp file
        temp_input.unlink()
        print(f"[DEBUG] Temp file cleaned up")
        
        # Return success response
        print(f"[DEBUG] Building response...")
        response_data = {
            "status": "success",
            "classification": category,
            "confidence": float(confidence),
            "format": image_format,
            "tags_anonymized": tags_anonymized,
            "paddle_regions": paddle_count,
            "easy_regions": easy_count,
            "total_regions": merged_count,
            "redacted": redacted_count,
            "skipped": skipped_count,
            "output_filename": output_filename,
            "preview_filename": preview_filename,
            "original_preview_filename": original_preview_filename
        }
        
        # Add MinIO fields if upload succeeded
        if minio_uri:
            response_data["minio_uri"] = minio_uri
            response_data["download_url"] = download_url
        
        print(f"[DEBUG] Returning response")
        print(f"[TIMING] Total processing time: {time.time() - start_time:.2f}s")
        return response_data
        
    except ImageValidationError as e:
        logger.error(f"Validation error: {e}")
        if temp_input.exists():
            temp_input.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation failed: {str(e)}"
        )
    
    except Exception as e:
        import traceback
        print(f"[ERROR] Full traceback:")
        traceback.print_exc()
        logger.error(f"Processing error: {e}", exc_info=True)
        if temp_input.exists():
            temp_input.unlink()
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@app.get("/result/{filename}")
async def get_result(filename: str):
    """Download or view anonymized result image."""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result file not found"
        )
    
    # Determine media type
    ext = file_path.suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.dcm': 'application/dicom',
        '.dicom': 'application/dicom'
    }
    
    media_type = media_type_map.get(ext, 'application/octet-stream')
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


# ---------------------------------------------------------------------------
# Pathology Detection — singleton (loaded once at startup)
# ---------------------------------------------------------------------------
_pathology_detector: Optional[PathologyDetector] = None


def _get_pathology_detector() -> PathologyDetector:
    """Lazy-load the pathology model on first request, then reuse."""
    global _pathology_detector
    if _pathology_detector is None:
        logger.info("Loading PathologyDetector (first request)...")
        _pathology_detector = PathologyDetector(
            confidence_threshold=0.6,   # calibrated (op_norm) scale
            max_results=2,              # top-1 / top-2 only for safety
        )
    return _pathology_detector


@app.post("/detect-pathology")
async def detect_pathology(file: UploadFile = File(...)):
    """
    Detect pathologies in a chest X-ray image.

    - Validates the image is a chest X-ray using CLIP.
    - Runs TorchXRayVision DenseNet inference.
    - Generates a Grad-CAM heatmap (returned as base64 PNG).
    - **Does NOT store the image or any sensitive data.**
    """
    temp_path = None
    try:
        # ---- 1. Read image bytes & save to temp file -------------------
        contents = await file.read()
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        temp_path = TEMP_DIR / f"pathology_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{suffix}"
        temp_path.write_bytes(contents)

        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image)

        # ---- 2. Validate: must be a chest X-ray (CLIP) ----------------
        classifier = MedicalImageClassifier()
        category, confidence, metadata = classifier.classify_image(str(temp_path))
        category_lower = category.lower()

        if "chest" not in category_lower:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Only chest X-rays are supported",
                },
            )

        # ---- 3. Pathology detection -----------------------------------
        detector = _get_pathology_detector()
        detection = detector.detect(image_np)

        # ---- 4. Grad-CAM heatmap + pseudo-localization -----------------
        model = detector.get_model()
        image_tensor = detector._preprocess(image_np)
        heatmap_result = generate_heatmap(
            model=model,
            image_tensor=image_tensor,
            original_image=image_np,
            target_class_idx=detection.get("top_class_idx"),
        )

        # Encode overlay image as base64 PNG
        heatmap_pil = Image.fromarray(heatmap_result["overlay_image"])
        buf = io.BytesIO()
        heatmap_pil.save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # ---- 5. Build response ----------------------------------------
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "pathologies": detection["pathologies"],
                "summary": detection.get("summary"),
                "heatmap": heatmap_b64,
                "bbox": heatmap_result["bbox"],
                "localization_note": heatmap_result["note"],
                "disclaimer": detection.get("disclaimer"),
                "warning": "This is AI-assisted detection, not a medical diagnosis.",
            },
        )

    except Exception as e:
        logger.error("Pathology detection failed: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
            },
        )
    finally:
        # Clean up temp file — never store patient images
        if temp_path and temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    logger.info("Starting Medical Image Anonymization API...")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
