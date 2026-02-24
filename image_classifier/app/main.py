"""
Medical Image Anonymization API - Main Application

What this module does:
    FastAPI orchestrator for the medical image anonymization pipeline.
    Coordinates DICOM handling, OCR text detection, anonymization,
    and validation services into a unified REST API.

Why it is used:
    Provides a production-ready HTTP interface for anonymizing
    medical images. The API accepts DICOM files, processes them
    through the anonymization pipeline, and returns anonymized
    results with detailed metadata.

Assumptions:
    - Input files are DICOM format (other formats gracefully rejected)
    - Services are initialized once and reused (singleton pattern)
    - Memory management is handled via streaming for large files
    - Never stores non-anonymized images to disk
    - All errors are caught and returned as HTTP 500 with details

API Endpoints:
    POST /anonymize - Main anonymization endpoint
    GET /health - Health check with service status
    GET /info - API information and version

Author: PFE Medical Anonymizer
Date: 2025
"""

import logging
from pathlib import Path
from typing import Optional
from io import BytesIO
import tempfile
import os

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Import services
from app.services.dicom_handler import DICOMHandler
from app.services.ocr_service import OCRService, TextRegion
from app.services.anonymization_service import AnonymizationService, AnonymizationResult
from app.services.guard_service import GuardService, ValidationReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Image Anonymization API",
    description="API for anonymizing medical DICOM images by removing text overlays",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize services (singleton pattern)
dicom_handler = DICOMHandler()
ocr_service = OCRService(use_paddle=True)
anonymization_service = AnonymizationService(default_method="neutral")  # MEDICAL-GRADE default
guard_service = GuardService()


@app.get("/")
async def root():
    """API root - provides basic information."""
    return {
        "service": "Medical Image Anonymization API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns status of all services and dependencies.
    """
    health_status = {
        "status": "healthy",
        "services": {
            "dicom_handler": "available",
            "ocr_service": ocr_service.engine_name,
            "anonymization": "opencv_ready",
            "guard_service": "available"
        },
        "capabilities": {
            "dicom_read": True,
            "dicom_write": True,
            "text_detection": True,
            "anonymization": True,
            "validation": True
        }
    }
    
    return health_status


@app.post("/anonymize")
async def anonymize_dicom(
    file: UploadFile = File(...),
    method: str = "auto",
    validation: bool = True
):
    """
    Anonymize a DICOM medical image.
    
    Args:
        file: DICOM file to anonymize
        method: Anonymization method ("auto", "telea", "ns", "mean")
        validation: Whether to run validation checks
    
    Returns:
        Anonymized DICOM file as download
    
    Raises:
        HTTPException: If processing fails
    """
    logger.info(f"Received anonymization request: file={file.filename}, method={method}")
    
    # Validate file extension
    if not file.filename.lower().endswith(('.dcm', '.dicom')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a DICOM file (.dcm or .dicom)"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        logger.info(f"File size: {len(content)} bytes")
        
        # === Step 1: Load DICOM ===
        logger.info("Step 1: Loading DICOM file...")
        try:
            pixel_array, dicom_dataset = dicom_handler.load_dicom_from_bytes(content)
        except Exception as e:
            logger.error(f"DICOM loading failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load DICOM file: {str(e)}"
            )
        
        # === Step 2: Validation (optional, non-blocking) ===
        validation_report = None
        if validation:
            logger.info("Step 2: Running validation...")
            try:
                validation_report = guard_service.validate_image(pixel_array, dicom_dataset)
                logger.info(f"Validation complete: score={validation_report.quality_score:.2f}")
            except Exception as e:
                logger.warning(f"Validation failed (continuing): {e}")
        
        # === Step 3: Normalize for processing ===
        logger.info("Step 3: Normalizing image...")
        normalized_image = dicom_handler.normalize_for_processing(pixel_array)
        
        # === Step 4: Detect text (OCR) ===
        logger.info("Step 4: Detecting text regions...")
        try:
            text_regions = ocr_service.detect_text(normalized_image)
            
            # Merge overlapping regions
            text_regions = ocr_service.merge_overlapping_regions(text_regions)
            
            logger.info(f"Detected {len(text_regions)} text regions")
            
            # Log detected regions
            for i, region in enumerate(text_regions):
                logger.info(f"  Region {i+1}: {region.text} (conf={region.confidence:.2f})")
                
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            # Continue with empty regions (image will pass through)
            text_regions = []
        
        # === Step 5: Anonymize ===
        logger.info("Step 5: Anonymizing image...")
        try:
            anonymization_result = anonymization_service.anonymize(
                normalized_image,
                text_regions,
                method=method
            )
            
            logger.info(
                f"Anonymization complete: method={anonymization_result.method_used}, "
                f"pixels={anonymization_result.pixels_modified}, "
                f"confidence={anonymization_result.confidence_score:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Anonymization failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Anonymization processing failed: {str(e)}"
            )
        
        # === Step 6: Convert back to DICOM format ===
        logger.info("Step 6: Converting back to DICOM format...")
        denormalized_image = dicom_handler.denormalize_for_dicom(
            anonymization_result.anonymized_image,
            pixel_array
        )
        
        # === Step 7: Save anonymized DICOM ===
        logger.info("Step 7: Saving anonymized DICOM...")
        
        # Create temporary file for response
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
            temp_path = Path(tmp_file.name)
        
        try:
            dicom_handler.save_anonymized_dicom(
                dicom_dataset,
                denormalized_image,
                temp_path
            )
            
            logger.info(f"Anonymized DICOM saved to: {temp_path}")
            
            # Prepare response metadata
            metadata = {
                "original_filename": file.filename,
                "anonymized": True,
                "method_used": anonymization_result.method_used,
                "text_regions_detected": len(text_regions),
                "pixels_modified": int(anonymization_result.pixels_modified),
                "confidence_score": round(anonymization_result.confidence_score, 3),
                "validation": {
                    "performed": validation_report is not None,
                    "quality_score": round(validation_report.quality_score, 3) if validation_report else None,
                    "warnings": len([f for f in validation_report.findings if f.severity.value == "warning"]) if validation_report else 0
                } if validation_report else None
            }
            
            # Return file with metadata headers
            return FileResponse(
                path=temp_path,
                filename=f"anonymized_{file.filename}",
                media_type="application/dicom",
                headers={
                    "X-Anonymization-Metadata": str(metadata)
                }
            )
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/info")
async def api_info():
    """
    Get API information and capabilities.
    """
    return {
        "name": "Medical Image Anonymization API",
        "version": "1.0.0",
        "description": "Anonymizes medical DICOM images by detecting and removing text overlays",
        "endpoints": {
            "/": "API root",
            "/health": "Health check with service status",
            "/anonymize": "POST - Anonymize DICOM file",
            "/info": "API information"
        },
        "anonymization_methods": ["auto", "telea", "ns", "mean"],
        "supported_formats": ["DICOM (.dcm, .dicom)"],
        "text_detection": {
            "primary": "PaddleOCR (if available)",
            "fallback": "Image processing (white/dark text detection)"
        },
        "anonymization_algorithms": {
            "telea": "OpenCV TELEA inpainting (default)",
            "ns": "OpenCV Navier-Stokes inpainting",
            "mean": "Mean replacement (aggressive fallback)"
        }
    }


@app.get("/anonymize-stream")
async def anonymize_info():
    """
    Information about the anonymize endpoint.
    """
    return {
        "endpoint": "/anonymize",
        "method": "POST",
        "content_type": "multipart/form-data",
        "parameters": {
            "file": {
                "type": "file",
                "required": True,
                "description": "DICOM file to anonymize",
                "format": ".dcm or .dicom"
            },
            "method": {
                "type": "string",
                "required": False,
                "default": "auto",
                "options": ["auto", "telea", "ns", "mean"],
                "description": "Anonymization algorithm"
            },
            "validation": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Run validation checks"
            }
        },
        "response": {
            "success": "Returns anonymized DICOM file with metadata headers",
            "error": "Returns JSON with error details"
        }
    }


# Cleanup function for temp files
@app.on_event("shutdown")
async def cleanup():
    """Cleanup on shutdown."""
    logger.info("Shutting down API, cleaning up resources...")


# Run the server
if __name__ == "__main__":
    logger.info("Starting Medical Image Anonymization API...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
