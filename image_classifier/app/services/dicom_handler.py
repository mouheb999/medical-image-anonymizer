"""
DICOM Handler Service

What this module does:
    Handles loading, validation, and saving of DICOM medical images.
    Extracts pixel data from DICOM files and prepares them for processing.
    Ensures anonymized images are saved back as valid DICOM format.

Why it is used:
    Medical images use the DICOM standard, which contains both image data
    and patient metadata. This service abstracts the complexity of DICOM
    handling, providing a clean interface for the anonymization pipeline.

Assumptions:
    - Input files are valid DICOM format (or gracefully handled if not)
    - Pixel data exists in the DICOM file
    - The service preserves all non-identifying metadata during anonymization

Author: PFE Medical Anonymizer
Date: 2025
"""

import pydicom
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import io
import logging

logger = logging.getLogger(__name__)


class DICOMHandler:
    """
    Service for handling DICOM medical image files.
    
    Provides methods to load DICOM files, extract pixel arrays,
    and save anonymized results while preserving metadata.
    """
    
    def __init__(self):
        """Initialize the DICOM handler with default settings."""
        self.supported_transfer_syntaxes = [
            '1.2.840.10008.1.2',  # Implicit VR Little Endian
            '1.2.840.10008.1.2.1',  # Explicit VR Little Endian
        ]
    
    def load_dicom(self, file_path: Path) -> Tuple[np.ndarray, pydicom.Dataset]:
        """
        Load a DICOM file and extract pixel data.
        
        Args:
            file_path: Path to the DICOM file
            
        Returns:
            Tuple of (pixel_array, dicom_dataset)
            
        Raises:
            ValueError: If file is not a valid DICOM
            RuntimeError: If pixel data cannot be extracted
        """
        try:
            logger.info(f"Loading DICOM file: {file_path}")
            
            # Read DICOM file
            ds = pydicom.dcmread(file_path, force=True)
            
            # Check if pixel data exists
            if not hasattr(ds, 'PixelData'):
                raise ValueError(f"No pixel data found in DICOM file: {file_path}")
            
            # Extract pixel array
            pixel_array = ds.pixel_array
            
            logger.info(f"DICOM loaded successfully: shape={pixel_array.shape}, "
                       f"dtype={pixel_array.dtype}")
            
            return pixel_array, ds
            
        except Exception as e:
            logger.error(f"Failed to load DICOM file {file_path}: {str(e)}")
            raise RuntimeError(f"DICOM loading failed: {str(e)}")
    
    def load_dicom_from_bytes(self, file_bytes: bytes) -> Tuple[np.ndarray, pydicom.Dataset]:
        """
        Load DICOM from bytes (for API uploads).
        
        Args:
            file_bytes: Raw bytes of DICOM file
            
        Returns:
            Tuple of (pixel_array, dicom_dataset)
        """
        try:
            logger.info("Loading DICOM from bytes")
            
            # Create BytesIO object
            dicom_stream = io.BytesIO(file_bytes)
            
            # Read DICOM
            ds = pydicom.dcmread(dicom_stream, force=True)
            
            if not hasattr(ds, 'PixelData'):
                raise ValueError("No pixel data found in uploaded DICOM")
            
            pixel_array = ds.pixel_array
            
            logger.info(f"DICOM loaded from bytes: shape={pixel_array.shape}")
            
            return pixel_array, ds
            
        except Exception as e:
            logger.error(f"Failed to load DICOM from bytes: {str(e)}")
            raise RuntimeError(f"DICOM loading from bytes failed: {str(e)}")
    
    def normalize_for_processing(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Normalize DICOM pixel array for OpenCV processing.
        
        Medical images often have 16-bit or 12-bit depth.
        This converts to 8-bit for OpenCV compatibility.
        
        Args:
            pixel_array: Raw DICOM pixel array
            
        Returns:
            Normalized 8-bit or 16-bit array suitable for OpenCV
        """
        logger.debug(f"Normalizing pixel array: dtype={pixel_array.dtype}")
        
        # Handle different bit depths
        if pixel_array.dtype == np.uint16:
            # Normalize 16-bit to 8-bit for OpenCV
            # Use windowing to preserve contrast
            p_min, p_max = np.percentile(pixel_array, [1, 99])
            normalized = np.clip((pixel_array - p_min) / (p_max - p_min) * 255, 0, 255)
            return normalized.astype(np.uint8)
        
        elif pixel_array.dtype == np.int16:
            # Handle signed 16-bit
            pixel_array = pixel_array.astype(np.float32)
            pixel_array = pixel_array - pixel_array.min()
            if pixel_array.max() > 0:
                pixel_array = (pixel_array / pixel_array.max()) * 255
            return pixel_array.astype(np.uint8)
        
        elif pixel_array.dtype == np.uint8:
            # Already 8-bit
            return pixel_array
        
        else:
            # Generic fallback
            logger.warning(f"Unexpected dtype {pixel_array.dtype}, converting to uint8")
            normalized = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
            return normalized.astype(np.uint8)
    
    def denormalize_for_dicom(
        self, 
        processed_array: np.ndarray, 
        original_array: np.ndarray
    ) -> np.ndarray:
        """
        Convert processed array back to original DICOM format.
        
        Args:
            processed_array: Array after anonymization (8-bit)
            original_array: Original array to match dtype/shape
            
        Returns:
            Array matching original DICOM specifications
        """
        logger.debug(f"Denormalizing: target dtype={original_array.dtype}")
        
        # Ensure 2D array
        if len(processed_array.shape) == 3:
            # Convert BGR to grayscale if needed
            processed_array = cv2.cvtColor(processed_array, cv2.COLOR_BGR2GRAY)
        
        # Match original dtype
        if original_array.dtype == np.uint16:
            # Scale back to 16-bit
            return (processed_array.astype(np.uint16) * 257)  # 255*257 = 65535
        elif original_array.dtype == np.int16:
            return processed_array.astype(np.int16)
        else:
            return processed_array.astype(original_array.dtype)
    
    def save_anonymized_dicom(
        self, 
        original_ds: pydicom.Dataset,
        anonymized_pixels: np.ndarray,
        output_path: Path
    ) -> None:
        """
        Save anonymized image as DICOM file.
        
        Removes identifying information while preserving
        necessary medical metadata.
        
        Args:
            original_ds: Original DICOM dataset
            anonymized_pixels: Processed pixel array
            output_path: Where to save the result
        """
        try:
            logger.info(f"Saving anonymized DICOM to: {output_path}")
            
            # Create a copy of the dataset
            ds = original_ds.copy()
            
            # Update pixel data
            ds.PixelData = anonymized_pixels.tobytes()
            
            # Anonymize patient information (DICOM standard tags)
            anonymization_tags = [
                (0x0010, 0x0010),  # PatientName
                (0x0010, 0x0020),  # PatientID
                (0x0010, 0x0030),  # PatientBirthDate
                (0x0010, 0x0040),  # PatientSex
                (0x0010, 0x1040),  # PatientAddress
                (0x0010, 0x2154),  # PatientTelephoneNumbers
                (0x0010, 0x21B0),  # AdditionalPatientHistory
                (0x0008, 0x0090),  # ReferringPhysicianName
                (0x0008, 0x1048),  # PhysiciansOfRecord
                (0x0008, 0x1050),  # PerformingPhysicianName
                (0x0008, 0x1070),  # OperatorsName
                (0x0008, 0x1155),  # ReferencedSOPInstanceUID
                (0x0020, 0x000D),  # StudyInstanceUID
                (0x0020, 0x000E),  # SeriesInstanceUID
                (0x0020, 0x0010),  # StudyID
                (0x0040, 0x0275),  # RequestAttributesSequence
                (0x0040, 0xA124),  # UID
                (0x0040, 0xA730),  # ReferencedRequestSequence
            ]
            
            # Remove or anonymize tags
            for tag in anonymization_tags:
                if tag in ds:
                    if tag == (0x0010, 0x0010):  # PatientName
                        ds[tag].value = "ANONYMOUS"
                    elif tag == (0x0010, 0x0020):  # PatientID
                        ds[tag].value = "ANON_ID"
                    else:
                        del ds[tag]
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            ds.save_as(output_path)
            
            logger.info(f"Anonymized DICOM saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save anonymized DICOM: {str(e)}")
            raise RuntimeError(f"DICOM save failed: {str(e)}")
    
    def get_metadata_summary(self, ds: pydicom.Dataset) -> Dict[str, Any]:
        """
        Extract non-identifying metadata summary for logging/validation.
        
        Args:
            ds: DICOM dataset
            
        Returns:
            Dictionary of safe metadata (no patient identifiers)
        """
        safe_metadata = {
            'modality': getattr(ds, 'Modality', 'Unknown'),
            'body_part': getattr(ds, 'BodyPartExamined', 'Unknown'),
            'image_size': f"{ds.Rows}x{ds.Columns}" if hasattr(ds, 'Rows') else 'Unknown',
            'bits_allocated': getattr(ds, 'BitsAllocated', 'Unknown'),
            'photometric_interpretation': getattr(ds, 'PhotometricInterpretation', 'Unknown'),
        }
        
        return safe_metadata
