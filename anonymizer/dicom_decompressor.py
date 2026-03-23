"""
dicom_decompressor.py
---------------------
Safe DICOM pixel decompression that works with numpy 1.x.
Uses gdcm for compressed transfer syntaxes, avoiding the
pylibjpeg numpy>=2.0 conflict.
"""

import logging
import numpy as np
import pydicom
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    ExplicitVRBigEndian,
)

logger = logging.getLogger(__name__)

# Transfer syntaxes that need decompression
COMPRESSED_SYNTAXES = {
    "1.2.840.10008.1.2.4.70": "JPEG Lossless",
    "1.2.840.10008.1.2.4.80": "JPEG-LS Lossless",
    "1.2.840.10008.1.2.4.81": "JPEG-LS Lossy",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
    "1.2.840.10008.1.2.4.91": "JPEG 2000 Lossy",
    "1.2.840.10008.1.2.5":    "RLE Lossless",
    "1.2.840.10008.1.2.4.50": "JPEG Baseline",
    "1.2.840.10008.1.2.4.51": "JPEG Extended",
}

def is_compressed(dataset: pydicom.Dataset) -> bool:
    """Check if DICOM dataset uses compressed transfer syntax."""
    try:
        ts = str(dataset.file_meta.TransferSyntaxUID)
        return ts in COMPRESSED_SYNTAXES
    except AttributeError:
        return False

def decompress_dicom(dataset: pydicom.Dataset) -> np.ndarray:
    """
    Safely decompress DICOM pixel data to numpy array.
    
    Strategy:
    1. If uncompressed: use dataset.pixel_array directly
    2. If compressed: use gdcm handler to decompress first
    3. If gdcm fails: raise clear error with install instructions
    
    Parameters
    ----------
    dataset: pydicom.Dataset
        The loaded DICOM dataset
        
    Returns
    -------
    np.ndarray
        Pixel array as numpy array (uint8 or uint16)
        
    Raises
    ------
    ValueError
        If decompression fails with clear error message
    """
    if not is_compressed(dataset):
        logger.debug("DICOM is uncompressed, reading directly")
        try:
            return dataset.pixel_array.copy()
        except Exception as e:
            raise ValueError(f"Failed to read uncompressed DICOM pixels: {e}")
    
    ts = str(dataset.file_meta.TransferSyntaxUID)
    compression_name = COMPRESSED_SYNTAXES.get(ts, ts)
    logger.info(f"Compressed DICOM detected: {compression_name}")
    
    # Try gdcm first (no numpy version conflict)
    try:
        import gdcm
        pydicom.config.pixel_data_handlers = ['gdcm']
        pixel_array = dataset.pixel_array.copy()
        logger.info(f"Successfully decompressed using gdcm: {pixel_array.shape}")
        return pixel_array
    except ImportError:
        logger.warning("gdcm not available, trying pydicom handlers")
    except Exception as e:
        logger.warning(f"gdcm decompression failed: {e}")
    
    # Try pydicom's built-in handler
    try:
        pydicom.config.pixel_data_handlers = ['numpy']
        pixel_array = dataset.pixel_array.copy()
        logger.info(f"Decompressed with numpy handler: {pixel_array.shape}")
        return pixel_array
    except Exception as e:
        logger.warning(f"numpy handler failed: {e}")
    
    # All methods failed
    raise ValueError(
        f"Cannot decompress DICOM with transfer syntax: {compression_name}. "
        f"Install gdcm: pip install python-gdcm"
    )

def normalize_to_uint8(pixel_array: np.ndarray) -> np.ndarray:
    """
    Normalize pixel array to uint8 for OCR processing.
    Handles uint16 DICOM pixels correctly.
    
    Parameters
    ----------
    pixel_array: np.ndarray
        Raw DICOM pixel array (uint8 or uint16)
        
    Returns
    -------
    np.ndarray
        Normalized uint8 array
    """
    if pixel_array.dtype == np.uint8:
        return pixel_array
    
    # uint16 → uint8 normalization
    min_val = pixel_array.min()
    max_val = pixel_array.max()
    
    if max_val == min_val:
        return np.zeros_like(pixel_array, dtype=np.uint8)
    
    normalized = ((pixel_array - min_val) / (max_val - min_val) * 255)
    return normalized.astype(np.uint8)
