"""anonymizer package - Medical image anonymization pipeline.

This package provides complete medical image anonymization for both DICOM
and regular image files (JPEG, PNG) including:
- Image validation (image_validator)
- Metadata PHI anonymization (metadata_anonymizer) - DICOM only
- Safe pixel redaction (pixel_redactor)
- Full pipeline orchestration (pipeline)

Example usage:
    from anonymizer import AnonymizationPipeline
    
    pipeline = AnonymizationPipeline(
        conf_threshold=0.5,
        padding=5,
        border_margin=100
    )
    
    # For DICOM files
    result = pipeline.process("input.dcm", "output_dir/")
    
    # For regular images
    result = pipeline.process("input.jpg", "output_dir/")
"""

from anonymizer.image_validator import ImageValidator, ImageValidationError
from anonymizer.metadata_anonymizer import MetadataAnonymizer
from anonymizer.pixel_redactor import PixelRedactor
from anonymizer.pipeline import AnonymizationPipeline

# Backward compatibility aliases
DICOMValidator = ImageValidator
DICOMValidationError = ImageValidationError

__all__ = [
    "ImageValidator",
    "ImageValidationError",
    "MetadataAnonymizer",
    "PixelRedactor",
    "AnonymizationPipeline",
    # Backward compatibility
    "DICOMValidator",
    "DICOMValidationError",
]
