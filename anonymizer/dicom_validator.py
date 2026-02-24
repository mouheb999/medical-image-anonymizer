"""dicom_validator.py - DICOM file validation for medical image anonymization pipeline.

This module provides validation for DICOM files to ensure they are valid,
contain required tags, and have pixel data before processing.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pydicom

logger = logging.getLogger(__name__)


class DICOMValidationError(Exception):
    """Exception raised when DICOM file validation fails.
    
    Attributes:
        message: Explanation of the validation failure
        path: Path to the file that failed validation (if applicable)
    """
    
    def __init__(self, message: str, path: str | None = None) -> None:
        self.message = message
        self.path = path
        super().__init__(self.message)


class DICOMValidator:
    """Validates DICOM files for processing in the anonymization pipeline.
    
    This class performs comprehensive validation checks on DICOM files including:
    - File existence and extension verification
    - pydicom parsing validation
    - Required DICOM tag presence (PixelData, Modality, PatientID)
    
    Example:
        >>> validator = DICOMValidator()
        >>> try:
        ...     dataset = validator.validate("path/to/image.dcm")
        ...     print(f"Valid DICOM with {len(dataset.pixel_array.shape)} dimensions")
        ... except DICOMValidationError as e:
        ...     print(f"Validation failed: {e.message}")
    """
    
    # Required DICOM tags for validation
    REQUIRED_TAGS = [
        (0x0008, 0x0060),  # Modality
        (0x0010, 0x0020),  # PatientID
    ]
    
    # Pixel Data tag
    PIXEL_DATA_TAG = (0x7FE0, 0x0010)
    
    def __init__(self) -> None:
        """Initialize the DICOM validator."""
        logger.debug("DICOMValidator initialized")
    
    def validate(self, path: str) -> "pydicom.Dataset":
        """Validate a DICOM file and return the loaded dataset.
        
        Performs comprehensive validation:
        1. Verifies file exists
        2. Checks file extension is .dcm
        3. Attempts to parse with pydicom
        4. Verifies PixelData tag is present
        5. Verifies required tags (Modality, PatientID) exist
        
        Args:
            path: Path to the DICOM file to validate
            
        Returns:
            The loaded pydicom Dataset if validation succeeds
            
        Raises:
            DICOMValidationError: If any validation check fails
            
        Example:
            >>> validator = DICOMValidator()
            >>> ds = validator.validate("image.dcm")
            >>> print(f"Modality: {ds.Modality}")
        """
        import pydicom
        
        file_path = Path(path)
        
        # Check file exists
        logger.debug(f"Validating DICOM file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise DICOMValidationError(
                f"File does not exist: {file_path}",
                str(file_path)
            )
        
        # Check extension
        if file_path.suffix.lower() != ".dcm":
            logger.error(f"Invalid file extension: {file_path.suffix}")
            raise DICOMValidationError(
                f"Invalid file extension '{file_path.suffix}'. Expected '.dcm'",
                str(file_path)
            )
        
        # Try to read the file
        try:
            dataset = pydicom.dcmread(file_path)
            logger.debug(f"Successfully read DICOM file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to read DICOM file: {e}")
            raise DICOMValidationError(
                f"Failed to read DICOM file: {str(e)}",
                str(file_path)
            ) from e
        
        # Verify PixelData tag is present
        if self.PIXEL_DATA_TAG not in dataset:
            logger.error("PixelData tag (7FE0,0010) not found in dataset")
            raise DICOMValidationError(
                "PixelData tag (7FE0,0010) is missing. File may not contain image data.",
                str(file_path)
            )
        
        logger.debug("PixelData tag verified")
        
        # Verify required tags
        missing_tags = []
        for tag in self.REQUIRED_TAGS:
            if tag not in dataset:
                tag_str = f"({tag[0]:04X},{tag[1]:04X})"
                missing_tags.append(tag_str)
                logger.warning(f"Required tag {tag_str} not found")
        
        if missing_tags:
            raise DICOMValidationError(
                f"Required DICOM tags missing: {', '.join(missing_tags)}",
                str(file_path)
            )
        
        # Log success
        modality = getattr(dataset, 'Modality', 'Unknown')
        patient_id = getattr(dataset, 'PatientID', 'Unknown')
        logger.info(
            f"DICOM validation passed: {file_path.name} | "
            f"Modality: {modality}, PatientID: {patient_id}"
        )
        
        return dataset
