"""image_validator.py - Image file validation for medical image anonymization pipeline.

This module provides validation for both DICOM and regular image files (JPEG, PNG)
to ensure they are valid and contain pixel data before processing.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy

if TYPE_CHECKING:
    import pydicom

logger = logging.getLogger(__name__)


class ImageValidationError(Exception):
    """Exception raised when image file validation fails.
    
    Attributes:
        message: Explanation of the validation failure
        path: Path to the file that failed validation (if applicable)
    """
    
    def __init__(self, message: str, path: str | None = None) -> None:
        self.message = message
        self.path = path
        super().__init__(self.message)


class ValidationResult(NamedTuple):
    """Result of image validation.
    
    Attributes:
        is_dicom: True if file is DICOM format
        dataset: pydicom Dataset if DICOM, None otherwise
        image_path: Path to the validated image file
        pixel_array: numpy array of image pixels (for non-DICOM)
    """
    is_dicom: bool
    dataset: "pydicom.Dataset | None"
    image_path: Path
    pixel_array: "numpy.ndarray | None"


class ImageValidator:
    """Validates image files (DICOM, JPEG, PNG) for the anonymization pipeline.
    
    This class validates both DICOM and regular image files:
    - DICOM: Validates tags (PixelData, Modality, PatientID)
    - JPEG/PNG: Validates file can be opened and contains pixel data
    
    Example:
        >>> validator = ImageValidator()
        >>> try:
        ...     result = validator.validate("image.dcm")
        ...     if result.is_dicom:
        ...         print(f"Valid DICOM: {result.dataset.Modality}")
        ...     else:
        ...         print(f"Valid image: {result.pixel_array.shape}")
        ... except ImageValidationError as e:
        ...     print(f"Validation failed: {e.message}")
    """
    
    # Supported image extensions
    DICOM_EXTENSIONS = {".dcm", ".dicom"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    ALL_EXTENSIONS = DICOM_EXTENSIONS | IMAGE_EXTENSIONS
    
    # Required DICOM tags for validation
    REQUIRED_DICOM_TAGS = [
        (0x0008, 0x0060),  # Modality
        (0x0010, 0x0020),  # PatientID
    ]
    
    # Pixel Data tag
    PIXEL_DATA_TAG = (0x7FE0, 0x0010)
    
    def __init__(self) -> None:
        """Initialize the image validator."""
        logger.debug("ImageValidator initialized")
    
    def validate(self, path: str) -> ValidationResult:
        """Validate an image file and return validation result.
        
        For DICOM files:
        - Validates extension, parsing, and required tags
        - Returns dataset for further processing
        
        For JPEG/PNG files:
        - Validates file can be opened with PIL
        - Loads pixel array as numpy
        
        Args:
            path: Path to the image file to validate
            
        Returns:
            ValidationResult containing:
                - is_dicom: True if DICOM format
                - dataset: pydicom Dataset (DICOM only)
                - image_path: Path object
                - pixel_array: numpy array (non-DICOM only)
            
        Raises:
            ImageValidationError: If validation fails
            
        Example:
            >>> validator = ImageValidator()
            >>> result = validator.validate("image.jpg")
            >>> print(f"Is DICOM: {result.is_dicom}")
        """
        import numpy as np
        
        file_path = Path(path)
        
        # Check file exists
        logger.debug(f"Validating image file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise ImageValidationError(
                f"File does not exist: {file_path}",
                str(file_path)
            )
        
        # Check extension
        ext = file_path.suffix.lower()
        if ext not in self.ALL_EXTENSIONS:
            logger.error(f"Invalid file extension: {ext}")
            raise ImageValidationError(
                f"Invalid file extension '{ext}'. "
                f"Supported: {', '.join(sorted(self.ALL_EXTENSIONS))}",
                str(file_path)
            )
        
        # Route to appropriate validator
        if ext in self.DICOM_EXTENSIONS:
            return self._validate_dicom(file_path)
        else:
            return self._validate_regular_image(file_path)
    
    def _validate_dicom(self, file_path: Path) -> ValidationResult:
        """Validate a DICOM file.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            ValidationResult with is_dicom=True and dataset
            
        Raises:
            ImageValidationError: If DICOM validation fails
        """
        import pydicom
        
        # Try to read the file
        try:
            dataset = pydicom.dcmread(file_path)
            logger.debug(f"Successfully read DICOM file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to read DICOM file: {e}")
            raise ImageValidationError(
                f"Failed to read DICOM file: {str(e)}",
                str(file_path)
            ) from e
        
        # Verify PixelData tag is present
        if self.PIXEL_DATA_TAG not in dataset:
            logger.error("PixelData tag (7FE0,0010) not found in dataset")
            raise ImageValidationError(
                "PixelData tag (7FE0,0010) is missing. File may not contain image data.",
                str(file_path)
            )
        
        logger.debug("PixelData tag verified")
        
        # Verify required tags
        missing_tags = []
        for tag in self.REQUIRED_DICOM_TAGS:
            if tag not in dataset:
                tag_str = f"({tag[0]:04X},{tag[1]:04X})"
                missing_tags.append(tag_str)
                logger.warning(f"Required tag {tag_str} not found")
        
        if missing_tags:
            raise ImageValidationError(
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
        
        return ValidationResult(
            is_dicom=True,
            dataset=dataset,
            image_path=file_path,
            pixel_array=None
        )
    
    def _validate_regular_image(self, file_path: Path) -> ValidationResult:
        """Validate a regular image file (JPEG, PNG, etc.).
        
        Args:
            file_path: Path to image file
            
        Returns:
            ValidationResult with is_dicom=False and pixel_array
            
        Raises:
            ImageValidationError: If image validation fails
        """
        from PIL import Image
        import numpy as np
        
        try:
            # Open and load image
            img = Image.open(file_path)
            
            # Convert to numpy array
            pixel_array = np.array(img)
            
            logger.debug(
                f"Successfully loaded image: {file_path} | "
                f"Shape: {pixel_array.shape}, Dtype: {pixel_array.dtype}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load image file: {e}")
            raise ImageValidationError(
                f"Failed to load image file: {str(e)}",
                str(file_path)
            ) from e
        
        # Verify image has pixel data
        if pixel_array.size == 0:
            raise ImageValidationError(
                "Image contains no pixel data",
                str(file_path)
            )
        
        logger.info(
            f"Image validation passed: {file_path.name} | "
            f"Size: {pixel_array.shape}, Mode: {img.mode}"
        )
        
        return ValidationResult(
            is_dicom=False,
            dataset=None,
            image_path=file_path,
            pixel_array=pixel_array
        )
    
    @staticmethod
    def is_dicom_file(path: str) -> bool:
        """Check if a file is a DICOM file based on extension.
        
        Args:
            path: Path to check
            
        Returns:
            True if DICOM extension, False otherwise
        """
        return Path(path).suffix.lower() in ImageValidator.DICOM_EXTENSIONS


# Backward compatibility alias
DICOMValidationError = ImageValidationError
DICOMValidator = ImageValidator
