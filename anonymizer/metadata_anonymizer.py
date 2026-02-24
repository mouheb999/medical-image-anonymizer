"""metadata_anonymizer.py - DICOM metadata PHI anonymization.

This module handles the anonymization of Protected Health Information (PHI)
from DICOM metadata tags while preserving the DICOM structure and all
non-PHI diagnostic information.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pydicom

logger = logging.getLogger(__name__)


class MetadataAnonymizer:
    """Anonymizes PHI-related DICOM metadata tags.
    
    This class replaces PHI tag values with "ANONYMIZED" while preserving
the DICOM tag structure for compliance. It also handles removal of
    overlay planes (Group 6000 tags) which are separate text annotation layers.
    
    PHI Tags Anonymized:
        - PatientName (0010,0010)
        - PatientID (0010,0020)
        - PatientBirthDate (0010,0030)
        - PatientSex (0010,0040)
        - StudyDate (0008,0020)
        - StudyTime (0008,0030)
        - OperatorsName (0008,1070)
        - InstitutionName (0008,0080)
        - PhysiciansOfRecord (0008,1048)
        - OtherPatientIDs (0010,1000)
        - ReferringPhysicianName (0008,0090)
        - PatientAddress (0010,1040)
        - PatientTelephoneNumbers (0010,2154)
    
    Example:
        >>> anonymizer = MetadataAnonymizer()
        >>> ds = pydicom.dcmread("input.dcm")
        >>> anonymized_ds, count = anonymizer.anonymize(ds)
        >>> print(f"Anonymized {count} tags")
    """
    
    # PHI tags to anonymize (Group, Element)
    PHI_TAGS = [
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x0040),  # PatientSex
        (0x0008, 0x0020),  # StudyDate
        (0x0008, 0x0030),  # StudyTime
        (0x0008, 0x1070),  # OperatorsName
        (0x0008, 0x0080),  # InstitutionName
        (0x0008, 0x1048),  # PhysiciansOfRecord
        (0x0010, 0x1000),  # OtherPatientIDs
        (0x0008, 0x0090),  # ReferringPhysicianName
        (0x0010, 0x1040),  # PatientAddress
        (0x0010, 0x2154),  # PatientTelephoneNumbers
    ]
    
    # Replacement value for PHI
    REPLACEMENT_VALUE = "ANONYMIZED"
    
    # Overlay plane group range (6000-601F)
    OVERLAY_GROUP_START = 0x6000
    OVERLAY_GROUP_END = 0x601F
    
    def __init__(self) -> None:
        """Initialize the metadata anonymizer."""
        logger.debug("MetadataAnonymizer initialized")
    
    def anonymize(self, ds: "pydicom.Dataset | None") -> tuple["pydicom.Dataset | None", int]:
        """Anonymize PHI tags in a DICOM dataset.
        
        Replaces all present PHI tag values with "ANONYMIZED" while
        preserving the tag structure. Also removes overlay planes
        (Group 6000 tags) which are text annotation layers.
        
        For non-DICOM files (JPEG/PNG), this method skips anonymization
        and returns the input unchanged with count=0.
        
        Args:
            ds: The pydicom Dataset to anonymize (modified in-place),
                or None for non-DICOM files
            
        Returns:
            A tuple containing:
                - The anonymized dataset (or None for non-DICOM)
                - The count of tags that were anonymized (0 for non-DICOM)
                
        Example:
            >>> ds = pydicom.dcmread("input.dcm")
            >>> anonymizer = MetadataAnonymizer()
            >>> ds, count = anonymizer.anonymize(ds)
            >>> print(f"Anonymized {count} PHI tags")
        """
        # Skip for non-DICOM files (None dataset)
        if ds is None:
            logger.info("Skipping metadata anonymization - non-DICOM file (no metadata to anonymize)")
            return None, 0
        
        logger.info("Starting metadata anonymization")
        
        anonymized_count = 0
        
        # Anonymize PHI tags
        for tag in self.PHI_TAGS:
            if tag in ds:
                tag_str = f"({tag[0]:04X},{tag[1]:04X})"
                original_value = str(ds[tag].value)[:50]  # Truncate for logging
                
                # Replace value but keep the tag
                ds[tag].value = self.REPLACEMENT_VALUE
                anonymized_count += 1
                
                logger.info(
                    f"Anonymized tag {tag_str}: "
                    f"'{original_value}' → '{self.REPLACEMENT_VALUE}'"
                )
        
        # Remove overlay planes (Group 6000 tags)
        overlay_count = self._remove_overlay_planes(ds)
        
        total_anonymized = anonymized_count + overlay_count
        
        logger.info(
            f"Metadata anonymization complete: "
            f"{anonymized_count} PHI tags + {overlay_count} overlay planes"
        )
        
        return ds, total_anonymized
    
    def _remove_overlay_planes(self, ds: "pydicom.Dataset") -> int:
        """Remove overlay plane tags (Group 6000) from the dataset.
        
        Overlay planes are separate text annotation layers stored in DICOM
        files. Removing them is 100% safe and touches no diagnostic pixels.
        
        Args:
            ds: The pydicom Dataset to process
            
        Returns:
            The number of overlay tags removed
        """
        overlay_tags = []
        
        # Find all overlay tags (Group 6000-601F)
        for tag in list(ds.keys()):  # Use list() to avoid dict change during iteration
            group = tag >> 16
            if self.OVERLAY_GROUP_START <= group <= self.OVERLAY_GROUP_END:
                overlay_tags.append(tag)
        
        # Remove overlay tags
        removed_count = 0
        for tag in overlay_tags:
            tag_str = f"({tag >> 16:04X},{tag & 0xFFFF:04X})"
            del ds[tag]
            removed_count += 1
            logger.info(f"Removed overlay plane tag {tag_str}")
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} overlay plane tags (Group 6000)")
        else:
            logger.debug("No overlay plane tags found")
        
        return removed_count
