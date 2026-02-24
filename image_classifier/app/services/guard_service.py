"""
Guard Service

What this module does:
    Provides soft validation and quality checks for medical images.
    Performs informational analysis without blocking the pipeline.
    Detects potential issues (wrong format, extreme sizes, etc.)
    and reports them for logging/monitoring.

Why it is used:
    Medical images require careful handling, but the pipeline
    should never hard-fail due to validation issues. This service
    provides "guard rails" - warnings and information without
    stopping processing. It helps identify edge cases and quality
    issues for post-processing review.

Assumptions:
    - Validation is advisory only, never blocking
    - False positives are acceptable (better to flag than miss)
    - Quality metrics are approximate and for logging only
    - The pipeline continues regardless of validation results

Author: PFE Medical Anonymizer
Date: 2025
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation findings."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationFinding:
    """Single validation finding/observation."""
    category: str
    message: str
    severity: ValidationSeverity
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for an image."""
    is_valid: bool  # Always True, but indicates if any ERROR level findings
    findings: List[ValidationFinding]
    quality_score: float  # 0-1 overall quality estimate
    recommendations: List[str]
    metadata: Dict[str, Any]


class GuardService:
    """
    Soft validation service for medical images.
    
    Provides informational checks without blocking processing.
    All methods return warnings/advice, never raise blocking errors.
    """
    
    def __init__(self):
        """Initialize guard service with default thresholds."""
        # Image quality thresholds
        self.min_resolution = 64  # Minimum dimension
        self.max_resolution = 10000  # Maximum dimension
        self.min_contrast = 10  # Min std dev of pixel values
        self.max_file_size_mb = 500  # Maximum file size
        
        # Medical image common characteristics
        self.common_modalities = ['CT', 'MR', 'XR', 'CR', 'DX', 'US', 'PT', 'NM']
        self.common_body_parts = ['CHEST', 'ABDOMEN', 'HEAD', 'PELVIS', 'EXTREMITY']
    
    def validate_image(
        self,
        pixel_array: np.ndarray,
        dicom_metadata: Optional[Any] = None
    ) -> ValidationReport:
        """
        Perform soft validation on an image.
        
        Args:
            pixel_array: Image pixel data
            dicom_metadata: Optional DICOM dataset for additional checks
            
        Returns:
            ValidationReport with findings and recommendations
        """
        findings = []
        recommendations = []
        
        # Basic image checks
        findings.extend(self._check_image_properties(pixel_array))
        
        # Quality checks
        findings.extend(self._check_image_quality(pixel_array))
        
        # DICOM-specific checks
        if dicom_metadata is not None and PYDICOM_AVAILABLE:
            findings.extend(self._check_dicom_metadata(dicom_metadata))
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings)
        
        # Collect metadata
        metadata = self._collect_metadata(pixel_array, dicom_metadata)
        
        # Check if any ERROR level findings
        has_errors = any(f.severity == ValidationSeverity.ERROR for f in findings)
        
        report = ValidationReport(
            is_valid=not has_errors,  # Still informational, not blocking
            findings=findings,
            quality_score=quality_score,
            recommendations=recommendations,
            metadata=metadata
        )
        
        # Log summary
        self._log_report(report)
        
        return report
    
    def _check_image_properties(self, pixel_array: np.ndarray) -> List[ValidationFinding]:
        """Check basic image properties."""
        findings = []
        
        # Check dimensions
        if len(pixel_array.shape) < 2:
            findings.append(ValidationFinding(
                category="dimensions",
                message="Image has less than 2 dimensions",
                severity=ValidationSeverity.ERROR,
                details={"shape": pixel_array.shape}
            ))
        else:
            h, w = pixel_array.shape[:2]
            
            # Check minimum size
            if h < self.min_resolution or w < self.min_resolution:
                findings.append(ValidationFinding(
                    category="dimensions",
                    message=f"Image resolution very low: {w}x{h}",
                    severity=ValidationSeverity.WARNING,
                    details={"width": w, "height": h, "min_required": self.min_resolution}
                ))
            
            # Check maximum size
            if h > self.max_resolution or w > self.max_resolution:
                findings.append(ValidationFinding(
                    category="dimensions",
                    message=f"Image resolution very high: {w}x{h}",
                    severity=ValidationSeverity.WARNING,
                    details={"width": w, "height": h, "max_allowed": self.max_resolution}
                ))
            
            # Check aspect ratio
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio > 10:
                findings.append(ValidationFinding(
                    category="dimensions",
                    message=f"Unusual aspect ratio: {aspect_ratio:.1f}:1",
                    severity=ValidationSeverity.INFO,
                    details={"aspect_ratio": aspect_ratio}
                ))
        
        # Check bit depth
        if pixel_array.dtype == np.uint8:
            findings.append(ValidationFinding(
                category="bit_depth",
                message="8-bit image (may be compressed or processed)",
                severity=ValidationSeverity.INFO,
                details={"dtype": str(pixel_array.dtype)}
            ))
        elif pixel_array.dtype in [np.uint16, np.int16]:
            findings.append(ValidationFinding(
                category="bit_depth",
                message="16-bit image (typical for medical)",
                severity=ValidationSeverity.INFO,
                details={"dtype": str(pixel_array.dtype)}
            ))
        else:
            findings.append(ValidationFinding(
                category="bit_depth",
                message=f"Unusual bit depth: {pixel_array.dtype}",
                severity=ValidationSeverity.WARNING,
                details={"dtype": str(pixel_array.dtype)}
            ))
        
        return findings
    
    def _check_image_quality(self, pixel_array: np.ndarray) -> List[ValidationFinding]:
        """Check image quality metrics."""
        findings = []
        
        # Calculate statistics
        p_min, p_max = pixel_array.min(), pixel_array.max()
        p_mean = pixel_array.mean()
        p_std = pixel_array.std()
        
        # Check contrast
        if p_std < self.min_contrast:
            findings.append(ValidationFinding(
                category="quality",
                message="Very low contrast detected",
                severity=ValidationSeverity.WARNING,
                details={"std_dev": float(p_std), "threshold": self.min_contrast}
            ))
        
        # Check for completely uniform regions (possible corruption)
        if p_std < 1:
            findings.append(ValidationFinding(
                category="quality",
                message="Nearly uniform image (possible corruption)",
                severity=ValidationSeverity.WARNING,
                details={"std_dev": float(p_std)}
            ))
        
        # Check for potential clipping
        if p_min == 0 or (p_max == 255 and pixel_array.dtype == np.uint8):
            findings.append(ValidationFinding(
                category="quality",
                message="Possible intensity clipping detected",
                severity=ValidationSeverity.INFO,
                details={"min": int(p_min), "max": int(p_max)}
            ))
        
        # Check dynamic range usage
        if pixel_array.dtype == np.uint16:
            effective_range = (p_max - p_min) / 65535.0
            if effective_range < 0.1:
                findings.append(ValidationFinding(
                    category="quality",
                    message="Low dynamic range usage",
                    severity=ValidationSeverity.INFO,
                    details={"effective_range": float(effective_range)}
                ))
        
        return findings
    
    def _check_dicom_metadata(self, ds: Any) -> List[ValidationFinding]:
        """Check DICOM-specific metadata."""
        findings = []
        
        # Check modality
        modality = getattr(ds, 'Modality', 'UNKNOWN')
        if modality not in self.common_modalities:
            findings.append(ValidationFinding(
                category="dicom",
                message=f"Uncommon modality: {modality}",
                severity=ValidationSeverity.INFO,
                details={"modality": modality, "common": self.common_modalities}
            ))
        
        # Check photometric interpretation
        photo_interp = getattr(ds, 'PhotometricInterpretation', 'UNKNOWN')
        if photo_interp not in ['MONOCHROME1', 'MONOCHROME2', 'RGB', 'YBR_FULL']:
            findings.append(ValidationFinding(
                category="dicom",
                message=f"Unusual photometric interpretation: {photo_interp}",
                severity=ValidationSeverity.WARNING,
                details={"photometric_interpretation": photo_interp}
            ))
        
        # Check for missing tags
        required_tags = ['Modality', 'Rows', 'Columns']
        missing_tags = [tag for tag in required_tags if not hasattr(ds, tag)]
        if missing_tags:
            findings.append(ValidationFinding(
                category="dicom",
                message=f"Missing recommended DICOM tags: {missing_tags}",
                severity=ValidationSeverity.WARNING,
                details={"missing_tags": missing_tags}
            ))
        
        return findings
    
    def _calculate_quality_score(self, findings: List[ValidationFinding]) -> float:
        """Calculate overall quality score from findings."""
        if not findings:
            return 1.0
        
        # Start with perfect score
        score = 1.0
        
        # Deduct based on severity
        for finding in findings:
            if finding.severity == ValidationSeverity.ERROR:
                score -= 0.3
            elif finding.severity == ValidationSeverity.WARNING:
                score -= 0.1
            elif finding.severity == ValidationSeverity.INFO:
                score -= 0.02
        
        return max(0.0, score)
    
    def _generate_recommendations(self, findings: List[ValidationFinding]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        for finding in findings:
            if finding.category == "dimensions" and finding.severity == ValidationSeverity.WARNING:
                recommendations.append(
                    "Consider resizing or verifying image dimensions"
                )
            elif finding.category == "quality" and finding.severity == ValidationSeverity.WARNING:
                recommendations.append(
                    "Low contrast may affect text detection accuracy"
                )
            elif finding.category == "dicom" and "Missing" in finding.message:
                recommendations.append(
                    "Verify DICOM file integrity and completeness"
                )
        
        # Add general recommendations if few findings
        if len(findings) < 3:
            recommendations.append(
                "Image appears suitable for anonymization"
            )
        
        return list(set(recommendations))  # Remove duplicates
    
    def _collect_metadata(
        self,
        pixel_array: np.ndarray,
        dicom_metadata: Optional[Any]
    ) -> Dict[str, Any]:
        """Collect safe metadata for reporting."""
        metadata = {
            "shape": list(pixel_array.shape),
            "dtype": str(pixel_array.dtype),
            "pixel_range": {
                "min": int(pixel_array.min()),
                "max": int(pixel_array.max()),
                "mean": float(pixel_array.mean()),
                "std": float(pixel_array.std())
            }
        }
        
        if dicom_metadata is not None and PYDICOM_AVAILABLE:
            metadata["modality"] = getattr(dicom_metadata, 'Modality', 'Unknown')
            metadata["photometric_interpretation"] = getattr(
                dicom_metadata, 'PhotometricInterpretation', 'Unknown'
            )
            metadata["bits_allocated"] = getattr(dicom_metadata, 'BitsAllocated', 'Unknown')
        
        return metadata
    
    def _log_report(self, report: ValidationReport) -> None:
        """Log validation report summary."""
        severity_counts = {
            "info": sum(1 for f in report.findings if f.severity == ValidationSeverity.INFO),
            "warning": sum(1 for f in report.findings if f.severity == ValidationSeverity.WARNING),
            "error": sum(1 for f in report.findings if f.severity == ValidationSeverity.ERROR)
        }
        
        logger.info(
            f"Validation complete: quality={report.quality_score:.2f}, "
            f"findings={len(report.findings)} "
            f"(INFO:{severity_counts['info']}, "
            f"WARN:{severity_counts['warning']}, "
            f"ERR:{severity_counts['error']})"
        )
        
        # Log warnings
        for finding in report.findings:
            if finding.severity == ValidationSeverity.WARNING:
                logger.warning(f"Validation: {finding.message}")
            elif finding.severity == ValidationSeverity.ERROR:
                logger.error(f"Validation: {finding.message}")
    
    def quick_check(self, pixel_array: np.ndarray) -> Tuple[bool, str]:
        """
        Quick validation check for simple pass/fail with message.
        
        Args:
            pixel_array: Image to check
            
        Returns:
            Tuple of (passed, message)
        """
        # Minimum checks only
        if len(pixel_array.shape) < 2:
            return False, "Invalid image dimensions"
        
        h, w = pixel_array.shape[:2]
        if h < self.min_resolution or w < self.min_resolution:
            return True, f"Low resolution: {w}x{h} (proceeding anyway)"
        
        return True, "Basic validation passed"
