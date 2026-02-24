"""
CLIP-based Medical Image Classifier with Threshold-based Filtering.

This is the improved production-ready version with:
- Class-based architecture
- Normalized embeddings + cosine similarity
- Text embedding caching
- Proper DICOM windowing + CLAHE
- Structured metadata output
- Robust error handling
- Backwards compatibility with original API

Author: Claude Improvements
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import cv2
import pydicom
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Disable gradients globally for inference optimization
torch.set_grad_enabled(False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DICOM windowing parameters for different anatomies
DICOM_WINDOWS = {
    "lungs": {"center": -400, "width": 1500},
    "bone": {"center": 500, "width": 2000},
    "soft_tissue": {"center": 50, "width": 400},
    "default": {"center": 0, "width": None}
}


# ---------- STAGE 1: Domain Detection ----------
DOMAIN_LABELS = [
    "a medical radiology x-ray image",
    "a natural photograph",
    "an animal photo",
    "a document or screenshot",
    "a computer generated abstract image"
]

# ---------- STAGE 2: Anatomical Classification ----------
ALLOWED_LABELS = [
    "a chest x-ray radiograph showing lungs and ribs",
    "a skull x-ray radiograph showing head bones",
    "a dental x-ray radiograph showing teeth",
    "pelvic hysterosalpingography x-ray showing uterus and fallopian tubes"
]

OTHER_MEDICAL_LABEL = [
    "a medical radiology image of another body part such as hand, leg, spine or abdomen"
]

MEDICAL_LABELS = ALLOWED_LABELS + OTHER_MEDICAL_LABEL

# Default thresholds
DOMAIN_THRESHOLD = 0.60
ANATOMY_THRESHOLD = 0.70
GAP_THRESHOLD = 0.20


@dataclass
class ClassificationResult:
    """Result container for classification output."""
    category: str
    confidence: float
    metadata: Dict


class MedicalImageClassifier:
    """
    A two-stage CLIP-based medical image classifier.
    
    Stage 1: Domain detection (medical vs. non-medical)
    Stage 2: Anatomical classification (allowed body parts vs. other medical)
    
    Args:
        domain_threshold: Minimum confidence for medical domain detection
        anatomy_threshold: Minimum confidence for anatomy classification
        gap_threshold: Minimum gap between top two predictions
        window_type: DICOM windowing type ("lungs", "bone", "soft_tissue", "default")
        model_name: Hugging Face model identifier
        device: Computation device ("cuda" or "cpu", auto-detected if None)
    """
    
    def __init__(
        self,
        domain_threshold: float = DOMAIN_THRESHOLD,
        anatomy_threshold: float = ANATOMY_THRESHOLD,
        gap_threshold: float = GAP_THRESHOLD,
        window_type: str = "bone",
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        self.domain_threshold = domain_threshold
        self.anatomy_threshold = anatomy_threshold
        self.gap_threshold = gap_threshold
        self.window_type = window_type
        self.model_name = model_name
        
        # Auto-detect device if not specified
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor once
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Text embedding cache
        self._text_embedding_cache: Dict[str, torch.Tensor] = {}
        
        logger.info("Classifier initialized successfully")
    
    def _get_text_embeddings(self, labels: List[str]) -> torch.Tensor:
        """Get cached text embeddings or compute and cache them."""
        cache_key = "|".join(labels)
        
        if cache_key not in self._text_embedding_cache:
            logger.debug(f"Computing text embeddings for {len(labels)} labels")
            
            # Process text only (no images needed for text embeddings)
            text_inputs = self.processor(
                text=labels,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            # Get text features and normalize
            with torch.no_grad():
                text_outputs = self.model.get_text_features(**text_inputs)
                # Handle different return types
                if hasattr(text_outputs, 'pooler_output'):
                    text_features = text_outputs.pooler_output
                elif hasattr(text_outputs, 'last_hidden_state'):
                    text_features = text_outputs.last_hidden_state[:, 0, :]  # Use CLS token
                else:
                    text_features = text_outputs  # Already a tensor
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            self._text_embedding_cache[cache_key] = text_features
        
        return self._text_embedding_cache[cache_key]
    
    def _apply_windowing(
        self, 
        pixel_array: np.ndarray, 
        center: int, 
        width: Optional[int]
    ) -> np.ndarray:
        """Apply DICOM windowing to pixel array."""
        if width is None:
            # Auto-stretch to min/max
            min_val = pixel_array.min()
            max_val = pixel_array.max()
        else:
            min_val = center - width // 2
            max_val = center + width // 2
        
        # Apply windowing
        windowed = np.clip(pixel_array, min_val, max_val)
        
        # Normalize to 0-255
        windowed = (windowed - min_val) / (max_val - min_val) * 255.0
        
        return windowed.astype(np.uint8)
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to image (applied on L channel for efficiency)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and preprocess image from file path.
        
        Supports DICOM (.dcm) with proper windowing and standard image formats.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed PIL Image in RGB format
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Loading image: {image_path}")
        
        if image_path.suffix.lower() == ".dcm":
            # Load DICOM
            dicom = pydicom.dcmread(str(image_path))
            pixel_array = dicom.pixel_array.astype(np.float32)
            
            # Handle photometric interpretation (inverted images)
            photometric = getattr(dicom, 'PhotometricInterpretation', 'MONOCHROME2')
            if photometric == 'MONOCHROME1':
                # Inverted image - reverse it
                pixel_array = pixel_array.max() - pixel_array
            
            # Get windowing parameters
            window_config = DICOM_WINDOWS.get(self.window_type, DICOM_WINDOWS["default"])
            
            # Check for DICOM windowing parameters in metadata
            center = getattr(dicom, 'WindowCenter', window_config["center"])
            width = getattr(dicom, 'WindowWidth', window_config["width"])
            
            # Handle multi-value window centers
            if isinstance(center, pydicom.multival.MultiValue):
                center = int(center[0])
            else:
                center = int(center)
            
            if isinstance(width, pydicom.multival.MultiValue):
                width = int(width[0]) if width[0] is not None else None
            else:
                width = int(width) if width is not None else None
            
            # Apply windowing
            windowed = self._apply_windowing(pixel_array, center, width)
            
            # Apply CLAHE for better contrast
            enhanced = self._apply_clahe(windowed)
            
            # Convert grayscale to RGB
            rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
            return Image.fromarray(rgb_image)
        
        else:
            # Standard image format
            return Image.open(image_path).convert("RGB")
    
    def _compute_similarity(
        self, 
        image: Image.Image, 
        text_embeddings: torch.Tensor
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings.
        
        Uses normalized embeddings and dot product for cosine similarity.
        Logits are scaled by 100 for numerical stability.
        """
        # Process image
        image_inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        
        # Get image features and normalize
        with torch.no_grad():
            image_outputs = self.model.get_image_features(**image_inputs)
            # Handle different return types
            if hasattr(image_outputs, 'pooler_output'):
                image_features = image_outputs.pooler_output
            elif hasattr(image_outputs, 'last_hidden_state'):
                image_features = image_outputs.last_hidden_state[:, 0, :]
            else:
                image_features = image_outputs
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Compute cosine similarity via dot product (both are L2 normalized)
        logits = (image_features @ text_embeddings.T) * 100.0
        
        # Convert to probabilities
        probs = logits.softmax(dim=-1)[0].cpu().numpy()
        
        return probs
    
    def classify_image(
        self, 
        image_path: Union[str, Path]
    ) -> Tuple[str, float, Dict]:
        """
        Classify a single image through the two-stage pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (category, confidence, metadata)
            - category: Classification result string
            - confidence: Confidence score (0-1)
            - metadata: Dictionary with detailed results
        """
        try:
            # Load and preprocess image
            image = self.load_image(image_path)
            
            # ---------- STAGE 1: Domain Detection ----------
            domain_text_emb = self._get_text_embeddings(DOMAIN_LABELS)
            domain_probs = self._compute_similarity(image, domain_text_emb)
            
            domain_best_idx = int(np.argmax(domain_probs))
            domain_conf = float(domain_probs[domain_best_idx])
            
            logger.info(
                f"Domain: {DOMAIN_LABELS[domain_best_idx]} ({domain_conf:.4f})"
            )
            
            # Check if medical
            if domain_best_idx != 0 or domain_conf < self.domain_threshold:
                metadata = {
                    "domain_probs": domain_probs.tolist(),
                    "anatomy_probs": [],
                    "domain_conf": domain_conf,
                    "anatomy_conf": 0.0,
                    "gap": 0.0
                }
                return "Rejected: Not medical", domain_conf, metadata
            
            # ---------- STAGE 2: Anatomy Classification ----------
            medical_text_emb = self._get_text_embeddings(MEDICAL_LABELS)
            anatomy_probs = self._compute_similarity(image, medical_text_emb)
            
            # Get sorted indices
            sorted_idx = np.argsort(anatomy_probs)[::-1]
            best_idx = int(sorted_idx[0])
            second_idx = int(sorted_idx[1])
            
            best_conf = float(anatomy_probs[best_idx])
            gap = float(anatomy_probs[best_idx] - anatomy_probs[second_idx])
            best_label = MEDICAL_LABELS[best_idx]
            
            logger.info(
                f"Top prediction: {best_label} ({best_conf:.4f}, gap={gap:.4f})"
            )
            
            # Build metadata
            metadata = {
                "domain_probs": domain_probs.tolist(),
                "anatomy_probs": anatomy_probs.tolist(),
                "domain_conf": domain_conf,
                "anatomy_conf": best_conf,
                "gap": gap
            }
            
            # Check if "other medical" body part
            if best_label == OTHER_MEDICAL_LABEL[0]:
                return "Rejected: Other medical body part", best_conf, metadata
            
            # Check thresholds for acceptance
            if best_conf >= self.anatomy_threshold and gap >= self.gap_threshold:
                return f"Accepted: {best_label}", best_conf, metadata
            else:
                return "Rejected: Uncertain anatomy", best_conf, metadata
                
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return "Error: File not found", 0.0, {
                "domain_probs": [],
                "anatomy_probs": [],
                "domain_conf": 0.0,
                "anatomy_conf": 0.0,
                "gap": 0.0
            }
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "Error: Classification failed", 0.0, {
                "domain_probs": [],
                "anatomy_probs": [],
                "domain_conf": 0.0,
                "anatomy_conf": 0.0,
                "gap": 0.0
            }
    
    def batch_classify(
        self, 
        image_paths: List[Union[str, Path]]
    ) -> List[Dict]:
        """
        Classify multiple images efficiently.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of result dictionaries with keys:
            - image: Image path
            - category: Classification result
            - confidence: Confidence score
            - metadata: Detailed results
        """
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing [{i}/{len(image_paths)}]: {image_path}")
            
            category, confidence, metadata = self.classify_image(image_path)
            
            results.append({
                "image": str(image_path),
                "category": category,
                "confidence": confidence,
                "metadata": metadata
            })
        
        return results
    
    def set_logging_level(self, level: Union[int, str]) -> None:
        """Set the logging level for the classifier."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)
    
    def clear_cache(self) -> None:
        """Clear the text embedding cache to free memory."""
        self._text_embedding_cache.clear()
        logger.info("Text embedding cache cleared")


# ============================================================================
# LEGACY / BACKWARDS COMPATIBLE API
# ============================================================================

# Global classifier instance for legacy API
_global_classifier: Optional[MedicalImageClassifier] = None


def _get_global_classifier() -> MedicalImageClassifier:
    """Get or create global classifier instance."""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = MedicalImageClassifier()
    return _global_classifier


def classify_image(image_path: Union[str, Path]) -> Tuple[str, float, Dict]:
    """
    Classify an image (legacy-compatible API).
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (category, confidence, metadata)
    """
    classifier = _get_global_classifier()
    return classifier.classify_image(image_path)


def batch_classify(
    image_paths: List[Union[str, Path]], 
    classifier: Optional[MedicalImageClassifier] = None
) -> List[Dict]:
    """
    Classify multiple images (convenience function).
    
    Args:
        image_paths: List of paths to image files
        classifier: Existing classifier instance (uses global if None)
        
    Returns:
        List of result dictionaries
    """
    if classifier is None:
        classifier = _get_global_classifier()
    
    return classifier.batch_classify(image_paths)


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image (legacy-compatible API).
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed PIL Image
    """
    classifier = _get_global_classifier()
    return classifier.load_image(image_path)


def clip_predict(image: Image.Image, labels: List[str]) -> np.ndarray:
    """
    Get CLIP predictions for image with given labels (legacy API).
    
    Args:
        image: PIL Image
        labels: List of text labels
        
    Returns:
        Probability distribution over labels
    """
    classifier = _get_global_classifier()
    text_embeddings = classifier._get_text_embeddings(labels)
    return classifier._compute_similarity(image, text_embeddings)


# ============================================================================
# TEST / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    image_path = "MRBRAIN.DCM"
    
    category, confidence, metadata = classify_image(image_path)
    
    print("Prediction:", category)
    print("Confidence:", round(confidence, 4))
    print("Metadata:", metadata)
