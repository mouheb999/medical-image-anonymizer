"""detect.py - Detect text in a real image file"""
import os
import sys

# Set env vars before importing paddle
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from pathlib import Path

# Add ocr to path
from ocr import TextDetector


def load_image(path: str) -> np.ndarray:
    """Load image from file (DICOM or regular image)."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    if path.suffix.lower() == ".dcm":
        import pydicom
        ds = pydicom.dcmread(path)
        img = ds.pixel_array
        
        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        
        # Normalize to uint8
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        from PIL import Image
        img = np.array(Image.open(path))
        
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
    
    return img


def main():
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try default image
        default = "person21_virus_53.jpeg"
        if Path(default).exists():
            image_path = default
        else:
            print("Error: No image provided and no default image found.")
            print("Usage: python detect.py <image_path>")
            sys.exit(1)
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Loading: {image_path}")
    image = load_image(image_path)
    print(f"Image size: {image.shape}")
    
    print("\nInitializing TextDetector...")
    detector = TextDetector(lang="en", conf_threshold=0.5)
    
    print("Detecting text...")
    regions = detector.detect_text(image)
    
    print(f"\n{'='*60}")
    print(f"RESULT: Found {len(regions)} text region(s)")
    print(f"{'='*60}")
    
    if len(regions) == 0:
        print("No text detected in this image.")
    else:
        for i, region in enumerate(regions, 1):
            print(f"\nRegion {i}:")
            print(f"  Polygon corners: {region.tolist()}")
            
            # Calculate bounding box
            x_coords = region[:, 0]
            y_coords = region[:, 1]
            print(f"  Bounding box: x={x_coords.min()}-{x_coords.max()}, y={y_coords.min()}-{y_coords.max()}")
            print(f"  Size: {x_coords.max() - x_coords.min()} x {y_coords.max() - y_coords.min()} pixels")
    
    return len(regions)


if __name__ == "__main__":
    sys.exit(main())
