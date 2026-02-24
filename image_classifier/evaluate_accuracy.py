"""
Medical Image Classifier Accuracy Evaluation Script

Evaluates classification accuracy against ground-truth labels derived from folder structure.

Author: Claude
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Import the classifier
from improved_medical_classifier import MedicalImageClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Expected class labels (must match folder names)
CLASSES = ["chest", "skull", "dental", "pelvic", "other_medical", "non_medical"]

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}


def map_prediction_to_label(category: str) -> str:
    """
    Map classifier output string to simplified label.
    
    Args:
        category: Raw classifier output string
        
    Returns:
        Simplified label for comparison with ground truth
    """
    category_lower = category.lower()
    
    if "chest x-ray" in category_lower:
        return "chest"
    elif "skull x-ray" in category_lower:
        return "skull"
    elif "dental x-ray" in category_lower:
        return "dental"
    elif "pelvic hysterosalpingography" in category_lower:
        return "pelvic"
    elif "other medical body part" in category_lower:
        return "other_medical"
    elif "not medical" in category_lower:
        return "non_medical"
    elif "uncertain anatomy" in category_lower:
        # Treat uncertain as incorrect - map to uncertain category
        return "uncertain"
    else:
        logger.warning(f"Unknown category format: {category}")
        return "uncertain"


def collect_images(dataset_path: Path) -> List[Tuple[Path, str]]:
    """
    Recursively collect all images and their ground-truth labels.
    
    Args:
        dataset_path: Root path to dataset folder
        
    Returns:
        List of (image_path, true_label) tuples
    """
    images = []
    
    for class_name in CLASSES:
        class_folder = dataset_path / class_name
        if not class_folder.exists():
            logger.warning(f"Class folder not found: {class_folder}")
            continue
        
        for ext in SUPPORTED_EXTENSIONS:
            for image_path in class_folder.rglob(f"*{ext}"):
                images.append((image_path, class_name))
    
    return images


def evaluate_image(
    classifier: MedicalImageClassifier, 
    image_path: Path
) -> Tuple[str, float, str]:
    """
    Classify a single image and map prediction to label.
    
    Args:
        classifier: Initialized classifier instance
        image_path: Path to image file
        
    Returns:
        Tuple of (predicted_label, confidence, raw_category)
    """
    raw_category, confidence, metadata = classifier.classify_image(image_path)
    predicted_label = map_prediction_to_label(raw_category)
    
    return predicted_label, confidence, raw_category


def run_evaluation(
    dataset_path: Path,
    classifier: MedicalImageClassifier,
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Run full evaluation on dataset.
    
    Args:
        dataset_path: Path to dataset root
        classifier: Initialized classifier
        
    Returns:
        Tuple of (results, y_true, y_pred)
    """
    images = collect_images(dataset_path)
    
    if not images:
        logger.error("No images found in dataset!")
        sys.exit(1)
    
    logger.info(f"Found {len(images)} images to evaluate")
    
    results = []
    y_true = []
    y_pred = []
    
    for i, (image_path, true_label) in enumerate(images, 1):
        logger.info(f"[{i}/{len(images)}] Processing: {image_path}")
        
        predicted_label, confidence, raw_category = evaluate_image(
            classifier, image_path
        )
        
        result = {
            "image": str(image_path),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "raw_category": raw_category,
        }
        results.append(result)
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
    
    return results, y_true, y_pred


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of computed metrics
    """
    # Filter out uncertain predictions for standard metrics
    valid_indices = [i for i, p in enumerate(y_pred) if p != "uncertain"]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    
    metrics = {
        "total_images": len(y_true),
        "uncertain_predictions": y_pred.count("uncertain"),
        "valid_predictions": len(y_true_valid),
    }
    
    if y_true_valid:
        metrics["accuracy"] = round(accuracy_score(y_true_valid, y_pred_valid), 4)
        metrics["precision_macro"] = round(
            precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0), 4
        )
        metrics["recall_macro"] = round(
            recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0), 4
        )
        metrics["f1_macro"] = round(
            f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0), 4
        )
        
        # Per-class metrics
        metrics["precision_per_class"] = precision_score(
            y_true_valid, y_pred_valid, labels=CLASSES, average=None, zero_division=0
        ).tolist()
        metrics["recall_per_class"] = recall_score(
            y_true_valid, y_pred_valid, labels=CLASSES, average=None, zero_division=0
        ).tolist()
        metrics["f1_per_class"] = f1_score(
            y_true_valid, y_pred_valid, labels=CLASSES, average=None, zero_division=0
        ).tolist()
    else:
        metrics["accuracy"] = 0.0
        metrics["precision_macro"] = 0.0
        metrics["recall_macro"] = 0.0
        metrics["f1_macro"] = 0.0
        metrics["precision_per_class"] = [0.0] * len(CLASSES)
        metrics["recall_per_class"] = [0.0] * len(CLASSES)
        metrics["f1_per_class"] = [0.0] * len(CLASSES)
    
    # Confusion matrix (for all classes + uncertain)
    all_labels = CLASSES + ["uncertain"]
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = all_labels
    
    return metrics


def print_report(results: List[Dict], metrics: Dict) -> None:
    """
    Print formatted evaluation report.
    
    Args:
        results: List of per-image results
        metrics: Computed metrics dictionary
    """
    print("\n" + "=" * 50)
    print("           ACCURACY REPORT")
    print("=" * 50)
    
    print(f"\nTotal Images: {metrics['total_images']}")
    print(f"Uncertain Predictions: {metrics['uncertain_predictions']}")
    print(f"Valid Predictions: {metrics['valid_predictions']}")
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    
    print("\n" + "-" * 50)
    print("Per-Class Metrics:")
    print("-" * 50)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)
    
    for i, class_name in enumerate(CLASSES):
        precision = round(metrics["precision_per_class"][i], 4)
        recall = round(metrics["recall_per_class"][i], 4)
        f1 = round(metrics["f1_per_class"][i], 4)
        print(f"{class_name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
    
    print("\n" + "-" * 50)
    print("Confusion Matrix:")
    print("-" * 50)
    
    labels = metrics["confusion_matrix_labels"]
    cm = metrics["confusion_matrix"]
    
    # Print header
    header = "{:>12}".format("") + "".join([f"{label[:8]:>10}" for label in labels])
    print(header)
    
    # Print rows
    for i, true_label in enumerate(labels):
        row = f"{true_label[:10]:>12}"
        for j in range(len(labels)):
            row += f"{cm[i][j]:>10}"
        print(row)
    
    print("\n" + "=" * 50)
    
    # Detailed classification report
    valid_results = [
        r for r in results 
        if r["predicted_label"] != "uncertain"
    ]
    
    if valid_results:
        y_true = [r["true_label"] for r in valid_results]
        y_pred = [r["predicted_label"] for r in valid_results]
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))


def save_results(
    results: List[Dict], 
    metrics: Dict, 
    output_path: Path
) -> None:
    """
    Save evaluation results to CSV files.
    
    Args:
        results: Per-image results
        metrics: Computed metrics
        output_path: Base path for output files
    """
    # Save per-image predictions
    predictions_file = output_path.parent / f"{output_path.stem}_predictions.csv"
    
    with open(predictions_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    logger.info(f"Predictions saved to: {predictions_file}")
    
    # Save confusion matrix
    cm_file = output_path.parent / f"{output_path.stem}_confusion_matrix.csv"
    
    with open(cm_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + metrics["confusion_matrix_labels"])
        for i, label in enumerate(metrics["confusion_matrix_labels"]):
            writer.writerow([label] + metrics["confusion_matrix"][i])
    
    logger.info(f"Confusion matrix saved to: {cm_file}")
    
    # Save metrics summary
    metrics_file = output_path.parent / f"{output_path.stem}_metrics.csv"
    
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Images", metrics["total_images"]])
        writer.writerow(["Uncertain Predictions", metrics["uncertain_predictions"]])
        writer.writerow(["Valid Predictions", metrics["valid_predictions"]])
        writer.writerow(["Overall Accuracy", metrics["accuracy"]])
        writer.writerow(["Macro Precision", metrics["precision_macro"]])
        writer.writerow(["Macro Recall", metrics["recall_macro"]])
        writer.writerow(["Macro F1", metrics["f1_macro"]])
        
        writer.writerow([])
        writer.writerow(["Per-Class Metrics"])
        writer.writerow(["Class", "Precision", "Recall", "F1"])
        for i, class_name in enumerate(CLASSES):
            writer.writerow([
                class_name,
                round(metrics["precision_per_class"][i], 4),
                round(metrics["recall_per_class"][i], 4),
                round(metrics["f1_per_class"][i], 4),
            ])
    
    logger.info(f"Metrics summary saved to: {metrics_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate MedicalImageClassifier accuracy"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset folder with class subfolders",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save results CSV",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging",
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Initialize classifier once
    logger.info("Initializing classifier...")
    classifier = MedicalImageClassifier()
    
    # Run evaluation
    results, y_true, y_pred = run_evaluation(dataset_path, classifier)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Print report
    print_report(results, metrics)
    
    # Save results if requested
    if args.save:
        save_results(results, metrics, Path(args.save))


if __name__ == "__main__":
    main()
