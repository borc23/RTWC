import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from test import check_labels_exist, load_yolo_labels, calculate_metrics_from_predictions
from PIL import Image
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics package not found. pip install ultralytics")
    sys.exit(1)

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


def run_test_inference(model_path, data_config, slicing_cfg, conf_threshold=0.25):
    """
    Run inference on test data and calculate metrics.
    Returns metrics dict.
    """
    

    has_labels, image_dir, label_dir = check_labels_exist(data_config, 'test')

    if not has_labels:
        print(f"{RED}Error: Test set has no labels. Cannot compare models.{RESET}")
        sys.exit(1)

    image_files = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.bmp"))
    )

    if not image_files:
        print(f"{RED}No test images found in {image_dir}{RESET}")
        sys.exit(1)

    model = YOLO(model_path)

    all_predictions = {}
    all_ground_truths = {}

    for img_path in tqdm(image_files, desc=f"Testing {Path(model_path).name}", leave=False):
        img = Image.open(img_path)
        img_width, img_height = img.size

        results = model.predict(
            img_path,
            imgsz=slicing_cfg.get('slice_width', 640),
            conf=conf_threshold,
            save=False,
            verbose=False
        )

        predictions = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                predictions.append({
                    'class': int(boxes.cls[i]),
                    'bbox': boxes.xyxy[i].tolist(),
                    'confidence': float(boxes.conf[i])
                })

        all_predictions[img_path.stem] = predictions

        gt_path = label_dir / f"{img_path.stem}.txt"
        all_ground_truths[img_path.stem] = load_yolo_labels(gt_path, img_width, img_height)

    metrics = calculate_metrics_from_predictions(
        all_predictions,
        all_ground_truths,
        iou_threshold=0.5
    )

    return metrics


def format_metric(value, is_better, is_count=False):
    """Format a metric value with color based on comparison."""
    if is_count:
        formatted = f"{int(value):>6}"
    else:
        formatted = f"{value:>6.4f}"

    if is_better is None:
        return formatted
    elif is_better:
        return f"{GREEN}{formatted}{RESET}"
    else:
        return f"{RED}{formatted}{RESET}"


def compare_metrics(current_metrics, new_metrics, higher_is_better):
    """
    Compare two sets of metrics and return comparison results.
    Returns dict with 'current_better', 'new_better', 'equal' for each metric.
    """
    comparison = {}
    for metric, higher_better in higher_is_better.items():
        current_val = current_metrics.get(metric, 0)
        new_val = new_metrics.get(metric, 0)

        if current_val == new_val:
            comparison[metric] = 'equal'
        elif higher_better:
            comparison[metric] = 'new_better' if new_val > current_val else 'current_better'
        else:
            comparison[metric] = 'new_better' if new_val < current_val else 'current_better'

    return comparison


def print_comparison_table(current_metrics, new_metrics):
    """Print a side-by-side comparison table with color coding."""
    higher_is_better = {
        'precision': True,
        'recall': True,
        'f1_score': True,
        'true_positives': True,
        'false_positives': False,
        'false_negatives': False,
    }

    comparison = compare_metrics(current_metrics, new_metrics, higher_is_better)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}              MODEL COMPARISON RESULTS{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"\n{BOLD}{'Metric':<20} {'Current Best':>15} {'New Model':>15}{RESET}")
    print(f"{'-'*50}")

    metrics_order = ['precision', 'recall', 'f1_score', 'true_positives', 'false_positives', 'false_negatives']

    for metric in metrics_order:
        current_val = current_metrics.get(metric, 0)
        new_val = new_metrics.get(metric, 0)
        comp = comparison.get(metric, 'equal')

        is_count = metric in ['true_positives', 'false_positives', 'false_negatives']

        current_better = comp == 'current_better'
        new_better = comp == 'new_better'

        current_formatted = format_metric(current_val, current_better if comp != 'equal' else None, is_count)
        new_formatted = format_metric(new_val, new_better if comp != 'equal' else None, is_count)

        display_name = metric.replace('_', ' ').title()
        print(f"{display_name:<20} {current_formatted:>15} {new_formatted:>15}")

    print(f"{'-'*50}")

    current_f1 = current_metrics.get('f1_score', 0)
    new_f1 = new_metrics.get('f1_score', 0)

    if new_f1 > current_f1:
        print(f"\n{GREEN}{BOLD}>>> New model is BETTER (higher F1 score){RESET}")
        return True
    elif new_f1 < current_f1:
        print(f"\n{RED}{BOLD}>>> Current model is BETTER (higher F1 score){RESET}")
        return False
    else:
        print(f"\n{YELLOW}{BOLD}>>> Models are EQUAL (same F1 score){RESET}")
        return None


def promote_model(source_path, dest_dir):
    """Copy the new model to the best_model directory."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for f in dest_dir.glob('*'):
        if f.is_file():
            f.unlink()

    dest_path = dest_dir / 'best.pt'
    shutil.copy2(source_path, dest_path)
    print(f"{GREEN}Model promoted to {dest_path}{RESET}")


def prompt_and_version_model():
    """Prompt user for version and run version_model command."""
    print(f"\n{BOLD}Do you want to update and version the best model? [y/N]: {RESET}", end='')
    response = input().strip().lower()

    if response in ['y', 'yes']:
        print(f"{BOLD}Enter version number (e.g., 1.0.0): {RESET}", end='')
        version = input().strip()

        if not version:
            print(f"{RED}No version provided. Skipping versioning.{RESET}")
            return False

        print(f"\n{GREEN}Running: ./scripts/run.sh version_model {version}{RESET}")
        result = subprocess.run(
            ['./scripts/run.sh', 'version_model', version],
            cwd=os.getcwd()
        )

        if result.returncode == 0:
            print(f"\n{GREEN}Model successfully versioned as v{version}!{RESET}")
            return True
        else:
            print(f"\n{RED}Versioning failed with exit code {result.returncode}{RESET}")
            return False
    else:
        print(f"{YELLOW}Skipping model update.{RESET}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare current best model with new model.")
    parser.add_argument('--current', type=str, default='models/best_model/best.pt',
                        help='Path to current best model.')
    parser.add_argument('--new', type=str, default='models/best.pt',
                        help='Path to new model to compare.')
    parser.add_argument('--data', type=str, default='data/data.yaml',
                        help='Path to data.yaml config.')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to training config.')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold.')
    parser.add_argument('--auto-promote', action='store_true',
                        help='Automatically promote new model if better.')
    parser.add_argument('--force-promote', action='store_true',
                        help='Promote new model regardless of comparison.')
    args = parser.parse_args()

    current_path = Path(args.current)
    new_path = Path(args.new)

    if not new_path.exists():
        print(f"{RED}Error: New model not found at {new_path}{RESET}")
        sys.exit(1)

    slicing_cfg = {
        'slice_width': 640,
        'slice_height': 640,
    }

    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'slicing' in config:
                slicing_cfg.update(config['slicing'])

    if not current_path.exists():
        print(f"{YELLOW}No current best model found at {current_path}{RESET}")
        print(f"{YELLOW}Promoting new model as the first best model.{RESET}")
        promote_model(new_path, 'models/best_model')
        prompt_and_version_model()
        return

    print(f"{BOLD}Comparing models...{RESET}")
    print(f"  Current best: {current_path}")
    print(f"  New model:    {new_path}")
    print(f"  Test data:    {args.data}")

    # Run inference on both models
    print(f"\n{BOLD}Running test inference on current best model...{RESET}")
    current_metrics = run_test_inference(current_path, args.data, slicing_cfg, args.conf)

    print(f"\n{BOLD}Running test inference on new model...{RESET}")
    new_metrics = run_test_inference(new_path, args.data, slicing_cfg, args.conf)

    # Print comparison
    new_is_better = print_comparison_table(current_metrics, new_metrics)

    # Handle promotion
    if args.force_promote:
        print(f"\n{YELLOW}Force promoting new model...{RESET}")
        promote_model(new_path, 'models/best_model')
        prompt_and_version_model()
    elif args.auto_promote and new_is_better:
        print(f"\n{GREEN}Auto-promoting new model...{RESET}")
        promote_model(new_path, 'models/best_model')
        prompt_and_version_model()
    elif new_is_better:
        # Interactive prompt when new model is better
        promote_model(new_path, 'models/best_model')
        prompt_and_version_model()
    elif new_is_better is None:
        # Models are equal - ask user if they want to update anyway
        print(f"\n{YELLOW}Models have equal F1 score.{RESET}")
        print(f"{BOLD}Do you want to update best model anyway? [y/N]: {RESET}", end='')
        response = input().strip().lower()
        if response in ['y', 'yes']:
            promote_model(new_path, 'models/best_model')
            prompt_and_version_model()
    else:
        # Current model is better
        print(f"\n{YELLOW}Current model remains the best. No changes made.{RESET}")


if __name__ == "__main__":
    main()
