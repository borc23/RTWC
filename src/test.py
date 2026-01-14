import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics package not found. pip install ultralytics")
    sys.exit(1)

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    print("Warning: SAHI not available. Using standard inference.")
    SAHI_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def check_labels_exist(data_config_path, split='test'):
    """Check if labels exist for the given split."""
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_path = Path(data_config.get('path', ''))
    image_path_str = data_config.get(split)
    
    if not image_path_str:
        return False, None, None
    
    image_dir = dataset_path / image_path_str
    label_dir = image_dir.parent.parent / 'labels' / image_dir.name
    
    has_labels = label_dir.exists() and any(label_dir.glob('*.txt'))
    
    return has_labels, image_dir, label_dir


def calculate_metrics_from_predictions(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate detection metrics from predictions and ground truths.
    Simple implementation for basic metrics.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for img_name, preds in predictions.items():
        gts = ground_truths.get(img_name, [])
        
        matched_gts = set()
        
        for pred in preds:
            pred_box = pred['bbox']
            pred_cls = pred['class']
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gts:
                    continue
                if gt['class'] != pred_cls:
                    continue
                    
                iou = calculate_iou(pred_box, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gts.add(best_gt_idx)
            else:
                total_fp += 1
        
        total_fn += len(gts) - len(matched_gts)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def load_yolo_labels(label_path, img_width, img_height):
    """Load YOLO format labels and convert to absolute coordinates."""
    labels = []
    if not label_path.exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                
                # Convert to absolute coordinates
                x1 = (cx - w/2) * img_width
                y1 = (cy - h/2) * img_height
                x2 = (cx + w/2) * img_width
                y2 = (cy + h/2) * img_height
                
                labels.append({
                    'class': cls,
                    'bbox': [x1, y1, x2, y2]
                })
    
    return labels


def run_inference(model_path, data_config, slicing_cfg, output_dir, conf_threshold=0.25, wandb_project=None):
    """
    Run inference on test data with optional slicing.
    """
    print(f"\n--- Running Test Inference ---")
    print(f"Model: {model_path}")
    print(f"Data config: {data_config}")
    
    # Check if test set has labels
    has_labels, image_dir, label_dir = check_labels_exist(data_config, 'test')
    
    if image_dir is None:
        print("Error: Test set not found in data config.")
        print("Make sure 'test' is defined in your data.yaml")
        sys.exit(1)
    
    print(f"Test images directory: {image_dir}")
    print(f"Labels available: {has_labels}")
    
    # Get image files
    image_files = sorted(
        list(image_dir.glob("*.jpg")) + 
        list(image_dir.glob("*.png")) + 
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.bmp"))
    )
    
    if not image_files:
        print(f"No images found in {image_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} test images")
    
    # Create output directories
    output_path = Path(output_dir)
    pred_labels_dir = output_path / 'predictions' / 'labels'
    pred_images_dir = output_path / 'predictions' / 'images'
    pred_labels_dir.mkdir(parents=True, exist_ok=True)
    pred_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Prepare for metrics calculation
    all_predictions = {}
    all_ground_truths = {}
    
    # Run inference
    for img_path in tqdm(image_files, desc="Running inference"):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Run prediction (with or without slicing)
        if SAHI_AVAILABLE and slicing_cfg.get('use_sahi', True):
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=model_path,
                confidence_threshold=conf_threshold,
                device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
            )
            
            result = get_sliced_prediction(
                str(img_path),
                detection_model,
                slice_height=slicing_cfg.get('slice_height', 640),
                slice_width=slicing_cfg.get('slice_width', 640),
                overlap_height_ratio=slicing_cfg.get('overlap_ratio', 0.2),
                overlap_width_ratio=slicing_cfg.get('overlap_ratio', 0.2),
            )
            
            # Extract predictions
            predictions = []
            for obj in result.object_prediction_list:
                predictions.append({
                    'class': obj.category.id,
                    'bbox': obj.bbox.to_xyxy(),
                    'confidence': obj.score.value
                })
        else:
            # Standard YOLO inference
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
        
        # Save predictions in YOLO format
        pred_label_path = pred_labels_dir / f"{img_path.stem}.txt"
        with open(pred_label_path, 'w') as f:
            for pred in predictions:
                x1, y1, x2, y2 = pred['bbox']
                cx = ((x1 + x2) / 2) / img_width
                cy = ((y1 + y2) / 2) / img_height
                w = (x2 - x1) / img_width
                h = (y2 - y1) / img_height
                f.write(f"{pred['class']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {pred['confidence']:.4f}\n")
        
        # Store for metrics
        all_predictions[img_path.stem] = predictions
        
        # Load ground truth if available
        if has_labels:
            gt_path = label_dir / f"{img_path.stem}.txt"
            all_ground_truths[img_path.stem] = load_yolo_labels(gt_path, img_width, img_height)
    
    # Calculate metrics if labels exist
    metrics = {
        'total_images': len(image_files),
        'total_predictions': sum(len(p) for p in all_predictions.values()),
        'has_labels': has_labels
    }
    
    if has_labels:
        detection_metrics = calculate_metrics_from_predictions(
            all_predictions, 
            all_ground_truths,
            iou_threshold=0.5
        )
        metrics.update(detection_metrics)
        
        print(f"\nTest Metrics (IoU@0.5):")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
    else:
        print("\nNo labels available - metrics not calculated")
    
    # Save metrics
    metrics_path = output_path / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")
    
    # Log to W&B if available
    if WANDB_AVAILABLE and wandb_project:
        wandb.init(project=wandb_project, name='test-inference', job_type='test')
        wandb.log(metrics)
        wandb.finish()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on test data.")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights.')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml config.')
    parser.add_argument('--config', type=str, default=None, help='Path to training config (for slicing params).')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save results.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--no-sahi', action='store_true', help='Disable SAHI sliced inference.')
    parser.add_argument('--wandb-project', type=str, default=None, help='W&B project name.')
    args = parser.parse_args()
    
    # Load slicing config
    slicing_cfg = {
        'slice_width': 640,
        'slice_height': 640,
        'overlap_ratio': 0.2,
        'min_area_ratio': 0.1,
        'use_sahi': not args.no_sahi
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'slicing' in config:
                slicing_cfg.update(config['slicing'])
    
    slicing_cfg['use_sahi'] = not args.no_sahi
    
    run_inference(
        model_path=args.model,
        data_config=args.data,
        slicing_cfg=slicing_cfg,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        wandb_project=args.wandb_project
    )


if __name__ == "__main__":
    main()
