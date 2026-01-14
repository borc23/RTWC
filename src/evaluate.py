import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics package not found. pip install ultralytics")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def run_evaluation(model_path, data_config, slicing_cfg, output_dir, wandb_project=None):
    """
    Run evaluation on the validation set.
    """
    print(f"\n--- Running Evaluation ---")
    print(f"Model: {model_path}")
    print(f"Data config: {data_config}")
    
    # Load model
    model = YOLO(model_path)
    
    # Prepare validation slicing config (no overlap for validation)
    val_slicing_cfg = deepcopy(slicing_cfg)
    val_slicing_cfg['overlap_ratio'] = 0.0
    
    # Run validation
    results = model.val(
        data=data_config,
        imgsz=slicing_cfg.get('slice_width', 640),
        batch=16,
        split='val',
        save_json=True,
        plots=True,
        project=output_dir,
        name='eval'
    )
    
    # Extract metrics
    metrics = {
        'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
        'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
        'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
        'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
    }
    
    # Per-class metrics if available
    if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
        per_class = {}
        for i, cls_idx in enumerate(results.box.ap_class_index):
            per_class[f'class_{int(cls_idx)}_AP50'] = float(results.box.ap50[i])
        metrics['per_class'] = per_class
    
    # Save metrics
    output_path = Path(output_dir) / 'eval_metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nEvaluation Results:")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"\nMetrics saved to: {output_path}")
    
    # Log to W&B if available
    if WANDB_AVAILABLE and wandb_project:
        wandb.init(project=wandb_project, name='evaluation', job_type='eval')
        wandb.log(metrics)
        wandb.finish()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on validation set.")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights.')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml config.')
    parser.add_argument('--config', type=str, default='./configs/train_config.yaml', help='Path to training config (for slicing params).')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Directory to save results.')
    parser.add_argument('--wandb-project', type=str, default=None, help='W&B project name.')
    args = parser.parse_args()
    
    # Load slicing config if provided
    slicing_cfg = {
        'slice_width': 640,
        'slice_height': 640,
        'overlap_ratio': 0.0,
        'min_area_ratio': 0.1
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'slicing' in config:
                slicing_cfg.update(config['slicing'])
    
    run_evaluation(
        model_path=args.model,
        data_config=args.data,
        slicing_cfg=slicing_cfg,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project
    )


if __name__ == "__main__":
    main()
