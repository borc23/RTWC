#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import sys
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
    print("Warning: wandb not installed. Experiment tracking disabled.")
    WANDB_AVAILABLE = False

from data import SlicedDetectionTrainer, build_augmentations


def setup_wandb(config, project_name, run_name):
    if not WANDB_AVAILABLE:
        return None
    
    try:
        if not os.environ.get("WANDB_API_KEY"):
            api_key = input("Enter your W&B API Key: ").strip()
            if not api_key:
                print("No API key provided. Skipping W&B tracking...")
                return None
            os.environ["WANDB_API_KEY"] = api_key

        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        print("API Key set successfully!")
    except (EOFError, KeyboardInterrupt):
        print("\nInput interrupted. Skipping W&B tracking...")
        return None

    try:
        wandb.init(project=project_name, name=run_name, config=config, reinit='return_previous')
        print(f"W&B tracking initialized: {wandb.run.url}")
        return wandb.run
    except Exception as e:
        print(f"Failed to initialize W&B: {e}")
        print("Continuing without W&B tracking...")
        return None


def export_metrics(results_dir, output_path):
    """Export training metrics to JSON for DVC tracking."""
    metrics = {}

    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        
        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                for key, value in last_row.items():
                    clean_key = (
                        key.strip().replace("/", "_").replace("(", "").replace(")", "")
                    )
                    try:
                        metrics[clean_key] = float(value)
                    except (ValueError, TypeError):
                        pass

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics exported to {output_path}")
    return metrics


def run_training(config):
    """Main function to run YOLOv8 training with slicing and W&B."""
    print("\n--- Preparing for training ---")

    aug_config = config.pop("augmentations", {})
    train_transform = build_augmentations(aug_config)
    slicing_cfg = config.pop("slicing")

    wandb_project = config.pop("wandb_project", "yolo-sliced-training")
    wandb_run_name = config.pop("wandb_run_name", None)
    metrics_output = config.pop("metrics_output", "outputs/train_metrics.json")

    wandb_run = setup_wandb(
        config={**config, "slicing": slicing_cfg, "augmentations": aug_config},
        project_name=wandb_project,
        run_name=wandb_run_name or config.get("name", "exp"),
    )

    yolo_overrides = config
    yolo_overrides["train_transform"] = train_transform
    yolo_overrides["val_transform"] = None
    yolo_overrides["slicing_cfg"] = slicing_cfg
    yolo_overrides["train_fraction"] = yolo_overrides.pop("train_fraction", 1.0)
    yolo_overrides["val_fraction"] = yolo_overrides.pop("val_fraction", 1.0)

    model_path = yolo_overrides.get("model")
    if not model_path:
        print("Error: Model path ('model:') must be specified in the config.")
        sys.exit(1)

    try:
        model = YOLO(model_path)
        results = model.train(trainer=SlicedDetectionTrainer, **yolo_overrides)

        os.makedirs(Path(metrics_output).parent, exist_ok=True)
        metrics = export_metrics(results.save_dir, metrics_output)

        # Copy best and last models to a fixed location for DVC tracking
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        best_pt = results.save_dir / "weights" / "best.pt"
        last_pt = results.save_dir / "weights" / "last.pt"

        if best_pt.exists():
            shutil.copy2(best_pt, models_dir / "best.pt")
            print(f"Copied best model to {models_dir / 'best.pt'}")

        if last_pt.exists():
            shutil.copy2(last_pt, models_dir / "last.pt")
            print(f"Copied last model to {models_dir / 'last.pt'}")

        if wandb_run and metrics:
            wandb.log({"final": metrics})

        print("--- Training complete ---")

    except Exception as e:
        print(f"\n--- An error occurred during training: {e} ---")
        raise
    finally:
        if wandb_run:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model with slicing and W&B tracking."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="W&B project name."
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="W&B run name."
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="outputs/train_metrics.json",
        help="Path to save metrics JSON.",
    )
    args, unknown_args = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.wandb_project:
        config["wandb_project"] = args.wandb_project
    if args.wandb_run_name:
        config["wandb_run_name"] = args.wandb_run_name
    config["metrics_output"] = args.metrics_output

    # Parse additional overrides
    override_parser = argparse.ArgumentParser()
    for key, value in config.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                arg_name = f"--{key}.{inner_key}"
                if inner_value is not None:
                    override_parser.add_argument(
                        arg_name, type=type(inner_value), default=None
                    )
                else:
                    override_parser.add_argument(arg_name, type=str, default=None)
        else:
            arg_name = f"--{key}"
            if value is not None:
                override_parser.add_argument(arg_name, type=type(value), default=None)
            else:
                override_parser.add_argument(arg_name, type=str, default=None)

    override_args = override_parser.parse_args(unknown_args)

    for key, value in vars(override_args).items():
        if value is not None:
            if "." in key:
                main_key, sub_key = key.split(".")
                if main_key not in config:
                    config[main_key] = {}
                config[main_key][sub_key] = value
            else:
                config[key] = value

    run_training(config)


if __name__ == "__main__":
    main()
