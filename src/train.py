"""
YOLOv8 Sliced Training with Weights & Biases Integration.
Supports DVC pipeline integration with metrics export.
"""
import argparse
import os
import sys
import yaml
import json
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from copy import deepcopy
import random
import functools

# Ensure necessary packages are installed
try:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
except ImportError:
    print("Ultralytics package not found. pip install ultralytics")
    sys.exit(1)

try:
    import albumentations as A
except ImportError:
    print("Albumentations package not found. pip install albumentations")
    sys.exit(1)

try:
    from sahi.slicing import get_slice_bboxes
except (ImportError, ModuleNotFoundError):
    print("Error: Could not import 'get_slice_bboxes' from 'sahi'. pip install sahi")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Experiment tracking disabled.")
    WANDB_AVAILABLE = False


def seed_worker(worker_id):
    """Seeds the worker process with a unique seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def yolo_to_abs_bbox(yolo_coords, img_width, img_height):
    class_id, center_x_norm, center_y_norm, w_norm, h_norm = yolo_coords
    center_x = center_x_norm * img_width
    center_y = center_y_norm * img_height
    w = w_norm * img_width
    h = h_norm * img_height
    xmin = center_x - w / 2
    ymin = center_y - h / 2
    xmax = center_x + w / 2
    ymax = center_y + h / 2
    return int(class_id), xmin, ymin, xmax, ymax


def abs_bbox_to_yolo(abs_coords, slice_width, slice_height):
    class_id, xmin, ymin, xmax, ymax = abs_coords
    w = xmax - xmin
    h = ymax - ymin
    center_x = xmin + w / 2
    center_y = ymin + h / 2
    center_x_norm = center_x / slice_width
    center_y_norm = center_y / slice_height
    w_norm = w / slice_width
    h_norm = h / slice_height
    return class_id, center_x_norm, center_y_norm, w_norm, h_norm


class SlicedDataset(Dataset):
    def __init__(self, data_config_path, split, slicing_cfg, augmentations=None, fraction=1.0):
        self.slicing_cfg = slicing_cfg
        self.augmentations = augmentations
        self.fraction = fraction
        self.split = split

        with open(data_config_path, "r") as f:
            self.data_config = yaml.safe_load(f)
        
        dataset_path = Path(self.data_config.get('path', ''))
        
        image_path_str = self.data_config.get(split)
        if not image_path_str:
            raise ValueError(f"'{split}' set not found in {data_config_path}")
        self.image_dir = dataset_path / image_path_str
        self.label_dir = self.image_dir.parent.parent / 'labels' / self.image_dir.name

        self.image_files = sorted(
            list(self.image_dir.glob("*.jpg")) + 
            list(self.image_dir.glob("*.png")) + 
            list(self.image_dir.glob("*.jpeg")) +
            list(self.image_dir.glob("*.bmp"))
        )

        print(f"Building slice index for '{split}' set...")
        slice_groups = self._create_slice_jobs()

        if self.fraction < 1.0:
            num_groups = int(len(slice_groups) * self.fraction)
            print(f"Using {self.fraction*100:.2f}% of data ({num_groups} images).")
            random.Random(42).shuffle(slice_groups)
            slice_groups = slice_groups[:num_groups]

        if split == 'train':
            random.Random(42).shuffle(slice_groups)
        
        self.slice_jobs = [job for group in slice_groups for job in group]

        if not self.slice_jobs:
            print(f"Warning: No slices generated for '{split}'. Check paths.")

    def _create_slice_jobs(self):
        grouped_jobs = []
        if not self.image_files:
            return grouped_jobs
            
        for image_path in tqdm(self.image_files, desc=f"Indexing {self.image_dir.name}"):
            label_path = self.label_dir / f"{image_path.stem}.txt"
            
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception:
                continue

            slice_bboxes = get_slice_bboxes(
                image_height=img_height,
                image_width=img_width,
                slice_height=self.slicing_cfg['slice_height'],
                slice_width=self.slicing_cfg['slice_width'],
                overlap_height_ratio=self.slicing_cfg['overlap_ratio'],
                overlap_width_ratio=self.slicing_cfg['overlap_ratio'],
                auto_slice_resolution=False
            )
            
            has_labels = label_path.exists() and label_path.stat().st_size > 0

            current_image_jobs = []
            for bbox in slice_bboxes:
                if self.slicing_cfg.get('include_negatives', False) is False and not has_labels:
                    continue
                current_image_jobs.append((image_path, label_path if has_labels else None, bbox, (img_width, img_height)))
            
            if current_image_jobs:
                grouped_jobs.append(current_image_jobs)

        return grouped_jobs

    def __len__(self):
        return len(self.slice_jobs)

    @functools.lru_cache(maxsize=16)
    def get_image(self, path):
        return ImageOps.exif_transpose(Image.open(path).convert("RGB"))

    @functools.lru_cache(maxsize=64)
    def get_labels(self, path):
        if not path or not path.exists():
            return []
        with open(path, "r") as f:
            return [[float(p) for p in line.strip().split()] for line in f.readlines()]

    def _load_slice_without_aug(self, idx):
        """Helper to load a single slice's data without applying augmentations."""
        image_path, label_path, slice_bbox, (img_width, img_height) = self.slice_jobs[idx]

        image = self.get_image(image_path)
        original_labels = self.get_labels(label_path)

        x_min, y_min, x_max, y_max = slice_bbox
        slice_img = image.crop((x_min, y_min, x_max, y_max))
        slice_img_np = np.array(slice_img, dtype=np.uint8)

        orig_slice_width = x_max - x_min
        orig_slice_height = y_max - y_min

        labels_yolo = []
        if original_labels:
            for yolo_label in original_labels:
                class_id, orig_xmin, orig_ymin, orig_xmax, orig_ymax = yolo_to_abs_bbox(yolo_label, img_width, img_height)
                orig_area = (orig_xmax - orig_xmin) * (orig_ymax - orig_ymin)
                inter_xmin = max(orig_xmin, x_min)
                inter_ymin = max(orig_ymin, y_min)
                inter_xmax = min(orig_xmax, x_max)
                inter_ymax = min(orig_ymax, y_max)

                if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
                    new_xmin_local = inter_xmin - x_min
                    new_ymin_local = inter_ymin - y_min
                    new_xmax_local = inter_xmax - x_min
                    new_ymax_local = inter_ymax - y_min
                    new_area = (new_xmax_local - new_xmin_local) * (new_ymax_local - new_ymin_local)

                    if new_area > 0 and orig_area > 0 and new_area / orig_area >= self.slicing_cfg['min_area_ratio']:
                        new_abs_bbox = (class_id, new_xmin_local, new_ymin_local, new_xmax_local, new_ymax_local)
                        new_yolo_label = abs_bbox_to_yolo(new_abs_bbox, orig_slice_width, orig_slice_height)
                        labels_yolo.append(list(new_yolo_label))
        
        return slice_img_np, labels_yolo, str(image_path), (orig_slice_height, orig_slice_width)

    def __getitem__(self, idx):
        slice_img_np, labels_yolo, image_path, ori_shape = self._load_slice_without_aug(idx)

        if self.augmentations:
            mosaic_metadata = []
            other_indices = [idx]
            while len(other_indices) < 4:
                new_idx = random.randint(0, len(self.slice_jobs) - 1)
                if new_idx not in other_indices:
                    other_indices.append(new_idx)
            other_indices.pop(0)

            for mosaic_idx in other_indices:
                other_img_np, other_labels, _, _ = self._load_slice_without_aug(mosaic_idx)
                other_bboxes_np = np.array([row[1:] for row in other_labels], dtype=np.float32)
                if not other_labels:
                    other_bboxes_np = other_bboxes_np.reshape(0, 4)
                mosaic_metadata.append({
                    'image': other_img_np,
                    'bboxes': other_bboxes_np,
                    'class_labels': [row[0] for row in other_labels],
                })

            primary_class_labels = [row[0] for row in labels_yolo]
            primary_bboxes_np = np.array([row[1:] for row in labels_yolo], dtype=np.float32)
            if not labels_yolo:
                primary_bboxes_np = primary_bboxes_np.reshape(0, 4)

            input_dict = {
                'image': slice_img_np,
                'bboxes': primary_bboxes_np,
                'class_labels': primary_class_labels,
                'mosaic_metadata': mosaic_metadata,
            }
            
            augmented = self.augmentations(**input_dict)
            slice_img_np = augmented['image']
            labels_yolo = [[augmented['class_labels'][i]] + list(bbox) for i, bbox in enumerate(augmented['bboxes'])]
            ori_shape = slice_img_np.shape[:2]

        labels = np.array(labels_yolo, dtype=np.float32)

        return {
            "img": slice_img_np,
            "cls": labels[:, 0:1] if labels.size > 0 else np.zeros((0, 1)),
            "bboxes": labels[:, 1:5] if labels.size > 0 else np.zeros((0, 4)),
            "im_file": image_path,
            "ori_shape": ori_shape,
            "ratio_pad": (1.0, 1.0),
        }
    
    @staticmethod
    def collate_fn(batch):
        imgs, clss, bboxes, im_files, ori_shapes, ratio_pads = [], [], [], [], [], []
        batch_idxs = []
        for i, item in enumerate(batch):
            imgs.append(item['img'])
            im_files.append(item['im_file'])
            ori_shapes.append(item['ori_shape'])
            ratio_pads.append(item['ratio_pad'])
            n = len(item['cls'])
            if n > 0:
                clss.append(item['cls'])
                bboxes.append(item['bboxes'])
                batch_idxs.append(np.full((n,), i))
        final_batch = {
            'img': torch.from_numpy(np.stack(imgs, 0)).permute(0, 3, 1, 2),
            'im_file': im_files,
            'ori_shape': ori_shapes,
            'ratio_pad': ratio_pads,
        }
        if batch_idxs:
            final_batch['batch_idx'] = torch.from_numpy(np.concatenate(batch_idxs, 0))
            final_batch['cls'] = torch.from_numpy(np.concatenate(clss, 0))
            final_batch['bboxes'] = torch.from_numpy(np.concatenate(bboxes, 0))
        else:
            final_batch['batch_idx'] = torch.empty(0)
            final_batch['cls'] = torch.empty(0, 1)
            final_batch['bboxes'] = torch.empty(0, 4)
        return final_batch


class SlicedDetectionTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None):
        self.train_transform = overrides.pop('train_transform', None)
        self.val_transform = overrides.pop('val_transform', None)
        self.slicing_cfg = overrides.pop('slicing_cfg', None)
        self.train_fraction = overrides.pop('train_fraction', 1.0)
        self.val_fraction = overrides.pop('val_fraction', 1.0)
        super().__init__(overrides=overrides, _callbacks=_callbacks)

    def get_dataloader(self, dataset_path, batch_size, mode='train', rank=-1):
        """Builds and returns the PyTorch DataLoader."""
        assert mode in ('train', 'val'), f"Invalid dataloader mode '{mode}'"

        dataset = self.build_dataset(dataset_path, mode)

        sampler = None
        shuffle = False
        if mode == 'train':
            if self.world_size > 1:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=rank,
                    shuffle=True
                )
            else:
                shuffle = True

        workers = getattr(self.args, 'workers', 8)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=workers,
            sampler=sampler,
            collate_fn=getattr(dataset, 'collate_fn', None),
            worker_init_fn=seed_worker if mode == 'train' and workers > 0 else None,
            pin_memory=True,
            persistent_workers=workers > 0
        )

    def build_dataset(self, img_path, mode='train', batch=None):
        data_path = self.args.data
        if mode == 'train':
            dataset = SlicedDataset(
                data_config_path=data_path,
                split="train",
                slicing_cfg=self.slicing_cfg,
                augmentations=self.train_transform,
                fraction=self.train_fraction
            )
            dataset.collate_fn = SlicedDataset.collate_fn
            return dataset
        elif mode == 'val':
            val_slicing_cfg = deepcopy(self.slicing_cfg)
            val_slicing_cfg['overlap_ratio'] = 0.0
            dataset = SlicedDataset(
                data_config_path=data_path,
                split="val",
                slicing_cfg=val_slicing_cfg,
                augmentations=self.val_transform,
                fraction=self.val_fraction
            )
            dataset.collate_fn = SlicedDataset.collate_fn
            return dataset
        else:
            return super().build_dataset(img_path, mode, batch)

    def plot_training_labels(self):
        print("Skipping dataset label plotting for custom sliced dataset.")
        pass


def build_augmentations(aug_cfg):
    """Builds an Albumentations augmentation pipeline from a configuration dictionary."""
    if not aug_cfg or not aug_cfg.get('enable', False):
        return None

    pipeline = []
    
    # Horizontal flip
    if aug_cfg.get('hflip', 0) > 0:
        pipeline.append(A.HorizontalFlip(p=aug_cfg['hflip']))
    
    # Color jitter
    if aug_cfg.get('color_jitter_p', 0) > 0:
        pipeline.append(A.ColorJitter(
            brightness=aug_cfg.get('color_jitter_brightness', 0.2),
            contrast=aug_cfg.get('color_jitter_contrast', 0.2),
            saturation=aug_cfg.get('color_jitter_saturation', 0.2),
            hue=aug_cfg.get('color_jitter_hue', 0.1),
            p=aug_cfg['color_jitter_p']
        ))
    
    # Gaussian noise
    if aug_cfg.get('gauss_noise_p', 0) > 0:
        pipeline.append(A.GaussNoise(p=aug_cfg['gauss_noise_p']))
    
    # Blur
    if aug_cfg.get('blur_p', 0) > 0:
        pipeline.append(A.Blur(
            blur_limit=aug_cfg.get('blur_limit', 3),
            p=aug_cfg['blur_p']
        ))

    if not pipeline:
        return None

    return A.Compose(pipeline, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def setup_wandb(config, project_name, run_name):
    """Initialize Weights & Biases tracking."""
    if not WANDB_AVAILABLE:
        return None
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        reinit=True
    )
    return wandb.run


def export_metrics(results_dir, output_path):
    """Export training metrics to JSON for DVC tracking."""
    metrics = {}
    
    # Try to load results from Ultralytics output
    results_csv = Path(results_dir) / "results.csv"
    if results_csv.exists():
        import csv
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                # Clean up column names and extract metrics
                for key, value in last_row.items():
                    clean_key = key.strip().replace('/', '_').replace('(', '').replace(')', '')
                    try:
                        metrics[clean_key] = float(value)
                    except (ValueError, TypeError):
                        pass
    
    # Save metrics
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics exported to {output_path}")
    return metrics


def run_training(config):
    """Main function to run YOLOv8 training with slicing and W&B."""
    print("\n--- Preparing for training ---")

    # Extract configs
    aug_config = config.pop('augmentations', {})
    train_transform = build_augmentations(aug_config)
    slicing_cfg = config.pop('slicing')
    
    # W&B setup
    wandb_project = config.pop('wandb_project', 'yolo-sliced-training')
    wandb_run_name = config.pop('wandb_run_name', None)
    metrics_output = config.pop('metrics_output', 'outputs/train_metrics.json')
    
    # Setup W&B
    wandb_run = setup_wandb(
        config={**config, 'slicing': slicing_cfg, 'augmentations': aug_config},
        project_name=wandb_project,
        run_name=wandb_run_name or config.get('name', 'exp')
    )

    # Prepare YOLO overrides
    yolo_overrides = config
    yolo_overrides['train_transform'] = train_transform
    yolo_overrides['val_transform'] = None
    yolo_overrides['slicing_cfg'] = slicing_cfg
    yolo_overrides['train_fraction'] = yolo_overrides.pop('train_fraction', 1.0)
    yolo_overrides['val_fraction'] = yolo_overrides.pop('val_fraction', 1.0)
    
    model_path = yolo_overrides.get('model')
    if not model_path:
        print("Error: Model path ('model:') must be specified in the config.")
        sys.exit(1)
    
    try:
        model = YOLO(model_path)
        results = model.train(trainer=SlicedDetectionTrainer, **yolo_overrides)
        
        # Export metrics for DVC
        results_dir = Path(yolo_overrides.get('project', 'runs')) / yolo_overrides.get('name', 'exp')
        os.makedirs(Path(metrics_output).parent, exist_ok=True)
        metrics = export_metrics(results_dir, metrics_output)
        
        # Log final metrics to W&B
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
    parser = argparse.ArgumentParser(description="Train YOLOv8 model with slicing and W&B tracking.")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to configuration file.')
    parser.add_argument('--wandb-project', type=str, default=None, help='W&B project name.')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name.')
    parser.add_argument('--metrics-output', type=str, default='outputs/train_metrics.json', help='Path to save metrics JSON.')
    args, unknown_args = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.wandb_project:
        config['wandb_project'] = args.wandb_project
    if args.wandb_run_name:
        config['wandb_run_name'] = args.wandb_run_name
    config['metrics_output'] = args.metrics_output

    # Parse additional overrides
    override_parser = argparse.ArgumentParser()
    for key, value in config.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                arg_name = f'--{key}.{inner_key}'
                if inner_value is not None:
                    override_parser.add_argument(arg_name, type=type(inner_value), default=None)
                else:
                    override_parser.add_argument(arg_name, type=str, default=None)
        else:
            arg_name = f'--{key}'
            if value is not None:
                override_parser.add_argument(arg_name, type=type(value), default=None)
            else:
                override_parser.add_argument(arg_name, type=str, default=None)
    
    override_args = override_parser.parse_args(unknown_args)

    for key, value in vars(override_args).items():
        if value is not None:
            if '.' in key:
                main_key, sub_key = key.split('.')
                if main_key not in config:
                    config[main_key] = {}
                config[main_key][sub_key] = value
            else:
                config[key] = value
    
    run_training(config)


if __name__ == "__main__":
    main()
