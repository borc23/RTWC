# YOLO Sliced Training Pipeline

A production-ready ML pipeline for training YOLOv8 models with image slicing, featuring:
- **DVC** for data and model versioning
- **Weights & Biases** for experiment tracking
- **SAHI** for sliced inference
- **Manual review gate** before pushing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DVC Pipeline Stages                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │ 
│  │ prepare  │──▶│  train   │──▶│ evaluate │──▶│   test   │      │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            |
│  data_stats.json  best.pt    eval_metrics   predictions/        │
│                      │                                          │
│                      └──────────▶ W&B Dashboard                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Initial Setup

```bash
# Clone/create your project
cd yolo-pipeline

# Run setup script
chmod +x scripts/*.sh
./scripts/setup.sh
```

### 2. Prepare Your Data

Copy your data to the `data/` directory:

```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/           # Optional
    ├── images/
    └── labels/
```

Track with DVC:
```bash
dvc add data
git add data.dvc .gitignore
git commit -m "Add training data"
```

### 3. Configure Training

Edit `configs/train_config.yaml` to match your needs:
- Model architecture
- Training hyperparameters
- Slicing parameters
- Augmentation settings

### 4. Login to W&B

```bash
wandb login
```

### 5. Run Pipeline

```bash
# Full pipeline
./scripts/run.sh full

# Or using DVC directly
dvc repro
```

### 6. Review & Push

After training completes:
```bash
./scripts/run.sh push
```

This will:
1. Show you all metrics
2. Compare with previous version
3. Let you decide whether to push
4. Optionally tag the release

## Directory Structure

```
yolo-pipeline/
├── configs/
│   └── train_config.yaml    # Training configuration
├── data/                    # Dataset (DVC tracked)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                  # Saved models (DVC tracked)
├── outputs/                 # Pipeline outputs
│   ├── data_stats.json
│   ├── train_metrics.json
│   ├── eval_metrics.json
│   ├── test_metrics.json
│   └── predictions/
├── runs/                    # Training runs
├── scripts/
│   ├── setup.sh
│   ├── run.sh
│   └── review_and_push.sh
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── test.py
├── dvc.yaml                 # Pipeline definition
├── params.yaml              # Tracked parameters
└── requirements.txt
```

## Common Commands

```bash
# Run full pipeline
./scripts/run.sh full

# Run specific stage
./scripts/run.sh train
./scripts/run.sh eval
./scripts/run.sh test

# Check status
./scripts/run.sh status

# View metrics
./scripts/run.sh metrics

# Review and push
./scripts/run.sh push

# Clean outputs
./scripts/run.sh clean
```

## DVC Commands

```bash
# Reproduce pipeline
dvc repro

# Run specific stage
dvc repro train

# Check what will run
dvc status

# View metrics
dvc metrics show

# Compare metrics
dvc metrics diff

# Push data/models to remote
dvc push

# Pull data/models from remote
dvc pull

# Checkout specific version
git checkout <commit>
dvc checkout
```

## Configuration Reference

### train_config.yaml

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data` | Path to data.yaml | `data/data.yaml` |
| `model` | Model architecture | `yolov8m.yaml` |
| `epochs` | Training epochs | 100 |
| `batch` | Batch size | 32 |
| `imgsz` | Image size | 640 |
| `slicing.slice_height` | Slice height | 640 |
| `slicing.slice_width` | Slice width | 640 |
| `slicing.overlap_ratio` | Overlap between slices | 0.2 |

### Augmentations

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable` | Enable augmentations | true |
| `hflip` | Horizontal flip prob | 0.5 |
| `color_jitter_p` | Color jitter prob | 0.75 |
| `gauss_noise_p` | Gaussian noise prob | 0.2 |
| `blur_p` | Blur probability | 0.5 |

## Workflow

### Training a New Model

1. Modify `configs/train_config.yaml` or `params.yaml`
2. Run `dvc repro`
3. Review results in W&B and locally
4. Run `./scripts/review_and_push.sh`
5. Tag version if satisfied

### Experimenting with Parameters

```bash
# Quick experiment with different epochs
python src/train.py --config configs/train_config.yaml --epochs 50

# Or modify params.yaml and run
dvc repro
```

### Restoring Previous Version

```bash
# List versions
git log --oneline

# Checkout specific version
git checkout <commit-hash>
dvc checkout

# Data and models now match that version
```

## Troubleshooting

### DVC remote not configured
```bash
dvc remote add -d local /path/to/dvc-storage
```

### W&B not logging
```bash
wandb login
# Or set environment variable
export WANDB_API_KEY=your_key
```

### Out of memory during training
- Reduce `batch` size
- Reduce `workers` count
- Use smaller model (yolov8n, yolov8s)

### Pipeline not detecting changes
```bash
dvc status  # Check what's changed
dvc repro -f  # Force rerun
```

## License

MIT License
