"""
Data Preparation Script.
Validates dataset structure, generates statistics, and prepares data.yaml.
"""
import argparse
import json
import yaml
import sys
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm


def validate_dataset(data_dir, splits=['train', 'val']):
    """
    Validate dataset structure and generate statistics.
    """
    data_path = Path(data_dir)
    stats = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'splits': {}
    }
    
    for split in splits:
        split_stats = {
            'images': 0,
            'labels': 0,
            'images_without_labels': 0,
            'labels_without_images': 0,
            'total_objects': 0,
            'class_distribution': defaultdict(int),
            'image_sizes': [],
            'corrupt_images': []
        }
        
        images_dir = data_path / split / 'images'
        labels_dir = data_path / split / 'labels'
        
        if not images_dir.exists():
            stats['errors'].append(f"Missing images directory: {images_dir}")
            stats['valid'] = False
            continue
        
        if not labels_dir.exists():
            stats['warnings'].append(f"Missing labels directory: {labels_dir}")
            labels_dir = None
        
        image_files = set()
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
            image_files.update(p.stem for p in images_dir.glob(ext))
        
        label_files = set()
        if labels_dir:
            label_files = set(p.stem for p in labels_dir.glob('*.txt'))
        
        split_stats['images'] = len(image_files)
        split_stats['labels'] = len(label_files)
        
        images_without_labels = image_files - label_files
        labels_without_images = label_files - image_files
        
        split_stats['images_without_labels'] = len(images_without_labels)
        split_stats['labels_without_images'] = len(labels_without_images)
        
        if labels_without_images:
            stats['warnings'].append(f"{split}: {len(labels_without_images)} labels without corresponding images")
        
        print(f"Validating {split} set...")
        for img_stem in tqdm(image_files, desc=f"Checking {split}"):
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                candidate = images_dir / f"{img_stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            
            if img_path:
                try:
                    with Image.open(img_path) as img:
                        split_stats['image_sizes'].append(img.size)
                except Exception as e:
                    split_stats['corrupt_images'].append(str(img_path))
            
            if labels_dir:
                label_path = labels_dir / f"{img_stem}.txt"
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls = int(parts[0])
                                split_stats['class_distribution'][cls] += 1
                                split_stats['total_objects'] += 1
        
        # Convert defaultdict to regular dict for JSON serialization
        split_stats['class_distribution'] = dict(split_stats['class_distribution'])
        
        if split_stats['image_sizes']:
            widths = [s[0] for s in split_stats['image_sizes']]
            heights = [s[1] for s in split_stats['image_sizes']]
            split_stats['image_size_stats'] = {
                'min_width': min(widths),
                'max_width': max(widths),
                'avg_width': sum(widths) / len(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'avg_height': sum(heights) / len(heights)
            }
        
        del split_stats['image_sizes']
        
        stats['splits'][split] = split_stats
    
    return stats


def create_data_yaml(data_dir, output_path, class_names=None, test_exists=False):
    """
    Create or update data.yaml for the dataset.
    """
    data_path = Path(data_dir).resolve()
    
    if class_names is None:
        existing_yaml = data_path / 'data.yaml'
        if existing_yaml.exists():
            with open(existing_yaml, 'r') as f:
                existing = yaml.safe_load(f)
                class_names = existing.get('names', {})
        else:
            all_classes = set()
            for split in ['train', 'val']:
                labels_dir = data_path / split / 'labels'
                if labels_dir.exists():
                    for label_file in labels_dir.glob('*.txt'):
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    all_classes.add(int(parts[0]))
            
            class_names = {i: f'class_{i}' for i in sorted(all_classes)}
    
    config = {
        'path': str(data_path),
        'train': 'train/images',
        'val': 'val/images',
        'names': class_names
    }
    
    if test_exists:
        config['test'] = 'test/images'
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate dataset for YOLO training.")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory.')
    parser.add_argument('--output-yaml', type=str, default=None, help='Output path for data.yaml.')
    parser.add_argument('--stats-output', type=str, default='outputs/data_stats.json', help='Output path for statistics.')
    parser.add_argument('--classes', type=str, nargs='+', default=None, help='Class names (in order).')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    splits = []
    for split in ['train', 'val', 'test']:
        if (data_dir / split / 'images').exists():
            splits.append(split)
    
    if not splits:
        print(f"Error: No valid splits found in {data_dir}")
        print("Expected structure:")
        print("  data/")
        print("    train/images/")
        print("    train/labels/")
        print("    val/images/")
        print("    val/labels/")
        sys.exit(1)
    
    print(f"Found splits: {splits}")
    
    stats = validate_dataset(data_dir, splits)
    
    print("\n" + "="*50)
    print("DATASET VALIDATION SUMMARY")
    print("="*50)
    
    for split, split_stats in stats['splits'].items():
        print(f"\n{split.upper()}:")
        print(f"  Images: {split_stats['images']}")
        print(f"  Labels: {split_stats['labels']}")
        print(f"  Total objects: {split_stats['total_objects']}")
        print(f"  Images without labels: {split_stats['images_without_labels']}")
        
        if split_stats['class_distribution']:
            print(f"  Class distribution:")
            for cls, count in sorted(split_stats['class_distribution'].items()):
                print(f"    Class {cls}: {count}")
        
        if split_stats.get('image_size_stats'):
            size_stats = split_stats['image_size_stats']
            print(f"  Image sizes: {size_stats['min_width']}x{size_stats['min_height']} to {size_stats['max_width']}x{size_stats['max_height']}")
    
    if stats['errors']:
        print("\nERRORS:")
        for err in stats['errors']:
            print(f"  ❌ {err}")
    
    if stats['warnings']:
        print("\nWARNINGS:")
        for warn in stats['warnings']:
            print(f"  ⚠️  {warn}")
    
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")
    
    if args.output_yaml:
        output_yaml = args.output_yaml
    else:
        output_yaml = data_dir / 'data.yaml'
    
    class_names = None
    if args.classes:
        class_names = {i: name for i, name in enumerate(args.classes)}
    
    config = create_data_yaml(
        data_dir, 
        output_yaml, 
        class_names=class_names,
        test_exists='test' in splits
    )
    print(f"Data config saved to: {output_yaml}")
    
    if not stats['valid']:
        print("\nDataset validation FAILED")
        sys.exit(1)
    else:
        print("\nDataset validation PASSED")


if __name__ == "__main__":
    main()
