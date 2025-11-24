#!/usr/bin/env python3
"""
LVIS Visualization Tool

Visualizes LVIS annotations on images:
- Bounding boxes (bbox_2d)
- Polygons (segmentation)

Usage:
    python vis_tools/visualize_lvis.py --num_samples 5
    python vis_tools/visualize_lvis.py --split val --mode polygon
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_lvis_annotations(annotation_path):
    """Load LVIS annotations."""
    print(f"Loading annotations from: {annotation_path}")
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    # Build lookup structures
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"  Loaded {len(images)} images")
    print(f"  Loaded {len(data['annotations'])} annotations")
    print(f"  Loaded {len(categories)} categories")
    
    return images, categories, annotations_by_image


def find_available_images(image_root, image_ids, max_check=100):
    """Find which images are actually downloaded."""
    available = []
    
    # Check a sample of images
    sample_ids = random.sample(list(image_ids), min(max_check, len(image_ids)))
    
    for img_id in sample_ids:
        img_info = image_ids[img_id]
        
        # Try multiple possible field names
        file_name = (
            img_info.get('coco_file_name') or 
            img_info.get('file_name') or
            img_info.get('coco_url', '').split('/')[-1]  # Extract from URL
        )
        
        if not file_name:
            # Skip if we can't determine filename
            continue
        
        img_path = os.path.join(image_root, file_name)
        
        if os.path.exists(img_path):
            available.append(img_id)
    
    return available


def coco_bbox_to_xyxy(bbox):
    """Convert COCO bbox [x,y,w,h] to [x1,y1,x2,y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def visualize_sample(
    image_path,
    annotations,
    categories,
    mode='bbox',
    output_path=None,
    show_labels=True,
    max_polygons=5
):
    """
    Visualize one sample with annotations.
    
    Args:
        image_path: Path to image
        annotations: List of LVIS annotations
        categories: Category mapping
        mode: 'bbox', 'polygon', or 'both'
        output_path: Save path (if None, display)
        show_labels: Show category labels
        max_polygons: Max polygons to draw (to avoid clutter)
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_array)
    ax.axis('off')
    
    # Color map
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    drawn_polygons = 0
    
    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]
        
        # Get category name
        cat_id = ann['category_id']
        cat_name = categories[cat_id]['name'] if cat_id in categories else 'unknown'
        
        # Draw bbox
        if mode in ['bbox', 'both']:
            bbox = ann['bbox']
            x1, y1, x2, y2 = coco_bbox_to_xyxy(bbox)
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                linestyle='-' if mode == 'bbox' else '--'
            )
            ax.add_patch(rect)
            
            # Add label
            if show_labels and mode == 'bbox':
                ax.text(
                    x1, y1 - 5,
                    cat_name,
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
                )
        
        # Draw polygon
        if mode in ['polygon', 'both'] and 'segmentation' in ann:
            if drawn_polygons >= max_polygons:
                continue
            
            segmentation = ann['segmentation']
            if isinstance(segmentation, list):
                for seg in segmentation:
                    if isinstance(seg, list) and len(seg) >= 6:
                        # Convert to polygon coordinates
                        points = np.array(seg).reshape(-1, 2)
                        
                        polygon = patches.Polygon(
                            points,
                            linewidth=1.5,
                            edgecolor=color,
                            facecolor=(*color[:3], 0.2),  # Semi-transparent
                            linestyle='-'
                        )
                        ax.add_patch(polygon)
                        
                        # Add label at polygon center
                        if show_labels:
                            center_x = points[:, 0].mean()
                            center_y = points[:, 1].mean()
                            ax.text(
                                center_x, center_y,
                                f"{cat_name}\n[{len(points)} coords]",
                                color='white',
                                fontsize=7,
                                ha='center',
                                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1)
                            )
                        
                        drawn_polygons += 1
                        break  # Only draw first part for clarity
    
    title = f"LVIS Visualization ({mode} mode)\n"
    title += f"{len(annotations)} annotations"
    if mode in ['polygon', 'both']:
        title += f", showing {drawn_polygons}/{min(len(annotations), max_polygons)} polygons"
    
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LVIS annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show 5 random samples with bboxes
  python vis_tools/visualize_lvis.py --num_samples 5
  
  # Show polygons instead
  python vis_tools/visualize_lvis.py --mode polygon --num_samples 3
  
  # Show both bbox and polygons
  python vis_tools/visualize_lvis.py --mode both --num_samples 2
  
  # Use validation split
  python vis_tools/visualize_lvis.py --split val
  
  # Save to file instead of display
  python vis_tools/visualize_lvis.py --save --output_dir vis_tools/output
        """
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val'],
        default='train',
        help='Dataset split (default: train)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['bbox', 'polygon', 'both'],
        default='polygon',
        help='Visualization mode (default: bbox)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3,
        help='Number of samples to visualize (default: 3)'
    )
    
    parser.add_argument(
        '--max_polygons',
        type=int,
        default=5,
        help='Max polygons per image (default: 5)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save images instead of displaying'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='vis_tools/output',
        help='Output directory for saved images'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Paths
    base_dir = '.'
    annotation_path = os.path.join(
        base_dir, 'lvis', 'raw', 'annotations', f'lvis_v1_{args.split}.json'
    )
    image_root = os.path.join(base_dir, 'lvis', 'raw', 'images', f'{args.split}2017')
    
    # Check if annotations exist
    if not os.path.exists(annotation_path):
        print(f"✗ Error: Annotation file not found: {annotation_path}")
        sys.exit(1)
    
    # Check if images are extracted
    if not os.path.exists(image_root):
        print(f"✗ Error: Image directory not found: {image_root}")
        print(f"\nPlease extract images first:")
        print(f"  cd {os.path.dirname(image_root)}")
        print(f"  unzip {args.split}2017.zip")
        sys.exit(1)
    
    print("="*60)
    print("LVIS Visualization Tool")
    print("="*60)
    print(f"  Split: {args.split}")
    print(f"  Mode: {args.mode}")
    print(f"  Samples: {args.num_samples}")
    print("="*60 + "\n")
    
    # Load annotations
    images, categories, annotations_by_image = load_lvis_annotations(annotation_path)
    
    # Find available images
    print(f"\nChecking for available images...")
    available_ids = find_available_images(image_root, images, max_check=200)
    
    if not available_ids:
        print(f"✗ No images found in {image_root}")
        print(f"\nMake sure images are extracted.")
        sys.exit(1)
    
    print(f"  Found {len(available_ids)} available images")
    
    # Filter to images with annotations
    available_with_anns = [
        img_id for img_id in available_ids 
        if img_id in annotations_by_image and len(annotations_by_image[img_id]) > 0
    ]
    
    if not available_with_anns:
        print(f"✗ No available images have annotations")
        sys.exit(1)
    
    print(f"  {len(available_with_anns)} have annotations")
    
    # Sample images
    sample_ids = random.sample(
        available_with_anns, 
        min(args.num_samples, len(available_with_anns))
    )
    
    print(f"\n{'='*60}")
    print(f"Visualizing {len(sample_ids)} samples...")
    print('='*60)
    
    # Create output directory if saving
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize each sample
    for idx, img_id in enumerate(sample_ids, 1):
        img_info = images[img_id]
        
        # Try multiple possible field names
        file_name = (
            img_info.get('coco_file_name') or 
            img_info.get('file_name') or
            img_info.get('coco_url', '').split('/')[-1]
        )
        
        if not file_name:
            print(f"  ✗ Skipping: cannot determine filename for image {img_id}")
            continue
        
        img_path = os.path.join(image_root, file_name)
        anns = annotations_by_image[img_id]
        
        print(f"\n[{idx}/{len(sample_ids)}] {file_name}")
        print(f"  Size: {img_info['width']}x{img_info['height']}")
        print(f"  Annotations: {len(anns)}")
        
        # Show annotation details
        if len(anns) <= 10:
            for i, ann in enumerate(anns):
                cat_name = categories[ann['category_id']]['name']
                has_seg = 'segmentation' in ann and ann['segmentation']
                seg_info = ""
                if has_seg and isinstance(ann['segmentation'], list):
                    num_parts = len(ann['segmentation'])
                    if num_parts > 0 and isinstance(ann['segmentation'][0], list):
                        num_coords = len(ann['segmentation'][0]) // 2
                        seg_info = f" [{num_coords} coords]"
                print(f"    [{i}] {cat_name}{seg_info}")
        else:
            print(f"    (showing first 10)")
            for i, ann in enumerate(anns[:10]):
                cat_name = categories[ann['category_id']]['name']
                has_seg = 'segmentation' in ann and ann['segmentation']
                seg_info = ""
                if has_seg and isinstance(ann['segmentation'], list):
                    num_parts = len(ann['segmentation'])
                    if num_parts > 0 and isinstance(ann['segmentation'][0], list):
                        num_coords = len(ann['segmentation'][0]) // 2
                        seg_info = f" [{num_coords} coords]"
                print(f"    [{i}] {cat_name}{seg_info}")
        
        # Prepare output path
        output_path = None
        if args.save:
            output_path = os.path.join(
                args.output_dir,
                f"{args.split}_{args.mode}_{idx:03d}_{os.path.splitext(file_name)[0]}.png"
            )
        
        # Visualize
        try:
            visualize_sample(
                img_path,
                anns,
                categories,
                mode=args.mode,
                output_path=output_path,
                show_labels=True,
                max_polygons=args.max_polygons
            )
        except Exception as e:
            print(f"  ✗ Error visualizing: {e}")
            continue
    
    print("\n" + "="*60)
    print("✓ Visualization Complete!")
    print("="*60)
    
    if args.save:
        print(f"\nImages saved to: {args.output_dir}")
    else:
        print(f"\nNote: Use --save to save images instead of displaying")


if __name__ == "__main__":
    main()

