
import argparse
import json
import os
import random
from collections import Counter, defaultdict

from PIL import Image

from src.config import ConfigLoader
from src.datasets.augmentation.builder import build_compose_from_config
from src.datasets.augmentation.ops import RandomCrop, SmallObjectZoomPaste

def analyze_dataset(jsonl_path, config_yaml, num_samples=20, seeds_per_sample=20):
    print(f"Analyzing {jsonl_path} with {config_yaml}...")
    
    # Load Config
    conf = ConfigLoader.load_yaml_with_extends(config_yaml)
    aug_cfg = conf.get("custom", {}).get("augmentation", {})
    
    # Ensure enabled
    if not aug_cfg.get("enabled", False):
        print("Augmentation disabled in config.")
        return

    # Build Pipeline
    pipeline = build_compose_from_config(aug_cfg)
    
    # Load Data
    records = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples: break
            records.append(json.loads(line))
            
    stats = {
        "crop_attempts": 0,
        "crop_triggered": 0,
        "crop_skip_reasons": Counter(),
        "zoom_attempts": 0,
        "zoom_added_objects": 0,
        "zoom_added_classes": Counter(),
        "total_objects_before": 0,
        "total_objects_after": 0
    }

    # Hack: set pipeline prob to 1.0 for relevant ops to stress test logic? 
    # Or keep as is to measure ACTUAL trigger rates?
    # The user asked for "random_crop trigger rate" -> implies actual config probability.
    
    root_dir = os.path.dirname(os.path.abspath(jsonl_path))
    os.environ["ROOT_IMAGE_DIR"] = root_dir

    for rec_idx, rec in enumerate(records):
        images = rec.get("images", [])
        objects = rec.get("objects", [])
        
        # Load images
        pil_images = []
        for img_path in images:
            if not os.path.isabs(img_path):
                img_path = os.path.join(root_dir, img_path)
            try:
                pil_images.append(Image.open(img_path).convert("RGB"))
            except Exception:
                pass
        
        if not pil_images: continue
        w, h = pil_images[0].size
        
        # Parse Geometries
        geoms = []
        for obj in objects:
            g = {}
            if "bbox_2d" in obj: g["bbox_2d"] = obj["bbox_2d"]
            elif "poly" in obj: g["poly"] = obj["poly"]
            elif "line" in obj: g["line"] = obj["line"]
            
            # Add desc for zoom whitelist check
            if "desc" in obj: g["desc"] = obj["desc"]
            
            if g: geoms.append(g)

        stats["total_objects_before"] += len(geoms) * seeds_per_sample

        for i in range(seeds_per_sample):
            rng = random.Random(2025 + rec_idx * 1000 + i)
            
            # Reset pipeline stats
            # We need to access individual ops to check their internal stats if available
            # But the standard Compose doesn't expose per-op stats easily unless we spy on them.
            # However, `RandomCrop` stores `last_skip_reason`.
            
            # Apply
            _, new_geoms = pipeline.apply(pil_images, geoms, width=w, height=h, rng=rng)
            
            stats["total_objects_after"] += len(new_geoms)
            
            # Inspect Ops
            for op in pipeline.ops:
                if isinstance(op, RandomCrop):
                    stats["crop_attempts"] += 1
                    if op.last_skip_reason:
                        stats["crop_skip_reasons"][op.last_skip_reason] += 1
                    elif op.last_kept_indices is not None:
                         stats["crop_triggered"] += 1
                
                # For zoom, we can infer it triggered if object count increased (and no crop)
                # Or we can check if the op has internal logging? 
                # The current SmallObjectZoomPaste doesn't expose "last_added_count" explicitly 
                # but adds `__aug_op` to geoms.
                
            # Count Zoom additions via metadata
            zoom_added = 0
            for g in new_geoms:
                if g.get("__aug_op") == "small_object_zoom_paste":
                    zoom_added += 1
                    desc = g.get("desc", "unknown")
                    # Extract class from desc "类别=XXX"
                    cls = "unknown"
                    if "类别=" in desc:
                        try:
                            cls = desc.split("类别=")[1].split(",")[0].split(" ")[0]
                        except:
                            cls = desc
                    else:
                        cls = desc
                    stats["zoom_added_classes"][cls] += 1
            
            stats["zoom_added_objects"] += zoom_added
            if zoom_added > 0:
                stats["zoom_attempts"] += 1 # roughly speaking, successful attempts

    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    analyze_dataset(args.jsonl, args.config)
