#!/usr/bin/env python3
"""
Sample subsets from converted JSONL datasets.

Supports multiple sampling strategies:
- stratified: Maintains frequency distribution (ideal for LVIS)
- uniform: Equal samples per category
- random: Pure random sampling
- top_k: Sample from most frequent K categories
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"! Error parsing line {line_num}: {e}")
    return samples


def save_jsonl(samples: List[Dict[str, Any]], path: str) -> None:
    """Save samples to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def analyze_dataset(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze category distribution in dataset.
    
    Returns:
        {
            "total_samples": int,
            "total_objects": int,
            "category_counts": {category: count},
            "samples_by_category": {category: [sample_indices]}
        }
    """
    category_counts = Counter()
    samples_by_category = defaultdict(list)
    total_objects = 0
    
    for idx, sample in enumerate(samples):
        objects = sample.get("objects", [])
        total_objects += len(objects)
        
        # Track which categories appear in this sample
        categories_in_sample = set()
        for obj in objects:
            cat = obj.get("desc", "unknown")
            categories_in_sample.add(cat)
            category_counts[cat] += 1
        
        # Add this sample to all categories it contains
        for cat in categories_in_sample:
            samples_by_category[cat].append(idx)
    
    return {
        "total_samples": len(samples),
        "total_objects": total_objects,
        "category_counts": dict(category_counts),
        "samples_by_category": dict(samples_by_category),
        "num_categories": len(category_counts)
    }


def sample_stratified(
    samples: List[Dict[str, Any]],
    num_samples: int,
    analysis: Dict[str, Any],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Stratified sampling: maintains category frequency distribution.
    
    Good for LVIS to preserve long-tail characteristics.
    Samples proportionally from each category's frequency.
    """
    random.seed(seed)
    
    category_counts = analysis["category_counts"]
    samples_by_category = analysis["samples_by_category"]
    total_objects = analysis["total_objects"]
    
    # Calculate target samples per category based on frequency
    target_per_category = {}
    for cat, count in category_counts.items():
        proportion = count / total_objects
        target_per_category[cat] = max(1, int(num_samples * proportion))
    
    # Sample from each category
    selected_indices = set()
    for cat, target in sorted(target_per_category.items(), key=lambda x: -x[1]):
        available = samples_by_category[cat]
        # Sample without replacement
        sample_count = min(target, len(available))
        selected = random.sample(available, sample_count)
        selected_indices.update(selected)
        
        if len(selected_indices) >= num_samples:
            break
    
    # Convert to list and trim if needed
    selected_indices = list(selected_indices)[:num_samples]
    return [samples[i] for i in sorted(selected_indices)]


def sample_uniform(
    samples: List[Dict[str, Any]],
    num_samples: int,
    analysis: Dict[str, Any],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Uniform sampling: equal samples per category.
    
    Good for balanced evaluation across all categories.
    """
    random.seed(seed)
    
    samples_by_category = analysis["samples_by_category"]
    num_categories = analysis["num_categories"]
    
    samples_per_category = max(1, num_samples // num_categories)
    
    selected_indices = set()
    for cat, available in samples_by_category.items():
        sample_count = min(samples_per_category, len(available))
        selected = random.sample(available, sample_count)
        selected_indices.update(selected)
    
    # Trim to exact count
    selected_indices = list(selected_indices)[:num_samples]
    return [samples[i] for i in sorted(selected_indices)]


def sample_random(
    samples: List[Dict[str, Any]],
    num_samples: int,
    analysis: Dict[str, Any],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Pure random sampling without replacement.
    """
    random.seed(seed)
    return random.sample(samples, min(num_samples, len(samples)))


def sample_top_k(
    samples: List[Dict[str, Any]],
    num_samples: int,
    analysis: Dict[str, Any],
    top_k: int = 100,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Sample from top-K most frequent categories.
    
    Good for focusing on common objects, faster training.
    """
    random.seed(seed)
    
    category_counts = analysis["category_counts"]
    samples_by_category = analysis["samples_by_category"]
    
    # Get top-K categories
    top_categories = sorted(category_counts.items(), key=lambda x: -x[1])[:top_k]
    top_cat_names = [cat for cat, _ in top_categories]
    
    print(f"  Top-{top_k} categories: {top_cat_names[:10]}...")
    
    # Pool all samples from top-K categories
    candidate_indices = set()
    for cat in top_cat_names:
        candidate_indices.update(samples_by_category[cat])
    
    # Random sample from pool
    candidate_indices = list(candidate_indices)
    selected = random.sample(candidate_indices, min(num_samples, len(candidate_indices)))
    
    return [samples[i] for i in sorted(selected)]


def print_stats(samples: List[Dict[str, Any]], title: str = "Dataset") -> None:
    """Print dataset statistics."""
    analysis = analyze_dataset(samples)
    
    print(f"\n{'='*60}")
    print(f"{title} Statistics")
    print(f"{'='*60}")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Total objects: {analysis['total_objects']}")
    print(f"  Unique categories: {analysis['num_categories']}")
    print(f"  Avg objects/sample: {analysis['total_objects'] / max(analysis['total_samples'], 1):.2f}")
    
    # Category frequency distribution
    counts = list(analysis['category_counts'].values())
    if counts:
        print(f"\n  Category frequency distribution:")
        print(f"    Min: {min(counts)}")
        print(f"    Max: {max(counts)}")
        print(f"    Mean: {sum(counts) / len(counts):.1f}")
        print(f"    Median: {sorted(counts)[len(counts)//2]}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sample subsets from JSONL datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sampling Strategies:

  stratified (recommended for LVIS):
    Maintains frequency distribution from original dataset.
    Good for preserving long-tail characteristics.
    
  uniform:
    Equal samples per category.
    Good for balanced evaluation.
    
  random:
    Pure random sampling.
    Simplest approach.
    
  top_k:
    Sample from K most frequent categories.
    Good for common object focus.

Examples:

  # Stratified sampling (5000 samples, all categories)
  python sample_dataset.py \\
    --input lvis/processed/train.jsonl \\
    --output lvis/processed/samples/train_5k_stratified.jsonl \\
    --num_samples 5000 \\
    --strategy stratified
  
  # Uniform sampling (3000 samples)
  python sample_dataset.py \\
    --input lvis/processed/train.jsonl \\
    --output lvis/processed/samples/train_3k_uniform.jsonl \\
    --num_samples 3000 \\
    --strategy uniform
  
  # Top-100 categories (1000 samples)
  python sample_dataset.py \\
    --input lvis/processed/train.jsonl \\
    --output lvis/processed/samples/train_1k_top100.jsonl \\
    --num_samples 1000 \\
    --strategy top_k \\
    --top_k 100
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of samples to select"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["stratified", "uniform", "random", "top_k"],
        default="stratified",
        help="Sampling strategy (default: stratified)"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top categories for top_k strategy (default: 100)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed statistics"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.input):
        print(f"✗ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("="*60)
    print("Dataset Sampling Tool")
    print("="*60)
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Seed: {args.seed}")
    print("="*60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    samples = load_jsonl(args.input)
    print(f"  Loaded {len(samples)} samples")
    
    if args.num_samples > len(samples):
        print(f"! Warning: Requested {args.num_samples} samples but only {len(samples)} available")
        args.num_samples = len(samples)
    
    # Analyze
    print("\n[2/4] Analyzing dataset...")
    analysis = analyze_dataset(samples)
    if args.stats:
        print_stats(samples, "Original Dataset")
    
    # Sample
    print(f"\n[3/4] Sampling with '{args.strategy}' strategy...")
    
    if args.strategy == "stratified":
        sampled = sample_stratified(samples, args.num_samples, analysis, args.seed)
    elif args.strategy == "uniform":
        sampled = sample_uniform(samples, args.num_samples, analysis, args.seed)
    elif args.strategy == "random":
        sampled = sample_random(samples, args.num_samples, analysis, args.seed)
    elif args.strategy == "top_k":
        sampled = sample_top_k(samples, args.num_samples, analysis, args.top_k, args.seed)
    else:
        print(f"✗ Unknown strategy: {args.strategy}")
        sys.exit(1)
    
    print(f"  Selected {len(sampled)} samples")
    
    # Save
    print(f"\n[4/4] Saving to: {args.output}")
    save_jsonl(sampled, args.output)
    
    # Print statistics
    if args.stats:
        print_stats(sampled, "Sampled Dataset")
    
    print("\n" + "="*60)
    print("✓ Sampling Complete!")
    print("="*60)
    print(f"\nOutput saved to: {args.output}")
    print(f"Samples: {len(sampled)}")
    
    # Quick validation
    sample_analysis = analyze_dataset(sampled)
    print(f"Categories: {sample_analysis['num_categories']}")
    print(f"Objects: {sample_analysis['total_objects']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

