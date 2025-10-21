#!/usr/bin/env python3
"""
Dataset Merger

Combines multiple processed datasets into unified training sets for scalable training.
Supports merging datasets while maintaining traceability and balanced teacher selection.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List


# Configure UTF-8 encoding for stdout/stderr if supported
try:
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass

try:
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass

logger = logging.getLogger(__name__)


class DatasetMerger:
    """Merges multiple processed datasets with intelligent teacher rebalancing."""

    def __init__(self, base_output_dir: str = "data"):
        """Initialize merger with base output directory."""
        self.base_output_dir = Path(base_output_dir)
        self.combined_dir = self.base_output_dir / "combined"
        self.combined_dir.mkdir(parents=True, exist_ok=True)

    def merge_datasets(
        self, dataset_names: List[str], max_teachers: int = 20, val_ratio: float = 0.2
    ) -> Dict[str, int]:
        """
        Merge multiple datasets into unified training sets.

        Args:
            dataset_names: List of dataset names to merge
            max_teachers: Maximum teacher samples to include
            val_ratio: Validation split ratio

        Returns:
            Dictionary with counts of merged samples
        """
        logger.info(f"🔄 Merging {len(dataset_names)} datasets: {dataset_names}")

        # Collect all samples from datasets
        all_train_samples = []
        all_val_samples = []
        all_teacher_samples = []

        dataset_stats = {}

        for dataset_name in dataset_names:
            dataset_dir = self.base_output_dir / dataset_name
            if not dataset_dir.exists():
                logger.warning(f"Dataset directory not found: {dataset_dir}")
                continue

            stats = self._load_dataset(
                dataset_dir, all_train_samples, all_val_samples, all_teacher_samples
            )
            dataset_stats[dataset_name] = stats
            logger.info(f"  {dataset_name}: {stats}")

        # Rebalance teacher samples across datasets
        balanced_teachers = self._rebalance_teachers(all_teacher_samples, max_teachers)

        # Write merged outputs
        output_stats = {
            "train": len(all_train_samples),
            "val": len(all_val_samples),
            "teacher": len(balanced_teachers),
            "total": len(all_train_samples)
            + len(all_val_samples)
            + len(balanced_teachers),
        }

        self._write_merged_outputs(
            all_train_samples, all_val_samples, balanced_teachers
        )
        self._write_merge_metadata(dataset_names, dataset_stats, output_stats)

        logger.info(f"✅ Merged datasets: {output_stats}")
        return output_stats

    def _load_dataset(
        self,
        dataset_dir: Path,
        all_train: List[Dict],
        all_val: List[Dict],
        all_teacher: List[Dict],
    ) -> Dict[str, int]:
        """Load samples from a single dataset."""
        stats = {"train": 0, "val": 0, "teacher": 0}

        # Load train samples
        train_file = dataset_dir / "train.jsonl"
        if train_file.exists():
            train_samples = self._load_jsonl(train_file)
            # Add dataset source metadata
            for sample in train_samples:
                sample["_dataset_source"] = dataset_dir.name
            all_train.extend(train_samples)
            stats["train"] = len(train_samples)

        # Load validation samples
        val_file = dataset_dir / "val.jsonl"
        if val_file.exists():
            val_samples = self._load_jsonl(val_file)
            for sample in val_samples:
                sample["_dataset_source"] = dataset_dir.name
            all_val.extend(val_samples)
            stats["val"] = len(val_samples)

        # Load teacher samples
        teacher_file = dataset_dir / "teacher.jsonl"
        if teacher_file.exists():
            teacher_samples = self._load_jsonl(teacher_file)
            for sample in teacher_samples:
                sample["_dataset_source"] = dataset_dir.name
            all_teacher.extend(teacher_samples)
            stats["teacher"] = len(teacher_samples)

        return stats

    def _rebalance_teachers(
        self, all_teachers: List[Dict], max_teachers: int
    ) -> List[Dict]:
        """Rebalance teacher samples to ensure diversity across datasets and labels."""
        if len(all_teachers) <= max_teachers:
            return all_teachers

        logger.info(f"🎓 Rebalancing {len(all_teachers)} teachers → {max_teachers}")

        # Group by dataset source
        teachers_by_dataset = {}
        for teacher in all_teachers:
            dataset = teacher.get("_dataset_source", "unknown")
            if dataset not in teachers_by_dataset:
                teachers_by_dataset[dataset] = []
            teachers_by_dataset[dataset].append(teacher)

        # Group by unique label combinations
        teachers_by_labels = {}
        for teacher in all_teachers:
            labels = set()
            for obj in teacher.get("objects", []):
                labels.add(obj.get("desc", ""))
            label_key = tuple(sorted(labels))
            if label_key not in teachers_by_labels:
                teachers_by_labels[label_key] = []
            teachers_by_labels[label_key].append(teacher)

        # Select balanced set prioritizing:
        # 1. Label diversity (different label combinations)
        # 2. Dataset diversity (equal representation)
        # 3. Geometry diversity (if available)

        selected_teachers = []
        dataset_counts = {name: 0 for name in teachers_by_dataset.keys()}
        used_label_combinations = set()

        # First pass: Select one teacher per unique label combination
        for label_key, teachers in teachers_by_labels.items():
            if len(selected_teachers) >= max_teachers:
                break

            # Prefer teachers from under-represented datasets
            teachers_sorted = sorted(
                teachers,
                key=lambda t: (
                    dataset_counts[t.get("_dataset_source", "unknown")],
                    -len(t.get("objects", [])),
                ),
            )

            selected_teacher = teachers_sorted[0]
            selected_teachers.append(selected_teacher)
            dataset_counts[selected_teacher.get("_dataset_source", "unknown")] += 1
            used_label_combinations.add(label_key)

        # Second pass: Fill remaining slots with dataset balance priority
        remaining_teachers = [t for t in all_teachers if t not in selected_teachers]
        while len(selected_teachers) < max_teachers and remaining_teachers:
            # Find teacher from least represented dataset
            min_count = min(dataset_counts.values())
            candidate_datasets = [
                name for name, count in dataset_counts.items() if count == min_count
            ]

            best_teacher = None
            for teacher in remaining_teachers:
                if teacher.get("_dataset_source", "unknown") in candidate_datasets:
                    best_teacher = teacher
                    break

            if best_teacher is None:
                best_teacher = remaining_teachers[0]

            selected_teachers.append(best_teacher)
            dataset_counts[best_teacher.get("_dataset_source", "unknown")] += 1
            remaining_teachers.remove(best_teacher)

        logger.info(f"  Selected teachers by dataset: {dataset_counts}")
        logger.info(
            f"  Unique label combinations covered: {len(used_label_combinations)}"
        )

        return selected_teachers

    def _write_merged_outputs(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        teacher_samples: List[Dict],
    ) -> None:
        """Write merged output files."""
        logger.info("💾 Writing merged outputs...")

        # Write individual files
        self._write_jsonl(train_samples, self.combined_dir / "train.jsonl")
        self._write_jsonl(val_samples, self.combined_dir / "val.jsonl")
        self._write_jsonl(teacher_samples, self.combined_dir / "teacher.jsonl")

        # Write combined file
        all_samples = teacher_samples + train_samples + val_samples
        self._write_jsonl(all_samples, self.combined_dir / "all_samples.jsonl")

        logger.info(
            f"  Written {len(train_samples)} train, {len(val_samples)} val, {len(teacher_samples)} teacher samples"
        )

    def _write_merge_metadata(
        self,
        dataset_names: List[str],
        dataset_stats: Dict[str, Dict],
        output_stats: Dict[str, int],
    ) -> None:
        """Write metadata about the merge process."""
        metadata = {
            "merged_datasets": dataset_names,
            "source_stats": dataset_stats,
            "output_stats": output_stats,
            "merge_timestamp": str(Path(__file__).stat().st_mtime),
        }

        with open(
            self.combined_dir / "merge_metadata.json", "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info("📊 Merge metadata written")

    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """Load JSONL file."""
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _write_jsonl(self, samples: List[Dict], file_path: Path) -> None:
        """Write JSONL file."""
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def list_available_datasets(self) -> List[str]:
        """List all available processed datasets."""
        datasets = []
        for item in self.base_output_dir.iterdir():
            if item.is_dir() and item.name != "combined":
                # Check if it has the expected files
                if (item / "train.jsonl").exists() or (
                    item / "all_samples.jsonl"
                ).exists():
                    datasets.append(item.name)
        return sorted(datasets)


def main():
    """Command line interface for dataset merging."""
    import argparse

    parser = argparse.ArgumentParser(description="Merge multiple processed datasets")
    parser.add_argument("--datasets", nargs="+", help="Dataset names to merge")
    parser.add_argument("--output_dir", default="data", help="Base output directory")
    parser.add_argument(
        "--max_teachers", type=int, default=20, help="Maximum teacher samples"
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--log_level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        encoding="utf-8",
    )

    merger = DatasetMerger(args.output_dir)

    if args.list:
        datasets = merger.list_available_datasets()
        print(f"Available datasets: {datasets}")
        return

    if not args.datasets:
        datasets = merger.list_available_datasets()
        if len(datasets) >= 2:
            print(f"Auto-merging all available datasets: {datasets}")
            args.datasets = datasets
        else:
            print("No datasets specified and less than 2 available")
            return

    result = merger.merge_datasets(args.datasets, args.max_teachers)
    print(f"\n✅ Merge complete: {result}")


if __name__ == "__main__":
    main()
