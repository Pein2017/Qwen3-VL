"""Quick probe for fusion packing groups and counts.
Usage: python scripts/debug_fusion_groups.py configs/fused_data/debug.yaml
This avoids model load by using a stub template.
"""
import sys
import types
from collections import Counter
from pathlib import Path

from src.config import ConfigLoader
from src.datasets.fusion import FusionConfig
from src.datasets.unified_fusion_dataset import FusionCaptionDataset
from src.packing.grouped_packing import GroupedPackingDataset

cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/fused_data/debug.yaml")

# Load YAML to resolve fusion_config path and sample limits
cfg_raw = ConfigLoader.load_yaml_with_extends(str(cfg_path))
custom = cfg_raw.get("custom", {})
fusion_cfg_path = custom.get("fusion_config")
train_sample_limit = custom.get("train_sample_limit") or custom.get("sample_limit")

if not fusion_cfg_path:
    raise SystemExit("custom.fusion_config not set in config")

# Resolve fusion config path (repo-relative allowed)
fusion_cfg_path = Path(fusion_cfg_path)
if not fusion_cfg_path.is_absolute():
    fusion_cfg_path = (Path.cwd() / fusion_cfg_path).resolve()

fusion_cfg = FusionConfig.from_file(str(fusion_cfg_path))

# Stub template only needs max_length + encode returning length/labels
stub_tmpl = types.SimpleNamespace(system=None, max_length=4096)
stub_tmpl.encode = lambda merged, return_length=False: {
    "input_ids": list(range(32)),
    "labels": [0] * 32,
    "metadata": merged.get("metadata", {}),
}

# Build dataset
train_ds = FusionCaptionDataset(
    fusion_config=fusion_cfg,
    base_template=stub_tmpl,
    user_prompt=custom.get("user_prompt", "user"),
    emit_norm=custom.get("emit_norm", "norm1000"),
    json_format=custom.get("json_format", "standard"),
    augmenter=None,
    bypass_prob=float(custom.get("bypass_prob", 0.0)),
    curriculum_state=None,
    use_summary=bool(custom.get("use_summary", False)),
    system_prompt_dense=None,
    system_prompt_summary=None,
    seed=42,
    shuffle=True,
    sample_limit=train_sample_limit,
    split="train",
)

print("epoch_counts", train_ds._epoch_counts)

pack_ds = GroupedPackingDataset(
    train_ds,
    template=stub_tmpl,
    packing_length=stub_tmpl.max_length,
    group_key_fn=lambda row: (row.get("metadata", {}) or {}).get("_fusion_source", "default"),
)
counts = Counter(pack_ds[i]["packed_group"] for i in range(len(pack_ds)))
print("pack_count", counts)
print("groups_first_30", [pack_ds[i]["packed_group"] for i in range(min(30, len(pack_ds)))])
