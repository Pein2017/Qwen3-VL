# Qwen3‑VL Docs

Status: Active
Scope: Canonical documentation index and directory map.
Owners: Documentation
Last updated: 2026-01-05
Related: [overview/README.md](overview/README.md), [data/README.md](data/README.md), [training/README.md](training/README.md), [runtime/README.md](runtime/README.md), [ops/README.md](ops/README.md), [reference/README.md](reference/README.md)

## Quick Navigation
- **Overview** → [overview/README.md](overview/README.md), [overview/ARCHITECTURE.md](overview/ARCHITECTURE.md), [overview/CHANGELOG.md](overview/CHANGELOG.md)
- **Data pipeline** → [data/README.md](data/README.md), [data/DATA_PREPROCESSING_PIPELINE.md](data/DATA_PREPROCESSING_PIPELINE.md), [data/DATA_JSONL_CONTRACT.md](data/DATA_JSONL_CONTRACT.md)
- **Training** → [training/README.md](training/README.md), [training/TRAINING_PLAYBOOK.md](training/TRAINING_PLAYBOOK.md), [training/REFERENCE.md](training/REFERENCE.md)
- **Runtime** → [runtime/README.md](runtime/README.md), [runtime/STAGE_A_RUNTIME.md](runtime/STAGE_A_RUNTIME.md), [runtime/STAGE_B_RUNTIME.md](runtime/STAGE_B_RUNTIME.md)
- **Reference** → [reference/README.md](reference/README.md), [reference/SCHEMA_CONSTITUTION.md](reference/SCHEMA_CONSTITUTION.md), [reference/PROMPTS_REFERENCE.md](reference/PROMPTS_REFERENCE.md), [reference/stage-B-knowledge-Chinese.md](reference/stage-B-knowledge-Chinese.md)
- **Operations** → [ops/README.md](ops/README.md), [ops/deployment.md](ops/deployment.md), [ops/UPSTREAM_DEPENDENCIES.md](ops/UPSTREAM_DEPENDENCIES.md), [ops/CODEX_MCP_INSTALLATION.md](ops/CODEX_MCP_INSTALLATION.md)
- **Specs & governance** → [openspec/AGENTS.md](../openspec/AGENTS.md), [openspec/project.md](../openspec/project.md)

### Suggested Reading Order
1. **Orientation** — [overview/README.md](overview/README.md), [overview/ARCHITECTURE.md](overview/ARCHITECTURE.md)
2. **Intake → schema** — [data/README.md](data/README.md), [data/DATA_PREPROCESSING_PIPELINE.md](data/DATA_PREPROCESSING_PIPELINE.md), [data/DATA_JSONL_CONTRACT.md](data/DATA_JSONL_CONTRACT.md), [data/DATA_AND_DATASETS.md](data/DATA_AND_DATASETS.md), [data/BBU_RRU_BUSINESS_KNOWLEDGE.md](data/BBU_RRU_BUSINESS_KNOWLEDGE.md)
3. **Augmentation & fusion** — [data/DATA_AUGMENTATION.md](data/DATA_AUGMENTATION.md), [data/POLYGON_SUPPORT.md](data/POLYGON_SUPPORT.md), [data/UNIFIED_FUSION_DATASET.md](data/UNIFIED_FUSION_DATASET.md)
4. **Training** — [training/TRAINING_PLAYBOOK.md](training/TRAINING_PLAYBOOK.md), [training/REFERENCE.md](training/REFERENCE.md), [training/GRPO_MS_SWIFT_PIPELINE.md](training/GRPO_MS_SWIFT_PIPELINE.md)
5. **Runtime** — [runtime/STAGE_A_RUNTIME.md](runtime/STAGE_A_RUNTIME.md), [runtime/STAGE_A_STAGE_B.md](runtime/STAGE_A_STAGE_B.md), [runtime/STAGE_B_RUNTIME.md](runtime/STAGE_B_RUNTIME.md)
6. **Reference & ops** — [reference/README.md](reference/README.md), [ops/README.md](ops/README.md)

### Documentation Ownership & Directory Map

| Directory | Primary doc(s) | Scope |
|-----------|----------------|-------|
| `src/` | [training/REFERENCE.md](training/REFERENCE.md), [training/TRAINING_PLAYBOOK.md](training/TRAINING_PLAYBOOK.md) | Core training/inference implementation (`src/sft.py`, datasets, trainers). |
| `src/stage_a/` | [runtime/STAGE_A_RUNTIME.md](runtime/STAGE_A_RUNTIME.md), [runtime/STAGE_A_STAGE_B.md](runtime/STAGE_A_STAGE_B.md) | Stage‑1 per-image object recognition and summary emission. |
| `src/stage_b/` | [runtime/STAGE_B_RUNTIME.md](runtime/STAGE_B_RUNTIME.md), [runtime/STAGE_A_STAGE_B.md](runtime/STAGE_A_STAGE_B.md) | Stage‑2 verdict loop (rule_search). |
| `data_conversion/` | [data/DATA_PREPROCESSING_PIPELINE.md](data/DATA_PREPROCESSING_PIPELINE.md), [data/DATA_AND_DATASETS.md](data/DATA_AND_DATASETS.md) (Conversion section) | Offline preprocessing from annotation exports; taxonomy, resize, validation. |
| `public_data/` | [data/PUBLIC_DATA.md](data/PUBLIC_DATA.md) | LVIS and auxiliary datasets (download, convert, sample, validate, visualize). |
| `scripts/` | [scripts/README.md](../scripts/README.md), [training/TRAINING_PLAYBOOK.md](training/TRAINING_PLAYBOOK.md), [runtime/STAGE_B_RUNTIME.md](runtime/STAGE_B_RUNTIME.md) | Canonical entrypoints: training, inference, Stage‑A/B launchers, dataset fusion. |
| `configs/` | [training/TRAINING_PLAYBOOK.md](training/TRAINING_PLAYBOOK.md), [training/REFERENCE.md](training/REFERENCE.md), [training/GRPO_MS_SWIFT_PIPELINE.md](training/GRPO_MS_SWIFT_PIPELINE.md) | YAML training presets and overlays (including `configs/train/grpo/` GRPO post‑training templates). |
| `openspec/` | [openspec/AGENTS.md](../openspec/AGENTS.md), [openspec/project.md](../openspec/project.md) | Change-management specs and proposal workflow. |
| `vis_tools/` | [data/DATA_AUGMENTATION.md](data/DATA_AUGMENTATION.md), [vis_tools/README_CROP_VIS.md](../vis_tools/README_CROP_VIS.md) | Visualization/debug scripts for augmentation and QA spot checks. |

Whenever you add or modify code in the directories above, update the associated doc in the same PR to keep the handbook current.
The auto-context router uses this directory map as its source of truth; update this map first, then sync the router context map.

## Full Index

### Overview
- [overview/README.md](overview/README.md)
- [overview/ARCHITECTURE.md](overview/ARCHITECTURE.md)
- [overview/CHANGELOG.md](overview/CHANGELOG.md)
- [overview/AUDIT_LOG.md](overview/AUDIT_LOG.md)

### Data
- [data/README.md](data/README.md)
- [data/DATA_PREPROCESSING_PIPELINE.md](data/DATA_PREPROCESSING_PIPELINE.md)
- [data/DATA_JSONL_CONTRACT.md](data/DATA_JSONL_CONTRACT.md)
- [data/DATA_AND_DATASETS.md](data/DATA_AND_DATASETS.md)
- [data/DATA_AUGMENTATION.md](data/DATA_AUGMENTATION.md)
- [data/POLYGON_SUPPORT.md](data/POLYGON_SUPPORT.md)
- [data/UNIFIED_FUSION_DATASET.md](data/UNIFIED_FUSION_DATASET.md)
- [data/PUBLIC_DATA.md](data/PUBLIC_DATA.md)
- [data/BBU_RRU_BUSINESS_KNOWLEDGE.md](data/BBU_RRU_BUSINESS_KNOWLEDGE.md)

### Training
- [training/README.md](training/README.md)
- [training/TRAINING_PLAYBOOK.md](training/TRAINING_PLAYBOOK.md)
- [training/REFERENCE.md](training/REFERENCE.md)
- [training/GRPO_MS_SWIFT_PIPELINE.md](training/GRPO_MS_SWIFT_PIPELINE.md)

### Runtime
- [runtime/README.md](runtime/README.md)
- [runtime/STAGE_A_RUNTIME.md](runtime/STAGE_A_RUNTIME.md)
- [runtime/STAGE_A_STAGE_B.md](runtime/STAGE_A_STAGE_B.md)
- [runtime/STAGE_B_RUNTIME.md](runtime/STAGE_B_RUNTIME.md)

### Operations
- [ops/README.md](ops/README.md)
- [ops/deployment.md](ops/deployment.md)
- [ops/UPSTREAM_DEPENDENCIES.md](ops/UPSTREAM_DEPENDENCIES.md)
- [ops/CODEX_MCP_INSTALLATION.md](ops/CODEX_MCP_INSTALLATION.md)

### Reference
- [reference/README.md](reference/README.md)
- [reference/SCHEMA_CONSTITUTION.md](reference/SCHEMA_CONSTITUTION.md)
- [reference/PROMPTS_REFERENCE.md](reference/PROMPTS_REFERENCE.md)
- [reference/DIAGNOSIS_AND_REVIEW.md](reference/DIAGNOSIS_AND_REVIEW.md)
- [reference/CODEX_SUBAGENTS_ORCHESTRATION.md](reference/CODEX_SUBAGENTS_ORCHESTRATION.md)
- [reference/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md](reference/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md)
- [reference/stage-B-knowledge-Chinese.md](reference/stage-B-knowledge-Chinese.md)

## Scripts & Tooling
- Script inventory: [scripts/README.md](../scripts/README.md)
- Visualization tooling: [vis_tools/README_CROP_VIS.md](../vis_tools/README_CROP_VIS.md)
