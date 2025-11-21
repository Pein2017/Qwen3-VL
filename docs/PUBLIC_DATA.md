# Public Data Module (`public_data/`) Overview

This document introduces the **public data** submodule under `public_data/` at the repo root (`/data/Qwen3-VL/public_data`).

The goal of this module is to provide **geometry-aware, tested pipelines** for turning public detection/segmentation datasets (starting with **LVIS**) into JSONL files that match the Qwen3-VL training contract and can be used as **auxiliary datasets** in training and fusion.

> For how these auxiliary datasets are fused with BBU training, see:
> `openspec/changes/update-geometry-poly-fusion/design.md` ("`public_data/` Integration" section).

---

## Scope and Responsibilities

`public_data/` is a self-contained mini-project focused on:

- **Dataset engineering** for public vision datasets (LVIS now; Objects365 / Open Images later).
- **Geometry-aware conversion** from source formats (e.g., COCO-style bbox + segmentation) to Qwen3-VL JSONL with:
  - `bbox_2d`: `[x1, y1, x2, y2]` in **pixel coordinates**.
- `poly`: `[..., xn, yn]` + `poly_points` for N-point polygons. Experiments can simplify polygons in two ways via fusion config: `poly_fallback: bbox_2d` (all polys → boxes) or `poly_max_points: N` (only polys with more than N vertices are downgraded).
- **Validation & tests** to catch schema or geometry errors early.
- **Visualization tools** to visually inspect bounding boxes and polygons.

`public_data/` deliberately does **not** contain training code; it produces JSONL files that are then consumed by the main training stack under `configs/` and `src/`.

---

## How `public_data/` Fits into Qwen3-VL

At the project level, `public_data/` plays three roles:

- **Producer of auxiliary training data**: converts public datasets (currently LVIS) into JSONL that matches the Qwen3-VL dense-caption schema.
- **Geometry bridge**: exposes `bbox_2d` and N-point polygon (`poly` + `poly_points`) geometries in **pixel space**, ready for downstream normalization to `norm1000` in templates.
- **Quality gate**: provides tests and validation scripts to catch schema / geometry issues before training.

In training configs under `configs/`, these JSONL files are referenced via `custom.train_jsonl` / `custom.val_jsonl`. For multi-dataset fusion (BBU + LVIS as auxiliary), the detailed behavior is specified in:

- `openspec/changes/update-geometry-poly-fusion/design.md` (see the "`public_data/` Integration" section) and the dataset wrapper registry under `src/datasets/wrappers`, which binds each auxiliary dataset to its domain, default template, and augmentation policy.

---

## Where to Find Operational Details

This document is a **high-level overview** for the main repo. For concrete commands, directory layout, and troubleshooting, see:

- `public_data/README.md` – the single source of truth for:
  - LVIS download and conversion commands
  - JSONL schema details (bbox + polygon)
  - Sampling / validation workflows
  - Integration examples and common issues

As new public datasets are added (Objects365, Open Images, ...), they should follow the same pattern inside `public_data/`, with this file remaining the entry point for how the submodule relates to the rest of Qwen3-VL.

## Smart-resize (shared preprocessor)

- `public_data/scripts/convert_lvis.py --smart-resize` invokes the shared `SmartResizePreprocessor` (pixel budget + grid alignment) to rewrite images and geometry. Outputs default to `public_data/lvis/resized_<factor>_<blocks>/`.
- Datasets loaded by `DenseCaptionDataset` or `MultiSourceFusionDataset` resolve relative image paths against the JSONL parent and can optionally apply the same smart-resize guard via env (`SMART_RESIZE_GUARD=true`, `SMART_RESIZE_GUARD_OUTPUT_DIR=<dir>`), ensuring portable paths without relying on CWD or symlinks.
