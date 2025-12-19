# Proposal: Multi-object and Line Copy-Paste Augmentation

## Summary
Add configuration-first, geometry-safe augmentation capabilities to densify small objects and cable-like lines by copy-pasting patches within and across images. The goal is to reduce the model's reliance on spatial priors ("objects only appear in canonical cabinet locations") and instead force it to visually recognize small items and thin lines wherever they appear on the canvas.

The change extends the existing `small_object_zoom_paste` PatchOp with optional background-aware placement and an object patch bank, and introduces two new PatchOps for object clusters and line segments. All new behavior is opt-in via YAML and compatible with the existing augmentation registry, curriculum, and telemetry.

## Motivation

Current augmentation covers affine transforms, random cropping with label filtering, and per-object small-object zoom-paste. This improves robustness but still leaves large background regions unoccupied and line objects underrepresented. Models tend to memorize typical layouts (e.g., screws or labels only near certain racks) and miss tiny objects or cables when they appear in unusual positions.

We want to:
- Increase the number and diversity of **small object** appearances, especially in visually empty areas.
- Expose **clusters** of related objects (labels + screws + connectors) in varied locations, not just original cabinets.
- Increase coverage for **line** geometries (cables/fibers) while preserving their curvature and recognizability.
- Keep augmentation infra simple: reusing existing PatchOp, geometry helpers, and YAML-driven configuration.

## Non-Goals

- No changes to Stage-B logic or downstream verdict guidance.
- No new dataset schema fields; all behavior lives inside augmentation/preprocessor.
- No change to existing configs unless explicitly updated; old augmentation configs remain valid and equivalent.

## High-level Design

1. **Extend `small_object_zoom_paste`**
   - Add optional background-aware placement: small objects are preferentially pasted into low-coverage regions using a coarse occupancy grid over the image.
   - Add support for multiple copies per target small object, gated by per-image caps and overlap rules.
   - Add an optional in-memory patch bank so small-object patches can be reused across images within a worker.
   - Default configuration preserves existing behavior (single copy, uniform placement, no bank).

2. **Add `object_cluster_copy_paste` PatchOp**
   - Build small multi-object clusters around seed objects by expanding their AABB and collecting nearby geometries.
   - Copy the resulting patch (image crop + cluster geometries) and paste it into background cells using the same occupancy and overlap logic as the extended `small_object_zoom_paste`.
   - Optionally source cluster patches from a patch bank in addition to the current image.

3. **Add `line_segment_copy_paste` PatchOp**
   - Select `line` geometries (cables/fibers) by length and optional class whitelist.
   - Extract either full-line or sliding-window segments, compute a small context AABB, and copy-paste the resulting patch.
   - Apply only mild scaling and translation to preserve line curvature and recognizability.
   - Place line segments into low-coverage regions while avoiding excessive overlap with existing objects or lines.
   - Optionally use a patch bank for lines.

4. **Reuse shared helpers**
   - Implement shared helpers inside `ops.py` for occupancy grid computation, background-cell sampling, and IoU-based placement rejection.
   - Introduce a small mixin for per-worker patch banks reused across the new PatchOps.

5. **Config and curriculum**
   - Expose all knobs via YAML under `custom.augmentation.ops` and make probabilities/scale ranges available to the existing curriculum scheduler.
   - Keep global `bypass_prob` semantics unchanged.

## Risks / Open Questions

- Over-aggressive densification could make training images unrealistically cluttered and harm convergence if probabilities and caps are misconfigured.
- Per-worker patch banks introduce stateful behavior across samples; we must define deterministic seeding and reasonable capacity limits.
- Cross-image patch reuse slightly blurs dataset boundaries; we should confirm this is acceptable for the current training regime.

