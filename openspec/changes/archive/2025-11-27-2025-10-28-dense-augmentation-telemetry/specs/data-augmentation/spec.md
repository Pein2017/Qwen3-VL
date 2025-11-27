## ADDED Requirements

### Requirement: Augmentation Telemetry and Safety
The augmentation pipeline SHALL emit telemetry for affine/crop safety (padding ratios, coverage, skip reasons) and enforce polygon-aware coverage during crops.

#### Scenario: Crop coverage logging
- **WHEN** `RandomCrop` runs on a sample with quads or polys
- **THEN** it computes coverage using polygon clipping (not AABB only) and logs per-epoch coverage/skip stats via the telemetry hook.

#### Scenario: Padding ratio telemetry
- **WHEN** canvas expansion/padding aligns images to the required multiple
- **THEN** the pipeline records the padding ratio (pad_area / final_area) in telemetry so runs can flag excessive padding.

#### Scenario: Failure on missing telemetry
- **WHEN** telemetry sink is unavailable or disabled
- **THEN** augmentation still proceeds, but missing sinks are reported once with a warning; training is not blocked.
