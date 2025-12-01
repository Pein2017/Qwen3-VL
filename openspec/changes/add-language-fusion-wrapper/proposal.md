# Proposal: add-language-fusion-wrapper

## Why
- Prevent language forgetting when training detection-heavy SFT runs.
- Integrate text-only datasets (COIG-CQIA) into the fusion pipeline with proper chat prompts.
- Ensure fusion supports message-style (no images) sources without breaking detection records.

## What
- Add a `chat` fusion wrapper for text-only JSONL.
- Provide safe chat prompts (system/user) for language sources.
- Allow fusion/builders to pass through pre-authored `messages` without injecting detection prompts.
- Ship a fusion config that mixes BBU target + LVIS + chat data, plus a derived training YAML.

## Impact/Risks
- Low code surface: wrapper registry, prompts, fusion config, builder pass-through.
- Risk: template mismatch or prompt override; mitigate with defaults and tests via `openspec validate` and a smoke config.

## Validation
- `openspec validate add-language-fusion-wrapper --strict`
- Config lint: ensure new YAML paths exist.
- Optional smoke: `scripts/train.sh --config configs/fused_data/last_6_langmix.yaml --debug` (small sample).
