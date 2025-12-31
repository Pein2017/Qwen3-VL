# Design: Prompt Profiles + Domain Packs (Training vs Inference)

## Goals
- Separate training prompts (format + task criterion only) from inference prompts (Stage‑A + Stage‑B) that include domain‑specific business knowledge.
- Centralize prompt composition in typed Python dataclasses.
- Eliminate duplicated safety rules and improve prompt selection transparency.

## Prompt Roles
Two explicit profiles are introduced for summary mode:
- **summary_train_min**: minimal training prompt, includes only format rules and task criterion
  (including evidence‑only / no‑hallucination constraints).
- **summary_runtime**: inference prompt, includes safety constraints + domain knowledge + mission rules (when applicable).

## Dataclass Model (Python)
Proposed new module: `src/prompts/summary_profiles.py` (exact names are adjustable).

Example structure:
- `SummaryPromptProfile`
  - `name`
  - `format_spec`
  - `task_criterion`
  - `safety_constraints` (optional)
  - `include_domain_pack`
  - `include_mission_rules`

Domain knowledge packs in `src/prompts/domain_packs.py`:
- `DomainKnowledgePack`
  - `domain` (`bbu|rru`)
  - `schema_hint`
  - `prior_rules`
  - `restrictions`
  - `mission_rules` (optional per‑mission additions)

These dataclasses are the source of truth for business knowledge blocks.

## Composition Order (Summary System Prompt)
1. Format spec
2. Task criterion
3. Safety constraints (runtime only)
4. Domain knowledge pack (runtime only)
5. Mission rules (runtime only, when mission provided)

This guarantees training prompts are minimal while inference prompts are rich and domain‑specific.

## Domain Resolution
- Stage‑A: domain resolved from CLI `--dataset` (`bbu|rru`).
- Stage‑B: domain resolved from config only:
  1) `config.domain_map[mission]` if configured
  2) `config.default_domain` fallback
- Missing or unknown domain MUST raise a validation error before prompt construction.

## Config/CLI Interface
- Training config:
  - `prompts.profile: summary_train_min | summary_runtime`
  - `prompts.domain: bbu | rru` (optional; only used by runtime profile)
  - `prompts.system`/`prompts.user` remain authoritative overrides.
- Stage‑A CLI:
  - Add `--prompt_profile` and `--domain` if needed; default to runtime profile and dataset.
- Stage‑B config:
  - Add `domain_map` and `default_domain` under the Stage‑B config section for domain resolution.
  - Stage‑B does not consume `ticket.domain`; domain must be resolvable from config.

## Compatibility Notes
- Default behavior:
  - Summary training uses `summary_train_min` unless overridden.
  - Stage‑A/Stage‑B inference uses `summary_runtime` with domain packs.
- Existing prompt overrides continue to work unchanged.

## Concise Inference Domain Block
Inference domain blocks SHALL be concise and avoid enumerating the full list of
item strings already present in training data. The block should serve as a
category‑level glossary only, and be appended via simple concatenation with the
training prompt.

Example (BBU, concise):
```
【BBU领域提示（简版）】
只做客观罗列；desc 原文保持不改写。
核心类别：BBU设备、挡风板、螺丝/光纤插头、光纤、电线、标签、需复核备注。
常见关系：安装螺丝/光纤插头（含BBU端/ODF端/接地螺丝）、光纤保护与弯曲半径、电线捆扎、标签可识别/无法识别。
```

RRU domain guidance remains pending and must be explicitly marked as such in
the domain pack registry until authored.
