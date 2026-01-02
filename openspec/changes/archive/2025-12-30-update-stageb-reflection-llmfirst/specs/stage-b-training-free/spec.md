## MODIFIED Requirements

### Requirement: Stage-B SHALL evaluate group tickets without gradient updates by orchestrating a training-free sampling and judging loop.
Stage-B SHALL reuse the same model for rollout and reflection, SHALL trigger reflection on label mismatch or low-agreement/uncertainty/manual-review signals, and SHALL fall back to manual review when reflection cannot justify the ground-truth verdict.
#### Scenario: Reflection-first loop with widened eligibility
- Given normalized Stage-A summaries and labels, when Stage-B runs rollout + selection, then it MUST trigger reflection if (a) the selected verdict disagrees with the label OR (b) selection signals low agreement, uncertainty, or an explicit manual-review flag.
- When reflection runs and cannot surface evidence to support the ground-truth verdict, it SHALL mark the ticket `need_manual_review` rather than force-aligning the verdict.
- Reflection MUST reuse the loaded rollout model (no secondary critic model) and operate without gradient updates.
- When training/running, Stage-B SHALL process one mission at a time with a single `initial_guidance.json`; concurrent multi-mission training with different guidance seeds is out of scope.

### Requirement: Stage-B SHALL emit reproducible verdict artifacts that satisfy existing downstream contracts.
Artifacts SHALL include strict format failure routing and multi-mission label coverage while preserving deterministic JSONL outputs per run.
#### Scenario: Selection completes for a mission batch with strict format handling
- When rollout parsing fails (missing verdict or reason, malformed schema, or empty payload), the system SHALL write the entry to `failure_malformed.jsonl`, enqueue it in `manual_review_queue.jsonl`, and exclude it from selection/trajectory exports; no auto-salvage or coerced `format_ok=True` is permitted.
- Selection JSONL exports SHALL remain deterministic per seed/run and include manual-review flags and warnings; reruns with the same `run_name` SHALL override (recreate empty) per-run artifacts (`trajectories.jsonl`, `selections.jsonl`, `reflection_cache/`, `manual_review_queue.jsonl`, `failure_malformed.jsonl`) to avoid cross-run contamination (no backups required).***
- When building grouped reports, the system SHALL merge labels from all provided `stage_a_paths` keyed by mission/group_id instead of only the first Stage-A file.

### Requirement: Stage-B SHALL persist mission guidance and trajectory logs as fail-fast file artifacts for reuse.
Guidance updates SHALL deduplicate/merge/summarize rules, SHALL reuse mission guidance per run instead of resetting from seed, and SHALL keep per-run hygiene for caches and review queues.
#### Scenario: Guidance is updated after reflection
- When reflection proposes operations, the repository SHALL merge or summarize overlapping guidance entries (deduplicate near-duplicates, reuse existing keys when text matches, and collapse overlapping intents) instead of blind appends.
- Mission guidance under `{run_dir}/{mission}/guidance.json` SHALL start from the mission seed once per run_name and be reused across epochs; rerunning with the same run_name MUST NOT discard prior reflected guidance unless explicitly reset.
- Per-run artifacts (`reflection_cache`, `manual_review_queue.jsonl`, `failure_malformed.jsonl`) SHALL be cleared or versioned at run start to avoid reusing stale entries across reruns; guidance writes remain atomic with snapshot retention.

### Requirement: Stage-B SHALL assemble prompts with mission experiences explicitly and abort when injection fails.
Reflection and rollout prompts MUST include Stage-A per-image summaries or parsed Reason segments to ground guidance edits and verdict fixes.
#### Scenario: Prompts include Stage-A evidence
- When building rollout prompts, the system SHALL prepend numbered guidance experiences and Stage-A per-image summaries; an empty experiences dict remains a hard failure.
- When building reflection prompts/bundles, the system SHALL include Stage-A per-image summaries or extracted Reason segments alongside selected/competing verdicts so guidance edits are grounded; prompts must fail fast if evidence cannot be loaded. If rollout already carries the summaries in the ticket/bundle, reflection SHALL reuse them; raw responses MAY be logged for audit but are not required inputs to reflection.

### Requirement: Stage-B SHALL record minimal deterministic signals for each candidate to support LLM reflection.
Signals SHALL be limited to label alignment and vote-strength style agreement; confidence/self-consistency knobs are not required and agreement flags MUST propagate into reflection eligibility.
#### Scenario: Signals are generated after rollout
- Signal extraction MUST emit `label_match` and vote-strength/low-agreement indicators derived from selection; confidence/self_consistency fields MAY be omitted and SHALL NOT gate decisions.
- Low-agreement/uncertainty/manual-review flags from selection MUST be preserved on candidates/bundles and make the batch reflection-eligible even when labels match.

### Requirement: Stage-B SHALL run reflection with the rollout model and no CriticEngine.
Stage-B reflection SHALL operate without a separate critic stage and SHALL use the rollout model for counterfactual reasoning.
#### Scenario: Reflection runs after each complete batch without critic outputs
- Given trajectories and selection outputs (with label/Stage-A evidence), when reflection executes, it MUST build proposals without any CriticEngine inputs; the model instance used for rollout is reused for reflection generation.
- Reflection outputs SHALL remain strict-JSON; parse failures are fatal and logged with `reflection.debug_info`.
- Reflection actions SHALL be limited to rule edits (`op`: add|update|delete) with optional rationale/evidence; request-for-more-info or abstain/keep branches are removed. Empty/invalid JSON results in no guidance change and the batch is logged as ineligible.

### Requirement: Stage-B SHALL defer manual review to reflection when no supporting evidence exists.
Reflection SHALL decide manual-review routing after examining Stage-A summaries and rollout candidates.
#### Scenario: Manual review only when reflection sees no support
- If all usable rollout candidates for a group fail to support the ground-truth label and reflection produces no guidance update for the batch, the system MUST enqueue that group to `manual_review_queue.jsonl` with a `no_support_after_reflection` reason.
- Label mismatches or low-agreement alone SHALL NOT auto-enqueue manual review before reflection; they only make the bundle eligible for reflection.

## RENAMED Requirements
- FROM: Stage-B SHALL include an LLM-based CriticEngine for per-candidate evaluations.
  TO: Stage-B SHALL run reflection with the rollout model and no CriticEngine.***
