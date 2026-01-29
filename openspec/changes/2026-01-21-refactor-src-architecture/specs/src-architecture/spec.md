# src-architecture (Delta)

## ADDED Requirements

### Requirement: `src` imports SHALL be lightweight and free of implicit side effects
Importing the top-level `src` package (and a defined set of lightweight subpackages) MUST NOT:
- import heavyweight runtime dependencies unless the caller explicitly imports a runtime/training entrypoint, and
- import heavyweight internal entrypoints as an indirect side effect.

For this change, **lightweight imports** are:
- `import src`
- `import src.utils`
- `import src.config`
- `import src.datasets.contracts`
- `import src.generation.contracts`

For this change, **heavyweight runtime dependencies** are defined (deterministically) as the following module names being present in `sys.modules` in a fresh interpreter:
- `torch`
- `transformers`
- `swift`
- `deepspeed` (if installed)

For this change, **heavyweight internal entrypoints** are:
- `src.sft`
- `src.stage_a.inference`
- `src.stage_b.runner`

Where heavyweight modules are exposed as convenience imports, the package SHALL use lazy import patterns (e.g., `__getattr__`) to preserve backward compatibility without incurring import-time cost.

Note: `from src.config import ConfigLoader` is considered an explicit training/config boundary import and MAY import heavyweight runtime dependencies.

#### Scenario: Importing lightweight packages does not trigger heavyweight imports
- **WHEN** a user executes `conda run -n ms python -c "import sys; import src, src.utils, src.config, src.datasets.contracts, src.generation.contracts; forbidden=['torch','transformers','swift','deepspeed','src.sft','src.stage_a.inference','src.stage_b.runner']; bad=[m for m in forbidden if m in sys.modules]; raise SystemExit(1 if bad else 0)"`
- **THEN** the command exits with status code `0`

#### Scenario: Heavyweight modules are imported only when explicitly requested
- **WHEN** a user executes `conda run -n ms python -c "import src; import src.sft"`
- **THEN** heavyweight dependencies MAY be imported

#### Scenario: ConfigLoader MAY import heavyweight dependencies when explicitly requested
- **WHEN** a user executes `conda run -n ms python -c "import src.config; from src.config import ConfigLoader"`
- **THEN** heavyweight dependencies MAY be imported

#### Scenario: Importing `src.sft` does not register GRPO rewards at import time
- **WHEN** a user executes `conda run -n ms python -c "from swift.plugin import orms; before=set(orms.keys()); import src.sft; from swift.plugin import orms as orms2; added=set(orms2.keys())-before; bad=[k for k in added if k.startswith(('summary.','dense.'))]; raise SystemExit(1 if bad else 0)"`
- **THEN** the command exits with status code `0`

### Requirement: CLI entrypoints SHALL remain stable while core logic moves into internal modules
The refactor SHALL preserve the public CLI entrypoints and their behavior:
- `python -m src.sft --config ...`
- `python -m src.stage_a.cli ...`
- `python -m src.stage_b.runner ...`

The CLI modules SHOULD become thin wrappers that validate inputs and delegate to internal pipeline/app modules.

#### Scenario: SFT CLI `--help` remains supported
- **WHEN** a user runs `conda run -n ms python -m src.sft --help`
- **THEN** the command exits with status code `0`
- **AND THEN** it prints a usage/help message

#### Scenario: Stage-A CLI `--help` remains supported
- **WHEN** a user runs `conda run -n ms python -m src.stage_a.cli --help`
- **THEN** the command exits with status code `0`
- **AND THEN** it prints a usage/help message

#### Scenario: Stage-B CLI `--help` remains supported
- **WHEN** a user runs `conda run -n ms python -m src.stage_b.runner --help`
- **THEN** the command exits with status code `0`
- **AND THEN** it prints a usage/help message

#### Scenario: Training config validation remains supported (no model required)
- **WHEN** a user runs `conda run -n ms python scripts/validate_sft_config.py --config configs/smoke/sft_dense_tiny.yaml`
- **THEN** the command exits with status code `0`

#### Scenario: Stage-B smoke validation remains supported (no model required)
- **WHEN** a user runs `conda run -n ms python scripts/stage_b_smoke.py`
- **THEN** the command exits with status code `0`

#### Scenario: Stage-B public `run_all` remains available (import-compatible)
- **WHEN** a user imports `run_all` from `src.stage_b`
- **THEN** the import succeeds
- **AND THEN** `run_all` orchestrates ingest → rollout → rule_search as before

### Requirement: Shared coercion helpers SHALL be centralized and reused
The codebase SHALL provide a single canonical coercion helper module for parsing common scalar values from unstructured mappings (e.g., YAML/JSON):
- boolean coercion (accepting the same string forms currently supported by training + Stage-B config loaders)
- int coercion
- float coercion

For this change, the accepted boolean string tokens (case-insensitive, whitespace-insensitive) SHALL be exactly:
- truthy: `true`, `1`, `yes`, `y`, `on`
- falsy: `false`, `0`, `no`, `n`, `off`

The shared coercion helpers MUST:
- accept values that are already of the target type (e.g., `bool` → `bool`) and return them unchanged
- treat string inputs as case-insensitive and whitespace-insensitive (`" YES "` → `True`)
- reject `bool` for int/float coercion (because `bool` is a subclass of `int`)
- raise `TypeError` for type mismatches (e.g., list/dict)
- raise `ValueError` for invalid string values (e.g., `"maybe"`)
- include a full field path in error messages when a path/context is provided

Downstream configuration loaders (training config, Stage-B config, dataset wrapper parsing) MUST use these shared helpers instead of reimplementing local variants.

#### Scenario: Boolean parsing is consistent across Stage-B and training config loaders
- **GIVEN** a config value `" YES "` for a boolean field
- **WHEN** the value is parsed by the training config loader and the Stage-B config loader
- **THEN** both parsers interpret the value as `True`
- **AND THEN** invalid boolean strings raise a `ValueError` naming the full field path

#### Scenario: Integer/float parsing rejects boolean inputs
- **GIVEN** a config value `true` for an integer or float field
- **WHEN** the value is parsed by the shared coercion helpers
- **THEN** it raises a `TypeError` naming the full field path

### Requirement: Summary JSON parsing/formatting SHALL be centralized
The codebase SHALL provide a single shared utility for Stage-A summary JSON handling used by Stage-A inference, Stage-A postprocessing, and Stage-B ingest/prompt assembly.

The utility SHALL support:
- extracting the summary JSON object from model output text (single-line or multi-line)
- formatting to a canonical single-line JSON string with stable separators (preserving current behavior)
- stripping optional domain/task header lines (`<DOMAIN=...>, <TASK=...>`) when present
- best-effort behavior: if no valid summary JSON object is found, return `None` rather than raising

The canonical formatter MUST preserve the existing contract:
- order top-level keys `统计`, `备注`, `分组统计` first (if present)
- preserve other keys after in their original parsed order (do not `sort_keys=True`)
- format using `ensure_ascii=False` and `separators=(", ", ": ")`

Stage-B MAY apply mission-specific redactions (e.g., removing `站点距离` evidence for non-RRU missions) after canonical formatting. Such redactions are intentionally outside the shared canonical formatter, and may cause Stage-B prompt strings to differ from Stage-A JSONL strings.

#### Scenario: Canonical summary JSON formatting is stable across call sites
- **GIVEN** a parsed summary JSON object mapping
- **WHEN** Stage-A formats the object using the shared canonical formatter
- **AND WHEN** Stage-B formats the same object using the shared canonical formatter
- **THEN** the formatted summary JSON is identical (byte-for-byte)

#### Scenario: Canonical formatter preserves extra top-level keys
- **GIVEN** a parsed summary JSON object mapping with the required `"统计"` key and an additional top-level key
- **WHEN** the shared canonical formatter is used
- **THEN** the formatted summary JSON includes the additional key (it is not dropped)

#### Scenario: Summary JSON extraction chooses the last JSON-line summary when multiple are present
- **GIVEN** model output text with two JSON objects on separate non-empty lines, and both contain the key `"统计"`
- **WHEN** the shared extractor is used
- **THEN** it returns the formatted JSON for the last such JSON line (closest to the end of the text)

#### Scenario: Summary JSON extraction fails softly when no JSON is present
- **GIVEN** model output text with no parseable JSON object containing the key `"统计"`
- **WHEN** the shared extractor is used
- **THEN** it returns `None`

### Requirement: Stage-A JSONL record payloads SHALL have explicit boundary validation
Stage-A group JSONL records SHALL be treated as a boundary contract and MUST be validated via a TypedDict + validator helper before writing or postprocessing.

The validator MUST enforce:
- mapping type at the top level
- required keys at the JSONL boundary: `group_id`, `mission`, `label`, `images`, `per_image`
- `group_id`, `mission`, `label` are strings
- `images` is a list of strings with length `N >= 1`
- `per_image` is a mapping with exactly `N` keys: `image_1` ... `image_N` (contiguous, no extras)
- each `per_image[image_i]` value is a non-empty string after stripping

Note: an empty per-image summary (including whitespace-only after stripping) is considered invalid model output. Implementations SHOULD fail fast on such outputs (e.g., by checking `if not text.strip(): ...`), and the validator makes this boundary invariant explicit before writing/postprocessing.

#### Scenario: Invalid Stage-A record fails fast with an actionable error
- **GIVEN** a Stage-A record with `per_image` set to a non-mapping value
- **WHEN** Stage-A attempts to write the record
- **THEN** it raises a `TypeError` or `ValueError` with a full field path (e.g., `stage_a.per_image`) and does not write a partial record

#### Scenario: Stage-A record image/per_image alignment is enforced
- **GIVEN** a Stage-A record with 3 images but only 2 `per_image` entries
- **WHEN** Stage-A validates the record for JSONL output
- **THEN** it raises a `ValueError` naming the full field path (e.g., `stage_a.per_image`) and does not write a partial record

#### Scenario: Stage-A record validator unit tests pass
- **WHEN** a user runs `conda run -n ms pytest -q tests/test_stage_a_record_validation.py`
- **THEN** the command exits with status code `0`

### Requirement: Dependency direction SHALL be enforced for foundation packages
To reduce circular-import pressure and isolate low-level utilities, the refactor SHALL enforce a minimal dependency direction:
- any module under `src.utils` MUST NOT import from module prefixes: `src.stage_a`, `src.stage_b`, `src.sft`, `src.trainers`, `src.training`
- any module under `src.generation` MUST NOT import from module prefixes: `src.stage_a`, `src.stage_b`, `src.sft`, `src.trainers`, `src.training`

The dependency direction SHALL be enforced via a repository script (`scripts/ci/check_import_boundaries.py`) and a small pytest wrapper test.

#### Scenario: Import-boundary check passes
- **WHEN** a user runs `conda run -n ms python scripts/ci/check_import_boundaries.py`
- **THEN** the command exits with status code `0`

#### Scenario: Pytest import-boundary test passes
- **WHEN** a user runs `conda run -n ms pytest -q tests/test_import_boundaries.py`
- **THEN** the command exits with status code `0`

### Requirement: Cross-module private helper imports SHALL be eliminated
Modules MUST NOT import underscore-prefixed helpers from other **top-level domain packages** under `src/`.

For this change, a top-level domain package is the first module segment under `src` (e.g., `src.datasets`, `src.stage_b`, `src.utils`).
Imports of underscore-prefixed helpers within the same top-level domain package are allowed.

If a helper is needed across modules, it SHALL be promoted into an explicit public utility module with a stable name and documented behavior.

This rule SHOULD be enforced by the repository import-boundary checks (AST-based) so new private imports are caught in CI.

#### Scenario: Fusion dataset does not rely on private helpers
- **WHEN** the unified fusion dataset needs shared fusion computations (quota computation, index sampling)
- **THEN** it calls a public helper module
- **AND THEN** it does not import underscore-prefixed functions from another module

### Requirement: Configuration-heavy construction surfaces SHALL use semantic grouping objects
When constructors or pipeline entrypoints require more than four interdependent configuration values, the implementation SHALL define a semantic grouping dataclass (e.g., `XOptions`, `XParams`) and pass that object through module boundaries.

The grouping objects MUST follow the Schema Constitution:
- defined as `@dataclass(frozen=True)`
- validate invariants in `__post_init__`
- parsed at boundaries via `from_mapping(...)` (or an equivalent explicit constructor), not by ad-hoc dict access throughout the pipeline
- placed in a domain-appropriate `schema`/`contracts` module (e.g., `src/stage_b/schema.py`, `src/datasets/contracts.py`)

#### Scenario: Dataset construction uses Options objects
- **WHEN** training constructs datasets for either single-dataset or fusion mode
- **THEN** dataset construction is driven by a small number of structured Options objects
- **AND THEN** the options are validated once at the boundary and reused downstream without re-parsing
