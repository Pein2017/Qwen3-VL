# Schema Constitution

Status: Active
Scope: Rules for modeling non-trivial data, schema selection, and validation.
Owners: Architecture
Last updated: 2026-01-05

## Purpose
Establish a consistent, scalable approach for modeling non-trivial data across Qwen3-VL. The goals are:
- Make data contracts explicit and discoverable.
- Reduce ad-hoc `dict`/`list` usage for meaningful structures.
- Improve validation, reviewability, and long-term maintainability.

## Definitions

### Boundary schema
A schema for data that crosses a boundary.

A boundary is any point where:
- data leaves one responsibility domain and enters another, or
- downstream code is expected to rely on stable semantic assumptions.

Typical boundaries include (but are not limited to):
- file / JSON / JSONL ingestion
- config loading
- disk or network I/O
- CLI and serving interfaces
- public module interfaces

Boundary schemas must validate at creation.

### Serving and CLI
- **Serving**: runtime HTTP/RPC interfaces that accept external requests (inference endpoints).
- **CLI**: command-line entrypoints that parse user input (for example `python -m src.sft --config ...`).

### Semantic grouping
A semantic grouping is a structured object that intentionally bundles related fields
which are meant to be interpreted together.

Semantic grouping is preferred when:
- multiple fields jointly represent one conceptual unit, or
- correctness depends on the relationship between fields rather than individual values.

Semantic groupings serve as carriers of meaning across boundaries and module interfaces.

### Non-trivial data (rubric C)
The rubric applies to function signatures, return types, and class attributes.

**Non-trivial** when a dict/list represents a record with multiple semantic fields, nested structures, or heterogeneous value types.

Data is also considered non-trivial when:
- the meaning of one field depends on the presence or interpretation of another field, or
- the structure is intended to be consumed as a conceptual whole by downstream code.

**Trivial** when the structure is a simple lookup or flat list of primitives (for example `Mapping[str, str | int | float | bool]` or `list[int]`), or a local-only helper expression.

## Type selection rules
Use the smallest structured type that satisfies the contract.

At boundaries, prefer a single semantic grouping over multiple loosely related parameters
or ad-hoc mappings.

1) **Pydantic BaseModel** (serving/CLI boundary)
- Use for serving/CLI/request validation, cross-field constraints, or rich error reporting.
- **Scope**: permitted only for serving/CLI boundary schemas unless a module already depends on Pydantic.

2) **dataclass (frozen=True)** (internal structured state)
- Use for internal domain/config/state objects.
- Validate in `__post_init__` and parse external mappings in a `from_mapping` classmethod.

3) **TypedDict + validator** (mapping-shaped dataset rows)
- Use for JSON/JSONL records that must stay mapping-shaped.
- Provide a `validate_*` helper and only `cast(...)` after validation.

4) **Explicitly unstructured mappings** (escape hatch)
- Allowed only when the payload is intentionally unstructured.
- Must be documented in a docstring and validated as a `Mapping` or `Sequence` at entry.
- Prefer isolating unstructured data in fields named `extra` or `raw` when feasible.

## Validation and error handling
- **Type mismatch**: raise `TypeError` with the full field path.
- **Value invalid**: raise `ValueError` with the full field path and constraint.
- Validate at boundaries, convert immediately to structured types, and keep `cast(...)` only after validation passes.

Validation is required at boundaries.
Within a boundary, internal transformations may assume validated invariants
and are not required to re-validate intermediate states.

## Function signatures and returns
- Non-trivial arguments must use structured types (dataclass, TypedDict, or Pydantic).
- Prefer an `Options` or `Params` object when inputs are interdependent or exceed four fields.
- Composite returns must use a named structure.

Functions that conceptually operate on or produce a single unit of meaning
should accept or return a corresponding semantic grouping,
even if the underlying data could be represented as multiple independent values.

## Naming and placement
- Use `XConfig`, `XParams`, `XOptions`, `XInput`, `XOutput`, `XRecord`, `XItem` naming.
- Place schemas in `contracts/` or `schema/` modules per domain.
- Keep validators adjacent to the schema definition.

Names should reflect the conceptual role of the data rather than its transport form.
Avoid names that expose only implementation details (e.g. `data_dict`, `tmp_result`).

## Examples (before/after)

### Example A: mapping -> dataclass
**Before**
```python
from typing import Any

def parse_optim(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_size": int(raw.get("batch_size", 4)),
        "lr": float(raw["lr"]),
    }
```

**After**
```python
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

@dataclass(frozen=True)
class OptimConfig:
    batch_size: int = 4
    lr: float = 1e-4

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OptimConfig":
        if not isinstance(raw, Mapping):
            raise TypeError("optim must be a mapping")
        batch_size = int(raw.get("batch_size", 4))
        lr = float(raw["lr"])
        if batch_size <= 0:
            raise ValueError("optim.batch_size must be > 0")
        if lr <= 0:
            raise ValueError("optim.lr must be > 0")
        return cls(batch_size=batch_size, lr=lr)
```

### Example B: dataset row -> TypedDict + validator
```python
from typing import Any, Mapping, Sequence, TypedDict, cast

class SampleRecord(TypedDict, total=False):
    image_id: str
    captions: Sequence[str]


def validate_sample(raw: Mapping[str, Any]) -> SampleRecord:
    if not isinstance(raw, Mapping):
        raise TypeError("sample must be a mapping")
    captions = raw.get("captions", [])
    if not isinstance(captions, Sequence):
        raise ValueError("sample.captions must be a sequence")
    return cast(SampleRecord, raw)
```

### Example C: boundary validation -> Pydantic
```python
from pydantic import BaseModel, Field

class RunRequest(BaseModel):
    config_path: str = Field(min_length=1)
    dry_run: bool = False
```

## Migration guidance
- Start at boundaries: replace raw mappings with `from_mapping` or validators.
- Move shared structures into `schema/` or `contracts/` modules.
- Keep `extra` for forward compatibility but avoid spreading unstructured payloads.

When refactoring existing code, prioritize:
- collapsing loosely related parameters into semantic groupings, and
- making boundary-crossing data explicit and named.

## Backward-compatibility policy
There are no backward-compatibility guarantees for the `src/` refactor. Internal call sites may change to align with the constitution.
