# Schema Constitution

Status: Active
Scope: Rules for modeling non-trivial data, schema selection, validation, and review checklist.
Owners: Architecture
Last updated: 2026-01-05
Related: [data/DATA_JSONL_CONTRACT.md](../data/DATA_JSONL_CONTRACT.md), [training/REFERENCE.md](../training/REFERENCE.md), `src/datasets/contracts.py`, `src/config/schema.py`

## Purpose
Establish a consistent, scalable approach for modeling non-trivial data across Qwen3-VL. The goals are:
- Make data contracts explicit and discoverable.
- Reduce ad-hoc `dict`/`list` usage for meaningful structures.
- Improve validation, reviewability, and long-term maintainability.

## Definitions

### Boundary schema
A schema for data that crosses a boundary (file/JSON/JSONL, config, disk, network/API, CLI input, or public module interface). Boundary schemas must validate at creation.

### Serving and CLI
- **Serving**: runtime HTTP/RPC interfaces that accept external requests (inference endpoints).
- **CLI**: command-line entrypoints that parse user input (for example `python -m src.sft --config ...`).

### Non-trivial data (rubric)
**Hard triggers (any one => non-trivial):**
- Boundary I/O (config, JSON/JSONL, disk, network, CLI, public interface)
- Nested structures (mapping of mappings, list of mappings, list of lists)
- Stored or shared across modules (not just a local expression)

**Soft triggers (two or more => non-trivial):**
- 3+ semantically meaningful fields
- Optional/union fields that affect behavior
- Cross-field invariants or validation rules

**Trivial data**: no hard triggers and at most one soft trigger.

## Type selection rules
Use the smallest structured type that satisfies the contract.

1) **Pydantic BaseModel** (boundary with complex validation)
- Use for serving/CLI/request validation, cross-field constraints, or rich error reporting.
- Prefer when validation or coercion is a requirement.

2) **dataclass (frozen=True)** (internal structured state)
- Use for internal domain/config/state objects.
- Validate in `__post_init__` and parse external mappings in a `from_mapping` classmethod.

3) **TypedDict + validator** (mapping-shaped dataset rows)
- Use for JSON/JSONL records that must stay mapping-shaped.
- Provide a `validate_*` helper and only `cast(...)` after validation.

4) **JsonValue alias** (unstructured data)
- Use only when data is truly unstructured and must remain so.
- Keep unstructured payloads isolated in a field named `extra` or `raw`.

## Validation and error handling
- **Type mismatch**: raise `TypeError` with the full field path.
- **Value invalid**: raise `ValueError` with the full field path and constraint.
- Validate at boundaries, convert immediately to structured types, and keep `cast(...)` only after validation passes.

## Function signatures and returns
- Non-trivial arguments must use structured types (dataclass, TypedDict, or Pydantic).
- Prefer an `Options` or `Params` object when inputs are interdependent or exceed four fields.
- Composite returns must use a named structure.

## Naming and placement
- Use `XConfig`, `XParams`, `XOptions`, `XInput`, `XOutput`, `XRecord`, `XItem` naming.
- Place schemas in `contracts/` or `schema/` modules per domain.
- Keep validators adjacent to the schema definition.

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

## Schema review checklist (canonical)
Use this checklist in code reviews or when adding/modifying schemas:
- Non-trivial rubric applied (hard triggers or two soft triggers).
- Boundary data validated at creation and converted immediately to structured types.
- Schema type chosen correctly (Pydantic, dataclass, TypedDict) with justification.
- `dict`/`list` used only for trivial or explicitly unstructured data.
- `extra`/`raw` unstructured fields are isolated and documented.
- Type errors vs value errors are distinct and include full field paths.
- `Optional[T]` used only when `None` is meaningful (not as a default placeholder).

## Migration guidance
- Start at boundaries: replace raw mappings with `from_mapping` or validators.
- Move shared structures into `schema/` or `contracts/` modules.
- Keep `extra` for forward compatibility but avoid spreading unstructured payloads.
