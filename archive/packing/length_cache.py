from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def build_cache_fingerprint(
    *,
    augmentation_cfg: Mapping[str, Any] | None,
    template_id: str,
    data_paths: Sequence[str],
) -> str:
    payload = {
        "augmentation": augmentation_cfg or {},
        "template": template_id,
        "data_paths": sorted(str(p) for p in data_paths),
    }
    blob = _safe_json(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class LengthCache:
    fingerprint: str
    lengths: Mapping[int, int]
    meta: Mapping[str, Any]

    @classmethod
    def load(cls, path: str) -> "LengthCache":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fp = payload.get("fingerprint") or payload.get("hash")
        lengths_raw = payload.get("lengths") or {}
        meta = payload.get("meta") or {}
        if not isinstance(fp, str) or not fp:
            raise ValueError("length cache missing fingerprint/hash")
        lengths: dict[int, int] = {}
        if isinstance(lengths_raw, Mapping):
            for k, v in lengths_raw.items():
                try:
                    key_int = int(k)
                    lengths[key_int] = int(v)
                except Exception as exc:
                    raise ValueError(f"invalid length entry {k!r}: {v!r}") from exc
        else:
            raise ValueError("length cache 'lengths' must be a mapping of sample_id -> length")
        return cls(fingerprint=fp, lengths=lengths, meta=meta)

    def save(self, path: str) -> None:
        payload = {
            "fingerprint": self.fingerprint,
            "lengths": {str(k): int(v) for k, v in self.lengths.items()},
            "meta": dict(self.meta),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(_safe_json(payload), encoding="utf-8")


def file_fingerprint(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    stat = p.stat()
    return {"path": str(p), "mtime": int(stat.st_mtime), "size": int(stat.st_size)}


def fingerprint_from_paths(paths: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        if p:
            out.append(file_fingerprint(p))
    return out
