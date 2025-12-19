#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hypothesis pool persistence for Stage-B reflection gating."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..types import HypothesisCandidate
from ..utils.chinese import normalize_spaces, to_simplified

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[，,。.!！？;；:：\-—_()（）\[\]{}<>《》“”\"'·~]", re.UNICODE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_signature(text: str) -> str:
    simplified = to_simplified(text or "")
    simplified = normalize_spaces(simplified).lower()
    simplified = simplified.replace("如果", "若").replace("当", "若").replace("就", "则")
    simplified = _PUNCT_RE.sub(" ", simplified)
    simplified = normalize_spaces(simplified)
    return simplified.strip()


@dataclass
class HypothesisRecord:
    signature: str
    text: str
    falsifier: Optional[str]
    dimension: Optional[str]
    status: str
    support_cycles: int
    support_cycle_ids: Tuple[int, ...]
    support_ticket_keys: Tuple[str, ...]
    first_seen: str
    last_seen: str
    promoted_at: Optional[str] = None
    rejected_at: Optional[str] = None

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "signature": self.signature,
            "text": self.text,
            "falsifier": self.falsifier,
            "dimension": self.dimension,
            "status": self.status,
            "support_cycles": self.support_cycles,
            "support_cycle_ids": list(self.support_cycle_ids),
            "support_ticket_keys": list(self.support_ticket_keys),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }
        if self.promoted_at:
            payload["promoted_at"] = self.promoted_at
        if self.rejected_at:
            payload["rejected_at"] = self.rejected_at
        return payload

    @staticmethod
    def from_payload(payload: Mapping[str, object]) -> "HypothesisRecord":
        signature = str(payload.get("signature") or "").strip()
        text = str(payload.get("text") or "").strip()
        falsifier = (
            str(payload.get("falsifier") or "").strip()
            if payload.get("falsifier") is not None
            else None
        )
        dimension = (
            str(payload.get("dimension") or "").strip()
            if payload.get("dimension") is not None
            else None
        )
        status = str(payload.get("status") or "candidate").strip()
        support_cycles = int(payload.get("support_cycles") or 0)

        cycle_ids_raw = payload.get("support_cycle_ids") or []
        cycle_ids: Tuple[int, ...]
        if isinstance(cycle_ids_raw, Sequence) and not isinstance(
            cycle_ids_raw, (str, bytes)
        ):
            cycle_ids = tuple(int(item) for item in cycle_ids_raw)
        else:
            cycle_ids = ()

        ticket_keys_raw = payload.get("support_ticket_keys") or []
        ticket_keys: Tuple[str, ...]
        if isinstance(ticket_keys_raw, Sequence) and not isinstance(
            ticket_keys_raw, (str, bytes)
        ):
            ticket_keys = tuple(str(item) for item in ticket_keys_raw if str(item).strip())
        else:
            ticket_keys = ()

        first_seen = str(payload.get("first_seen") or "")
        last_seen = str(payload.get("last_seen") or "")
        promoted_at = (
            str(payload.get("promoted_at") or "").strip()
            if payload.get("promoted_at") is not None
            else None
        )
        rejected_at = (
            str(payload.get("rejected_at") or "").strip()
            if payload.get("rejected_at") is not None
            else None
        )
        return HypothesisRecord(
            signature=signature,
            text=text,
            falsifier=falsifier,
            dimension=dimension,
            status=status,
            support_cycles=support_cycles,
            support_cycle_ids=cycle_ids,
            support_ticket_keys=ticket_keys,
            first_seen=first_seen,
            last_seen=last_seen,
            promoted_at=promoted_at,
            rejected_at=rejected_at,
        )


class HypothesisPool:
    """Mission-scoped hypothesis pool with deterministic promotion gating."""

    def __init__(
        self,
        pool_path: Path,
        events_path: Path,
        *,
        min_support_cycles: int = 2,
        min_unique_ticket_keys: int = 6,
    ) -> None:
        if min_support_cycles <= 0:
            raise ValueError("min_support_cycles must be > 0")
        if min_unique_ticket_keys <= 0:
            raise ValueError("min_unique_ticket_keys must be > 0")
        self.pool_path = pool_path
        self.events_path = events_path
        self.min_support_cycles = min_support_cycles
        self.min_unique_ticket_keys = min_unique_ticket_keys
        self._cache: Optional[Dict[str, HypothesisRecord]] = None

    def load(self) -> Dict[str, HypothesisRecord]:
        if self._cache is not None:
            return self._cache
        if not self.pool_path.exists():
            self._cache = {}
            return {}
        with self.pool_path.open("r", encoding="utf-8") as fh:
            raw_payload = json.load(fh) or {}
        if not isinstance(raw_payload, Mapping):
            raise ValueError("hypotheses.json must be a mapping of signature -> record")
        parsed: Dict[str, HypothesisRecord] = {}
        for key, value in raw_payload.items():
            if not isinstance(value, Mapping):
                continue
            record = HypothesisRecord.from_payload(value)
            signature = record.signature or str(key)
            if signature:
                parsed[signature] = record
        self._cache = parsed
        return parsed

    def invalidate(self) -> None:
        self._cache = None

    def _write(self, payload: Mapping[str, HypothesisRecord]) -> None:
        serializable = {
            signature: record.to_payload() for signature, record in payload.items()
        }
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        with self.pool_path.open("w", encoding="utf-8") as fh:
            json.dump(serializable, fh, ensure_ascii=False, indent=2)
        self._cache = dict(payload)

    def _append_event(self, payload: Mapping[str, object]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")

    def _eligible_for_promotion(self, record: HypothesisRecord) -> bool:
        if record.status != "candidate":
            return False
        unique_keys = len(record.support_ticket_keys)
        return (
            record.support_cycles >= self.min_support_cycles
            and unique_keys >= self.min_unique_ticket_keys
        )

    def build_current_evidence_map(
        self, hypotheses: Sequence[HypothesisCandidate]
    ) -> Dict[str, Tuple[str, ...]]:
        evidence_map: Dict[str, List[str]] = {}
        for hyp in hypotheses:
            signature = _normalize_signature(hyp.text)
            if not signature:
                continue
            bucket = evidence_map.setdefault(signature, [])
            for key in hyp.evidence:
                if key:
                    bucket.append(key)
        return {
            signature: tuple(dict.fromkeys(keys))
            for signature, keys in evidence_map.items()
        }

    def record_proposals(
        self,
        hypotheses: Sequence[HypothesisCandidate],
        *,
        reflection_cycle: int,
        epoch: int,
        allow_promote: bool = True,
    ) -> Tuple[HypothesisRecord, ...]:
        pool = dict(self.load())
        now = _now().isoformat()
        eligible: Dict[str, HypothesisRecord] = {}

        for hyp in hypotheses:
            signature = _normalize_signature(hyp.text)
            if not signature:
                continue

            record = pool.get(signature)
            if record is None:
                record = HypothesisRecord(
                    signature=signature,
                    text=hyp.text.strip(),
                    falsifier=hyp.falsifier.strip() if hyp.falsifier else None,
                    dimension=hyp.dimension.strip() if hyp.dimension else None,
                    status="candidate",
                    support_cycles=0,
                    support_cycle_ids=(),
                    support_ticket_keys=(),
                    first_seen=now,
                    last_seen=now,
                    promoted_at=None,
                    rejected_at=None,
                )

            support_cycles = set(record.support_cycle_ids)
            support_keys = set(record.support_ticket_keys)
            if reflection_cycle not in support_cycles:
                support_cycles.add(reflection_cycle)
            for key in hyp.evidence:
                if key:
                    support_keys.add(key)

            record.support_cycle_ids = tuple(sorted(support_cycles))
            record.support_cycles = len(record.support_cycle_ids)
            record.support_ticket_keys = tuple(sorted(support_keys))
            record.last_seen = now

            pool[signature] = record

            self._append_event(
                {
                    "timestamp": now,
                    "event": "proposed",
                    "signature": signature,
                    "text": hyp.text.strip(),
                    "falsifier": hyp.falsifier.strip() if hyp.falsifier else None,
                    "dimension": hyp.dimension.strip() if hyp.dimension else None,
                    "ticket_keys": list(hyp.evidence),
                    "reflection_cycle": reflection_cycle,
                    "epoch": epoch,
                }
            )

            if allow_promote and self._eligible_for_promotion(record):
                eligible[signature] = record

        self._write(pool)

        return tuple(eligible[signature] for signature in sorted(eligible.keys()))

    def mark_promoted(
        self,
        signatures: Iterable[str],
        *,
        reflection_cycle: int,
        epoch: int,
    ) -> Tuple[HypothesisRecord, ...]:
        pool = dict(self.load())
        now = _now().isoformat()
        promoted: List[HypothesisRecord] = []
        for signature in signatures:
            record = pool.get(signature)
            if record is None:
                continue
            if record.status == "promoted":
                continue
            record.status = "promoted"
            record.promoted_at = now
            record.last_seen = now
            pool[signature] = record
            promoted.append(record)
            self._append_event(
                {
                    "timestamp": now,
                    "event": "promoted",
                    "signature": signature,
                    "text": record.text,
                    "falsifier": record.falsifier,
                    "dimension": record.dimension,
                    "support_cycles": record.support_cycles,
                    "unique_ticket_keys": len(record.support_ticket_keys),
                    "ticket_keys": list(record.support_ticket_keys),
                    "reflection_cycle": reflection_cycle,
                    "epoch": epoch,
                }
            )
        if promoted:
            self._write(pool)
        return tuple(promoted)

    def mark_rejected(
        self,
        signatures: Iterable[str],
        *,
        reflection_cycle: int,
        epoch: int,
        reason: Optional[str] = None,
    ) -> Tuple[HypothesisRecord, ...]:
        pool = dict(self.load())
        now = _now().isoformat()
        rejected: List[HypothesisRecord] = []
        reason_text = reason.strip() if isinstance(reason, str) and reason.strip() else None
        for signature in signatures:
            record = pool.get(signature)
            if record is None:
                continue
            if record.status in {"promoted", "rejected"}:
                continue
            record.status = "rejected"
            record.rejected_at = now
            record.last_seen = now
            pool[signature] = record
            rejected.append(record)
            payload = {
                "timestamp": now,
                "event": "rejected",
                "signature": signature,
                "text": record.text,
                "falsifier": record.falsifier,
                "dimension": record.dimension,
                "support_cycles": record.support_cycles,
                "unique_ticket_keys": len(record.support_ticket_keys),
                "ticket_keys": list(record.support_ticket_keys),
                "reflection_cycle": reflection_cycle,
                "epoch": epoch,
            }
            if reason_text:
                payload["reason"] = reason_text
            self._append_event(payload)
        if rejected:
            self._write(pool)
        return tuple(rejected)
