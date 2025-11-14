#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mission guidance management for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Union, cast

from ..types import (
    ExperienceMetadata,
    ExperienceOperation,
    MissionGuidance,
    ReflectionProposal,
)

logger = logging.getLogger(__name__)



class MissionGuidanceError(RuntimeError):
    """Raised when mission guidance files are malformed."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_timestamp_microsecond(dt: datetime) -> str:
    """Format timestamp with microsecond resolution for snapshot filenames."""
    return dt.strftime("%Y%m%d-%H%M%S-%f")


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise MissionGuidanceError(f"updated_at must be ISO string, got {value!r}")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise MissionGuidanceError(f"Invalid updated_at timestamp: {value!r}") from exc


def _parse_experiences_dict(
    mission: str, payload: Mapping[str, object]
) -> Dict[str, str]:
    """Parse experiences dict from payload."""

    experiences_raw = payload.get("experiences")
    if experiences_raw is None:
        raise MissionGuidanceError(f"Mission {mission} experiences must be present")
    if not isinstance(experiences_raw, Mapping):
        raise MissionGuidanceError(
            f"Mission {mission} experiences must be a mapping, got {type(experiences_raw).__name__}"
        )
    experiences = {}
    for key, value in experiences_raw.items():
        if not isinstance(value, str):
            raise MissionGuidanceError(
                f"Mission {mission} experience {key!r} must be a string"
            )
        if not value.strip():
            raise MissionGuidanceError(
                f"Mission {mission} experience {key!r} must be non-empty"
            )
        experiences[str(key)] = value.strip()
    if not experiences:
        raise MissionGuidanceError(
            f"Mission {mission} experiences dict must be non-empty"
        )
    return experiences


def _parse_metadata_dict(
    mission: str, payload: Mapping[str, object]
) -> Dict[str, ExperienceMetadata]:
    metadata_raw = payload.get("metadata")
    if metadata_raw is None:
        return {}
    if not isinstance(metadata_raw, Mapping):
        raise MissionGuidanceError(
            f"Mission {mission} metadata must be a mapping if present"
        )
    metadata: Dict[str, ExperienceMetadata] = {}
    for key, value in metadata_raw.items():
        if not isinstance(value, Mapping):
            raise MissionGuidanceError(
                f"Mission {mission} metadata for {key!r} must be a mapping"
            )
        try:
            metadata[str(key)] = ExperienceMetadata.from_payload(value)
        except ValueError as exc:
            raise MissionGuidanceError(
                f"Mission {mission} metadata for {key!r} is invalid: {exc}"
            ) from exc
    return metadata


def _parse_mission_section(
    mission: str, payload: Mapping[str, object]
) -> MissionGuidance:
    focus_value = payload.get("focus")
    focus = (
        focus_value.strip()
        if isinstance(focus_value, str) and focus_value.strip()
        else None
    )

    # Current schema only: require explicit 'step' and 'experiences'
    step_raw = payload.get("step")
    if isinstance(step_raw, bool) or step_raw is None:
        raise MissionGuidanceError(f"Mission {mission} step must be an integer")
    try:
        step = int(cast(Union[int, str, float], step_raw))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise MissionGuidanceError(
            f"Mission {mission} step must be an integer, got {step_raw!r}"
        ) from exc

    updated_at = _parse_datetime(payload.get("updated_at"))

    # Require 'experiences' mapping; no legacy 'guidance' fallback
    experiences = _parse_experiences_dict(mission, payload)

    metadata = _parse_metadata_dict(mission, payload)

    return MissionGuidance(
        mission=mission,
        focus=focus,
        experiences=experiences,
        step=step,
        updated_at=updated_at,
        metadata=metadata,
    )


class GuidanceRepository:
    """Persistence helper for mission guidance files and snapshots."""

    def __init__(
        self,
        path: str | Path,
        *,
        retention: int,
    ) -> None:
        if retention <= 0:
            raise ValueError("retention must be > 0")

        # Use object.__setattr__ to be compatible with frozen dataclass subclasses in tests
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(self, "retention", retention)
        object.__setattr__(self, "_cache", None)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def ensure_initialized(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump({}, fh, ensure_ascii=False, indent=2)
        self._cache = {}


    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def load(self) -> Dict[str, MissionGuidance]:
        if self._cache is not None:
            return self._cache

        if not self.path.exists():
            self.ensure_initialized()
            return {}

        with self.path.open("r", encoding="utf-8") as fh:
            raw_payload = json.load(fh) or {}

        if not isinstance(raw_payload, MutableMapping):
            raise MissionGuidanceError("Guidance file must be a mapping")

        parsed = {
            str(mission): _parse_mission_section(str(mission), section)
            for mission, section in raw_payload.items()
            if isinstance(section, Mapping)
        }
        self._cache = parsed
        return parsed

    def get(self, mission: str) -> MissionGuidance:
        guidance_map = self.load()
        if mission not in guidance_map:
            raise MissionGuidanceError(f"Mission {mission} guidance not found")
        return guidance_map[mission]

    def invalidate(self) -> None:
        self._cache = None

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def update_mission(
        self,
        mission: str,
        *,
        focus: Optional[str] = None,
        experiences: Optional[Dict[str, str]] = None,
    ) -> MissionGuidance:
        guidance_map = self.load()
        if mission not in guidance_map:
            raise MissionGuidanceError(f"Mission {mission} guidance not found")

        current = guidance_map[mission]
        new_focus = (
            focus.strip() if isinstance(focus, str) and focus.strip() else current.focus
        )
        new_experiences = (
            experiences if experiences is not None else current.experiences
        )
        if not new_experiences:
            raise MissionGuidanceError(
                f"Mission {mission} experiences dict must be non-empty"
            )

        updated = MissionGuidance(
            mission=current.mission,
            focus=new_focus,
            experiences=new_experiences,
            step=current.step + 1,
            updated_at=_now(),
            metadata=dict(current.metadata),
        )

        guidance_map = dict(guidance_map)
        guidance_map[mission] = updated
        self._write(guidance_map)
        return updated

    @staticmethod
    def _next_experience_key(experiences: Mapping[str, str]) -> str:
        used = set(experiences.keys())
        index = 0
        while True:
            candidate = f"G{index}"
            if candidate not in used:
                return candidate
            index += 1

    def _build_updated_guidance(
        self,
        current: MissionGuidance,
        *,
        reflection_id: str,
        source_group_ids: Sequence[str],
        operations: Sequence[ExperienceOperation],
    ) -> MissionGuidance:
        experiences = dict(current.experiences)
        metadata = dict(current.metadata)
        now = _now()

        applied_any = False
        source_fallback = tuple(str(gid) for gid in source_group_ids)

        for op in operations:
            normalized_op = op.op
            if normalized_op not in {"upsert", "remove", "merge"}:
                raise MissionGuidanceError(
                    f"Unsupported experience operation '{normalized_op}'"
                )

            key = (op.key or "").strip() or None

            if normalized_op == "remove":
                if key is None:
                    continue
                if key in experiences:
                    del experiences[key]
                    metadata.pop(key, None)
                    applied_any = True
                else:
                    logger.warning(
                        f"remove operation skipped for missing key '{key}' in mission {current.mission}"
                    )
                continue

            text = (op.text or "").strip()
            if not text:
                continue

            if key is None:
                key = self._next_experience_key(experiences)
            else:
                key = str(key)

            experiences[key] = text

            # Build combined sources (proposal evidence + fallback group ids)
            combined_sources = []
            seen_sources = set()
            for source in list(op.evidence or ()) + list(source_fallback):
                source_str = str(source)
                if source_str and source_str not in seen_sources:
                    combined_sources.append(source_str)
                    seen_sources.add(source_str)

            rationale = (op.rationale or "").strip() or None

            # If this is a merge, remove merged_from keys without reindexing
            if normalized_op == "merge" and op.merged_from:
                for mkey in op.merged_from:
                    mkey_str = (mkey or "").strip()
                    if not mkey_str or mkey_str == key:
                        if mkey_str == key:
                            logger.warning(
                                f"merge operation includes target key '{key}' in merged_from for mission {current.mission}; skipping"
                            )
                        continue
                    if mkey_str in experiences:
                        try:
                            del experiences[mkey_str]
                        except KeyError:  # pragma: no cover - defensive
                            pass
                        metadata.pop(mkey_str, None)
                    else:
                        logger.warning(
                            f"merge operation skipped missing source key '{mkey_str}' in mission {current.mission}"
                        )

            metadata[key] = ExperienceMetadata(
                updated_at=now,
                reflection_id=reflection_id,
                sources=tuple(combined_sources),
                rationale=rationale,
            )
            applied_any = True

        if not applied_any:
            raise MissionGuidanceError("Reflection refine proposal did not modify guidance")

        # Enforce non-empty experiences dict
        if not experiences:
            raise MissionGuidanceError(
                f"Mission {current.mission} experiences dict must be non-empty after operations"
            )

        return MissionGuidance(
            mission=current.mission,
            focus=current.focus,
            experiences=experiences,
            step=current.step + 1,
            updated_at=now,
            metadata=metadata,
        )

    def _resolve_operations(
        self,
        *,
        mission: str,
        proposal: ReflectionProposal,
        reflection_id: str,
        source_group_ids: Sequence[str],
        operations: Optional[Sequence[ExperienceOperation]],
        parsed_experiences: Optional[Dict[str, str]],
    ) -> Sequence[ExperienceOperation]:
        if operations is not None:
            return operations
        if getattr(proposal, "operations", None):
            return tuple(proposal.operations)
        if parsed_experiences is not None:
            return [
                ExperienceOperation(
                    op="upsert",
                    key=key,
                    text=value,
                    rationale=None,
                    evidence=tuple(source_group_ids),
                )
                for key, value in parsed_experiences.items()
            ]
        raise MissionGuidanceError(
            f"Reflection refine proposal for mission {mission} requires operations"
        )

    def apply_reflection(
        self,
        mission: str,
        *,
        proposal: ReflectionProposal,
        reflection_id: str,
        source_group_ids: Sequence[str],
        applied_epoch: Optional[int] = None,
        operations: Optional[Sequence[ExperienceOperation]] = None,
        parsed_experiences: Optional[Dict[str, str]] = None,
    ) -> MissionGuidance:
        """Apply reflection proposal by merging incremental experience edits."""

        guidance_map = self.load()
        if mission not in guidance_map:
            raise MissionGuidanceError(f"Mission {mission} guidance not found")

        current = guidance_map[mission]
        action = proposal.action

        if action == "noop":
            return current
        if action != "refine":  # pragma: no cover - defensive branch
            raise MissionGuidanceError(f"Unsupported reflection action: {action}")

        ops = self._resolve_operations(
            mission=mission,
            proposal=proposal,
            reflection_id=reflection_id,
            source_group_ids=source_group_ids,
            operations=operations,
            parsed_experiences=parsed_experiences,
        )

        updated = self._build_updated_guidance(
            current,
            reflection_id=reflection_id,
            source_group_ids=source_group_ids,
            operations=ops,
        )

        guidance_map = dict(guidance_map)
        guidance_map[mission] = updated
        self._write(guidance_map)
        return updated

    def preview_reflection(
        self,
        mission: str,
        *,
        proposal: ReflectionProposal,
        reflection_id: str,
        source_group_ids: Sequence[str],
        operations: Optional[Sequence[ExperienceOperation]] = None,
        parsed_experiences: Optional[Dict[str, str]] = None,
    ) -> MissionGuidance:
        """Build a non-persistent mission guidance preview for a reflection."""

        guidance_map = self.load()
        if mission not in guidance_map:
            raise MissionGuidanceError(f"Mission {mission} guidance not found")

        current = guidance_map[mission]
        action = proposal.action

        if action == "noop":
            return current
        if action != "refine":
            raise MissionGuidanceError(f"Unsupported reflection action: {action}")

        ops = self._resolve_operations(
            mission=mission,
            proposal=proposal,
            reflection_id=reflection_id,
            source_group_ids=source_group_ids,
            operations=operations,
            parsed_experiences=parsed_experiences,
        )

        return self._build_updated_guidance(
            current,
            reflection_id=reflection_id,
            source_group_ids=source_group_ids,
            operations=ops,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _write(self, payload: Mapping[str, MissionGuidance]) -> None:
        serializable: Dict[str, object] = {
            mission: guidance.to_payload() for mission, guidance in payload.items()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare snapshot directory
        snapshot_dir = self.path.parent / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write via temp file + rename
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            # Use builtins.open to align with tests that patch open()
            with open(temp_path, "w", encoding="utf-8") as fh:  # type: ignore[arg-type]
                json.dump(serializable, fh, ensure_ascii=False, indent=2)
            # Atomic rename to live path
            temp_path.replace(self.path)
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass
            raise

        # Create snapshot of the NEW live file (after atomic rename)
        timestamp = _format_timestamp_microsecond(_now())
        snapshot_path = snapshot_dir / f"guidance-{timestamp}.json"
        try:
            shutil.copy2(self.path, snapshot_path)
        except Exception:
            # Snapshot best-effort; do not block on failure
            pass

        # Prune old snapshots after write
        self._prune_snapshots(snapshot_dir)

        self._cache = dict(payload)

    def _prune_snapshots(self, snapshot_dir: Path) -> None:
        snapshots = sorted(snapshot_dir.glob("guidance-*.json"), reverse=True)
        for index, path in enumerate(snapshots):
            if index >= self.retention:
                try:
                    path.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass


__all__ = ["GuidanceRepository", "MissionGuidanceError"]
