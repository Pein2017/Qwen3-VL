from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC

_NUMERIC_TYPES = (int, float)


def _is_sequence_of_numbers(value: SequenceABC) -> bool:
    return len(value) == 2 and all(isinstance(item, _NUMERIC_TYPES) for item in value)


@dataclass(frozen=True)
class NumericParam:
    values: Tuple[float, ...]

    @staticmethod
    def from_raw(value: Any) -> 'NumericParam':
        if isinstance(value, bool):
            raise ValueError('Boolean is not accepted for numeric parameters')
        if isinstance(value, _NUMERIC_TYPES):
            return NumericParam((float(value),))
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
            if _is_sequence_of_numbers(value):
                first, second = value
                return NumericParam((float(first), float(second)))
        raise ValueError(
            "Numeric parameters must be scalars or 2-element numeric ranges"
        )

    def interpolate(self, other: 'NumericParam', progress: float) -> 'NumericParam':
        if len(self.values) != len(other.values):
            raise ValueError("Cannot interpolate parameters with mismatched dimensions")
        clamped_progress = max(0.0, min(1.0, progress))
        return NumericParam(
            tuple(
                prev + clamped_progress * (target - prev)
                for prev, target in zip(self.values, other.values)
            )
        )

    def to_python_value(self) -> float | list[float]:
        if len(self.values) == 1:
            return float(self.values[0])
        return [float(v) for v in self.values]


@dataclass(frozen=True)
class CurriculumPhase:
    until_step: int | None
    until_percent: float | None
    bypass_prob: NumericParam | None
    op_overrides: Dict[str, Dict[str, NumericParam]]


@dataclass(frozen=True)
class _PhaseDescriptor:
    start_step: int
    end_step: int
    prev_bypass: NumericParam
    prev_ops: Dict[str, Dict[str, NumericParam]]
    target_bypass: NumericParam
    target_ops: Dict[str, Dict[str, NumericParam]]


class AugmentationCurriculumScheduler:
    def __init__(
        self,
        base_bypass: NumericParam,
        base_ops: Mapping[str, Mapping[str, NumericParam]],
        phases: Sequence[CurriculumPhase],
    ) -> None:
        if not phases:
            raise ValueError("Curriculum must contain at least one phase")
        self._base_bypass = base_bypass
        self._base_ops = base_ops
        self._raw_phases = list(phases)
        self._phases: list[_PhaseDescriptor] = []
        self._final_ops: Dict[str, Dict[str, NumericParam]] = {}
        self._final_bypass: NumericParam | None = None
        self._requires_total_steps = any(
            phase.until_percent is not None for phase in self._raw_phases
        )
        has_steps = any(phase.until_step is not None for phase in self._raw_phases)
        if self._requires_total_steps and has_steps:
            raise ValueError("Mixing until_step and until_percent is not supported")
        if not self._requires_total_steps:
            self._build_descriptors(base_bypass, base_ops, phases)

    @classmethod
    def from_config(
        cls,
        base_bypass: float,
        op_meta: Iterable[Mapping[str, Any]],
        curriculum_raw: Mapping[str, Any],
    ) -> 'AugmentationCurriculumScheduler':
        if 'phases' not in curriculum_raw:
            raise ValueError("Curriculum config requires 'phases' list")
        phases_raw = curriculum_raw['phases']
        if not isinstance(phases_raw, SequenceABC):
            raise TypeError("curriculum.phases must be a sequence")
        base_bypass_param = NumericParam.from_raw(base_bypass)
        base_ops = _build_base_ops(op_meta)
        phases: list[CurriculumPhase] = []
        prev_boundary = 0.0
        for idx, raw_phase in enumerate(phases_raw):
            if not isinstance(raw_phase, MappingABC):
                raise TypeError(f"phase[{idx}] must be a mapping")
            until_step_raw = raw_phase.get('until_step')
            until_percent_raw = raw_phase.get('until_percent')
            if until_step_raw is None and until_percent_raw is None:
                raise ValueError(
                    f"phase[{idx}] requires 'until_step' or 'until_percent'"
                )
            until_step: int | None = None
            until_percent: float | None = None
            if until_percent_raw is not None:
                try:
                    up = float(until_percent_raw)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"phase[{idx}].until_percent must be a number in (0, 100]"
                    )
                if up <= 0 or up > 100:
                    raise ValueError(
                        f"phase[{idx}].until_percent must be in (0, 100], got {up}"
                    )
                until_percent = up / 100.0 if up > 1 else up
                if until_percent <= prev_boundary:
                    raise ValueError(
                        f"phase[{idx}] until_percent must be > previous ({prev_boundary})"
                    )
                prev_boundary = until_percent
            else:
                try:
                    until_step = int(until_step_raw)
                except (TypeError, ValueError):
                    raise ValueError(f"phase[{idx}].until_step must be an integer")
                if until_step <= prev_boundary:
                    raise ValueError(
                        f"phase[{idx}] until_step must be > previous ({prev_boundary})"
                    )
                prev_boundary = float(until_step)
            bypass = raw_phase.get('bypass_prob')
            bypass_param = None
            if bypass is not None:
                bypass_param = NumericParam.from_raw(bypass)
                if not 0.0 <= bypass_param.values[0] <= 1.0:
                    raise ValueError(
                        f"phase[{idx}].bypass_prob must be between 0 and 1"
                    )
            ops_raw = raw_phase.get('ops', {})
            if not isinstance(ops_raw, MappingABC):
                raise TypeError(f"phase[{idx}].ops must be a mapping")
            op_overrides: Dict[str, Dict[str, NumericParam]] = {}
            for op_name, params in ops_raw.items():
                if op_name not in base_ops:
                    raise ValueError(
                        f"phase[{idx}] references unknown op '{op_name}'"
                    )
                if not isinstance(params, MappingABC):
                    raise TypeError(
                        f"phase[{idx}].ops['{op_name}'] must be a mapping"
                    )
                field_overrides: Dict[str, NumericParam] = {}
                for param_name, value in params.items():
                    numeric_value = NumericParam.from_raw(value)
                    base_param = base_ops[op_name].get(param_name)
                    if base_param is None:
                        raise ValueError(
                            f"phase[{idx}] override for '{op_name}.{param_name}' has no numeric base"
                        )
                    if len(base_param.values) != len(numeric_value.values):
                        raise ValueError(
                            f"phase[{idx}] override for '{op_name}.{param_name}' must match base dimension"
                        )
                    field_overrides[param_name] = numeric_value
                op_overrides[op_name] = field_overrides
            phases.append(
                CurriculumPhase(
                    until_step=until_step,
                    until_percent=until_percent,
                    bypass_prob=bypass_param,
                    op_overrides=op_overrides,
                )
            )
            prev_boundary = prev_boundary
        return cls(base_bypass_param, base_ops, phases)

    def _build_descriptors(
        self,
        base_bypass: NumericParam,
        base_ops: Mapping[str, Mapping[str, NumericParam]],
        phases: Sequence[CurriculumPhase],
    ) -> None:
        self._phases.clear()
        prev_bypass = base_bypass
        prev_ops = {name: dict(params) for name, params in base_ops.items()}
        start_step = 0
        for phase in phases:
            target_bypass = prev_bypass
            if phase.bypass_prob is not None:
                target_bypass = phase.bypass_prob
            target_ops = {name: dict(params) for name, params in prev_ops.items()}
            for op_name, overrides in phase.op_overrides.items():
                op_target = target_ops.setdefault(op_name, {})
                op_target.update(overrides)
            descriptor = _PhaseDescriptor(
                start_step=start_step,
                end_step=int(phase.until_step) if phase.until_step is not None else start_step,
                prev_bypass=prev_bypass,
                prev_ops=_deepcopy_ops(prev_ops),
                target_bypass=target_bypass,
                target_ops=_deepcopy_ops(target_ops),
            )
            self._phases.append(descriptor)
            prev_bypass = target_bypass
            prev_ops = target_ops
            start_step = phase.until_step
        self._final_ops = {name: dict(params) for name, params in prev_ops.items()}
        self._final_bypass = prev_bypass

    def get_state(self, global_step: int) -> Dict[str, Any]:
        if self._requires_total_steps and self._final_bypass is None:
            raise ValueError(
                "Curriculum with until_percent requires total_steps; call set_total_steps() first"
            )
        if global_step < 0:
            global_step = 0
        for descriptor in self._phases:
            if global_step <= descriptor.end_step:
                return self._interpolate(descriptor, global_step)
        return {
            'bypass_prob': self._final_bypass.to_python_value(),
            'ops': _numeric_ops_to_python(self._final_ops),
        }

    def _interpolate(self, descriptor: _PhaseDescriptor, step: int) -> Dict[str, Any]:
        span = descriptor.end_step - descriptor.start_step
        progress = 1.0
        if span > 0:
            progress = (step - descriptor.start_step) / span
        effective_bypass = descriptor.prev_bypass.interpolate(
            descriptor.target_bypass, progress
        )
        effective_ops: Dict[str, Dict[str, NumericParam]] = {}
        for op_name, target_params in descriptor.target_ops.items():
            prev_params = descriptor.prev_ops.get(op_name, {})
            entry: Dict[str, NumericParam] = {}
            for param_name, target_value in target_params.items():
                prev_value = prev_params.get(param_name, target_value)
                entry[param_name] = prev_value.interpolate(target_value, progress)
            effective_ops[op_name] = entry
        return {
            'bypass_prob': effective_bypass.to_python_value(),
            'ops': _numeric_ops_to_python(effective_ops),
        }

    def set_total_steps(self, total_steps: int) -> None:
        if not self._requires_total_steps:
            return
        if total_steps <= 0:
            raise ValueError("total_steps must be positive to resolve percent curriculum")
        boundaries: list[int] = []
        prev = 0
        for phase in self._raw_phases:
            if phase.until_percent is None:
                raise ValueError("percent curriculum expected until_percent")
            step_boundary = max(1, int(round(total_steps * phase.until_percent)))
            if step_boundary <= prev:
                raise ValueError(
                    "Resolved curriculum boundaries must be strictly increasing after scaling to steps"
                )
            boundaries.append(step_boundary)
            prev = step_boundary
        phases_with_steps: list[CurriculumPhase] = []
        for phase, step in zip(self._raw_phases, boundaries):
            phases_with_steps.append(
                CurriculumPhase(
                    until_step=step,
                    until_percent=None,
                    bypass_prob=phase.bypass_prob,
                    op_overrides=phase.op_overrides,
                )
            )
        self._build_descriptors(self._base_bypass, self._base_ops, phases_with_steps)


def _numeric_ops_to_python(
    ops: Mapping[str, Mapping[str, NumericParam]]
) -> Dict[str, Dict[str, float | list[float]]]:
    return {
        name: {param: numeric.to_python_value() for param, numeric in params.items()}
        for name, params in ops.items()
    }


def _deepcopy_ops(
    ops: Mapping[str, Mapping[str, NumericParam]]
) -> Dict[str, Dict[str, NumericParam]]:
    return {name: dict(params) for name, params in ops.items()}


def _build_base_ops(
    op_meta: Iterable[Mapping[str, Any]]
) -> Dict[str, Dict[str, NumericParam]]:
    base_ops: Dict[str, Dict[str, NumericParam]] = {}
    for entry in op_meta:
        if not isinstance(entry, MappingABC):
            continue
        name = entry.get('name')
        params = entry.get('params', {})
        if not name or not isinstance(params, MappingABC):
            continue
        numeric_params: Dict[str, NumericParam] = {}
        for param_name, value in params.items():
            try:
                numeric_param = NumericParam.from_raw(value)
            except ValueError:
                continue
            numeric_params[param_name] = numeric_param
        base_ops.setdefault(name, {}).update(numeric_params)
    return base_ops
