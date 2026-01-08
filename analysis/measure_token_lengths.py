from __future__ import annotations

import argparse
import copy
import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, cast


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return int(round(d0 + d1))


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return int(math.ceil(value / multiple) * multiple)


@dataclass(frozen=True)
class LengthStats:
    n: int
    min_v: int
    p50: int
    p90: int
    p95: int
    p99: int
    max_v: int
    mean: float
    stdev: float


def _summarize(values: list[int]) -> LengthStats:
    if not values:
        raise ValueError("summarize requires at least one value")
    values_sorted = sorted(values)
    mean = statistics.fmean(values_sorted)
    stdev = statistics.pstdev(values_sorted) if len(values_sorted) > 1 else 0.0
    return LengthStats(
        n=len(values_sorted),
        min_v=values_sorted[0],
        p50=_percentile(values_sorted, 50),
        p90=_percentile(values_sorted, 90),
        p95=_percentile(values_sorted, 95),
        p99=_percentile(values_sorted, 99),
        max_v=values_sorted[-1],
        mean=float(mean),
        stdev=float(stdev),
    )


def _count_image_tokens(input_ids: Iterable[int], image_token_id: int | None) -> int:
    if image_token_id is None:
        return 0
    return sum(1 for tok in input_ids if tok == image_token_id)


def _count_labeled_tokens(labels: Any) -> int:
    if labels is None:
        return 0
    # labels can be list[int] or a tensor; avoid torch dependency.
    try:
        iterator = labels.tolist()
    except Exception:
        iterator = labels
    return sum(1 for v in iterator if int(v) != -100)


def main() -> None:
    # This script only needs tokenizer/processor/template on CPU.
    # ms-swift TrainArguments initializes DeepSpeed when GPUs are visible; in a single-process
    # analysis run, exposing multiple GPUs can trigger a device_map/DeepSpeed compatibility error.
    # Force a single visible GPU unless the caller already constrained it.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif "," in cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible.split(",", 1)[0].strip()

    parser = argparse.ArgumentParser(
        description=(
            "Measure token length distribution (prompt + image tokens + assistant tokens) "
            "for a GRPO/SFT dataset after augmentation."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training YAML path, e.g. configs/train/grpo/dense_2048.yaml",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of dataset samples to measure (train split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset epoch seed (affects fusion scheduling and augmentation RNG).",
    )
    args = parser.parse_args()

    from src.config.loader import ConfigLoader
    from src.config.schema import TrainingConfig
    from src.datasets.fusion import FusionConfig
    from src.datasets.unified_fusion_dataset import FusionCaptionDataset

    from swift.llm.model.register import get_model_tokenizer  # pyright: ignore[reportUnknownVariableType]
    from swift.llm.template import get_template  # pyright: ignore[reportUnknownVariableType]
    from swift.llm.template.template_inputs import TemplateInputs
    from swift.utils.utils import remove_response  # pyright: ignore[reportUnknownVariableType]

    raw_cfg = ConfigLoader.load_yaml_with_extends(args.config)
    prompts = ConfigLoader.resolve_prompts(raw_cfg)
    training_config = TrainingConfig.from_mapping(raw_cfg, prompts)

    model_path_raw = training_config.model.get("model")
    model_path = str(model_path_raw) if model_path_raw is not None else ""
    if not model_path or model_path == "None":
        raise ValueError("model.model must be set in the resolved config")

    template_cfg = training_config.template
    template_type = str(template_cfg.get("template") or "")
    if not template_type:
        raise ValueError("template.template must be set in the resolved config")
    # ms-swift get_template() does not accept arbitrary keys like 'system'; set those after instantiation.
    template_system = template_cfg.get("system")
    template_kwargs = {
        k: v for k, v in template_cfg.items() if k not in {"template", "system"}
    }

    max_model_len = int(training_config.global_max_length or 0)
    if max_model_len <= 0:
        raise ValueError("global_max_length must be set in the resolved config")

    # Build processor/tokenizer without loading model weights.
    _model, processor = get_model_tokenizer(
        model_id_or_path=model_path,
        load_model=False,
        max_model_len=max_model_len,
    )
    template = get_template(template_type, processor, **template_kwargs)
    template.set_mode("train")
    if template_system not in (None, ""):
        try:
            setattr(template, "system", str(template_system))
        except Exception:
            pass

    custom = training_config.custom
    fusion_path = custom.fusion_config
    if not fusion_path:
        raise ValueError(
            "custom.fusion_config is required for this measurement script."
        )
    fusion_config = FusionConfig.from_file(fusion_path)

    augmenter = None
    bypass_prob = float(getattr(custom, "bypass_prob", 0.0))
    aug_cfg = custom.augmentation
    if isinstance(aug_cfg, dict) and aug_cfg.get("enabled", True):
        from src.datasets.augmentation import ops as _register_ops  # noqa: F401
        from src.datasets.augmentation.builder import build_compose_from_config

        _ = _register_ops
        augmenter = build_compose_from_config(aug_cfg)
        bypass_prob = float(aug_cfg.get("bypass_prob", bypass_prob))

    # Replicate sft.py system-prompt logic (dense + summary) so token lengths match training.
    system_prompt_dense = getattr(template, "system", None)
    try:
        if system_prompt_dense is None or bool(custom.use_summary):
            from src.config.prompts import build_dense_system_prompt

            system_prompt_dense = build_dense_system_prompt(custom.json_format)
    except Exception:
        pass

    system_prompt_summary = custom.system_prompt_summary
    if system_prompt_summary is None:
        try:
            from src.prompts.stage_a_summary import build_stage_a_system_prompt

            system_prompt_summary = build_stage_a_system_prompt()
        except Exception:
            system_prompt_summary = None

    dataset = FusionCaptionDataset(
        fusion_config=fusion_config,
        base_template=template,
        user_prompt=custom.user_prompt,
        emit_norm=custom.emit_norm,
        json_format=custom.json_format,
        assistant_prefix_format=custom.assistant_prefix_format,
        augmenter=augmenter,
        bypass_prob=bypass_prob,
        curriculum_state=None,
        use_summary=bool(custom.use_summary),
        system_prompt_dense=system_prompt_dense,
        system_prompt_summary=system_prompt_summary,
        seed=args.seed,
        sample_limit=None,
        split="train",
    )
    dataset.set_epoch(0)

    image_token_id = getattr(template, "image_token_id", None)

    prompt_lens: list[int] = []
    prompt_image_tokens: list[int] = []
    sft_total_lens: list[int] = []
    sft_assistant_lens: list[int] = []
    sft_image_tokens: list[int] = []

    dense_prompt_lens: list[int] = []
    summary_prompt_lens: list[int] = []
    dense_sft_total_lens: list[int] = []
    summary_sft_total_lens: list[int] = []

    n = min(int(args.samples), len(dataset))
    for idx in range(n):
        sample = dataset[idx]
        messages = sample.get("messages")
        if not isinstance(messages, list):
            raise TypeError("dataset sample must contain 'messages' as a list")
        messages_list = cast(list[object], messages)

        # Generation prompt: remove assistant response and re-encode exactly like GRPO trainer does.
        messages_prompt = copy.deepcopy(messages_list)
        remove_response(messages_prompt)
        prompt_inputs = TemplateInputs.from_dict({"messages": messages_prompt})
        prompt_encoded = template.encode(prompt_inputs)
        prompt_ids = prompt_encoded.get("input_ids")
        if prompt_ids is None:
            raise ValueError(
                "template.encode did not return input_ids for prompt encoding"
            )
        prompt_len = len(prompt_ids)
        prompt_lens.append(prompt_len)
        prompt_image_tokens.append(_count_image_tokens(prompt_ids, image_token_id))

        # SFT-style full sequence: dataset already encodes assistant content/labels.
        input_ids = sample.get("input_ids")
        if input_ids is None:
            raise ValueError("dataset sample missing input_ids")
        total_len = len(input_ids)
        sft_total_lens.append(total_len)
        sft_image_tokens.append(_count_image_tokens(input_ids, image_token_id))
        assistant_len = _count_labeled_tokens(sample.get("labels"))
        sft_assistant_lens.append(assistant_len)

        mode = (
            sample.get("metadata", {}).get("_fusion_mode")
            if isinstance(sample.get("metadata"), dict)
            else None
        )
        if mode == "summary":
            summary_prompt_lens.append(prompt_len)
            summary_sft_total_lens.append(total_len)
        else:
            dense_prompt_lens.append(prompt_len)
            dense_sft_total_lens.append(total_len)

    rlhf = training_config.rlhf
    max_completion_length = int(rlhf.get("max_completion_length") or 0)
    if max_completion_length <= 0:
        raise ValueError(
            "rlhf.max_completion_length must be set and > 0 in the resolved config"
        )

    prompt_stats = _summarize(prompt_lens)
    total_stats = _summarize(sft_total_lens)
    assistant_stats = _summarize(sft_assistant_lens)
    prompt_img_stats = _summarize(prompt_image_tokens)

    needed_max_len_p99 = prompt_stats.p99 + max_completion_length
    needed_max_len_max = prompt_stats.max_v + max_completion_length
    recommended = _ceil_to_multiple(needed_max_len_p99 + 128, 256)
    recommended_hard = _ceil_to_multiple(needed_max_len_max + 128, 256)

    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Fusion: {fusion_path}")
    print(f"Measured samples: {prompt_stats.n} (dataset size={len(dataset)})")
    print(f"image_token_id: {image_token_id}")
    print(f"rlhf.max_completion_length: {max_completion_length}")
    print("=" * 80)

    def _print_stats(name: str, st: LengthStats) -> None:
        print(
            f"{name}: n={st.n} min={st.min_v} p50={st.p50} p90={st.p90} "
            f"p95={st.p95} p99={st.p99} max={st.max_v} mean={st.mean:.1f} stdev={st.stdev:.1f}"
        )

    _print_stats("prompt_len (generation)", prompt_stats)
    _print_stats("prompt_image_tokens", prompt_img_stats)
    _print_stats("sft_total_len (train seq)", total_stats)
    _print_stats("sft_assistant_len (labels!=-100)", assistant_stats)
    if dense_prompt_lens:
        _print_stats("prompt_len_dense", _summarize(dense_prompt_lens))
    if summary_prompt_lens:
        _print_stats("prompt_len_summary", _summarize(summary_prompt_lens))
    if dense_sft_total_lens:
        _print_stats("sft_total_len_dense", _summarize(dense_sft_total_lens))
    if summary_sft_total_lens:
        _print_stats("sft_total_len_summary", _summarize(summary_sft_total_lens))

    print("-" * 80)
    print(
        f"Needed vLLM max_model_len (p99 prompt + max_completion): {needed_max_len_p99}"
    )
    print(
        f"Needed vLLM max_model_len (max prompt + max_completion): {needed_max_len_max}"
    )
    print(f"Recommended vLLM max_model_len (p99+headroom, ceil256): {recommended}")
    print(
        f"Conservative vLLM max_model_len (max+headroom, ceil256): {recommended_hard}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
