#!/usr/bin/env python3
"""Analyze ms-swift / TRL GRPO `logging.jsonl` files.

Produces:
- CSV exports for train/eval metrics
- Plots for reward trajectories, reward components, KL/loss/LR, and length stats

Usage:
  conda run -n ms python analysis/grpo_analyze_logging.py \
    --log output/.../logging.jsonl --out analysis/grpo_reward_collapse_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_step(value: Any) -> int | None:
    """Parse `global_step/max_steps` like '100/1252' into 100."""
    if isinstance(value, str) and "/" in value:
        left = value.split("/", 1)[0].strip()
        try:
            return int(left)
        except Exception:
            return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _split_train_eval(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for r in rows:
        if any(k.startswith("eval_") for k in r.keys()):
            eval_rows.append(r)
        else:
            train_rows.append(r)
    return train_rows, eval_rows


def _strip_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _to_frame(
    rows: list[dict[str, Any]], *, prefix_to_strip: str | None
) -> pd.DataFrame:
    if prefix_to_strip:
        rows = [_strip_prefix(r, prefix_to_strip) for r in rows]
    df = pd.DataFrame(rows)
    if "global_step/max_steps" in df.columns:
        df["step"] = df["global_step/max_steps"].map(_parse_step)
    elif "step" in df.columns:
        df["step"] = df["step"].map(_parse_step)
    else:
        df["step"] = None
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)
    df = df.sort_values("step").reset_index(drop=True)
    return df


def _plot_reward_trajectory(
    train: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    _ = ax.plot(train["step"], train["reward"], label="train/reward", linewidth=2)
    if not eval_df.empty:
        _ = ax.plot(
            eval_df["step"],
            eval_df["reward"],
            label="eval/reward",
            linewidth=2,
            marker="o",
        )
    _ = ax.set_title("Reward trajectory")
    _ = ax.set_xlabel("global_step")
    _ = ax.set_ylabel("reward (group mean)")
    _ = ax.grid(True, alpha=0.3)
    _ = ax.legend()
    _ = fig.tight_layout()
    _ = fig.savefig(out_dir / "reward.png", dpi=200)
    plt.close(fig)


def _plot_components(df: pd.DataFrame, title: str, out_path: Path) -> None:
    component_cols = [
        "rewards/SummaryFormatReward/mean",
        "rewards/SummaryHeaderReward/mean",
        "rewards/SummaryParsePenalty/mean",
        "rewards/SummaryDatasetReward/mean",
        "rewards/SummaryContentF1Reward/mean",
    ]
    existing = [c for c in component_cols if c in df.columns]
    if not existing:
        return

    n = len(existing)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 3.5 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for idx, col in enumerate(existing):
        ax = axes_flat[idx]
        _ = ax.plot(df["step"], df[col], linewidth=2)
        _ = ax.set_title(col)
        _ = ax.set_xlabel("global_step")
        _ = ax.set_ylabel("mean")
        _ = ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        _ = ax.axis("off")

    _ = fig.suptitle(title, y=1.02)
    _ = fig.tight_layout()
    _ = fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_kl_loss_lr(train: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)

    # KL
    ax = axes[0]
    if "kl" in train.columns:
        _ = ax.plot(train["step"], train["kl"], label="train/kl", linewidth=2)
    if not eval_df.empty and "kl" in eval_df.columns:
        _ = ax.plot(
            eval_df["step"], eval_df["kl"], label="eval/kl", linewidth=2, marker="o"
        )
    _ = ax.set_ylabel("KL (sequence-summed)")
    _ = ax.grid(True, alpha=0.3)
    _ = ax.legend()

    # Loss
    ax = axes[1]
    if "loss" in train.columns:
        _ = ax.plot(train["step"], train["loss"], label="train/loss", linewidth=2)
    if not eval_df.empty and "loss" in eval_df.columns:
        _ = ax.plot(
            eval_df["step"], eval_df["loss"], label="eval/loss", linewidth=2, marker="o"
        )
    _ = ax.set_ylabel("loss")
    _ = ax.grid(True, alpha=0.3)
    _ = ax.legend()

    # LR (note: may reflect only one param-group)
    ax = axes[2]
    if "learning_rate" in train.columns:
        _ = ax.plot(
            train["step"],
            train["learning_rate"],
            label="train/learning_rate",
            linewidth=2,
        )
    _ = ax.set_xlabel("global_step")
    _ = ax.set_ylabel("learning_rate (logged)")
    _ = ax.grid(True, alpha=0.3)
    _ = ax.legend()

    _ = fig.suptitle("KL / loss / learning rate", y=1.02)
    _ = fig.tight_layout()
    _ = fig.savefig(out_dir / "kl_loss_lr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_lengths(train: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for prefix, df in [("train", train), ("eval", eval_df)]:
        if df.empty:
            continue
        mean_col = "completions/mean_length"
        min_col = "completions/min_length"
        max_col = "completions/max_length"
        if mean_col not in df.columns:
            continue
        _ = ax.plot(
            df["step"], df[mean_col], label=f"{prefix}/mean_length", linewidth=2
        )
        if min_col in df.columns:
            _ = ax.plot(
                df["step"],
                df[min_col],
                label=f"{prefix}/min_length",
                linewidth=1,
                alpha=0.8,
            )
        if max_col in df.columns:
            _ = ax.plot(
                df["step"],
                df[max_col],
                label=f"{prefix}/max_length",
                linewidth=1,
                alpha=0.8,
            )

    _ = ax.set_title("Completion length stats (aggregated)")
    _ = ax.set_xlabel("global_step")
    _ = ax.set_ylabel("tokens")
    _ = ax.grid(True, alpha=0.3)
    _ = ax.legend()
    _ = fig.tight_layout()
    _ = fig.savefig(out_dir / "lengths.png", dpi=200)
    plt.close(fig)


def _write_summary_md(
    train: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path
) -> None:
    lines: list[str] = []
    lines.append("# GRPO logging.jsonl summary")
    lines.append("")
    if train.empty:
        lines.append("No train rows found.")
        _ = (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        return

    def fmt(x: float) -> str:
        return f"{x:.6g}"

    reward_min_step = int(train.loc[train["reward"].idxmin(), "step"])  # type: ignore
    reward_max_step = int(train.loc[train["reward"].idxmax(), "step"])  # type: ignore
    lines.append("## Train")
    lines.append(f"- rows: {len(train)}")
    lines.append(
        f"- step range: {int(train['step'].min())} → {int(train['step'].max())}"
    )
    lines.append(
        f"- reward min/max: {fmt(float(train['reward'].min()))} @ step {reward_min_step} → {fmt(float(train['reward'].max()))} @ step {reward_max_step}"
    )
    if "kl" in train.columns:
        kl_max_step = int(train.loc[train["kl"].idxmax(), "step"])  # type: ignore
        lines.append(f"- kl max: {fmt(float(train['kl'].max()))} @ step {kl_max_step}")

    if not eval_df.empty:
        lines.append("")
        lines.append("## Eval")
        lines.append(f"- rows: {len(eval_df)}")
        eval_reward_min_step = int(eval_df.loc[eval_df["reward"].idxmin(), "step"])  # type: ignore
        eval_reward_max_step = int(eval_df.loc[eval_df["reward"].idxmax(), "step"])  # type: ignore
        lines.append(
            f"- eval_reward min/max: {fmt(float(eval_df['reward'].min()))} @ step {eval_reward_min_step} → {fmt(float(eval_df['reward'].max()))} @ step {eval_reward_max_step}"
        )
        if "kl" in eval_df.columns:
            lines.append(f"- eval_kl max: {fmt(float(eval_df['kl'].max()))}")

    _ = (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--log", type=str, required=True, help="Path to logging.jsonl")
    _ = ap.add_argument("--out", type=str, required=True, help="Output directory")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)
    _ = out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(log_path)
    train_rows, eval_rows = _split_train_eval(rows)
    train_df = _to_frame(train_rows, prefix_to_strip=None)
    eval_df = _to_frame(eval_rows, prefix_to_strip="eval_")

    # Persist CSVs for quick diffs/grep
    train_df.to_csv(out_dir / "train_metrics.csv", index=False)
    eval_df.to_csv(out_dir / "eval_metrics.csv", index=False)

    _plot_reward_trajectory(train_df, eval_df, out_dir)
    _plot_components(
        train_df,
        "Train reward components (mean)",
        out_dir / "reward_components_train.png",
    )
    if not eval_df.empty:
        _plot_components(
            eval_df,
            "Eval reward components (mean)",
            out_dir / "reward_components_eval.png",
        )
    _plot_kl_loss_lr(train_df, eval_df, out_dir)
    _plot_lengths(train_df, eval_df, out_dir)
    _write_summary_md(train_df, eval_df, out_dir)


if __name__ == "__main__":
    main()
