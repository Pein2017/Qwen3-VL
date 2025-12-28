from __future__ import annotations

import textwrap

import pytest

from src.stage_b.config import load_stage_b_config


def _base_rule_search_yaml(*, extra_rule_search: str = "") -> str:
    content = f"""
    seed: 7
    stage_a_paths:
      - /tmp/stage_a.jsonl
    guidance:
      path: /tmp/guidance.json
      retention: 3
    output:
      root: /tmp/output
      run_name: test-run
    reflection:
      decision_prompt_path: /tmp/decision.txt
      ops_prompt_path: /tmp/ops.txt
      batch_size: 1
    model:
      model_name_or_path: /tmp/model
      torch_dtype: auto
      device_map: auto
    runner:
      epochs: 1
      per_rank_rollout_batch_size: 1
    rule_search:
      proposer_prompt_path: /tmp/proposer.txt
      train_sampler:
        grid:
          - temperature: 0.7
            top_p: 0.9
            max_new_tokens: 64
        samples_per_decode: 2
      eval_sampler:
        grid:
          - temperature: 0.1
            top_p: 0.8
            max_new_tokens: 32
        samples_per_decode: 1
    {extra_rule_search}
    """
    return textwrap.dedent(content).strip()


def test_rule_search_rejects_legacy_validate_keys(tmp_path) -> None:
    content = _base_rule_search_yaml(
        extra_rule_search="""
      validate_size: 64
    """,
    )
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="legacy keys"):
        load_stage_b_config(config_path)


def test_rule_search_requires_train_sampler(tmp_path) -> None:
    content = textwrap.dedent(
        """
        seed: 7
        stage_a_paths:
          - /tmp/stage_a.jsonl
        guidance:
          path: /tmp/guidance.json
          retention: 3
        output:
          root: /tmp/output
          run_name: test-run
        reflection:
          decision_prompt_path: /tmp/decision.txt
          ops_prompt_path: /tmp/ops.txt
          batch_size: 1
        model:
          model_name_or_path: /tmp/model
          torch_dtype: auto
          device_map: auto
        runner:
          epochs: 1
          per_rank_rollout_batch_size: 1
        rule_search:
          proposer_prompt_path: /tmp/proposer.txt
          eval_sampler:
            grid:
              - temperature: 0.1
                top_p: 0.8
                max_new_tokens: 32
            samples_per_decode: 1
        """
    ).strip()
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(content, encoding="utf-8")
    with pytest.raises(KeyError, match="train_sampler"):
        load_stage_b_config(config_path)
