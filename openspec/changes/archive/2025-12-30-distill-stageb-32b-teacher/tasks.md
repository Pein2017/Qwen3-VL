# Tasks: distill-stageb-32b-teacher

1. **Spec & Design**
   - [ ] Align with `specs/stage-b-training-free` and `specs/sft-training` on terminology（Stage‑A 摘要、guidance、Verdict/Reason、chatml 模板）。
   - [ ] Finalize this OpenSpec change and validate via `openspec validate distill-stageb-32b-teacher --strict`.

2. **Stage‑B Distillation Mode**
   - [x] Add a `stage_b_distillation` config block to Stage‑B YAML（e.g., `configs/stage_b/bbu_line.yaml`），默认 **enabled**，支持输出路径配置。
   - [x] Extend `src/stage_b/runner.py` / sampling pipeline so that, when distill mode is enabled, it assembles chatml‑style `messages`（system/user/assistant）基于当前 mission/guidance/Stage‑A 摘要与 **选中 Verdict/Reason**。
   - [x] Persist **仅收敛 epoch（guidance 当轮无更新）中每个 group 的选中对话** 到 `{output.root}/{run_name}/{mission}/distill_chatml.jsonl`，仅含 `group_id`, `mission`, `label`, `messages`，并在重跑时覆盖旧文件。
   - [x] Ensure distill logging 不改变现有 `trajectories.jsonl` / `selections.jsonl` / `guidance.json` 行为；distill 日志不包含 reflection 文本。

3. **Teacher Integration（Qwen3‑32B）**
   - [x] Provide an example Stage‑B config that uses `model_cache/models/Qwen/Qwen3-32` 作为 `model_name_or_path`，仅用于离线 distill run。
   - [ ] Verify Stage‑B runner 在该 Teacher checkpoint 下可以正常读取 Stage‑A 摘要并输出 Verdict/Reason（格式通过现有 parsers）。

4. **Distillation Corpus Preparation**
   - [x] Implement a small script/notebook 将 distill_chatml.jsonl 切分为 train/val（例如 `data/stage_b/distill_chatml.train.jsonl`），并可选过滤明显与历史 label 冲突的样本。
   - [ ] Document负样本优先级策略（例如 label=fail & verdict=fail 样本可标记为高价值，用于后续配比实验）。

5. **Fusion & SFT Wiring**
   - [x] Add a new fusion source（例如 `stageb_teacher`）到合适的 fusion 配置中，使用 `dataset: chat`, `template: chatml`, `mode: dense`，指向 distill_chatml 语料。
   - [ ] Update一个 summary/fusion SFT YAML，使其在 dense+summary+chat 的基础上引入 stageb_teacher 任务，并记录推荐 ratio 初始值。

6. **Validation & Docs**
   - [ ] Run a small‑scale distill SFT 实验（tiny JSONL + few epochs），对比 Stage‑B 上 label_match 与误放行率（尤其 label=fail 时 `verdict=pass` 比例）在 dense+summary vs dense+summary+distill 两个模型下的变化。
   - [ ] Update `docs/runtime/STAGE_A_STAGE_B.md` 和 `docs/stage-B-knowledge-Chinese.md`，描述 Teacher 蒸馏流程与统一模型的角色分工。
   - [ ] Optionally，在 `docs/experiments/` 记录一次蒸馏实验结果与推荐超参。
