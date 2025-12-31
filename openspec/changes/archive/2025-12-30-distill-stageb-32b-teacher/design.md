# Design: Stage‑B Teacher Distillation with Qwen3‑32B

## 概览

我们引入一个离线「Teacher 模式」的 Stage‑B 流水线，用 Qwen3‑32B 读取 Stage‑A 摘要与 guidance，生成 Verdict/Reason 对话，并将完整对话存为 chatml 语料；之后，通过标准 SFT/fusion 机制将这些语料蒸馏回统一的 Qwen3‑VL‑4B 模型。

核心思想：

- **Stage‑B 仍然 training‑free**：生产判定不更新权重，Teacher 只在离线蒸馏中使用。
- **Teacher→Student**：Teacher 负责在复杂任务 prompt 下给出较理性的 verdict 行为；Student（4B）学习在相同 prompt 下模仿 Teacher 的输出模式。
- **chatml 对话对齐**：蒸馏语料直接符合 `chatml` 模板，避免后处理。

## 架构组件

### 1. Distillation Run 模式（Stage‑B Runner）

在现有 `src/stage_b/runner.py` 结构上增加一个可选分支，尽量复用当前的配置（如 `configs/stage_b/bbu_line.yaml`）：

- 配置开关（示意）：
  - `stage_b_distillation.enabled: bool`（默认 false）
  - `stage_b_distillation.log_chatml_path: str`（例如 `{output.root}/{run_name}/{mission}/distill_chatml.jsonl`）
  - 可选：`stage_b_distillation.include_reflection: bool`（是否也记录 reflection 对话）。
- 当 `enabled=true` 时，Runner 在每次 rollout 完成后：
  - 组装一条 chatml 消息序列：
    - `system`: mission 说明 + guidance 文本 + 输出协议（两行 Verdict/Reason）。
    - `user`: 承载 Stage‑A per_image 摘要 + 当前组 label + 任务说明（“请根据摘要给出审核通过/不通过及理由…”）。
    - `assistant`: 当前候选的 `verdict` + `reason` 文本，合并为一个回答（保持现有两行协议即可）。
  - 将该 messages 序列以及必要 metadata 写入 distill_chatml.jsonl：
    - `{"group_id", "mission", "label", "messages": [...], "decode": {...}}`。
- 该日志独立于现有 `trajectories.jsonl` / `selections.jsonl`，不改变原有 artifacts。

模型选择与配置复用：

- Stage‑B 自身仍被视为在 **prompt-space 上“训练”规则** 的 runner：通过多 epoch + reflection 调整 guidance，而不更新模型参数。
- 当需要蒸馏时，推荐采用“两阶段”流程：
1. 使用现有 4B 模型或 Teacher 32B 在 Stage‑B 标准配置下跑若干 epoch、开启 reflection，以 label 为锚逐步收敛 guidance（prompt‑space 训练过程），直到某个 epoch 反思未对 guidance 做出任何更新（收敛）。
2. 在收敛当轮（最后一个 guidance 无更新的 epoch）直接写出 distill 语料：
   - `stage_b_distillation.enabled` 默认开启；无需额外触发脚本；
   - `model.model_name_or_path` 可指向 Teacher：`model_cache/models/Qwen/Qwen3-32`；
   - 仅记录该收敛 epoch 中 **每个 group 的选中候选**（非全部候选）的单轮 chatml 对话；
   - distill 日志路径覆盖写入（重跑同 mission 时覆盖旧文件），不包含 reflection 文本。
- 标准 production run 仍使用统一的 4B 模型和现有 Stage‑B 配置（使用相同 guidance 快照）；Teacher distill run 与生产 run 通过不同 YAML 区分，无需在代码中引入复杂模式切换。

### 2. Distillation 语料构建

在 distill run 输出的 `distill_chatml.jsonl` 基础上：

- 编写轻量脚本（或 dataset wrapper）：
  - 过滤或打标 Teacher 输出与历史 label 明显冲突的记录；
  - 可为「label=fail & verdict=fail」等高价值样本打上高权重标签（在 fusion 配置中体现）。
- 语料格式直接满足 `dataset: chat`, `template: chatml` 的要求：
  - `messages[0].role == "system"`,
  - `messages[1].role == "user"`,
  - `messages[-1].role == "assistant"`。
  - 元数据只需 `{group_id, mission, label, messages}`，无额外 decode/epoch 字段。

### 3. SFT / Fusion 集成

在 fusion 配置中新增一个 text‑only 源，例如：

```yaml
  - name: stageb_teacher
    dataset: chat
    train_jsonl: data/stage_b/distill_chatml.train.jsonl
    ratio: 0.05   # 示例，实际需实验
    template: chatml
    mode: dense
    sample_without_replacement: true
    augmentation_enabled: false
    curriculum_enabled: false
```

并在 summary/fusion SFT YAML 中开启该源，形成多任务：

- dense caption（bbu/rru/lvis）
- summary（bbu_summary）
- chat（coig_cqia 等）
- stageb_teacher（Teacher 蒸馏 verdict，对应 Stage‑B prompt 格式）

LoRA/冻结策略（Student）：

- Student 以当前 summary 模型检查点作为 base，继续冻结视觉与对齐模块，仅在 LLM 顶层做 LoRA；
- 控制 Stage‑B 蒸馏源的 `ratio`，避免其主导训练，并通过少量 epoch 作为「后训练」步骤微调，而非完整重训。

## 关键设计选择

1. **为何使用 chatml 而非自定义模板？**
   - 现有语言数据（coig_cqia）已使用 `chatml` 模板；
   - 复用模板可直接重用 `src/datasets/builders/jsonlines` 的 chat 路径与 prompts 配置；
   - 减少额外模板/解析复杂度。

2. **为何不直接用 KD（logits 蒸馏）？**
   - 当前训程与 infra 已对「SFT on chatml」路径较为成熟；
   - 直接文本蒸馏（Teacher 对话 → Student SFT）更符合现有 `sft-training` / `fusion-dataset` 规范；
   - 不需要在训练期同时加载 Teacher，训练成本更可控。

3. **Stage‑B training‑free 规范如何保持？**
   - 本次改动不改变生产 Stage‑B 判定逻辑；
   - distill 模式为离线工具，与 `specs/stage-b-training-free` 保持兼容：在线 run 依旧不更新模型参数。
