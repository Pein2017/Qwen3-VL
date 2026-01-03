# Proposal: 利用 Qwen3‑32B 生成 Stage‑B 伪标注并蒸馏回统一模型

## 背景

- 项目当前采用两阶段质检流程：
  - **Stage‑A**：dense caption + summary，用 Qwen3‑VL‑4B 模型输出每图事实摘要（无坐标）。
  - **Stage‑B**：基于 Stage‑A 摘要 + mission guidance 的 training‑free 推理，输出组级 Verdict/Reason，并通过 reflection 迭代 guidance。
- 工程约束：生产环境只能常驻 **一个** Qwen3‑VL‑4B 权重（无法并行加载 dense-only / summary-only / verdict-only 多模型，也不适合频繁切 LoRA）。
- 现状问题：
  - 纯 dense SFT 的 4B 模型在 Stage‑B prompt 下，对负面摘要（如“弯曲半径不合理 / 不符合要求 / 无保护措施 / 标签/无法识别 / 未按要求配备挡风板”）仍较敏感，误放行可控。
  - 引入 summary.yaml 后得到的「dense+summary 统一模型」，虽然 Stage‑A 摘要质量仍然良好，但 **同样的摘要输入下 Stage‑B 更倾向输出 `pass`，对负项不敏感**。
  - Stage‑B 本身是 prompt‑only（参见 `specs/stage-b-training-free`），没有针对 verdict 行为的 SFT，因此「summary 任务重写了顶层语义」，而 verdict 模式没有被显式保护。

## 目标

在保持单一 4B 模型部署的前提下：

1. 复用现有 Stage‑B runner 与 artifacts 结构，引入一个 **Teacher 模式**：
  - 使用本地更强的 **Qwen3‑32B** 文本模型作为 Stage‑B Teacher；
  - 在基本沿用 `configs/stage_b/bbu_line.yaml` 的前提下，仅调整 `model.model_name_or_path` 指向 Teacher，并增加少量 `stage_b_distillation` 配置，用于输出 `distill_chatml.jsonl`；
   - 读取固定的 Stage‑A 摘要 + guidance + 标签，在 **已收敛的 prompt-space（guidance 不再变化）** 下离线生成高质量 Verdict/Reason（以及可选反思）。
2. 将上述 Teacher 轨迹落盘为 **chatml 对话格式** 的 JSONL 语料（基于收敛后的 guidance 快照）：
   - 输入端完整记录 system + user（含 mission、guidance、Stage‑A 摘要）；
   - 输出端记录 assistant 的 Verdict/Reason（以及可选 reflection 文本）；
   - 语料可直接作为 `dataset: chat`, `template: chatml` 的 SFT 数据源，无需再转换。
3. 在原有 dense+summary SFT 基础上追加一个 **Stage‑B verdict 多任务 SFT 头**：
   - 以当前的 summary 模型检查点作为 Student 初始权重，仅在 LLM 顶层做轻量 SFT；
   - 使用 Teacher 生成的高质量 chatml 对话作为监督（类似现有 `coig_cqia` 的 `dataset: chat` 源）；
   - 恢复并强化模型在 Stage‑B 模式下对负项的敏感度，降低误放行；
   - 同时保留 dense caption 和 summary 任务的表现（通过数据配比与轻量 LoRA 控制影响范围）。

非目标：

- 不改变 Stage‑B training‑free 的主设计：生产判定仍通过 prompt + guidance 完成，不在在线循环中更新权重。
- 不引入新的外部存储或服务端组件；Teacher 32B 仅在离线蒸馏阶段使用。

## 影响面（高层）

- **Stage‑B runner**（`src/stage_b/`）：
  - 新增一个「distillation run 模式」配置（默认开启），允许：
    - 切换 `model.model_name_or_path` 为 Teacher Qwen3‑32B；
    - 在 **guidance 收敛（某轮 reflection 无更新） 的最后一轮 rollout 中**，为每个 group 的 **选中候选** 记录单轮 chatml 对话（system/user/assistant）到独立 JSONL；
    - 重跑同 mission 时覆盖写入 distill 文件。
  - 保持现有 trajectories/selections/guidance 行为不变，distill 模式只增加额外 logging（不包含 reflection 内容）。
- **SFT/Fusion 配置**（`configs/fusion/` + `configs/fused_data/`）：
  - 新增一个 text‑only `dataset: chat` 目标，指向蒸馏出的 Stage‑B JSONL；
  - 复用 `template: chatml`，与当前 coig_cqia 源保持一致。
- **文档**：
  - 更新 `docs/runtime/STAGE_A_STAGE_B.md` / `docs/reference/stage-B-knowledge-Chinese.md`，描述 Teacher 蒸馏与统一模型的关系；
  - 在 `docs/training/REFERENCE.md` 或 `docs/experiments/` 增补蒸馏 runbook。

## 风险与缓解

- Teacher 误判风险：Qwen3‑32B 仍可能在个别 ticket 上偏离人工标签。
  - 缓解：在构造 SFT 语料时加入自动过滤（如强制 label=fail 时 verdict=fail，或将冲突样本降权/标记），高风险 mission 做人工抽检。
- 多任务干扰：Stage‑B 蒸馏任务如果比重过大，可能再次影响 dense/summary 能力。
  - 缓解：通过 fusion ratio 控制 Stage‑B 蒸馏样本占比，对含明显负项的样本适度过采样而非整体放大。
- 复杂度上升：Stage‑B runner 新增 distill 模式与 chatml 轨迹，需确保不破坏现有 training‑free path。
  - 缓解：以 feature flag / config 键显式开启蒸馏模式，默认保持当前行为；增加最小化单测或验证脚本验证 schema。
