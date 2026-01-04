# Change Proposal — add-stageb-guidance-scaffold-hypothesis-gating

## Why

Stage‑B 的 reflection 当前会直接把“本批次”归纳到的规则写入 `guidance.json`（G1+），这在多 mission、复杂噪声与多温度 rollout 场景下容易出现两个系统性问题：

1) **结构性不变量被误改/丢失**  
   例如“全局/局部取舍”“证据覆盖（缺少局部图即不通过）”“多主体取主证”等属于 mission‑wise 的结构脚手架。一旦它们被写在可变的 G 规则里，就会被后续反思改写，从而导致推理框架漂移与 FN 激增。

2) **噪声规则进入 guidance，造成规则膨胀与错误学习**  
   例如“复核/不应直接/佐证”类第三状态变体、以及偶然 batch 中共现的噪声维度（如品牌/局部 OCR 误识别），容易被反思当作“可泛化规则”写入 guidance。即使后续可被 lifecycle 清理，仍会在短期内显著伤害指标。

本变更引入两件事：
- **mission‑wise 的不可变结构层（scaffold）**：从 seed `initial_guidance` 注入，并且永远不允许被 reflection 修改；
- **极简 hypothesis 门控（delayed promotion）**：reflection 先提出“可证伪 hypothesis”，系统跨批次累计支持证据，满足门槛后才晋升为 G 规则写入 guidance。

## What

1) **Guidance 分层（scaffold vs mutable guidance）**
   - `initial_guidance` 的每个 mission 在 `experiences` 内新增只读 scaffold keys：`S1..Sn`（mission‑wise）。
   - `S*` 的内容用于表达结构性不变量（证据覆盖、全局优先、多主体取主证、禁止第三状态等），并且在 Stage‑B prompt 中始终可见。
   - reflection 允许操作 `G0+`（允许更新 mission checklist），但 `S*` MUST be read-only and MUST NOT be targeted by ops.
   - `G0` MUST remain present (non-empty) and MUST NOT be removed.

2) **HypothesisPool（候选规则池） + 简化门控**
   - ops pass 不再“默认直接写入 guidance”，而是输出 `hypotheses[]`（可证伪、可泛化、二值化）。
   - 系统将 `hypotheses[]` 以 mission‑scoped 的形式落盘，并跨批次累计支持证据。
   - 达到门槛（如“至少 2 个不同反思批次 + 至少 K 个不同 group_id 支持”）后，才将 hypothesis 晋升为 `G*` 经验写入 `guidance.json`。

3) **把第三状态彻底挡在 guidance 外**
   - `hypotheses[*].text` 与任何可晋升的规则文本 MUST 禁止出现 “复核/不应直接/佐证/证据不足/待定”等第三状态表达（含常见变体）。
   - 如果某个样本只能导出第三状态结论，应由 decision pass 进入 stop‑gradient 队列，而非进入 hypothesis 或 guidance。

## Impact

- **准确率与稳定性**：结构脚手架不再漂移；噪声规则需要跨批次重复支持才能入库，降低 epoch2/3 那类“规则污染导致 FN 暴涨”的风险。
- **学习速度**：会更慢一些（延迟晋升），但更稳、更可控；适合多 mission 可扩展。
- **可观测性**：新增 hypothesis 池与 promotion 事件的可审计产物，便于定位“为什么某条规则入库/为何没入库”。

## Non-Goals

- 不引入规则可执行 DSL/自动判定引擎；hypothesis 的有效性仍通过跨批次 evidence 累计与后续 hit/miss lifecycle 进行淘汰。
- 不在本变更中重构 Stage‑A 摘要质量或引入新的图像侧特征。

## Success Criteria

- mission‑wise scaffold (`S*`) 在整个 run 生命周期内保持不变，且不依赖任何可变 G 规则的关键词触发。
- guidance 规则数量增速显著下降，且第三状态表达不进入 `guidance.json`。
- hypothesis 晋升可解释：每条晋升规则能追溯到多个批次的 `group_id` 支持证据。
