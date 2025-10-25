## ADDED Requirements

### Requirement: One group per JSONL line (text-only)
- Each sample represents a group; required fields:
  - group_id: string
  - task_type: string in {"BBU安装方式检查（正装）","BBU接地线检查","BBU线缆布放","挡风板安装检查"}
  - group_label: "通过" | "不通过"
  - stage_a_summaries: dict {图片_i: 中文单行摘要}
  - messages: list[dict] (system in Chinese; user includes task focus and Stage-A summaries as plain text)
- Stage-B does not include any image inputs.

#### Scenario: Minimal sample
- A JSONL record contains messages referencing 图片_1..图片_k summaries and has group_label=通过

### Requirement: Model output contract（两行输出）
- 第一行：严格为 “通过” 或 “不通过”
- 第二行：理由（中文自然语言；可引用 图片_i；不强制固定词表）
- 不允许多余行；允许末尾空白字符；空理由将被格式或长度奖励惩罚

#### Scenario: 有效输出
- completion:
  通过\n理由: 基于图片_1…
- 接受；标签奖励由第一行判定

#### Scenario: 无效输出（大小写或词形错误）
- completion:
  通过了\n…
- 格式奖励=0；标签奖励忽略该判定

### Requirement: 当前奖励集合（v1）
- 仅启用：标签奖励（匹配 通过/不通过 与 group_label）、格式奖励（严格两行格式）
- 一致性奖励将在后续迭代加入；数据加载器需将 stage_a_summaries 透传给奖励

#### Scenario: 奖励输入透传
- 奖励函数可读取 group_label、stage_a_summaries、task_type 用于判定
