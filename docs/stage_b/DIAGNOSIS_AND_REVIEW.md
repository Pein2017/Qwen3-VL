# Stage-B 流程与现状诊断

## 目录
- [1. 整体背景](#1-整体背景)
- [2. Stage-B 核心流程](#2-stage-b-核心流程)
- [3. 最近实现：Hit/Miss 生命周期管理](#3-最近实现hitmiss-生命周期管理)
- [4. 当前问题诊断](#4-当前问题诊断)
- [5. 数据分析与根本原因](#5-数据分析与根本原因)
- [6. 需要Code Review与改进建议的地方](#6-需要code-review与改进建议的地方)

---

## 1. 整体背景

### 1.1 系统架构

这是一个**AI质检系统**，分为两个阶段：

```
Stage-A（基础识别）
  ├─ 输入：图片 + 任务描述(Focus)
  ├─ 输出：每张图的结构化摘要(Summary)
  └─ 目标：提取关键信息（有什么对象、是否完整等）

        ↓

Stage-B（分组判决）
  ├─ 输入：多张图的摘要集合 + 人工标签 + 学到的规则(Guidance)
  ├─ 处理流程：Rollout → Selection → Reflection
  └─ 输出：pass/fail 判决 + 学到的规则
```

### 1.3 生产部署约束：单模型双阶段

- 最终线上环境中，Stage‑A 摘要与 Stage‑B 判决共用同一个 Qwen3‑VL 模型（同一组权重 / LoRA 组合），通过不同 prompt 切换任务。
- 这意味着：任何针对 Stage‑A summary‑mode 的 SFT/LoRA 微调都可能影响 Stage‑B 的 verdict 习惯；训练与配置调整需要显式考虑“既能做 dense/summary 识别，又能做 group‑ticket 判决”的统一能力。
- 本文档中的诊断与改进建议（prompt 调整、LoRA 层数收缩、数据配比等）都默认遵守这一单模型约束。

### 1.2 核心价值主张

**"训练免费的端到端学习"**
- ❌ 不微调模型（昂贵、慢）
- ✅ 通过Reflection从错误中学习规则
- ✅ 规则逐步改进，指导下一轮预测

### 1.3 任务具体例子

**任务：挡风板安装检查**

```
Focus (检查重点)：
  "至少需要检测到BBU设备并根据情况判断是否需要安装挡风板。
   若需要安装，则判断是否符合要求。"

样本：
  - 图片：[Image1, Image2, Image3]
  - Stage-A摘要：
      Image1: "三个BBU设备，其中两个未配备挡风板..."
      Image2: "一个BBU设备，无需安装"
      Image3: "BBU设备按要求配备了挡风板..."
  - 人工标签：fail (因为有些BBU设备未配备)

期望：
  Rollout生成多个候选，每个包含Verdict(pass/fail) + Reason
  Selection挑选最优候选
  如果预测错误，Reflection学习规则来纠正
```

---

## 2. Stage-B 核心流程

### 2.1 三步走架构

```
┌─────────────────────────────────────────────────────┐
│ Stage-B Pipeline (纯LLM，无模型微调)                  │
└─────────────────────────────────────────────────────┘

Step 1: ROLLOUT (生成多个候选)
  ┌──────────────────────────────────┐
  │ 输入：Group Ticket               │
  │  - Focus: 检查重点               │
  │  - 摘要: Stage-A输出             │
  │  - Guidance: 当前规则库          │
  │                                  │
  │ 处理：在不同温度下采样          │
  │  - 温度 0.1: 保守，重复率高      │
  │  - 温度 0.4: 中等                │
  │  - 温度 0.7: 创意，多样性        │
  │                                  │
  │ 输出：N个候选                    │
  │  [Candidate0, Candidate1, ...]  │
  │  每个包含：                      │
  │    - verdict: pass/fail          │
  │    - reason: 详细推理            │
  │    - format_ok: 格式检查         │
  └──────────────────────────────────┘
              ↓

Step 2: SELECTION (挑选最优候选)
  ┌──────────────────────────────────┐
  │ 投票机制                         │
  │  - 统计各verdict的count          │
  │  - 计算vote_strength            │
  │  - 当vote_strength < 阈值        │
  │    → 标记为manual_review        │
  │                                  │
  │ 输出：Selection结果              │
  │  - selected_candidate_index      │
  │  - verdict: 最终判决             │
  │  - reason: 选中候选的reason      │
  │  - label_match: 是否与标签匹配   │
  └──────────────────────────────────┘
              ↓

Step 3: REFLECTION (从错误中学习)
  ┌──────────────────────────────────┐
  │ 证据判断 + 规则学习              │
  │                                  │
  │ 前置条件：收集batch_size个结果   │
  │                                  │
  │ LLM任务：                        │
  │  1. 分析候选的reason            │
  │  2. 判断是否有证据支持标签       │
  │  3. 如果有：提取可学习的规则     │
  │  4. 如果无：标记为no_evidence   │
  │                                  │
  │ 输出：Proposal                   │
  │  - action: refine / noop         │
  │  - operations: 规则操作列表      │
  │  - uncertainty_note: "no_ev..."  │
  │  - confidence: 规则置信度        │
  │                                  │
  │ 后续处理：                       │
  │  - 如果有证据 → 更新Guidance    │
  │  - 如果无证据 → 标记manual_review│
  └──────────────────────────────────┘
              ↓
          输出结果
```

### 2.2 关键数据结构

**TrajectoryWithSignals** (每个候选及其信号)
```python
@dataclass
class DeterministicSignals:
    label_match: Optional[bool]  # 该候选是否与标签匹配
    self_consistency: Optional[float]
    conflict_flag: bool  # 预测与标签冲突
    needs_manual_review: bool
    vote_strength: Optional[float]  # Selection时的投票强度
    low_agreement: bool  # 投票不一致
    ...

@dataclass
class TrajectoryWithSignals:
    parsed: ParsedCandidate  # Rollout输出
    signals: DeterministicSignals  # 信号标注
```

**ExperienceMetadata** (规则的生命周期)
```python
@dataclass
class ExperienceMetadata:
    updated_at: datetime
    reflection_id: str
    sources: Tuple[str, ...]  # 哪些样本支持该规则
    rationale: Optional[str]

    # 生命周期字段（新增）
    hit_count: int = 0  # 预测正确的次数
    miss_count: int = 0  # 预测错误的次数
    confidence: float = 1.0  # = hit / (hit + miss + eps)
```

### 2.3 关键文件

| 文件 | 功能 |
|------|------|
| `src/stage_b/runner.py` | 主流程：Rollout → Selection → Reflection循环 |
| `src/stage_b/rollout.py` | Rollout采样器 |
| `src/stage_b/reflection/engine.py` | Reflection LLM调用 + 规则提取 |
| `src/stage_b/io/guidance.py` | Guidance读写 + 规则管理 |
| `src/stage_b/config.py` | 配置Schema |
| `src/stage_b/types.py` | 数据结构定义 |

---

## 3. 最近实现：Hit/Miss 生命周期管理

### 3.1 设计目标

**问题：** Reflection学到的规则可能是短暂过拟合，需要自动清理低质量规则

**解决方案：**
- 追踪每条规则的**命中次数**(hit_count)和**失误次数**(miss_count)
- 计算置信度: `confidence = hit / (hit + miss + 1e-5)`
- 自动移除: `confidence < 0.35 AND miss_count >= 3`

### 3.2 实现流程

```
规则初始化
  ↓
hit_count=1, miss_count=0, confidence=1.0
  ↓
[每个Epoch]
  ├─ Reflection应用规则 → 记录last_applied_rule_keys
  │
  ├─ Sample预测成功 → 该规则隐含命中（计数器增加）
  │                   [当前未显式实现]
  │
  ├─ Sample预测失败 → 调用increment_miss_count()
  │                   miss_count++
  │                   confidence重新计算
  │
  └─ Epoch结束 → cleanup_low_confidence()
                  移除低置信度规则
```

### 3.3 代码实现位置

#### 3.3.1 在Runner中追踪已应用规则

**文件：** `src/stage_b/runner.py:319`

```python
last_applied_rule_keys: List[str] = []  # Track rules applied in last reflection
```

#### 3.3.2 从Reflection提取规则操作

**文件：** `src/stage_b/runner.py:570-583` （批Reflection）和 `678-691` （epoch末Reflection）

```python
# Extract and track applied rules
if outcome.applied:
    applied_keys = []
    for op in outcome.operations:
        # Track both upsert (add/update) operations
        if op.op == "upsert" and op.key:
            applied_keys.append(op.key)
        elif op.op == "upsert" and op.text:
            # For new rules without explicit key, use text as identifier
            applied_keys.append(f"_text_{hash(op.text) % (10**8)}")
    last_applied_rule_keys = applied_keys
    logger.debug(f"Reflection applied {len(applied_keys)} rule(s): {applied_keys}")
```

#### 3.3.3 当预测失败时更新miss_count

**文件：** `src/stage_b/runner.py:569-576` （批处理） 和 `669-676` （epoch末）

```python
# If winning candidate still mismatches label, enqueue manual review
if win_cand and win_cand.signals and win_cand.signals.label_match is False:
    _append_jsonl(manual_review_path, {...})
    # Update miss_count for rules that led to this failed prediction
    if last_applied_rule_keys:
        mission_guidance_repo.increment_miss_count(
            mission, last_applied_rule_keys
        )
```

#### 3.3.4 Epoch末自动清理

**文件：** `src/stage_b/runner.py:697-712`

```python
# Epoch-end cleanup: remove low-confidence rules
if config.guidance_lifecycle and config.guidance_lifecycle.enable_auto_cleanup:
    try:
        removed = mission_guidance_repo.cleanup_low_confidence(
            mission,
            confidence_threshold=config.guidance_lifecycle.confidence_drop_threshold,
            min_miss_before_drop=config.guidance_lifecycle.min_miss_before_drop,
        )
        if removed:
            logger.info(f"Epoch {epoch}: cleanup_low_confidence removed {len(removed)} rule(s): {removed}")
    except Exception as exc:
        logger.warning(f"Failed to perform cleanup at epoch {epoch}: {exc}")
```

#### 3.3.5 GuidanceRepository中的实现

**文件：** `src/stage_b/io/guidance.py`

```python
def increment_miss_count(
    self,
    mission: str,
    experience_keys: Sequence[str],
) -> None:
    """Increment miss count for experiences that led to incorrect predictions."""
    # 1. 加载当前guidance
    # 2. 对每个key，更新metadata
    # 3. 重新计算confidence
    # 4. 保存回文件

def cleanup_low_confidence(
    self,
    mission: str,
    *,
    confidence_threshold: float = 0.35,
    min_miss_before_drop: int = 3,
) -> List[str]:
    """Remove low-confidence experiences from guidance."""
    # 1. 遍历所有规则的metadata
    # 2. 找出 confidence < 阈值 AND miss_count >= 最小值 的规则
    # 3. 删除这些规则
    # 4. 保存回文件
    # 5. 返回删除的规则列表
```

### 3.4 配置

**文件：** `src/stage_b/config.py`

```python
@dataclass(frozen=True)
class GuidanceLifecycleConfig:
    """Guidance lifecycle management configuration."""
    confidence_drop_threshold: float = 0.35
    min_miss_before_drop: int = 3
    enable_auto_cleanup: bool = True  # Auto-cleanup at each epoch end
```

**在YAML中启用：**

```yaml
# configs/stage_b/debug.yaml
guidance_lifecycle:
  confidence_drop_threshold: 0.35
  min_miss_before_drop: 3
  enable_auto_cleanup: true
```

---

## 4. 当前问题诊断

### 4.1 实验设置

```
运行ID：debug-0.1-0.4-0.7
模型：Qwen3-VL (Summary微调版本)
数据集：3个样本
任务：挡风板安装检查 (标签都是fail)
Epochs：3
Rollout温度：[0.1, 0.4, 0.7]
```

### 4.2 结果统计

| 指标 | 数值 | 说明 |
|------|------|------|
| **总样本数** | 3 | 都标记为fail |
| **Rollout生成的Candidates** | 22个 | 跨3个样本，多温度采样 |
| **Candidates中verdict=pass** | 22个 | **100%** 🔴 |
| **Candidates中verdict=fail** | 0个 | **0%** |
| **Selection结果(最终预测)** | 9个 | 3样本×3epochs |
| **其中label_match=true** | 0个 | **0% 准确率** 🔴 |
| **其中label_match=false** | 9个 | **100% 错误** |
| **需要Manual Review** | 20个 | 占 69% 的记录 |

### 4.3 Manual Review原因分布

```
label_mismatch_after_reflection:  9条 (45%)
  → 预测=pass，标签=fail，模型与标签不符
  → Reflection学到了规则，但无法改变已生成的candidates

no_support_after_reflection:      6条 (30%)
  → Reason内容支持pass，但规则要求fail，没有证据支持fail
  → 产生矛盾

format_error:                     5条 (25%)
  → Rollout/Summary生成了格式错误的内容
  → 无法被解析
```

### 4.4 Reflection学到的规则

```
规则文本：
"当BBU设备按要求配备了挡风板,但挡风板只显示部分,
 无法判断安装是否牢固情况,判定为不过"

应用于：
- QC-TEMP-20241022-0015062
- QC-TEMP-20241122-0015316
- QC-TEMP-20241202-0015446
```

**问题：**
- ✅ 规则逻辑正确，符合人工标签
- ❌ 但所有Candidates都已经是verdict=pass
- ❌ 这个规则永远无法被应用（改变预测）

---

## 5. 数据分析与根本原因

### 5.1 样本详细分析

#### 样本1: QC-TEMP-20241022-0015062

```
标签: fail (因为有BBU设备未配备挡风板)

Rollout生成8个候选，全部verdict=pass：

Candidate 0 (温度0.4)：
  Verdict: pass
  Reason: "Image1: 标签/无法识别;BBU设备/中兴,显示完整,机柜空间充足需要安装/
           这个BBU设备按要求配备了挡风板;BBU设备/中兴,只显示部分,
           机柜空间充足需要安装/这个BBU设备未按要求配备挡风板;..."

  ⚠️ Reason同时提到：
     - ✓ "按要求配备了挡风板" (正面)
     - ✓ "未按要求配备挡风板" (负面)
     - → 但最终判定为 pass ❌

Candidate 1-7: 类似，全是pass
```

#### 样本2: QC-TEMP-20241122-0015316

```
标签: fail (挡风板只显示部分，无法判断是否牢固)

Rollout生成9个候选，全部verdict=pass：

Candidate 0-8：
  Verdict: pass
  Reason: "这个BBU设备按要求配备了挡风板,但挡风板只显示部分,
           无法判断安装是否牢固情况"

  ⚠️ 同样的现象：
     - Reason明确说了"无法判断"（这应该是不过的表现）
     - 但最终判定为 pass ❌
```

#### 样本3: QC-TEMP-20241202-0015446

```
标签: fail (未拍摄到BBU设备)

Rollout生成5个候选，全部verdict=pass：

Candidate 0-4：
  Verdict: pass
  Reason: "无法判断是否足够空间安装挡风板,无需安装"

  ⚠️ 特点：
     - 全是"无法判断"的内容
     - 应该倾向fail或uncertain
     - 但判定为 pass ❌
```

### 5.2 根本原因分析

#### **假设：Summary模型引入的正向偏倚**

```
原始流程（无Summary）：
  Image → 直接分析 → Verdict ✓

当前流程（有Summary）：
  Image → Summary模型(信息提取) → Rollout分析 → Verdict ❌
           ↓
        已被污染的输入
```

**Summary模型的问题：**

1. **任务定义问题**
   - Summary被训练成**信息提取**任务
   - 它学会了"总结有什么"，不是"判断是否通过"
   - 优秀的总结特征：完整、平衡、有组织
   - 这些特征自然地倾向正面表述

2. **输入信息过滤**
   - Summary从原始Image中提取关键信息
   - 这个过程做了**隐性的简化和权重调整**
   - 可能强调了"设备完整性"而淡化了"不符合项"

3. **Rollout的过度依赖**
   - Rollout基于Summary的输出来生成Verdict
   - 如果Summary中文的"气质"是正面的，Verdict就倾向pass
   - 即使Reason里有负面内容，"词汇组织"已经被Summary影响了

### 5.3 对比假设

如果使用**无Summary版本**的数据，会看到什么？
- 预期：verdict分布更均匀，包含pass和fail
- 预期：有些候选会是fail，能与标签匹配
- 预期：Reflection能学到规则并立即应用（有fail→fail的对应）

### 5.4 Reflection的无奈

```
Reflection期望的循环：
  Epoch1: 候选=[pass, fail, pass] → 学到规则：当X时→fail
  Epoch2: 新的candidates + 新规则 → verdict从pass变成fail ✓
  Epoch3: 更多样本验证规则 → 置信度提升

当前实际的情况：
  Epoch1: 候选=[pass, pass, pass] → 学到规则：当Y时→fail
  Epoch2: 新的candidates + 新规则 → verdict仍然[pass, pass, pass] ❌
          规则已学，但无法应用（没有fail候选）
  Epoch3: 重复...
```

---

## 6. 需要Code Review与改进建议的地方

### 6.1 核心问题区域

#### 区域1：Summary的使用方式

**问题位置：** `src/stage_b/rollout.py` （Rollout prompt）

**当前假设：** Rollout直接基于Summary生成Verdict

**需要检查：**
1. Rollout的prompt是否过度依赖Summary的整体表述？
2. 是否缺少"强制考虑负面信息"的约束？
3. 能否将Summary的输出重新结构化，以减少偏倚？

**改进建议方向：**
- [ ] 直接使用原始Image而不是Summary（快速验证假设）
- [ ] 修改Rollout的prompt，显式要求考虑负面模式
- [ ] 在Selection阶段加入"有任何fail候选就标记风险"的逻辑
- [ ] 重新微调Summary，让它关注"问题点"而不仅仅是"完整信息"

#### 区域2：Hit/Miss生命周期管理的完整性

**问题位置：** `src/stage_b/runner.py` + `src/stage_b/io/guidance.py`

**当前实现检查清单：**

- [x] 追踪last_applied_rule_keys ✓
- [x] 从Reflection提取applied rules ✓
- [x] 调用increment_miss_count() ✓
- [ ] **缺失：increment_hit_count()** ❌

**问题：**
- 只在预测失败时更新miss_count
- 从未显式更新hit_count
- 导致hit_count永远=1（规则创建时）

**改进建议：**
```python
# 当样本预测成功时，也应该更新hit_count
if win_cand and win_cand.signals and win_cand.signals.label_match is True:
    if last_applied_rule_keys:
        mission_guidance_repo.increment_hit_count(
            mission, last_applied_rule_keys
        )
```

#### 区域3：Reflection的证据判断逻辑

**问题位置：** `src/stage_b/reflection/engine.py` （reflection prompt）

**当前设计：**
- Reflection判断"是否有证据支持标签"
- 如果无证据，标记为no_evidence_for_label

**问题：**
- 当所有candidates都是pass，理由中没有fail的证据
- Reflection虽然学到了规则，但产生矛盾（理由说pass，规则要fail）

**改进建议：**
- [ ] 调整Reflection的prompt，让它也考虑"理由与规则的一致性"
- [ ] 如果理由与标签矛盾，应该更激进地提出规则修正
- [ ] 考虑让Reflection也能"修正已生成的candidates"（如果可能）

#### 区域4：Selection策略

**问题位置：** `src/stage_b/scoring/selection.py`

**当前策略：** 投票，取vote_strength最高的verdict

**问题：**
- 当所有candidates都是pass，投票结果自然也是pass
- 即使标签是fail，也无法被纠正

**改进建议：**
- [ ] 如果预测与标签严重冲突，考虑返回uncertainty（而不是强行选择）
- [ ] 增加一个"反向投票"机制：如果有任何fail候选，即使少数也标记风险
- [ ] 考虑基于Guidance的规则来调整vote权重

### 6.2 值得重新审视的设计决策

#### 1. Summary的必要性

**当前设计：**
```
Image → Summary(SFT微调模型) → Rollout分析 → Verdict
```

**问题：**
- Summary引入了偏倚
- 增加了流程复杂度

**替代方案：**
```
Image → Rollout直接分析 → Verdict
（使用Focus来指导Rollout，而不是Summary）
```

**建议：**
- [ ] 运行无Summary版本，对比效果
- [ ] 如果有效，永久移除Summary
- [ ] 如果无效，说明Summary不是主要原因

#### 2. 规则学习的粒度

**当前设计：**
- Reflection从失败样本中学习规则
- 一条规则可能基于多个样本证据

**问题：**
- 当样本本身有噪声（Summary偏倚导致的错误），学到的规则也有问题
- Hit/Miss跟踪无法纠正这种系统性偏倚

**改进建议：**
- [ ] 考虑样本来源的质量（是否来自Summary）
- [ ] 为来自噪声源的规则降权
- [ ] 增加规则的"版本"概念，允许规则被推翻和替换

#### 3. Manual Review队列的处理

**当前设计：**
- 20个样本进入manual_review_queue
- 需要人工决策

**问题：**
- 69%的样本进入人工复核，这太高了
- 这种情况应该触发**系统级别的诊断告警**

**改进建议：**
- [ ] 添加质量监控：manual_review比例 > 50% → 红色告警
- [ ] 触发告警时，自动运行诊断（对比无Summary版本、检查Rollout偏倚等）
- [ ] 考虑自动回滚到上一个良好状态

### 6.3 代码质量检查项

#### 规则名称生成（Hash冲突风险）

**位置：** `src/stage_b/runner.py:587`

```python
applied_keys.append(f"_text_{hash(op.text) % (10**8)}")
```

**风险：**
- Hash冲突虽然概率小，但不是零
- 不同的规则可能生成相同的key

**改进建议：**
```python
# 使用更稳定的标识符
import hashlib
text_hash = hashlib.md5(op.text.encode()).hexdigest()[:8]
applied_keys.append(f"_text_{text_hash}")
```

#### increment_miss_count实现检查

**位置：** `src/stage_b/io/guidance.py`

**需要确保：**
- [ ] 正确处理不存在的key（创建新entry vs 忽略）
- [ ] 原子性：修改metadata后一定要save
- [ ] 边界条件：miss_count + hit_count 不会溢出
- [ ] 日志记录：每次修改都有trace

#### 配置的默认值

**位置：** `src/stage_b/config.py`

```python
class GuidanceLifecycleConfig:
    confidence_drop_threshold: float = 0.35  # 合理吗？
    min_miss_before_drop: int = 3            # 样本太少时过激？
    enable_auto_cleanup: bool = True         # 应该默认开？
```

**问题：**
- 这些值是通过什么标准确定的？
- 是否有数据支持这些选择？

**改进建议：**
- [ ] 添加配置的文档说明
- [ ] 进行敏感性分析（尝试不同的阈值）
- [ ] 考虑不同的任务可能需要不同的参数

---

## 7. 建议的调查与验证步骤

### 第一阶段：验证Summary是否是根本原因（1天）

```bash
# 1. 运行无Summary的版本（如果存在）
bash scripts/stage_b.sh --config debug_no_summary.yaml

# 2. 对比结果
python tools/compare_results.py \
  --result1 output_post/stage_b/debug-0.1-0.4-0.7 \
  --result2 output_post/stage_b/debug_no_summary
```

**预期：**
- ✓ 如果pass率显著降低，证实Summary是罪魁祸首
- ✓ 如果verdict分布变为pass/fail混合，证实
- ✗ 如果还是全pass，说明问题在Rollout或Selection

### 第二阶段：改进Rollout Prompt（1-2天）

```python
# 在Rollout的prompt中加入反偏倚约束
ROLLOUT_SYSTEM_PROMPT += """
重要提醒：即使摘要中强调了"符合要求"的方面，
如果同时提到任何"不符合""未配备""无法判断"等负面信息，
都应该倾向于fail判定。不要被摘要的整体"完整性"迷惑。
"""
```

### 第三阶段：补全Hit/Miss机制（1天）

```python
# 添加increment_hit_count()
# 完善confidence计算
# 测试cleanup流程
```

### 第四阶段：质量监控与告警（1-2天）

```python
# 添加Stage-B级别的质量检查
if manual_review_ratio > 0.5:
    logger.critical("High manual review ratio detected!")
    logger.critical("Potential data quality or model bias issue")
    # 触发诊断报告
```

---

## 8. 总结：给新同事的建议

### 核心问题
> **Summary微调模型导致Rollout全部输出pass，使整个学习机制失效**

### 立即调查
1. **确认Summary是否是主因**
   - 对比有Summary和无Summary的结果
   - 如果有Summary是主因，下一步是修改Rollout或放弃Summary

2. **如果不是Summary**
   - 检查Rollout的微调数据质量
   - 检查Rollout的prompt是否过于乐观

### 短期改进（可并行进行）
1. 修改Rollout prompt，加入反偏倚约束
2. 补全Hit/Miss机制（添加increment_hit_count）
3. 添加质量监控（manual_review告警）

### 中期改进
1. 重新评估Summary的必要性
2. 如果保留Summary，重新微调它（关注问题点而不仅是完整性）
3. 进行敏感性分析，调优lifecycle参数

### 长期方向
1. 考虑一个专门的"Judgment模型"替代Summary + Rollout的两层结构
2. 建立自动化诊断工具，监控model bias
3. 实现更完善的版本控制和回滚机制

---

## 附录A：Key文件导航

```
src/stage_b/
├── runner.py                  ← 主循环（Hit/Miss更新位置）
├── rollout.py                 ← Rollout采样（可能有Summary偏倚）
├── reflection/
│   └── engine.py              ← Reflection + 规则提取
├── scoring/
│   └── selection.py           ← Selection投票逻辑
├── io/
│   └── guidance.py            ← Guidance读写 + Hit/Miss方法
├── config.py                  ← 配置（GuidanceLifecycleConfig）
└── types.py                   ← 数据结构（ExperienceMetadata）

configs/
└── stage_b/
    └── debug.yaml             ← 配置示例

docs/
├── README.md                  ← 目录
├── training/REFERENCE.md      ← 架构图
└── stage_b/
    └── [本文件]
```

---

## 附录B：快速参考 - Reflection Learn Loop

```
┌────────────────────────────────────────────────────────┐
│ Reflection学习循环的预期工作流程                         │
└────────────────────────────────────────────────────────┘

理想情况（有足够的fail候选）：

Epoch 1, Sample A:
  ├─ Rollout: [pass(80%), fail(20%)]
  ├─ Selection: 投票选fail（因为有fail候选）
  ├─ 预测=fail，标签=fail ✓
  └─ Reflection: 学到规则 "当X时→fail"
                 hit_count=1

Epoch 2, Sample B:
  ├─ Rollout: [pass(85%), fail(15%)]  (新数据)
  ├─ Selection: 规则权重 → 倾向fail
  ├─ 预测=fail，标签=fail ✓
  └─ Reflection: 规则验证成功
                 hit_count=2, miss_count=0

───────────────────────────────────────

当前实际情况（Summary偏倚导致无fail候选）：

Epoch 1, Sample A:
  ├─ Rollout: [pass(100%), fail(0%)]  ❌ ALL PASS
  ├─ Selection: 只能选pass
  ├─ 预测=pass，标签=fail ✗
  └─ Reflection: 学到规则 "当Y时→fail"
                 但无法验证（因为下一轮还是全pass）

Epoch 2, Sample B:
  ├─ Rollout: [pass(100%), fail(0%)]  ❌ ALL PASS (重复问题)
  ├─ Selection: 只能选pass
  ├─ 预测=pass，标签=fail ✗
  └─ Reflection: 无法验证规则，产生矛盾
                 → no_evidence_for_label

───────────────────────────────────────

修复路径：

选项A: 去掉Summary，直接用Image + Focus
  → 恢复足够的fail候选
  → 学习循环重新启动

选项B: 修改Rollout prompt
  → 显式要求考虑负面信息
  → 如果Summary中提到问题，倾向fail
  → 增加fail候选的比例

选项C: 修改Selection策略
  → 反向投票：有任何fail候选就标记风险
  → 而不是完全依赖多数投票
```
