# Polygon Support in LVIS Conversion

## 概念统一：多边形表示

所有几何类型统一为**点列表**表示：

| 类型 | 格式 | 点数 | 说明 |
|------|------|------|------|
| `bbox_2d` | `[x1, y1, x2, y2]` | 2 | 隐式矩形（对角点） |
| `quad` | `[x1, y1, ..., xn, yn]` | N ≥ 3 | 任意封闭多边形 |
| `line` | `[x1, y1, ..., xn, yn]` + `line_points` | N ≥ 2 | 开放折线 |

**关键设计**：
- `quad` 不再限制为4个点，而是**任意N点封闭多边形**
- 添加 `quad_points` 字段记录点数（类似 `line_points`）
- bbox 是特殊的2点多边形

---

## LVIS Segmentation 转换

### 输入格式（LVIS）
```json
{
  "segmentation": [
    [327.26, 354.04, 329.51, 338.67, 323.14, 341.67, ...],  // Part 1
    [100.5, 200.3, ...]  // Part 2 (可选)
  ]
}
```

### 输出格式（Qwen3-VL）
```json
{
  "objects": [
    {
      "quad": [327.26, 354.04, 329.51, 338.67, 323.14, 341.67, ...],
      "quad_points": 8,
      "desc": "person"
    },
    {
      "quad": [100.5, 200.3, ...],
      "quad_points": 12,
      "desc": "person"
    }
  ]
}
```

**转换规则**：
- 每个 segmentation part 转换为一个独立的 `quad` 对象
- `quad_points` = `len(quad) / 2`
- 最少3个点（6个坐标值）
- 坐标值必须是偶数个

---

## 使用方法

### 1. bbox-only 模式（标准）
```bash
python scripts/convert_lvis.py --split train
```
输出：只包含 `bbox_2d`

### 2. polygon 模式（完整分割）
```bash
python scripts/convert_lvis.py --split train --use-polygon
```
输出：包含 `quad`（N点多边形）

### 3. 测试模式
```bash
python scripts/convert_lvis.py --test --split train
```
自动启用 polygon 模式，转换10个样本验证

---

## 数据统计

LVIS 多边形点数分布（预估）：
- 3-5 点：~5%（简单形状）
- 6-10 点：~30%（常见物体）
- 11-20 点：~45%（详细轮廓）
- 20+ 点：~20%（复杂形状）

**优势**：
- ✅ 完整保留 LVIS 的精细分割标注
- ✅ 向量化表示，适合 V-LLM
- ✅ 1203 类别 × 精细轮廓 = 高质量训练数据

---

## 验证

```bash
# 验证输出格式
python scripts/validate_jsonl.py lvis/processed/train.jsonl

# 检查点：
# - quad 坐标数量 = quad_points * 2
# - quad_points >= 3
# - 所有坐标在图像范围内（允许小幅超出）
```

---

## 与 Qwen3-VL 格式对齐

需要在 Qwen3-VL 项目中放松 `quad` 验证：

```python
# src/datasets/preprocessors/dense_caption.py
# 修改前：
assert len(obj["quad"]) == 8, "quad must have 8 values"

# 修改后：
assert len(obj["quad"]) >= 6 and len(obj["quad"]) % 2 == 0, \
    "quad must have at least 6 values (3 points) and even number of coords"
assert "quad_points" in obj, "quad requires quad_points field"
assert len(obj["quad"]) == obj["quad_points"] * 2, \
    "quad length must match quad_points * 2"
```

类似 `line` 的处理方式。

---

## 性能考虑

**内存**：
- 平均每个多边形 ~10 点 = 20 个 float = 160 bytes
- vs bbox (4 个 float = 32 bytes)
- 增加约 5x 内存，但仍然远小于像素 mask

**序列化**：
- 模板需要将多边形坐标序列化为文本
- 建议归一化到 `norm1000`（与 bbox 一致）
- 示例：`<|quad|>[(123,456),(234,567),...]<|quad_end|>`

---

## 总结

通过**放松 quad 为任意多边形**，实现了：
- ✅ 概念统一（点列表）
- ✅ 完整利用 LVIS 分割数据
- ✅ 保持向量化（非像素级）
- ✅ 与现有格式兼容（只需放松验证）

**下一步**：等 LVIS 下载完成后测试转换！

