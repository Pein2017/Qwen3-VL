# LVIS Data Pipeline - Next Steps

**Status**: ✅ Dataset downloaded and extracted  
**Date**: 2025-10-27

---

## 📊 Current Status

| Component | Status | Details |
|-----------|--------|---------|
| Annotations | ✅ Complete | train (1.1GB), val (192MB) |
| Images | ✅ Complete | train (118,287), val (5,000) |
| Conversion | ⏳ Ready | Waiting to run |
| Visualization | ✅ Tested | Working correctly |

---

## 🚀 Step-by-Step Guide

### Step 1: Verify Data (Optional but Recommended)

先可视化几个样本，确认数据完整性：

```bash
cd /data/public_data

# 生成5张可视化图片（bbox + polygon）
conda run -n ms python vis_tools/visualize_lvis.py \
  --num_samples 5 \
  --save \
  --mode both
```

查看生成的图片：`vis_tools/output/*.png`

---

### Step 2: Convert to Qwen3-VL JSONL Format

选择以下三种方式之一：

#### 方式A: 完整转换（推荐，包含多边形）

```bash
cd /data/public_data

# 转换训练集和验证集，包含多边形标注
conda run -n ms python scripts/convert_lvis.py --use-polygon
```

**输出**：
- `lvis/processed/lvis_train.jsonl` (~1-2 GB)
- `lvis/processed/lvis_val.jsonl` (~100-200 MB)
- `lvis/stats/conversion_stats.json`

**时间**: 约10-20分钟（取决于CPU）

#### 方式B: 仅边界框（更快）

如果只需要bbox，不需要多边形：

```bash
cd /data/public_data

# 只转换bbox
conda run -n ms python scripts/convert_lvis.py
```

**时间**: 约5-10分钟

#### 方式C: 快速测试（10个样本）

先转换少量数据测试流程：

```bash
cd /data/public_data

# 只转换10个样本，快速验证
conda run -n ms python scripts/convert_lvis.py --use-polygon --test
```

**输出**: `lvis/processed/lvis_train.jsonl` (10 samples)  
**时间**: 约10秒

---

### Step 3: Validate Converted Data

转换完成后，验证输出格式：

```bash
cd /data/public_data

# 验证训练集
conda run -n ms python scripts/validate_jsonl.py \
  lvis/processed/lvis_train.jsonl

# 验证验证集
conda run -n ms python scripts/validate_jsonl.py \
  lvis/processed/lvis_val.jsonl
```

**预期输出**：
- Schema validation passed ✓
- Image paths verified ✓
- Bounding boxes valid ✓
- Polygons valid ✓

---

### Step 4: Create Sample Subsets (Optional)

如果计算资源有限，创建采样子集：

#### 小型数据集（1000样本，用于快速实验）

```bash
cd /data/public_data

conda run -n ms python scripts/sample_dataset.py \
  --input lvis/processed/lvis_train.jsonl \
  --output lvis/processed/lvis_train_1k.jsonl \
  --num_samples 1000 \
  --strategy stratified
```

#### 中型数据集（10000样本）

```bash
conda run -n ms python scripts/sample_dataset.py \
  --input lvis/processed/lvis_train.jsonl \
  --output lvis/processed/lvis_train_10k.jsonl \
  --num_samples 10000 \
  --strategy stratified
```

#### 按类别采样

```bash
# 只保留特定类别
conda run -n ms python scripts/sample_dataset.py \
  --input lvis/processed/lvis_train.jsonl \
  --output lvis/processed/lvis_train_vehicles.jsonl \
  --categories "car,truck,bus,motorcycle,bicycle"
```

---

### Step 5: Inspect Converted Data

查看转换后的JSONL格式：

```bash
cd /data/public_data

# 查看第一个样本
head -1 lvis/processed/lvis_train.jsonl | python -m json.tool
```

**期望格式**：
```json
{
  "images": ["path/to/image.jpg"],
  "objects": [
    {
      "bbox_2d": [x1, y1, x2, y2],
      "desc": "category_name"
    },
    {
      "quad": [x1, y1, ..., xn, yn],
      "quad_points": 5,
      "desc": "another_category"
    }
  ],
  "width": 1920,
  "height": 1080,
  "summary": "optional scene description"
}
```

---

## 🔍 Quick Reference

### 完整工作流（推荐）

```bash
cd /data/public_data

# 1. 可视化验证
conda run -n ms python vis_tools/visualize_lvis.py --num_samples 5 --save

# 2. 快速测试转换
conda run -n ms python scripts/convert_lvis.py --use-polygon --test

# 3. 检查测试结果
head -1 lvis/processed/lvis_train.jsonl | python -m json.tool

# 4. 如果测试通过，完整转换
conda run -n ms python scripts/convert_lvis.py --use-polygon

# 5. 验证完整数据
conda run -n ms python scripts/validate_jsonl.py lvis/processed/lvis_train.jsonl

# 6. 创建1k采样子集（用于快速实验）
conda run -n ms python scripts/sample_dataset.py \
  --input lvis/processed/lvis_train.jsonl \
  --output lvis/processed/lvis_train_1k.jsonl \
  --num_samples 1000 \
  --strategy stratified
```

---

## 📁 转换后的目录结构

```
lvis/
├── raw/
│   ├── annotations/
│   │   ├── lvis_v1_train.json
│   │   └── lvis_v1_val.json
│   └── images/
│       ├── train2017/  (118,287 images)
│       └── val2017/    (5,000 images)
├── processed/
│   ├── lvis_train.jsonl         # 完整训练集
│   ├── lvis_val.jsonl           # 完整验证集
│   ├── lvis_train_1k.jsonl      # 1k采样（可选）
│   └── lvis_train_10k.jsonl     # 10k采样（可选）
├── stats/
│   └── conversion_stats.json    # 转换统计
└── metadata/
    └── category_names.txt       # 类别列表
```

---

## 🎯 集成到 Qwen3-VL

转换完成后，在 Qwen3-VL 配置中使用：

```yaml
# /data/Qwen3-VL/configs/your_experiment.yaml

custom:
  train_jsonl: ./lvis/processed/lvis_train_1k.jsonl
  val_jsonl: ./lvis/processed/lvis_val.jsonl
  emit_norm: norm1000  # LVIS coordinates are in pixels
  
  # Polygon support
  images_per_user_turn: 1
  
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  
model:
  model: /path/to/Qwen3-VL-4B-Instruct
  
template:
  template: qwen3_vl
  max_length: 4096
```

然后启动训练：

```bash
cd /data/Qwen3-VL
conda run -n ms bash scripts/train.sh \
  config=/data/Qwen3-VL/configs/your_experiment.yaml \
  gpus=0
```

---

## ⚠️ 注意事项

1. **磁盘空间**: 确保有足够空间存储转换后的JSONL文件（~2-3GB）
2. **内存**: 完整转换需要约8-16GB RAM
3. **时间**: 完整转换约需10-20分钟
4. **Conda环境**: 始终使用 `conda run -n ms` 前缀

---

## 🐛 故障排查

### 问题：转换速度很慢

**解决**：
- 使用 `--test` 先测试小样本
- 考虑只转换bbox（不用 `--use-polygon`）
- 检查磁盘I/O性能

### 问题：内存不足

**解决**：
- 分批转换（修改脚本添加batch处理）
- 先转换val集（较小）
- 使用采样创建更小的子集

### 问题：验证失败

**解决**：
- 检查 `validate_jsonl.py` 的错误信息
- 确认图像路径可访问
- 查看 `lvis/stats/conversion_stats.json` 的统计信息

---

## 📚 相关文档

- `LVIS_QUICKSTART.md` - LVIS快速入门
- `POLYGON_SUPPORT.md` - 多边形支持说明
- `vis_tools/README.md` - 可视化工具文档
- `tests/README.md` - 测试文档

---

## ✅ 下一步目标

完成转换后，你应该有：

- [x] 完整的LVIS数据集（原始格式）
- [ ] 转换后的Qwen3-VL JSONL文件
- [ ] 验证通过的数据
- [ ] （可选）采样的训练子集
- [ ] 准备好集成到训练流程

**推荐命令**（复制粘贴执行）：

```bash
cd /data/public_data && \
conda run -n ms python scripts/convert_lvis.py --use-polygon --test && \
echo "=== Test conversion complete! Check the output above. ===" && \
echo "If it looks good, run the full conversion:" && \
echo "  conda run -n ms python scripts/convert_lvis.py --use-polygon"
```

