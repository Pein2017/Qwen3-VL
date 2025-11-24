# LVIS Visualization Tools

可视化工具用于展示LVIS数据集的标注。

## 功能

- ✅ 展示边界框（bbox）
- ✅ 展示多边形分割（polygon）
- ✅ 随机采样图片
- ✅ 支持保存或显示
- ✅ 显示类别标签和点数

## 快速开始

### 前提条件

确保图像已解压：
```bash
cd ./lvis/raw/images
unzip train2017.zip  # 或 val2017.zip
```

### 基础用法

```bash
cd public_data

# 展示3张带bbox的训练图片
conda run -n ms python vis_tools/visualize_lvis.py

# 展示5张带多边形的图片
conda run -n ms python vis_tools/visualize_lvis.py \
  --mode polygon \
  --num_samples 5

# 同时显示bbox和多边形
conda run -n ms python vis_tools/visualize_lvis.py \
  --mode both \
  --num_samples 2

# 使用验证集
conda run -n ms python vis_tools/visualize_lvis.py \
  --split val

# 保存图片而不是显示
conda run -n ms python vis_tools/visualize_lvis.py \
  --save \
  --output_dir vis_tools/output
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--split` | train | 数据集分割（train/val） |
| `--mode` | bbox | 可视化模式（bbox/polygon/both） |
| `--num_samples` | 3 | 展示图片数量 |
| `--max_polygons` | 5 | 每张图最多显示的多边形数 |
| `--save` | False | 保存图片而不是显示 |
| `--output_dir` | vis_tools/output | 输出目录 |
| `--seed` | 42 | 随机种子 |

## 可视化模式

### 1. bbox 模式（默认）

只显示边界框，适合快速查看目标位置：
```bash
conda run -n ms python vis_tools/visualize_lvis.py --mode bbox
```

**特点**：
- 彩色边界框
- 显示类别标签
- 清晰简洁

### 2. polygon 模式

显示精细分割多边形：
```bash
conda run -n ms python vis_tools/visualize_lvis.py --mode polygon
```

**特点**：
- 显示完整轮廓
- 半透明填充
- 标注点数
- 最多显示5个多边形（避免过度拥挤）

### 3. both 模式

同时显示bbox和多边形：
```bash
conda run -n ms python vis_tools/visualize_lvis.py --mode both
```

**特点**：
- bbox用虚线
- polygon用实线
- 便于对比

## 示例

### 示例1: 快速检查训练数据
```bash
# 随机展示5张图
conda run -n ms python vis_tools/visualize_lvis.py \
  --num_samples 5 \
  --mode bbox
```

### 示例2: 检查多边形质量
```bash
# 展示多边形标注，保存结果
conda run -n ms python vis_tools/visualize_lvis.py \
  --mode polygon \
  --num_samples 10 \
  --save
```

### 示例3: 验证集可视化
```bash
# 验证集，同时显示bbox和多边形
conda run -n ms python vis_tools/visualize_lvis.py \
  --split val \
  --mode both \
  --num_samples 3
```

## 输出说明

### 控制台输出

```
============================================================
LVIS Visualization Tool
============================================================
  Split: train
  Mode: polygon
  Samples: 3
============================================================

Loading annotations from: ./lvis/raw/annotations/lvis_v1_train.json
  Loaded 100170 images
  Loaded 1270141 annotations
  Loaded 1203 categories

Checking for available images...
  Found 156 available images
  156 have annotations

============================================================
Visualizing 3 samples...
============================================================

[1/3] 000000391895.jpg
  Size: 640x428
  Annotations: 1
    [0] zebra, 35pts

[2/3] 000000370765.jpg
  Size: 640x480
  Annotations: 6
    [0] dog, 23pts
    [1] dog, 45pts
    ...
```

### 可视化图片

- **图片**: 原始COCO图像
- **标注**: 彩色bbox或多边形
- **标签**: 类别名称（polygon模式显示点数）
- **标题**: 显示模式和标注统计

## 注意事项

1. **图像必须已解压**: 脚本会检查图像目录
2. **内存使用**: 如果图片很大，建议减少`--num_samples`
3. **显示限制**: 多边形模式默认最多显示5个（避免太拥挤）
4. **随机性**: 使用`--seed`可固定随机选择

## 故障排查

### 问题: 找不到图像

```bash
✗ Error: Image directory not found
```

**解决**: 解压图像文件
```bash
cd ./lvis/raw/images
unzip train2017.zip
```

### 问题: matplotlib显示错误

如果无法显示图形窗口，使用`--save`参数保存图片：
```bash
conda run -n ms python vis_tools/visualize_lvis.py --save
```

### 问题: 没有可用图像

如果只检查到很少图像，说明下载还在进行中。等待下载完成或使用已下载的验证集。

## 依赖

- Python 3.x
- matplotlib
- Pillow (PIL)
- numpy

这些应该已在`ms`环境中安装。

## 与转换器对比

| 工具 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `visualize_lvis.py` | 可视化原始LVIS数据 | LVIS JSON + 图像 | PNG可视化 |
| `convert_lvis.py` | 转换为Qwen3-VL格式 | LVIS JSON + 图像 | JSONL |

**建议工作流**：
1. 用`visualize_lvis.py`检查原始数据质量
2. 用`convert_lvis.py`转换数据
3. 用转换后的JSONL训练模型

