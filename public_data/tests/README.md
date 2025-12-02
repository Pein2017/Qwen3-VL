# LVIS Converter Tests

测试套件用于验证 LVIS 数据集转换功能，无需图像文件即可运行。

## 快速开始

```bash
cd public_data
bash tests/run_tests.sh
```

## 测试内容

### ✅ Test 1: Annotation Loading
- 验证 LVIS JSON 标注文件加载
- 检查类别分布（Frequent/Common/Rare）
- 统计图像和标注数量

**数据规模**：
- 图像：100,170 张
- 标注：1,270,141 个
- 类别：1,203 个（f:405, c:461, r:337）

### ✅ Test 2: BBox Conversion
- 测试边界框转换逻辑
- COCO格式 `[x,y,w,h]` → Qwen3-VL格式 `[x1,y1,x2,y2]`
- 验证坐标裁剪和边界检查

**示例输出**：
```json
{
  "bbox_2d": [91.01, 89.24, 479.86, 358.16],
  "desc": "zebra"
}
```

### ✅ Test 3: Polygon Conversion
- 测试多边形分割转换
- 支持任意 N 点多边形（N ≥ 3）
- 统计多边形点数分布

**发现**：
- 最小：5 点多边形
- 最大：314 点多边形（精细轮廓）
- 常见：15-35 点范围
- **4点多边形极少**（~5%）

**示例输出**：
```json
{
  "poly": [327.26, 354.04, 329.51, 338.67, ...],
  "poly_points": 35,
  "desc": "person"
}
```

### ✅ Test 4: Qwen3-VL Format Compliance
- 验证输出格式符合 DATA_AND_DATASETS.md 规范
- 检查必需字段：`images`, `objects`, `width`, `height`
- 验证几何类型：`bbox_2d`, `poly`（N点多边形）
- 验证 `poly_points` 字段一致性

### ✅ Test 5: Polygon Cap Functionality (test_poly_cap.py)
- 测试多边形顶点上限功能（`poly_max_points`）
- 验证超过上限的多边形自动转换为 `bbox_2d`
- 验证数据合约合规性（仅 `bbox_2d` 和 `poly`，无 `line`/`quad`）
- 统计转换数量（`poly_to_bbox_capped`）

**测试场景**：
- 4点/8点/12点多边形 → 保持为 `poly`
- 15点/18点多边形 → 转换为 `bbox_2d`（默认上限12）

## 多边形统计（前20张图）

| 点数范围 | 数量 | 占比 |
|---------|------|------|
| 5-10 点 | 24 | 11.8% |
| 11-20 点 | 51 | 25.1% |
| 21-40 点 | 69 | 34.0% |
| 41-70 点 | 35 | 17.2% |
| 71-100 点 | 16 | 7.9% |
| 100+ 点 | 8 | 3.9% |

**结论**：
- ✅ LVIS 提供丰富的多边形分割数据
- ✅ 大部分为 10-40 点的详细轮廓
- ✅ 适合训练精细的视觉理解能力

## 运行环境

**必需**：
- conda 环境：`ms`
- Python 3.x
- 已下载的 LVIS 标注文件

**不需要**：
- 图像文件（测试使用 mock 目录）
- GPU

## 文件说明

```
tests/
├── README.md                    # 本文件
├── run_tests.sh                 # 测试运行脚本（使用ms环境）
├── test_lvis_converter.py       # 转换器测试套件
└── test_poly_cap.py             # 多边形顶点上限功能测试
```

## 下载完成后的完整测试

当图像下载完成后，运行完整转换测试：

```bash
# 测试10个样本（带图像验证）
cd public_data
conda run -n ms python scripts/convert_lvis.py --test --split train

# 查看输出
conda run -n ms python scripts/validate_jsonl.py \
  lvis/processed/test_conversion.jsonl
```

## 故障排查

### 问题：找不到 conda 环境
```bash
conda env list  # 检查环境
conda activate ms  # 手动激活
```

### 问题：无法导入模块
```bash
# 确保从 /data/public_data 目录运行
cd public_data
conda run -n ms python tests/test_lvis_converter.py
```

## 预期结果

✅ 所有测试应该通过：

**test_lvis_converter.py** (4个测试)：
```
✓ PASS - Annotation Loading
✓ PASS - BBox Conversion
✓ PASS - Polygon Conversion
✓ PASS - Format Compliance

Total: 4/4 tests passed
```

**test_poly_cap.py** (2个测试)：
```
✓ PASS - Polygon Cap Functionality
✓ PASS - Data Contract Compliance

Total: 2/2 tests passed
```
