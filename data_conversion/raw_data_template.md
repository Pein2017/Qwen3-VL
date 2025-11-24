# 数据堂标注平台输出格式规范

## 概述

本文档定义了数据堂标注平台的原始数据输出格式，包含完整的标注结果、质检信息、统计数据和图片信息。

## 主要数据结构

输出数据采用 JSON 格式，包含以下四个主要部分：

- **markResult**: 标注结果数据
- **qualityResult**: 质检结果数据  
- **workload**: 统计信息
- **info**: 图片基本信息

## 完整样例

```json
{
  "qualityResult": {
    "type": "FeatureCollection",
    "features": []
  },
  "markResult": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": {
          "objectId": 4,
          "layerId": 5,
          "content": {
            "5": "33/34",
            "label": "biao1juxing",
            "S1juxing01": "S1juxing01-01",
            "xialaduoxuian": ["0"]
          },
          "ocrContent": {},
          "generateMode": 1,
          "quality": {
            "errorType": {
              "attributeError": [],
              "targetError": [],
              "otherError": ""
            },
            "changes": {
              "attribute": [],
              "remark": "",
              "target": []
            },
            "qualityStatus": "unqualified"
          },
          "labelColor": [162, 242, 157],
          "area": 16643.55555555555,
          "groups": []
        },
        "geometry": {
          "type": "ExtentPolygon",
          "coordinates": [
            [540.8333333333334, 149.49999999999997],
            [641.5, 149.49999999999997],
            [641.5, 314.8333333333333],
            [540.8333333333334, 314.8333333333333],
            [540.8333333333334, 149.49999999999997]
          ]
        }
      }
    ]
  },
  "workload": {
    "tagCount": 1,
    "qualifiedCount": 0,
    "unqualifiedCount": 0,
    "tagCount1": 1,
    "qualifiedCount1": 0,
    "unqualifiedCount1": 0,
    "tagCount2": 0,
    "tagCount3": 0,
    "label": {
      "biao1juxing": 1
    },
    "allMissingCount": 0,
    "missingCount": 0,
    "extent": {
      "tagCount": 1,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 1,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {
        "biao1juxing": 1
      }
    },
    "cubeUnstandard": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "line": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "parallelogram": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "cubeTrapezium": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "point": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "square": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "tiltRectangle": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "polygon": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "trapezium": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "ellipse": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "circle": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    },
    "cubeStandard": {
      "tagCount": 0,
      "qualifiedCount": 0,
      "unqualifiedCount": 0,
      "tagCount1": 0,
      "qualifiedCount1": 0,
      "unqualifiedCount1": 0,
      "label": {}
    }
  },
  "info": {
    "depth": 3,
    "width": 1219,
    "height": 650
  },
  "version": "4.0.0"
}
```

## 数据字段详细说明

### 顶层字段

| 字段 | 含义 | 类型 | 说明 |
|------|------|------|------|
| `markResult` | 标注结果 | object | 包含所有标注数据 |
| `qualityResult` | 画叉质检结果 | object | 质检相关数据 |
| `workload` | 统计信息 | object | 标注结果统计 |
| `info` | 图片信息 | object | 图片的宽、高、深度信息 |
| `version` | 版本信息 | string | 当前标注结果使用的模板版本 |

### markResult 结构

#### markResult 主要字段

| 字段 | 含义 | 类型 | 说明 |
|------|------|------|------|
| `type` | 标注结果类别 | string | 固定值 "FeatureCollection" |
| `features` | 标注结果列表 | array | 包含所有标注对象 |
| `imageResult` | 整图属性结果 | object | 图片级别的属性信息 |

#### markResult.features 字段详解

##### 基本结构

| 字段 | 含义 | 类型 | 说明 |
|------|------|------|------|
| `type` | 标注类别 | string | 固定值 "Feature" |
| `geometry` | 标注图形 | object | 几何形状数据 |
| `properties` | 标注图形信息 | object | 属性和元数据 |

##### geometry 字段

| 字段 | 含义 | 类型 | 取值及含义 |
|------|------|------|----------|
| `type` | 标注图形类别 | string | **Polygon**: 多边形<br>**LineString**: 线<br>**ExtentPolygon**: 矩形<br>**Square**: 四边形<br>**TiltRectangle**: 自由矩形<br>**Point**: 点<br>**Circle**: 圆<br>**Ellipse**: 椭圆<br>**Cube**: 3D 拉框<br>**Trapezium**: 梯形 |
| `coordinates` | 标注图形坐标 | array | 具体坐标点，详见图形坐标说明 |
| `lineType` | 贝塞尔曲线数据 | string/array | **L**: 普通点<br>**C**: 控制点<br>Polygon时为array，LineString时为string |
| `cubeMode` | 3D拉框绘制模式 | number | **1**: 标准立方体<br>**2**: 立方体<br>**3**: 梯型立方体 |

##### properties 字段

| 字段 | 含义 | 类型 | 说明 |
|------|------|------|------|
| `objectId` | 标注图形ID | number | 标注图形唯一标识 |
| `generateMode` | 标注框类别 | number | **1**: 手动框<br>**2**: 预识别 |
| `content` | 标签属性信息 | object | 键值对形式的属性数据 |
| `content.label` | 标签信息 | string | 多级标签用"/"分割 |
| `ocrContent` | OCR识别结果 | object | 智能OCR识别的结果 |
| `groups` | 组合信息 | array | 图形组合数据 |
| `labelColor` | 标签颜色 | array | RGB颜色值 |
| `area` | 图形面积 | number | 仅闭合图形有此字段 |
| `quality` | 质检信息 | object | 质检相关数据 |

##### quality 质检信息

| 字段 | 含义 | 类型 | 取值说明 |
|------|------|------|----------|
| `qualityStatus` | 质检结果 | string | **passed**: 合格<br>**failed**: 不合格<br>**changed**: 已修改<br>**unqualified**: 未质检 |
| `errorType` | 质检错误信息 | object | 包含各类错误信息 |
| `errorType.targetError` | 目标错误 | array | **unfit**: 不贴合<br>**annotationObject**: 对象不符<br>**other**: 其他 |
| `errorType.attributeError` | 属性错误 | array | 属性英文名集合 |
| `errorType.otherError` | 其他错误 | string | 其他错误描述 |
| `changes` | 修改状态 | object | 修改相关信息 |

#### markResult.imageResult 字段

| 字段 | 含义 | 类型 | 说明 |
|------|------|------|------|
| `content` | 图片属性结果 | object | 图片级别属性信息 |
| `quality` | 质检结果 | object | 图片级别质检信息 |
| `quality.qualityStatus` | 质检状态 | string | **passed**: 合格<br>**failed**: 不合格<br>**changed**: 已修改<br>**unqualified**: 未质检 |
| `quality.errorType.attributeError` | 属性错误列表 | array | 属性英文名组成的集合 |

### qualityResult 结构

| 字段 | 含义 | 类型 | 说明 |
|------|------|------|------|
| `type` | 质检结果类别 | string | 固定值 "FeatureCollection" |
| `features` | 质检图形列表 | array | 质检标记点列表 |

#### qualityResult.features 字段

| 字段 | 含义 | 类型 | 取值说明 |
|------|------|------|----------|
| `type` | 标注类别 | string | 固定值 "Feature" |
| `geometry.type` | 质检图形类别 | string | 固定值 "Point" |
| `geometry.coordinates` | 质检图形坐标 | array | 质检点坐标 |
| `properties.errorType` | 质检错误信息 | array | **unfit**: 边框不贴合<br>**wrongMark**: 标注对象不符<br>**leakage**: 漏标<br>**attributeError**: 属性错误 |
| `title` | 显示文字 | string | 质检标记显示的文字 |

### workload 统计信息

#### 整体统计

| 字段 | 含义 | 类型 | 说明 |
|------|------|------|------|
| `tagCount` | 框总数 | number | 所有标注框数量 |
| `qualifiedCount` | 合格框数 | number | 质检合格的框数量 |
| `unqualifiedCount` | 不合格框数 | number | 质检不合格的框数量 |
| `tagCount1` | 手动框数 | number | 手动标注的框数量 |
| `qualifiedCount1` | 手动框合格数 | number | 手动框中质检合格数量 |
| `unqualifiedCount1` | 手动框不合格数 | number | 手动框中质检不合格数量 |
| `tagCount2` | 总智能识别数 | number | 智能识别的框数量 |
| `tagCount3` | 识别未修改数 | number | 智能识别未修改的框数量 |
| `allMissingCount` | 漏标总数量 | number | 所有漏标数量 |
| `missingCount` | 未修改漏标总数量 | number | 未修改的漏标数量 |

#### 按图形类型统计

每种图形类型都包含以下统计字段：

- `{type}.tagCount`: 该类型图形总数
- `{type}.qualifiedCount`: 该类型图形合格数
- `{type}.unqualifiedCount`: 该类型图形不合格数
- `{type}.tagCount1`: 该类型手动框数
- `{type}.qualifiedCount1`: 该类型手动框合格数
- `{type}.unqualifiedCount1`: 该类型手动框不合格数
- `{type}.label`: 该类型标签统计信息

**支持的图形类型：**

- `extent`: 矩形图形
- `polygon`: 多边形图形
- `square`: 四边形图形
- `tiltRectangle`: 自由矩形图形
- `parallelogram`: 平行四边形图形
- `point`: 点图形
- `line`: 线图形
- `trapezium`: 梯形图形
- `circle`: 圆形图形
- `ellipse`: 椭圆形图形
- `cubeStandard`: 标准3D框图形
- `cubeUnstandard`: 非标准3D框图形
- `cubeTrapezium`: 梯形3D框图形

### info 图片信息

| 字段 | 含义 | 类型 | 说明 |
|------|------|------|------|
| `width` | 图片宽度 | number | 像素单位 |
| `height` | 图片高度 | number | 像素单位 |
| `depth` | 图片深度 | number | 颜色深度 |

## 图形坐标说明

### 矩形 (ExtentPolygon)

二维数组，顺时针存储点坐标，闭合图形长度为5。

```json
{
  "geometry": {
    "type": "ExtentPolygon",
    "coordinates": [
      [9103.3864224138, 2975.7977011494318],
      [10573.984123563223, 2975.7977011494318],
      [10573.984123563223, 4550.202298850579],
      [9103.3864224138, 4550.202298850579],
      [9103.3864224138, 2975.7977011494318]
    ]
  }
}
```

### 线 (LineString)

二维数组，按绘制顺序存储坐标。

```json
{
  "geometry": {
    "type": "LineString",
    "coordinates": [
      [13982.310560344828, 2595.17241379311],
      [13988.077610153257, 3356.422988505753],
      [13993.844659961685, 4117.673563218395],
      [13999.611709770115, 4878.924137931039],
      [13895.80481321839, 5986.197701149428]
    ],
    "lineType": "LCCLL",
    "lineMode": 1
  }
}
```

**lineType 说明：**
- **L**: 普通点坐标
- **C**: 贝塞尔曲线控制点坐标

**lineMode 说明：**
- **1**: 折线
- **2**: 竖线  
- **3**: 横线

### 平行四边形/斜矩形/梯形

三维数组，逆时针存储点坐标。

```json
{
  "type": "Parallelogram",
  "coordinates": [
    [
      [11196.825502873564, 9100.40459770115],
      [12425.207112068965, 11332.252873563219],
      [15245.294468390804, 10536.4],
      [14016.912859195403, 8304.551724137931],
      [11196.825502873564, 9100.40459770115]
    ]
  ]
}
```

**几何类型：**
- **Parallelogram**: 平行四边形
- **TiltRectangle**: 斜矩形
- **Trapezium**: 梯形

### 多边形/四边形

三维数组，按绘制顺序存储坐标，可包含环框。

```json
{
  "type": "Polygon",
  "coordinates": [
    [
      [10435.57492816092, 7474.0965517241375],
      [13376.77033045977, 6903.158620689655],
      [13322.254867410575, 7487.21018451706],
      [12800.065349616858, 7848.954789272031],
      [12511.712859195402, 8321.852873563219],
      [10435.57492816092, 7474.0965517241375]
    ],
    [
      [12061.88297413793, 7387.590804597701],
      [12096.485272988506, 7906.625287356322],
      [12338.701364942528, 8010.432183908046],
      [12753.928951149424, 7543.301149425287],
      [12061.88297413793, 7387.590804597701]
    ]
  ],
  "lineType": [
    "LLCCLL",
    "LLLLL"
  ]
}
```

**几何类型：**
- **Polygon**: 多边形
- **Square**: 四边形

## 与项目集成说明

本格式为数据堂标注平台的原始输出格式，需要通过数据转换管道处理为项目所需的格式：

1. **坐标转换**: 将复杂的几何坐标转换为标准bbox格式
2. **标签提取**: 从content.label字段提取标准化标签信息
3. **质检过滤**: 根据quality.qualityStatus过滤低质量标注
4. **格式标准化**: 转换为JSONL格式用于模型训练

相关转换脚本位于 `data_conversion/` 目录。