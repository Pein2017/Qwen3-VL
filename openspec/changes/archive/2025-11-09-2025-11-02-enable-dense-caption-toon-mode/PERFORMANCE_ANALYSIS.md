# TOON Mode Performance Analysis

## Executive Summary

Training performance in TOON mode is significantly worse than JSON format. Analysis reveals **10 implementation and design issues**, with **3 critical problems** that likely explain the poor performance:

1. **❌ CRITICAL: Prompt ambiguity** - System prompt mentions both `norm100/norm1000`, confusing the model about which normalization to use
2. **❌ CRITICAL: No concrete examples** - Complex novel format has no example in prompt for model to follow
3. **❌ CRITICAL: Format unfamiliarity** - CSV-style format less represented in Chinese LLM pre-training vs JSON

## Token Efficiency Comparison

The TOON format **does** achieve the intended token reduction:

```
JSON:  250 characters (~62 tokens) - 20% structural overhead
TOON:  143 characters (~35 tokens) - 42.8% reduction ✓
```

**However**, token efficiency doesn't translate to training performance if the model can't learn the format effectively.

---

## Detailed Issue Analysis

### 1. Prompt Instruction Issues (CRITICAL)

#### Issue 1.1: Ambiguous Normalization Specification
**Location**: `src/config/prompts.py:44`

```python
# CURRENT (WRONG):
"3) 坐标使用 norm100/norm1000 整数（由 emit_norm 决定）：..."
```

**Problem**: 
- Mentions BOTH `norm100` and `norm1000` in same instruction
- Model doesn't know which one to use
- `emit_norm` is a config variable the model never sees
- This creates ambiguity: should output be 0-100 or 0-1000 range?

**Fix**:
```python
# SHOULD BE:
"3) 坐标使用 norm1000 整数（范围 0..1000）：..."
```

#### Issue 1.2: No Concrete Example
**Problem**:
- TOON is a novel, custom format not seen in pre-training
- System prompt has detailed rules but NO example output
- JSON format benefits from being familiar from pre-training
- Without examples, model must infer format from description alone

**Fix**: Add example section to system prompt:
```python
SYSTEM_PROMPT_TOON = (
    "你是图像密集标注助手。只返回一个 TOON 表格，不要 JSON 或额外文本。\n\n"
    "输出要求：\n"
    "1) 首行使用 objs[N]{type,desc,xs}: 表头，N 为对象数量...\n"
    "...\n\n"
    "示例输出（3 个对象）：\n"
    "objs[3]{type,desc,xs}:\n"
    "  0,BBU设备/华为/正常安装,100,200,300,400\n"
    "  1,标签/清晰,可见,50,100,150,100,150,150,50,150\n"
    "  2,光纤/黄色,有保护/走线规范,50,100,150,200,250,300,350,400\n\n"
    "先验规则：\n" + PRIOR_RULES
)
```

#### Issue 1.3: Type ID Memorization Burden
**Problem**:
- Arbitrary integer mapping: `0=bbox_2d, 1=quad, 2=line`
- JSON uses semantic keys (`"bbox_2d"`, `"quad"`, `"line"`) that are self-documenting
- Integer IDs add cognitive load and are error-prone

**Consideration**: Could use semantic tokens instead:
```
bbox,BBU设备/华为/正常安装,100,200,300,400
quad,标签/清晰,可见,50,100,150,100,150,150,50,150
line,光纤/黄色,有保护/走线规范,50,100,150,200,250,300
```

---

### 2. Format Design Issues

#### Issue 2.1: CSV Format Unfamiliarity
**Problem**:
- Qwen models heavily trained on JSON (Chinese web, code, structured data)
- CSV format likely underrepresented in Chinese pre-training corpus
- Format distance from pre-training distribution hurts few-shot learning

**Evidence**:
- JSON: Ubiquitous in web APIs, config files, documentation
- CSV: Less common in Chinese technical content vs English

#### Issue 2.2: Header Meta-Syntax Complexity
**Current**: `objs[N]{type,desc,xs}:`

**Problem**:
- Custom notation not seen in pre-training
- Square brackets for count: `[N]`
- Curly braces for field schema: `{type,desc,xs}`
- Colon suffix: `:`
- Model must learn this entirely novel meta-language

**Impact**: Autoregressive difficulty - must predict exact count `N` before generating objects

#### Issue 2.3: Coordinate Sequence Ambiguity
**Example**: `2,光纤,50,100,150,200,250,300,350,400`

**Problem**:
- No structural markers between coordinate pairs
- Hard to parse: Is `50,100` one point or two values?
- Especially problematic for lines with many points (up to 20+ coords)
- JSON arrays provide `[x1,y1,x2,y2]` bracket boundaries

---

### 3. Learning Dynamics Issues

#### Issue 3.1: Sparse Semantic Tokens
**JSON format**:
```json
{"object_1": {"desc": "BBU设备", "bbox_2d": [100, 200, 300, 400]}}
```
- Rich structural tokens: `{`, `}`, `[`, `]`, `"`, `:`, `,`
- Field names: `"object_1"`, `"desc"`, `"bbox_2d"`
- **50 structural characters (20% of total)** provide learning scaffolding

**TOON format**:
```
objs[1]{type,desc,xs}:
  0,BBU设备,100,200,300,400
```
- Mostly numeric tokens for coordinates
- Less semantic structure for model to latch onto
- Harder to learn structural patterns

#### Issue 3.2: Error Brittleness
- CSV: Single wrong comma breaks entire line parsing
- JSON: Errors often localized to one field/object
- TOON format more brittle during early training

#### Issue 3.3: Autoregressive Count Constraint
**TOON requires**:
1. Predict exact count `N` in header BEFORE seeing objects
2. Must emit exactly `N` rows
3. Early-binding constraint that's hard to satisfy

**JSON allows**:
- Incremental object generation
- No upfront count commitment
- More flexible autoregressive path

---

## Recommended Fixes (Priority Order)

### ✅ IMMEDIATE (Priority 1) - Fix Prompt Clarity

These changes require NO format modification, only prompt improvements:

1. **Fix normalization ambiguity** in `src/config/prompts.py`:
   ```python
   # Line 44, change:
   "3) 坐标使用 norm100/norm1000 整数（由 emit_norm 决定）：..."
   # To:
   "3) 坐标使用 norm1000 整数（范围 0..1000）：..."
   ```

2. **Add concrete example** to `SYSTEM_PROMPT_TOON`:
   ```python
   "示例输出：\n"
   "objs[3]{type,desc,xs}:\n"
   "  0,BBU设备/华为/正常安装,100,200,300,400\n"
   "  0,挡风板/已安装,符合规范,150,250,350,450\n"
   "  2,光纤/黄色,有保护/走线规范,50,100,150,200,250,300\n\n"
   ```

3. **Strengthen type ID explanation**:
   ```python
   "type 取值：0(矩形bbox)、1(四边形quad)、2(折线line)"
   ```

### 🔧 SHORT-TERM (Priority 2) - Validate Implementation

4. **Dump training samples** to verify correct serialization:
   - Set `dump_conversation_text: true` in config
   - Manually inspect first 10-20 samples
   - Verify TOON format matches specification

5. **Check for format mixing**:
   - Ensure validation set uses TOON format when training with TOON
   - Verify no JSON samples leak into TOON training

6. **Add format validation callback**:
   - Parse assistant outputs during training
   - Log format errors and malformed outputs
   - Track format correctness as a metric

### 🎯 MEDIUM-TERM (Priority 3) - Format Improvements

7. **Consider semantic type tokens** (requires retraining):
   ```
   # Instead of: 0,BBU设备,...
   # Use:        bbox,BBU设备,...
   ```

8. **Add coordinate pair grouping for lines**:
   ```
   # Instead of: 2,光纤,50,100,150,200,250,300
   # Consider:   line,光纤,50:100,150:200,250:300
   ```

9. **Simplify header format**:
   ```
   # Instead of: objs[N]{type,desc,xs}:
   # Consider:   objs[N]:  (schema is implied)
   ```

---

## Testing & Validation

### Quick Implementation Test

Run this to verify TOON serialization is working correctly:

```bash
# From repo root
conda run -n ms python << 'EOF'
import sys
import os
# Add project root to path (relative to current working directory)
_project_root = os.getcwd()
sys.path.insert(0, _project_root)

from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.datasets.builders.toon import decode_toon_payload

# Test sample
sample = {
    "images": ["test.jpg"],
    "objects": [
        {"desc": "BBU设备/华为/正常安装", "bbox": [100, 200, 300, 400]},
        {"desc": "光纤/黄色,有保护/走线规范", "line": [[50, 100], [150, 200], [250, 300]]}
    ],
    "width": 1000,
    "height": 1000
}

# Build with TOON mode
builder = JSONLinesBuilder(
    user_prompt="Test",
    system_prompt="Test",
    emit_norm="norm1000",
    toon_mode=True
)

# Process sample
result = builder.build([sample])
assistant_text = result[0]["messages"][1]["content"][0]["text"]

print("TOON Output:")
print(assistant_text)
print("\n" + "="*60)

# Verify round-trip
try:
    decoded = decode_toon_payload(assistant_text)
    print("✓ Round-trip successful")
    print("Decoded objects:", decoded.keys())
except Exception as e:
    print("✗ Round-trip FAILED:", e)
EOF
```

### Training Verification Steps

1. **Enable conversation dumps**:
   ```yaml
   # In configs/toon/stage_1_gkd.yaml
   custom:
     dump_conversation_text: true
     dump_conversation_path: output/toon_samples.txt
   ```

2. **Examine first batch**:
   ```bash
   head -n 100 output/toon_samples.txt
   ```
   Verify:
   - Header format is correct: `objs[N]{type,desc,xs}:`
   - Type IDs are 0, 1, or 2
   - Coordinates are in 0-1000 range
   - No JSON leakage

3. **Compare token distributions**:
   - Compute token frequency distribution for JSON vs TOON
   - Check if TOON has expected reduction in structural tokens

---

## Hypothesis Testing

To determine root cause, run these ablations:

### Test 1: JSON with Examples (Control)
- Keep JSON format
- Add concrete example to JSON prompt
- Expected: Slight improvement from example

### Test 2: TOON with Fixed Prompts
- Use TOON format
- Apply Priority 1 fixes (clear norm, add example)
- Expected: **Significant improvement if prompt was the issue**

### Test 3: Hybrid Format
- Keep JSON structure but use integer type IDs
- Tests if CSV format is the main problem

Expected outcome:
- If Test 2 matches JSON performance → Prompt was the issue ✓
- If Test 2 still poor → Format design is the issue
- Test 3 isolates CSV vs JSON structure impact

---

## Conclusion

**Root Cause Assessment**:

Primary suspects for poor TOON performance:
1. **Prompt ambiguity** (norm100/norm1000) - HIGH CONFIDENCE
2. **Missing examples** for novel format - HIGH CONFIDENCE  
3. **CSV format unfamiliarity** - MEDIUM CONFIDENCE

**Recommended Action**:
1. **Immediately apply Priority 1 fixes** (prompt improvements)
2. **Re-run training** and compare results
3. If still poor, consider **format redesign** (Priority 3 changes)

The token efficiency gain (42.8%) is real, but training effectiveness matters more than token count. Format must be learnable for the model to benefit.

---

## References

- Original proposal: `openspec/changes/2025-11-02-enable-dense-caption-toon-mode/proposal.md`
- Implementation: `src/datasets/builders/toon.py`, `src/config/prompts.py`
- Config: `configs/toon/stage_1_gkd.yaml`

