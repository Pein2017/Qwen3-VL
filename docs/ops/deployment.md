# Deployment API Contract

Status: Active
Scope: Runtime deployment contract for `deployment/inference.py::predict()` and server expectations.
Owners: Ops + Runtime
Last updated: 2026-01-02
Related: [deployment/inference.py](../../deployment/inference.py), [deployment/server.py](../../deployment/server.py), [runtime/STAGE_A_STAGE_B.md](../runtime/STAGE_A_STAGE_B.md)

## Standardized deployment API: `inference.py::predict()`

### Function name

* Must be exactly: **`predict`**

### Call convention (inputs)

`server.py` will call it with **keyword arguments**. Your function must accept at least:

```python
def predict(
    images,                    # List[PIL.Image.Image]
    device=None,               # optional (torch.device / str / None)
    model_registry=None,       # optional dict of preloaded models
    task_configs=None,         # optional dict of task configs
    extra_info=None,           # optional dict or str
    **kwargs,                  # must tolerate extra fields
) -> dict:
    ...
```

**Minimum requirement:** it must accept `images` and not crash if the others are passed.

### `images`

* Type: `List[PIL.Image.Image]`
* Semantics: one request can contain **multiple images**; output must align per image.

---

## Required return schema (outputs)

`predict()` must return a **dict** with at least these keys:

```python
{
  "identify_result": identify_result,   # REQUIRED
  "if_total_pass": if_total_pass,       # REQUIRED
  "annotated_images": annotated_images  # OPTIONAL
}
```

### `identify_result` (required)

* Type: **list**
* Shape: **one entry per input image**
* Each image entry: a list of subtask results
* Each subtask result: a 3-item list/tuple of **strings**:

```python
identify_result = [
  [  # image 0
    [task_name, status, need_check_msg],
    [task_name, status, need_check_msg],
  ],
  [  # image 1
    [task_name, status, need_check_msg],
  ],
]
```

Where:

* `task_name`: arbitrary human-readable string (your check name)
* `status`: must be exactly `"Pass"` or `"Fail"`
* `need_check_msg`: must be exactly one of:

  * `"该子问题需要判断"`
  * `"该子问题不需要判断"`

### `if_total_pass` (required)

* Type: `bool`
* Repo-wide implied semantics:

  * `False` if **any** subtask anywhere is `["*", "Fail", "该子问题需要判断"]`
  * otherwise `True`

### `annotated_images` (optional)

Must be **JSON-serializable** if present.

Safest choice:

* `None`, **or**
* `List[str]` base64-encoded images, **same length as `images`** (positional alignment)

---

## Error behavior (what’s expected)

* If `images` is empty, many tasks raise `ValueError("输入图片列表为空")`.
* In general: **raise exceptions** on invalid inputs; standardized `server.py` catches and wraps errors into its standard error JSON.
* Don’t return alternative shapes on error.

---

## Minimal compliance checklist (for wrapping your model)

1. Accept `images` (list of PIL images)
2. Return a `dict`
3. Always include:

   * `identify_result` as a list aligned to `len(images)`
   * `if_total_pass` as a bool
4. Optional `annotated_images`: `None` or list of base64 strings aligned to `images`

---

## Tiny reference template (structure only)

```python
def predict(images, device=None, model_registry=None, task_configs=None, extra_info=None, **kwargs):
    if not images:
        raise ValueError("输入图片列表为空")

    identify_result = []
    any_fail_need_check = False

    for img in images:
        # run your model here, produce per-image subtasks
        per_image = []

        # example subtask output
        task_name = "MyCheck"
        status = "Pass"   # or "Fail"
        need_check_msg = "该子问题需要判断"  # or 不需要判断
        per_image.append([task_name, status, need_check_msg])

        if status == "Fail" and need_check_msg == "该子问题需要判断":
            any_fail_need_check = True

        identify_result.append(per_image)

    return {
        "identify_result": identify_result,
        "if_total_pass": (not any_fail_need_check),
        "annotated_images": None,
    }
```
