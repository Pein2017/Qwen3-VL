from __future__ import annotations

import subprocess
import sys


def test_import_smoke_is_lightweight() -> None:
    code = (
        "import sys; "
        "import src, src.utils, src.config, src.datasets.contracts, src.generation.contracts; "
        "forbidden=['torch','transformers','swift','deepspeed','src.sft','src.stage_a.inference','src.stage_b.runner']; "
        "bad=[m for m in forbidden if m in sys.modules]; "
        "raise SystemExit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Lightweight import smoke failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )
