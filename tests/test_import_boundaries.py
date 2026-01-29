from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "ci" / "check_import_boundaries.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Import-boundary check failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )
