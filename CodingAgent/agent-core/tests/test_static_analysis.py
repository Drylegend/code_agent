"""Static analysis guard: runs ruff F821 (undefined names) over agent_core.

This test ensures that import errors like the missing `import json` in gateway.py
are caught at test time, not in production.
"""

import subprocess
import sys
from pathlib import Path

import pytest


AGENT_CORE_DIR = Path(__file__).resolve().parent.parent / "agent_core"


def test_ruff_f821_no_undefined_names():
    """Fail the test suite if any undefined name (F821) exists in agent_core."""
    ruff_exe = Path(sys.executable).parent / "ruff"
    if not ruff_exe.exists() and not ruff_exe.with_suffix(".exe").exists():
        pytest.skip("ruff is not installed in the current environment")

    result = subprocess.run(
        [str(ruff_exe), "check", "--select", "F821", str(AGENT_CORE_DIR)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        pytest.fail(
            f"ruff F821 (undefined names) found errors in agent_core:\n{result.stdout}\n{result.stderr}"
        )
