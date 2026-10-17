"""Tests for eval harness v0 execution and verification."""

from __future__ import annotations

import pytest

from evals.runner import run_single_task
from evals.tasks import EVAL_TASKS


@pytest.mark.anyio
@pytest.mark.parametrize("task", EVAL_TASKS, ids=[t.task_id for t in EVAL_TASKS])
async def test_eval_task_scripted_passes(task):
    """Every eval task in EVAL_TASKS must pass deterministically in scripted mode."""
    record = await run_single_task(task, mode="scripted")
    assert record["success"] is True, f"Task {task.task_id} failed: {record['reason']}"
    assert record["unintended_file_changes"] == []
    expected_errors = 1 if task.task_id == "task_6_l2_reject_retry" else 0
    assert record["tool_errors"] == expected_errors
