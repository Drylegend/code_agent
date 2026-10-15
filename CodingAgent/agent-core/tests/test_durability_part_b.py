"""Unit tests for Phase 7.8 Slim Part B — Durability.

Tests:
1. Read-after-write verification on disk in write_file and apply_patch:
   - Catches content mismatch
   - Safely rolls back to previous snapshot
   - Verifies restoration
2. Two-phase audit:
   - write_intent followed by write_committed on success
   - write_intent followed by write_failed on failure
   - Distinguishes interrupted writes
3. Startup recovery:
   - Identifies uncommitted write intents
   - Marks them interrupted
   - Cleans leftover temporary files
   - Never blindly overwrites target
   - Writes recovery audit records to audit log
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import pytest
from unittest.mock import patch

from agent_core.snapshots import SnapshotManager
from agent_core.tools import (
    AuditLogger,
    PathGuard,
    PathGuardError,
    ToolGateway,
    create_default_registry,
)
from agent_core.tools.apply_patch import ApplyPatchInput, ApplyPatchTool, EditSpec
from agent_core.tools.base import ToolContext
from agent_core.tools.write_file import WriteFileInput, WriteFileTool


@pytest.mark.anyio
async def test_01_write_file_two_phase_audit_committed_flow(tmp_path: Path):
    """Proves write_file logs write_intent and write_committed in intent journal."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    audit_file = tmp_path / "audit.jsonl"
    store_dir = tmp_path / "store"

    audit_logger = AuditLogger(audit_file)
    snap_mgr = SnapshotManager(store_dir=store_dir)
    write_tool = WriteFileTool(snapshot_manager=snap_mgr, audit_logger=audit_logger)

    ctx = ToolContext(session_id="sess-b1", workspace_root=workspace, audit_logger=audit_logger)
    res = await write_tool.execute(
        WriteFileInput(path="sample.txt", content="Initial durable content\n"),
        ctx,
    )
    assert res.success is True

    # Check intent log
    assert audit_logger.intent_log_path.exists()
    intent_lines = [
        json.loads(line)
        for line in audit_logger.intent_log_path.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    assert len(intent_lines) == 2
    assert intent_lines[0]["phase"] == "write_intent"
    assert intent_lines[0]["path"] == "sample.txt"
    assert intent_lines[1]["phase"] == "write_committed"
    assert intent_lines[1]["intent_id"] == intent_lines[0]["intent_id"]


@pytest.mark.anyio
async def test_02_write_file_read_after_write_mismatch_triggers_rollback(tmp_path: Path):
    """Proves read-after-write mismatch triggers safe rollback to previous snapshot."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "victim.txt"
    target.write_bytes(b"Original safe text\n")
    before_sha = hashlib.sha256(b"Original safe text\n").hexdigest()

    audit_file = tmp_path / "audit.jsonl"
    store_dir = tmp_path / "store"
    audit_logger = AuditLogger(audit_file)
    snap_mgr = SnapshotManager(store_dir=store_dir)
    write_tool = WriteFileTool(snapshot_manager=snap_mgr, audit_logger=audit_logger)

    ctx = ToolContext(session_id="sess-b2", workspace_root=workspace, audit_logger=audit_logger)

    # Hook os.replace to corrupt the file on disk immediately after replacement
    real_replace = os.replace

    def corrupting_replace(src, dst):
        real_replace(src, dst)
        Path(dst).write_bytes(b"Corrupted disk bytes mismatching expected SHA\n")

    with patch("os.replace", side_effect=corrupting_replace):
        res = await write_tool.execute(
            WriteFileInput(path="victim.txt", content="New content to replace\n", overwrite=True),
            ctx,
        )

    assert res.success is False
    assert "Read-after-write verification failed" in (res.error or "")
    assert "Rollback to previous snapshot verified" in (res.error or "")

    # Target file must have been rolled back to original text
    assert target.read_text(encoding="utf-8") == "Original safe text\n"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before_sha

    # Check intent log contains write_failed
    intent_lines = [
        json.loads(line)
        for line in audit_logger.intent_log_path.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    assert len(intent_lines) == 2
    assert intent_lines[0]["phase"] == "write_intent"
    assert intent_lines[1]["phase"] == "write_failed"
    assert "mismatch" in intent_lines[1]["error"]


@pytest.mark.anyio
async def test_03_apply_patch_two_phase_audit_committed_flow(tmp_path: Path):
    """Proves apply_patch logs write_intent and write_committed in intent journal."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "code.py"
    target.write_text("def hello():\n    return 42\n", encoding="utf-8")

    audit_file = tmp_path / "audit.jsonl"
    store_dir = tmp_path / "store"
    audit_logger = AuditLogger(audit_file)
    snap_mgr = SnapshotManager(store_dir=store_dir)
    patch_tool = ApplyPatchTool(snapshot_manager=snap_mgr, audit_logger=audit_logger)

    ctx = ToolContext(session_id="sess-b3", workspace_root=workspace, audit_logger=audit_logger)
    res = await patch_tool.execute(
        ApplyPatchInput(
            path="code.py",
            edits=[EditSpec(old_text="42", new_text="100")],
        ),
        ctx,
    )
    assert res.success is True

    intent_lines = [
        json.loads(line)
        for line in audit_logger.intent_log_path.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    assert len(intent_lines) == 2
    assert intent_lines[0]["phase"] == "write_intent"
    assert intent_lines[1]["phase"] == "write_committed"


@pytest.mark.anyio
async def test_04_apply_patch_read_after_write_mismatch_rollback(tmp_path: Path):
    """Proves apply_patch rolls back on read-after-write mismatch."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "math.py"
    original_code = "x = 1\ny = 2\n"
    target.write_text(original_code, encoding="utf-8")

    audit_file = tmp_path / "audit.jsonl"
    store_dir = tmp_path / "store"
    audit_logger = AuditLogger(audit_file)
    snap_mgr = SnapshotManager(store_dir=store_dir)
    patch_tool = ApplyPatchTool(snapshot_manager=snap_mgr, audit_logger=audit_logger)

    ctx = ToolContext(session_id="sess-b4", workspace_root=workspace, audit_logger=audit_logger)

    real_replace = os.replace

    def corrupting_replace(src, dst):
        real_replace(src, dst)
        Path(dst).write_bytes(b"corrupted disk bytes\n")

    with patch("os.replace", side_effect=corrupting_replace):
        res = await patch_tool.execute(
            ApplyPatchInput(
                path="math.py",
                edits=[EditSpec(old_text="y = 2", new_text="y = 3")],
            ),
            ctx,
        )

    assert res.success is False
    assert "Read-after-write verification failed" in (res.error or "")
    # Original file restored
    assert target.read_text(encoding="utf-8") == original_code



def test_05_startup_recovery_marks_interrupted_cleans_temp_and_preserves_target(tmp_path: Path):
    """Proves recover_uncommitted_intents marks interrupted writes, cleans temp files, never overwrites target."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target_file = workspace / "important.txt"
    target_file.write_text("Safe untouched target\n", encoding="utf-8")

    # Simulate leftover temp file from an interrupted write
    temp_file = workspace / ".tmp_write_orphan.bin"
    temp_file.write_text("Uncommitted leftover bytes\n", encoding="utf-8")
    assert temp_file.exists()

    audit_file = tmp_path / "audit.jsonl"
    audit_logger = AuditLogger(audit_file)

    # Manually log an uncommitted write_intent to simulate process killed before replace/committed
    intent_id = "intent_crash_test_123"
    audit_logger.log_write_intent(
        session_id="sess-crash",
        tool="write_file",
        intent_id=intent_id,
        path="important.txt",
        temp_file=str(temp_file),
        before_sha256="dummy_before",
        after_sha256="dummy_after",
        snapshot_seq=1,
        workspace_root=workspace,
    )

    # Also log a completed intent to verify it is NOT marked interrupted
    done_id = "intent_normal_456"
    audit_logger.log_write_intent(
        session_id="sess-crash",
        tool="write_file",
        intent_id=done_id,
        path="other.txt",
        temp_file=None,
        before_sha256=None,
        after_sha256=None,
        snapshot_seq=2,
        workspace_root=workspace,
    )
    audit_logger.log_write_committed(
        session_id="sess-crash",
        tool="write_file",
        intent_id=done_id,
        path="other.txt",
        after_sha256=None,
        snapshot_seq=2,
    )

    # Execute startup recovery
    recovered = audit_logger.recover_uncommitted_intents(workspace_root=workspace)

    assert len(recovered) == 1
    assert recovered[0]["intent_id"] == intent_id
    assert recovered[0]["phase"] == "write_interrupted"
    assert recovered[0]["outcome"] == "interrupted"

    # Leftover temporary file must have been cleaned up
    assert not temp_file.exists()

    # Target file was NEVER overwritten
    assert target_file.read_text(encoding="utf-8") == "Safe untouched target\n"

    # Recovery audit record exists in the main audit log
    main_lines = [
        json.loads(line)
        for line in audit_file.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    assert len(main_lines) == 1
    assert main_lines[0]["tool"] == "recovery"
    assert main_lines[0]["phase"] == "write_interrupted"
    assert main_lines[0]["outcome"] == "interrupted"


@pytest.mark.anyio
async def test_06_cancellation_shielding_completes_write_and_commit(tmp_path: Path):
    """Proves that external task cancellation (e.g. Stop button) does NOT cut off

    the replace + verify + commit sequence; execution is shielded so that the write
    and two-phase commit complete cleanly before CancelledError is propagated.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target_file = workspace / "shielded.txt"

    audit_file = tmp_path / "audit.jsonl"
    store_dir = tmp_path / "store"
    audit_logger = AuditLogger(audit_file)
    snap_mgr = SnapshotManager(store_dir=store_dir)
    write_tool = WriteFileTool(snapshot_manager=snap_mgr, audit_logger=audit_logger)

    registry = create_default_registry(include_write_tools=False)
    registry.register(write_tool)

    guard = PathGuard(workspace)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace,
        path_guard=guard,
    )

    real_replace = os.replace
    task = None

    def cancelling_replace(src, dst):
        if task is not None:
            task.cancel()
        real_replace(src, dst)

    with patch("os.replace", side_effect=cancelling_replace):
        task = asyncio.create_task(
            gateway.invoke(
                session_id="sess-cancel-shield",
                tool_name="write_file",
                raw_args={"path": "shielded.txt", "content": "Shielded durable write content\n"},
            )
        )

        # The outer task must report CancelledError
        with pytest.raises(asyncio.CancelledError):
            await task

    # File on disk MUST be completely written and verified
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == "Shielded durable write content\n"

    # Intent log must have BOTH write_intent and write_committed (never cut off or dangling)
    assert audit_logger.intent_log_path.exists()
    intent_lines = [
        json.loads(line)
        for line in audit_logger.intent_log_path.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    assert len(intent_lines) == 2
    assert intent_lines[0]["phase"] == "write_intent"
    assert intent_lines[1]["phase"] == "write_committed"
    assert intent_lines[1]["intent_id"] == intent_lines[0]["intent_id"]


def test_07_recovery_does_not_delete_temp_file_outside_workspace_and_no_glob(tmp_path: Path):
    """Proves recover_uncommitted_intents strictly validates temp_file inside workspace_root

    and never deletes paths outside workspace or globs unrelated directories.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_temp = outside_dir / ".tmp_write_forbidden.bin"
    outside_temp.write_text("Outside secret temp\n", encoding="utf-8")

    # Unrelated temp in workspace not recorded in intent log
    unrelated_temp = workspace / ".tmp_write_unrelated.bin"
    unrelated_temp.write_text("Unrelated temp file\n", encoding="utf-8")

    audit_file = tmp_path / "audit.jsonl"
    audit_logger = AuditLogger(audit_file)

    # Intent records outside temp file
    audit_logger.log_write_intent(
        session_id="sess-rec-sec",
        tool="write_file",
        intent_id="intent_escape_attempt",
        path="test.txt",
        temp_file=str(outside_temp),
        before_sha256="dummy",
        after_sha256="dummy",
        snapshot_seq=1,
        workspace_root=workspace,
    )

    # Run recovery
    recovered = audit_logger.recover_uncommitted_intents(workspace_root=workspace)
    assert len(recovered) == 1

    # Outside file MUST NOT be deleted (failed relative_to workspace check)
    assert outside_temp.exists()

    # Unrelated temp MUST NOT be deleted (no globbing)
    assert unrelated_temp.exists()


def test_08_pathguard_blocks_access_to_audit_intents(tmp_path: Path):
    """Proves PathGuard strictly blocks any tool attempt to access audit_intents.jsonl in agent-store."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    guard = PathGuard(workspace)
    audit_logger = AuditLogger(tmp_path / "agent-store" / "audit" / "audit.jsonl")

    # audit_intents.jsonl is outside workspace
    intent_path = audit_logger.intent_log_path
    assert not intent_path.is_relative_to(workspace)

    # Validation must raise PathGuardError
    with pytest.raises(PathGuardError, match=r"resolves outside workspace root"):
        guard.validate_path(str(intent_path))

    with pytest.raises(PathGuardError, match=r"resolves outside workspace root"):
        guard.validate_write_path(str(intent_path))

