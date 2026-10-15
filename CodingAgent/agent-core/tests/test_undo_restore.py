"""Focused tests for Phase 7 Undo/restore requirement.

Tests:
1. Successful Undo for apply_patch (restores file to original bytes when hash is unchanged).
2. Successful Undo for write_file:
   - For an existing modified file (restores previous content).
   - For a newly created file (deletes the newly created file).
3. Refusal when the current file hash differs from snapshot after_sha256:
   - File is NOT overwritten or deleted.
   - Clear warning returned indicating file changed since snapshot.
4. Direct UI/API WebSocket wiring for Undo (edit.undo -> edit.undo.result):
   - Success path returns ok=True and summary.
   - Refusal path returns ok=False and warning/error.
   - Does NOT invoke the model or generate chat streaming.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from agent_core.protocol import (
    ClientEditUndoMessage,
    ServerEditUndoResultMessage,
    parse_client_message,
)
from agent_core.server import handle_connection
from agent_core.snapshots import SnapshotManager, restore_snapshot
from agent_core.tools.apply_patch import ApplyPatchInput, ApplyPatchTool, EditSpec as PatchEdit
from agent_core.tools.base import ToolContext
from agent_core.tools.write_file import WriteFileInput, WriteFileTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def agent_store(tmp_path: Path) -> Path:
    store = tmp_path / "agent_store"
    store.mkdir()
    return store


@pytest.fixture
def snapshot_manager(agent_store: Path) -> SnapshotManager:
    return SnapshotManager(store_dir=agent_store)


@pytest.fixture
def patch_tool(snapshot_manager: SnapshotManager) -> ApplyPatchTool:
    return ApplyPatchTool(snapshot_manager=snapshot_manager)


@pytest.fixture
def write_tool(snapshot_manager: SnapshotManager) -> WriteFileTool:
    return WriteFileTool(snapshot_manager=snapshot_manager)


# ── Test 1: Successful Undo for apply_patch with unchanged hash ─────────────

@pytest.mark.anyio
async def test_undo_apply_patch_success_when_hash_unchanged(
    workspace: Path, patch_tool: ApplyPatchTool, snapshot_manager: SnapshotManager
):
    target = workspace / "code.py"
    original_bytes = b"def add(a, b):\n    return a + b\n"
    target.write_bytes(original_bytes)

    ctx = ToolContext(workspace_root=workspace, session_id="sess-undo-patch-1")
    patch_args = ApplyPatchInput(
        path="code.py",
        edits=[PatchEdit(old_text="return a + b", new_text="return a + b  # sum")],
    )
    res = await patch_tool.execute(patch_args, ctx)
    assert res.success is True
    assert res.output.get("snapshot_seq") == 1
    after_sha256 = res.output.get("after_sha256")
    assert after_sha256 is not None
    assert hashlib.sha256(target.read_bytes()).hexdigest() == after_sha256
    assert target.read_bytes() != original_bytes

    # Undo with verify_current_hash=True
    restore_res = restore_snapshot(
        seq=1,
        session_id="sess-undo-patch-1",
        workspace_root=workspace,
        store_dir=snapshot_manager.store_dir,
        verify_current_hash=True,
    )

    assert restore_res.success is True
    assert restore_res.action == "restored"
    assert target.read_bytes() == original_bytes


# ── Test 2: Successful Undo for write_file (modified and new files) ─────────

@pytest.mark.anyio
async def test_undo_write_file_existing_and_new(
    workspace: Path, write_tool: WriteFileTool, snapshot_manager: SnapshotManager
):
    # Case A: Existing file modified
    existing = workspace / "existing.txt"
    original_existing = b"Hello V1"
    existing.write_bytes(original_existing)

    ctx = ToolContext(workspace_root=workspace, session_id="sess-undo-write")
    res1 = await write_tool.execute(
        WriteFileInput(path="existing.txt", content="Hello V2", overwrite=True),
        ctx,
    )
    assert res1.success is True
    assert res1.output.get("snapshot_seq") == 1
    assert existing.read_bytes() == b"Hello V2"

    # Restore existing file
    restore_res1 = restore_snapshot(
        seq=1,
        session_id="sess-undo-write",
        workspace_root=workspace,
        store_dir=snapshot_manager.store_dir,
        verify_current_hash=True,
    )
    assert restore_res1.success is True
    assert restore_res1.action == "restored"
    assert existing.read_bytes() == original_existing

    # Case B: Newly created file undone (deleted)
    res2 = await write_tool.execute(
        WriteFileInput(path="brand_new.txt", content="New file content"),
        ctx,
    )
    assert res2.success is True
    assert res2.output.get("snapshot_seq") == 2
    assert (workspace / "brand_new.txt").exists()

    restore_res2 = restore_snapshot(
        seq=2,
        session_id="sess-undo-write",
        workspace_root=workspace,
        store_dir=snapshot_manager.store_dir,
        verify_current_hash=True,
    )
    assert restore_res2.success is True
    assert restore_res2.action == "deleted"
    assert not (workspace / "brand_new.txt").exists()


# ── Test 3: Refusal when current file hash differs from snapshot after_sha256 ─

@pytest.mark.anyio
async def test_undo_refusal_when_current_hash_mismatch(
    workspace: Path, patch_tool: ApplyPatchTool, snapshot_manager: SnapshotManager
):
    target = workspace / "file.py"
    target.write_bytes(b"initial line\n")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-mismatch")
    res = await patch_tool.execute(
        ApplyPatchInput(path="file.py", edits=[PatchEdit(old_text="initial", new_text="modified")]),
        ctx,
    )
    assert res.success is True
    assert res.output.get("snapshot_seq") == 1
    assert target.read_bytes() == b"modified line\n"

    # External user / third-party change occurs after snapshot
    target.write_bytes(b"subsequently edited by external user or another tool\n")
    external_bytes = target.read_bytes()

    # Attempt Undo with verify_current_hash=True
    restore_res = restore_snapshot(
        seq=1,
        session_id="sess-mismatch",
        workspace_root=workspace,
        store_dir=snapshot_manager.store_dir,
        verify_current_hash=True,
    )

    # Must REFUSE, must NOT overwrite
    assert restore_res.success is False
    assert restore_res.action in ("refused", "conflict")
    assert "changed since the snapshot was created" in restore_res.error
    assert restore_res.error is not None
    # File must be untouched
    assert target.read_bytes() == external_bytes


@pytest.mark.anyio
async def test_undo_refusal_when_new_file_hash_mismatch(
    workspace: Path, write_tool: WriteFileTool, snapshot_manager: SnapshotManager
):
    ctx = ToolContext(workspace_root=workspace, session_id="sess-new-mismatch")
    res = await write_tool.execute(
        WriteFileInput(path="created.txt", content="initial content"),
        ctx,
    )
    assert res.success is True
    target = workspace / "created.txt"
    assert target.exists()

    # User alters created file before undo
    target.write_text("user added important work here", encoding="utf-8")

    restore_res = restore_snapshot(
        seq=1,
        session_id="sess-new-mismatch",
        workspace_root=workspace,
        store_dir=snapshot_manager.store_dir,
        verify_current_hash=True,
    )

    # Must REFUSE deletion because content changed
    assert restore_res.success is False
    assert restore_res.action in ("refused", "conflict")
    assert "changed since the snapshot was created" in restore_res.error
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "user added important work here"


# ── Test 4: UI/API WebSocket wiring for Undo ─────────────────────────────────

class FakeWebSocket:
    """Mock WebSocket implementing async iteration and send for server handle_connection."""

    def __init__(self, messages: List[Dict[str, Any]]):
        self.messages = [json.dumps(m) for m in messages]
        self.sent_messages: List[Dict[str, Any]] = []
        self.remote_address = ("127.0.0.1", 12345)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)

    async def send(self, data: str):
        self.sent_messages.append(json.loads(data))


@pytest.mark.anyio
async def test_websocket_undo_direct_wiring_success_and_refusal(
    workspace: Path, agent_store: Path, monkeypatch: pytest.MonkeyPatch
):
    from unittest.mock import MagicMock
    from agent_core.config import ModelConfig, ServerConfig
    from agent_core.sessions import SessionManager

    # Set AGENT_CORE_STORE_DIR to agent_store
    monkeypatch.setenv("AGENT_CORE_STORE_DIR", str(agent_store))
    from agent_core.snapshots import get_snapshot_manager
    snap_mgr = get_snapshot_manager(store_dir=agent_store)
    snap_mgr.snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Setup server dependencies
    cfg = ServerConfig(
        host="127.0.0.1",
        port=9999,
        auth_token="test-token",
        log_file=workspace / "test.log",
        models={"local-fast": ModelConfig(alias="local-fast", model="ollama/qwen2.5:3b")},
    )
    sm = SessionManager()
    logger = MagicMock()

    # 1. Prepare target file and write via WriteFileTool to create snapshot
    target = workspace / "test_api.txt"
    target.write_text("Version 1 original\n", encoding="utf-8")

    tool = WriteFileTool(snapshot_manager=snap_mgr)
    ctx = ToolContext(workspace_root=workspace, session_id="sess-ws-undo")
    res = await tool.execute(
        WriteFileInput(path="test_api.txt", content="Version 2 edited\n", overwrite=True),
        ctx,
    )
    assert res.success is True
    assert res.output["snapshot_seq"] == 1
    assert target.read_text(encoding="utf-8") == "Version 2 edited\n"

    # 2. Simulate WebSocket client sending edit.undo message
    undo_msg = {
        "type": "edit.undo",
        "payload": {
            "session_id": "sess-ws-undo",
            "seq": 1,
            "call_id": "call-12345",
            "workspace_root": str(workspace),
        },
    }

    ws = FakeWebSocket([undo_msg])
    await handle_connection(ws, cfg, sm, logger, tool_gateway=None)  # type: ignore

    # Verify outgoing message was edit.undo.result with ok=True
    undo_results = [m for m in ws.sent_messages if m.get("type") == "edit.undo.result"]
    assert len(undo_results) == 1
    undo_payload = undo_results[0]["payload"]
    assert undo_payload["ok"] is True
    assert undo_payload["seq"] == 1
    assert undo_payload["call_id"] == "call-12345"
    assert "Restored test_api.txt" in undo_payload["summary"]
    # Verify file is restored
    assert target.read_text(encoding="utf-8") == "Version 1 original\n"

    # 3. Simulate another edit followed by tampering and verify refusal over WS
    res2 = await tool.execute(
        WriteFileInput(path="test_api.txt", content="Version 3 edited\n", overwrite=True),
        ctx,
    )
    assert res2.success is True
    seq2 = res2.output["snapshot_seq"]

    # Tamper with file
    target.write_text("Tampered externally\n", encoding="utf-8")

    undo_tampered_msg = {
        "type": "edit.undo",
        "payload": {
            "session_id": "sess-ws-undo",
            "seq": seq2,
            "call_id": "call-67890",
            "workspace_root": str(workspace),
        },
    }

    ws2 = FakeWebSocket([undo_tampered_msg])
    await handle_connection(ws2, cfg, sm, logger, tool_gateway=None)  # type: ignore

    undo_results2 = [m for m in ws2.sent_messages if m.get("type") == "edit.undo.result"]
    assert len(undo_results2) == 1
    undo_payload2 = undo_results2[0]["payload"]
    assert undo_payload2["ok"] is False
    assert undo_payload2["seq"] == seq2
    assert undo_payload2["call_id"] == "call-67890"
    assert "changed since the snapshot was created" in undo_payload2["error"]
    # Ensure tampered file was not overwritten
    assert target.read_text(encoding="utf-8") == "Tampered externally\n"


@pytest.mark.anyio
async def test_changes_list_and_changes_undo_websocket(
    workspace: Path, agent_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """A4 & A5. changes.list and changes.undo WebSocket handlers."""
    from unittest.mock import MagicMock
    from agent_core.config import ModelConfig, ServerConfig
    from agent_core.sessions import SessionManager

    monkeypatch.setenv("AGENT_CORE_STORE_DIR", str(agent_store))
    from agent_core.snapshots import get_snapshot_manager
    snap_mgr = get_snapshot_manager(store_dir=agent_store)
    snap_mgr.snapshots_dir.mkdir(parents=True, exist_ok=True)

    cfg = ServerConfig(
        host="127.0.0.1",
        port=9999,
        auth_token="test-token",
        log_file=workspace / "test.log",
        models={"local-fast": ModelConfig(alias="local-fast", model="ollama/qwen2.5:3b")},
    )
    sm = SessionManager()
    logger = MagicMock()

    # 1. Create a file and edit it to produce snapshot 1
    target = workspace / "doc.txt"
    target.write_text("Line 1 original\nLine 2 original\n", encoding="utf-8")

    tool = WriteFileTool(snapshot_manager=snap_mgr)
    ctx = ToolContext(workspace_root=workspace, session_id="sess-changes-1")
    res1 = await tool.execute(
        WriteFileInput(path="doc.txt", content="Line 1 original\nLine 2 modified\n", overwrite=True),
        ctx,
    )
    assert res1.success is True
    seq1 = res1.output["snapshot_seq"]

    # 2. Query changes.list
    list_msg = {
        "type": "changes.list",
        "payload": {
            "workspace_root": str(workspace),
            "session_id": "sess-changes-1",
        },
    }
    ws = FakeWebSocket([list_msg])
    await handle_connection(ws, cfg, sm, logger, tool_gateway=None)  # type: ignore

    list_results = [m for m in ws.sent_messages if m.get("type") == "changes.list.result"]
    assert len(list_results) == 1
    changes = list_results[0]["payload"]["changes"]
    assert len(changes) == 1
    c1 = changes[0]
    assert c1["seq"] == seq1
    assert c1["path"] == "doc.txt"
    assert c1["can_undo"] is True
    assert c1["status"] == "clean"
    assert c1["is_legacy"] is False
    assert c1["tool"] == "write_file"

    # 3. Undo change 1 via changes.undo (normal clean undo)
    undo_msg = {
        "type": "changes.undo",
        "payload": {
            "session_id": "sess-changes-1",
            "seq": seq1,
            "workspace_root": str(workspace),
        },
    }
    ws_undo = FakeWebSocket([undo_msg])
    await handle_connection(ws_undo, cfg, sm, logger, tool_gateway=None)  # type: ignore

    undo_results = [m for m in ws_undo.sent_messages if m.get("type") == "changes.undo.result"]
    assert len(undo_results) == 1
    assert undo_results[0]["payload"]["ok"] is True
    assert undo_results[0]["payload"]["action"] == "restored"
    assert target.read_text(encoding="utf-8") == "Line 1 original\nLine 2 original\n"


@pytest.mark.anyio
async def test_changes_undo_conflict_refusal_and_force(
    workspace: Path, agent_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """A5. Conflict checking: refuse when modified unless force=True."""
    from unittest.mock import MagicMock
    from agent_core.config import ModelConfig, ServerConfig
    from agent_core.sessions import SessionManager

    monkeypatch.setenv("AGENT_CORE_STORE_DIR", str(agent_store))
    from agent_core.snapshots import get_snapshot_manager
    snap_mgr = get_snapshot_manager(store_dir=agent_store)

    cfg = ServerConfig(
        host="127.0.0.1",
        port=9999,
        auth_token="test-token",
        log_file=workspace / "test.log",
        models={"local-fast": ModelConfig(alias="local-fast", model="ollama/qwen2.5:3b")},
    )
    sm = SessionManager()
    logger = MagicMock()

    target = workspace / "force_test.txt"
    target.write_text("v1", encoding="utf-8")

    tool = WriteFileTool(snapshot_manager=snap_mgr)
    ctx = ToolContext(workspace_root=workspace, session_id="sess-force")
    res = await tool.execute(
        WriteFileInput(path="force_test.txt", content="v2", overwrite=True),
        ctx,
    )
    seq = res.output["snapshot_seq"]

    # External modification -> conflict
    target.write_text("v3_externally_modified", encoding="utf-8")

    # 1. Normal changes.undo without force -> conflict refusal
    undo_msg = {
        "type": "changes.undo",
        "payload": {
            "session_id": "sess-force",
            "seq": seq,
            "workspace_root": str(workspace),
            "force": False,
        },
    }
    ws1 = FakeWebSocket([undo_msg])
    await handle_connection(ws1, cfg, sm, logger, tool_gateway=None)  # type: ignore

    res1 = [m for m in ws1.sent_messages if m.get("type") == "changes.undo.result"][0]["payload"]
    assert res1["ok"] is False
    assert res1["action"] == "conflict"
    assert "changed since the snapshot was created" in res1["error"]
    assert target.read_text(encoding="utf-8") == "v3_externally_modified"

    # 2. changes.undo WITH force=True -> succeeds and restores v1
    undo_force_msg = {
        "type": "changes.undo",
        "payload": {
            "session_id": "sess-force",
            "seq": seq,
            "workspace_root": str(workspace),
            "force": True,
        },
    }
    ws2 = FakeWebSocket([undo_force_msg])
    await handle_connection(ws2, cfg, sm, logger, tool_gateway=None)  # type: ignore

    res2 = [m for m in ws2.sent_messages if m.get("type") == "changes.undo.result"][0]["payload"]
    assert res2["ok"] is True
    assert res2["action"] == "restored"
    assert target.read_text(encoding="utf-8") == "v1"


@pytest.mark.anyio
async def test_legacy_snapshot_requires_explicit_workspace(
    workspace: Path, agent_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """A1. Legacy snapshots (no workspace_root) require explicit confirmation and cannot infer workspace."""
    from unittest.mock import MagicMock
    from agent_core.config import ModelConfig, ServerConfig
    from agent_core.sessions import SessionManager

    monkeypatch.setenv("AGENT_CORE_STORE_DIR", str(agent_store))
    from agent_core.snapshots import get_snapshot_manager
    snap_mgr = get_snapshot_manager(store_dir=agent_store)

    # Manually create a legacy snapshot record with workspace_root=None
    s_dir = snap_mgr._get_session_dir("sess-legacy")
    snap_file = s_dir / "1_legacy.bin"
    snap_file.write_bytes(b"Legacy content original")
    sha = hashlib.sha256(b"Legacy content original").hexdigest()

    rec = {
        "seq": 1,
        "path": "legacy.txt",
        "existed": True,
        "before_sha256": sha,
        "after_sha256": hashlib.sha256(b"Legacy edited").hexdigest(),
        "timestamp": "2026-01-01T00:00:00Z",
        "tool": "write_file",
        "snapshot_file": "1_legacy.bin",
        "workspace_root": None,  # Legacy!
    }
    with open(s_dir / "index.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    cfg = ServerConfig(
        host="127.0.0.1",
        port=9999,
        auth_token="test-token",
        log_file=workspace / "test.log",
        models={"local-fast": ModelConfig(alias="local-fast", model="ollama/qwen2.5:3b")},
    )
    sm = SessionManager()
    logger = MagicMock()

    # Create target in workspace
    target = workspace / "legacy.txt"
    target.write_text("Legacy edited", encoding="utf-8")

    # 1. Attempt undo without explicit workspace_root in message -> refused
    msg_no_ws = {
        "type": "changes.undo",
        "payload": {
            "session_id": "sess-legacy",
            "seq": 1,
            # No workspace_root provided!
        },
    }
    ws1 = FakeWebSocket([msg_no_ws])
    await handle_connection(ws1, cfg, sm, logger, tool_gateway=None)  # type: ignore

    res1 = [m for m in ws1.sent_messages if m.get("type") == "changes.undo.result"][0]["payload"]
    assert res1["ok"] is False
    assert "Legacy snapshot requires explicit user confirmation" in res1["error"]

    # 2. Provide explicit workspace_root -> succeeds
    msg_with_ws = {
        "type": "changes.undo",
        "payload": {
            "session_id": "sess-legacy",
            "seq": 1,
            "workspace_root": str(workspace),
        },
    }
    ws2 = FakeWebSocket([msg_with_ws])
    await handle_connection(ws2, cfg, sm, logger, tool_gateway=None)  # type: ignore

    res2 = [m for m in ws2.sent_messages if m.get("type") == "changes.undo.result"][0]["payload"]
    assert res2["ok"] is True
    assert res2["action"] == "restored"
    assert target.read_bytes() == b"Legacy content original"


@pytest.mark.anyio
async def test_snapshot_corruption_refuses_restore(
    workspace: Path, agent_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """A2. Integrity check: corrupted snapshot file SHA-256 mismatch refuses restore before write."""
    monkeypatch.setenv("AGENT_CORE_STORE_DIR", str(agent_store))
    from agent_core.snapshots import get_snapshot_manager
    snap_mgr = get_snapshot_manager(store_dir=agent_store)

    target = workspace / "corrupt_test.txt"
    target.write_text("Original text before edit\n", encoding="utf-8")

    tool = WriteFileTool(snapshot_manager=snap_mgr)
    ctx = ToolContext(workspace_root=workspace, session_id="sess-corrupt")
    res = await tool.execute(
        WriteFileInput(path="corrupt_test.txt", content="New edited text\n", overwrite=True),
        ctx,
    )
    seq = res.output["snapshot_seq"]

    # Tamper with the snapshot file on disk in agent_store
    s_dir = snap_mgr._get_session_dir("sess-corrupt")
    records = snap_mgr.list_snapshots("sess-corrupt")
    snap_file = s_dir / records[-1]["snapshot_file"]
    snap_file.write_bytes(b"MALICIOUS_OR_CORRUPTED_BYTES")

    # Attempt restore
    restore_res = snap_mgr.restore(
        session_id="sess-corrupt",
        seq=seq,
        workspace_root=workspace,
        verify_current_hash=True,
    )

    assert restore_res.success is False
    assert restore_res.action == "corrupted"
    assert "Snapshot integrity verification failed" in restore_res.error
    # Target file must NOT have been modified or overwritten
    assert target.read_text(encoding="utf-8") == "New edited text\n"


@pytest.mark.anyio
async def test_restore_target_safety_pathguard_denial(
    workspace: Path, agent_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """A3. Restore target safety: restore uses the exact same security boundary as writes."""
    monkeypatch.setenv("AGENT_CORE_STORE_DIR", str(agent_store))
    from agent_core.snapshots import get_snapshot_manager
    snap_mgr = get_snapshot_manager(store_dir=agent_store)

    # Manually inject snapshot record pointing to a protected file (.env or traversal)
    s_dir = snap_mgr._get_session_dir("sess-traversal")
    snap_bin = s_dir / "1_env.bin"
    snap_bin.write_bytes(b"SECRET_KEY=123")
    sha = hashlib.sha256(b"SECRET_KEY=123").hexdigest()

    rec = {
        "seq": 1,
        "path": "../secret.env",  # Traversal!
        "existed": True,
        "before_sha256": sha,
        "after_sha256": sha,
        "timestamp": "2026-01-01T00:00:00Z",
        "tool": "write_file",
        "snapshot_file": "1_env.bin",
        "workspace_root": str(workspace),
    }
    with open(s_dir / "index.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    restore_res = snap_mgr.restore(
        session_id="sess-traversal",
        seq=1,
        workspace_root=workspace,
    )

    assert restore_res.success is False
    assert restore_res.action == "denied"
    assert "security policy" in restore_res.error.lower() or "escapes" in restore_res.error.lower()


@pytest.mark.anyio
async def test_model_cannot_invoke_undo(workspace: Path):
    """A5 Security requirement: Model must NEVER be able to invoke changes.undo."""
    from agent_core.tools import ToolGateway, create_default_registry, PathGuard, AuditLogger
    from agent_core.server import parse_text_mode_tool_calls

    registry = create_default_registry(include_write_tools=True)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=AuditLogger(workspace / ".audit.jsonl"),
        workspace_root=workspace,
        path_guard=PathGuard(workspace),
    )

    # 1. ToolGateway rejection
    res = await gateway.invoke(
        session_id="sess-model-undo",
        tool_name="changes.undo",
        raw_args={"seq": 1},
        workspace_root=workspace,
    )
    assert ("not registered" in (res.error or "").lower()) or ("unknown tool" in (res.denial_reason or "").lower())

    # 2. Text-mode parser rejection
    model_output = '{"name": "changes.undo", "arguments": {"seq": 1}}'
    parsed = parse_text_mode_tool_calls(model_output, registry)
    assert len(parsed) == 0

    model_output_2 = '{"name": "restore_snapshot", "arguments": {"seq": 1}}'
    parsed_2 = parse_text_mode_tool_calls(model_output_2, registry)
    assert len(parsed_2) == 0

