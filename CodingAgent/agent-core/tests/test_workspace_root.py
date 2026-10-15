"""Unit and integration tests for Phase 6: Explicit Workspace / Project Root.

Verifies:
1. validate_workspace_root validation and resolution rules.
2. Session-level workspace root binding and multi-session isolation.
3. ToolGateway and PathGuard enforcement against explicit session workspace roots.
4. Denial of tool calls when no workspace root is configured (no cwd fallback).
5. Cross-session boundary enforcement.
6. WebSocket server handling of explicit workspace roots and INVALID_WORKSPACE_ROOT errors.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from agent_core.protocol import (
    ClientChatSendMessage,
    create_chat_send,
    parse_client_message,
)
from agent_core.server import (
    ConnectionWrapper,
    handle_connection,
    run_chat_generation,
)
from agent_core.sessions import Session, SessionManager
from agent_core.tools import (
    AuditLogger,
    PathGuard,
    ToolGateway,
    create_default_registry,
)
from agent_core.workspace import WorkspaceError, validate_workspace_root


# ── Step A: Workspace Validator Tests ─────────────────────────────────────────


def test_1_validate_workspace_root_valid_directory(tmp_path: Path):
    """Valid absolute existing directory returns canonical Path."""
    target = tmp_path / "valid_dir"
    target.mkdir()
    res = validate_workspace_root(str(target))
    assert res == target.resolve()
    assert res.is_dir()


def test_2_validate_workspace_root_rejects_relative_path():
    """Relative paths must be rejected with WorkspaceError."""
    with pytest.raises(WorkspaceError, match="must be an absolute path"):
        validate_workspace_root("relative/path/to/dir")

    with pytest.raises(WorkspaceError, match="must be an absolute path"):
        validate_workspace_root(".")


def test_3_validate_workspace_root_rejects_nonexistent_directory(tmp_path: Path):
    """Non-existent directory must be rejected."""
    nonexistent = tmp_path / "ghost_folder_123"
    with pytest.raises(WorkspaceError, match="does not exist"):
        validate_workspace_root(str(nonexistent))


def test_4_validate_workspace_root_rejects_file(tmp_path: Path):
    """Regular files must be rejected (must be a directory)."""
    sample_file = tmp_path / "file.txt"
    sample_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="must be a directory, not a file"):
        validate_workspace_root(str(sample_file))


def test_5_validate_workspace_root_rejects_empty_and_none():
    """None and whitespace-only strings must be rejected."""
    with pytest.raises(WorkspaceError, match="cannot be None"):
        validate_workspace_root(None)

    with pytest.raises(WorkspaceError, match="cannot be empty"):
        validate_workspace_root("")

    with pytest.raises(WorkspaceError, match="cannot be empty"):
        validate_workspace_root("   ")


def test_6_validate_workspace_root_does_not_categorically_reject_unc_structure(tmp_path: Path):
    """UNC format is not rejected in structure validation (fails only if non-existent)."""
    # A fake UNC path fails because it does not exist, NOT because it starts with \\
    with pytest.raises(WorkspaceError, match="does not exist"):
        validate_workspace_root(r"\\nonexistent_server\share\dir")


# ── Step B: Session Binding & Isolation Tests ─────────────────────────────────


def test_7_session_workspace_root_binding(tmp_path: Path):
    """Session stores and updates explicit workspace root."""
    ws = tmp_path / "session_ws"
    ws.mkdir()

    session = Session("sess-bind-1")
    assert session.workspace_root is None

    session.set_workspace_root(ws)
    assert session.workspace_root == ws.resolve()


def test_8_session_manager_isolation(tmp_path: Path):
    """Session A and Session B maintain isolated, distinct workspace roots."""
    dir_a = tmp_path / "project_a"
    dir_b = tmp_path / "project_b"
    dir_a.mkdir()
    dir_b.mkdir()

    sm = SessionManager()
    session_a = sm.get_or_create("sess-a", workspace_root=dir_a)
    session_b = sm.get_or_create("sess-b", workspace_root=dir_b)

    assert session_a.workspace_root == dir_a.resolve()
    assert session_b.workspace_root == dir_b.resolve()
    assert session_a.workspace_root != session_b.workspace_root

    # Calling get_or_create again without root preserves existing root
    session_a_again = sm.get_or_create("sess-a")
    assert session_a_again.workspace_root == dir_a.resolve()


# ── Step C: ToolGateway & PathGuard Multi-Root Enforcement ────────────────────


@pytest.mark.anyio
async def test_9_tool_gateway_uses_per_session_workspace_root(tmp_path: Path):
    """ToolGateway invokes tool using the explicitly supplied per-session workspace_root."""
    dir_a = tmp_path / "ws_a"
    dir_a.mkdir()
    (dir_a / "file_a.txt").write_text("Hello from A", encoding="utf-8")

    registry = create_default_registry()
    audit_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(audit_file)

    # ToolGateway created without fixed workspace_root
    gateway = ToolGateway(registry=registry, audit_logger=logger, workspace_root=None)

    # Invoke tool with dir_a as session workspace_root
    res = await gateway.invoke(
        session_id="sess-a",
        tool_name="read_file",
        raw_args={"path": "file_a.txt"},
        workspace_root=dir_a,
    )

    assert res.success is True
    assert res.authorized is True
    assert res.output["content"] == "Hello from A"


@pytest.mark.anyio
async def test_10_tool_gateway_rejects_when_no_workspace_root_configured(tmp_path: Path):
    """ToolGateway denies invocation when neither session nor gateway has a root (no cwd fallback)."""
    registry = create_default_registry()
    audit_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(audit_file)

    gateway = ToolGateway(registry=registry, audit_logger=logger, workspace_root=None)

    res = await gateway.invoke(
        session_id="sess-noroot",
        tool_name="list_directory",
        raw_args={"path": "."},
        workspace_root=None,
    )

    assert res.success is False
    assert res.authorized is False
    assert "No workspace root configured" in (res.denial_reason or "")


@pytest.mark.anyio
async def test_11_cross_session_workspace_isolation_denied(tmp_path: Path):
    """Session A cannot read files from Session B's workspace even if relative escaping."""
    dir_a = tmp_path / "ws_a"
    dir_b = tmp_path / "ws_b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_b / "secret_b.txt").write_text("Secret in B", encoding="utf-8")

    registry = create_default_registry()
    audit_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(audit_file)
    gateway = ToolGateway(registry=registry, audit_logger=logger, workspace_root=None)

    # Attempting to read B's file from A's root via path traversal
    res = await gateway.invoke(
        session_id="sess-a",
        tool_name="read_file",
        raw_args={"path": "../ws_b/secret_b.txt"},
        workspace_root=dir_a,
    )

    assert res.success is False
    assert res.authorized is False
    assert "Path authorization denied" in (res.denial_reason or "")


# ── Step D: Protocol & Server Error Handling Tests ─────────────────────────────


def test_12_protocol_chat_send_supports_workspace_root():
    """ClientChatSendMessage accepts and validates optional workspace_root."""
    msg = create_chat_send(
        session_id="sess-proto",
        text="List files",
        model_alias="local-fast",
        workspace_root=r"D:\my_workspace",
    )
    raw_json = msg.model_dump_json()
    parsed = parse_client_message(raw_json)

    assert isinstance(parsed, ClientChatSendMessage)
    assert parsed.payload.workspace_root == r"D:\my_workspace"


class FakeWebSocket:
    """Mock WebSocket implementing async iteration and send."""

    def __init__(self, messages: List[str]):
        self.messages = list(messages)
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
async def test_13_server_rejects_missing_workspace_root(tmp_path: Path):
    """Server sends INVALID_WORKSPACE_ROOT when no workspace root was provided."""
    from agent_core.config import ModelConfig, ServerConfig
    from agent_core.server import handle_connection

    cfg = ServerConfig(
        host="127.0.0.1",
        port=9999,
        auth_token="test-token",
        log_file=tmp_path / "test.log",
        models={"local-fast": ModelConfig(alias="local-fast", model="ollama/qwen2.5:3b")},
    )
    sm = SessionManager()
    logger = MagicMock()

    msg = create_chat_send(
        session_id="sess-no-ws",
        text="Hello",
        model_alias="local-fast",
        workspace_root=None,
    )

    fake_conn = FakeWebSocket([msg.model_dump_json()])
    await handle_connection(fake_conn, cfg, sm, logger, tool_gateway=None)  # type: ignore

    assert len(fake_conn.sent_messages) == 1
    err_envelope = fake_conn.sent_messages[0]
    assert err_envelope["type"] == "chat.error"
    assert err_envelope["payload"]["code"] == "INVALID_WORKSPACE_ROOT"
    assert "must be explicitly specified" in err_envelope["payload"]["message"]


@pytest.mark.anyio
async def test_14_server_rejects_invalid_workspace_root(tmp_path: Path):
    """Server sends INVALID_WORKSPACE_ROOT when client sends a nonexistent workspace root."""
    from agent_core.config import ModelConfig, ServerConfig
    from agent_core.server import handle_connection

    cfg = ServerConfig(
        host="127.0.0.1",
        port=9999,
        auth_token="test-token",
        log_file=tmp_path / "test.log",
        models={"local-fast": ModelConfig(alias="local-fast", model="ollama/qwen2.5:3b")},
    )
    sm = SessionManager()
    logger = MagicMock()

    msg = create_chat_send(
        session_id="sess-invalid-ws",
        text="Hello",
        model_alias="local-fast",
        workspace_root=str(tmp_path / "nonexistent_folder_abc"),
    )

    fake_conn = FakeWebSocket([msg.model_dump_json()])
    await handle_connection(fake_conn, cfg, sm, logger, tool_gateway=None)  # type: ignore

    assert len(fake_conn.sent_messages) == 1
    err_envelope = fake_conn.sent_messages[0]
    assert err_envelope["type"] == "chat.error"
    assert err_envelope["payload"]["code"] == "INVALID_WORKSPACE_ROOT"
    assert "does not exist" in err_envelope["payload"]["message"]

