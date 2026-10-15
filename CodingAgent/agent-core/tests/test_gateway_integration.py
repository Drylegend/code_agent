"""Regression tests: end-to-end tool execution through the REAL path.

Exercises the REAL path:
  WebSocket client -> session -> tool loop -> approval gate -> write
Using a scripted fake LLM emitting write_file and apply_patch.

Covers:
  - L2 (Ask): approve, reject, timeout for write_file
  - L2 (Ask): approve, reject, timeout for apply_patch (the exact path that hit the NameError)
  - L3 (Scoped): auto-approve for write_file
  - L3 (Scoped): auto-approve for apply_patch
  - Internal tool exceptions: full traceback logged, audit entry recorded with exception_type,
    user receives clear error in UI, model gets generic error without internals.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from agent_core.config import ModelConfig
from agent_core.server import ConnectionWrapper, run_chat_generation
from agent_core.sessions import Session
from agent_core.snapshots import SnapshotManager
from agent_core.tools import (
    ApprovalManager,
    AuditLogger,
    PathGuard,
    ToolGateway,
    create_default_registry,
)


# --- Test Helpers & Fakes ---

class MockWsConnection:
    """Mock WebSocket connection capturing all outbound JSON messages."""

    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    def get_messages_of_type(self, msg_type: str) -> List[Dict[str, Any]]:
        return [m for m in self.sent if m.get("type") == msg_type]


class FakeChoice:
    def __init__(self, message: Any) -> None:
        self.message = message


class FakeMessage:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class FakeResponse:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None) -> None:
        self.choices = [FakeChoice(FakeMessage(content=content, tool_calls=tool_calls))]
        self.usage = None


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = type("Function", (), {"name": name, "arguments": arguments})()


@pytest.fixture
def test_env(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    store = tmp_path / "store"
    store.mkdir(parents=True, exist_ok=True)
    audit_file = store / "audit.jsonl"
    audit_logger = AuditLogger(log_path=str(audit_file))
    snap_mgr = SnapshotManager(store_dir=store)

    return {
        "workspace": workspace,
        "store": store,
        "audit_file": audit_file,
        "audit_logger": audit_logger,
        "snap_mgr": snap_mgr,
    }


def _make_stack(env: dict, default_timeout: float = 5.0):
    approval_mgr = ApprovalManager(audit_logger=env["audit_logger"], default_timeout=default_timeout)
    registry = create_default_registry(include_write_tools=True, snapshot_manager=env["snap_mgr"])
    gateway = ToolGateway(
        registry=registry,
        audit_logger=env["audit_logger"],
        workspace_root=env["workspace"],
        path_guard=PathGuard(env["workspace"]),
        approval_manager=approval_mgr,
    )
    mock_ws = MockWsConnection()
    conn_wrapper = ConnectionWrapper(mock_ws)
    approval_mgr.set_event_sink(lambda payload: conn_wrapper.send(json.dumps(payload)))
    model_cfg = ModelConfig(
        alias="test-coder",
        model="ollama/qwen2.5-coder:7b",
        tool_mode="native",
        temperature=0.0,
        max_tokens=500,
    )
    return approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg


# ── write_file L2 Tests ───────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_real_path_write_file_l2_approve(test_env):
    """Real path: WebSocket client -> session -> tool loop -> approval -> write (L2 Approve)."""
    env = test_env
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "hello.txt"

    session = Session(session_id="sess-wf-app", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(2)
    session.add_user_message("Create hello.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="Writing file",
                tool_calls=[
                    FakeToolCall("call_wf_1", "write_file", json.dumps({"path": str(target), "content": "Hello World"}))
                ],
            )
        return FakeResponse(content="Done.")

    async def client_responder():
        for _ in range(50):
            await asyncio.sleep(0.02)
            reqs = mock_ws.get_messages_of_type("approval_request")
            if reqs:
                req = reqs[0]
                assert req["tool"] == "write_file"
                assert req["path"] == str(target)
                await approval_mgr.submit_response(req["request_id"], "approve", session.session_id)
                return
        raise TimeoutError("Approval request not received")

    client_task = asyncio.create_task(client_responder())
    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-1",
            conn_wrapper=conn_wrapper,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )
    await client_task

    assert target.exists()
    assert target.read_text(encoding="utf-8") == "Hello World"


@pytest.mark.anyio
async def test_real_path_write_file_l2_reject(test_env):
    """Real path: write_file rejected at L2 does not modify filesystem."""
    env = test_env
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "rejected.txt"

    session = Session(session_id="sess-wf-rej", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(2)
    session.add_user_message("Create rejected.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="Writing",
                tool_calls=[
                    FakeToolCall("call_wf_rej", "write_file", json.dumps({"path": str(target), "content": "Reject Me"}))
                ],
            )
        return FakeResponse(content="I see the request was rejected.")

    async def client_responder():
        for _ in range(50):
            await asyncio.sleep(0.02)
            reqs = mock_ws.get_messages_of_type("approval_request")
            if reqs:
                await approval_mgr.submit_response(reqs[0]["request_id"], "reject", session.session_id)
                return
        raise TimeoutError("Approval request not received")

    client_task = asyncio.create_task(client_responder())
    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-2",
            conn_wrapper=conn_wrapper,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )
    await client_task

    assert not target.exists()


@pytest.mark.anyio
async def test_real_path_write_file_l2_timeout(test_env):
    """Real path: write_file times out at L2 without modifying filesystem."""
    env = test_env
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env, default_timeout=0.05)
    target = env["workspace"] / "timeout.txt"

    session = Session(session_id="sess-wf-tout", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(2)
    session.add_user_message("Create timeout.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="Writing",
                tool_calls=[
                    FakeToolCall("call_wf_tout", "write_file", json.dumps({"path": str(target), "content": "Timeout"}))
                ],
            )
        return FakeResponse(content="Operation timed out.")

    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-3",
            conn_wrapper=conn_wrapper,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    assert not target.exists()


# ── write_file L3 Auto-Approve ───────────────────────────────────────────────

@pytest.mark.anyio
async def test_real_path_write_file_l3_auto_approve(test_env):
    """Real path: write_file at L3 auto-approves without emitting approval_request."""
    env = test_env
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "auto_wf.txt"

    session = Session(session_id="sess-wf-l3", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(3)
    session.add_user_message("Create auto_wf.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="Auto writing",
                tool_calls=[
                    FakeToolCall("call_wf_auto", "write_file", json.dumps({"path": str(target), "content": "Auto L3"}))
                ],
            )
        return FakeResponse(content="Done.")

    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-4",
            conn_wrapper=conn_wrapper,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    assert target.exists()
    assert target.read_text(encoding="utf-8") == "Auto L3"
    assert len(mock_ws.get_messages_of_type("approval_request")) == 0


# ── apply_patch L2 Tests (THE REGRESSION PATH) ───────────────────────────────

@pytest.mark.anyio
async def test_real_path_apply_patch_l2_approve(test_env):
    """Real path: apply_patch at L2 approval (the exact path that failed with NameError: json)."""
    env = test_env
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "source.py"
    target.write_text("def hello():\n    return 'old'\n", encoding="utf-8")

    session = Session(session_id="sess-ap-app", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(2)
    session.add_user_message("Patch source.py")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="Patching",
                tool_calls=[
                    FakeToolCall(
                        "call_ap_1",
                        "apply_patch",
                        json.dumps({
                            "path": str(target),
                            "edits": [{"old_text": "'old'", "new_text": "'new'"}],
                        }),
                    )
                ],
            )
        return FakeResponse(content="Patch applied.")

    async def client_responder():
        for _ in range(50):
            await asyncio.sleep(0.02)
            reqs = mock_ws.get_messages_of_type("approval_request")
            if reqs:
                req = reqs[0]
                assert req["tool"] == "apply_patch"
                assert req["path"] == str(target)
                assert req["diff_preview"] is not None
                await approval_mgr.submit_response(req["request_id"], "approve", session.session_id)
                return
        raise TimeoutError("Approval request not received")

    client_task = asyncio.create_task(client_responder())
    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-5",
            conn_wrapper=conn_wrapper,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )
    await client_task

    content = target.read_text(encoding="utf-8")
    assert "'new'" in content
    assert "'old'" not in content


@pytest.mark.anyio
async def test_real_path_apply_patch_l2_reject(test_env):
    """Real path: apply_patch at L2 rejected leaves file unmodified."""
    env = test_env
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "keep.txt"
    target.write_text("initial text", encoding="utf-8")

    session = Session(session_id="sess-ap-rej", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(2)
    session.add_user_message("Patch keep.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="Patching",
                tool_calls=[
                    FakeToolCall(
                        "call_ap_rej",
                        "apply_patch",
                        json.dumps({
                            "path": str(target),
                            "edits": [{"old_text": "initial", "new_text": "corrupt"}],
                        }),
                    )
                ],
            )
        return FakeResponse(content="Rejected.")

    async def client_responder():
        for _ in range(50):
            await asyncio.sleep(0.02)
            reqs = mock_ws.get_messages_of_type("approval_request")
            if reqs:
                await approval_mgr.submit_response(reqs[0]["request_id"], "reject", session.session_id)
                return
        raise TimeoutError("Approval request not received")

    client_task = asyncio.create_task(client_responder())
    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-6",
            conn_wrapper=conn_wrapper,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )
    await client_task

    assert target.read_text(encoding="utf-8") == "initial text"


@pytest.mark.anyio
async def test_real_path_apply_patch_l2_timeout(test_env):
    """Real path: apply_patch at L2 timeout leaves file unmodified."""
    env = test_env
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env, default_timeout=0.05)
    target = env["workspace"] / "tout_patch.txt"
    target.write_text("original content", encoding="utf-8")

    session = Session(session_id="sess-ap-tout", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(2)
    session.add_user_message("Patch tout_patch.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="Patching",
                tool_calls=[
                    FakeToolCall(
                        "call_ap_tout",
                        "apply_patch",
                        json.dumps({
                            "path": str(target),
                            "edits": [{"old_text": "original", "new_text": "modified"}],
                        }),
                    )
                ],
            )
        return FakeResponse(content="Timed out.")

    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-7",
            conn_wrapper=conn_wrapper,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    assert target.read_text(encoding="utf-8") == "original content"


# ── apply_patch L3 Auto-Approve ──────────────────────────────────────────────

@pytest.mark.anyio
async def test_real_path_apply_patch_l3_auto_approve(test_env):
    """Real path: apply_patch at L3 auto-approves without approval_request."""
    env = test_env
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "auto_patch.txt"
    target.write_text("alpha beta gamma", encoding="utf-8")

    session = Session(session_id="sess-ap-l3", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(3)
    session.add_user_message("Patch auto_patch.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="Auto patching",
                tool_calls=[
                    FakeToolCall(
                        "call_ap_auto",
                        "apply_patch",
                        json.dumps({
                            "path": str(target),
                            "edits": [{"old_text": "beta", "new_text": "BETA"}],
                        }),
                    )
                ],
            )
        return FakeResponse(content="Patched.")

    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-8",
            conn_wrapper=conn_wrapper,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    assert target.read_text(encoding="utf-8") == "alpha BETA gamma"
    assert len(mock_ws.get_messages_of_type("approval_request")) == 0


# ── Internal Tool Exception Handling ─────────────────────────────────────────

@pytest.mark.anyio
async def test_internal_exception_logged_audit_and_ui_error(test_env):
    """Internal tool exceptions must log full traceback, record audit entry with exception_type,
    send clear error to UI, and supply short generic error without internals to model.
    """
    env = test_env
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "error_test.txt"

    session = Session(session_id="sess-exc", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(3)
    session.add_user_message("Run tool that raises exception")

    # Monkey-patch write_file execute to raise a simulated internal error
    tool = gateway.registry.get("write_file")
    async def broken_execute(*args, **kwargs):
        raise RuntimeError("Simulated internal database connection failure")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="Writing file",
                tool_calls=[
                    FakeToolCall("call_exc", "write_file", json.dumps({"path": str(target), "content": "data"}))
                ],
            )
        return FakeResponse(content="Handled error.")

    with patch.object(tool, "execute", side_effect=broken_execute), \
         patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-exc",
            conn_wrapper=conn_wrapper,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    # 1. UI receives clear error
    tool_results = mock_ws.get_messages_of_type("tool.result")
    assert len(tool_results) == 1
    tr = tool_results[0]
    assert tr["ok"] is False
    assert "Internal error while executing 'write_file'" in tr["summary"]
    # Not just raw exception name
    assert tr["summary"] != "Tool error: RuntimeError"

    # 2. Audit log records entry with exception_type
    audit_lines = [json.loads(line) for line in env["audit_file"].read_text(encoding="utf-8").strip().splitlines()]
    exc_entries = [e for e in audit_lines if e.get("outcome") == "error"]
    assert len(exc_entries) >= 1
    last_exc = exc_entries[-1]
    exc_type = last_exc.get("exception_type") or last_exc.get("extra", {}).get("exception_type")
    assert exc_type == "RuntimeError"

    # 3. Model observation in history has generic error without internal traceback
    tool_obs = [m for m in session.messages if m.get("role") == "tool"]
    assert len(tool_obs) == 1
    content_str = tool_obs[0]["content"]
    assert "An internal error occurred while executing this tool" in content_str
    # Verify internals are not leaked to model
    assert "Simulated internal database" not in content_str
    assert "Traceback" not in content_str
