"""Tests for real unified diff approval preview in ToolGateway.

Covers:
  - apply_patch: real unified diff against current file
  - write_file to existing file: unified diff
  - write_file to new file: size + first lines preview
  - Preview cap at 4 KB with truncation marker
  - Fallback to JSON dump when file doesn't exist (apply_patch)
  - Integration with approval_request event
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
from agent_core.tools.gateway import (
    _cap_preview,
    _compute_apply_patch_diff,
    _compute_write_file_preview,
    MAX_PREVIEW_SIZE,
)


# --- Unit Tests for helper functions ---


class TestCapPreview:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert _cap_preview(text) == text

    def test_long_text_truncated(self):
        text = "x" * (MAX_PREVIEW_SIZE + 100)
        result = _cap_preview(text)
        assert len(result) <= MAX_PREVIEW_SIZE + 50  # marker text
        assert "preview truncated at 4 KB" in result

    def test_exact_boundary(self):
        text = "x" * MAX_PREVIEW_SIZE
        assert _cap_preview(text) == text


class TestComputeWriteFilePreview:
    def test_new_file_shows_size_and_lines(self, tmp_path):
        preview = _compute_write_file_preview("new.txt", "line1\nline2\nline3", tmp_path)
        assert "New file:" in preview
        assert "3 lines" in preview
        assert "line1" in preview

    def test_existing_file_shows_unified_diff(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content\n", encoding="utf-8")
        preview = _compute_write_file_preview("existing.txt", "new content\n", tmp_path)
        assert "---" in preview
        assert "+++" in preview
        assert "-old content" in preview
        assert "+new content" in preview

    def test_no_changes(self, tmp_path):
        f = tmp_path / "same.txt"
        f.write_text("same content", encoding="utf-8")
        preview = _compute_write_file_preview("same.txt", "same content", tmp_path)
        assert preview == "(no changes)"

    def test_large_new_file_truncated(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(500))
        preview = _compute_write_file_preview("big.txt", content, tmp_path)
        assert "New file:" in preview
        # Should show first 20 lines
        assert "line 0" in preview
        assert "more lines" in preview


class TestComputeApplyPatchDiff:
    def test_basic_patch_diff(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    return 'old'\n", encoding="utf-8")
        args = {
            "path": "code.py",
            "edits": [{"old_text": "'old'", "new_text": "'new'"}],
        }
        preview = _compute_apply_patch_diff("code.py", args, tmp_path)
        assert "---" in preview
        assert "+++" in preview
        assert "-    return 'old'" in preview
        assert "+    return 'new'" in preview

    def test_nonexistent_file_falls_back_to_json(self, tmp_path):
        args = {
            "path": "missing.py",
            "edits": [{"old_text": "x", "new_text": "y"}],
        }
        preview = _compute_apply_patch_diff("missing.py", args, tmp_path)
        parsed = json.loads(preview)
        assert isinstance(parsed, list)

    def test_no_change_patch(self, tmp_path):
        f = tmp_path / "noop.txt"
        f.write_text("abc", encoding="utf-8")
        args = {"path": "noop.txt", "edits": [{"old_text": "xyz", "new_text": "replaced"}]}
        preview = _compute_apply_patch_diff("noop.txt", args, tmp_path)
        # old_text not found, so replace does nothing -> "(no changes)"
        assert preview == "(no changes)"

    def test_replace_all_patch(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("cat dog cat dog", encoding="utf-8")
        args = {
            "path": "multi.txt",
            "edits": [{"old_text": "cat", "new_text": "CAT"}],
            "replace_all": True,
        }
        preview = _compute_apply_patch_diff("multi.txt", args, tmp_path)
        assert "+CAT dog CAT dog" in preview

    def test_large_diff_truncated(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(1000))
        f = tmp_path / "big.py"
        f.write_text(content, encoding="utf-8")
        # Replace every line marker
        args = {
            "path": "big.py",
            "edits": [{"old_text": "line", "new_text": "modified_line"}],
            "replace_all": True,
        }
        preview = _compute_apply_patch_diff("big.py", args, tmp_path)
        if len(content) > MAX_PREVIEW_SIZE:
            assert "preview truncated" in preview


# --- Integration: approval_request carries real diff ---


class MockWsConnection:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    def get_messages_of_type(self, msg_type: str) -> List[Dict[str, Any]]:
        return [m for m in self.sent if m.get("type") == msg_type]


class FakeChoice:
    def __init__(self, message):
        self.message = message

class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

class FakeResponse:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [FakeChoice(FakeMessage(content=content, tool_calls=tool_calls))]
        self.usage = None

class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = type("Fn", (), {"name": name, "arguments": arguments})()


@pytest.fixture
def env(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = tmp_path / "store"
    store.mkdir()
    audit_file = store / "audit.jsonl"
    audit_logger = AuditLogger(log_path=str(audit_file))
    snap_mgr = SnapshotManager(store_dir=store)
    return {
        "workspace": workspace,
        "store": store,
        "audit_logger": audit_logger,
        "snap_mgr": snap_mgr,
    }


def _make_stack(env, default_timeout=5.0):
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
    model_cfg = ModelConfig(alias="test", model="ollama/qwen2.5:3b")
    return approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg


@pytest.mark.anyio
async def test_apply_patch_approval_carries_real_diff(env):
    """The approval_request event for apply_patch should carry a real unified diff."""
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "file.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    session = Session(session_id="sess-diff", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(2)
    session.add_user_message("patch file.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="patching",
                tool_calls=[FakeToolCall("c1", "apply_patch", json.dumps({
                    "path": str(target), "edits": [{"old_text": "beta", "new_text": "BETA"}]
                }))],
            )
        return FakeResponse(content="done")

    async def client():
        for _ in range(50):
            await asyncio.sleep(0.02)
            reqs = mock_ws.get_messages_of_type("approval_request")
            if reqs:
                req = reqs[0]
                diff = req.get("diff_preview", "")
                # Must be a real diff, not JSON
                assert "---" in diff, f"Expected unified diff header, got: {diff[:200]}"
                assert "+++" in diff
                assert "-beta" in diff
                assert "+BETA" in diff
                await approval_mgr.submit_response(req["request_id"], "approve", session.session_id)
                return
        raise TimeoutError("no approval_request")

    task = asyncio.create_task(client())
    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session, model_config=model_cfg, request_id="r1",
            conn_wrapper=conn_wrapper, logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )
    await task
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


@pytest.mark.anyio
async def test_write_file_new_approval_shows_size_preview(env):
    """write_file to a new file should show size + first lines in diff_preview."""
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "new.txt"

    session = Session(session_id="sess-wfn", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(2)
    session.add_user_message("create new.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="writing",
                tool_calls=[FakeToolCall("c2", "write_file", json.dumps({
                    "path": str(target), "content": "hello world\nline 2\n"
                }))],
            )
        return FakeResponse(content="done")

    async def client():
        for _ in range(50):
            await asyncio.sleep(0.02)
            reqs = mock_ws.get_messages_of_type("approval_request")
            if reqs:
                req = reqs[0]
                diff = req.get("diff_preview", "")
                assert "New file:" in diff
                assert "hello world" in diff
                await approval_mgr.submit_response(req["request_id"], "approve", session.session_id)
                return
        raise TimeoutError("no approval_request")

    task = asyncio.create_task(client())
    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session, model_config=model_cfg, request_id="r2",
            conn_wrapper=conn_wrapper, logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )
    await task


@pytest.mark.anyio
async def test_write_file_existing_approval_shows_diff(env):
    """write_file overwriting an existing file should show a unified diff."""
    approval_mgr, gateway, mock_ws, conn_wrapper, model_cfg = _make_stack(env)
    target = env["workspace"] / "over.txt"
    target.write_text("original content\n", encoding="utf-8")

    session = Session(session_id="sess-wfo", workspace_root=str(env["workspace"]))
    session.set_autonomy_level(2)
    session.add_user_message("overwrite over.txt")

    step = 0
    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return FakeResponse(
                content="overwriting",
                tool_calls=[FakeToolCall("c3", "write_file", json.dumps({
                    "path": str(target), "content": "new content\n", "overwrite": True,
                }))],
            )
        return FakeResponse(content="done")

    async def client():
        for _ in range(50):
            await asyncio.sleep(0.02)
            reqs = mock_ws.get_messages_of_type("approval_request")
            if reqs:
                req = reqs[0]
                diff = req.get("diff_preview", "")
                assert "---" in diff
                assert "-original content" in diff
                assert "+new content" in diff
                await approval_mgr.submit_response(req["request_id"], "approve", session.session_id)
                return
        raise TimeoutError("no approval_request")

    task = asyncio.create_task(client())
    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session, model_config=model_cfg, request_id="r3",
            conn_wrapper=conn_wrapper, logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )
    await task
