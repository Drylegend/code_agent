"""Tests for Phase 8 Part 3: Context size measurement and list_directory caps/ignores.

Verifies:
1. list_directory applies default ignores (.git, node_modules, target, __pycache__, .venv, dist, agent-store, logs).
2. list_directory prefers non-recursive by default.
3. list_directory supports recursive=True with proper directory pruning.
4. list_directory caps results at 200 by default with truncated=True.
5. Per-request token breakdown logs: system prompt, tool schemas, history, tool results.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from agent_core.config import ModelConfig
from agent_core.llm import SYSTEM_PROMPT
from agent_core.server import ConnectionWrapper, run_chat_generation
from agent_core.sessions import Session
from agent_core.tools import (
    ApprovalManager,
    AuditLogger,
    ListDirectoryInput,
    ListDirectoryTool,
    PathGuard,
    ToolContext,
    ToolGateway,
    create_default_registry,
)


@pytest.fixture
def workspace(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


# ── list_directory Ignore Tests ──────────────────────────────────────────────

@pytest.mark.anyio
async def test_list_directory_default_ignores(workspace):
    """list_directory must ignore .git, node_modules, target, __pycache__, .venv, dist, agent-store, logs."""
    ignored_names = [
        ".git",
        "node_modules",
        "target",
        "__pycache__",
        ".venv",
        "dist",
        "agent-store",
        "logs",
    ]
    for name in ignored_names:
        (workspace / name).mkdir()
        (workspace / name / "nested.txt").write_text("inner")

    # Visible items
    (workspace / "src").mkdir()
    (workspace / "package.json").write_text("{}")
    (workspace / "README.md").write_text("# Hello")

    tool = ListDirectoryTool()
    ctx = ToolContext(session_id="sess-list-1", workspace_root=workspace)
    res = await tool.execute(ListDirectoryInput(path="."), ctx)

    assert res.success is True
    entry_names = [e["name"] for e in res.output["entries"]]

    # Visible files must be present
    assert "src" in entry_names
    assert "package.json" in entry_names
    assert "README.md" in entry_names

    # None of the ignored directories should be in entries
    for ignored in ignored_names:
        assert ignored not in entry_names

    # All ignored directories should be recorded in skipped
    skipped = res.output["skipped"]
    for ignored in ignored_names:
        assert ignored in skipped


# ── list_directory Non-recursive Default & Recursive Option ──────────────────

@pytest.mark.anyio
async def test_list_directory_non_recursive_by_default(workspace):
    """list_directory must be non-recursive by default."""
    sub = workspace / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("inner")

    tool = ListDirectoryTool()
    ctx = ToolContext(session_id="sess-list-2", workspace_root=workspace)
    res = await tool.execute(ListDirectoryInput(path="."), ctx)

    assert res.success is True
    assert res.output["recursive"] is False
    entry_names = [e["name"] for e in res.output["entries"]]
    assert "subdir" in entry_names
    # nested.txt should NOT be in top-level non-recursive entries
    assert "subdir/nested.txt" not in entry_names
    assert "nested.txt" not in entry_names


@pytest.mark.anyio
async def test_list_directory_recursive_option(workspace):
    """list_directory with recursive=True must enumerate nested files while respecting ignores."""
    sub1 = workspace / "src"
    sub1.mkdir()
    (sub1 / "index.ts").write_text("console.log(1);")
    sub2 = sub1 / "components"
    sub2.mkdir()
    (sub2 / "Button.tsx").write_text("export const Button = () => null;")

    # Ignored dir inside src
    (sub1 / "__pycache__").mkdir()
    (sub1 / "__pycache__" / "cached.pyc").write_text("cache")

    tool = ListDirectoryTool()
    ctx = ToolContext(session_id="sess-list-3", workspace_root=workspace)
    res = await tool.execute(ListDirectoryInput(path=".", recursive=True), ctx)

    assert res.success is True
    assert res.output["recursive"] is True
    entry_names = [e["name"] for e in res.output["entries"]]

    assert "src" in entry_names
    assert "src/index.ts" in entry_names
    assert "src/components" in entry_names
    assert "src/components/Button.tsx" in entry_names
    assert "__pycache__" in res.output["skipped"]
    assert "src/__pycache__" not in entry_names
    assert "src/__pycache__/cached.pyc" not in entry_names


# ── list_directory 200 Cap & Truncated Flag ─────────────────────────────────

@pytest.mark.anyio
async def test_list_directory_result_cap_200(workspace):
    """list_directory caps results at 200 and sets truncated: True."""
    # Create 250 files
    for i in range(250):
        (workspace / f"file_{i:03d}.txt").write_text(f"content {i}")

    tool = ListDirectoryTool()
    ctx = ToolContext(session_id="sess-list-4", workspace_root=workspace)
    res = await tool.execute(ListDirectoryInput(path="."), ctx)

    assert res.success is True
    assert res.output["truncated"] is True
    assert len(res.output["entries"]) == 200
    assert res.output["total_found"] == 250
    assert "truncated at 200 entries" in res.output["message"]


# ── Token Breakdown Measurement ──────────────────────────────────────────────

@pytest.mark.anyio
async def test_token_breakdown_logged_fresh_and_multi_turn(workspace, caplog):
    """Measure and verify token breakdown for a fresh session and a multi-turn session."""
    registry = create_default_registry(include_write_tools=True)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=AuditLogger(str(workspace / "audit.jsonl")),
        workspace_root=workspace,
        path_guard=PathGuard(workspace),
        approval_manager=ApprovalManager(AuditLogger(str(workspace / "audit.jsonl"))),
    )

    class MockWs:
        async def send(self, data: str) -> None:
            pass

    conn = ConnectionWrapper(MockWs())
    model_cfg = ModelConfig(
        alias="test-coder",
        model="ollama/qwen2.5-coder:7b",
        tool_mode="native",
        temperature=0.0,
    )

    # 1. Fresh session (Turn 1)
    fresh_session = Session(session_id="sess-fresh", workspace_root=str(workspace))
    fresh_session.add_user_message("Hello, please list the directory.")

    async def fake_llm(**kwargs):
        return type("Resp", (), {
            "choices": [type("C", (), {
                "message": type("M", (), {"content": "Directory listing complete.", "tool_calls": None})(),
            })()],
            "usage": None,
        })()

    with caplog.at_level(logging.INFO), \
         patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=fresh_session,
            model_config=model_cfg,
            request_id="req-fresh",
            conn_wrapper=conn,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    # Verify log contains token breakdown
    breakdown_logs = [r.message for r in caplog.records if "Per-request context token breakdown" in r.message]
    assert len(breakdown_logs) >= 1
    fresh_log = breakdown_logs[0]
    assert "system_prompt=" in fresh_log
    assert "tool_schemas=" in fresh_log
    assert "history=" in fresh_log
    assert "tool_results=" in fresh_log
    print(f"\n[MEASUREMENT] Fresh session first request: {fresh_log}")

    # 2. 5-turn session simulation
    five_turn_session = Session(session_id="sess-5turn", workspace_root=str(workspace))
    for turn in range(1, 6):
        five_turn_session.add_user_message(f"Turn {turn}: inspect file {turn}")
        five_turn_session.add_assistant_message(f"Turn {turn}: checking file {turn}")
        five_turn_session.add_tool_result(
            tool_call_id=f"call_{turn}",
            name="read_file",
            content=f"Contents of file {turn} with simulated code lines...\nline 1\nline 2\nline 3\n",
        )

    with caplog.at_level(logging.INFO), \
         patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=five_turn_session,
            model_config=model_cfg,
            request_id="req-5turn",
            conn_wrapper=conn,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    breakdown_logs_all = [r.message for r in caplog.records if "Per-request context token breakdown" in r.message]
    five_turn_log = breakdown_logs_all[-1]
    assert "tool_results=" in five_turn_log
    print(f"[MEASUREMENT] 5-turn session request: {five_turn_log}")
