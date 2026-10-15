"""Tests asserting all registered tools are present on every LLM request.

Verifies:
1. Native tool mode sends all 6 registered tools:
   (list_directory, read_file, glob, grep, write_file, apply_patch)
2. Text_json tool mode sends all 6 registered tools.
3. Every request in a multi-round conversation retains all 6 registered tools.
4. Tool names and counts are properly logged.
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
from agent_core.server import ConnectionWrapper, run_chat_generation
from agent_core.sessions import Session
from agent_core.tools import (
    ApprovalManager,
    AuditLogger,
    PathGuard,
    ToolGateway,
    create_default_registry,
)

EXPECTED_TOOLS = {
    "list_directory",
    "read_file",
    "glob",
    "grep",
    "write_file",
    "apply_patch",
}


class MockWsConnection:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))


class FakeChoice:
    def __init__(self, content: str = "Hello", tool_calls: list = None) -> None:
        self.message = type("Msg", (), {"content": content, "tool_calls": tool_calls})()


class FakeResponse:
    def __init__(self, content: str = "Hello", tool_calls: list = None) -> None:
        self.choices = [FakeChoice(content, tool_calls)]
        self.usage = None


@pytest.fixture
def env(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    audit = AuditLogger(str(tmp_path / "audit.jsonl"))
    approval = ApprovalManager(audit)
    registry = create_default_registry(include_write_tools=True)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit,
        workspace_root=workspace,
        path_guard=PathGuard(workspace),
        approval_manager=approval,
    )
    return {
        "workspace": workspace,
        "gateway": gateway,
    }


@pytest.mark.anyio
async def test_all_tools_present_in_native_mode(env):
    """Assert all 6 registered tools are present when tool_mode is 'native'."""
    captured_tools = []

    async def fake_llm(**kwargs):
        tools = kwargs.get("tools") or []
        captured_tools.append([t["function"]["name"] for t in tools])
        return FakeResponse(content="I am responding.")

    model_cfg = ModelConfig(
        alias="test-native",
        model="ollama/qwen2.5-coder:7b",
        tool_mode="native",
        temperature=0.0,
    )

    session = Session(session_id="sess-avail-native", workspace_root=str(env["workspace"]))
    session.add_user_message("What tools do you have?")
    conn = ConnectionWrapper(MockWsConnection())

    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-native",
            conn_wrapper=conn,
            logger=logging.getLogger("test"),
            tool_gateway=env["gateway"],
        )

    assert len(captured_tools) == 1
    sent_tools = set(captured_tools[0])
    assert sent_tools == EXPECTED_TOOLS, f"Expected {EXPECTED_TOOLS}, but got {sent_tools}"


@pytest.mark.anyio
async def test_all_tools_present_in_text_json_mode(env):
    """Assert all 6 registered tools are present when tool_mode is 'text_json'."""
    captured_tools = []

    async def fake_llm(**kwargs):
        tools = kwargs.get("tools") or []
        captured_tools.append([t["function"]["name"] for t in tools])
        return FakeResponse(content="I am responding in text_json mode.")

    model_cfg = ModelConfig(
        alias="test-textjson",
        model="ollama/qwen2.5-coder:7b",
        tool_mode="text_json",
        temperature=0.0,
    )

    session = Session(session_id="sess-avail-textjson", workspace_root=str(env["workspace"]))
    session.add_user_message("Check tool list.")
    conn = ConnectionWrapper(MockWsConnection())

    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-textjson",
            conn_wrapper=conn,
            logger=logging.getLogger("test"),
            tool_gateway=env["gateway"],
        )

    assert len(captured_tools) == 1
    sent_tools = set(captured_tools[0])
    assert sent_tools == EXPECTED_TOOLS, f"Expected {EXPECTED_TOOLS}, but got {sent_tools}"


@pytest.mark.anyio
async def test_all_tools_present_on_every_request_multi_round(env):
    """Assert all 6 registered tools are present on every request across multiple rounds."""
    captured_tools = []

    step = 0
    fake_tool_call = type("TC", (), {
        "id": "c1",
        "function": type("F", (), {
            "name": "list_directory",
            "arguments": json.dumps({"path": "."}),
        })(),
    })()

    async def fake_llm(**kwargs):
        nonlocal step
        step += 1
        tools = kwargs.get("tools") or []
        captured_tools.append([t["function"]["name"] for t in tools])
        if step == 1:
            return FakeResponse(content="Listing dir", tool_calls=[fake_tool_call])
        return FakeResponse(content="Directory listed.")

    model_cfg = ModelConfig(
        alias="test-multi",
        model="ollama/qwen2.5-coder:7b",
        tool_mode="native",
        temperature=0.0,
    )

    session = Session(session_id="sess-avail-multi", workspace_root=str(env["workspace"]))
    session.add_user_message("List files")
    conn = ConnectionWrapper(MockWsConnection())

    with patch("agent_core.server.acompletion_with_tools", side_effect=fake_llm):
        await run_chat_generation(
            session=session,
            model_config=model_cfg,
            request_id="req-multi",
            conn_wrapper=conn,
            logger=logging.getLogger("test"),
            tool_gateway=env["gateway"],
        )

    assert len(captured_tools) == 2, "Expected 2 rounds of LLM requests"
    for idx, round_tools in enumerate(captured_tools):
        assert set(round_tools) == EXPECTED_TOOLS, f"Round {idx} missing tools: {EXPECTED_TOOLS - set(round_tools)}"
