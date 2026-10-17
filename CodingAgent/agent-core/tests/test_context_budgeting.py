"""Tests for Phase 9 Part 1: Real Context Accounting and Budgeting.

Verifies:
1. ModelConfig validation of num_ctx and usable_budget (usable_budget <= num_ctx <= context_window).
2. models.yaml contains explicit num_ctx and usable_budget for local-fast and local-coder.
3. num_ctx is passed in litellm requests (both acompletion_with_tools and stream_chat_completion).
4. Tool results fed to the model are capped with a truncation marker and query narrowing hint,
   while full results are sent to UI and audit.
5. Ground truth token counts from Ollama (prompt_eval_count / eval_count) are extracted and logged.
6. Prompt truncation is detected when tokens reach or exceed num_ctx, emitting a warning and UI event.
7. Context usage warnings fire at 40% ("consider compacting") and 60% ("warning").
8. Context status events and done message payloads carry complete context accounting metadata.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from agent_core.config import ModelConfig, load_models_config
from agent_core.llm import acompletion_with_tools, stream_chat_completion
from agent_core.protocol import (
    ContextStatusPayload,
    ServerContextStatusMessage,
    TokenUsage,
    create_context_status,
)
from agent_core.server import (
    ConnectionWrapper,
    MAX_TOOL_RESULT_FOR_MODEL_CHARS,
    format_tool_result_for_model,
    run_chat_generation,
)
from agent_core.sessions import Session
from agent_core.tools import GatewayInvocationResult, ToolGateway, create_default_registry


# ── 1. ModelConfig & YAML Configuration Tests ─────────────────────────────────

def test_model_config_validates_budget_and_ctx():
    """usable_budget <= num_ctx <= context_window must be enforced."""
    # Valid config
    cfg = ModelConfig(
        alias="fast",
        model="ollama/qwen2.5:3b",
        context_window=131072,
        num_ctx=8192,
        usable_budget=6144,
    )
    assert cfg.num_ctx == 8192
    assert cfg.usable_budget == 6144

    # num_ctx > context_window raises ValidationError
    with pytest.raises(ValidationError, match="cannot exceed model context_window"):
        ModelConfig(
            alias="fast",
            model="ollama/qwen2.5:3b",
            context_window=4096,
            num_ctx=8192,
            usable_budget=2048,
        )

    # usable_budget > num_ctx raises ValidationError
    with pytest.raises(ValidationError, match="cannot exceed num_ctx"):
        ModelConfig(
            alias="fast",
            model="ollama/qwen2.5:3b",
            context_window=32768,
            num_ctx=8192,
            usable_budget=10000,
        )


def test_models_yaml_contains_explicit_num_ctx():
    """Verify models.yaml has explicit num_ctx and usable_budget for aliases."""
    models_yaml_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    config = load_models_config(models_yaml_path)

    assert "local-fast" in config
    fast = config["local-fast"]
    assert fast.num_ctx == 8192
    assert fast.usable_budget == 6144
    assert fast.context_window == 32768

    assert "local-coder" in config
    coder = config["local-coder"]
    assert coder.num_ctx == 16384
    assert coder.usable_budget == 12288
    assert coder.context_window == 32768


# ── 2. LiteLLM num_ctx Parameter Passing ──────────────────────────────────────

@pytest.mark.anyio
async def test_acompletion_passes_num_ctx():
    """acompletion_with_tools must pass model_config.num_ctx to litellm."""
    cfg = ModelConfig(
        alias="fast",
        model="ollama/qwen2.5:3b",
        num_ctx=8192,
        usable_budget=6144,
    )
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = type("Resp", (), {
            "choices": [type("C", (), {"message": type("M", (), {"content": "ok", "tool_calls": None})()})()],
            "usage": None,
        })()

        await acompletion_with_tools(
            model_config=cfg,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert mock_llm.called
        kwargs = mock_llm.call_args[1]
        assert kwargs.get("num_ctx") == 8192


@pytest.mark.anyio
async def test_stream_chat_passes_num_ctx():
    """stream_chat_completion must pass model_config.num_ctx to litellm."""
    cfg = ModelConfig(
        alias="fast",
        model="ollama/qwen2.5:3b",
        num_ctx=8192,
        usable_budget=6144,
    )
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        async def fake_stream():
            yield type("Chunk", (), {
                "choices": [type("C", (), {"delta": type("D", (), {"content": "hi"})()})()],
                "usage": None,
            })()

        mock_llm.return_value = fake_stream()

        chunks = []
        async for delta, _ in stream_chat_completion(cfg, [{"role": "user", "content": "hi"}]):
            if delta:
                chunks.append(delta)

        assert chunks == ["hi"]
        assert mock_llm.call_args[1].get("num_ctx") == 8192


# ── 3. Tool Result Capping ────────────────────────────────────────────────────

def test_format_tool_result_caps_large_output():
    """Outputs exceeding budget are capped with truncation marker and hint."""
    large_text = "line of code\n" * 1000  # ~13,000 characters
    inv_res = GatewayInvocationResult(
        tool_name="read_file",
        authorized=True,
        success=True,
        output={"content": large_text},
    )

    formatted = format_tool_result_for_model(inv_res, max_chars=1000)

    assert "TOOL RESULT TRUNCATED" in formatted
    assert "Hint: narrow your query" in formatted
    # Check that body was capped close to 1000 chars (plus marker and tags)
    assert len(formatted) < 2000
    assert "UNTRUSTED DATA" in formatted


def test_format_tool_result_keeps_small_output_intact():
    """Outputs within budget are not truncated."""
    inv_res = GatewayInvocationResult(
        tool_name="read_file",
        authorized=True,
        success=True,
        output={"content": "hello world"},
    )
    formatted = format_tool_result_for_model(inv_res)
    assert "TOOL RESULT TRUNCATED" not in formatted
    assert "hello world" in formatted


# ── 4. End-to-End Truncation Detection & Warning Events ────────────────────────

class MockWs:
    def __init__(self):
        self.sent: List[Dict[str, Any]] = []

    async def send(self, data: str):
        self.sent.append(json.loads(data))


def _make_mock_gateway(tmp_path: Path):
    reg = create_default_registry(include_write_tools=False)
    audit = type("MockAudit", (), {"log_invocation": lambda *a, **kw: None})()
    return ToolGateway(registry=reg, audit_logger=audit, workspace_root=tmp_path)


@pytest.mark.anyio
async def test_truncation_detection_emits_warning_event(tmp_path: Path, caplog):
    """When prompt tokens exceed num_ctx, log warning, emit UI event with truncated=True."""
    mock_ws = MockWs()
    conn = ConnectionWrapper(mock_ws)
    gateway = _make_mock_gateway(tmp_path)

    cfg = ModelConfig(
        alias="test-fast",
        model="ollama/qwen2.5:3b",
        context_window=8192,
        num_ctx=2048,
        usable_budget=1500,
    )

    session = Session(session_id="sess-trunc", workspace_root=str(tmp_path))
    session.add_user_message("Big request")

    # Simulate LLM returning prompt_tokens = 2048 (at num_ctx)
    fake_resp = type("Resp", (), {
        "choices": [type("C", (), {
            "message": type("M", (), {"content": "Answer", "tool_calls": None})(),
        })()],
        "usage": type("U", (), {
            "prompt_tokens": 2048,
            "completion_tokens": 20,
            "total_tokens": 2068,
        })(),
    })()

    with caplog.at_level(logging.WARNING), \
         patch("agent_core.server.acompletion_with_tools", return_value=fake_resp):
        await run_chat_generation(
            session=session,
            model_config=cfg,
            request_id="req-trunc",
            conn_wrapper=conn,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    # 1. Verify warning was logged
    truncation_logs = [r.message for r in caplog.records if "PROMPT TRUNCATION DETECTED" in r.message]
    assert len(truncation_logs) >= 1

    # 2. Verify context.status message was emitted
    status_events = [m for m in mock_ws.sent if m.get("type") == "context.status"]
    assert len(status_events) >= 1
    ev = status_events[0]["payload"]
    assert ev["truncated"] is True
    assert ev["prompt_tokens"] == 2048
    assert ev["num_ctx"] == 2048
    assert ev["usage_percent"] > 100.0

    # 3. Verify chat.done message carries complete context accounting
    done_events = [m for m in mock_ws.sent if m.get("type") == "chat.done"]
    assert len(done_events) >= 1
    done_usage = done_events[0]["payload"]["usage"]
    assert done_usage["truncated"] is True
    assert done_usage["num_ctx"] == 2048
    assert done_usage["usable_budget"] == 1500


@pytest.mark.anyio
async def test_context_usage_percent_warnings(tmp_path: Path):
    """At 40% shows 'consider compacting', at 60% shows warning."""
    mock_ws = MockWs()
    conn = ConnectionWrapper(mock_ws)
    gateway = _make_mock_gateway(tmp_path)

    cfg = ModelConfig(
        alias="test-budget",
        model="ollama/qwen2.5:3b",
        context_window=10000,
        num_ctx=8000,
        usable_budget=5000,
    )

    # Test 45% of usable_budget: 2250 tokens
    session40 = Session(session_id="sess-40", workspace_root=str(tmp_path))
    session40.add_user_message("Query")
    resp40 = type("Resp", (), {
        "choices": [type("C", (), {"message": type("M", (), {"content": "ok", "tool_calls": None})()})()],
        "usage": type("U", (), {"prompt_tokens": 2250, "completion_tokens": 10, "total_tokens": 2260})(),
    })()

    with patch("agent_core.server.acompletion_with_tools", return_value=resp40):
        await run_chat_generation(
            session=session40,
            model_config=cfg,
            request_id="req-40",
            conn_wrapper=conn,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    status_events = [m for m in mock_ws.sent if m.get("type") == "context.status"]
    assert any("Consider compacting" in (m["payload"].get("warning") or "") for m in status_events)

    # Test 65% of usable_budget: 3250 tokens
    mock_ws.sent.clear()
    session60 = Session(session_id="sess-60", workspace_root=str(tmp_path))
    session60.add_user_message("Query")
    resp60 = type("Resp", (), {
        "choices": [type("C", (), {"message": type("M", (), {"content": "ok", "tool_calls": None})()})()],
        "usage": type("U", (), {"prompt_tokens": 3250, "completion_tokens": 10, "total_tokens": 3260})(),
    })()

    with patch("agent_core.server.acompletion_with_tools", return_value=resp60):
        await run_chat_generation(
            session=session60,
            model_config=cfg,
            request_id="req-60",
            conn_wrapper=conn,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    status_events60 = [m for m in mock_ws.sent if m.get("type") == "context.status"]
    assert any("Context warning" in (m["payload"].get("warning") or "") for m in status_events60)


@pytest.mark.anyio
async def test_truncation_detection_fires_when_prompt_exceeds_num_ctx(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Proves truncation detection fires when prompt tokens exceed num_ctx with simulated Ollama response."""
    mock_ws = MockWs()
    conn = ConnectionWrapper(mock_ws)
    gateway = _make_mock_gateway(tmp_path)

    cfg = ModelConfig(
        alias="test-exceed",
        model="ollama/qwen2.5:3b",
        context_window=8192,
        num_ctx=2048,
        usable_budget=1536,
    )

    session = Session(session_id="sess-exceed", workspace_root=str(tmp_path))
    session.add_user_message("Excessive prompt request")

    # Simulate Ollama returning prompt_tokens = 2500 (exceeds num_ctx of 2048)
    simulated_ollama_resp = type("Resp", (), {
        "choices": [type("C", (), {
            "message": type("M", (), {"content": "Processed after cut", "tool_calls": None})(),
        })()],
        "usage": type("U", (), {
            "prompt_tokens": 2500,
            "completion_tokens": 15,
            "total_tokens": 2515,
        })(),
    })()

    with caplog.at_level(logging.WARNING), \
         patch("agent_core.server.acompletion_with_tools", return_value=simulated_ollama_resp):
        await run_chat_generation(
            session=session,
            model_config=cfg,
            request_id="req-exceed",
            conn_wrapper=conn,
            logger=logging.getLogger("test"),
            tool_gateway=gateway,
        )

    # 1. Truncation warning was logged with exact token comparison
    truncation_logs = [r.message for r in caplog.records if "PROMPT TRUNCATION DETECTED" in r.message]
    assert len(truncation_logs) == 1
    assert "prompt_tokens=2500 >= num_ctx=2048" in truncation_logs[0]

    # 2. context.status message emitted with truncated: True and usage > 100%
    status_events = [m for m in mock_ws.sent if m.get("type") == "context.status"]
    assert len(status_events) >= 1
    ev = status_events[0]["payload"]
    assert ev["truncated"] is True
    assert ev["prompt_tokens"] == 2500
    assert ev["num_ctx"] == 2048
    assert ev["usage_percent"] > 100.0
    assert "Context TRUNCATED" in ev["warning"]

    # 3. Final chat.done message flags truncated: True
    done_events = [m for m in mock_ws.sent if m.get("type") == "chat.done"]
    assert len(done_events) >= 1
    assert done_events[0]["payload"]["usage"]["truncated"] is True

