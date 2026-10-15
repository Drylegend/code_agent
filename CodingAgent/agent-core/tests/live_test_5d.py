"""Live verification script for Phase 5D against local Ollama local-fast (qwen2.5:3b)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import sys

from agent_core.config import ModelConfig
from agent_core.server import ConnectionWrapper, run_chat_generation
from agent_core.sessions import Session
from agent_core.tools import (
    AuditLogger,
    PathGuard,
    ToolGateway,
    create_default_registry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_test_5d")


class CapturingConnectionWrapper:
    """Collects all sent WebSocket message payloads."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, data: str) -> None:
        parsed = json.loads(data)
        self.messages.append(parsed)
        msg_type = parsed.get("type")
        if msg_type == "tool.call":
            print(f"\n[WS EVENT tool.call] tool={parsed.get('tool')} args={parsed.get('args')} call_id={parsed.get('call_id')}")
        elif msg_type == "tool.result":
            preview = parsed.get("preview") or ""
            preview_len = len(preview)
            print(f"[WS EVENT tool.result] call_id={parsed.get('call_id')} ok={parsed.get('ok')} summary='{parsed.get('summary')}' preview_len={preview_len}")
        elif msg_type == "chat.delta":
            print(parsed.get("payload", {}).get("text", ""), end="", flush=True)
        elif msg_type == "chat.done":
            print(f"\n[WS EVENT chat.done] usage={parsed.get('payload', {}).get('usage')}")


async def main() -> None:
    workspace_root = Path(r"D:\personal_coding_agent\CodingAgent").resolve()
    audit_file = workspace_root.parent / "test_5d_audit.jsonl"
    if audit_file.exists():
        audit_file.unlink()

    audit_logger = AuditLogger(audit_file)
    registry = create_default_registry()
    path_guard = PathGuard(workspace_root)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace_root,
        path_guard=path_guard,
    )

    model_config = ModelConfig(
        alias="local-fast",
        model="ollama/qwen2.5:3b",
        api_base="http://127.0.0.1:11434",
    )

    session = Session("live-5d-session")

    # TEST 1: list_directory
    print("=" * 60)
    print("TEST 1: List the files in my workspace and tell me what kind of project it appears to be.")
    print("=" * 60)

    prompt_1 = "List the files in my workspace and tell me what kind of project it appears to be."
    session.add_user_message(prompt_1)
    conn_1 = CapturingConnectionWrapper()

    await run_chat_generation(
        session=session,
        model_config=model_config,
        request_id="req-live-1",
        conn_wrapper=conn_1,  # type: ignore
        logger=logger,
        tool_gateway=gateway,
    )

    # Verification for Test 1
    tool_calls_1 = [m for m in conn_1.messages if m.get("type") == "tool.call"]
    tool_results_1 = [m for m in conn_1.messages if m.get("type") == "tool.result"]
    dones_1 = [m for m in conn_1.messages if m.get("type") == "chat.done"]

    assert len(tool_calls_1) >= 1, "Expected at least 1 tool.call event"
    assert len(tool_results_1) >= 1, "Expected at least 1 tool.result event"
    assert len(dones_1) == 1, "Expected exactly 1 chat.done event"

    first_call = tool_calls_1[0]
    first_result = tool_results_1[0]
    assert first_call["tool"] == "list_directory"
    assert first_call["call_id"] == first_result["call_id"]
    assert first_result["ok"] is True
    print("\n[OK] Test 1 verified: tool.call and tool.result emitted, correlated by call_id, response streamed.")

    # TEST 2: read_file
    print("\n" + "=" * 60)
    print("TEST 2: Read the main README or project documentation file and summarize what this project does.")
    print("=" * 60)

    prompt_2 = "Read the main README or project documentation file and summarize what this project does."
    session.add_user_message(prompt_2)
    conn_2 = CapturingConnectionWrapper()

    await run_chat_generation(
        session=session,
        model_config=model_config,
        request_id="req-live-2",
        conn_wrapper=conn_2,  # type: ignore
        logger=logger,
        tool_gateway=gateway,
    )

    # Verification for Test 2
    tool_calls_2 = [m for m in conn_2.messages if m.get("type") == "tool.call"]
    tool_results_2 = [m for m in conn_2.messages if m.get("type") == "tool.result"]
    dones_2 = [m for m in conn_2.messages if m.get("type") == "chat.done"]

    assert len(tool_calls_2) >= 1, "Expected at least 1 tool.call event for read_file"
    assert len(tool_results_2) >= 1, "Expected at least 1 tool.result event for read_file"
    assert len(dones_2) == 1, "Expected exactly 1 chat.done event"

    read_call = tool_calls_2[0]
    read_result = tool_results_2[0]
    assert read_call["tool"] in ("read_file", "list_directory")
    assert read_call["call_id"] == read_result["call_id"]
    assert read_result["ok"] is True
    if read_result.get("preview"):
        assert len(read_result["preview"]) <= 4200, "Preview must be bounded to <= 4 KB"

    print("\n[OK] Test 2 verified: tool activity emitted with bounded preview and streamed response.")

    # Check audit log
    records = []
    with open(audit_file, "r", encoding="utf-8") as af:
        for line in af:
            if line.strip():
                records.append(json.loads(line))
    print(f"\n[OK] Audit records written: {len(records)}")
    assert len(records) >= 2, "Expected at least 2 audit log records"

    from agent_core.tools.verify_audit import verify_audit_log
    is_valid, errors = verify_audit_log(audit_file)
    assert is_valid, f"Audit hash chain must be intact: {errors}"
    print(f"[OK] Audit hash chain integrity verified ({len(records)} valid records).")

    # Cleanup audit file
    if audit_file.exists():
        audit_file.unlink()

    print("\n================ ALL LIVE TESTS PASSED! ================")


if __name__ == "__main__":
    asyncio.run(main())
