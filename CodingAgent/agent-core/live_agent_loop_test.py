"""Live integration test for Phase 5C generic agent loop with local-fast."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from agent_core.config import load_models_config
from agent_core.server import ConnectionWrapper, run_chat_generation
from agent_core.sessions import Session
from agent_core.tools import (
    AuditLogger,
    PathGuard,
    ToolGateway,
    create_default_registry,
)


class LiveTestConnection:
    """Mock connection collector for live integration testing."""

    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.done: dict | None = None
        self.errors: list[dict] = []

    async def send(self, data: str) -> None:
        msg = json.loads(data)
        msg_type = msg.get("type")
        if msg_type == "chat.delta":
            text = msg["payload"]["text"]
            self.deltas.append(text)
            print(text, end="", flush=True)
        elif msg_type == "chat.done":
            self.done = msg
            print("\n[LIVE TEST] Received chat.done", flush=True)
        elif msg_type == "chat.error":
            self.errors.append(msg)
            print(f"\n[LIVE TEST] Error: {msg}", flush=True)


async def main() -> None:
    logger = logging.getLogger("live_test")
    logger.setLevel(logging.INFO)

    models = load_models_config()
    model_cfg = models["local-fast"]
    print(f"[LIVE TEST] Model: {model_cfg.alias} -> {model_cfg.model} at {model_cfg.api_base}")

    # Set workspace root to CodingAgent directory
    workspace_root = Path(__file__).resolve().parent.parent
    audit_file = workspace_root / "logs" / "live_test_audit.jsonl"
    audit_logger = AuditLogger(audit_file)
    registry = create_default_registry()
    path_guard = PathGuard(workspace_root)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace_root,
        path_guard=path_guard,
    )

    session = Session("sess-live-test")
    user_prompt = "List the files in my workspace and tell me what kind of project it appears to be."
    session.add_user_message(user_prompt)
    print(f"[LIVE TEST] User Prompt: '{user_prompt}'\n")

    conn = LiveTestConnection()

    print("--- Streaming Model Response ---")
    await run_chat_generation(
        session=session,
        model_config=model_cfg,
        request_id="live-req-1",
        conn_wrapper=conn,  # type: ignore
        logger=logger,
        tool_gateway=gateway,
    )
    print("\n--------------------------------")

    # Assertions
    msgs = session.get_messages()
    print(f"\n[LIVE TEST] Total session messages: {len(msgs)}")
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "assistant" and "tool_calls" in m:
            tc_names = [tc["function"]["name"] for tc in m["tool_calls"]]
            print(f"  [{i}] {role} tool_calls: {tc_names}")
        elif role == "tool":
            print(f"  [{i}] tool result for '{m.get('name')}': {len(m.get('content', ''))} chars")
        else:
            snippet = str(m.get("content", ""))[:80].replace("\n", " ")
            print(f"  [{i}] {role}: {snippet}...")

    # Verification:
    # 1. Must have executed list_directory
    tool_messages = [m for m in msgs if m.get("role") == "tool" and m.get("name") == "list_directory"]
    assert len(tool_messages) >= 1, "Expected list_directory to be invoked"

    # 2. Must have final assistant response
    assert conn.done is not None, "Expected chat.done message"
    final_text = "".join(conn.deltas)
    assert len(final_text) > 0, "Expected non-empty final response"

    print("\n[LIVE TEST] SUCCESS! local-fast called list_directory and answered based on the result.")


if __name__ == "__main__":
    asyncio.run(main())
