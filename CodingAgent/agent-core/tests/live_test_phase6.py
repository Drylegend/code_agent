"""Live verification script for Phase 6 against Ollama local-fast.

Explicit workspace root:
D:\\personal_coding_agent\\CodingAgent\\agent-shell

Confirms:
1. list_directory(".") sees only agent-shell contents
2. agent-core/logs are NOT visible
3. read_file("package.json") reads agent-shell/package.json
4. read_file("../README.md") is denied
5. Full live LLM tool-calling loop adheres strictly to the explicit workspace root
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import sys

# Ensure agent_core is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core.config import ModelConfig
from agent_core.server import ConnectionWrapper, run_chat_generation
from agent_core.sessions import Session
from agent_core.tools import (
    AuditLogger,
    ToolGateway,
    create_default_registry,
)
from agent_core.workspace import validate_workspace_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_test_phase6")


class CapturingConnectionWrapper:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, data: str) -> None:
        parsed = json.loads(data)
        self.messages.append(parsed)
        msg_type = parsed.get("type")
        if msg_type == "tool.call":
            print(f"\n[WS EVENT tool.call] tool={parsed.get('tool')} args={parsed.get('args')}")
        elif msg_type == "tool.result":
            preview = parsed.get("preview") or ""
            print(f"[WS EVENT tool.result] ok={parsed.get('ok')} summary='{parsed.get('summary')}' preview_len={len(preview)}")
        elif msg_type == "chat.delta":
            print(parsed.get("payload", {}).get("text", ""), end="", flush=True)
        elif msg_type == "chat.done":
            print(f"\n[WS EVENT chat.done]")


async def main() -> None:
    raw_target_ws = r"D:\personal_coding_agent\CodingAgent\agent-shell"
    validated_ws = validate_workspace_root(raw_target_ws)
    print(f"Validated explicit workspace root: {validated_ws}")
    assert validated_ws.name == "agent-shell", f"Expected agent-shell, got {validated_ws.name}"

    store_base = Path(os.environ.get("AGENT_CORE_STORE_DIR", r"D:\agent-store")).resolve()
    test_audit_dir = store_base / "test_audit"
    test_audit_dir.mkdir(parents=True, exist_ok=True)
    audit_file = test_audit_dir / "phase6_live_audit.jsonl"
    if audit_file.exists():
        audit_file.unlink()

    audit_logger = AuditLogger(audit_file)
    registry = create_default_registry()

    # ToolGateway created WITHOUT hardcoded workspace root; relies purely on per-session root
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=None,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Check 1: Direct ToolGateway list_directory(".")
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 1: list_directory('.') against agent-shell")
    print("=" * 60)

    res_list = await gateway.invoke(
        session_id="session-live-phase6",
        tool_name="list_directory",
        raw_args={"path": "."},
        workspace_root=validated_ws,
    )

    assert res_list.success is True, f"list_directory failed: {res_list.error}"
    assert res_list.authorized is True, f"list_directory denied: {res_list.denial_reason}"
    entry_names = [e["name"] for e in res_list.output["entries"]]
    print(f"Listed {len(entry_names)} entries in agent-shell: {entry_names}")

    # Confirm agent-shell contents are present
    assert "package.json" in entry_names, "package.json should be present in agent-shell"
    assert "src" in entry_names, "src directory should be present in agent-shell"

    # Confirm agent-core and logs are NOT visible
    assert "agent-core" not in entry_names, "CRITICAL: agent-core must NOT be visible!"
    assert "logs" not in entry_names, "CRITICAL: logs must NOT be visible!"
    print("[PASS] Check 1: agent-shell contents visible; agent-core/logs NOT visible.")

    # ──────────────────────────────────────────────────────────────────────────
    # Check 2: Direct ToolGateway read_file("package.json")
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 2: read_file('package.json') against agent-shell")
    print("=" * 60)

    res_read = await gateway.invoke(
        session_id="session-live-phase6",
        tool_name="read_file",
        raw_args={"path": "package.json"},
        workspace_root=validated_ws,
    )

    assert res_read.success is True, f"read_file failed: {res_read.error}"
    assert res_read.authorized is True, f"read_file denied: {res_read.denial_reason}"
    pkg_content = res_read.output["content"]
    assert '"name": "agent-shell"' in pkg_content, f"Expected agent-shell package.json, got: {pkg_content[:200]}"
    print("[PASS] Check 2: Successfully read agent-shell/package.json.")

    # ──────────────────────────────────────────────────────────────────────────
    # Check 3: Direct ToolGateway read_file("../README.md")
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 3: read_file('../README.md') escaping agent-shell")
    print("=" * 60)

    res_escape = await gateway.invoke(
        session_id="session-live-phase6",
        tool_name="read_file",
        raw_args={"path": "../README.md"},
        workspace_root=validated_ws,
    )

    assert res_escape.success is False, "read_file outside workspace should fail"
    assert res_escape.authorized is False, "read_file outside workspace should be unauthorized"
    print(f"Denial reason: {res_escape.denial_reason}")
    assert "Path authorization denied" in (res_escape.denial_reason or "") or "escapes" in (res_escape.denial_reason or "").lower()
    print("[PASS] Check 3: read_file('../README.md') was strictly denied.")

    # ──────────────────────────────────────────────────────────────────────────
    # Check 4: Live Ollama LLM tool loop with explicit session workspace root
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 4: Live LLM tool execution with agent-shell workspace root")
    print("=" * 60)

    model_config = ModelConfig(
        alias="local-fast",
        model="ollama/qwen2.5:3b",
        api_base="http://127.0.0.1:11434",
    )

    session = Session("sess-phase6-llm-1")
    session.set_workspace_root(validated_ws)
    prompt = "Please list the files in the workspace directory to see what is here."
    session.add_user_message(prompt)

    conn = CapturingConnectionWrapper()
    await run_chat_generation(
        session=session,
        model_config=model_config,
        request_id="req-phase6-live",
        conn_wrapper=conn,  # type: ignore
        logger=logger,
        tool_gateway=gateway,
    )

    tool_calls = [m for m in conn.messages if m.get("type") == "tool.call"]
    tool_results = [m for m in conn.messages if m.get("type") == "tool.result"]
    dones = [m for m in conn.messages if m.get("type") == "chat.done"]

    assert len(tool_calls) >= 1, "Expected at least 1 tool.call event"
    assert len(tool_results) >= 1, "Expected at least 1 tool.result event"
    assert len(dones) == 1, "Expected exactly 1 chat.done event"

    first_call = tool_calls[0]
    assert first_call["tool"] == "list_directory"
    first_result = tool_results[0]
    assert first_result["ok"] is True
    print(f"\n[PASS] Check 4: Live LLM successfully used list_directory confined to {validated_ws.name}.")

    print("\n" + "=" * 60)
    print("ALL PHASE 6 LIVE VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
