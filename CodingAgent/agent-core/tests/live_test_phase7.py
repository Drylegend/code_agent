"""Live verification test for Phase 7 (Safe File Editing & Snapshots) against Ollama local-fast.

Workspace Root: D:\\personal_coding_agent\\CodingAgent\\agent-shell

Verifies:
1. Live path traversal denial (attempt to edit/read ../README.md) -> ok=False, PathGuard denial
2. Live denylist denial (attempt to edit/write .env) -> ok=False, PathGuard denial
3. Live edit on README.md (adds temporary section) -> tool executes, snapshot created, diff verified
4. Live rollback via restore_snapshot -> exact byte restoration confirmed
5. Cryptographic audit log hash chain validation via verify_audit_log
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
from pathlib import Path
import sys

# Ensure agent_core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core.config import ModelConfig
from agent_core.server import run_chat_generation
from agent_core.sessions import Session
from agent_core.snapshots import get_snapshot_manager, restore_snapshot
from agent_core.tools import (
    AuditLogger,
    PathGuard,
    ToolGateway,
    create_default_registry,
)
from agent_core.tools.verify_audit import verify_audit_log
from agent_core.workspace import validate_workspace_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_test_phase7")


def get_external_test_audit_path(test_name: str) -> Path:
    """Resolve an external audit log path located outside the workspace under D:\agent-store\test_audit."""
    store_base = Path(os.environ.get("AGENT_CORE_STORE_DIR", r"D:\agent-store")).resolve()
    audit_dir = store_base / "test_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    target = audit_dir / f"{test_name}.jsonl"
    if target.exists():
        target.unlink()
    return target


class CapturingConnectionWrapper:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.tool_calls: list[dict] = []
        self.tool_results: list[dict] = []

    async def send(self, data: str) -> None:
        parsed = json.loads(data)
        self.messages.append(parsed)
        msg_type = parsed.get("type")
        if msg_type == "tool.call":
            self.tool_calls.append(parsed)
            print(f"\n[WS EVENT tool.call] tool={parsed.get('tool')} args={parsed.get('args')}")
        elif msg_type == "tool.result":
            self.tool_results.append(parsed)
            print(f"[WS EVENT tool.result] ok={parsed.get('ok')} summary='{parsed.get('summary')}'")
        elif msg_type == "chat.delta":
            print(parsed.get("payload", {}).get("text", ""), end="", flush=True)
        elif msg_type == "chat.done":
            print(f"\n[WS EVENT chat.done]")


async def test_live_traversal_denial(workspace_root: Path, model_config: ModelConfig) -> bool:
    print("\n" + "=" * 60)
    print("STEP 1: LIVE PATH TRAVERSAL DENIAL (../README.md)")
    print("=" * 60)

    # 1. Gateway direct check
    registry = create_default_registry(include_write_tools=True)
    audit_path = get_external_test_audit_path("live_traversal_audit")
    audit_logger = AuditLogger(audit_path)
    path_guard = PathGuard(workspace_root)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace_root,
        path_guard=path_guard,
    )

    res_direct = await gateway.invoke(
        session_id="live-traversal-direct",
        tool_name="write_file",
        raw_args={"path": "../README.md", "content": "traversal test"},
        workspace_root=workspace_root,
    )
    print(f"Gateway direct write to '../README.md': success={res_direct.success}, error={res_direct.error}")
    assert res_direct.success is False
    assert res_direct.authorized is False
    print("--> Direct gateway traversal check PASSED: denied.")

    # 2. Live LLM Loop check
    session = Session("live-traversal-sess")
    session.set_workspace_root(workspace_root)
    session.add_user_message(
        "Please use write_file to modify '../README.md' and write 'traversal test'.",
    )

    conn = CapturingConnectionWrapper()
    await run_chat_generation(
        session=session,
        model_config=model_config,
        request_id="req-traversal",
        conn_wrapper=conn,  # type: ignore
        logger=logger,
        tool_gateway=gateway,
    )

    if conn.tool_results:
        for tr in conn.tool_results:
            print(f"Tool result from model call: ok={tr.get('ok')}, summary={tr.get('summary')}")
            assert tr.get("ok") is False, "Expected traversal tool call to fail!"
        print("--> SUCCESS: Model tool call to '../README.md' was blocked with ok=False.")
    else:
        print("--> Model refused traversal prompt without emitting tool call.")

    return True


async def test_live_denylist_denial(workspace_root: Path, model_config: ModelConfig) -> bool:
    print("\n" + "=" * 60)
    print("STEP 2: LIVE DENYLIST DENIAL (.env)")
    print("=" * 60)

    # 1. Gateway direct check
    registry = create_default_registry(include_write_tools=True)
    audit_path = get_external_test_audit_path("live_denylist_audit")
    audit_logger = AuditLogger(audit_path)
    path_guard = PathGuard(workspace_root)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace_root,
        path_guard=path_guard,
    )

    res_direct = await gateway.invoke(
        session_id="live-denylist-direct",
        tool_name="write_file",
        raw_args={"path": ".env", "content": "SECRET_KEY=12345"},
        workspace_root=workspace_root,
    )
    print(f"Gateway direct write to '.env': success={res_direct.success}, error={res_direct.error}")
    assert res_direct.success is False
    assert res_direct.authorized is False
    assert not (workspace_root / ".env").exists()
    print("--> Direct gateway denylist check PASSED: denied.")

    # 2. Live LLM Loop check
    session = Session("live-denylist-sess")
    session.set_workspace_root(workspace_root)
    session.add_user_message(
        "Please write a file named '.env' with the content 'SECRET_KEY=12345'.",
    )

    conn = CapturingConnectionWrapper()
    await run_chat_generation(
        session=session,
        model_config=model_config,
        request_id="req-denylist",
        conn_wrapper=conn,  # type: ignore
        logger=logger,
        tool_gateway=gateway,
    )

    assert not (workspace_root / ".env").exists(), "CRITICAL: .env file was created!"
    if conn.tool_results:
        for tr in conn.tool_results:
            print(f"Tool result from model call: ok={tr.get('ok')}, summary={tr.get('summary')}")
            assert tr.get("ok") is False, "Expected .env tool call to fail!"
        print("--> SUCCESS: Model tool call to '.env' was blocked with ok=False.")
    else:
        print("--> Model refused .env write prompt without emitting tool call.")

    return True


async def test_live_edit_and_rollback(workspace_root: Path, model_config: ModelConfig) -> bool:
    print("\n" + "=" * 60)
    print("STEP 3: LIVE FILE EDITING & SNAPSHOT ROLLBACK ON README.md")
    print("=" * 60)

    readme_path = workspace_root / "README.md"
    assert readme_path.exists(), f"README.md not found at {readme_path}"
    original_bytes = readme_path.read_bytes()
    original_text = readme_path.read_text(encoding="utf-8")
    print(f"Original README.md size: {len(original_bytes)} bytes")

    session_id = "live-edit-sess"
    session = Session(session_id)
    session.set_workspace_root(workspace_root)

    # Prompt model to read first then edit
    session.add_user_message(
        "First, use read_file to inspect README.md. Then, use apply_patch to append the following section at the very end of README.md:\n\n"
        "## Phase 7 Verification\n"
        "Safe file editing has been verified on Windows.\n",
    )

    registry = create_default_registry(include_write_tools=True)
    audit_path = get_external_test_audit_path("live_edit_audit")
    audit_logger = AuditLogger(audit_path)
    path_guard = PathGuard(workspace_root)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace_root,
        path_guard=path_guard,
    )

    conn = CapturingConnectionWrapper()
    await run_chat_generation(
        session=session,
        model_config=model_config,
        request_id="req-edit-live",
        conn_wrapper=conn,  # type: ignore
        logger=logger,
        tool_gateway=gateway,
    )

    # Check if model executed an edit
    new_bytes = readme_path.read_bytes()
    new_text = readme_path.read_text(encoding="utf-8")

    if new_bytes == original_bytes:
        print("\nModel did not edit via LLM prompt directly. Executing direct apply_patch to test live snapshot rollback.")
        patch_res = await gateway.invoke(
            session_id=session_id,
            tool_name="apply_patch",
            raw_args={
                "path": "README.md",
                "edits": [
                    {
                        "old_text": original_text,
                        "new_text": original_text + "\n## Phase 7 Verification\nSafe file editing has been verified on Windows.\n",
                    }
                ],
            },
            workspace_root=workspace_root,
        )
        assert patch_res.success is True
        new_bytes = readme_path.read_bytes()
        new_text = readme_path.read_text(encoding="utf-8")

    assert new_bytes != original_bytes, "README.md was not modified!"
    print(f"\nModified README.md size: {len(new_bytes)} bytes")

    # Print Unified Diff
    print("\n--- UNIFIED DIFF OF EDIT ---")
    diff = difflib.unified_diff(
        original_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile="README.md.orig",
        tofile="README.md.modified",
    )
    diff_text = "".join(diff)
    print(diff_text or "[No diff generated]")
    print("-" * 28)

    # Verify snapshot was captured
    mgr = get_snapshot_manager()
    snapshots = mgr.list_snapshots(session_id)
    print(f"Recorded pre-write snapshots for session '{session_id}': {len(snapshots)}")
    assert len(snapshots) >= 1, "No pre-write snapshot found!"
    latest_seq = snapshots[-1]["seq"]
    print(f"Latest snapshot seq: {latest_seq}, details: {snapshots[-1]}")

    # Perform snapshot rollback
    print(f"\nRestoring snapshot seq {latest_seq} via restore_snapshot...")
    restore_res = restore_snapshot(
        seq=latest_seq,
        session_id=session_id,
        workspace_root=workspace_root,
    )
    print(f"Restore outcome: {restore_res.to_dict()}")
    assert restore_res.success is True, f"restore_snapshot failed: {restore_res.error}"

    # Confirm exact byte-for-byte restoration
    restored_bytes = readme_path.read_bytes()
    assert restored_bytes == original_bytes, "Restored bytes DO NOT match original bytes!"
    print(f"--> SUCCESS: README.md restored byte-for-byte ({len(restored_bytes)} bytes match perfectly).")

    # Verify audit log integrity
    print(f"\nVerifying audit log hash chain at {audit_path}...")
    is_valid, errors = verify_audit_log(audit_path)
    print(f"Audit log valid: {is_valid}, errors: {errors}")
    assert is_valid is True, f"Audit log verification failed: {errors}"
    assert len(errors) == 0

    # Cleanup external test audit files
    if audit_path.exists():
        audit_path.unlink()
    for extra in audit_path.parent.glob("live_*.jsonl"):
        extra.unlink(missing_ok=True)

    # Strictly assert that the workspace root does NOT contain any audit logs
    workspace_audit_files = (
        list(workspace_root.glob("*.jsonl"))
        + list(workspace_root.glob(".*audit*.jsonl"))
        + list(workspace_root.glob("*audit*.jsonl"))
    )
    assert len(workspace_audit_files) == 0, f"Found forbidden audit logs inside workspace: {workspace_audit_files}"

    print("\n--> ALL LIVE VERIFICATION CRITERIA PASSED SUCCESSFULLY.")
    return True


async def main() -> None:
    raw_target_ws = r"D:\personal_coding_agent\CodingAgent\agent-shell"
    validated_ws = validate_workspace_root(raw_target_ws)
    print(f"Validated Workspace Root: {validated_ws}")

    model_config = ModelConfig(
        alias="local-fast",
        model="ollama/qwen2.5:3b",
        api_base="http://127.0.0.1:11434",
    )

    await test_live_traversal_denial(validated_ws, model_config)
    await test_live_denylist_denial(validated_ws, model_config)
    await test_live_edit_and_rollback(validated_ws, model_config)


if __name__ == "__main__":
    asyncio.run(main())
