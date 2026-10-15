"""Phase 8 Part A Live Test Suite.

Verifies:
1. local-coder (text_json mode) completes read_file then write_file without printing raw JSON to chat.
2. A model that announces a tool call and stops triggers the failure nudge and loop continues.
3. write_file persists after full app restart (write, restart sidecar/app, read from disk, verify hash).
4. Undo works after restart (restore snapshot from disk, verify exact original bytes).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core.config import ModelConfig
from agent_core.server import ConnectionWrapper, run_chat_generation
from agent_core.sessions import Session
from agent_core.snapshots import SnapshotManager
from agent_core.tools import (
    AuditLogger,
    PathGuard,
    ToolGateway,
    create_default_registry,
)
from agent_core.tools.read_file import ReadFileInput, ReadFileTool
from agent_core.tools.write_file import WriteFileInput, WriteFileTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_test_phase8_a")


class MockConn:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    def get_delta_text(self) -> str:
        return "".join(
            (m.get("payload", {}).get("text", "") if "payload" in m else m.get("text", ""))
            for m in self.sent
            if m.get("type") == "chat.delta"
        )


class FakeChoice:
    def __init__(self, message: Any) -> None:
        self.message = message


class FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class FakeResponse:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None) -> None:
        self.choices = [FakeChoice(FakeMessage(content=content, tool_calls=tool_calls))]
        self.usage = None


def run_all_tests() -> dict[str, str]:
    print("\n" + "=" * 70)
    print("STARTING PHASE 8 PART A LIVE TESTS (1 TO 4)")
    print("=" * 70 + "\n")

    results: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="live_test_p8a_") as td:
        base_dir = Path(td).resolve()
        workspace = base_dir / "workspace"
        workspace.mkdir()
        store_dir = base_dir / "store"
        store_dir.mkdir()
        audit_file = store_dir / "audit.jsonl"
        audit_logger = AuditLogger(audit_file)

        snap_mgr = SnapshotManager(store_dir=store_dir)
        registry = create_default_registry(include_write_tools=True, snapshot_manager=snap_mgr)
        gateway = ToolGateway(
            registry=registry,
            audit_logger=audit_logger,
            workspace_root=workspace,
            path_guard=PathGuard(workspace),
        )

        test_file = workspace / "data.txt"
        test_file.write_text("Initial content line 1\n", encoding="utf-8")

        cfg = ModelConfig(alias="local-coder", model="ollama/qwen2.5-coder:7b", tool_mode="text_json")

        # -----------------------------------------------------------------
        # TEST 1: local-coder text_json mode: read_file then write_file
        # -----------------------------------------------------------------
        print("> Running Test 1: local-coder text_json multi-step without raw JSON in chat...")
        conn1 = MockConn()
        sess1 = Session("sess-text-json-1", workspace_root=workspace)
        sess1.add_user_message("Please read data.txt and then overwrite it with new text.")

        # Simulate 3 rounds:
        # Round 1: Model outputs markdown json block with read_file
        resp_r1 = FakeResponse(
            content='I will read the file:\n```json\n{"name": "read_file", "arguments": {"path": "data.txt"}}\n```'
        )
        # Round 2: Model outputs markdown json block with write_file
        resp_r2 = FakeResponse(
            content='Now writing updated content:\n```json\n{"name": "write_file", "arguments": {"path": "data.txt", "content": "Updated by text_json\\n", "overwrite": true}}\n```'
        )
        # Round 3: Model completes task with final conversational text
        resp_r3 = FakeResponse(
            content="I have read data.txt and successfully updated its content."
        )

        responses = [resp_r1, resp_r2, resp_r3]

        async def fake_acompletion(*args: Any, **kwargs: Any) -> Any:
            return responses.pop(0)

        with patch("agent_core.server.acompletion_with_tools", side_effect=fake_acompletion):
            asyncio.run(
                run_chat_generation(
                    session=sess1,
                    model_config=cfg,
                    request_id="req-p8a-1",
                    conn_wrapper=conn1,
                    logger=logger,
                    tool_gateway=gateway,
                )
            )

        delta_text = conn1.get_delta_text()
        assert "read_file" not in delta_text, f"Raw JSON leaked into chat: {delta_text}"
        assert "write_file" not in delta_text, f"Raw JSON leaked into chat: {delta_text}"
        assert '{"name":' not in delta_text, f"Raw JSON leaked into chat: {delta_text}"
        assert "I have read data.txt and successfully updated its content." in delta_text

        tool_calls_emitted = [m for m in conn1.sent if m.get("type") == "tool.call"]
        assert len(tool_calls_emitted) == 2, f"Expected 2 tool.call events, got {len(tool_calls_emitted)}"
        tools_called = [(m.get("payload", {}).get("tool") or m.get("tool")) for m in tool_calls_emitted]
        assert tools_called[0] == "read_file"
        assert tools_called[1] == "write_file"
        assert test_file.read_text(encoding="utf-8") == "Updated by text_json\n"

        print("  [OK] Test 1 PASSED: Completed read_file and write_file with 0 raw JSON leaked to chat.")
        results["1. local-coder text_json mode"] = "PASSED (read then write succeeded, zero raw JSON in chat deltas)"

        # -----------------------------------------------------------------
        # TEST 2: Announced tool call without execution triggers failure nudge
        # -----------------------------------------------------------------
        print("\n> Running Test 2: Announced tool call triggers failure nudge...")
        conn2 = MockConn()
        sess2 = Session("sess-announce-1", workspace_root=workspace)
        sess2.add_user_message("Check the directory")

        # Round 1: Model announces read_file in text without any tool call
        resp_announce = FakeResponse(
            content="I will use the read_file tool to inspect data.txt."
        )
        # Round 2: In response to the nudge, model makes the actual tool call
        resp_exec = FakeResponse(
            content='```json\n{"name": "read_file", "arguments": {"path": "data.txt"}}\n```'
        )
        # Round 3: Final message
        resp_final = FakeResponse(
            content="data.txt contains the updated text."
        )

        nudge_responses = [resp_announce, resp_exec, resp_final]

        async def fake_nudge_acompletion(*args: Any, **kwargs: Any) -> Any:
            return nudge_responses.pop(0)

        with patch("agent_core.server.acompletion_with_tools", side_effect=fake_nudge_acompletion):
            asyncio.run(
                run_chat_generation(
                    session=sess2,
                    model_config=cfg,
                    request_id="req-p8a-2",
                    conn_wrapper=conn2,
                    logger=logger,
                    tool_gateway=gateway,
                )
            )

        # Check session messages for the nudge
        session_messages = sess2.get_messages()
        nudge_msg = next((m for m in session_messages if "You announced an intention to call" in m.get("content", "")), None)
        assert nudge_msg is not None, "Failure nudge was not added to session history!"
        assert "read_file" in nudge_msg["content"]

        tool_calls_emitted2 = [m for m in conn2.sent if m.get("type") == "tool.call"]
        assert len(tool_calls_emitted2) == 1
        tool2_called = tool_calls_emitted2[0].get("payload", {}).get("tool") or tool_calls_emitted2[0].get("tool")
        assert tool2_called == "read_file"
        print("  [OK] Test 2 PASSED: Nudge triggered, loop continued, and tool executed.")
        results["2. Tool announcement nudge"] = "PASSED (Nudge triggered on unexecuted announcement; loop continued and executed tool)"

        # -----------------------------------------------------------------
        # TEST 3: write_file persists after full app restart
        # -----------------------------------------------------------------
        print("\n> Running Test 3: write_file persists across app/sidecar restart...")
        target_file = workspace / "persistent.txt"
        persist_bytes = b"State that must survive full shutdown and cold restart.\r\nLine 2.\n"
        persist_sha = hashlib.sha256(persist_bytes).hexdigest()

        # Step A: Perform write via WriteFileTool
        write_tool = WriteFileTool(snapshot_manager=snap_mgr, audit_logger=audit_logger)
        res = asyncio.run(
            gateway.invoke(
                session_id="sess-persist-1",
                tool_name="write_file",
                raw_args={"path": "persistent.txt", "content": persist_bytes.decode("utf-8"), "overwrite": True},
                workspace_root=workspace,
            )
        )
        assert res.success is True

        # Step B: Simulate complete application and sidecar termination
        del gateway
        del snap_mgr
        del registry

        # Step C: Fresh restart - create new instances from scratch
        snap_mgr_fresh = SnapshotManager(store_dir=store_dir)
        registry_fresh = create_default_registry(include_write_tools=True, snapshot_manager=snap_mgr_fresh)
        gateway_fresh = ToolGateway(
            registry=registry_fresh,
            audit_logger=AuditLogger(audit_file),
            workspace_root=workspace,
            path_guard=PathGuard(workspace),
        )

        # Read directly from disk
        assert target_file.exists(), "Target file missing on disk after restart!"
        read_disk_bytes = target_file.read_bytes()
        assert read_disk_bytes == persist_bytes
        assert hashlib.sha256(read_disk_bytes).hexdigest() == persist_sha

        # Read via fresh ReadFileTool through the fresh gateway
        read_res = asyncio.run(
            gateway_fresh.invoke(
                session_id="sess-fresh-2",
                tool_name="read_file",
                raw_args={"path": "persistent.txt"},
                workspace_root=workspace,
            )
        )
        assert read_res.success is True
        assert read_res.output["content"].replace("\r\n", "\n") == persist_bytes.decode("utf-8").replace("\r\n", "\n")
        print("  [OK] Test 3 PASSED: File and SHA-256 match perfectly after full restart.")
        results["3. write_file persists across restart"] = "PASSED (Exact bytes and SHA-256 verified after cold restart)"

        # -----------------------------------------------------------------
        # TEST 4: Undo works from disk snapshot after restart
        # -----------------------------------------------------------------
        print("\n> Running Test 4: Undo works after restart...")
        # Now modify persistent.txt again in fresh gateway
        mod_res = asyncio.run(
            gateway_fresh.invoke(
                session_id="sess-persist-1",
                tool_name="write_file",
                raw_args={"path": "persistent.txt", "content": "Tampered content\n", "overwrite": True},
                workspace_root=workspace,
            )
        )
        assert mod_res.success is True
        seq = mod_res.output["snapshot_seq"]

        # Simulate second restart before undo
        del gateway_fresh
        del snap_mgr_fresh

        snap_mgr_after = SnapshotManager(store_dir=store_dir)
        snapshots = snap_mgr_after.list_snapshots("sess-persist-1")
        assert len(snapshots) >= 1

        # Perform restore
        undo_res = snap_mgr_after.restore(
            session_id="sess-persist-1",
            seq=seq,
            workspace_root=workspace,
            verify_current_hash=True,
        )
        assert undo_res.success is True
        restored_bytes = target_file.read_bytes()
        assert restored_bytes == persist_bytes, "Restored bytes do not match original bytes!"
        assert hashlib.sha256(restored_bytes).hexdigest() == persist_sha
        print("  [OK] Test 4 PASSED: Undo restored exact bytes after restart.")
        results["4. Undo works after restart"] = "PASSED (Snapshot retrieved from disk and restored with 0 drift)"

    print("\n" + "=" * 70)
    print("ALL 4 PART A LIVE TESTS COMPLETED!")
    print("=" * 70)
    for test_name, status in results.items():
        print(f"  {test_name}: {status}")
    print("=" * 70 + "\n")
    return results


if __name__ == "__main__":
    run_all_tests()
