"""Phase 8 Part B Live Test Suite.

End-to-end live test verifying:
1. ToolGateway pauses on write_file under Autonomy Level 2 (Ask).
2. Approval request event is emitted with request_id, tool, path, and bytes.
3. User approval response ("approve") unblocks the gateway, executes write, and writes file to disk.
4. User rejection response ("reject") safely aborts the tool call and denies write without modifying disk.
5. Hash chain audit log records approval_requested, approval_granted, and approval_rejected.
"""

from __future__ import annotations

import asyncio
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
    ApprovalManager,
    create_default_registry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_test_phase8_b")


class MockConn:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send(self, data: str) -> None:
        parsed = json.loads(data)
        self.sent.append(parsed)

    def get_approval_requests(self) -> List[Dict[str, Any]]:
        return [
            m for m in self.sent
            if m.get("type") == "approval_request"
        ]


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


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = type("Function", (), {"name": name, "arguments": arguments})()


def run_live_test_phase8_b() -> dict[str, str]:
    print("\n" + "=" * 70)
    print("STARTING PHASE 8 PART B LIVE APPROVAL GATE TEST")
    print("=" * 70 + "\n")

    results: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="live_test_p8b_") as td:
        base_dir = Path(td).resolve()
        workspace = base_dir / "workspace"
        workspace.mkdir()
        store_dir = base_dir / "store"
        store_dir.mkdir()
        audit_file = store_dir / "audit.jsonl"
        audit_logger = AuditLogger(audit_file)

        snap_mgr = SnapshotManager(store_dir=store_dir)
        approval_mgr = ApprovalManager(audit_logger=audit_logger, default_timeout=10.0)
        registry = create_default_registry(include_write_tools=True, snapshot_manager=snap_mgr)
        gateway = ToolGateway(
            registry=registry,
            audit_logger=audit_logger,
            workspace_root=workspace,
            path_guard=PathGuard(workspace),
            approval_manager=approval_mgr,
        )

        model_cfg = ModelConfig(
            alias="local-coder",
            model="ollama/qwen2.5-coder:7b",
            tool_mode="native",
            temperature=0.0,
            max_tokens=500,
        )

        # -------------------------------------------------------------
        # TEST 1: write_file pauses at L2, user Approves, write executes
        # -------------------------------------------------------------
        session1 = Session(session_id="sess-live-approve", workspace_root=str(workspace))
        session1.set_autonomy_level(2)
        target_file1 = workspace / "approved_live.txt"

        mock_conn1 = MockConn()
        conn_wrapper1 = ConnectionWrapper(mock_conn1)
        approval_mgr.set_event_sink(lambda p: conn_wrapper1.send(json.dumps(p)))

        step_counter = 0

        async def fake_chat_approve(**kwargs: Any) -> FakeResponse:
            nonlocal step_counter
            step_counter += 1
            if step_counter == 1:
                return FakeResponse(
                    content="I will write the file.",
                    tool_calls=[
                        FakeToolCall(
                            call_id="call_app_1",
                            name="write_file",
                            arguments=json.dumps({"path": str(target_file1), "content": "Live Approved Content"}),
                        )
                    ],
                )
            return FakeResponse(content="File has been written successfully.")

        async def approve_background_client():
            # Poll for approval_request in mock_conn1
            for _ in range(100):
                await asyncio.sleep(0.05)
                reqs = mock_conn1.get_approval_requests()
                if reqs:
                    req = reqs[0]
                    print(f"  [MockClient] Received approval_request: id={req['request_id']} tool={req['tool']}")
                    assert req["tool"] == "write_file"
                    assert req["path"] == str(target_file1)
                    assert req["bytes"] == len("Live Approved Content".encode("utf-8"))
                    # Send approval response
                    success = await approval_mgr.submit_response(
                        request_id=req["request_id"],
                        decision="approve",
                        session_id=session1.session_id,
                    )
                    assert success is True
                    print(f"  [MockClient] Approved request {req['request_id']}")
                    return
            raise TimeoutError("Approval request was not received in time")

        async def run_approve_flow():
            client_task = asyncio.create_task(approve_background_client())
            session1.add_user_message("Please write the approved file.")
            with patch("agent_core.server.acompletion_with_tools", side_effect=fake_chat_approve):
                await run_chat_generation(
                    session=session1,
                    model_config=model_cfg,
                    request_id="req-live-app-1",
                    conn_wrapper=conn_wrapper1,
                    logger=logger,
                    tool_gateway=gateway,
                )
            await client_task

        asyncio.run(run_approve_flow())

        assert target_file1.exists(), "Target file must exist after approval"
        assert target_file1.read_text(encoding="utf-8") == "Live Approved Content"
        print("  [PASS] Test 1 Passed: Write paused at L2, approved, and executed.\n")
        results["test1_live_approval_granted"] = "PASS"

        # -------------------------------------------------------------
        # TEST 2: write_file pauses at L2, user Rejects, write denied
        # -------------------------------------------------------------
        session2 = Session(session_id="sess-live-reject", workspace_root=str(workspace))
        session2.set_autonomy_level(2)
        target_file2 = workspace / "rejected_live.txt"

        mock_conn2 = MockConn()
        conn_wrapper2 = ConnectionWrapper(mock_conn2)
        approval_mgr.set_event_sink(lambda p: conn_wrapper2.send(json.dumps(p)))

        step_counter2 = 0

        async def fake_chat_reject(**kwargs: Any) -> FakeResponse:
            nonlocal step_counter2
            step_counter2 += 1
            if step_counter2 == 1:
                return FakeResponse(
                    content="I will attempt to write.",
                    tool_calls=[
                        FakeToolCall(
                            call_id="call_rej_1",
                            name="write_file",
                            arguments=json.dumps({"path": str(target_file2), "content": "Live Rejected Content"}),
                        )
                    ],
                )
            return FakeResponse(content="Understood, write was cancelled.")

        async def reject_background_client():
            for _ in range(100):
                await asyncio.sleep(0.05)
                reqs = mock_conn2.get_approval_requests()
                if reqs:
                    req = reqs[0]
                    print(f"  [MockClient] Received approval_request: id={req['request_id']} tool={req['tool']}")
                    success = await approval_mgr.submit_response(
                        request_id=req["request_id"],
                        decision="reject",
                        session_id=session2.session_id,
                    )
                    assert success is True
                    print(f"  [MockClient] Rejected request {req['request_id']}")
                    return
            raise TimeoutError("Approval request was not received in time")

        async def run_reject_flow():
            client_task = asyncio.create_task(reject_background_client())
            session2.add_user_message("Please write the rejected file.")
            with patch("agent_core.server.acompletion_with_tools", side_effect=fake_chat_reject):
                await run_chat_generation(
                    session=session2,
                    model_config=model_cfg,
                    request_id="req-live-rej-1",
                    conn_wrapper=conn_wrapper2,
                    logger=logger,
                    tool_gateway=gateway,
                )
            await client_task

        asyncio.run(run_reject_flow())

        assert not target_file2.exists(), "Target file must NOT exist after rejection"
        print("  [PASS] Test 2 Passed: Write paused at L2, rejected, and aborted safely.\n")
        results["test2_live_approval_rejected"] = "PASS"

        # -------------------------------------------------------------
        # TEST 3: Verify Audit Log Hash Chain contains approval entries
        # -------------------------------------------------------------
        lines = [json.loads(l) for l in audit_file.read_text(encoding="utf-8").strip().splitlines()]
        outcomes = [l["outcome"] for l in lines]
        assert "approval_requested" in outcomes
        assert "approval_granted" in outcomes
        assert "approval_rejected" in outcomes

        for i in range(1, len(lines)):
            assert lines[i]["previous_hash"] == lines[i - 1]["hash"]
        print("  [PASS] Test 3 Passed: Tamper-evident audit log contains approval chain.\n")
        results["test3_audit_chain_verified"] = "PASS"

    print("=" * 70)
    print("ALL PHASE 8 PART B LIVE TESTS PASSED!")
    print("=" * 70)
    return results


if __name__ == "__main__":
    res = run_live_test_phase8_b()
    print("\nSummary Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
