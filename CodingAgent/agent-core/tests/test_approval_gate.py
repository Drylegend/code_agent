import asyncio
import hashlib
import json
import pytest
from pathlib import Path
from agent_core.sessions import Session
from agent_core.tools.registry import create_default_registry
from agent_core.tools.gateway import ToolGateway
from agent_core.tools.approval import ApprovalManager
from agent_core.tools.audit import AuditLogger


@pytest.fixture
def temp_workspace(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture
def temp_audit_log(tmp_path: Path):
    audit_file = tmp_path / "audit" / "audit.jsonl"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    return AuditLogger(log_path=str(audit_file))


@pytest.mark.anyio
async def test_approval_approve(temp_workspace: Path, temp_audit_log: AuditLogger):
    """Test write_file at Level 2 is paused and proceeds when user approves."""
    approval_manager = ApprovalManager(audit_logger=temp_audit_log, default_timeout=5.0)
    registry = create_default_registry(include_write_tools=True, audit_logger=temp_audit_log)
    gateway = ToolGateway(registry=registry, audit_logger=temp_audit_log, approval_manager=approval_manager)

    session = Session(session_id="sess-approve-1", workspace_root=str(temp_workspace))
    session.set_autonomy_level(2)

    emitted_requests = []
    approval_manager.set_emitter(lambda req: emitted_requests.append(req))

    target_file = temp_workspace / "approved.txt"

    async def approve_task():
        while not emitted_requests:
            await asyncio.sleep(0.01)
        req = emitted_requests[0]
        assert req.tool_name == "write_file"
        assert req.path == str(target_file)
        success = await approval_manager.submit_response(
            request_id=req.request_id,
            decision="approve",
            session_id=session.session_id,
        )
        assert success is True

    approve_bg = asyncio.create_task(approve_task())

    result = await gateway.invoke(
        session_id=session.session_id,
        tool_name="write_file",
        raw_args={"path": str(target_file), "content": "Approved Content"},
        workspace_root=str(temp_workspace),
        autonomy_level=session.autonomy_level,
    )
    await approve_bg

    assert result.success is True
    assert target_file.read_text(encoding="utf-8") == "Approved Content"


@pytest.mark.anyio
async def test_approval_reject(temp_workspace: Path, temp_audit_log: AuditLogger):
    """Test write_file at Level 2 is paused and denied when user rejects."""
    approval_manager = ApprovalManager(audit_logger=temp_audit_log, default_timeout=5.0)
    registry = create_default_registry(include_write_tools=True, audit_logger=temp_audit_log)
    gateway = ToolGateway(registry=registry, audit_logger=temp_audit_log, approval_manager=approval_manager)

    session = Session(session_id="sess-reject-1", workspace_root=str(temp_workspace))
    session.set_autonomy_level(2)

    emitted_requests = []
    approval_manager.set_emitter(lambda req: emitted_requests.append(req))

    target_file = temp_workspace / "rejected.txt"

    async def reject_task():
        while not emitted_requests:
            await asyncio.sleep(0.01)
        req = emitted_requests[0]
        success = await approval_manager.submit_response(
            request_id=req.request_id,
            decision="reject",
            session_id=session.session_id,
        )
        assert success is True

    reject_bg = asyncio.create_task(reject_task())

    result = await gateway.invoke(
        session_id=session.session_id,
        tool_name="write_file",
        raw_args={"path": str(target_file), "content": "Rejected Content"},
        workspace_root=str(temp_workspace),
        autonomy_level=session.autonomy_level,
    )
    await reject_bg

    assert result.success is False
    assert "rejected by user" in (result.denial_reason or "").lower() or "rejected by user" in (result.error or "").lower()
    assert not target_file.exists()


@pytest.mark.anyio
async def test_approval_timeout(temp_workspace: Path, temp_audit_log: AuditLogger):
    """Test approval request fails safe on timeout."""
    approval_manager = ApprovalManager(audit_logger=temp_audit_log, default_timeout=0.1)
    registry = create_default_registry(include_write_tools=True, audit_logger=temp_audit_log)
    gateway = ToolGateway(registry=registry, audit_logger=temp_audit_log, approval_manager=approval_manager)

    session = Session(session_id="sess-timeout-1", workspace_root=str(temp_workspace))
    session.set_autonomy_level(2)

    target_file = temp_workspace / "timed_out.txt"

    result = await gateway.invoke(
        session_id=session.session_id,
        tool_name="write_file",
        raw_args={"path": str(target_file), "content": "Timeout Content"},
        workspace_root=str(temp_workspace),
        autonomy_level=session.autonomy_level,
    )

    assert result.success is False
    err_str = f"{result.denial_reason} {result.error}".lower()
    assert "timed out" in err_str or "rejected" in err_str or "denied" in err_str
    assert not target_file.exists()


@pytest.mark.anyio
async def test_approval_cancel(temp_workspace: Path, temp_audit_log: AuditLogger):
    """Test cancellation (Stop button) aborts pending approval."""
    approval_manager = ApprovalManager(audit_logger=temp_audit_log, default_timeout=5.0)
    registry = create_default_registry(include_write_tools=True, audit_logger=temp_audit_log)
    gateway = ToolGateway(registry=registry, audit_logger=temp_audit_log, approval_manager=approval_manager)

    session = Session(session_id="sess-cancel-1", workspace_root=str(temp_workspace))
    session.set_autonomy_level(2)

    emitted_requests = []
    approval_manager.set_emitter(lambda req: emitted_requests.append(req))

    target_file = temp_workspace / "cancelled.txt"

    async def cancel_task():
        while not emitted_requests:
            await asyncio.sleep(0.01)
        # Simulate user pressing Stop button
        approval_manager.cancel_session_requests(session.session_id)

    cancel_bg = asyncio.create_task(cancel_task())

    result = await gateway.invoke(
        session_id=session.session_id,
        tool_name="write_file",
        raw_args={"path": str(target_file), "content": "Cancelled Content"},
        workspace_root=str(temp_workspace),
        autonomy_level=session.autonomy_level,
    )
    await cancel_bg

    assert result.success is False
    err_str = f"{result.denial_reason} {result.error}".lower()
    assert "cancelled" in err_str or "rejected" in err_str or "denied" in err_str
    assert not target_file.exists()


@pytest.mark.anyio
async def test_approval_stale_hash(temp_workspace: Path, temp_audit_log: AuditLogger):
    """Test that submitting approval with altered arguments hash is rejected."""
    approval_manager = ApprovalManager(audit_logger=temp_audit_log, default_timeout=5.0)

    raw_args = {"path": str(temp_workspace / "foo.txt"), "content": "Original"}
    norm_str = json.dumps(raw_args, sort_keys=True, separators=(",", ":"))
    real_hash = hashlib.sha256(norm_str.encode("utf-8")).hexdigest()

    req_task = asyncio.create_task(
        approval_manager.request_approval(
            session_id="sess-stale",
            tool_name="write_file",
            validated_args=raw_args,
            path=raw_args["path"],
            timeout=5.0,
        )
    )
    await asyncio.sleep(0.01)
    req_id = list(approval_manager._pending_requests.keys())[0]

    tampered_hash = hashlib.sha256(b'tampered').hexdigest()
    # Attempt to submit approval with tampered hash
    success = await approval_manager.submit_response(
        request_id=req_id,
        decision="approve",
        session_id="sess-stale",
        args_hash=tampered_hash,
    )
    assert success is False
    assert not req_task.done()

    # Now submit with correct hash
    success2 = await approval_manager.submit_response(
        request_id=req_id,
        decision="approve",
        session_id="sess-stale",
        args_hash=real_hash,
    )
    assert success2 is True
    res = await req_task
    assert res is True


@pytest.mark.anyio
async def test_approval_replayed_request_id(temp_workspace: Path, temp_audit_log: AuditLogger):
    """Test that a request_id cannot be reused or replayed once consumed."""
    approval_manager = ApprovalManager(audit_logger=temp_audit_log, default_timeout=5.0)

    req_task = asyncio.create_task(
        approval_manager.request_approval(
            session_id="sess-replay",
            tool_name="write_file",
            validated_args={"foo": "bar"},
            path="some/file.txt",
            timeout=5.0,
        )
    )
    await asyncio.sleep(0.01)
    req_id = list(approval_manager._pending_requests.keys())[0]

    first = await approval_manager.submit_response(
        request_id=req_id,
        decision="approve",
        session_id="sess-replay",
    )
    assert first is True
    assert await req_task is True

    # Replay
    second = await approval_manager.submit_response(
        request_id=req_id,
        decision="approve",
        session_id="sess-replay",
    )
    assert second is False


@pytest.mark.anyio
async def test_approval_model_isolation():
    """Verify that approval and undo methods cannot be called as tools by the model."""
    registry = create_default_registry()
    tools = registry.list_tools()

    tool_names = [t.name for t in tools]
    assert "approval.response" not in tool_names
    assert "approval_response" not in tool_names
    assert "changes.undo" not in tool_names
    assert "undo" not in tool_names

    assert registry.get("approval.response") is None
    assert registry.get("approval_response") is None
    assert registry.get("changes.undo") is None


@pytest.mark.anyio
async def test_autonomy_level_scoped_auto_approves(temp_workspace: Path, temp_audit_log: AuditLogger):
    """Test that Level 3 (Scoped) auto-approves workspace writes without waiting for approval."""
    approval_manager = ApprovalManager(audit_logger=temp_audit_log, default_timeout=1.0)
    registry = create_default_registry(include_write_tools=True, audit_logger=temp_audit_log)
    gateway = ToolGateway(registry=registry, audit_logger=temp_audit_log, approval_manager=approval_manager)

    session = Session(session_id="sess-scoped-1", workspace_root=str(temp_workspace))
    session.set_autonomy_level(3)

    emitted_requests = []
    approval_manager.set_emitter(lambda req: emitted_requests.append(req))

    target_file = temp_workspace / "scoped_write.txt"
    result = await gateway.invoke(
        session_id=session.session_id,
        tool_name="write_file",
        raw_args={"path": str(target_file), "content": "Auto Approved In Scoped"},
        workspace_root=str(temp_workspace),
        autonomy_level=session.autonomy_level,
    )

    assert result.success is True
    assert len(emitted_requests) == 0  # No approval request emitted!
    assert target_file.read_text(encoding="utf-8") == "Auto Approved In Scoped"


@pytest.mark.anyio
async def test_approval_audit_logging(temp_workspace: Path, tmp_path: Path):
    """Test that approval requests, approvals, rejections, and timeouts are logged to the audit chain."""
    audit_file = tmp_path / "audit" / "audit.jsonl"
    audit_logger = AuditLogger(log_path=str(audit_file))
    approval_manager = ApprovalManager(audit_logger=audit_logger, default_timeout=0.1)

    # 1. Request + Approve
    t1 = asyncio.create_task(
        approval_manager.request_approval(
            session_id="sess-audit-1",
            tool_name="write_file",
            validated_args={"p": "1"},
            path="test1.txt",
            timeout=5.0,
        )
    )
    await asyncio.sleep(0.01)
    req1_id = list(approval_manager._pending_requests.keys())[0]
    await approval_manager.submit_response(req1_id, "approve", "sess-audit-1")
    await t1

    # 2. Request + Reject
    t2 = asyncio.create_task(
        approval_manager.request_approval(
            session_id="sess-audit-2",
            tool_name="apply_patch",
            validated_args={"p": "2"},
            path="test2.txt",
            timeout=5.0,
        )
    )
    await asyncio.sleep(0.01)
    req2_id = list(approval_manager._pending_requests.keys())[0]
    await approval_manager.submit_response(req2_id, "reject", "sess-audit-2")
    await t2

    # 3. Request + Timeout
    t3 = asyncio.create_task(
        approval_manager.request_approval(
            session_id="sess-audit-3",
            tool_name="write_file",
            validated_args={"p": "3"},
            path="test3.txt",
            timeout=0.05,
        )
    )
    res3 = await t3
    assert res3 is False

    # Verify audit log entries
    assert audit_file.exists()
    lines = [json.loads(l) for l in audit_file.read_text(encoding="utf-8").strip().splitlines()]
    outcomes = [l["outcome"] for l in lines]

    assert "approval_requested" in outcomes
    assert "approval_granted" in outcomes
    assert "approval_rejected" in outcomes
    assert "approval_expired" in outcomes

    # Check hash chaining integrity
    for i in range(1, len(lines)):
        assert lines[i]["previous_hash"] == lines[i - 1]["hash"]
