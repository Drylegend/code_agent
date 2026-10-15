"""Security and isolation unit tests for Phase 7 production audit log storage.

Proves:
1. Production audit path is outside the workspace root and resolves under D:\\agent-store\\.
2. Workspace write_file cannot write to or overwrite the audit log (absolute, relative, direct).
3. Workspace apply_patch cannot modify the audit log (absolute, relative, direct).
4. Workspace read_file cannot read the audit log.
5. Workspace file deletion cannot remove the audit log (and no deletion tool is exposed).
6. Trusted audit subsystem still appends valid cryptographic records through ToolGateway.
7. PathGuard explicitly protects the agent store even if a workspace artificially encloses it.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import pytest

from agent_core.config import (
    DEFAULT_AGENT_STORE_DIR,
    ModelConfig,
    ServerConfig,
    find_default_audit_log_path,
    find_default_store_dir,
    load_server_config,
)
from agent_core.server import run_chat_generation
from agent_core.sessions import Session
from agent_core.tools import (
    AuditLogger,
    PathGuard,
    PathGuardError,
    ToolGateway,
    create_default_registry,
)
from agent_core.tools.apply_patch import ApplyPatchInput, ApplyPatchTool, EditSpec
from agent_core.tools.base import ToolContext
from agent_core.tools.read_file import ReadFileInput, ReadFileTool
from agent_core.tools.verify_audit import verify_audit_log
from agent_core.tools.write_file import WriteFileInput, WriteFileTool


def test_01_production_audit_path_outside_workspace_and_resolves_under_agent_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Proves production audit path defaults outside workspace and resolves under D:\\agent-store\\."""
    # Ensure unconfigured environment defaults to D:\agent-store
    monkeypatch.delenv("AGENT_CORE_AUDIT_LOG_PATH", raising=False)
    monkeypatch.delenv("AGENT_CORE_STORE_DIR", raising=False)

    store_dir = find_default_store_dir()
    assert store_dir == DEFAULT_AGENT_STORE_DIR
    assert store_dir == Path(r"D:\agent-store")

    prod_audit_path = find_default_audit_log_path()
    assert prod_audit_path.is_absolute()
    assert Path(r"D:\agent-store") in prod_audit_path.parents or prod_audit_path.parent == Path(r"D:\agent-store\audit")
    assert prod_audit_path.name == "audit.jsonl"

    # Workspace must NOT contain production audit log
    workspace_root = Path(r"D:\personal_coding_agent\CodingAgent\agent-shell").resolve()
    with pytest.raises(ValueError):
        prod_audit_path.relative_to(workspace_root)

    # ServerConfig defaults audit_log_path to find_default_audit_log_path()
    cfg = ServerConfig(
        host="127.0.0.1",
        port=9999,
        auth_token="dummy-auth-token",
        log_file=tmp_path / "server.log",
        models={"local-fast": ModelConfig(alias="local-fast", model="ollama/qwen2.5:3b")},
    )
    assert cfg.audit_log_path == prod_audit_path
    assert cfg.audit_log_path.resolve() == prod_audit_path.resolve()


def test_02_audit_path_environment_override_resolves_correctly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Proves AGENT_CORE_AUDIT_LOG_PATH and AGENT_CORE_STORE_DIR allow controlled override."""
    custom_store = tmp_path / "custom_store"
    monkeypatch.setenv("AGENT_CORE_STORE_DIR", str(custom_store))
    monkeypatch.delenv("AGENT_CORE_AUDIT_LOG_PATH", raising=False)

    assert find_default_store_dir() == custom_store.resolve()
    assert find_default_audit_log_path() == (custom_store / "audit" / "audit.jsonl").resolve()

    custom_audit = tmp_path / "custom_audit" / "audit.jsonl"
    monkeypatch.setenv("AGENT_CORE_AUDIT_LOG_PATH", str(custom_audit))
    assert find_default_audit_log_path() == custom_audit.resolve()


@pytest.mark.anyio
async def test_03_workspace_write_file_cannot_write_to_audit_log(tmp_path: Path):
    """Proves workspace write_file cannot write to or overwrite the audit log."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external_store = tmp_path / "external_store"
    external_store.mkdir()
    audit_file = external_store / "audit.jsonl"

    audit_logger = AuditLogger(audit_file)
    registry = create_default_registry(include_write_tools=True)
    path_guard = PathGuard(workspace)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace,
        path_guard=path_guard,
    )

    # 1. Attempt write_file with absolute path to audit log
    res_abs = await gateway.invoke(
        session_id="sess-tamper-write",
        tool_name="write_file",
        raw_args={"path": str(audit_file), "content": "TAMPER ATTEMPT 1"},
        workspace_root=workspace,
    )
    assert res_abs.success is False
    assert res_abs.authorized is False
    assert "denied" in (res_abs.denial_reason or "").lower() or "outside workspace" in (res_abs.denial_reason or "").lower()

    # 2. Attempt write_file with overwrite=True to audit log
    res_overwrite = await gateway.invoke(
        session_id="sess-tamper-write",
        tool_name="write_file",
        raw_args={"path": str(audit_file), "content": "TAMPER ATTEMPT 2", "overwrite": True},
        workspace_root=workspace,
    )
    assert res_overwrite.success is False
    assert res_overwrite.authorized is False

    # 3. Attempt relative path traversal from workspace to audit log
    rel_traversal = os.path.relpath(str(audit_file), str(workspace))
    res_rel = await gateway.invoke(
        session_id="sess-tamper-write",
        tool_name="write_file",
        raw_args={"path": rel_traversal, "content": "TAMPER ATTEMPT 3"},
        workspace_root=workspace,
    )
    assert res_rel.success is False
    assert res_rel.authorized is False

    # 4. Attempt direct execution of WriteFileTool bypassing gateway
    write_tool = WriteFileTool()
    tool_res = await write_tool.execute(
        WriteFileInput(path=str(audit_file), content="DIRECT TAMPER"),
        ToolContext(session_id="sess-tamper-write", workspace_root=workspace),
    )
    assert tool_res.success is False
    assert "Path security denied" in (tool_res.error or "")

    # Ensure none of the tamper payloads entered the audit log as file content
    if audit_file.exists():
        raw_content = audit_file.read_text(encoding="utf-8")
        assert "TAMPER ATTEMPT 1" not in raw_content
        assert "TAMPER ATTEMPT 2" not in raw_content
        assert "TAMPER ATTEMPT 3" not in raw_content
        assert "DIRECT TAMPER" not in raw_content


@pytest.mark.anyio
async def test_04_workspace_apply_patch_cannot_modify_audit_log(tmp_path: Path):
    """Proves workspace apply_patch cannot modify the audit log."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external_store = tmp_path / "external_store"
    external_store.mkdir()
    audit_file = external_store / "audit.jsonl"

    audit_logger = AuditLogger(audit_file)
    # Log an initial legitimate event to populate the file
    audit_logger.log(
        session_id="sess-init",
        tool="list_directory",
        args={"path": "."},
        outcome="success",
        duration=0.01,
    )
    original_audit_text = audit_file.read_text(encoding="utf-8")

    registry = create_default_registry(include_write_tools=True)
    path_guard = PathGuard(workspace)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace,
        path_guard=path_guard,
    )

    # 1. Attempt apply_patch with absolute path
    res_abs = await gateway.invoke(
        session_id="sess-tamper-patch",
        tool_name="apply_patch",
        raw_args={
            "path": str(audit_file),
            "edits": [{"old_text": "list_directory", "new_text": "hacked_tool"}],
        },
        workspace_root=workspace,
    )
    assert res_abs.success is False
    assert res_abs.authorized is False

    # 2. Attempt apply_patch with relative traversal path
    rel_traversal = os.path.relpath(str(audit_file), str(workspace))
    res_rel = await gateway.invoke(
        session_id="sess-tamper-patch",
        tool_name="apply_patch",
        raw_args={
            "path": rel_traversal,
            "edits": [{"old_text": "list_directory", "new_text": "hacked_tool"}],
        },
        workspace_root=workspace,
    )
    assert res_rel.success is False
    assert res_rel.authorized is False

    # 3. Attempt direct ApplyPatchTool execution bypassing gateway
    patch_tool = ApplyPatchTool()
    tool_res = await patch_tool.execute(
        ApplyPatchInput(
            path=str(audit_file),
            edits=[EditSpec(old_text="list_directory", new_text="hacked_tool")],
        ),
        ToolContext(session_id="sess-tamper-patch", workspace_root=workspace),
    )
    assert tool_res.success is False
    assert "Path security denied" in (tool_res.error or "")

    # Ensure the original audit record was never patched
    current_audit_text = audit_file.read_text(encoding="utf-8")
    assert "hacked_tool" not in current_audit_text


@pytest.mark.anyio
async def test_05_workspace_read_file_cannot_read_audit_log(tmp_path: Path):
    """Proves workspace read_file cannot inspect or extract the audit log."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external_store = tmp_path / "external_store"
    external_store.mkdir()
    audit_file = external_store / "audit.jsonl"

    audit_logger = AuditLogger(audit_file)
    audit_logger.log(
        session_id="sess-secret-audit",
        tool="test",
        args={"param": "secret_data"},
        outcome="success",
        duration=0.01,
    )

    registry = create_default_registry(include_write_tools=True)
    path_guard = PathGuard(workspace)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace,
        path_guard=path_guard,
    )

    # 1. Attempt gateway read_file with absolute path
    res_abs = await gateway.invoke(
        session_id="sess-inspect",
        tool_name="read_file",
        raw_args={"path": str(audit_file)},
        workspace_root=workspace,
    )
    assert res_abs.success is False
    assert res_abs.authorized is False

    # 2. Attempt direct ReadFileTool execution bypassing gateway
    read_tool = ReadFileTool()
    tool_res = await read_tool.execute(
        ReadFileInput(path=str(audit_file)),
        ToolContext(session_id="sess-inspect", workspace_root=workspace),
    )
    assert tool_res.success is False
    assert "Path validation failed" in (tool_res.error or "") or "Path security denied" in (tool_res.error or "")


def test_06_workspace_tools_do_not_expose_file_deletion():
    """Proves that default production workspace tools do not include any file deletion tools."""
    registry = create_default_registry(include_write_tools=True)
    tool_names = [tool.name for tool in registry.list_tools()]

    # Production tools are strictly list_directory, read_file, write_file, apply_patch, glob, grep
    assert set(tool_names) == {"list_directory", "read_file", "write_file", "apply_patch", "glob", "grep"}
    for dangerous in ("delete_file", "remove_file", "unlink", "rmdir", "delete"):
        assert dangerous not in tool_names


@pytest.mark.anyio
async def test_07_trusted_audit_subsystem_appends_records_and_chain_validates(tmp_path: Path):
    """Proves trusted audit subsystem successfully appends valid cryptographic records through ToolGateway."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external_store = tmp_path / "external_store"
    external_store.mkdir()
    audit_file = external_store / "audit.jsonl"

    audit_logger = AuditLogger(audit_file)
    registry = create_default_registry(include_write_tools=True)
    path_guard = PathGuard(workspace)
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        workspace_root=workspace,
        path_guard=path_guard,
    )

    # 1. Legitimate write inside workspace
    res1 = await gateway.invoke(
        session_id="sess-valid",
        tool_name="write_file",
        raw_args={"path": "hello.txt", "content": "Hello World\n"},
        workspace_root=workspace,
    )
    assert res1.success is True

    # 2. Denied attempt to access external path
    res2 = await gateway.invoke(
        session_id="sess-valid",
        tool_name="write_file",
        raw_args={"path": str(audit_file), "content": "evil"},
        workspace_root=workspace,
    )
    assert res2.success is False

    # 3. Legitimate patch inside workspace
    res3 = await gateway.invoke(
        session_id="sess-valid",
        tool_name="apply_patch",
        raw_args={
            "path": "hello.txt",
            "edits": [{"old_text": "Hello World\n", "new_text": "Hello Antigravity\n"}],
        },
        workspace_root=workspace,
    )
    assert res3.success is True

    # Confirm audit events were appended
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    rec1 = json.loads(lines[0])
    assert rec1["tool"] == "write_file"
    assert rec1["outcome"] == "success"

    rec2 = json.loads(lines[1])
    assert rec2["tool"] == "write_file"
    assert rec2["outcome"] == "denied"

    rec3 = json.loads(lines[2])
    assert rec3["tool"] == "apply_patch"
    assert rec3["outcome"] == "success"

    # Cryptographic verification must pass completely
    valid, errors = verify_audit_log(audit_file)
    assert valid is True
    assert len(errors) == 0


def test_08_path_guard_explicitly_protects_agent_store_even_if_enclosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Proves PathGuard rejects writes into agent store even if workspace is artificially set to an enclosing directory."""
    parent_dir = tmp_path / "enclosing_root"
    workspace = parent_dir / "workspace"
    store_dir = parent_dir / "agent-store"
    workspace.mkdir(parents=True)
    store_dir.mkdir(parents=True)

    monkeypatch.setenv("AGENT_CORE_STORE_DIR", str(store_dir))

    # Artificially set workspace to parent_dir (enclosing both)
    guard = PathGuard(parent_dir)

    # Attempt write into agent store (snapshots)
    with pytest.raises(PathGuardError, match=r"Write access to agent store"):
        guard.validate_write_path(store_dir / "snapshots" / "snap.bin")

    # Attempt write to audit log inside agent store (blocked by audit storage policy)
    with pytest.raises(PathGuardError, match=r"Access to audit log storage is denied"):
        guard.validate_write_path(store_dir / "audit" / "audit.jsonl")

    # Attempt write to audit file name within workspace root
    with pytest.raises(PathGuardError, match=r"protected configuration file 'audit\.jsonl'|audit/log file"):
        guard.validate_write_path(parent_dir / "audit.jsonl")

    # Attempt read from audit log storage
    with pytest.raises(PathGuardError, match=r"Access to audit log storage is denied"):
        guard.validate_path(store_dir / "audit" / "audit.jsonl")
