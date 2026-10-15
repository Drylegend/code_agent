"""Unit tests for Phase 5A: Tool Foundation, PathGuard, and Audit Security."""

import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import pytest
from pydantic import BaseModel, Field

from agent_core.tools.audit import AuditLogger, GENESIS_HASH
from agent_core.tools.base import ToolContext, ToolResult, ToolSpec
from agent_core.tools.gateway import AuthorizationResult, ToolGateway
from agent_core.tools.path_guard import DEFAULT_DENY_PATTERNS, PathGuard, PathGuardError
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.verify_audit import verify_audit_log


# ── Dummy Tool Fixtures for Gateway Tests ─────────────────────────────────────


class DummyReadArgs(BaseModel):
    path: str = Field(..., description="File path to read")


class DummyReadTool(ToolSpec):
    name = "dummy_read"
    description = "Test read-only tool"
    input_model = DummyReadArgs
    read_only = True

    async def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        assert isinstance(args, DummyReadArgs)
        return ToolResult(success=True, output=f"Read file at {args.path}")


class DummyWriteArgs(BaseModel):
    path: str = Field(..., description="File path to write")
    content: str = Field(..., description="Content to write")


class DummyWriteTool(ToolSpec):
    name = "dummy_write"
    description = "Test mutating tool"
    input_model = DummyWriteArgs
    read_only = False

    async def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        return ToolResult(success=True, output="File written")


# ── PathGuard Unit Tests ───────────────────────────────────────────────────────


def test_1_dot_dot_escape(tmp_path: Path):
    guard = PathGuard(tmp_path)
    with pytest.raises(PathGuardError, match=r"escapes workspace root|resolves outside"):
        guard.validate_path(tmp_path / "subdir" / ".." / ".." / "outside.txt")


def test_2_absolute_path_outside_workspace(tmp_path: Path):
    guard = PathGuard(tmp_path)
    outside_dir = tmp_path.parent / "completely_outside_dir"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("outside")

    with pytest.raises(PathGuardError, match=r"resolves outside workspace root"):
        guard.validate_path(outside_file)


def test_3_windows_drive_switch(tmp_path: Path):
    guard = PathGuard(tmp_path)
    # Check current drive, pick a different drive letter
    curr_drive = tmp_path.resolve().drive.upper()
    alt_drive = "C:" if curr_drive != "C:" else "D:"
    fake_alt_path = f"{alt_drive}\\Windows\\System32\\calc.exe"

    with pytest.raises(PathGuardError, match=r"different drive"):
        guard.validate_path(fake_alt_path)


def test_4_unc_path_outside_workspace(tmp_path: Path):
    guard = PathGuard(tmp_path)
    unc_path = r"\\fake-server\share\secret.txt"
    with pytest.raises(PathGuardError, match=r"UNC path outside the workspace"):
        guard.validate_path(unc_path)


def test_5_symlink_pointing_outside_workspace(tmp_path: Path):
    guard = PathGuard(tmp_path)
    outside_file = tmp_path.parent / "outside_target.txt"
    outside_file.write_text("secret target")

    symlink_path = tmp_path / "leak_link.txt"
    try:
        os.symlink(outside_file, symlink_path)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip(
                "Skipping symlink test: Windows requires Developer Mode or elevated privileges for os.symlink"
            )
        raise

    with pytest.raises(PathGuardError, match=r"resolves outside workspace root"):
        guard.validate_path(symlink_path)


def test_6_windows_junction_pointing_outside_workspace(tmp_path: Path):
    guard = PathGuard(tmp_path)
    outside_dir = tmp_path.parent / "outside_junction_target"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "target_file.txt").write_text("hello")

    junction_path = tmp_path / "inside_junction"

    # On Windows, create a directory junction via _winapi
    try:
        import _winapi

        _winapi.CreateJunction(str(outside_dir), str(junction_path))
    except (ImportError, AttributeError, OSError) as exc:
        pytest.skip(f"Skipping junction test: directory junction creation unavailable ({exc})")

    try:
        # Accessing a file through the junction should be detected as pointing outside workspace
        with pytest.raises(PathGuardError, match=r"resolves outside workspace root"):
            guard.validate_path(junction_path / "target_file.txt")
    finally:
        # Cleanup junction
        try:
            os.rmdir(junction_path)
        except Exception:
            pass


def test_7_denylisted_env_files(tmp_path: Path):
    guard = PathGuard(tmp_path)
    with pytest.raises(PathGuardError, match=r"protected file pattern '\.env\*'"):
        guard.validate_path(".env")

    with pytest.raises(PathGuardError, match=r"protected file pattern '\.env\*'"):
        guard.validate_path("subfolder/.env.production")


def test_8_denylisted_pem_files(tmp_path: Path):
    guard = PathGuard(tmp_path)
    with pytest.raises(PathGuardError, match=r"protected file pattern '\*\.pem'"):
        guard.validate_path("certs/server.pem")


def test_9_denylisted_credentials_and_secrets(tmp_path: Path):
    guard = PathGuard(tmp_path)
    with pytest.raises(PathGuardError, match=r"protected file pattern 'credentials\*'"):
        guard.validate_path("credentials.json")

    with pytest.raises(PathGuardError, match=r"protected file pattern 'secrets\*'"):
        guard.validate_path("config/secrets.yaml")

    with pytest.raises(PathGuardError, match=r"protected path pattern '\.ssh'|id_rsa"):
        guard.validate_path(".ssh/id_rsa")


def test_10_valid_path_inside_workspace(tmp_path: Path):
    guard = PathGuard(tmp_path)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    code_file = src_dir / "app.py"
    code_file.write_text("print('hello')")

    resolved = guard.validate_path("src/app.py")
    assert resolved == code_file.resolve()
    assert resolved.exists()


def test_11_resolved_symlink_inside_workspace(tmp_path: Path):
    guard = PathGuard(tmp_path)
    real_file = tmp_path / "real.txt"
    real_file.write_text("internal content")

    link_path = tmp_path / "link_internal.txt"
    try:
        os.symlink(real_file, link_path)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip(
                "Skipping symlink test: Windows requires Developer Mode or elevated privileges for os.symlink"
            )
        raise

    resolved = guard.validate_path(link_path)
    assert resolved == real_file.resolve()


# ── ToolRegistry Unit Tests ────────────────────────────────────────────────────


def test_12_registry_duplicate_tool_rejection():
    registry = ToolRegistry()
    tool1 = DummyReadTool()
    tool2 = DummyReadTool()

    registry.register(tool1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool2)


# ── ToolGateway Unit Tests ─────────────────────────────────────────────────────


def test_13_invalid_pydantic_arguments_rejected_by_gateway(tmp_path: Path):
    async def run():
        audit_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(audit_file)
        registry = ToolRegistry()
        registry.register(DummyReadTool())

        gateway = ToolGateway(
            registry=registry,
            audit_logger=audit,
            workspace_root=tmp_path,
        )

        # Missing required argument 'path'
        res = await gateway.invoke("sess-1", "dummy_read", {})
        assert res.success is False
        assert res.authorized is False
        assert "Argument validation error" in (res.error or "")

        # Verify audit record was written as denied
        valid, errors = verify_audit_log(audit_file)
        assert valid is True, errors

    asyncio.run(run())


def test_14_unauthorized_tool_rejected_by_gateway(tmp_path: Path):
    async def run():
        audit_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(audit_file)
        registry = ToolRegistry()
        registry.register(DummyWriteTool())  # read_only = False

        gateway = ToolGateway(
            registry=registry,
            audit_logger=audit,
            workspace_root=tmp_path,
        )

        res = await gateway.invoke(
            "sess-1", "dummy_write", {"path": "file.txt", "content": "data"}
        )
        assert res.success is False
        assert res.authorized is False
        assert "write/action tools are currently disabled" in (res.denial_reason or "")

    asyncio.run(run())


# ── AuditLogger & Hash Chain Unit Tests ────────────────────────────────────────


def test_15_audit_record_is_written(tmp_path: Path):
    async def run():
        audit_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(audit_file)
        registry = ToolRegistry()
        registry.register(DummyReadTool())

        gateway = ToolGateway(
            registry=registry,
            audit_logger=audit,
            workspace_root=tmp_path,
        )

        valid_file = tmp_path / "hello.txt"
        valid_file.write_text("content")

        res = await gateway.invoke("sess-123", "dummy_read", {"path": "hello.txt"})
        assert res.success is True

        # Check file exists and has 1 line
        assert audit_file.exists()
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        rec = json.loads(lines[0])
        assert rec["session_id"] == "sess-123"
        assert rec["tool"] == "dummy_read"
        assert rec["outcome"] == "success"
        assert rec["previous_hash"] == GENESIS_HASH
        assert len(rec["hash"]) == 64

    asyncio.run(run())


def test_16_audit_hash_chain_validates(tmp_path: Path):
    async def run():
        audit_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(audit_file)
        registry = ToolRegistry()
        registry.register(DummyReadTool())

        gateway = ToolGateway(
            registry=registry,
            audit_logger=audit,
            workspace_root=tmp_path,
        )

        # Perform multiple operations
        f1 = tmp_path / "a.txt"
        f1.write_text("a")
        f2 = tmp_path / "b.txt"
        f2.write_text("b")

        await gateway.invoke("sess-1", "dummy_read", {"path": "a.txt"})
        await gateway.invoke("sess-1", "dummy_read", {"path": "b.txt"})
        await gateway.invoke("sess-1", "dummy_read", {"path": "nonexistent.txt"})

        valid, errors = verify_audit_log(audit_file)
        assert valid is True, f"Audit chain validation failed: {errors}"

    asyncio.run(run())


def test_17_tampered_audit_record_is_detected(tmp_path: Path):
    async def run():
        audit_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(audit_file)
        registry = ToolRegistry()
        registry.register(DummyReadTool())

        gateway = ToolGateway(
            registry=registry,
            audit_logger=audit,
            workspace_root=tmp_path,
        )

        f1 = tmp_path / "file.txt"
        f1.write_text("data")

        await gateway.invoke("sess-1", "dummy_read", {"path": "file.txt"})
        await gateway.invoke("sess-1", "dummy_read", {"path": "file.txt"})

        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        # Tamper with the first record's outcome
        rec1 = json.loads(lines[0])
        rec1["outcome"] = "tampered_outcome"
        lines[0] = json.dumps(rec1)

        audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Verification must detect the tampering
        valid, errors = verify_audit_log(audit_file)
        assert valid is False
        assert any("Tamper detected" in err or "Broken chain" in err for err in errors)

    asyncio.run(run())
