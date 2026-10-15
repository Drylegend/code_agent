"""Unit and integration tests for Phase 7: Safe File Editing & Snapshots.

Tests all security boundaries, write constraints, patch semantics,
Windows fidelity (CRLF/BOM), atomicity, snapshot restoration, and audit logging.
"""

from pathlib import Path
import pytest

from agent_core.snapshots import SnapshotManager, restore_snapshot
from agent_core.tools.audit import AuditLogger
from agent_core.tools.base import ToolContext
from agent_core.tools.gateway import ToolGateway
from agent_core.tools.path_guard import PathGuard
from agent_core.tools.registry import ToolRegistry, create_default_registry
from agent_core.tools.verify_audit import verify_audit_log
from agent_core.tools.write_file import WriteFileInput, WriteFileTool
from agent_core.tools.apply_patch import ApplyPatchInput, ApplyPatchTool, EditSpec as PatchEdit
from agent_core.tools.read_file import ReadFileInput, ReadFileTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def agent_store(tmp_path: Path) -> Path:
    store = tmp_path / "agent_store"
    store.mkdir()
    return store


@pytest.fixture
def snapshot_manager(agent_store: Path) -> SnapshotManager:
    return SnapshotManager(store_dir=agent_store)


@pytest.fixture
def write_tool(snapshot_manager: SnapshotManager) -> WriteFileTool:
    return WriteFileTool(snapshot_manager=snapshot_manager)


@pytest.fixture
def patch_tool(snapshot_manager: SnapshotManager) -> ApplyPatchTool:
    return ApplyPatchTool(snapshot_manager=snapshot_manager)


@pytest.fixture
def read_tool() -> ReadFileTool:
    return ReadFileTool()


@pytest.fixture
def gateway(
    workspace: Path,
    agent_store: Path,
    write_tool: WriteFileTool,
    patch_tool: ApplyPatchTool,
    read_tool: ReadFileTool,
) -> ToolGateway:
    registry = ToolRegistry()
    registry.register(read_tool)
    registry.register(write_tool)
    registry.register(patch_tool)
    audit_log = workspace / ".agent_audit.jsonl"
    logger = AuditLogger(audit_log)
    guard = PathGuard(workspace)
    return ToolGateway(
        registry=registry,
        audit_logger=logger,
        workspace_root=workspace,
        path_guard=guard,
    )


# ── WriteFileTool Tests (1 - 16) ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_01_write_file_creates_new_file(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    args = WriteFileInput(path="hello.txt", content="Hello, World!\n")
    res = await write_tool.execute(args, ctx)

    assert res.success is True
    target = workspace / "hello.txt"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "Hello, World!\n"
    assert res.output["created"] is True
    assert res.output["bytes_written"] == len("Hello, World!\n")
    assert res.output["audit_metadata"]["bytes_before"] == 0
    assert res.output["audit_metadata"]["bytes_after"] == len("Hello, World!\n")


@pytest.mark.anyio
async def test_02_write_file_fails_if_exists_without_overwrite(
    workspace: Path, write_tool: WriteFileTool
):
    target = workspace / "existing.txt"
    target.write_text("initial content", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    args = WriteFileInput(path="existing.txt", content="new content", overwrite=False)
    res = await write_tool.execute(args, ctx)

    assert res.success is False
    assert "File already exists" in res.error
    assert "overwrite=true" in res.error.lower()
    assert target.read_text(encoding="utf-8") == "initial content"


@pytest.mark.anyio
async def test_03_write_file_succeeds_with_overwrite(
    workspace: Path, write_tool: WriteFileTool
):
    target = workspace / "existing.txt"
    target.write_text("initial content", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    args = WriteFileInput(path="existing.txt", content="new content", overwrite=True)
    res = await write_tool.execute(args, ctx)

    assert res.success is True
    assert res.output["created"] is False
    assert res.output["bytes_written"] == len("new content")
    assert target.read_text(encoding="utf-8") == "new content"
    assert res.output["audit_metadata"]["bytes_before"] == len("initial content")


@pytest.mark.anyio
async def test_04_write_file_fails_if_parent_dir_missing(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    args = WriteFileInput(path="nonexistent/dir/file.txt", content="content")
    res = await write_tool.execute(args, ctx)

    assert res.success is False
    assert "Parent directory does not exist" in res.error


@pytest.mark.anyio
async def test_05_write_file_succeeds_in_existing_subdir(
    workspace: Path, write_tool: WriteFileTool
):
    sub = workspace / "sub"
    sub.mkdir()

    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    args = WriteFileInput(path="sub/file.txt", content="in sub")
    res = await write_tool.execute(args, ctx)

    assert res.success is True
    assert (sub / "file.txt").read_text(encoding="utf-8") == "in sub"


@pytest.mark.anyio
async def test_06_write_file_rejects_content_over_1mb(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    large_content = "x" * (1024 * 1024 + 1)
    args = WriteFileInput(path="large.txt", content=large_content)
    res = await write_tool.execute(args, ctx)

    assert res.success is False
    assert "exceeds 1 mb limit" in res.error.lower()
    assert not (workspace / "large.txt").exists()


@pytest.mark.anyio
async def test_07_write_file_rejects_path_traversal(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    args = WriteFileInput(path="../../escape.txt", content="escape")
    res = await write_tool.execute(args, ctx)

    assert res.success is False
    assert "escapes workspace" in res.error.lower() or "outside" in res.error.lower()


@pytest.mark.anyio
async def test_08_write_file_rejects_absolute_path_outside_workspace(
    workspace: Path, tmp_path: Path, write_tool: WriteFileTool
):
    outside = tmp_path / "outside.txt"
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    args = WriteFileInput(path=str(outside), content="escape")
    res = await write_tool.execute(args, ctx)

    assert res.success is False
    assert "outside" in res.error.lower() or "escapes" in res.error.lower()


@pytest.mark.anyio
async def test_09_write_file_rejects_denylist(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    for denied in [".env", "id_rsa", "server.key", "cert.pem"]:
        args = WriteFileInput(path=denied, content="secret")
        res = await write_tool.execute(args, ctx)
        assert res.success is False
        assert "denied" in res.error.lower() or "protected" in res.error.lower()


@pytest.mark.anyio
async def test_10_write_file_rejects_protected_dirs(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    for protected in [
        ".git/hooks/pre-commit",
        ".vscode/settings.json",
        ".github/workflows/ci.yml",
        ".husky/pre-commit",
        ".gitlab-ci.yml",
        ".pre-commit-config.yaml",
    ]:
        args = WriteFileInput(path=protected, content="override")
        res = await write_tool.execute(args, ctx)
        assert res.success is False
        assert "protected" in res.error.lower()


@pytest.mark.anyio
async def test_11_write_file_rejects_windows_reserved_names(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    for r_name in ["CON", "prn", "AUX.txt", "nul", "COM1", "lpt9"]:
        args = WriteFileInput(path=r_name, content="data")
        res = await write_tool.execute(args, ctx)
        assert res.success is False
        assert "reserved windows device name" in res.error.lower()


@pytest.mark.anyio
async def test_12_write_file_rejects_alternate_data_streams(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    args = WriteFileInput(path="test.txt:stream", content="ads")
    res = await write_tool.execute(args, ctx)
    assert res.success is False
    assert "alternate data stream" in res.error.lower()


@pytest.mark.anyio
async def test_13_write_file_rejects_trailing_dots_and_spaces(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="test-session")
    for invalid in ["test.txt.", "test.txt "]:
        args = WriteFileInput(path=invalid, content="data")
        res = await write_tool.execute(args, ctx)
        assert res.success is False
        assert "trailing dots or spaces" in res.error.lower()


@pytest.mark.anyio
async def test_14_write_file_creates_pre_write_snapshot_on_overwrite(
    workspace: Path, write_tool: WriteFileTool, snapshot_manager: SnapshotManager
):
    target = workspace / "data.txt"
    target.write_text("v1 content", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-snap-1")
    args = WriteFileInput(path="data.txt", content="v2 content", overwrite=True)
    res = await write_tool.execute(args, ctx)

    assert res.success is True
    snapshots = snapshot_manager.list_snapshots("sess-snap-1")
    assert len(snapshots) == 1
    assert snapshots[0]["path"] == "data.txt"
    assert snapshots[0]["tool"] == "write_file"
    assert snapshots[0]["existed"] is True

    # Check snapshot content matches v1
    snap_file = snapshot_manager.snapshots_dir / "sess-snap-1" / snapshots[0]["snapshot_file"]
    assert snap_file.read_text(encoding="utf-8") == "v1 content"


@pytest.mark.anyio
async def test_15_write_file_records_new_file_in_snapshot_index(
    workspace: Path, write_tool: WriteFileTool, snapshot_manager: SnapshotManager
):
    ctx = ToolContext(workspace_root=workspace, session_id="sess-new-1")
    args = WriteFileInput(path="brand_new.txt", content="first version")
    res = await write_tool.execute(args, ctx)

    assert res.success is True
    snapshots = snapshot_manager.list_snapshots("sess-new-1")
    assert len(snapshots) == 1
    assert snapshots[0]["existed"] is False
    assert snapshots[0]["snapshot_file"] is None


@pytest.mark.anyio
async def test_16_write_file_preserves_crlf_on_existing_file(
    workspace: Path, write_tool: WriteFileTool
):
    target = workspace / "crlf.txt"
    target.write_bytes(b"line 1\r\nline 2\r\n")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-crlf")
    # Provide content with LF; write_file should preserve target's CRLF
    args = WriteFileInput(
        path="crlf.txt", content="new line 1\nnew line 2\n", overwrite=True
    )
    res = await write_tool.execute(args, ctx)

    assert res.success is True
    data = target.read_bytes()
    assert b"\r\n" in data
    assert b"new line 1\r\nnew line 2\r\n" == data


# ── ApplyPatchTool Tests (17 - 29) ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_17_apply_patch_single_edit(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "code.py"
    target.write_text("def hello():\n    return False\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-1")
    args = ApplyPatchInput(
        path="code.py",
        edits=[PatchEdit(old_text="return False", new_text="return True")],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is True
    assert res.output["edits_applied"] == 1
    assert target.read_text(encoding="utf-8") == "def hello():\n    return True\n"


@pytest.mark.anyio
async def test_18_apply_patch_multiple_sequential_edits(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "multi.txt"
    target.write_text("apple\nbanana\ncherry\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-2")
    args = ApplyPatchInput(
        path="multi.txt",
        edits=[
            PatchEdit(old_text="apple", new_text="apricot"),
            PatchEdit(old_text="banana", new_text="blueberry"),
        ],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is True
    assert res.output["edits_applied"] == 2
    assert (
        target.read_text(encoding="utf-8") == "apricot\nblueberry\ncherry\n"
    )


@pytest.mark.anyio
async def test_19_apply_patch_fails_if_file_not_found(
    workspace: Path, patch_tool: ApplyPatchTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-3")
    args = ApplyPatchInput(
        path="ghost.txt",
        edits=[PatchEdit(old_text="a", new_text="b")],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is False
    assert "file does not exist" in res.error.lower()


@pytest.mark.anyio
async def test_20_apply_patch_fails_if_old_text_not_found(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "code.py"
    target.write_text("def foo():\n    print('hello')\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-4")
    args = ApplyPatchInput(
        path="code.py",
        edits=[PatchEdit(old_text="def bar():", new_text="def baz():")],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is False
    assert "old_text not found" in res.error
    # File unchanged
    assert target.read_text(encoding="utf-8") == "def foo():\n    print('hello')\n"


@pytest.mark.anyio
async def test_21_apply_patch_fails_if_multiple_matches_without_replace_all(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "dups.txt"
    target.write_text("var x = 1;\nvar x = 1;\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-5")
    args = ApplyPatchInput(
        path="dups.txt",
        edits=[PatchEdit(old_text="var x = 1;", new_text="var x = 2;")],
        replace_all=False,
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is False
    assert "found 2 occurrences" in res.error.lower()
    assert "replace_all is false" in res.error.lower()


@pytest.mark.anyio
async def test_22_apply_patch_succeeds_with_replace_all(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "dups.txt"
    target.write_text("var item_x = 1;\nvar item_x = 1;\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-6")
    args = ApplyPatchInput(
        path="dups.txt",
        edits=[PatchEdit(old_text="var item_x = 1;", new_text="var item_x = 2;")],
        replace_all=True,
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is True
    assert target.read_text(encoding="utf-8") == "var item_x = 2;\nvar item_x = 2;\n"


@pytest.mark.anyio
async def test_23_apply_patch_fails_if_file_exceeds_1mb(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "huge.txt"
    target.write_bytes(b"x" * (1024 * 1024 + 5))

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-7")
    args = ApplyPatchInput(
        path="huge.txt",
        edits=[PatchEdit(old_text="x", new_text="y")],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is False
    assert "exceeds 1 mb limit" in res.error.lower()


@pytest.mark.anyio
async def test_24_apply_patch_fails_if_patch_exceeds_256kb(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "small.txt"
    target.write_text("hello", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-8")
    large_patch = "x" * (256 * 1024 + 1)
    args = ApplyPatchInput(
        path="small.txt",
        edits=[PatchEdit(old_text="hello", new_text=large_patch)],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is False
    assert "exceeds 256 kb limit" in res.error.lower()


@pytest.mark.anyio
async def test_25_apply_patch_rejects_binary_file(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "binary.bin"
    target.write_bytes(b"\x00\x01\x02\x03\x04")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-9")
    args = ApplyPatchInput(
        path="binary.bin",
        edits=[PatchEdit(old_text="hello", new_text="world")],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is False
    assert "binary file" in res.error.lower()


@pytest.mark.anyio
async def test_26_apply_patch_preserves_crlf(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "crlf_file.py"
    target.write_bytes(b"line1 = 'a'\r\nline2 = 'b'\r\nline3 = 'c'\r\n")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-10")
    # Notice old_text uses LF (\n); matching should tolerate and output CRLF
    args = ApplyPatchInput(
        path="crlf_file.py",
        edits=[PatchEdit(old_text="line2 = 'b'", new_text="line2 = 'beta'")],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is True
    raw = target.read_bytes()
    assert b"\r\n" in raw
    assert b"line1 = 'a'\r\nline2 = 'beta'\r\nline3 = 'c'\r\n" == raw


@pytest.mark.anyio
async def test_27_apply_patch_preserves_lf(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "lf_file.py"
    target.write_bytes(b"line1 = 'a'\nline2 = 'b'\nline3 = 'c'\n")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-11")
    args = ApplyPatchInput(
        path="lf_file.py",
        edits=[PatchEdit(old_text="line2 = 'b'", new_text="line2 = 'beta'")],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is True
    raw = target.read_bytes()
    assert b"\r\n" not in raw
    assert b"line1 = 'a'\nline2 = 'beta'\nline3 = 'c'\n" == raw


@pytest.mark.anyio
async def test_28_apply_patch_preserves_utf8_bom(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "bom.txt"
    bom_content = b"\xef\xbb\xbfTitle\r\nHello World\r\n"
    target.write_bytes(bom_content)

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-12")
    args = ApplyPatchInput(
        path="bom.txt",
        edits=[PatchEdit(old_text="Hello World", new_text="Hello Antigravity")],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is True
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"Hello Antigravity\r\n" in raw


@pytest.mark.anyio
async def test_29_apply_patch_provides_closest_match_hint(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "source.py"
    target.write_text("function calculateTotalAmount() {\n  return 42;\n}\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-patch-13")
    # Typo: calculateTotalAmont vs calculateTotalAmount
    args = ApplyPatchInput(
        path="source.py",
        edits=[PatchEdit(old_text="function calculateTotalAmont() {", new_text="function calc() {")],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is False
    assert "closest matching line" in res.error.lower()
    assert "calculateTotalAmount" in res.error


# ── Gateway, Registry & Audit Tests (30 - 33) ────────────────────────────────


@pytest.mark.anyio
async def test_30_tool_gateway_write_file_and_apply_patch_routed_through_write_guard(
    gateway: ToolGateway, workspace: Path
):
    # Valid write_file
    res = await gateway.invoke(
        session_id="sess-gw-1",
        tool_name="write_file",
        raw_args={"path": "test.txt", "content": "gw write"},
        workspace_root=workspace,
    )
    assert res.success is True
    assert (workspace / "test.txt").read_text(encoding="utf-8") == "gw write"

    # Valid apply_patch
    patch_args = {
        "path": "test.txt",
        "edits": [{"old_text": "gw write", "new_text": "gw patched"}],
    }
    res_patch = await gateway.invoke(
        session_id="sess-gw-1",
        tool_name="apply_patch",
        raw_args=patch_args,
        workspace_root=workspace,
    )
    assert res_patch.success is True
    assert (workspace / "test.txt").read_text(encoding="utf-8") == "gw patched"

    # Blocked write_file to protected path
    res_block = await gateway.invoke(
        session_id="sess-gw-1",
        tool_name="write_file",
        raw_args={"path": ".git/hooks/post-commit", "content": "bad"},
        workspace_root=workspace,
    )
    assert res_block.success is False
    assert "access denied" in res_block.error.lower() or "protected" in res_block.error.lower()


@pytest.mark.anyio
async def test_31_tool_gateway_audit_logging_with_hashes_and_no_content(
    gateway: ToolGateway, workspace: Path
):
    secret_text = "super_secret_password_do_not_log_me"

    res = await gateway.invoke(
        session_id="sess-gw-audit",
        tool_name="write_file",
        raw_args={"path": "conf_data.txt", "content": secret_text},
        workspace_root=workspace,
    )
    assert res.success is True

    audit_file = workspace / ".agent_audit.jsonl"
    assert audit_file.exists()
    audit_lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) == 1

    audit_entry = audit_lines[0]
    # Check that file content is NOT in the audit log
    assert secret_text not in audit_entry

    # Check that SHA-256 hashes and byte counts are present
    assert "before_sha256" in audit_entry
    assert "after_sha256" in audit_entry
    assert "bytes_before" in audit_entry
    assert "bytes_after" in audit_entry


@pytest.mark.anyio
async def test_32_tool_gateway_rejects_arbitrary_mutating_tool(
    workspace: Path,
):
    class MaliciousTool(WriteFileTool):
        name = "shell_exec"
        read_only = False

    registry = ToolRegistry()
    registry.register(MaliciousTool())
    guard = PathGuard(workspace)
    gw = ToolGateway(
        registry=registry,
        audit_logger=AuditLogger(workspace / ".audit.jsonl"),
        workspace_root=workspace,
        path_guard=guard,
    )
    res = await gw.invoke(
        session_id="sess-gw-deny",
        tool_name="shell_exec",
        raw_args={"path": "test.txt", "content": "bad"},
        workspace_root=workspace,
    )
    assert res.success is False
    assert "not authorized" in res.error.lower() or "only read_only" in res.error.lower()


def test_33_tool_registry_with_write_tools():
    reg = create_default_registry(include_write_tools=True)
    tool_names = [t.name for t in reg.list_tools()]
    assert set(tool_names) == {"list_directory", "read_file", "write_file", "apply_patch"}
    defs = reg.get_openai_tool_definitions()
    assert len(defs) == 4
    def_names = [d["function"]["name"] for d in defs]
    assert "write_file" in def_names
    assert "apply_patch" in def_names


# ── Phase 7 Addendum H Specific Tests ─────────────────────────────────────────


@pytest.mark.anyio
async def test_34_addendum_git_hooks_and_vscode_denied(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="sess-addendum-h")
    # Test .git/hooks/x
    res1 = await write_tool.execute(
        WriteFileInput(path=".git/hooks/x", content="malicious hook"), ctx
    )
    assert res1.success is False
    assert "protected" in res1.error.lower()

    # Test .vscode/tasks.json
    res2 = await write_tool.execute(
        WriteFileInput(path=".vscode/tasks.json", content="{}"), ctx
    )
    assert res2.success is False
    assert "protected" in res2.error.lower()


@pytest.mark.anyio
async def test_35_addendum_mixed_case_env_denied(
    workspace: Path, write_tool: WriteFileTool
):
    ctx = ToolContext(workspace_root=workspace, session_id="sess-addendum-case")
    for name in [".ENV", ".Env", ".eNv", ".GIT/config", ".VSCode/settings.json"]:
        res = await write_tool.execute(
            WriteFileInput(path=name, content="val"), ctx
        )
        assert res.success is False
        assert "protected or denied" in res.error.lower() or "protected" in res.error.lower()


@pytest.mark.anyio
async def test_36_addendum_restore_snapshot_exact_bytes(
    workspace: Path,
    write_tool: WriteFileTool,
    patch_tool: ApplyPatchTool,
    snapshot_manager: SnapshotManager,
):
    target = workspace / "important.dat"
    original_bytes = b"\xef\xbb\xbforiginal header\r\nline2\r\n"
    target.write_bytes(original_bytes)

    ctx = ToolContext(workspace_root=workspace, session_id="sess-restore")

    # Modify via apply_patch
    res = await patch_tool.execute(
        ApplyPatchInput(
            path="important.dat",
            edits=[PatchEdit(old_text="line2", new_text="corrupted")],
        ),
        ctx,
    )
    assert res.success is True
    assert target.read_bytes() != original_bytes

    # Now restore snapshot seq 1
    restore_res = restore_snapshot(
        seq=1,
        session_id="sess-restore",
        workspace_root=workspace,
        store_dir=snapshot_manager.store_dir,
    )
    assert restore_res.success is True
    assert target.read_bytes() == original_bytes


@pytest.mark.anyio
async def test_37_addendum_restore_snapshot_undo_new_file(
    workspace: Path,
    write_tool: WriteFileTool,
    snapshot_manager: SnapshotManager,
):
    ctx = ToolContext(workspace_root=workspace, session_id="sess-undo-new")
    res = await write_tool.execute(
        WriteFileInput(path="temporary.txt", content="temp content"), ctx
    )
    assert res.success is True
    assert (workspace / "temporary.txt").exists()

    # Restore snapshot seq 1 (undo creation => file deleted)
    restore_res = restore_snapshot(
        seq=1,
        session_id="sess-undo-new",
        workspace_root=workspace,
        store_dir=snapshot_manager.store_dir,
    )
    assert restore_res.success is True
    assert not (workspace / "temporary.txt").exists()


@pytest.mark.anyio
async def test_38_addendum_snapshot_unreachable_via_read_file(
    workspace: Path,
    agent_store: Path,
    write_tool: WriteFileTool,
    read_tool: ReadFileTool,
    snapshot_manager: SnapshotManager,
):
    target = workspace / "doc.txt"
    target.write_text("doc v1", encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-unreach")
    await write_tool.execute(
        WriteFileInput(path="doc.txt", content="doc v2", overwrite=True), ctx
    )

    snapshots = snapshot_manager.list_snapshots("sess-unreach")
    assert len(snapshots) == 1
    snap_name = snapshots[0]["snapshot_file"]
    assert snap_name is not None

    full_snap_path = snapshot_manager.snapshots_dir / "sess-unreach" / snap_name
    assert full_snap_path.exists()

    # Attempt to read snapshot file using read_file tool with absolute path outside workspace
    res_read = await read_tool.execute(ReadFileInput(path=str(full_snap_path)), ctx)
    assert res_read.success is False
    assert "outside" in res_read.error.lower() or "escapes" in res_read.error.lower()


@pytest.mark.anyio
async def test_39_addendum_multi_edit_atomicity(
    workspace: Path, patch_tool: ApplyPatchTool
):
    target = workspace / "atomic.txt"
    original = "first\nsecond\nthird\n"
    target.write_text(original, encoding="utf-8")

    ctx = ToolContext(workspace_root=workspace, session_id="sess-atomic")
    # First edit matches, second edit fails
    args = ApplyPatchInput(
        path="atomic.txt",
        edits=[
            PatchEdit(old_text="first", new_text="FIRST"),
            PatchEdit(old_text="NONEXISTENT", new_text="SOMETHING"),
        ],
    )
    res = await patch_tool.execute(args, ctx)

    assert res.success is False
    assert "old_text not found" in res.error
    # Must remain completely unchanged
    assert target.read_text(encoding="utf-8") == original


def test_40_addendum_snapshot_storage_capping(tmp_path: Path):
    store = tmp_path / "capping_store"
    ws = tmp_path / "capping_ws"
    ws.mkdir()
    store.mkdir()

    # Small limits: max 3 snapshots, max 500 bytes
    manager = SnapshotManager(
        store_dir=store, max_snapshots_per_session=3, max_bytes_per_session=500
    )

    for i in range(1, 6):
        file_path = ws / f"f{i}.txt"
        file_path.write_text(f"content of file {i} " * 5, encoding="utf-8")
        manager.create_pre_write_snapshot(
            session_id="cap-sess",
            target_file=file_path,
            relative_path=f"f{i}.txt",
            tool="write_file",
        )

    snapshots = manager.list_snapshots("cap-sess")
    # Should be capped at 3
    assert len(snapshots) <= 3


@pytest.mark.anyio
async def test_41_verify_audit_hash_chain(
    gateway: ToolGateway, workspace: Path
):
    await gateway.invoke(
        session_id="sess-verify-chain",
        tool_name="write_file",
        raw_args={"path": "a.txt", "content": "111"},
        workspace_root=workspace,
    )
    await gateway.invoke(
        session_id="sess-verify-chain",
        tool_name="write_file",
        raw_args={"path": "a.txt", "content": "222", "overwrite": True},
        workspace_root=workspace,
    )
    await gateway.invoke(
        session_id="sess-verify-chain",
        tool_name="apply_patch",
        raw_args={"path": "a.txt", "edits": [{"old_text": "222", "new_text": "333"}]},
        workspace_root=workspace,
    )

    audit_file = workspace / ".agent_audit.jsonl"
    is_valid, errors = verify_audit_log(audit_file)

    assert is_valid is True
    assert len(errors) == 0
