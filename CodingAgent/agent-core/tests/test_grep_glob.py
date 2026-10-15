import json
import os
import pytest
from pathlib import Path

from agent_core.tools.base import ToolContext
from agent_core.tools.glob import GlobTool, GlobInput
from agent_core.tools.grep import GrepTool, GrepInput
from agent_core.tools.registry import create_default_registry
from agent_core.tools.gateway import ToolGateway
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


@pytest.fixture
def tool_context(temp_workspace: Path, temp_audit_log: AuditLogger):
    return ToolContext(
        session_id="sess-grep-glob-1",
        workspace_root=temp_workspace,
        audit_logger=temp_audit_log,
    )


# ==============================================================================
# GLOB UNIT TESTS
# ==============================================================================


@pytest.mark.anyio
async def test_glob_standard_and_recursive(temp_workspace: Path, tool_context: ToolContext):
    """Test standard and recursive glob patterns return sorted relative paths."""
    (temp_workspace / "src" / "components").mkdir(parents=True, exist_ok=True)
    (temp_workspace / "src" / "utils").mkdir(parents=True, exist_ok=True)
    (temp_workspace / "docs").mkdir(parents=True, exist_ok=True)

    (temp_workspace / "src" / "components" / "App.tsx").write_text("code", encoding="utf-8")
    (temp_workspace / "src" / "components" / "Button.tsx").write_text("code", encoding="utf-8")
    (temp_workspace / "src" / "utils" / "helper.ts").write_text("code", encoding="utf-8")
    (temp_workspace / "docs" / "readme.md").write_text("docs", encoding="utf-8")
    (temp_workspace / "root.ts").write_text("code", encoding="utf-8")

    tool = GlobTool()

    # Recursive tsx match
    res1 = await tool.execute(GlobInput(pattern="**/*.tsx"), tool_context)
    assert res1.success is True
    assert res1.output["matches"] == ["src/components/App.tsx", "src/components/Button.tsx"]
    assert res1.output["truncated"] is False

    # Scoped path search
    res2 = await tool.execute(GlobInput(pattern="*.tsx", path="src/components"), tool_context)
    assert res2.success is True
    assert res2.output["matches"] == ["src/components/App.tsx", "src/components/Button.tsx"]

    # All ts/tsx match
    res3 = await tool.execute(GlobInput(pattern="**/*.ts"), tool_context)
    assert res3.success is True
    assert "root.ts" in res3.output["matches"]
    assert "src/utils/helper.ts" in res3.output["matches"]


@pytest.mark.anyio
async def test_glob_skips_default_dirs(temp_workspace: Path, tool_context: ToolContext):
    """Test that default ignored directories (.git, node_modules, etc.) are skipped."""
    (temp_workspace / ".git").mkdir()
    (temp_workspace / ".git" / "config.txt").write_text("git config", encoding="utf-8")

    (temp_workspace / "node_modules" / "pkg").mkdir(parents=True)
    (temp_workspace / "node_modules" / "pkg" / "index.js").write_text("pkg", encoding="utf-8")

    (temp_workspace / "target" / "debug").mkdir(parents=True)
    (temp_workspace / "target" / "debug" / "binary.bin").write_text("bin", encoding="utf-8")

    (temp_workspace / "__pycache__").mkdir()
    (temp_workspace / "__pycache__" / "cached.pyc").write_text("pyc", encoding="utf-8")

    (temp_workspace / ".venv" / "lib").mkdir(parents=True)
    (temp_workspace / ".venv" / "lib" / "site.py").write_text("site", encoding="utf-8")

    (temp_workspace / "agent-store").mkdir()
    (temp_workspace / "agent-store" / "data.json").write_text("{}", encoding="utf-8")

    (temp_workspace / "valid.py").write_text("print('hello')", encoding="utf-8")

    tool = GlobTool()
    res = await tool.execute(GlobInput(pattern="**/*"), tool_context)

    assert res.success is True
    matches = res.output["matches"]
    assert "valid.py" in matches
    assert not any(".git" in m for m in matches)
    assert not any("node_modules" in m for m in matches)
    assert not any("target" in m for m in matches)
    assert not any("__pycache__" in m for m in matches)
    assert not any(".venv" in m for m in matches)
    assert not any("agent-store" in m for m in matches)


@pytest.mark.anyio
async def test_glob_cap_and_truncation(temp_workspace: Path, tool_context: ToolContext):
    """Test result capping at 200 items with truncated=True."""
    files_dir = temp_workspace / "lots_of_files"
    files_dir.mkdir()
    for i in range(250):
        (files_dir / f"item_{i:03d}.txt").write_text("content", encoding="utf-8")

    tool = GlobTool()
    res = await tool.execute(GlobInput(pattern="**/*.txt"), tool_context)

    assert res.success is True
    assert res.output["count"] == 200
    assert res.output["total_matches"] == 250
    assert res.output["truncated"] is True
    assert len(res.output["matches"]) == 200


@pytest.mark.anyio
async def test_glob_pathguard_containment_and_traversal(temp_workspace: Path, tool_context: ToolContext):
    """Test PathGuard rejects traversal paths outside workspace."""
    tool = GlobTool()
    res = await tool.execute(GlobInput(pattern="*.py", path="../outside"), tool_context)

    assert res.success is False
    assert "path validation failed" in res.error.lower() or "outside" in res.error.lower()


@pytest.mark.anyio
async def test_glob_symlink_escape(temp_workspace: Path, tmp_path: Path, tool_context: ToolContext):
    """Test that symlinks resolving outside the workspace are not followed or included."""
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("secret", encoding="utf-8")

    link_path = temp_workspace / "link_to_outside"
    try:
        os.symlink(outside_dir, link_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported in this test environment without elevation")

    tool = GlobTool()
    res = await tool.execute(GlobInput(pattern="**/*.txt"), tool_context)

    assert res.success is True
    assert not any("secret.txt" in m for m in res.output["matches"])


# ==============================================================================
# GREP UNIT TESTS
# ==============================================================================


@pytest.mark.anyio
async def test_grep_literal_match(temp_workspace: Path, tool_context: ToolContext):
    """Test case-sensitive and case-insensitive literal string search."""
    file1 = temp_workspace / "example.py"
    file1.write_text("def find_my_target():\n    return 'TARGET_VAL'\n", encoding="utf-8")

    file2 = temp_workspace / "example2.py"
    file2.write_text("target_val = 123\n", encoding="utf-8")

    tool = GrepTool()

    # Case-insensitive (default)
    res_ci = await tool.execute(GrepInput(pattern="target_val"), tool_context)
    assert res_ci.success is True
    assert res_ci.output["count"] == 2
    assert res_ci.output["matches"][0]["file"] == "example.py"
    assert res_ci.output["matches"][0]["line_number"] == 2
    assert "TARGET_VAL" in res_ci.output["matches"][0]["line"]
    assert res_ci.output["matches"][1]["file"] == "example2.py"

    # Case-sensitive
    res_cs = await tool.execute(GrepInput(pattern="TARGET_VAL", case_sensitive=True), tool_context)
    assert res_cs.success is True
    assert res_cs.output["count"] == 1
    assert res_cs.output["matches"][0]["file"] == "example.py"


@pytest.mark.anyio
async def test_grep_regex_match(temp_workspace: Path, tool_context: ToolContext):
    """Test regular expression search with capture/classes."""
    file1 = temp_workspace / "service.py"
    file1.write_text("port = 8080\nhost = '127.0.0.1'\nport = 9000\n", encoding="utf-8")

    tool = GrepTool()
    res = await tool.execute(GrepInput(pattern=r"port\s*=\s*\d+", regex=True), tool_context)

    assert res.success is True
    assert res.output["count"] == 2
    assert res.output["matches"][0]["line_number"] == 1
    assert res.output["matches"][1]["line_number"] == 3


@pytest.mark.anyio
async def test_grep_invalid_regex_actionable_error(temp_workspace: Path, tool_context: ToolContext):
    """Test actionable error message when user supplies an invalid regular expression."""
    tool = GrepTool()
    res = await tool.execute(GrepInput(pattern="[unclosed_bracket", regex=True), tool_context)

    assert res.success is False
    assert "invalid regex" in res.error.lower()
    assert "try regex=false for a literal search" in res.error.lower()


@pytest.mark.anyio
async def test_grep_pattern_length_cap(temp_workspace: Path, tool_context: ToolContext):
    """Test that pattern exceeding 500 characters is rejected."""
    long_pattern = "a" * 501
    tool = GrepTool()
    res = await tool.execute(GrepInput(pattern=long_pattern), tool_context)

    assert res.success is False
    assert "maximum limit of 500" in res.error.lower() or "exceeds" in res.error.lower()


@pytest.mark.anyio
async def test_grep_binary_skip(temp_workspace: Path, tool_context: ToolContext):
    """Test binary files with null bytes are skipped and reported."""
    bin_file = temp_workspace / "data.bin"
    bin_file.write_bytes(b"hello world\x00some binary data")

    text_file = temp_workspace / "text.txt"
    text_file.write_text("hello world from text\n", encoding="utf-8")

    tool = GrepTool()
    res = await tool.execute(GrepInput(pattern="hello world"), tool_context)

    assert res.success is True
    assert res.output["count"] == 1
    assert res.output["matches"][0]["file"] == "text.txt"
    assert res.output["files_skipped_binary"] == 1


@pytest.mark.anyio
async def test_grep_file_size_cap(temp_workspace: Path, tool_context: ToolContext):
    """Test files > 1 MB are skipped and counted in files_skipped_large."""
    large_file = temp_workspace / "large.txt"
    # Write 1.1 MB of text
    large_file.write_text("target_text\n" * 90000, encoding="utf-8")

    normal_file = temp_workspace / "normal.txt"
    normal_file.write_text("target_text\n", encoding="utf-8")

    tool = GrepTool()
    res = await tool.execute(GrepInput(pattern="target_text"), tool_context)

    assert res.success is True
    assert res.output["count"] == 1
    assert res.output["matches"][0]["file"] == "normal.txt"
    assert res.output["files_skipped_large"] == 1


@pytest.mark.anyio
async def test_grep_line_length_cap_and_snippet_truncation(temp_workspace: Path, tool_context: ToolContext):
    """Test that line length is capped and snippets are truncated with ellipses."""
    very_long_line = "prefix " + ("x" * 300) + " target_word " + ("y" * 5000) + "\n"
    target_file = temp_workspace / "minified.js"
    target_file.write_text(very_long_line, encoding="utf-8")

    tool = GrepTool()
    res = await tool.execute(GrepInput(pattern="target_word"), tool_context)

    assert res.success is True
    assert res.output["count"] == 1
    snippet = res.output["matches"][0]["line"]
    assert len(snippet) <= 203  # 200 + '...'
    assert snippet.endswith("...")


@pytest.mark.anyio
async def test_grep_result_cap_and_truncation(temp_workspace: Path, tool_context: ToolContext):
    """Test that matching lines exceeding max_results are truncated."""
    target_file = temp_workspace / "multi.txt"
    target_file.write_text("match_me\n" * 50, encoding="utf-8")

    tool = GrepTool()
    res = await tool.execute(GrepInput(pattern="match_me", max_results=10), tool_context)

    assert res.success is True
    assert res.output["count"] == 10
    assert res.output["total_matches"] == 50
    assert res.output["truncated"] is True


@pytest.mark.anyio
async def test_grep_pathguard_containment(temp_workspace: Path, tool_context: ToolContext):
    """Test PathGuard prevents grep traversal attempts outside workspace."""
    tool = GrepTool()
    res = await tool.execute(GrepInput(pattern="foo", path="../../"), tool_context)

    assert res.success is False
    assert "path validation failed" in res.error.lower() or "outside" in res.error.lower()


@pytest.mark.anyio
async def test_grep_and_glob_gateway_audit_logging(temp_workspace: Path, temp_audit_log: AuditLogger):
    """Test that grep and glob invocations through ToolGateway are audit logged and chained."""
    registry = create_default_registry(audit_logger=temp_audit_log)
    gateway = ToolGateway(registry=registry, audit_logger=temp_audit_log)

    (temp_workspace / "test.py").write_text("def find_me(): pass\n", encoding="utf-8")

    # 1. Execute glob via gateway
    res_glob = await gateway.invoke(
        session_id="sess-audit-gg",
        tool_name="glob",
        raw_args={"pattern": "*.py"},
        workspace_root=str(temp_workspace),
    )
    assert res_glob.success is True

    # 2. Execute grep via gateway
    res_grep = await gateway.invoke(
        session_id="sess-audit-gg",
        tool_name="grep",
        raw_args={"pattern": "find_me"},
        workspace_root=str(temp_workspace),
    )
    assert res_grep.success is True

    # Verify audit entries
    assert temp_audit_log.log_path.exists()
    lines = [json.loads(l) for l in temp_audit_log.log_path.read_text(encoding="utf-8").strip().splitlines()]
    tools_logged = [l["tool"] for l in lines]

    assert "glob" in tools_logged
    assert "grep" in tools_logged

    # Verify tamper-evident hash chain integrity
    for i in range(1, len(lines)):
        assert lines[i]["previous_hash"] == lines[i - 1]["hash"]
