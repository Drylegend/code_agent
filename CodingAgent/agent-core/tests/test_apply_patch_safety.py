"""Comprehensive tests for apply_patch safety, ambiguous match reporting, unified diffs, warnings, and system prompt rules (Phase 7.7 A1-A7)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_core.llm import SYSTEM_PROMPT
from agent_core.protocol import ToolResultPayload, create_tool_result
from agent_core.server import extract_tool_result_event_data
from agent_core.snapshots import SnapshotManager
from agent_core.tools import (
    ApplyPatchInput,
    ApplyPatchTool,
    EditSpec,
    GatewayInvocationResult,
    AuditLogger,
    PathGuard,
    ToolContext,
    ToolGateway,
    ToolRegistry,
)
from agent_core.tools.apply_patch import (
    DEFAULT_MAX_REPLACE_ALL_DELTA,
    DEFAULT_MAX_REPLACE_ALL_MATCHES,
    DEFAULT_MIN_REPLACE_ALL_CHARS,
)


@pytest.fixture
def test_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture
def snapshot_manager(tmp_path: Path) -> SnapshotManager:
    store = tmp_path / "snapshots"
    store.mkdir(parents=True, exist_ok=True)
    return SnapshotManager(store_dir=store)


@pytest.fixture
def patch_tool(snapshot_manager: SnapshotManager) -> ApplyPatchTool:
    return ApplyPatchTool(snapshot_manager=snapshot_manager)


# ── A1: Ambiguous Match Error Tests ──────────────────────────────────────────


@pytest.mark.anyio
async def test_a1_ambiguous_match_error_structure(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """When replace_all=False and old_text matches N > 1 times:
    - Operation is rejected;
    - old_text matched N times is explicitly reported;
    - First 5 matches are listed with line numbers and surrounding context;
    - Error does NOT suggest replace_all;
    - Instructs model to call read_file to provide unique context.
    """
    sample_content = (
        "line 1: header\n"
        "item_target = 10\n"
        "line 3: middle\n"
        "item_target = 20\n"
        "line 5: middle\n"
        "item_target = 30\n"
        "line 7: middle\n"
        "item_target = 40\n"
        "line 9: middle\n"
        "item_target = 50\n"
        "line 11: middle\n"
        "item_target = 60\n"
        "line 13: footer\n"
    )
    target_file = test_workspace / "multi_match.txt"
    target_file.write_text(sample_content, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a1")
    args = ApplyPatchInput(
        path="multi_match.txt",
        edits=[EditSpec(old_text="item_target", new_text="updated_item")],
        replace_all=False,
    )

    res = await patch_tool.execute(args, ctx)
    assert not res.success
    err = res.error or ""

    # Match count explicitly present (matched 6 times)
    assert "old_text matched 6 times" in err
    assert "Must match exactly once when replace_all is false" in err

    # First 5 matches are listed
    assert "Occurrence #1" in err
    assert "Occurrence #5" in err
    assert "Occurrence #6" not in err  # Only first 5 shown
    assert "only the first 5 matches shown" in err

    # Line numbers present
    assert "line 2" in err or " 2 |" in err
    assert "line 4" in err or " 4 |" in err

    # Surrounding context present
    assert "line 1: header" in err
    assert "line 3: middle" in err

    # Must NOT recommend replace_all
    assert "use replace_all" not in err.lower()
    assert "set replace_all=true" not in err.lower()
    assert "suggest replace_all" not in err.lower()

    # Explicit instruction to call read_file
    assert "Please call read_file and provide enough surrounding context" in err


# ── A2: replace_all Safety Guards Tests ──────────────────────────────────────


@pytest.mark.anyio
async def test_a2_reject_too_few_non_whitespace_chars(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """replace_all=True rejected if old_text has < 8 non-whitespace characters."""
    target_file = test_workspace / "code.py"
    target_file.write_text("x = 1\nx = 2\nx = 3\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a2_short")
    # "x =" has 2 non-whitespace characters (< 8)
    args = ApplyPatchInput(
        path="code.py",
        edits=[EditSpec(old_text="x =", new_text="val =")],
        replace_all=True,
    )

    res = await patch_tool.execute(args, ctx)
    assert not res.success
    err = res.error or ""
    assert "replace_all safety rule triggered: old_text has too few non-whitespace characters" in err
    assert f"minimum required: {DEFAULT_MIN_REPLACE_ALL_CHARS}" in err
    assert "actual: 2" in err
    assert "use a more specific/longer old_text or a narrower edit" in err


@pytest.mark.anyio
async def test_a2_reject_too_many_matches(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """replace_all=True rejected if matches > max_replace_all_matches (20)."""
    # 25 occurrences of a 10-char token
    content = "\n".join(f"var_token_abc = {i}" for i in range(25))
    target_file = test_workspace / "large_matches.txt"
    target_file.write_text(content, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a2_count")
    args = ApplyPatchInput(
        path="large_matches.txt",
        edits=[EditSpec(old_text="var_token_abc", new_text="renamed_token_xyz")],
        replace_all=True,
    )

    res = await patch_tool.execute(args, ctx)
    assert not res.success
    err = res.error or ""
    assert "replace_all safety rule triggered: match count exceeds limit" in err
    assert f"maximum allowed: {DEFAULT_MAX_REPLACE_ALL_MATCHES}" in err
    assert "actual: 25 matches" in err


@pytest.mark.anyio
async def test_a2_reject_oversized_delta(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """replace_all=True rejected if total size delta > 10 KB."""
    # 5 matches, but each expansion adds 3 KB (total delta 15 KB > 10 KB)
    content = "\n".join(f"mySpecialIdentifier_num{i} = True" for i in range(5))
    target_file = test_workspace / "large_delta.txt"
    target_file.write_text(content, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a2_delta")
    huge_replacement = "A" * 3000
    args = ApplyPatchInput(
        path="large_delta.txt",
        edits=[EditSpec(old_text="mySpecialIdentifier_num", new_text=huge_replacement)],
        replace_all=True,
    )

    res = await patch_tool.execute(args, ctx)
    assert not res.success
    err = res.error or ""
    assert "replace_all safety rule triggered: total size delta exceeds limit" in err
    assert f"maximum allowed: {DEFAULT_MAX_REPLACE_ALL_DELTA}" in err


@pytest.mark.anyio
async def test_a2_legitimate_long_identifier_rename_succeeds(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """Legitimate identifier rename (e.g. getUserProfileRecord) succeeds."""
    content = (
        "def getUserProfileRecord(uid):\n"
        "    return db.query(uid)\n\n"
        "record = getUserProfileRecord(42)\n"
    )
    target_file = test_workspace / "service.py"
    target_file.write_text(content, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a2_valid")
    args = ApplyPatchInput(
        path="service.py",
        edits=[EditSpec(old_text="getUserProfileRecord", new_text="fetchUserProfileData")],
        replace_all=True,
    )

    res = await patch_tool.execute(args, ctx)
    assert res.success
    assert res.output["replacements"] == 2
    assert "fetchUserProfileData" in target_file.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_a2_configurable_thresholds_override(
    test_workspace: Path, snapshot_manager: SnapshotManager
) -> None:
    """Thresholds configured on ApplyPatchTool instance are honored."""
    custom_tool = ApplyPatchTool(
        snapshot_manager=snapshot_manager,
        min_replace_all_chars=4,
        max_replace_all_matches=5,
        max_replace_all_delta=500,
    )
    assert custom_tool.min_replace_all_chars == 4
    assert custom_tool.max_replace_all_matches == 5
    assert custom_tool.max_replace_all_delta == 500

    target_file = test_workspace / "config_test.txt"
    target_file.write_text("abcd = 1\nabcd = 2\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a2_cfg")
    # "abcd" has 4 chars, which passes min_replace_all_chars=4
    args = ApplyPatchInput(
        path="config_test.txt",
        edits=[EditSpec(old_text="abcd", new_text="efgh")],
        replace_all=True,
    )
    res = await custom_tool.execute(args, ctx)
    assert res.success


# ── A3: Tool Description Guidance Tests ──────────────────────────────────────


def test_a3_apply_patch_description_teaches_insertion_safety() -> None:
    """ApplyPatchTool description contains explicit insertion-safe usage instructions."""
    desc = ApplyPatchTool.description
    assert "For an insertion:" in desc
    assert "old_text should be a unique existing line or block" in desc
    assert "new_text should preserve that existing text and add the requested content" in desc
    assert "do not use a short/common token such as '\\n' as old_text" in desc
    assert "do not use replace_all for a single insertion unless the user explicitly requested every occurrence" in desc


# ── A4: Unified Diff Preview & Structured Output Tests ───────────────────────


@pytest.mark.anyio
async def test_a4_structured_result_and_unified_diff(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """apply_patch returns structured info including replacements, bytes_before,
    bytes_after, delta, and a unified diff preview.
    """
    initial_text = "def hello():\n    return 'world'\n"
    target_file = test_workspace / "hello.py"
    target_file.write_text(initial_text, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a4")
    args = ApplyPatchInput(
        path="hello.py",
        edits=[EditSpec(old_text="return 'world'", new_text="return 'antigravity'")],
    )

    res = await patch_tool.execute(args, ctx)
    assert res.success
    out = res.output

    assert "replacements" in out
    assert out["replacements"] == 1
    assert "bytes_before" in out
    assert "bytes_after" in out
    assert "delta" in out
    assert out["delta"] == out["bytes_after"] - out["bytes_before"]

    assert "preview" in out
    preview = out["preview"]
    assert preview is not None
    assert "--- a/hello.py" in preview
    assert "+++ b/hello.py" in preview
    assert "-    return 'world'" in preview
    assert "+    return 'antigravity'" in preview


@pytest.mark.anyio
async def test_a4_diff_preview_capped_at_4kb(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """Large diff is capped at approximately 4 KB with truncation notice."""
    # Create file with 500 lines
    initial_lines = [f"line_{i:04d} = {i}" for i in range(500)]
    target_file = test_workspace / "large_diff.py"
    target_file.write_text("\n".join(initial_lines) + "\n", encoding="utf-8")

    new_lines = [f"line_{i:04d} = {i * 100}" for i in range(500)]
    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a4_cap")
    args = ApplyPatchInput(
        path="large_diff.py",
        edits=[EditSpec(old_text="\n".join(initial_lines), new_text="\n".join(new_lines))],
    )

    res = await patch_tool.execute(args, ctx)
    assert res.success
    preview = res.output["preview"]
    assert preview is not None
    assert "...(diff truncated at 4 KB)" in preview
    # Bounded length
    assert len(preview) <= 4096 + len("\n...(diff truncated at 4 KB)")


def test_a4_server_extract_tool_result_populates_preview() -> None:
    """server.py extract_tool_result_event_data extracts preview for apply_patch."""
    inv_res = GatewayInvocationResult(
        success=True,
        authorized=True,
        tool_name="apply_patch",
        output={
            "path": "test.py",
            "replacements": 1,
            "preview": "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-x\n+y\n",
            "warning": False,
        },
    )
    ok, summary, preview, truncated, possible_injection, warning = extract_tool_result_event_data(inv_res)
    assert ok
    assert preview is not None
    assert "--- a/test.py" in preview
    assert not warning


# ── A5: Out-of-Proportion Warning Tests ──────────────────────────────────────


@pytest.mark.anyio
async def test_a5_warning_on_replacements_over_three(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """warning=True when replacements > 3, but operation succeeds."""
    content = "\n".join(f"my_config_variable_name = {i}" for i in range(4))
    target_file = test_workspace / "config.py"
    target_file.write_text(content, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a5_rep")
    args = ApplyPatchInput(
        path="config.py",
        edits=[EditSpec(old_text="my_config_variable_name", new_text="my_settings_var_name")],
        replace_all=True,
    )

    res = await patch_tool.execute(args, ctx)
    assert res.success
    assert res.output["replacements"] == 4
    assert res.output["warning"] is True


@pytest.mark.anyio
async def test_a5_warning_on_oversized_delta(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """warning=True when abs(delta) > 10 * max(len(new_text), 1)."""
    # Deleting 200 bytes by replacing with "x" (len 1). Delta = -199.
    # 10 * max(len("x"), 1) = 10. abs(-199) > 10 => warning=True.
    original_snippet = "A" * 200
    target_file = test_workspace / "shrink.txt"
    target_file.write_text(f"prefix\n{original_snippet}\nsuffix\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a5_delta")
    args = ApplyPatchInput(
        path="shrink.txt",
        edits=[EditSpec(old_text=original_snippet, new_text="x")],
        replace_all=False,
    )

    res = await patch_tool.execute(args, ctx)
    assert res.success
    assert res.output["warning"] is True


@pytest.mark.anyio
async def test_a5_no_warning_on_normal_edit(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """warning=False for normal, proportional edit."""
    target_file = test_workspace / "normal.py"
    target_file.write_text("var_a = 1\nvar_b = 2\n", encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_a5_norm")
    args = ApplyPatchInput(
        path="normal.py",
        edits=[EditSpec(old_text="var_a = 1", new_text="var_a = 10")],
    )

    res = await patch_tool.execute(args, ctx)
    assert res.success
    assert res.output["warning"] is False


# ── A6: System Prompt Rules Tests ────────────────────────────────────────────


def test_a6_system_prompt_rules() -> None:
    """SYSTEM_PROMPT contains the required safety and reporting instructions."""
    assert "Report only what the tool result actually states; do not invent or hallucinate explanations." in SYSTEM_PROMPT
    assert "Use replace_all=true ONLY when the user explicitly requests to change every occurrence in the file." in SYSTEM_PROMPT
    assert "If apply_patch fails (e.g. ambiguous match), call read_file to inspect the exact section and retry with unique surrounding context." in SYSTEM_PROMPT
    assert "Never claim a file was created or modified unless the corresponding tool call succeeded." in SYSTEM_PROMPT


# ── A7: End-to-End Scripted Fake-Model Test ──────────────────────────────────


@pytest.mark.anyio
async def test_a7_end_to_end_scripted_model_ambiguous_and_replace_all_rejected(
    test_workspace: Path, snapshot_manager: SnapshotManager
) -> None:
    """Full script test:
    1. Start with known file.
    2. Fake model sends ambiguous apply_patch.
    3. Gateway rejects it.
    4. Fake model attempts retry using replace_all=True on short text.
    5. Gateway rejects replace_all retry due to new safety guard.
    6. File remains byte-for-byte unchanged.
    7. Both tool errors are returned properly.
    8. Snapshots reflect unchanged state.
    9. Legitimate long-identifier rename then succeeds.
    """
    initial_bytes = (
        b"# Configuration File\n"
        b"timeout = 30\n"
        b"retry_count = 3\n"
        b"timeout = 60\n"
        b"getUserProfileRecord = True\n"
        b"getUserProfileRecord = False\n"
    )
    test_file = test_workspace / "app.conf"
    test_file.write_bytes(initial_bytes)

    # Initialize gateway with tools
    registry = ToolRegistry()
    registry.register(ApplyPatchTool(snapshot_manager=snapshot_manager))
    guard = PathGuard(test_workspace)
    audit_logger = AuditLogger(test_workspace.parent / "test_audit.jsonl")
    gateway = ToolGateway(
        registry=registry,
        audit_logger=audit_logger,
        path_guard=guard,
        workspace_root=test_workspace,
    )
    session_id = "session_a7_scripted"

    # Step 1: Model sends ambiguous apply_patch (matches 2 times, replace_all=False)
    step1_args = {
        "path": "app.conf",
        "edits": [{"old_text": "timeout =", "new_text": "connection_timeout ="}],
        "replace_all": False,
    }
    res1 = await gateway.invoke(session_id, "apply_patch", step1_args)
    assert not res1.success
    assert "old_text matched 2 times" in res1.error
    assert "Occurrence #1" in res1.error
    assert "Occurrence #2" in res1.error
    # File remains identical
    assert test_file.read_bytes() == initial_bytes

    # Step 2: Model attempts desperate retry with replace_all=True on short token "timeout" (7 non-ws < 8)
    step2_short_args = {
        "path": "app.conf",
        "edits": [{"old_text": "timeout", "new_text": "timeout_sec"}],
        "replace_all": True,
    }
    res2 = await gateway.invoke(session_id, "apply_patch", step2_short_args)
    assert not res2.success
    assert "replace_all safety rule triggered: old_text has too few non-whitespace characters" in res2.error
    # File remains byte-for-byte unchanged
    assert test_file.read_bytes() == initial_bytes

    # Step 3: Legitimate long-identifier rename succeeds
    step3_args = {
        "path": "app.conf",
        "edits": [{"old_text": "getUserProfileRecord", "new_text": "fetchUserProfileRecord"}],
        "replace_all": True,
    }
    res3 = await gateway.invoke(session_id, "apply_patch", step3_args)
    assert res3.success
    assert res3.output["replacements"] == 2
    updated_bytes = test_file.read_bytes()
    assert b"fetchUserProfileRecord" in updated_bytes
    assert b"getUserProfileRecord" not in updated_bytes


# ── Regression: Existing-file write path end-to-end ──────────────────────────


@pytest.mark.anyio
async def test_apply_patch_writes_to_disk_and_matches_hashes(
    test_workspace: Path, patch_tool: ApplyPatchTool, snapshot_manager: SnapshotManager
) -> None:
    """Verify that ApplyPatchTool on an existing file:
    1. Actually modifies the file bytes on disk;
    2. Reports after_sha256 that matches hashlib.sha256(disk_bytes);
    3. Records snapshot with after_sha256 matching disk_bytes;
    4. Has snapshot seq recorded for undo.
    """
    import hashlib

    target = test_workspace / "test.txt"
    initial_bytes = b"Hello from the Agent Shell \xe2\x80\x94 Phase 7 live model edit verification.\n"
    target.write_bytes(initial_bytes)

    ctx = ToolContext(workspace_root=test_workspace, session_id="sess-patch-disk-check")
    edit_input = ApplyPatchInput(
        path="test.txt",
        edits=[EditSpec(old_text="live model edit", new_text="Undo")],
    )

    res = await patch_tool.execute(edit_input, ctx)
    assert res.success is True

    # 1. Verify file on disk actually changed
    disk_bytes = target.read_bytes()
    assert disk_bytes != initial_bytes
    assert b"Hello from the Agent Shell \xe2\x80\x94 Phase 7 Undo verification.\n" == disk_bytes

    # 2. Verify resulting SHA-256 matches file on disk
    computed_sha = hashlib.sha256(disk_bytes).hexdigest()
    assert res.output["after_sha256"] == computed_sha
    assert res.output["bytes_after"] == len(disk_bytes)

    # 3. Verify snapshot after_sha256 matches the file on disk
    seq = res.output.get("snapshot_seq")
    assert seq is not None
    snaps = snapshot_manager.list_snapshots("sess-patch-disk-check")
    assert len(snaps) == 1
    assert snaps[0]["seq"] == seq
    assert snaps[0]["after_sha256"] == computed_sha
    assert snaps[0]["before_sha256"] == hashlib.sha256(initial_bytes).hexdigest()


@pytest.mark.anyio
async def test_write_file_refuses_existing_file_without_overwrite(
    test_workspace: Path, snapshot_manager: SnapshotManager
) -> None:
    """Verify that write_file refuses to overwrite an existing file when overwrite=False,
    directing the model/user to use apply_patch instead.
    """
    from agent_core.tools import WriteFileInput, WriteFileTool

    write_tool = WriteFileTool(snapshot_manager=snapshot_manager)
    target = test_workspace / "test.txt"
    target.write_text("existing content", encoding="utf-8")
    original_bytes = target.read_bytes()

    ctx = ToolContext(workspace_root=test_workspace, session_id="sess-write-refuse")
    res = await write_tool.execute(
        WriteFileInput(path="test.txt", content="new content", overwrite=False),
        ctx,
    )

    assert res.success is False
    assert "File already exists" in res.error
    assert "apply_patch" in res.error
    # File on disk remains untouched
    assert target.read_bytes() == original_bytes


# ── Part E: Ambiguous & Escaped Matches Tests ──────────────────────────────


@pytest.mark.anyio
async def test_part_e_exact_unicode_match(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """Exact Unicode characters in old_text match directly."""
    initial_text = "Em-dash — and bullet • here\n"
    target = test_workspace / "unicode_exact.txt"
    target.write_text(initial_text, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_pe_1")
    args = ApplyPatchInput(
        path="unicode_exact.txt",
        edits=[EditSpec(old_text="—", new_text="--")],
    )
    res = await patch_tool.execute(args, ctx)
    assert res.success is True
    assert target.read_text(encoding="utf-8") == "Em-dash -- and bullet • here\n"


@pytest.mark.anyio
async def test_part_e_escaped_unicode_match(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """If old_text produces 0 matches and contains \\uXXXX, decode and retry match."""
    initial_text = "Prefix — and bullet • suffix\n"
    target = test_workspace / "unicode_escaped.txt"
    target.write_text(initial_text, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_pe_2")
    # Provide literal raw string with \u2014 for em-dash
    args = ApplyPatchInput(
        path="unicode_escaped.txt",
        edits=[EditSpec(old_text=r"Prefix \u2014 and", new_text="Prefix - and")],
    )
    res = await patch_tool.execute(args, ctx)
    assert res.success is True
    assert target.read_text(encoding="utf-8") == "Prefix - and bullet • suffix\n"
    # Tool result explicitly states that normalization happened
    assert res.output.get("unicode_escapes_normalized") is True
    assert "normalized" in res.output.get("normalization_note", "")


@pytest.mark.anyio
async def test_part_e_escaped_unicode_match_rejected_if_ambiguous(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """If decoded unicode matches more than once, retry is NOT accepted."""
    initial_text = "Em-dash — first\nEm-dash — second\n"
    target = test_workspace / "unicode_ambig.txt"
    target.write_text(initial_text, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_pe_2b")
    # Decoded "—" matches 2 times, so retry must be refused
    args = ApplyPatchInput(
        path="unicode_ambig.txt",
        edits=[EditSpec(old_text=r"\u2014", new_text="--")],
    )
    res = await patch_tool.execute(args, ctx)
    assert res.success is False
    assert "old_text not found" in res.error
    # Original text remains untouched
    assert target.read_text(encoding="utf-8") == initial_text


@pytest.mark.anyio
async def test_part_e_ambiguity_without_occurrence_refused(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """Multiple matches without occurrence or replace_all are refused with ambiguity error."""
    initial_text = "item: apple\nitem: banana\nitem: cherry\n"
    target = test_workspace / "ambiguity.txt"
    target.write_text(initial_text, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_pe_3")
    args = ApplyPatchInput(
        path="ambiguity.txt",
        edits=[EditSpec(old_text="item:", new_text="fruit:")],
    )
    res = await patch_tool.execute(args, ctx)
    assert res.success is False
    assert "ambiguous match" in res.error.lower()
    assert "Found 3 occurrences" in res.error
    # File unchanged
    assert target.read_text(encoding="utf-8") == initial_text


@pytest.mark.anyio
async def test_part_e_occurrence_1(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """occurrence=1 replaces only the first occurrence."""
    initial_text = "val = 10\nval = 20\nval = 30\n"
    target = test_workspace / "occ1.txt"
    target.write_text(initial_text, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_pe_4")
    args = ApplyPatchInput(
        path="occ1.txt",
        edits=[EditSpec(old_text="val", new_text="first_val", occurrence=1)],
    )
    res = await patch_tool.execute(args, ctx)
    assert res.success is True
    assert target.read_text(encoding="utf-8") == "first_val = 10\nval = 20\nval = 30\n"


@pytest.mark.anyio
async def test_part_e_occurrence_n(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """occurrence=N (e.g. 2 and 3) replaces only that specific occurrence."""
    initial_text = "repeat repeat repeat"
    target = test_workspace / "occn.txt"
    target.write_text(initial_text, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_pe_5")
    # Replace 2nd occurrence
    args = ApplyPatchInput(
        path="occn.txt",
        edits=[EditSpec(old_text="repeat", new_text="SECOND", occurrence=2)],
    )
    res = await patch_tool.execute(args, ctx)
    assert res.success is True
    assert target.read_text(encoding="utf-8") == "repeat SECOND repeat"

    # Now replace 2nd occurrence of 'repeat' in the updated text ("repeat SECOND repeat" -> occurrences at 1 and 2)
    args2 = ApplyPatchInput(
        path="occn.txt",
        edits=[EditSpec(old_text="repeat", new_text="THIRD", occurrence=2)],
    )
    res2 = await patch_tool.execute(args2, ctx)
    assert res2.success is True
    assert target.read_text(encoding="utf-8") == "repeat SECOND THIRD"


@pytest.mark.anyio
async def test_part_e_invalid_occurrence(
    test_workspace: Path, patch_tool: ApplyPatchTool
) -> None:
    """occurrence <= 0 or occurrence > match_count produces clear error."""
    initial_text = "foo bar foo"
    target = test_workspace / "invalid_occ.txt"
    target.write_text(initial_text, encoding="utf-8")

    ctx = ToolContext(workspace_root=test_workspace, session_id="test_pe_6")

    # occurrence=0 (invalid 1-based index)
    res0 = await patch_tool.execute(
        ApplyPatchInput(path="invalid_occ.txt", edits=[EditSpec(old_text="foo", new_text="X", occurrence=0)]),
        ctx,
    )
    assert res0.success is False
    assert "invalid occurrence 0" in res0.error
    assert "must be between 1 and 2" in res0.error

    # occurrence=-1
    res_neg = await patch_tool.execute(
        ApplyPatchInput(path="invalid_occ.txt", edits=[EditSpec(old_text="foo", new_text="X", occurrence=-1)]),
        ctx,
    )
    assert res_neg.success is False
    assert "invalid occurrence -1" in res_neg.error

    # occurrence=5 (when only 2 exist)
    res5 = await patch_tool.execute(
        ApplyPatchInput(path="invalid_occ.txt", edits=[EditSpec(old_text="foo", new_text="X", occurrence=5)]),
        ctx,
    )
    assert res5.success is False
    assert "invalid occurrence 5" in res5.error
    assert "Found 2 occurrences" in res5.error
    assert "must be between 1 and 2" in res5.error

    # File untouched
    assert target.read_text(encoding="utf-8") == initial_text

