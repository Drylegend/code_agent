"""apply_patch tool implementation for deterministic, sequential text replacements."""

from __future__ import annotations

import codecs
import difflib
import hashlib
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import ToolContext, ToolResult, ToolSpec
from .path_guard import PathGuard, PathGuardError
from ..snapshots import SnapshotError, get_snapshot_manager

MAX_PATCH_SIZE = 256 * 1024  # 256 KB
MAX_RESULT_FILE_SIZE = 1 * 1024 * 1024  # 1 MB
BINARY_HEADER_SIZE = 4096

UNICODE_ESCAPE_PATTERN = re.compile(r"\\u[0-9a-fA-F]{4}")


def decode_unicode_escapes(text: str) -> str:
    """Decode literal \\uXXXX sequences into their unicode characters."""
    return UNICODE_ESCAPE_PATTERN.sub(
        lambda m: chr(int(m.group(0)[2:], 16)),
        text,
    )


class EditSpec(BaseModel):
    """Specification for an individual text replacement within a patch."""

    old_text: str = Field(..., min_length=1, description="Exact unique text to replace")
    new_text: str = Field(..., description="Replacement text")
    occurrence: Optional[int] = Field(
        default=None,
        description="1-based match index if repeated",
    )


class ApplyPatchInput(BaseModel):
    """Input parameters for apply_patch."""

    path: str = Field(
        ...,
        description="File path relative to workspace root",
    )
    edits: List[EditSpec] = Field(
        ...,
        min_length=1,
        description="Sequential text replacements to apply",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_text",
    )


def is_binary_content(header: bytes) -> bool:
    """Check if byte chunk represents binary data."""
    if b"\x00" in header:
        return True
    return False


def find_closest_region_hint(source: str, query: str) -> str:
    """Find a contextual hint for an unmatched old_text snippet."""
    query_lines = [q.strip() for q in query.splitlines() if q.strip()]
    if not query_lines:
        return ""
    first_needle = query_lines[0]
    source_lines = source.splitlines()
    matches = difflib.get_close_matches(first_needle, source_lines, n=1, cutoff=0.6)
    if matches:
        return f"Closest matching line in file: '{matches[0]}'"
    return "Check indentation, whitespace, and exact surrounding context."


DEFAULT_MIN_REPLACE_ALL_CHARS = 8
DEFAULT_MAX_REPLACE_ALL_MATCHES = 20
DEFAULT_MAX_REPLACE_ALL_DELTA = 10 * 1024


class ApplyPatchTool(ToolSpec):
    """Apply deterministic sequential text replacements to an existing workspace file."""

    name = "apply_patch"
    description = (
        "Apply targeted text edits to an existing text file. Each edit specifies old_text and new_text. "
        "By default, each old_text must match exactly once. "
        "For an insertion: old_text should be a unique existing line or block; new_text should preserve that "
        "existing text and add the requested content; do not use a short/common token such as '\\n' as old_text; "
        "do not use replace_all for a single insertion unless the user explicitly requested every occurrence."
    )
    input_model = ApplyPatchInput
    read_only = False

    min_replace_all_chars: int = DEFAULT_MIN_REPLACE_ALL_CHARS
    max_replace_all_matches: int = DEFAULT_MAX_REPLACE_ALL_MATCHES
    max_replace_all_delta: int = DEFAULT_MAX_REPLACE_ALL_DELTA

    def __init__(
        self,
        snapshot_manager: Optional[Any] = None,
        min_replace_all_chars: Optional[int] = None,
        max_replace_all_matches: Optional[int] = None,
        max_replace_all_delta: Optional[int] = None,
        audit_logger: Optional[Any] = None,
    ) -> None:
        self.snapshot_manager = snapshot_manager
        self.audit_logger = audit_logger
        if min_replace_all_chars is not None:
            self.min_replace_all_chars = min_replace_all_chars
        if max_replace_all_matches is not None:
            self.max_replace_all_matches = max_replace_all_matches
        if max_replace_all_delta is not None:
            self.max_replace_all_delta = max_replace_all_delta

    async def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """Execute deterministic sequential patch with pre-write snapshot."""
        if not isinstance(args, ApplyPatchInput):
            return ToolResult(success=False, error="Invalid arguments for apply_patch")

        # 1. Enforce patch input size limit before any file operations
        total_patch_size = sum(
            len(e.old_text.encode("utf-8")) + len(e.new_text.encode("utf-8"))
            for e in args.edits
        )
        if total_patch_size > MAX_PATCH_SIZE:
            return ToolResult(
                success=False,
                error=f"Total patch size ({total_patch_size} bytes) exceeds 256 KB limit ({MAX_PATCH_SIZE} bytes)",
            )

        # 2. Path security validation
        guard = PathGuard(context.workspace_root)
        try:
            target_path = guard.validate_write_path(args.path)
        except PathGuardError as exc:
            return ToolResult(success=False, error=f"Path security denied: {exc}")

        # 3. File existence and type checks
        if not target_path.exists():
            return ToolResult(
                success=False,
                error=f"File does not exist: '{args.path}'. apply_patch requires an existing file.",
            )

        if not target_path.is_file():
            return ToolResult(
                success=False,
                error=f"Target path is a directory, not a file: '{args.path}'",
            )

        # 4. Size and binary check
        raw_original = target_path.read_bytes()
        bytes_before = len(raw_original)
        if bytes_before > MAX_RESULT_FILE_SIZE:
            return ToolResult(
                success=False,
                error=f"File size ({bytes_before} bytes) exceeds 1 MB limit ({MAX_RESULT_FILE_SIZE} bytes)",
            )
        header = raw_original[:BINARY_HEADER_SIZE]
        if is_binary_content(header):
            return ToolResult(
                success=False,
                error=f"Cannot apply patch to binary file: '{args.path}'",
            )

        # 5. Detect line ending, BOM, and decode
        has_bom = raw_original.startswith(codecs.BOM_UTF8)
        content_bytes = raw_original[len(codecs.BOM_UTF8):] if has_bom else raw_original

        try:
            content_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            return ToolResult(
                success=False,
                error=f"File '{args.path}' is not valid UTF-8 text: {exc}",
            )

        # Detect line endings in original file
        is_crlf = "\r\n" in content_text

        # Normalize working buffer to LF for consistent matching
        normalized_content = content_text.replace("\r\n", "\n")

        # 6. Apply edits sequentially in memory
        total_replacements = 0
        working_content = normalized_content
        any_escapes_normalized = False

        for idx, edit in enumerate(args.edits, start=1):
            if not edit.old_text:
                return ToolResult(
                    success=False,
                    error=f"Edit #{idx} failed: old_text cannot be empty",
                )

            norm_old = edit.old_text.replace("\r\n", "\n")
            norm_new = edit.new_text.replace("\r\n", "\n")

            match_count = working_content.count(norm_old)

            if match_count == 0 and UNICODE_ESCAPE_PATTERN.search(norm_old):
                decoded_old = decode_unicode_escapes(norm_old)
                # Retry is accepted ONLY if the decoded old_text matches exactly once
                if working_content.count(decoded_old) == 1:
                    norm_old = decoded_old
                    match_count = 1
                    any_escapes_normalized = True

            if match_count == 0:
                hint = find_closest_region_hint(working_content, norm_old)
                return ToolResult(
                    success=False,
                    error=f"Edit #{idx} failed: old_text not found in '{args.path}'. Match count: 0. {hint}",
                )

            if edit.occurrence is not None:
                if edit.occurrence <= 0 or edit.occurrence > match_count:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Edit #{idx} failed: invalid occurrence {edit.occurrence} for '{args.path}'. "
                            f"Found {match_count} occurrence{'s' if match_count != 1 else ''} "
                            f"(occurrence must be between 1 and {match_count})."
                        ),
                    )

                # Replace ONLY that specific 1-based occurrence
                parts = working_content.split(norm_old)
                k = edit.occurrence
                working_content = norm_old.join(parts[:k]) + norm_new + norm_old.join(parts[k:])
                total_replacements += 1

            elif args.replace_all:
                non_ws_count = len("".join(norm_old.split()))
                if non_ws_count < self.min_replace_all_chars:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Edit #{idx} failed: replace_all safety rule triggered: old_text has too few non-whitespace characters "
                            f"(actual: {non_ws_count}, minimum required: {self.min_replace_all_chars}). "
                            f"Please use a more specific/longer old_text or a narrower edit."
                        ),
                    )

                if match_count > self.max_replace_all_matches:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Edit #{idx} failed: replace_all safety rule triggered: match count exceeds limit "
                            f"(actual: {match_count} matches, maximum allowed: {self.max_replace_all_matches}). "
                            f"Please use a more specific/longer old_text or a narrower edit."
                        ),
                    )

                single_delta = len(norm_new.encode("utf-8")) - len(norm_old.encode("utf-8"))
                edit_delta = single_delta * match_count
                if abs(edit_delta) > self.max_replace_all_delta:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Edit #{idx} failed: replace_all safety rule triggered: total size delta exceeds limit "
                            f"(actual: {abs(edit_delta)} bytes, maximum allowed: {self.max_replace_all_delta} bytes). "
                            f"Please use a more specific/longer old_text or a narrower edit."
                        ),
                    )

                working_content = working_content.replace(norm_old, norm_new)
                total_replacements += match_count

            elif match_count > 1:
                # Find all matching occurrences and extract up to 5 with bounded surrounding context
                match_indices: List[int] = []
                search_pos = 0
                while len(match_indices) < 5:
                    found_idx = working_content.find(norm_old, search_pos)
                    if found_idx == -1:
                        break
                    match_indices.append(found_idx)
                    search_pos = found_idx + max(len(norm_old), 1)

                content_lines = working_content.split("\n")
                match_snippets: List[str] = []
                for m_num, m_pos in enumerate(match_indices, start=1):
                    line_no = working_content[:m_pos].count("\n") + 1
                    # 1 line before and 1 line after match
                    start_ctx = max(0, line_no - 2)
                    end_ctx = min(len(content_lines), line_no + norm_old.count("\n") + 1)
                    ctx_lines = [
                        f"  {ln + 1:4d} | {content_lines[ln]}"
                        for ln in range(start_ctx, end_ctx)
                    ]
                    match_snippets.append(f"Occurrence #{m_num} (line {line_no}):\n" + "\n".join(ctx_lines))

                matches_block = "\n\n".join(match_snippets)
                limit_note = " (only the first 5 matches shown; there are more matches in file)" if match_count > 5 else ""

                return ToolResult(
                    success=False,
                    error=(
                        f"Edit #{idx} failed: ambiguous match in '{args.path}'. "
                        f"Found {match_count} occurrences (old_text matched {match_count} times). "
                        f"Must match exactly once when replace_all is false.\n\n"
                        f"First matching occurrences{limit_note}:\n{matches_block}\n\n"
                        f"Please call read_file and provide enough surrounding context to make old_text unique."
                    ),
                )
            else:
                working_content = working_content.replace(norm_old, norm_new, 1)
                total_replacements += 1

        # 7. Restore original line endings and BOM
        if is_crlf:
            final_text = working_content.replace("\n", "\r\n")
        else:
            final_text = working_content

        final_bytes = (codecs.BOM_UTF8 if has_bom else b"") + final_text.encode("utf-8")
        bytes_after = len(final_bytes)
        delta = bytes_after - bytes_before

        # Generate actual unified diff preview
        diff_lines = list(
            difflib.unified_diff(
                content_text.splitlines(keepends=True),
                final_text.splitlines(keepends=True),
                fromfile=f"a/{args.path}",
                tofile=f"b/{args.path}",
            )
        )
        diff_str = "".join(diff_lines)
        DIFF_PREVIEW_LIMIT = 4096
        if len(diff_str) > DIFF_PREVIEW_LIMIT:
            diff_preview = diff_str[:DIFF_PREVIEW_LIMIT] + "\n...(diff truncated at 4 KB)"
        else:
            diff_preview = diff_str

        # Calculate out-of-proportion warning flag (does NOT block edit)
        total_new_len = sum(len(e.new_text) for e in args.edits)
        warning = (total_replacements > 3) or (abs(delta) > 10 * max(total_new_len, 1))

        # 8. Check resulting file size limit
        if bytes_after > MAX_RESULT_FILE_SIZE:
            return ToolResult(
                success=False,
                error=f"Resulting file size ({bytes_after} bytes) exceeds 1 MB limit ({MAX_RESULT_FILE_SIZE} bytes)",
            )

        # 9. Pre-write snapshot
        snapshot_mgr = self.snapshot_manager or get_snapshot_manager()
        after_sha256 = hashlib.sha256(final_bytes).hexdigest()

        try:
            snap_record = snapshot_mgr.create_pre_write_snapshot(
                session_id=context.session_id,
                target_file=target_path,
                relative_path=args.path,
                tool=self.name,
                after_sha256=after_sha256,
                workspace_root=str(context.workspace_root) if context.workspace_root else None,
            )
        except SnapshotError as exc:
            return ToolResult(
                success=False,
                error=f"Pre-write snapshot failed: {exc}",
            )

        before_sha256 = snap_record.get("before_sha256")

        audit_logger = self.audit_logger or getattr(context, "audit_logger", None)
        intent_id = f"intent_{context.session_id}_{snap_record.get('seq')}_{os.urandom(4).hex()}"
        write_start_time = time.perf_counter()

        # 10. Atomic write via temporary file in same directory
        temp_file: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target_path.parent,
                delete=False,
                prefix=".tmp_patch_",
                suffix=".bin",
            ) as tf:
                temp_file = Path(tf.name)
                tf.write(final_bytes)
                tf.flush()
                os.fsync(tf.fileno())

            # Log write intent before replace
            if audit_logger and hasattr(audit_logger, "log_write_intent"):
                audit_logger.log_write_intent(
                    session_id=context.session_id,
                    tool=self.name,
                    intent_id=intent_id,
                    path=args.path,
                    temp_file=str(temp_file),
                    before_sha256=before_sha256,
                    after_sha256=after_sha256,
                    snapshot_seq=snap_record.get("seq"),
                    workspace_root=context.workspace_root,
                )

            # Re-verify path immediately before replace (narrow TOCTOU window)
            guard.validate_write_path(args.path)

            os.replace(temp_file, target_path)
            temp_file = None

            # Read-after-write verification on disk
            written_bytes = target_path.read_bytes()
            written_sha256 = hashlib.sha256(written_bytes).hexdigest()
            if written_sha256 != after_sha256:
                if audit_logger and hasattr(audit_logger, "log_write_failed"):
                    audit_logger.log_write_failed(
                        session_id=context.session_id,
                        tool=self.name,
                        intent_id=intent_id,
                        path=args.path,
                        error=f"Read-after-write SHA mismatch: expected {after_sha256}, got {written_sha256}",
                        duration=time.perf_counter() - write_start_time,
                    )
                # Safely restore previous snapshot
                rollback_res = snapshot_mgr.restore(
                    session_id=context.session_id,
                    seq=snap_record.get("seq"),
                    workspace_root=context.workspace_root,
                    force=True,
                )
                verify_ok = False
                if rollback_res.success:
                    if target_path.exists() and hashlib.sha256(target_path.read_bytes()).hexdigest() == before_sha256:
                        verify_ok = True

                return ToolResult(
                    success=False,
                    error=(
                        f"Read-after-write verification failed for '{args.path}': "
                        f"content on disk did not match expected SHA-256. "
                        f"Rollback to previous snapshot {'verified' if verify_ok else 'failed'}."
                    ),
                )

            # Log write committed
            if audit_logger and hasattr(audit_logger, "log_write_committed"):
                audit_logger.log_write_committed(
                    session_id=context.session_id,
                    tool=self.name,
                    intent_id=intent_id,
                    path=args.path,
                    after_sha256=after_sha256,
                    duration=time.perf_counter() - write_start_time,
                    snapshot_seq=snap_record.get("seq"),
                )

        except PermissionError as exc:
            if audit_logger and hasattr(audit_logger, "log_write_failed"):
                audit_logger.log_write_failed(
                    session_id=context.session_id,
                    tool=self.name,
                    intent_id=intent_id,
                    path=args.path,
                    error=str(exc),
                    duration=time.perf_counter() - write_start_time,
                )
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)
            return ToolResult(
                success=False,
                error=f"File is locked or in use by another process: '{args.path}'. Please retry when the file is available.",
            )
        except Exception as exc:
            if audit_logger and hasattr(audit_logger, "log_write_failed"):
                audit_logger.log_write_failed(
                    session_id=context.session_id,
                    tool=self.name,
                    intent_id=intent_id,
                    path=args.path,
                    error=str(exc),
                    duration=time.perf_counter() - write_start_time,
                )
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)
            return ToolResult(
                success=False,
                error=f"Failed to apply patch to '{args.path}': {exc}",
            )
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)


        audit_metadata = {
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "delta": delta,
            "operation": "apply_patch",
            "path": args.path,
            "replacement_count": total_replacements,
            "unicode_escapes_normalized": any_escapes_normalized,
        }

        output = {
            "path": args.path,
            "replacements": total_replacements,
            "edits_applied": len(args.edits),
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "delta": delta,
            "preview": diff_preview,
            "warning": warning,
            "bytes_written": bytes_after,
            "snapshot_seq": snap_record.get("seq"),
            "after_sha256": after_sha256,
            "unicode_escapes_normalized": any_escapes_normalized,
            "audit_metadata": audit_metadata,
        }
        if any_escapes_normalized:
            output["normalization_note"] = "Unicode escapes in old_text were normalized to match target content."

        return ToolResult(success=True, output=output)

