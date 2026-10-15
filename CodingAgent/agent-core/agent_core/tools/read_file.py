"""read_file tool implementation with binary detection, 64KB/2000-line limits, and injection detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .base import ToolContext, ToolResult, ToolSpec
from .injection_detector import detect_instruction_injection
from .path_guard import PathGuard, PathGuardError

# Constraints
MAX_LINES_PER_CALL = 2000
MAX_BYTES_PER_CALL = 64 * 1024  # 64 KB limit
BINARY_CHECK_HEADER_SIZE = 4096


def is_binary_content(header_bytes: bytes) -> bool:
    """Detect if header bytes indicate binary content."""
    if not header_bytes:
        return False

    # Check for null bytes (definitive binary indicator)
    if b"\x00" in header_bytes:
        return True

    # Attempt to decode as UTF-8
    try:
        header_text = header_bytes.decode("utf-8")
        # Check ratio of non-printable control characters
        non_printable = sum(
            1
            for ch in header_text
            if ord(ch) < 32 and ch not in ("\n", "\r", "\t", "\b")
        )
        if len(header_text) > 0 and (non_printable / len(header_text)) > 0.2:
            return True
    except UnicodeDecodeError:
        return True

    return False


class ReadFileInput(BaseModel):
    """Input parameters for reading a file."""

    path: str = Field(..., description="Path of the file to read (relative to workspace root)")
    start_line: Optional[int] = Field(
        default=None,
        gt=0,
        le=10_000_000,
        description="1-based start line number (default: 1)",
    )
    end_line: Optional[int] = Field(
        default=None,
        gt=0,
        le=10_000_000,
        description="1-based end line number (inclusive)",
    )

    @model_validator(mode="after")
    def validate_range(self) -> ReadFileInput:
        """Validate start_line <= end_line when both are specified."""
        if self.start_line is not None and self.end_line is not None:
            if self.start_line > self.end_line:
                raise ValueError(
                    f"start_line ({self.start_line}) cannot be greater than end_line ({self.end_line})"
                )
        return self


class ReadFileTool(ToolSpec):
    """Read file contents safely from the workspace with chunking and security guards."""

    name = "read_file"
    description = (
        "Read contents of a text file in the workspace. Supports 1-based start_line and end_line, "
        "enforces 64 KB and 2000-line limits per call, and refuses binary files."
    )
    input_model = ReadFileInput
    read_only = True

    async def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """Execute file reading safely."""
        if not isinstance(args, ReadFileInput):
            return ToolResult(success=False, error="Invalid arguments for read_file")

        guard = PathGuard(context.workspace_root)
        try:
            resolved_file = guard.validate_path(args.path)
        except PathGuardError as exc:
            return ToolResult(success=False, error=f"Path validation failed: {exc}")

        if not resolved_file.exists():
            return ToolResult(success=False, error=f"File does not exist: {args.path}")

        if resolved_file.is_dir():
            return ToolResult(
                success=False,
                error=f"Path is a directory, not a file: {args.path}",
            )

        # 1. Binary check
        try:
            with open(resolved_file, "rb") as bf:
                header = bf.read(BINARY_CHECK_HEADER_SIZE)
                if is_binary_content(header):
                    return ToolResult(
                        success=False,
                        error="File appears to be binary and cannot be read as text.",
                    )
        except OSError as exc:
            return ToolResult(
                success=False,
                error=f"Cannot inspect file: {exc.strerror or str(exc)}",
            )

        # 2. Line numbering calculation
        start_line = args.start_line or 1
        requested_end_line = args.end_line

        # 3. Read content line-by-line to avoid unbounded memory consumption
        collected_lines: List[str] = []
        accumulated_bytes = 0
        current_line_num = 1
        actual_start_line = start_line
        actual_end_line = start_line - 1
        truncated = False
        next_start_line: Optional[int] = None
        message: Optional[str] = None
        used_replacement_decoding = False

        try:
            # First try strict UTF-8
            encoding_mode = "utf-8"
            errors_mode = "strict"
            try:
                with open(resolved_file, "r", encoding=encoding_mode, errors=errors_mode) as test_f:
                    test_f.read(1024)
            except UnicodeDecodeError:
                # Safe fallback to utf-8 replacement characters
                errors_mode = "replace"
                used_replacement_decoding = True

            with open(resolved_file, "r", encoding=encoding_mode, errors=errors_mode) as f:
                for line in f:
                    if current_line_num < start_line:
                        current_line_num += 1
                        continue

                    if requested_end_line is not None and current_line_num > requested_end_line:
                        # Beyond requested end line
                        break

                    # Check max lines limit
                    if len(collected_lines) >= MAX_LINES_PER_CALL:
                        truncated = True
                        next_start_line = current_line_num
                        message = (
                            f"Output truncated at line {actual_end_line} / {MAX_LINES_PER_CALL}-line limit. "
                            f"Request start_line={next_start_line} to continue."
                        )
                        break

                    # Check 64 KB byte limit
                    line_bytes = len(line.encode("utf-8"))
                    if accumulated_bytes + line_bytes > MAX_BYTES_PER_CALL and collected_lines:
                        truncated = True
                        next_start_line = current_line_num
                        message = (
                            f"Output truncated at line {actual_end_line} / 64 KB limit. "
                            f"Request start_line={next_start_line} to continue."
                        )
                        break

                    collected_lines.append(line)
                    accumulated_bytes += line_bytes
                    actual_end_line = current_line_num
                    current_line_num += 1

                # If loop completed without breaking, check if more lines remain in file
                if not truncated:
                    remaining_line = f.readline()
                    if remaining_line:
                        # There was more in the file after requested_end_line
                        pass

        except OSError as exc:
            return ToolResult(
                success=False,
                error=f"Error reading file: {exc.strerror or str(exc)}",
            )

        content = "".join(collected_lines)

        # 4. Untrusted content / prompt injection detection
        possible_injection = detect_instruction_injection(content)
        warning: Optional[str] = None
        if possible_injection:
            warning = "Possible instruction injection phrase detected in file content."

        if used_replacement_decoding:
            repl_warning = "File contains non-UTF-8 characters decoded with replacement."
            warning = f"{warning} {repl_warning}".strip() if warning else repl_warning

        # Construct safe relative path representation for response
        try:
            display_path = (
                resolved_file.relative_to(context.workspace_root).as_posix()
            )
        except ValueError:
            display_path = args.path

        return ToolResult(
            success=True,
            output={
                "path": display_path,
                "content": content,
                "start_line": actual_start_line if collected_lines else start_line,
                "end_line": actual_end_line if collected_lines else start_line,
                "truncated": truncated,
                "next_start_line": next_start_line,
                "possible_injection": possible_injection,
                "warning": warning,
                "message": message,
            },
        )
