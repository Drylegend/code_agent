"""grep tool implementation for pure Python workspace content search."""

from __future__ import annotations

import os
from pathlib import Path, PurePath
import re
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .base import ToolContext, ToolResult, ToolSpec
from .glob import matches_glob_pattern
from .path_guard import PathGuard, PathGuardError

MAX_PATTERN_LENGTH = 500
MAX_LINE_LENGTH = 4096
MAX_SNIPPET_LENGTH = 200
MAX_FILE_SIZE_BYTES = 1024 * 1024  # 1 MB
MAX_RESULTS_HARD_CAP = 200
BINARY_CHECK_BYTES = 1024

DEFAULT_IGNORED_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "agent-store",
    "logs",
    ".agents",
    "store",
    ".pytest_cache",
}


class GrepInput(BaseModel):
    """Input parameters for grep tool."""

    pattern: str = Field(
        ...,
        description="Search pattern string or regular expression. Use grep instead of repeated read_file calls to find text or code patterns.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Optional directory or file path relative to workspace root (defaults to workspace root).",
    )
    glob: Optional[str] = Field(
        default=None,
        description="Optional glob pattern to filter candidate files, e.g. '*.py' or '**/*.ts'.",
    )
    regex: bool = Field(
        default=False,
        description="Whether to interpret pattern as a regular expression. Defaults to False (exact literal string search).",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether search is case-sensitive. Defaults to False.",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=MAX_RESULTS_HARD_CAP,
        description="Maximum number of matching lines to return (default 100, capped at 200).",
    )


def is_binary_file(path: Path) -> bool:
    """Check if file contains null bytes in its initial chunk."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(BINARY_CHECK_BYTES)
            return b"\x00" in chunk
    except Exception:
        return True


class GrepTool(ToolSpec):
    """Search workspace files for matching strings or regular expressions."""

    name = "grep"
    description = (
        "Search file contents in the workspace for text or regex patterns without reading each file completely. "
        "Supports literal search (default) and regex search (regex=true), file path scoping, and glob filtering. "
        "Automatically skips binary files, files >1MB, .git, and node_modules. "
        "Returns structured results with file paths, 1-based line numbers, and truncated snippets."
    )
    input_model = GrepInput
    read_only = True

    async def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        if not isinstance(args, GrepInput):
            return ToolResult(success=False, error="Invalid arguments for grep")

        pattern = args.pattern
        if not pattern:
            return ToolResult(success=False, error="Search pattern cannot be empty")

        if len(pattern) > MAX_PATTERN_LENGTH:
            return ToolResult(
                success=False,
                error=f"Pattern length ({len(pattern)}) exceeds maximum limit of {MAX_PATTERN_LENGTH} characters",
            )

        # Compile regex if requested
        compiled_regex: Optional[re.Pattern[str]] = None
        if args.regex:
            flags = 0 if args.case_sensitive else re.IGNORECASE
            try:
                compiled_regex = re.compile(pattern, flags)
            except re.error as exc:
                return ToolResult(
                    success=False,
                    error=f"invalid regex: {exc}, try regex=false for a literal search",
                )
        else:
            needle = pattern if args.case_sensitive else pattern.lower()

        guard = PathGuard(context.workspace_root)
        target_subpath = args.path.strip() if args.path and args.path.strip() else "."

        try:
            resolved_target = guard.validate_path(target_subpath)
        except PathGuardError as exc:
            return ToolResult(success=False, error=f"Path validation failed: {exc}")

        if not resolved_target.exists():
            return ToolResult(success=False, error=f"Path does not exist: {target_subpath}")

        workspace_root = context.workspace_root.resolve()
        candidate_files: List[Path] = []

        if resolved_target.is_file():
            # Check symlink containment
            if resolved_target.is_symlink():
                try:
                    guard.validate_path(resolved_target.resolve())
                except PathGuardError:
                    return ToolResult(success=False, error=f"Symlink resolves outside workspace: {target_subpath}")
            candidate_files.append(resolved_target)
        else:
            # Walk directory tree
            for root_str, dirs, files in os.walk(resolved_target, followlinks=False):
                current_dir = Path(root_str)

                # Skip outside symlinks
                if current_dir.is_symlink():
                    try:
                        guard.validate_path(current_dir.resolve())
                    except PathGuardError:
                        dirs[:] = []
                        continue

                dirs[:] = [
                    d for d in dirs
                    if d not in DEFAULT_IGNORED_DIRS and not d.startswith(".git")
                ]

                for fname in files:
                    file_path = current_dir / fname

                    if file_path.is_symlink():
                        try:
                            guard.validate_path(file_path.resolve())
                        except PathGuardError:
                            continue

                    # Filter by glob if provided
                    if args.glob:
                        try:
                            rel_from_ws = file_path.resolve().relative_to(workspace_root).as_posix()
                        except ValueError:
                            rel_from_ws = fname

                        if not (
                            matches_glob_pattern(fname, args.glob)
                            or matches_glob_pattern(rel_from_ws, args.glob)
                        ):
                            continue

                    candidate_files.append(file_path)

        # Sort candidate files for deterministic output
        candidate_files.sort()

        matches: List[Dict[str, Any]] = []
        files_searched = 0
        files_skipped_binary = 0
        files_skipped_large = 0
        total_matches_count = 0
        max_allowed_results = min(args.max_results, MAX_RESULTS_HARD_CAP)
        is_truncated = False

        for file_path in candidate_files:
            try:
                st = file_path.stat()
                if st.st_size > MAX_FILE_SIZE_BYTES:
                    files_skipped_large += 1
                    continue
            except Exception:
                continue

            if is_binary_file(file_path):
                files_skipped_binary += 1
                continue

            try:
                rel_path = file_path.resolve().relative_to(workspace_root).as_posix()
            except ValueError:
                rel_path = file_path.name

            files_searched += 1

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_idx, line in enumerate(f, start=1):
                        # Cap line length to prevent ReDoS / CPU lockups
                        capped_line = line[:MAX_LINE_LENGTH]

                        matched = False
                        if args.regex:
                            if compiled_regex and compiled_regex.search(capped_line):
                                matched = True
                        else:
                            haystack = capped_line if args.case_sensitive else capped_line.lower()
                            if needle in haystack:
                                matched = True

                        if matched:
                            total_matches_count += 1
                            if len(matches) < max_allowed_results:
                                clean_line = capped_line.rstrip("\r\n")
                                if len(clean_line) > MAX_SNIPPET_LENGTH:
                                    snippet = clean_line[:MAX_SNIPPET_LENGTH] + "..."
                                else:
                                    snippet = clean_line

                                matches.append({
                                    "file": rel_path,
                                    "line_number": line_idx,
                                    "line": snippet,
                                })
                            else:
                                is_truncated = True
            except Exception:
                continue

        return ToolResult(
            success=True,
            output={
                "matches": matches,
                "count": len(matches),
                "total_matches": total_matches_count,
                "truncated": is_truncated or total_matches_count > len(matches),
                "files_searched": files_searched,
                "files_skipped_binary": files_skipped_binary,
                "files_skipped_large": files_skipped_large,
            },
        )
