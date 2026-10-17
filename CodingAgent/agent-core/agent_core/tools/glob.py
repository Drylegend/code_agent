"""glob tool implementation for fast workspace file pattern matching."""

from __future__ import annotations

import os
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .base import ToolContext, ToolResult, ToolSpec
from .path_guard import PathGuard, PathGuardError

DEFAULT_MAX_RESULTS = 200
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


class GlobInput(BaseModel):
    """Input parameters for glob tool."""

    pattern: str = Field(
        ...,
        description="Glob pattern to search for files (e.g. '**/*.py')",
    )
    path: Optional[str] = Field(
        default=None,
        description="Directory path relative to workspace root",
    )


def matches_glob_pattern(rel_path: str, pattern: str) -> bool:
    """Evaluate if a relative path matches a glob pattern, supporting ** for 0+ directories."""
    if not pattern:
        return False
    if pattern in ("**", "**/*", "*"):
        return True
    pure = PurePath(rel_path)
    if pure.match(pattern):
        return True
    if pattern.startswith("**/"):
        sub = pattern[3:]
        if pure.match(sub):
            return True
    return False


class GlobTool(ToolSpec):
    """Locate files in the workspace matching a glob pattern."""

    name = "glob"
    description = "Search for files matching a glob pattern in the workspace."
    input_model = GlobInput
    read_only = True

    async def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        if not isinstance(args, GlobInput):
            return ToolResult(success=False, error="Invalid arguments for glob")

        guard = PathGuard(context.workspace_root)
        search_subpath = args.path.strip() if args.path and args.path.strip() else "."

        try:
            resolved_dir = guard.validate_path(search_subpath)
        except PathGuardError as exc:
            return ToolResult(success=False, error=f"Path validation failed: {exc}")

        if not resolved_dir.exists():
            return ToolResult(success=False, error=f"Directory does not exist: {search_subpath}")

        if not resolved_dir.is_dir():
            return ToolResult(success=False, error=f"Path is not a directory: {search_subpath}")

        pattern = args.pattern.strip()
        if not pattern:
            return ToolResult(success=False, error="Glob pattern cannot be empty")

        matched_rel_paths: List[str] = []
        workspace_root = context.workspace_root.resolve()

        # Traverse directory tree using os.walk, pruning ignored dirs
        for root_str, dirs, files in os.walk(resolved_dir, followlinks=False):
            current_path = Path(root_str)

            # Prevent following symlinks that resolve outside workspace
            if current_path.is_symlink():
                try:
                    guard.validate_path(current_path.resolve())
                except PathGuardError:
                    dirs[:] = []
                    continue

            # Prune ignored directory names in-place
            dirs[:] = [
                d for d in dirs
                if d not in DEFAULT_IGNORED_DIRS and not d.startswith(".git")
            ]

            # Evaluate files and directories
            for fname in files:
                file_full = current_path / fname

                # Check symlink target containment
                if file_full.is_symlink():
                    try:
                        guard.validate_path(file_full.resolve())
                    except PathGuardError:
                        continue

                try:
                    rel_from_ws = file_full.resolve().relative_to(workspace_root).as_posix()
                except ValueError:
                    continue

                try:
                    rel_from_search = file_full.relative_to(resolved_dir).as_posix()
                except ValueError:
                    rel_from_search = rel_from_ws

                # Match against relative path or filename
                if (
                    matches_glob_pattern(rel_from_search, pattern)
                    or matches_glob_pattern(rel_from_ws, pattern)
                    or matches_glob_pattern(fname, pattern)
                ):
                    matched_rel_paths.append(rel_from_ws)

        # Sort paths for deterministic results
        matched_rel_paths.sort()
        total_matches = len(matched_rel_paths)
        truncated = total_matches > DEFAULT_MAX_RESULTS
        returned_matches = matched_rel_paths[:DEFAULT_MAX_RESULTS]

        return ToolResult(
            success=True,
            output={
                "matches": returned_matches,
                "count": len(returned_matches),
                "total_matches": total_matches,
                "truncated": truncated,
            },
        )
