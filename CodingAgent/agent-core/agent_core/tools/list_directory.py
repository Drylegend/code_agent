"""list_directory tool implementation confined to workspace boundaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .base import ToolContext, ToolResult, ToolSpec
from .path_guard import PathGuard, PathGuardError

# Default standard directories to omit from directory listings
DEFAULT_SKIPPED_DIRS = {
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
}


class ListDirectoryInput(BaseModel):
    """Input parameters for listing a directory."""

    path: str = Field(
        default=".",
        description="Path of the directory to list (relative to workspace root)",
    )
    max_entries: int = Field(
        default=200,
        gt=0,
        le=1000,
        description="Maximum entries to return (1-1000)",
    )

    @field_validator("path", mode="before")
    @classmethod
    def normalize_empty_path(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return "."
        return v


class ListDirectoryTool(ToolSpec):
    """List directory contents within the workspace root with security guards."""

    name = "list_directory"
    description = (
        "List files and subdirectories in a directory within the workspace. "
        "Returns item names, types ('file' or 'dir'), and file sizes in bytes."
    )
    input_model = ListDirectoryInput
    read_only = True

    async def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """Execute directory enumeration safely."""
        if not isinstance(args, ListDirectoryInput):
            return ToolResult(
                success=False,
                error="Invalid arguments for list_directory",
            )

        guard = PathGuard(context.workspace_root)
        try:
            resolved_dir = guard.validate_path(args.path)
        except PathGuardError as exc:
            return ToolResult(
                success=False,
                error=f"Path validation failed: {exc}",
            )

        if not resolved_dir.exists():
            return ToolResult(
                success=False,
                error=f"Directory does not exist: {args.path}",
            )

        if not resolved_dir.is_dir():
            return ToolResult(
                success=False,
                error=f"Path is a file, not a directory: {args.path}",
            )

        entries: List[Dict[str, Any]] = []
        skipped_reported: List[str] = []

        try:
            # Enumerate directory without recursive descent
            with os.scandir(resolved_dir) as it:
                for entry in it:
                    name = entry.name

                    # Omit and record default ignored directories
                    if entry.is_dir(follow_symlinks=False) and name in DEFAULT_SKIPPED_DIRS:
                        skipped_reported.append(name)
                        continue

                    # Basic metadata collection
                    is_dir = entry.is_dir(follow_symlinks=True)
                    item_type = "dir" if is_dir else "file"
                    size: Optional[int] = None

                    if not is_dir:
                        try:
                            size = entry.stat(follow_symlinks=True).st_size
                        except OSError:
                            size = None

                    entries.append({
                        "name": name,
                        "type": item_type,
                        "size": size,
                    })

        except OSError as exc:
            return ToolResult(
                success=False,
                error=f"Cannot read directory: {exc.strerror or str(exc)}",
            )

        # Deterministic case-insensitive name sorting
        entries.sort(key=lambda e: e["name"].lower())
        skipped_reported.sort(key=str.lower)

        total_found = len(entries)
        truncated = total_found > args.max_entries
        message: Optional[str] = None

        if truncated:
            entries = entries[: args.max_entries]
            message = (
                f"Directory listing truncated at {args.max_entries} entries (total: {total_found}). "
                "Request a narrower directory or another path to inspect the remaining entries."
            )

        # Construct safe relative path representation for response
        try:
            display_path = (
                resolved_dir.relative_to(context.workspace_root).as_posix() or "."
            )
        except ValueError:
            display_path = args.path

        return ToolResult(
            success=True,
            output={
                "path": display_path,
                "entries": entries,
                "skipped": skipped_reported,
                "truncated": truncated,
                "message": message,
            },
        )
