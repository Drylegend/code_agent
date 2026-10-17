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
    "venv",
    "dist",
    "agent-store",
    "logs",
}


class ListDirectoryInput(BaseModel):
    """Input parameters for listing a directory."""

    path: str = Field(
        default=".",
        description="Directory path relative to workspace root (default '.')",
    )
    recursive: bool = Field(
        default=False,
        description="Enumerate subdirectories recursively (default false)",
    )
    max_entries: int = Field(
        default=200,
        gt=0,
        le=1000,
        description="Max entries to return (default 200)",
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
    description = "List files and subdirectories in a workspace directory."
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
            if args.recursive:
                for root, dirs, files in os.walk(resolved_dir, topdown=True, followlinks=False):
                    # Prune default ignored directories in-place
                    ignored = [d for d in dirs if d in DEFAULT_SKIPPED_DIRS]
                    for d in ignored:
                        skipped_reported.append(d)
                        dirs.remove(d)

                    rel_root = Path(root).relative_to(resolved_dir)

                    for d in dirs:
                        rel_item = (rel_root / d).as_posix()
                        entries.append({
                            "name": rel_item,
                            "type": "dir",
                            "size": None,
                        })

                    for f in files:
                        rel_item = (rel_root / f).as_posix()
                        full_p = Path(root) / f
                        try:
                            sz = full_p.stat(follow_symlinks=False).st_size
                        except OSError:
                            sz = None
                        entries.append({
                            "name": rel_item,
                            "type": "file",
                            "size": sz,
                        })
            else:
                # Non-recursive: enumerate immediate children
                with os.scandir(resolved_dir) as it:
                    for entry in it:
                        name = entry.name

                        # Omit and record default ignored directories
                        if entry.is_dir(follow_symlinks=False) and name in DEFAULT_SKIPPED_DIRS:
                            skipped_reported.append(name)
                            continue

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
        skipped_reported = sorted(list(set(skipped_reported)), key=str.lower)

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
                "entry_count": len(entries),
                "total_found": total_found,
                "skipped": skipped_reported,
                "truncated": truncated,
                "recursive": args.recursive,
                "message": message,
            },
        )
