"""Explicit workspace root validation and resolution for agent-core."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


class WorkspaceError(Exception):
    """Exception raised when an explicit workspace root path fails validation."""

    pass


def validate_workspace_root(raw_path: Union[str, Path, None]) -> Path:
    """Validate, resolve, and canonicalize an explicit workspace root path.

    Enforces:
    1. Path must not be None or empty string.
    2. Path must be an absolute path (or UNC path). Relative paths are rejected.
    3. Path must exist on the filesystem.
    4. Path must be a directory, not a regular file or device.
    5. Path is canonicalized (.resolve()) to ensure a stable security boundary.
    6. UNC paths are allowed if they exist and point to a directory.
    7. Never falls back to process cwd or implicit parent directories.

    Args:
        raw_path: The raw workspace root path supplied by the client or caller.

    Returns:
        Canonical absolute Path to an existing directory.

    Raises:
        WorkspaceError: If validation fails.
    """
    if raw_path is None:
        raise WorkspaceError("Workspace root path cannot be None")

    raw_str = str(raw_path).strip()
    if not raw_str:
        raise WorkspaceError("Workspace root path cannot be empty")

    path_obj = Path(raw_str)

    # Must be absolute on the host OS (handles Windows drive letters and UNC paths)
    if not path_obj.is_absolute():
        raise WorkspaceError(
            f"Workspace root must be an absolute path: '{raw_str}'"
        )

    try:
        resolved = path_obj.resolve()
    except Exception as exc:
        raise WorkspaceError(
            f"Cannot resolve workspace root path '{raw_str}': {exc}"
        ) from exc

    if not resolved.exists():
        raise WorkspaceError(
            f"Workspace root directory does not exist: '{raw_str}'"
        )

    if not resolved.is_dir():
        raise WorkspaceError(
            f"Workspace root must be a directory, not a file: '{raw_str}'"
        )

    return resolved
