"""Windows-safe workspace path guard and denylist enforcement for agent-core."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import List, Optional, Union


class PathGuardError(Exception):
    """Exception raised when a path violates workspace containment or denylist rules."""

    pass


# Default protected patterns to deny access to sensitive files/directories
DEFAULT_DENY_PATTERNS = [
    ".env*",
    "*.pem",
    "*.key",
    "id_rsa*",
    ".git/config",
    ".aws",
    ".ssh",
    "credentials*",
    "secrets*",
]

# Write-specific protected execution configuration paths and directories
DEFAULT_PROTECTED_WRITE_DIRS = {
    ".git",
    ".vscode",
    ".github",
    ".husky",
}

DEFAULT_PROTECTED_WRITE_FILES = {
    ".pre-commit-config.yaml",
    ".gitlab-ci.yml",
    ".agent_audit.jsonl",
    "audit.jsonl",
}

# Windows reserved device names (case-insensitive)
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

AGENT_SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent  # .../agent-core
DEFAULT_AGENT_STORE_ROOT = Path(r"D:\agent-store")


class PathGuard:
    """Validates and confines requested filesystem paths strictly within a workspace root.

    Provides Windows-specific protections:
    - Resolves all paths, symlinks, and directory junctions before containment checks.
    - Rejects path traversal (..) escapes.
    - Rejects Windows drive-letter switches (e.g. C: vs D:).
    - Rejects unauthorized UNC network paths.
    - Rejects Alternate Data Streams (ADS) and reserved device names.
    - Denies access to secret and credential file patterns.
    - Denies writes to protected execution configs (.git, .vscode, .github, .husky, CI configs).
    - Prevents write access into agent store, logs, or agent source code.
    """

    def __init__(
        self,
        workspace_root: Union[str, Path],
        denylist_patterns: Optional[List[str]] = None,
        protected_write_dirs: Optional[List[str]] = None,
        protected_write_files: Optional[List[str]] = None,
    ) -> None:
        raw_root = Path(workspace_root).resolve()
        if not raw_root.exists():
            # Allow workspace path even if created dynamically in tests, but canonicalize
            pass
        self.workspace_root = raw_root
        self.denylist_patterns: List[str] = (
            list(denylist_patterns)
            if denylist_patterns is not None
            else list(DEFAULT_DENY_PATTERNS)
        )
        self.protected_write_dirs = (
            set(d.lower() for d in protected_write_dirs)
            if protected_write_dirs is not None
            else set(DEFAULT_PROTECTED_WRITE_DIRS)
        )
        self.protected_write_files = (
            set(f.lower() for f in protected_write_files)
            if protected_write_files is not None
            else set(DEFAULT_PROTECTED_WRITE_FILES)
        )

    def _check_windows_quirks(self, raw_str: str) -> None:
        """Check for Windows-specific evasion attempts (ADS, reserved names, trailing dots)."""
        # 1. Alternate Data Stream check (e.g. "file.txt:stream" or "a.txt:evil")
        # Drive letter at index 1 (e.g. "C:\...") is permitted
        clean_str = raw_str
        if len(clean_str) >= 2 and clean_str[1] == ":" and clean_str[0].isalpha():
            clean_str = clean_str[2:]

        if ":" in clean_str:
            raise PathGuardError(f"Alternate data streams (':') are not allowed: '{raw_str}'")

        # 2. Trailing dots and spaces check on path components
        # Normalize separators
        normalized = raw_str.replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        for part in parts:
            if part in (".", ".."):
                continue
            if part.endswith(".") or part.endswith(" "):
                raise PathGuardError(f"Paths with trailing dots or spaces are not allowed: '{part}'")

            # 3. Reserved Windows device names check (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
            base_stem = part.split(".")[0].upper()
            if base_stem in WINDOWS_RESERVED_NAMES:
                raise PathGuardError(f"Path references reserved Windows device name '{base_stem}': '{raw_str}'")

    def validate_path(self, requested_path: Union[str, Path]) -> Path:
        """Validate, resolve, and confine a requested path within the workspace root.

        Args:
            requested_path: Relative or absolute path string or Path object.

        Returns:
            Resolved, canonical Path guaranteed to be inside workspace_root and not denylisted.

        Raises:
            PathGuardError: If the path escapes the workspace root, switches drives,
                            points to a UNC share, or matches a denylisted pattern.
        """
        raw_input_str = str(requested_path)
        if not raw_input_str or not raw_input_str.strip():
            raise PathGuardError("Path cannot be empty")

        # Pre-resolution Windows quirk checks (rejects trailing spaces/dots, ADS, reserved names)
        self._check_windows_quirks(raw_input_str)

        raw_str = raw_input_str.strip()

        # 1. UNC checks
        is_unc_input = raw_str.startswith(("\\\\", "//"))
        is_ws_unc = str(self.workspace_root).startswith(("\\\\", "//"))
        if is_unc_input and not is_ws_unc:
            raise PathGuardError("Path is a UNC path outside the workspace")

        # 2. Resolve target path
        req_p = Path(raw_str)
        if req_p.is_absolute():
            resolved_target = req_p.resolve()
        else:
            resolved_target = (self.workspace_root / req_p).resolve()

        resolved_str = str(resolved_target)
        if resolved_str.startswith(("\\\\", "//")) and not is_ws_unc:
            raise PathGuardError("Path is a UNC path outside the workspace")

        # 3. Windows drive-letter containment check
        if resolved_target.drive and self.workspace_root.drive:
            if resolved_target.drive.upper() != self.workspace_root.drive.upper():
                raise PathGuardError(
                    f"Path is on a different drive: '{resolved_target.drive}' vs workspace drive '{self.workspace_root.drive}'"
                )

        # 4. Containment check (case-normalized for Windows compatibility)
        norm_target_str = os.path.normcase(str(resolved_target))
        norm_ws_str = os.path.normcase(str(self.workspace_root))

        try:
            norm_target_path = Path(norm_target_str)
            norm_ws_path = Path(norm_ws_str)
            rel_path = norm_target_path.relative_to(norm_ws_path)

            if ".." in rel_path.parts:
                raise PathGuardError("Path escapes workspace root")
        except ValueError:
            raise PathGuardError("Path resolves outside workspace root")

        # 5. Denylist check (evaluated on the relative path inside workspace)
        self.check_denylist(rel_path)

        # 6. Explicitly deny access to audit log storage even if workspace encloses it
        try:
            store_dir_env = os.environ.get("AGENT_CORE_STORE_DIR")
            store_root = Path(store_dir_env).resolve() if store_dir_env else DEFAULT_AGENT_STORE_ROOT
            audit_rel = resolved_target.relative_to(store_root)
            if "audit" in [p.lower() for p in audit_rel.parts] or "audit" in resolved_target.name.lower():
                raise PathGuardError("Access to audit log storage is denied by security policy")
        except ValueError:
            pass

        return resolved_target

    def validate_write_path(self, requested_path: Union[str, Path]) -> Path:
        """Validate path specifically for modifying operations (write_file / apply_patch).

        Enforces all validate_path checks plus:
        - Protection for .git, .vscode, .github, .husky directories.
        - Protection for CI and workflow execution configs.
        - Protection against modifying agent-core internal storage, logs, or source code.
        - Rejection of symlinks and reparse points on the target file.

        Returns:
            Resolved canonical Path ready for safe write.

        Raises:
            PathGuardError: If any write security policy is violated.
        """
        resolved_target = self.validate_path(requested_path)

        # Re-check relative path inside workspace root
        norm_target_str = os.path.normcase(str(resolved_target))
        norm_ws_str = os.path.normcase(str(self.workspace_root))
        rel_path = Path(norm_target_str).relative_to(Path(norm_ws_str))

        parts_lower = [p.lower() for p in rel_path.parts]
        filename_lower = rel_path.name.lower()

        # 1. Deny writes to protected directories (.git, .vscode, .github, .husky)
        for part in parts_lower:
            if part in self.protected_write_dirs:
                raise PathGuardError(
                    f"Write access to protected directory '{part}' is denied by security policy"
                )

        # 2. Deny writes to protected configuration files
        if filename_lower in self.protected_write_files:
            raise PathGuardError(
                f"Write access to protected configuration file '{rel_path.name}' is denied by security policy"
            )

        # Also check wildcard matching for audit and log files
        if fnmatch.fnmatch(filename_lower, "*.log") or filename_lower.endswith(".jsonl"):
            if "audit" in filename_lower or filename_lower.startswith(".agent"):
                raise PathGuardError(
                    f"Write access to audit/log file '{rel_path.name}' is denied by security policy"
                )

        # 3. Deny writes into agent-core source code if workspace happens to enclose it
        try:
            resolved_target.relative_to(AGENT_SOURCE_ROOT)
            raise PathGuardError("Write access to agent-core source code is denied by security policy")
        except ValueError:
            pass

        # 4. Deny writes into agent store or audit storage if workspace happens to enclose it
        try:
            store_dir_env = os.environ.get("AGENT_CORE_STORE_DIR")
            store_root = Path(store_dir_env).resolve() if store_dir_env else DEFAULT_AGENT_STORE_ROOT
            resolved_target.relative_to(store_root)
            raise PathGuardError("Write access to agent store or audit storage is denied by security policy")
        except ValueError:
            pass

        # 5. Check if existing target is a symlink or reparse point
        if resolved_target.is_symlink() or os.path.islink(resolved_target):
            raise PathGuardError("Write access through symlinks or reparse points is denied")

        return resolved_target

    def check_denylist(self, rel_path: Path) -> None:
        """Check if any segment or filename matches a denylisted pattern.

        Raises:
            PathGuardError: If a denylisted pattern is matched.
        """
        posix_rel = rel_path.as_posix().lower()
        parts_lower = [p.lower() for p in rel_path.parts]
        filename_lower = rel_path.name.lower()

        for pattern in self.denylist_patterns:
            pat_lower = pattern.lower()

            # Match against filename (e.g. .env, id_rsa, cert.pem, credentials.json)
            if filename_lower and fnmatch.fnmatch(filename_lower, pat_lower):
                raise PathGuardError(
                    f"Access to protected file pattern '{pattern}' is denied"
                )

            # Match against directory parts (e.g. .aws, .ssh)
            for part in parts_lower:
                if fnmatch.fnmatch(part, pat_lower):
                    raise PathGuardError(
                        f"Access to protected path pattern '{pattern}' is denied"
                    )

            # Match against full relative path (e.g. .git/config)
            if (
                fnmatch.fnmatch(posix_rel, pat_lower)
                or posix_rel == pat_lower
                or posix_rel.endswith(f"/{pat_lower}")
            ):
                raise PathGuardError(
                    f"Access to protected path pattern '{pattern}' is denied"
                )
