"""write_file tool implementation confined strictly within workspace boundaries."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .base import ToolContext, ToolResult, ToolSpec
from .path_guard import PathGuard, PathGuardError
from ..snapshots import SnapshotError, get_snapshot_manager

MAX_WRITE_CONTENT_SIZE = 1 * 1024 * 1024  # 1 MB


class WriteFileInput(BaseModel):
    """Input parameters for writing or creating a file."""

    path: str = Field(
        ...,
        description="File path relative to workspace root",
    )
    content: str = Field(
        ...,
        description="Complete file content to write",
    )
    overwrite: bool = Field(
        default=False,
        description="Set true to overwrite an existing file",
    )


class WriteFileTool(ToolSpec):
    """Safely write or create text files within the workspace root."""

    name = "write_file"
    description = "Create a file or overwrite an existing file in the workspace."
    input_model = WriteFileInput
    read_only = False

    def __init__(
        self,
        snapshot_manager: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
    ) -> None:
        self.snapshot_manager = snapshot_manager
        self.audit_logger = audit_logger

    async def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """Execute atomic file write with pre-write snapshotting."""
        if not isinstance(args, WriteFileInput):
            return ToolResult(success=False, error="Invalid arguments for write_file")

        # 1. Enforce size limit before any disk operations
        raw_bytes = args.content.encode("utf-8")
        if len(raw_bytes) > MAX_WRITE_CONTENT_SIZE:
            return ToolResult(
                success=False,
                error=f"Content size ({len(raw_bytes)} bytes) exceeds 1 MB limit ({MAX_WRITE_CONTENT_SIZE} bytes)",
            )

        # 2. Path security validation
        guard = PathGuard(context.workspace_root)
        try:
            target_path = guard.validate_write_path(args.path)
        except PathGuardError as exc:
            return ToolResult(success=False, error=f"Path security denied: {exc}")

        # 3. Parent directory existence check
        parent_dir = target_path.parent
        if not parent_dir.exists():
            return ToolResult(
                success=False,
                error=f"Parent directory does not exist: '{args.path}'. Parent directories are not automatically created.",
            )

        # 4. Overwrite rule check
        existed = target_path.exists()
        if existed:
            if target_path.is_dir():
                return ToolResult(
                    success=False,
                    error=f"Target path is a directory, not a file: '{args.path}'",
                )
            if not args.overwrite:
                return ToolResult(
                    success=False,
                    error=f"File already exists: '{args.path}'. To modify an existing file, use apply_patch or set overwrite=true.",
                )

        # 5. Windows fidelity (detect and preserve existing line endings if replacing)
        content_to_write = args.content
        if existed:
            try:
                orig_head = target_path.read_bytes()[:4096]
                if b"\r\n" in orig_head:
                    normalized = content_to_write.replace("\r\n", "\n")
                    content_to_write = normalized.replace("\n", "\r\n")
            except Exception:
                pass

        final_bytes = content_to_write.encode("utf-8")
        bytes_after = len(final_bytes)
        after_sha256 = hashlib.sha256(final_bytes).hexdigest()

        # 6. Pre-write snapshot (outside workspace)
        snapshot_mgr = self.snapshot_manager or get_snapshot_manager()

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
        bytes_before = target_path.stat().st_size if existed else 0

        audit_logger = self.audit_logger or getattr(context, "audit_logger", None)
        intent_id = f"intent_{context.session_id}_{snap_record.get('seq')}_{os.urandom(4).hex()}"
        write_start_time = time.perf_counter()

        # 7. Atomic write via temporary file in same directory
        temp_file: Optional[Path] = None
        try:
            # Unguessable temp file in parent_dir
            with tempfile.NamedTemporaryFile(
                dir=parent_dir,
                delete=False,
                prefix=".tmp_write_",
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

            # Narrow TOCTOU window: re-verify target path immediately before replace
            guard.validate_write_path(args.path)

            # Atomic replace
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
                    if existed:
                        if target_path.exists() and hashlib.sha256(target_path.read_bytes()).hexdigest() == before_sha256:
                            verify_ok = True
                    else:
                        if not target_path.exists():
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
                error=f"Failed to write file '{args.path}': {exc}",
            )
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)


        audit_metadata = {
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "operation": "write_file",
            "path": args.path,
        }

        output = {
            "path": args.path,
            "bytes_written": bytes_after,
            "created": not existed,
            "snapshot_seq": snap_record.get("seq"),
            "after_sha256": after_sha256,
            "audit_metadata": audit_metadata,
        }

        return ToolResult(success=True, output=output)

