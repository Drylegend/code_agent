"""Pre-write snapshot and rollback management for agent-core.

Stores original file bytes and creation records outside the workspace before every
modifying write operation. Provides deterministic rollback via restore_snapshot.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

DEFAULT_STORE_DIR = Path(__file__).resolve().parent.parent / "store"
DEFAULT_MAX_BYTES_PER_SESSION = 50 * 1024 * 1024  # 50 MB
DEFAULT_MAX_SNAPSHOTS_PER_SESSION = 200


class SnapshotError(Exception):
    """Base exception for snapshot management errors."""

    pass


class SnapshotStorageLimitError(SnapshotError):
    """Raised when snapshot storage limit for a session is reached."""

    pass


class SnapshotNotFoundError(SnapshotError):
    """Raised when a requested snapshot sequence cannot be found."""

    pass


class RestoreResult:
    """Result of restoring a snapshot."""

    def __init__(
        self,
        success: bool,
        seq: int,
        path: str,
        action: str,  # 'restored' or 'deleted'
        restored_sha256: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.success = success
        self.seq = seq
        self.path = path
        self.action = action
        self.restored_sha256 = restored_sha256
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "seq": self.seq,
            "path": self.path,
            "action": self.action,
            "restored_sha256": self.restored_sha256,
            "error": self.error,
        }


class SnapshotManager:
    """Manages pre-write snapshots outside workspace boundaries."""

    def __init__(
        self,
        store_dir: Optional[Union[str, Path]] = None,
        max_bytes_per_session: int = DEFAULT_MAX_BYTES_PER_SESSION,
        max_snapshots_per_session: int = DEFAULT_MAX_SNAPSHOTS_PER_SESSION,
    ) -> None:
        raw_store = store_dir or os.environ.get("AGENT_CORE_STORE_DIR", DEFAULT_STORE_DIR)
        self.store_dir = Path(raw_store).resolve()
        self.snapshots_dir = self.store_dir / "snapshots"
        self.max_bytes_per_session = max_bytes_per_session
        self.max_snapshots_per_session = max_snapshots_per_session
        self._lock = threading.Lock()

        # Ensure snapshot base directory exists
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        """Return the directory containing snapshots for a specific session."""
        safe_session = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        if not safe_session:
            safe_session = "default"
        s_dir = self.snapshots_dir / safe_session
        s_dir.mkdir(parents=True, exist_ok=True)
        return s_dir

    def _get_index_path(self, session_id: str) -> Path:
        return self._get_session_dir(session_id) / "index.jsonl"

    def _get_session_usage(self, session_id: str) -> Tuple[int, int]:
        """Return (total_bytes, count) of snapshots stored for this session."""
        s_dir = self._get_session_dir(session_id)
        total_bytes = 0
        count = 0
        for entry in s_dir.glob("*.bin"):
            if entry.is_file():
                total_bytes += entry.stat().st_size
                count += 1
        return total_bytes, count

    def get_next_seq(self, session_id: str) -> int:
        """Determine next sequence number for this session."""
        index_file = self._get_index_path(session_id)
        if not index_file.exists():
            return 1
        last_seq = 0
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    seq = data.get("seq", 0)
                    if seq > last_seq:
                        last_seq = seq
                except json.JSONDecodeError:
                    continue
        return last_seq + 1

    def _prune_oldest(self, session_id: str, new_bytes: int = 0) -> None:
        """Prune oldest snapshots when count or byte limit is reached."""
        s_dir = self._get_session_dir(session_id)
        records = self.list_snapshots(session_id)
        if not records:
            return

        current_bytes, current_count = self._get_session_usage(session_id)
        while records and (
            current_count >= self.max_snapshots_per_session
            or (current_bytes + new_bytes > self.max_bytes_per_session)
        ):
            oldest = records.pop(0)
            snap_file = oldest.get("snapshot_file")
            if snap_file:
                p = s_dir / snap_file
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            current_bytes, current_count = self._get_session_usage(session_id)

        index_file = self._get_index_path(session_id)
        with open(index_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def create_pre_write_snapshot(
        self,
        session_id: str,
        target_file: Path,
        relative_path: str,
        tool: str,
        after_sha256: Optional[str] = None,
        workspace_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record pre-write state before modifying or creating a file.

        Args:
            session_id: Current conversational session identifier.
            target_file: Absolute path to the file on disk.
            relative_path: Path relative to the workspace root.
            tool: Invoked tool name ('write_file' or 'apply_patch').
            after_sha256: Predicted or calculated SHA256 of new content.

        Returns:
            Index record dictionary.

        Raises:
            SnapshotStorageLimitError: If a single snapshot exceeds total session limit.
        """
        with self._lock:
            existed = target_file.exists() and target_file.is_file()
            new_size = target_file.stat().st_size if existed else 0

            if existed and new_size > self.max_bytes_per_session:
                raise SnapshotStorageLimitError(
                    f"Single snapshot size ({new_size} bytes) exceeds maximum session limit ({self.max_bytes_per_session} bytes)"
                )

            # Prune oldest if necessary to make room
            self._prune_oldest(session_id, new_bytes=new_size)

            s_dir = self._get_session_dir(session_id)
            seq = self.get_next_seq(session_id)
            now_iso = datetime.now(timezone.utc).isoformat()

            before_sha256: Optional[str] = None
            snapshot_filename: Optional[str] = None

            if existed:
                original_bytes = target_file.read_bytes()
                before_sha256 = hashlib.sha256(original_bytes).hexdigest()
                prefix = before_sha256[:8]
                snapshot_filename = f"{seq}_{prefix}.bin"
                snapshot_path = s_dir / snapshot_filename
                snapshot_path.write_bytes(original_bytes)

            # Canonicalize workspace_root for cross-session persistence
            canonical_ws = str(Path(workspace_root).resolve()).replace("\\", "/") if workspace_root else None

            record = {
                "seq": seq,
                "path": str(relative_path).replace("\\", "/"),
                "existed": existed,
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "timestamp": now_iso,
                "tool": tool,
                "snapshot_file": snapshot_filename,
                "workspace_root": canonical_ws,
            }

            index_path = self._get_index_path(session_id)
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            return record

    def list_snapshots(self, session_id: str) -> List[Dict[str, Any]]:
        """List all indexed snapshots for a session."""
        index_file = self._get_index_path(session_id)
        if not index_file.exists():
            return []
        records = []
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def list_snapshots_by_workspace(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        include_legacy: bool = True,
    ) -> List[Dict[str, Any]]:
        """List all snapshots across all sessions for a given canonical workspace root.

        Enriches records with legacy status, undo availability, and conflict/corruption status.
        Survives agent-core restarts and desktop app restarts.
        """
        canonical_ws = (
            str(Path(workspace_root).resolve()).replace("\\", "/")
            if workspace_root is not None
            else None
        )
        results: List[Dict[str, Any]] = []
        if not self.snapshots_dir.exists():
            return results

        for session_dir in self.snapshots_dir.iterdir():
            if not session_dir.is_dir():
                continue
            index_file = session_dir / "index.jsonl"
            if not index_file.exists():
                continue
            session_id = session_dir.name
            with open(index_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    rec_ws = rec.get("workspace_root")
                    is_legacy = (rec_ws is None or str(rec_ws).strip() == "")

                    if canonical_ws is not None:
                        if rec_ws != canonical_ws and not (is_legacy and include_legacy):
                            continue

                    # Determine file and snapshot status
                    status = "clean"
                    can_undo = True
                    current_sha: Optional[str] = None

                    effective_ws_str = canonical_ws or rec_ws
                    effective_ws = Path(effective_ws_str).resolve() if effective_ws_str else None

                    if effective_ws is not None:
                        target_file = (effective_ws / rec["path"]).resolve()
                        if target_file.exists() and target_file.is_file():
                            try:
                                current_sha = hashlib.sha256(target_file.read_bytes()).hexdigest()
                            except Exception:
                                pass

                    # Snapshot integrity check for existed=True files
                    snap_file = rec.get("snapshot_file")
                    if rec.get("existed") and snap_file:
                        snap_path = session_dir / snap_file
                        if not snap_path.exists():
                            status = "corrupted"
                            can_undo = False
                        else:
                            try:
                                snap_bytes = snap_path.read_bytes()
                                snap_sha = hashlib.sha256(snap_bytes).hexdigest()
                                if snap_sha != rec.get("before_sha256"):
                                    status = "corrupted"
                                    can_undo = False
                            except Exception:
                                status = "corrupted"
                                can_undo = False

                    if status != "corrupted":
                        if is_legacy:
                            status = "legacy"
                        elif current_sha and rec.get("after_sha256") and current_sha != rec["after_sha256"]:
                            if current_sha == rec.get("before_sha256"):
                                status = "undone"
                            else:
                                status = "conflict"
                        elif not rec.get("existed") and effective_ws and not (effective_ws / rec["path"]).exists():
                            status = "undone"

                    results.append({
                        "seq": rec.get("seq"),
                        "session_id": session_id,
                        "path": rec.get("path"),
                        "workspace_root": rec_ws,
                        "tool": rec.get("tool"),
                        "existed": rec.get("existed", True),
                        "before_sha256": rec.get("before_sha256"),
                        "after_sha256": rec.get("after_sha256"),
                        "timestamp": rec.get("timestamp"),
                        "snapshot_file": snap_file,
                        "is_legacy": is_legacy,
                        "can_undo": can_undo,
                        "status": status,
                        "current_sha256": current_sha,
                    })

        # Sort by timestamp ascending
        results.sort(key=lambda r: r.get("timestamp", ""))
        return results

    def restore(
        self,
        session_id: str,
        seq: int,
        workspace_root: Path,
        verify_current_hash: bool = False,
        force: bool = False,
    ) -> RestoreResult:
        """Restore a file in the workspace to its pre-write state for snapshot seq.

        Integrity and security checks:
        1. Target path validated through PathGuard.validate_write_path.
        2. Snapshot file SHA-256 verified against before_sha256 before any write.
        3. Conflict check against after_sha256 unless force=True.
        4. Read-after-write verification on disk.

        Args:
            session_id: Conversational session identifier.
            seq: Sequence number of the snapshot to restore.
            workspace_root: Absolute canonical workspace root.
            verify_current_hash: If True, refuses if current file SHA-256
                                differs from expected after_sha256.
            force: If True, bypasses conflict check (allows restore over modified file).

        Returns:
            RestoreResult describing outcome.
        """
        with self._lock:
            s_dir = self._get_session_dir(session_id)
            records = self.list_snapshots(session_id)
            matching = [r for r in records if r.get("seq") == seq]
            if not matching:
                raise SnapshotNotFoundError(f"Snapshot with seq {seq} not found for session '{session_id}'")

            record = matching[-1]
            rel_path_str = record["path"]
            ws_resolved = Path(workspace_root).resolve()

            # A3. Restore target safety: exact same PathGuard as write_file and apply_patch
            from .tools.path_guard import PathGuard, PathGuardError
            guard = PathGuard(ws_resolved)
            try:
                target_path = guard.validate_write_path(rel_path_str)
            except (PathGuardError, ValueError) as exc:
                return RestoreResult(
                    success=False,
                    seq=seq,
                    path=rel_path_str,
                    action="denied",
                    error=f"Restore target path rejected by security policy: {exc}",
                )

            existed = record.get("existed", True)
            original_bytes: Optional[bytes] = None
            actual_snap_sha: Optional[str] = None

            # A2. Integrity check before restore: verify snapshot file before any write
            if existed:
                snap_name = record.get("snapshot_file")
                if not snap_name:
                    return RestoreResult(
                        success=False,
                        seq=seq,
                        path=rel_path_str,
                        action="corrupted",
                        error="Index indicates file existed before edit but snapshot_file is missing",
                    )

                snap_path = s_dir / snap_name
                if not snap_path.exists():
                    return RestoreResult(
                        success=False,
                        seq=seq,
                        path=rel_path_str,
                        action="corrupted",
                        error=f"Snapshot file not found on disk: {snap_path}",
                    )

                original_bytes = snap_path.read_bytes()
                actual_snap_sha = hashlib.sha256(original_bytes).hexdigest()
                expected_before = record.get("before_sha256")
                if expected_before and actual_snap_sha != expected_before:
                    return RestoreResult(
                        success=False,
                        seq=seq,
                        path=rel_path_str,
                        action="corrupted",
                        error=(
                            f"Snapshot integrity verification failed for '{rel_path_str}'. "
                            f"Snapshot file SHA-256 ({actual_snap_sha[:8]}) does not match expected "
                            f"before_sha256 ({expected_before[:8]}). Snapshot is corrupted or modified."
                        ),
                    )

            # Conflict checking: Ensure file hasn't changed since edit unless force=True
            if verify_current_hash and not force:
                expected_after = record.get("after_sha256")
                if expected_after is not None:
                    if target_path.exists() and target_path.is_file():
                        current_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
                    else:
                        current_sha = None
                    if current_sha != expected_after:
                        return RestoreResult(
                            success=False,
                            seq=seq,
                            path=rel_path_str,
                            action="conflict",
                            error=(
                                f"File '{rel_path_str}' has changed since the snapshot was created. "
                                f"Expected SHA-256 {expected_after[:8]}, but found {current_sha[:8] if current_sha else 'missing'}. "
                                "Undo aborted to prevent overwriting newer changes."
                            ),
                        )

            # Execution
            if not existed:
                # File was newly created by this edit, so undo = delete
                if target_path.exists():
                    target_path.unlink()
                return RestoreResult(
                    success=True,
                    seq=seq,
                    path=rel_path_str,
                    action="deleted",
                    restored_sha256=None,
                )
            else:
                assert original_bytes is not None
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(original_bytes)

                # Read-after-write verification on disk
                written_bytes = target_path.read_bytes()
                written_sha = hashlib.sha256(written_bytes).hexdigest()
                if written_sha != actual_snap_sha:
                    return RestoreResult(
                        success=False,
                        seq=seq,
                        path=rel_path_str,
                        action="failed",
                        error="Read-after-write verification failed during snapshot restoration.",
                    )

                return RestoreResult(
                    success=True,
                    seq=seq,
                    path=rel_path_str,
                    action="restored",
                    restored_sha256=actual_snap_sha,
                )


# Global singleton instance
_default_manager: Optional[SnapshotManager] = None


def get_snapshot_manager(store_dir: Optional[Union[str, Path]] = None) -> SnapshotManager:
    """Retrieve or create default SnapshotManager instance."""
    global _default_manager
    if _default_manager is None or store_dir is not None:
        _default_manager = SnapshotManager(store_dir=store_dir)
    return _default_manager


def restore_snapshot(
    seq: int,
    session_id: str = "default",
    workspace_root: Optional[Union[str, Path]] = None,
    store_dir: Optional[Union[str, Path]] = None,
    verify_current_hash: bool = False,
    force: bool = False,
) -> RestoreResult:
    """Non-agent function / CLI command to undo an edit by restoring a snapshot."""
    if workspace_root is None:
        raise ValueError("workspace_root must be provided to restore_snapshot")
    ws = Path(workspace_root).resolve()
    manager = get_snapshot_manager(store_dir=store_dir)
    return manager.restore(
        session_id=session_id,
        seq=seq,
        workspace_root=ws,
        verify_current_hash=verify_current_hash,
        force=force,
    )

