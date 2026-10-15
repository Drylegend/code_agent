"""Cryptographically linked append-only JSONL audit logger for agent-core."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Dict, Optional

GENESIS_HASH = "0" * 64


def compute_record_hash(record_without_hash: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hex digest for an audit record payload."""
    canonical = json.dumps(record_without_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLogger:
    """Thread-safe append-only audit logger enforcing a tamper-evident hash chain."""

    def __init__(
        self,
        log_path: Union[str, Path],
        intent_log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.log_path = Path(log_path).resolve()
        if intent_log_path is not None:
            self.intent_log_path = Path(intent_log_path).resolve()
        else:
            self.intent_log_path = self.log_path.parent / (self.log_path.stem + "_intents.jsonl")
        self._lock = threading.RLock()
        self._last_hash: str = GENESIS_HASH
        self._intent_last_hash: str = GENESIS_HASH

        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.intent_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize last_hash from existing file if present
        self._initialize_last_hash()
        self._initialize_intent_last_hash()

    def _initialize_last_hash(self) -> None:
        """Read the existing audit log to find the latest valid hash in the chain."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            self._last_hash = GENESIS_HASH
            return

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "hash" in data:
                        self._last_hash = data["hash"]
                except json.JSONDecodeError:
                    pass

    def _initialize_intent_last_hash(self) -> None:
        """Read the existing intent log to find the latest valid hash in the chain."""
        if not self.intent_log_path.exists() or self.intent_log_path.stat().st_size == 0:
            self._intent_last_hash = GENESIS_HASH
            return

        with open(self.intent_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "hash" in data:
                        self._intent_last_hash = data["hash"]
                except json.JSONDecodeError:
                    pass


    def log(
        self,
        session_id: str,
        tool: str,
        args: Dict[str, Any],
        outcome: str,
        duration: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an audit event into the append-only JSONL file.

        Args:
            session_id: Current conversational session identifier.
            tool: Name of the invoked tool.
            args: Invocation arguments (file/patch contents will be sanitized).
            outcome: Execution outcome ('success', 'denied', or 'error').
            duration: Duration of execution in seconds.
            extra: Optional additional metadata fields (e.g. before/after sha256).

        Returns:
            The complete written audit record including previous_hash and hash.
        """
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()

            # Sanitize args to ensure no large strings/file contents are stored
            sanitized_args: Dict[str, Any] = {}
            for k, v in args.items():
                k_lower = k.lower()
                if k_lower in ("content", "patch"):
                    val_str = str(v)
                    sanitized_args[k] = f"[omitted: {len(val_str)} chars]"
                elif k_lower == "edits" and isinstance(v, list):
                    sanitized_args[k] = f"[{len(v)} edits omitted]"
                elif isinstance(v, (str, int, float, bool)) or v is None:
                    sanitized_args[k] = v
                elif isinstance(v, (list, tuple)):
                    sanitized_args[k] = list(v)
                else:
                    sanitized_args[k] = str(v)

            record_payload = {
                "timestamp": now_iso,
                "session_id": session_id,
                "tool": tool,
                "args": sanitized_args,
                "outcome": outcome,
                "duration": round(float(duration), 6),
                "previous_hash": self._last_hash,
            }

            if extra:
                for k, v in extra.items():
                    if k not in record_payload and k != "hash":
                        record_payload[k] = v

            rec_hash = compute_record_hash(record_payload)
            full_record = dict(record_payload)
            full_record["hash"] = rec_hash

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(full_record) + "\n")

            self._last_hash = rec_hash
            return full_record

    def _log_intent_record(
        self,
        session_id: str,
        tool: str,
        args: Dict[str, Any],
        outcome: str,
        duration: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an entry in the dedicated two-phase intent journal."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            record_payload = {
                "timestamp": now_iso,
                "session_id": session_id,
                "tool": tool,
                "args": args,
                "outcome": outcome,
                "duration": round(float(duration), 6),
                "previous_hash": self._intent_last_hash,
            }
            if extra:
                for k, v in extra.items():
                    if k not in record_payload and k != "hash":
                        record_payload[k] = v

            rec_hash = compute_record_hash(record_payload)
            full_record = dict(record_payload)
            full_record["hash"] = rec_hash

            with open(self.intent_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(full_record) + "\n")

            self._intent_last_hash = rec_hash
            return full_record

    def log_write_intent(
        self,
        session_id: str,
        tool: str,
        intent_id: str,
        path: str,
        temp_file: Optional[Union[str, Path]],
        before_sha256: Optional[str],
        after_sha256: Optional[str],
        snapshot_seq: Optional[int] = None,
        workspace_root: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Record the intent phase of a two-phase write operation."""
        return self._log_intent_record(
            session_id=session_id,
            tool=tool,
            args={"path": path},
            outcome="intent",
            duration=0.0,
            extra={
                "phase": "write_intent",
                "intent_id": intent_id,
                "path": str(path).replace("\\", "/"),
                "temp_file": str(temp_file).replace("\\", "/") if temp_file else None,
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "snapshot_seq": snapshot_seq,
                "workspace_root": str(Path(workspace_root).resolve()).replace("\\", "/") if workspace_root else None,
            },
        )

    def log_write_committed(
        self,
        session_id: str,
        tool: str,
        intent_id: str,
        path: str,
        after_sha256: Optional[str],
        duration: float = 0.0,
        snapshot_seq: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Record the committed phase of a two-phase write operation."""
        return self._log_intent_record(
            session_id=session_id,
            tool=tool,
            args={"path": path},
            outcome="committed",
            duration=duration,
            extra={
                "phase": "write_committed",
                "intent_id": intent_id,
                "path": str(path).replace("\\", "/"),
                "after_sha256": after_sha256,
                "snapshot_seq": snapshot_seq,
            },
        )

    def log_write_failed(
        self,
        session_id: str,
        tool: str,
        intent_id: str,
        path: str,
        error: str,
        duration: float = 0.0,
    ) -> Dict[str, Any]:
        """Record the failed phase of a two-phase write operation."""
        return self._log_intent_record(
            session_id=session_id,
            tool=tool,
            args={"path": path},
            outcome="failed",
            duration=duration,
            extra={
                "phase": "write_failed",
                "intent_id": intent_id,
                "path": str(path).replace("\\", "/"),
                "error": error,
            },
        )

    def recover_uncommitted_intents(
        self, workspace_root: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """Identify uncommitted write intents, clean leftover temp files, and write recovery audit records.

        Guarantees:
        - Interrupted writes are marked with write_interrupted audit records.
        - Leftover temporary files are cleaned up safely.
        - The target file is never blindly overwritten.
        """
        with self._lock:
            if not self.intent_log_path.exists() or self.intent_log_path.stat().st_size == 0:
                return []

            open_intents: Dict[str, Dict[str, Any]] = {}
            with open(self.intent_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    phase = record.get("phase")
                    intent_id = record.get("intent_id")
                    if not intent_id:
                        continue

                    if phase == "write_intent":
                        open_intents[intent_id] = record
                    elif phase in ("write_committed", "write_failed", "write_interrupted"):
                        open_intents.pop(intent_id, None)

            if not open_intents:
                return []

            recovered: List[Dict[str, Any]] = []
            for intent_id, intent_record in open_intents.items():
                cleaned_files: List[str] = []

                # Resolve and validate workspace root
                ws = None
                if workspace_root:
                    try:
                        ws = Path(workspace_root).resolve()
                    except Exception:
                        ws = None
                elif intent_record.get("workspace_root"):
                    try:
                        ws = Path(intent_record["workspace_root"]).resolve()
                    except Exception:
                        ws = None

                # Clean ONLY the exact temp file recorded in the intent record,
                # strictly validated to be inside the recorded workspace_root.
                # Never glob target directories.
                temp_file_str = intent_record.get("temp_file")
                if temp_file_str:
                    try:
                        tf = Path(temp_file_str).resolve()
                        if (
                            ws is not None
                            and tf.is_relative_to(ws)
                            and tf.is_file()
                            and tf.name.startswith(".tmp_")
                        ):
                            tf.unlink(missing_ok=True)
                            cleaned_files.append(str(tf).replace("\\", "/"))
                    except Exception:
                        pass

                # 3. Log recovery in intent journal (marking intent as interrupted)
                self._log_intent_record(
                    session_id=intent_record.get("session_id", "system"),
                    tool="recovery",
                    args={"intent_id": intent_id, "path": intent_record.get("path")},
                    outcome="interrupted",
                    duration=0.0,
                    extra={
                        "phase": "write_interrupted",
                        "event": "startup_recovery",
                        "intent_id": intent_id,
                        "path": intent_record.get("path"),
                        "cleaned_temp_files": cleaned_files,
                    },
                )

                # 4. Write recovery audit record to main audit log
                rec_payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": intent_record.get("session_id", "system"),
                    "tool": "recovery",
                    "args": {"intent_id": intent_id, "path": intent_record.get("path")},
                    "outcome": "interrupted",
                    "duration": 0.0,
                    "previous_hash": self._last_hash,
                    "phase": "write_interrupted",
                    "event": "startup_recovery",
                    "intent_id": intent_id,
                    "path": intent_record.get("path"),
                    "cleaned_temp_files": cleaned_files,
                }
                rec_hash = compute_record_hash(rec_payload)
                full_rec = dict(rec_payload)
                full_rec["hash"] = rec_hash

                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(full_rec) + "\n")

                self._last_hash = rec_hash
                recovered.append(full_rec)

            return recovered

    def get_last_hash(self) -> str:
        """Return the current tip of the hash chain."""
        with self._lock:
            return self._last_hash


