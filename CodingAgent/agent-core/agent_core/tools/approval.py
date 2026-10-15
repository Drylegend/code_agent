"""Deterministic tool-layer approval gate for agent-core."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Callable, Coroutine, Dict, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from .audit import AuditLogger

logger = logging.getLogger("approval_gate")

DEFAULT_APPROVAL_TIMEOUT = 120.0  # seconds


class ApprovalRequest(BaseModel):
    request_id: str
    session_id: str
    tool_name: str
    path: str
    diff_preview: Optional[str] = None
    byte_size: Optional[int] = None
    args_hash: str
    normalized_args: Dict[str, Any]
    created_at: float
    timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT


class ApprovalManager:
    """Manages pending approval requests, event emission, timeouts, and verification."""

    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
        default_timeout: float = DEFAULT_APPROVAL_TIMEOUT,
    ) -> None:
        self.audit_logger = audit_logger
        self.default_timeout = default_timeout
        self._pending_requests: Dict[str, ApprovalRequest] = {}
        self._pending_futures: Dict[str, asyncio.Future[bool]] = {}
        self._consumed_request_ids: Set[str] = set()
        self._event_sink: Optional[Callable[[Dict[str, Any]], Any]] = None
        self._lock = asyncio.Lock()

    def set_event_sink(
        self, sink: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Register the WebSocket dispatch callback to send approval_request events."""
        self._event_sink = sink

    def set_emitter(
        self, emitter: Callable[[ApprovalRequest], Any]
    ) -> None:
        """Register an emitter callback taking an ApprovalRequest object."""
        self._event_sink = lambda payload: emitter(
            self._pending_requests.get(payload.get("request_id"))
        )

    @staticmethod
    def compute_args_hash(raw_args: Dict[str, Any]) -> str:
        """Compute deterministic SHA256 of normalized tool arguments."""
        canonical = json.dumps(raw_args, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def request_approval(
        self,
        session_id: str,
        tool_name: str,
        validated_args: Dict[str, Any],
        path: str,
        diff_preview: Optional[str] = None,
        byte_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Create an approval request, log audit event, notify client, and await decision.

        Returns:
            True if user approved and args_hash verified; False if rejected or timed out.
        """
        request_id = f"apr_{uuid4().hex[:12]}"
        args_hash = self.compute_args_hash(validated_args)
        now = asyncio.get_running_loop().time()
        effective_timeout = timeout if timeout is not None else self.default_timeout

        req = ApprovalRequest(
            request_id=request_id,
            session_id=session_id,
            tool_name=tool_name,
            path=path,
            diff_preview=diff_preview,
            byte_size=byte_size,
            args_hash=args_hash,
            normalized_args=validated_args,
            created_at=now,
            timeout_seconds=effective_timeout,
        )

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        async with self._lock:
            self._pending_requests[request_id] = req
            self._pending_futures[request_id] = future

        # 1. Log audit: approval_requested
        if self.audit_logger is not None:
            self.audit_logger.log(
                session_id=session_id,
                tool=tool_name,
                args=validated_args,
                outcome="approval_requested",
                duration=0.0,
                extra={"request_id": request_id, "args_hash": args_hash},
            )

        # 2. Emit approval_request over WebSocket
        if self._event_sink is not None:
            event_payload = {
                "type": "approval_request",
                "request_id": request_id,
                "session_id": session_id,
                "tool": tool_name,
                "path": path,
                "diff_preview": diff_preview,
                "bytes": byte_size,
                "timeout": effective_timeout,
            }
            try:
                res = self._event_sink(event_payload)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.error("Failed to emit approval_request event: %s", exc)

        # 3. Await user decision with timeout
        try:
            decision = await asyncio.wait_for(future, timeout=effective_timeout)
            return decision
        except asyncio.TimeoutError:
            logger.warning("Approval request %s timed out after %.1f seconds", request_id, effective_timeout)
            async with self._lock:
                self._pending_requests.pop(request_id, None)
                self._pending_futures.pop(request_id, None)
                self._consumed_request_ids.add(request_id)

            if self.audit_logger is not None:
                self.audit_logger.log(
                    session_id=session_id,
                    tool=tool_name,
                    args=validated_args,
                    outcome="approval_expired",
                    duration=effective_timeout,
                    extra={"request_id": request_id, "args_hash": args_hash},
                )
            return False
        except asyncio.CancelledError:
            # Stop button / session request cancellation
            async with self._lock:
                self._pending_requests.pop(request_id, None)
                self._pending_futures.pop(request_id, None)
                self._consumed_request_ids.add(request_id)

            if self.audit_logger is not None:
                self.audit_logger.log(
                    session_id=session_id,
                    tool=tool_name,
                    args=validated_args,
                    outcome="approval_rejected",
                    duration=0.0,
                    extra={"request_id": request_id, "args_hash": args_hash, "reason": "cancelled"},
                )
            return False

    async def submit_response(
        self,
        request_id: str,
        decision: str,
        session_id: Optional[str] = None,
        args_hash: Optional[str] = None,
    ) -> bool:
        """Process incoming user approval_response (Approve or Reject).

        Returns:
            True if request was found and handled, False if unknown/expired/replayed.
        """
        async with self._lock:
            if request_id in self._consumed_request_ids or request_id not in self._pending_requests:
                logger.warning("Attempted to respond to invalid, replayed, or expired request_id=%s", request_id)
                return False

            req = self._pending_requests[request_id]

            if session_id and req.session_id != session_id:
                logger.warning("Session mismatch for request_id=%s: %s != %s", request_id, session_id, req.session_id)
                return False

            if args_hash and req.args_hash != args_hash:
                logger.warning("Args hash mismatch for request_id=%s: %s != %s", request_id, args_hash, req.args_hash)
                return False

            self._pending_requests.pop(request_id)
            future = self._pending_futures.pop(request_id, None)
            self._consumed_request_ids.add(request_id)

        if future is None or future.done():
            return False

        normalized_decision = decision.lower().strip()
        is_approved = normalized_decision in ("approve", "approved", "yes")

        outcome = "approval_granted" if is_approved else "approval_rejected"
        if self.audit_logger is not None:
            self.audit_logger.log(
                session_id=req.session_id,
                tool=req.tool_name,
                args=req.normalized_args,
                outcome=outcome,
                duration=0.0,
                extra={"request_id": request_id, "args_hash": req.args_hash},
            )

        future.set_result(is_approved)
        return True

    def cancel_session_requests(self, session_id: str) -> None:
        """Cancel all pending approval requests for a session (e.g. on Stop button)."""
        to_cancel = []
        for req_id, req in list(self._pending_requests.items()):
            if req.session_id == session_id:
                to_cancel.append(req_id)

        for req_id in to_cancel:
            self._pending_requests.pop(req_id, None)
            fut = self._pending_futures.pop(req_id, None)
            self._consumed_request_ids.add(req_id)
            if fut and not fut.done():
                fut.cancel()

    def clear(self) -> None:
        """Reset state (simulates sidecar restart)."""
        for fut in self._pending_futures.values():
            if not fut.done():
                fut.cancel()
        self._pending_requests.clear()
        self._pending_futures.clear()
        self._consumed_request_ids.clear()
