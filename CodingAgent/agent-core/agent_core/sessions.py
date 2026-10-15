"""In-memory session management and history tracking for agent-core."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_tool_call(tc: Any) -> Dict[str, Any]:
    """Ensure a tool call object is converted to a standard OpenAI-style dictionary."""
    if isinstance(tc, dict):
        return dict(tc)
    if hasattr(tc, "model_dump"):
        return tc.model_dump()
    if hasattr(tc, "dict"):
        return tc.dict()

    tc_id = getattr(tc, "id", None)
    fn = getattr(tc, "function", None)
    fn_name = getattr(fn, "name", "") if fn else ""
    fn_args = getattr(fn, "arguments", "{}") if fn else "{}"
    if not isinstance(fn_args, str):
        fn_args = json.dumps(fn_args)

    return {
        "id": str(tc_id) if tc_id else "",
        "type": "function",
        "function": {
            "name": fn_name,
            "arguments": fn_args,
        },
    }


class Session:
    """Represents a single conversational session with its message history, workspace root, and active task."""

    def __init__(self, session_id: str, workspace_root: Optional[Path] = None) -> None:
        self.session_id: str = session_id
        self.workspace_root: Optional[Path] = workspace_root
        self.autonomy_level: int = 2  # Level 2 "Ask" (default), Level 3 "Scoped"
        self.messages: List[Dict[str, Any]] = []
        self.active_task: Optional[asyncio.Task] = None
        self._lock: asyncio.Lock = asyncio.Lock()

    def set_autonomy_level(self, level: int) -> None:
        """Update session autonomy level (2 for Ask, 3 for Scoped)."""
        if level not in (2, 3):
            raise ValueError(f"Invalid autonomy level: {level}. Allowed levels are 2 (Ask) and 3 (Scoped).")
        self.autonomy_level = level

    def get_autonomy_level(self) -> int:
        return self.autonomy_level

    def set_workspace_root(self, root: Path) -> None:
        """Bind an explicit validated workspace root to this session."""
        self.workspace_root = root

    def add_system_message(self, content: str) -> None:
        """Prepend or set system message."""
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content: str) -> None:
        """Append a user message to the session history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Append a completed assistant message to the session history."""
        self.messages.append({"role": "assistant", "content": content})

    def add_assistant_tool_calls(
        self, tool_calls: List[Any], content: Optional[str] = None
    ) -> None:
        """Append an assistant message containing one or more tool calls."""
        normalized = [_normalize_tool_call(tc) for tc in tool_calls]
        self.messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": normalized,
        })

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        """Append a tool result message correlating to a specific tool_call_id."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        })

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return a copy of the current message history."""
        return copy.deepcopy(self.messages)

    def cancel_active_task(self) -> bool:
        """Cancel the currently active generation task, if any."""
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
            return True
        return False


class SessionManager:
    """In-memory registry of active sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    def get_or_create(
        self, session_id: str, workspace_root: Optional[Path] = None
    ) -> Session:
        """Retrieve an existing session or initialize a new one with optional workspace root."""
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id, workspace_root=workspace_root)
        elif workspace_root is not None:
            self._sessions[session_id].set_workspace_root(workspace_root)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by identifier if it exists."""
        return self._sessions.get(session_id)

    def cancel(self, session_id: str) -> bool:
        """Cancel active generation for a given session."""
        session = self._sessions.get(session_id)
        if session:
            return session.cancel_active_task()
        return False

    def clear(self) -> None:
        """Clear all in-memory sessions (for test isolation)."""
        for session in self._sessions.values():
            session.cancel_active_task()
        self._sessions.clear()
