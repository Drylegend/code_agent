"""Security-enforcing ToolGateway choke point for agent-core."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from .audit import AuditLogger
from .base import ToolContext, ToolResult, ToolSpec
from .path_guard import PathGuard, PathGuardError
from .registry import ToolRegistry

logger = logging.getLogger("agent_core.gateway")

# Maximum size for diff preview text sent to the UI (4 KB).
MAX_PREVIEW_SIZE = 4096


def _cap_preview(text: str) -> str:
    """Cap preview text at MAX_PREVIEW_SIZE with truncation marker."""
    if len(text) > MAX_PREVIEW_SIZE:
        return text[:MAX_PREVIEW_SIZE] + "\n... (preview truncated at 4 KB)"
    return text


def _read_file_safe(path: Path) -> Optional[str]:
    """Read a file's text content, returning None on any error."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None


def _compute_write_file_preview(
    target_path_str: str,
    new_content: str,
    workspace_root: Path,
) -> str:
    """Compute a preview for write_file approval requests.

    - For an existing file: unified diff of old vs new content.
    - For a new file: byte size + first N lines.
    """
    resolved = workspace_root / target_path_str
    if resolved.exists() and resolved.is_file():
        old_content = _read_file_safe(resolved)
        if old_content is not None:
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{target_path_str}",
                tofile=f"b/{target_path_str}",
                lineterm="",
            )
            diff_text = "\n".join(diff)
            if not diff_text:
                return "(no changes)"
            return _cap_preview(diff_text)
        else:
            return _cap_preview(f"(existing file unreadable; overwriting with {len(new_content.encode('utf-8'))} bytes)")
    else:
        # New file
        byte_size = len(new_content.encode("utf-8"))
        lines = new_content.splitlines()
        max_preview_lines = 20
        if len(lines) <= max_preview_lines:
            first_lines = "\n".join(lines)
        else:
            first_lines = "\n".join(lines[:max_preview_lines]) + f"\n... ({len(lines) - max_preview_lines} more lines)"
        preview = f"New file: {byte_size} bytes, {len(lines)} lines\n{first_lines}"
        return _cap_preview(preview)


def _compute_apply_patch_diff(
    target_path_str: str,
    args_data: Dict[str, Any],
    workspace_root: Path,
) -> str:
    """Compute a real unified diff for apply_patch approval requests.

    Simulates the edits against the current file content to produce a true diff.
    Falls back to JSON dump of edits if the file can't be read or simulation fails.
    """
    resolved = workspace_root / target_path_str
    if not resolved.exists() or not resolved.is_file():
        edits = args_data.get("edits", [])
        return _cap_preview(json.dumps(edits, indent=2))

    old_content = _read_file_safe(resolved)
    if old_content is None:
        edits = args_data.get("edits", [])
        return _cap_preview(json.dumps(edits, indent=2))

    # Simulate edits to produce the new content
    new_content = old_content
    edits: List[Dict[str, Any]] = args_data.get("edits", [])
    replace_all = bool(args_data.get("replace_all", False))

    try:
        for edit in edits:
            old_text = edit.get("old_text", "")
            new_text = edit.get("new_text", "")
            if not old_text:
                continue
            if replace_all:
                new_content = new_content.replace(old_text, new_text)
            else:
                # Replace first occurrence only (default)
                new_content = new_content.replace(old_text, new_text, 1)
    except Exception:
        edits_list = args_data.get("edits", [])
        return _cap_preview(json.dumps(edits_list, indent=2))

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{target_path_str}",
        tofile=f"b/{target_path_str}",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    if not diff_text:
        return "(no changes)"
    return _cap_preview(diff_text)


class AuthorizationResult(BaseModel):
    """Result of tool authorization evaluation."""

    authorized: bool = Field(..., description="Whether invocation is authorized")
    reason: Optional[str] = Field(default=None, description="Reason if invocation is denied")


class GatewayInvocationResult(BaseModel):
    """Structured response returned by ToolGateway.invoke."""

    success: bool = Field(..., description="Whether invocation was authorized and succeeded")
    tool_name: str = Field(..., description="Name of the invoked tool")
    output: Any = Field(default=None, description="Output payload produced by the tool")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    authorized: bool = Field(default=True, description="Whether invocation passed authorization")
    denial_reason: Optional[str] = Field(default=None, description="Reason if authorization was denied")


AUTHORIZED_WRITE_TOOLS = {"write_file", "apply_patch"}


class DefaultToolAuthorizer:
    """Default authorization strategy.

    Enforces:
    1. Only approved read-only tools or explicitly authorized write tools (write_file, apply_patch) are authorized.
    2. Any argument representing a filesystem path must pass workspace containment and security checks.
    3. Write tools must pass write-specific protected paths checks (via validate_write_path).
    """

    def __init__(self, path_guard: Optional[PathGuard] = None) -> None:
        self.path_guard = path_guard

    def authorize(
        self,
        session_id: str,
        tool: ToolSpec,
        args: BaseModel,
    ) -> AuthorizationResult:
        # Check authorization of non-read-only tools
        is_authorized_write = not tool.read_only and tool.name in AUTHORIZED_WRITE_TOOLS
        if not tool.read_only and not is_authorized_write:
            return AuthorizationResult(
                authorized=False,
                reason=f"Tool '{tool.name}' is not authorized: write/action tools are currently disabled.",
            )

        # Inspect fields for path parameters and check against PathGuard
        if self.path_guard is not None:
            data = args.model_dump()
            for key, val in data.items():
                if "path" in key.lower() and isinstance(val, (str, Path)):
                    try:
                        if is_authorized_write:
                            self.path_guard.validate_write_path(val)
                        else:
                            self.path_guard.validate_path(val)
                    except PathGuardError as exc:
                        return AuthorizationResult(
                            authorized=False,
                            reason=f"Path authorization denied: {exc}",
                        )

        return AuthorizationResult(authorized=True)


class ToolGateway:
    """The authoritative execution choke point for all agent tools.

    Execution order:
    1. Look up tool in ToolRegistry.
    2. Validate args using tool's Pydantic input model.
    3. Call authorize(session_id, tool, validated_args).
    4. If authorization fails, log audit event and return structured denied result.
    5. If authorized, execute tool.
    6. Record audit event with outcome, duration, and cryptographic metadata.
    7. Return structured GatewayInvocationResult.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        audit_logger: AuditLogger,
        workspace_root: Optional[Union[str, Path]] = None,
        authorizer: Optional[Callable[[str, ToolSpec, BaseModel], AuthorizationResult]] = None,
        path_guard: Optional[PathGuard] = None,
        approval_manager: Optional[Any] = None,
        get_session_autonomy: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.registry = registry
        self.audit_logger = audit_logger
        self.workspace_root = Path(workspace_root).resolve() if workspace_root is not None else None
        self.path_guard = path_guard or (PathGuard(self.workspace_root) if self.workspace_root is not None else None)
        self._authorizer = authorizer
        self.approval_manager = approval_manager
        self.get_session_autonomy = get_session_autonomy

    @property
    def authorizer(self) -> Callable[[str, ToolSpec, BaseModel], AuthorizationResult]:
        if self._authorizer is not None:
            return self._authorizer
        guard = self.path_guard or (PathGuard(self.workspace_root) if self.workspace_root is not None else None)
        return DefaultToolAuthorizer(guard).authorize

    @authorizer.setter
    def authorizer(self, value: Optional[Callable[[str, ToolSpec, BaseModel], AuthorizationResult]]) -> None:
        self._authorizer = value

    async def invoke(
        self,
        session_id: str,
        tool_name: str,
        raw_args: Dict[str, Any],
        workspace_root: Optional[Union[str, Path]] = None,
        autonomy_level: Optional[int] = None,
    ) -> GatewayInvocationResult:
        """Execute a tool through the secure gateway pipeline.

        No code can bypass this choke point to execute tools.
        """
        # Determine effective workspace root
        target_root = workspace_root if workspace_root is not None else self.workspace_root
        if target_root is None:
            self.audit_logger.log(
                session_id=session_id,
                tool=tool_name,
                args=raw_args,
                outcome="denied",
                duration=0.0,
            )
            return GatewayInvocationResult(
                success=False,
                tool_name=tool_name,
                authorized=False,
                denial_reason="No workspace root configured for session",
                error="No workspace root configured for session",
            )

        effective_root = Path(target_root).resolve()
        effective_guard = (
            self.path_guard
            if (self.path_guard is not None and effective_root == self.workspace_root)
            else PathGuard(effective_root)
        )

        # 1. Look up tool in registry
        tool = self.registry.get(tool_name)
        if tool is None:
            self.audit_logger.log(
                session_id=session_id,
                tool=tool_name,
                args=raw_args,
                outcome="denied",
                duration=0.0,
            )
            return GatewayInvocationResult(
                success=False,
                tool_name=tool_name,
                authorized=False,
                denial_reason=f"Unknown tool '{tool_name}'",
                error=f"Tool '{tool_name}' is not registered",
            )

        # 2. Validate arguments using tool's input_model
        try:
            validated_args = tool.input_model.model_validate(raw_args)
        except ValidationError as exc:
            self.audit_logger.log(
                session_id=session_id,
                tool=tool_name,
                args=raw_args,
                outcome="denied",
                duration=0.0,
            )
            return GatewayInvocationResult(
                success=False,
                tool_name=tool_name,
                authorized=False,
                denial_reason="Invalid arguments schema",
                error=f"Argument validation error: {exc.errors(include_url=False)}",
            )

        # 3. Call authorizer
        if self._authorizer is not None:
            auth_result = self._authorizer(session_id, tool, validated_args)
        else:
            auth_result = DefaultToolAuthorizer(effective_guard).authorize(
                session_id, tool, validated_args
            )

        # 4. Handle authorization failure
        if not auth_result.authorized:
            self.audit_logger.log(
                session_id=session_id,
                tool=tool_name,
                args=validated_args.model_dump(),
                outcome="denied",
                duration=0.0,
            )
            return GatewayInvocationResult(
                success=False,
                tool_name=tool_name,
                authorized=False,
                denial_reason=auth_result.reason,
                error=auth_result.reason or "Authorization denied",
            )

        start_time = time.perf_counter()
        raw_args_dict = validated_args.model_dump() if validated_args is not None else (raw_args if isinstance(raw_args, dict) else {})

        try:
            # 4b. Approval Gate Check for Write Tools
            effective_autonomy = (
                autonomy_level
                if autonomy_level is not None
                else (self.get_session_autonomy(session_id) if self.get_session_autonomy else 2)
            )
            if not tool.read_only and tool_name in AUTHORIZED_WRITE_TOOLS:
                # Level 2 "Ask" requires approval for write tools
                if effective_autonomy == 2 and self.approval_manager is not None:
                    args_data = validated_args.model_dump()
                    target_path = str(args_data.get("path", ""))
                    diff_preview: Optional[str] = None
                    byte_size: Optional[int] = None
                    if tool_name == "write_file":
                        content = args_data.get("content", "")
                        byte_size = len(str(content).encode("utf-8"))
                        diff_preview = _compute_write_file_preview(
                            target_path, content, effective_root,
                        )
                    elif tool_name == "apply_patch":
                        diff_preview = _compute_apply_patch_diff(
                            target_path, args_data, effective_root,
                        )

                    approved = await self.approval_manager.request_approval(
                        session_id=session_id,
                        tool_name=tool_name,
                        validated_args=args_data,
                        path=target_path,
                        diff_preview=diff_preview,
                        byte_size=byte_size,
                    )
                    if not approved:
                        self.audit_logger.log(
                            session_id=session_id,
                            tool=tool_name,
                            args=args_data,
                            outcome="denied",
                            duration=0.0,
                        )
                        return GatewayInvocationResult(
                            success=False,
                            tool_name=tool_name,
                            authorized=False,
                            denial_reason="Tool execution rejected by user",
                            error="Tool execution rejected by user",
                        )

            # 5. Execute authorized tool
            context = ToolContext(
                session_id=session_id,
                workspace_root=effective_root,
                audit_logger=self.audit_logger,
            )

            # Shield tool execution so destructive writes, replace, verify,
            # and commit sequences cannot be cut off midway by external cancellation (e.g. Stop button).
            exec_task = asyncio.create_task(tool.execute(validated_args, context))
            was_cancelled = False
            try:
                result = await asyncio.shield(exec_task)
            except asyncio.CancelledError:
                was_cancelled = True
                # Defer cancellation: wait for the shielded tool to complete its write/commit cleanly
                result = await exec_task

            duration = time.perf_counter() - start_time
            outcome = "success" if result.success else "error"

            # 6. Extract audit metadata if provided
            extra_audit: Optional[Dict[str, Any]] = None
            if isinstance(result.output, dict):
                audit_meta = result.output.get("audit_metadata")
                if isinstance(audit_meta, dict):
                    extra_audit = audit_meta

            # Record audit event
            self.audit_logger.log(
                session_id=session_id,
                tool=tool_name,
                args=validated_args.model_dump(),
                outcome=outcome,
                duration=duration,
                extra=extra_audit,
            )

            if was_cancelled:
                raise asyncio.CancelledError()

            # 7. Return structured result
            return GatewayInvocationResult(
                success=result.success,
                tool_name=tool_name,
                output=result.output,
                error=result.error,
                authorized=True,
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            duration = time.perf_counter() - start_time
            tb_str = traceback.format_exc()
            logger.error(
                "Internal tool execution error in '%s' for session '%s': %s\n%s",
                tool_name, session_id, exc, tb_str,
            )

            self.audit_logger.log(
                session_id=session_id,
                tool=tool_name,
                args=raw_args_dict,
                outcome="error",
                duration=duration,
                extra={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )

            # User sees a clear error in the UI; model gets a short generic one without internals
            return GatewayInvocationResult(
                success=False,
                tool_name=tool_name,
                authorized=True,
                error=f"Internal error while executing '{tool_name}'. The error has been logged.",
            )
