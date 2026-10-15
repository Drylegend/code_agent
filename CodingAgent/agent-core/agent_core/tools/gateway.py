"""Security-enforcing ToolGateway choke point for agent-core."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from .audit import AuditLogger
from .base import ToolContext, ToolResult, ToolSpec
from .path_guard import PathGuard, PathGuardError
from .registry import ToolRegistry


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
                elif tool_name == "apply_patch":
                    diff_preview = args_data.get("patch") or json.dumps(args_data.get("edits", []))

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
        start_time = time.perf_counter()

        try:
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
            clean_err = str(exc).splitlines()[0] if str(exc) else "Internal tool error"

            self.audit_logger.log(
                session_id=session_id,
                tool=tool_name,
                args=validated_args.model_dump(),
                outcome="error",
                duration=duration,
            )

            return GatewayInvocationResult(
                success=False,
                tool_name=tool_name,
                authorized=True,
                error=f"Tool execution failed: {clean_err}",
            )
