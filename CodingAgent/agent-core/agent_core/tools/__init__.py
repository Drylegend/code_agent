"""Tools and security subsystem for agent-core."""

from .base import ToolContext, ToolResult, ToolSpec
from .registry import ToolRegistry, create_default_registry
from .gateway import ToolGateway, AuthorizationResult, GatewayInvocationResult
from .path_guard import PathGuard, PathGuardError
from .audit import AuditLogger
from .list_directory import ListDirectoryTool, ListDirectoryInput
from .read_file import ReadFileTool, ReadFileInput
from .write_file import WriteFileTool, WriteFileInput
from .apply_patch import ApplyPatchTool, ApplyPatchInput, EditSpec
from .approval import ApprovalManager, ApprovalRequest

__all__ = [
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "ToolRegistry",
    "create_default_registry",
    "ToolGateway",
    "AuthorizationResult",
    "GatewayInvocationResult",
    "PathGuard",
    "PathGuardError",
    "AuditLogger",
    "detect_instruction_injection",
    "ListDirectoryTool",
    "ListDirectoryInput",
    "ReadFileTool",
    "ReadFileInput",
    "WriteFileTool",
    "WriteFileInput",
    "ApplyPatchTool",
    "ApplyPatchInput",
    "EditSpec",
]
