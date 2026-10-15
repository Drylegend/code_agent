"""Base tool abstractions and specifications for agent-core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ConfigDict, Field


class ToolContext(BaseModel):
    """Execution context provided to tools during invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(..., description="Unique conversational session identifier")
    workspace_root: Path = Field(..., description="Canonical resolved workspace root directory")
    audit_logger: Optional[Any] = Field(default=None, description="Optional AuditLogger instance")


class ToolResult(BaseModel):
    """Standardized result returned by tool execution."""

    success: bool = Field(..., description="Whether the tool execution succeeded")
    output: Any = Field(default=None, description="Structured or text output of the tool")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")


class ToolSpec(ABC):
    """Abstract base specification for any agent-core tool.

    Each concrete tool must define:
    - name: Unique identifier for the tool (e.g. 'read_file')
    - description: Human and LLM readable explanation of tool capabilities
    - input_model: Pydantic model class defining and validating input arguments
    - read_only: Boolean flag denoting whether the tool is strictly read-only
    - execute: Async execution method taking validated arguments and context
    """

    name: str
    description: str
    input_model: Type[BaseModel]
    read_only: bool = True

    @abstractmethod
    async def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """Execute the tool with validated arguments and context.

        Args:
            args: Validated instance of self.input_model.
            context: ToolContext containing workspace_root and session_id.

        Returns:
            ToolResult indicating success/failure and output/error.
        """
        raise NotImplementedError
