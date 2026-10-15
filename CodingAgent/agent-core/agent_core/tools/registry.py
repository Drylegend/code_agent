"""Central tool registry for agent-core."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import ToolSpec


class ToolRegistry:
    """Registry maintaining available ToolSpec instances and generating OpenAI schemas."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        """Register a new tool specification.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolSpec]:
        """Retrieve a registered tool by its exact name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolSpec]:
        """List all currently registered tool specifications."""
        return list(self._tools.values())

    def get_openai_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generate OpenAI-compatible tool definitions from registered tools.

        Formats schemas according to OpenAI Function Calling standard using
        Pydantic v2 `model_json_schema()`.
        """
        definitions: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            schema = tool.input_model.model_json_schema()

            # Clean schema top-level metadata if present
            schema.pop("title", None)

            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                },
            })
        return definitions

    def clear(self) -> None:
        """Clear all registered tools (primarily used for test isolation)."""
        self._tools.clear()


def create_default_registry(
    include_write_tools: bool = False,
    snapshot_manager: Optional[Any] = None,
    audit_logger: Optional[Any] = None,
) -> ToolRegistry:
    """Create and populate the default production tool registry.

    Registers:
    - list_directory
    - read_file
    And if include_write_tools is True (production mode):
    - write_file
    - apply_patch
    """
    from .list_directory import ListDirectoryTool
    from .read_file import ReadFileTool
    from .glob import GlobTool
    from .grep import GrepTool

    registry = ToolRegistry()
    registry.register(ListDirectoryTool())
    registry.register(ReadFileTool())
    registry.register(GlobTool())
    registry.register(GrepTool())

    if include_write_tools:
        from .write_file import WriteFileTool
        from .apply_patch import ApplyPatchTool

        registry.register(WriteFileTool(snapshot_manager=snapshot_manager, audit_logger=audit_logger))
        registry.register(ApplyPatchTool(snapshot_manager=snapshot_manager, audit_logger=audit_logger))

    return registry
