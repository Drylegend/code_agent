"""Deterministic scripted LLM provider for CI test execution.

Simulates valid LiteLLM responses for the 8 eval tasks, exercising the real:
  Session -> run_chat_generation -> ToolGateway -> ApprovalManager -> tool execution -> audit/snapshots
pipeline deterministically without external model dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class ScriptedChoice:
    def __init__(self, message: Any) -> None:
        self.message = message


class ScriptedMessage:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or None


class ScriptedToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = type("Fn", (), {"name": name, "arguments": arguments})()


class ScriptedResponse:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None, prompt_tokens: int = 100, completion_tokens: int = 20) -> None:
        self.choices = [ScriptedChoice(ScriptedMessage(content=content, tool_calls=tool_calls))]
        self.usage = type("Usage", (), {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })()


def get_scripted_llm_for_task(task_id: str):
    """Return an async callable that yields scripted responses for a specific task."""
    step = 0

    async def fake_llm(**kwargs) -> ScriptedResponse:
        nonlocal step
        step += 1
        messages: List[Dict[str, Any]] = kwargs.get("messages", [])

        if task_id.startswith("task_1"):
            if step == 1:
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t1",
                            "write_file",
                            json.dumps({"path": "greeting.txt", "content": "Hello, World!", "overwrite": True}),
                        )
                    ],
                )
            return ScriptedResponse(content="Created greeting.txt with 'Hello, World!'.")

        elif task_id.startswith("task_2"):
            if step == 1:
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t2",
                            "apply_patch",
                            json.dumps({
                                "path": "settings.py",
                                "edits": [{"old_text": "DEBUG = False", "new_text": "DEBUG = True"}],
                            }),
                        )
                    ],
                )
            return ScriptedResponse(content="Updated settings.py to DEBUG = True.")

        elif task_id.startswith("task_3"):
            if step == 1:
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t3",
                            "grep",
                            json.dumps({"path": "src", "pattern": "SECRET_CREDENTIAL_KEY"}),
                        )
                    ],
                )
            return ScriptedResponse(content="The file containing 'SECRET_CREDENTIAL_KEY' is beta.py.")

        elif task_id.startswith("task_4"):
            if step == 1:
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t4_glob",
                            "glob",
                            json.dumps({"pattern": "docs/**/*.md"}),
                        )
                    ],
                )
            elif step == 2:
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t4_patch",
                            "apply_patch",
                            json.dumps({
                                "path": "docs/guide.md",
                                "edits": [{"old_text": "Follow these steps.", "new_text": "Follow these steps.\n<!-- reviewed -->"}],
                            }),
                        )
                    ],
                )
            return ScriptedResponse(content="Found markdown files with glob and appended '<!-- reviewed -->' to docs/guide.md.")

        elif task_id.startswith("task_5"):
            if step == 1:
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t5_read",
                            "read_file",
                            json.dumps({"path": "calc.py"}),
                        )
                    ],
                )
            elif step == 2:
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t5_patch",
                            "apply_patch",
                            json.dumps({
                                "path": "calc.py",
                                "edits": [{"old_text": "return a + b", "new_text": "return a * b"}],
                            }),
                        )
                    ],
                )
            return ScriptedResponse(content="Inspected calc.py and updated multiply() to return a * b.")

        elif task_id.startswith("task_6b"):
            if step == 1:
                # Turn 1: attempt 1 (rejected)
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t6b_1",
                            "write_file",
                            json.dumps({"path": "deploy.txt", "content": "status: attempt_1\n", "overwrite": True}),
                        )
                    ],
                )
            elif step == 2:
                # Turn 1 completion: acknowledge rejection
                return ScriptedResponse(content="The user rejected the modification to deploy.txt. Please provide an alternative instruction.")
            elif step == 3:
                # Turn 2: user asks for approved_v2
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t6b_2",
                            "write_file",
                            json.dumps({"path": "deploy.txt", "content": "status: approved_v2\n", "overwrite": True}),
                        )
                    ],
                )
            return ScriptedResponse(content="Updated deploy.txt to 'status: approved_v2'.")

        elif task_id.startswith("task_6"):
            if step == 1:
                # First attempt: write attempt_1 (will be rejected by harness approval policy)
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t6_1",
                            "write_file",
                            json.dumps({"path": "deploy.txt", "content": "status: attempt_1\n", "overwrite": True}),
                        )
                    ],
                )
            return ScriptedResponse(content="The user rejected the modification to deploy.txt. I will not proceed. Please let me know if you would like an alternative.")

        elif task_id.startswith("task_7"):
            if step == 1:
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t7_read",
                            "read_file",
                            json.dumps({"path": "inventory.txt"}),
                        )
                    ],
                )
            elif step == 2:
                return ScriptedResponse(
                    content=None,
                    tool_calls=[
                        ScriptedToolCall(
                            "call_t7_patch",
                            "apply_patch",
                            json.dumps({
                                "path": "inventory.txt",
                                "edits": [{"old_text": "price: 20", "new_text": "price: 25"}],
                                "replace_all": False,
                            }),
                        )
                    ],
                )
            return ScriptedResponse(content="Updated the second price to 25 without replace_all.")

        elif task_id.startswith("task_8"):
            # Refusal task: state inability to run shell commands, no tool calls
            return ScriptedResponse(
                content="I cannot execute terminal commands or run `rm -rf /` because I do not have shell or terminal access. I can only inspect and edit files within the workspace."
            )

        return ScriptedResponse(content="Completed task.")

    return fake_llm
