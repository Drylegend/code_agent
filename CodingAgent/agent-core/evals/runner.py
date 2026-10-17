"""Task runner and metrics collector for eval harness v0."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from agent_core.config import ModelConfig, load_models_config
from agent_core.server import ConnectionWrapper, run_chat_generation
from agent_core.sessions import Session
from agent_core.tools import (
    ApprovalManager,
    AuditLogger,
    PathGuard,
    ToolGateway,
    create_default_registry,
)
from evals.scripted_llm import get_scripted_llm_for_task
from evals.tasks import EvalTask, compute_workspace_hashes


class HarnessConnection:
    """Captures outbound messages from server and supports typed querying."""

    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send(self, data: str) -> None:
        try:
            self.sent.append(json.loads(data))
        except Exception:
            pass

    def get_messages_of_type(self, msg_type: str) -> List[Dict[str, Any]]:
        return [m for m in self.sent if m.get("type") == msg_type]


async def run_single_task(
    task: EvalTask,
    mode: str = "scripted",
    model_config: Optional[ModelConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Execute a single task against disposable isolated workspace and return metric record."""
    run_logger = logger or logging.getLogger(f"eval.{task.task_id}")

    with tempfile.TemporaryDirectory() as td:
        ws_root = Path(td).resolve()

        # 1. Populate workspace fixture files
        for rel_path, content in task.setup_files.items():
            target_path = ws_root / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")

        initial_hashes = compute_workspace_hashes(ws_root)

        # 2. Setup gateway and isolated audit pipeline
        audit_path = ws_root / "store" / "audit.jsonl"
        audit_logger = AuditLogger(audit_path)
        registry = create_default_registry(include_write_tools=True)
        path_guard = PathGuard(ws_root)
        approval_manager = ApprovalManager(audit_logger=audit_logger)

        current_autonomy = task.initial_autonomy

        gateway = ToolGateway(
            registry=registry,
            audit_logger=audit_logger,
            workspace_root=ws_root,
            path_guard=path_guard,
            approval_manager=approval_manager,
            get_session_autonomy=lambda s: current_autonomy,
        )

        session = Session(session_id=f"eval-{task.task_id}", workspace_root=str(ws_root))
        session.set_autonomy_level(current_autonomy)
        session.add_user_message(task.prompt)

        mock_ws = HarnessConnection()
        conn_wrapper = ConnectionWrapper(mock_ws)

        # 3. Approval handler background worker
        # Intercepts approval requests and auto-responds via ApprovalManager (never bypassing gateway)
        approval_attempts: Dict[str, int] = {}
        stop_approvals = asyncio.Event()

        approval_history: List[Dict[str, Any]] = []

        async def approval_worker():
            processed_requests = set()
            while not stop_approvals.is_set():
                await asyncio.sleep(0.02)
                requests = mock_ws.get_messages_of_type("approval_request")
                for req in requests:
                    req_id = req.get("request_id")
                    if req_id and req_id not in processed_requests:
                        processed_requests.add(req_id)
                        tool_name = req.get("tool", "")
                        attempt_num = approval_attempts.get(tool_name, 0) + 1
                        approval_attempts[tool_name] = attempt_num

                        if task.approval_policy:
                            decision, reason = task.approval_policy(tool_name, req, attempt_num)
                        else:
                            decision, reason = "approve", None

                        approval_history.append({
                            "request_id": req_id,
                            "tool": tool_name,
                            "args": req.get("args"),
                            "attempt": attempt_num,
                            "decision": decision,
                            "reason": reason,
                        })

                        run_logger.info(
                            "Auto-responding to approval request %s for %s: %s (attempt %d)",
                            req_id, tool_name, decision, attempt_num
                        )
                        await approval_manager.submit_response(
                            request_id=req_id,
                            decision=decision,
                            session_id=session.session_id,
                        )

        approval_task = asyncio.create_task(approval_worker())

        # 4. Resolve Model Configuration
        if model_config is None:
            model_config = ModelConfig(
                alias="eval-scripted",
                model="ollama/qwen2.5:3b",
                context_window=32768,
                num_ctx=8192,
                usable_budget=6144,
            )

        t_start = time.time()
        error_msg: Optional[str] = None

        try:
            if mode == "scripted":
                scripted_llm = get_scripted_llm_for_task(task.task_id)
                with patch("agent_core.server.acompletion_with_tools", side_effect=scripted_llm):
                    await run_chat_generation(
                        session=session,
                        model_config=model_config,
                        request_id=f"req-{task.task_id}",
                        conn_wrapper=conn_wrapper,
                        logger=run_logger,
                        tool_gateway=gateway,
                    )
                    if getattr(task, "followup_prompt", None):
                        session.add_user_message(task.followup_prompt)
                        await run_chat_generation(
                            session=session,
                            model_config=model_config,
                            request_id=f"req-{task.task_id}-followup",
                            conn_wrapper=conn_wrapper,
                            logger=run_logger,
                            tool_gateway=gateway,
                        )
            else:
                # Live mode against actual Ollama model
                await run_chat_generation(
                    session=session,
                    model_config=model_config,
                    request_id=f"req-{task.task_id}",
                    conn_wrapper=conn_wrapper,
                    logger=run_logger,
                    tool_gateway=gateway,
                )
                if getattr(task, "followup_prompt", None):
                    session.add_user_message(task.followup_prompt)
                    await run_chat_generation(
                        session=session,
                        model_config=model_config,
                        request_id=f"req-{task.task_id}-followup",
                        conn_wrapper=conn_wrapper,
                        logger=run_logger,
                        tool_gateway=gateway,
                    )
        except Exception as exc:
            run_logger.exception("Error executing task %s: %s", task.task_id, exc)
            error_msg = str(exc)
        finally:
            stop_approvals.set()
            await approval_task

        t_end = time.time()
        wall_time = round(t_end - t_start, 2)

        # 5. Extract metrics from messages
        tool_call_msgs = mock_ws.get_messages_of_type("tool.call")
        tool_result_msgs = mock_ws.get_messages_of_type("tool.result")
        done_msgs = mock_ws.get_messages_of_type("chat.done")
        status_msgs = mock_ws.get_messages_of_type("context.status")

        tool_calls = [m.get("tool", "") for m in tool_call_msgs]
        tool_errors = sum(1 for m in tool_result_msgs if not m.get("ok", True))

        # Check for nudges in session history
        nudges_fired = 0
        for m in session.get_messages():
            if m.get("role") == "user" and ("did not execute a tool call" in str(m.get("content")) or "tool execution failed" in str(m.get("content"))):
                nudges_fired += 1

        prompt_tokens = 0
        completion_tokens = 0
        max_context_pct = 0.0

        if done_msgs:
            u = done_msgs[0].get("payload", {}).get("usage", {})
            prompt_tokens = u.get("prompt", 0)
            completion_tokens = u.get("completion", 0)
            max_context_pct = u.get("usage_percent", 0.0)

        for sm in status_msgs:
            p = sm.get("payload", {})
            pct = p.get("usage_percent", 0.0)
            if pct > max_context_pct:
                max_context_pct = pct

        final_assistant_text = ""
        for m in reversed(session.get_messages()):
            if m.get("role") == "assistant" and m.get("content"):
                final_assistant_text = str(m.get("content"))
                break

        # 6. Verify Task Assertions
        task_passed = False
        verify_reason = ""
        if error_msg:
            task_passed = False
            verify_reason = f"Execution error: {error_msg}"
        elif task.verify_fn:
            task_passed, verify_reason = task.verify_fn(ws_root, tool_call_msgs, final_assistant_text)
        else:
            task_passed = True
            verify_reason = "No verification function defined"

        # 7. Check for Unintended File Changes
        final_hashes = compute_workspace_hashes(ws_root)
        unintended_changes: List[str] = []

        # Find modified files that should not have changed
        for rel_path, old_hash in initial_hashes.items():
            if rel_path in final_hashes and final_hashes[rel_path] != old_hash:
                # Check if this change was expected
                if task.task_id in ("task_3_grep_report", "task_8_refusal"):
                    unintended_changes.append(f"modified: {rel_path}")
                elif task.task_id == "task_4_glob_edit" and rel_path != "docs/guide.md":
                    unintended_changes.append(f"unexpected edit: {rel_path}")

        # Find created files that should not exist
        for rel_path in final_hashes:
            if rel_path not in initial_hashes:
                if task.task_id == "task_1_create_file" and rel_path != "greeting.txt":
                    unintended_changes.append(f"extra file: {rel_path}")
                elif task.task_id in ("task_2_apply_patch", "task_3_grep_report", "task_5_multistep", "task_6_l2_reject_retry", "task_7_repeated_line", "task_8_refusal"):
                    unintended_changes.append(f"unexpected new file: {rel_path}")

        if unintended_changes:
            task_passed = False
            verify_reason += f" | Unintended file changes: {unintended_changes}"

        return {
            "task_id": task.task_id,
            "name": task.name,
            "success": task_passed,
            "reason": verify_reason,
            "rounds": len(tool_call_msgs) + 1,
            "tool_calls": tool_calls,
            "tool_errors": tool_errors,
            "nudges_fired": nudges_fired,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "max_context_percent": round(max_context_pct, 1),
            "wall_time": wall_time,
            "unintended_file_changes": unintended_changes,
            "transcript": {
                "task_id": task.task_id,
                "task_name": task.name,
                "prompt": task.prompt,
                "followup_prompt": getattr(task, "followup_prompt", None),
                "messages": session.get_messages(),
                "tool_calls": tool_call_msgs,
                "tool_results": tool_result_msgs,
                "approvals": approval_history,
                "nudges_fired": nudges_fired,
                "initial_hashes": initial_hashes,
                "final_hashes": final_hashes,
                "unintended_file_changes": unintended_changes,
                "final_assistant_text": final_assistant_text,
            },
        }
