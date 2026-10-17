"""CLI entry point for the agent-core eval harness.

Usage:
    python -m evals.run --mode scripted
    python -m evals.run --mode live --model local-coder
    python -m evals.run --mode live --model local-fast
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

from agent_core.config import ModelConfig, load_models_config
from evals.runner import run_single_task
from evals.tasks import EVAL_TASKS, EvalTask, get_task_by_id


def resolve_eval_store_dir() -> Path:
    """Resolve directory to store eval results outside the workspace."""
    base_dir = Path(r"D:\agent-store\evals")
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    except Exception:
        # Fallback to local agent-store if D: drive is not writable
        fallback = Path(__file__).resolve().parent.parent / "agent-store" / "evals"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def write_eval_reports(
    run_meta: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write results.json and summary.md to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    md_path = output_dir / "summary.md"

    full_payload = {
        "metadata": run_meta,
        "results": results,
    }
    json_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")

    # Generate Markdown Summary
    passed_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    pass_rate = (passed_count / total_count * 100.0) if total_count > 0 else 0.0

    lines = [
        f"# Eval Run Summary: {run_meta['mode'].upper()} mode",
        "",
        f"- **Timestamp:** {run_meta['timestamp']}",
        f"- **Mode:** `{run_meta['mode']}`",
        f"- **Model:** `{run_meta.get('model_alias', 'N/A')}` ({run_meta.get('model_name', 'scripted')})",
        f"- **Pass Rate:** **{passed_count}/{total_count} ({pass_rate:.1f}%)**",
        f"- **Total Wall Time:** {run_meta['total_wall_time']:.2f}s",
        "",
        "## Task Results",
        "",
        "| Task ID | Name | Status | Rounds | Tools Called | Errors | Nudges | Tokens (P/C) | Max Ctx % | Time |",
        "|---------|------|--------|--------|--------------|--------|--------|--------------|-----------|------|",
    ]

    for r in results:
        status_icon = "PASS" if r["success"] else "FAIL"
        tools_str = ", ".join(r["tool_calls"]) if r["tool_calls"] else "none"
        tokens_str = f"{r['prompt_tokens']} / {r['completion_tokens']}"
        lines.append(
            f"| `{r['task_id']}` | {r['name']} | **{status_icon}** | {r['rounds']} | `{tools_str}` | {r['tool_errors']} | {r['nudges_fired']} | {tokens_str} | {r['max_context_percent']}% | {r['wall_time']}s |"
        )

    lines.append("")
    lines.append("## Details & Diagnostics")
    lines.append("")
    for r in results:
        lines.append(f"### {r['name']} (`{r['task_id']}`)")
        lines.append(f"- **Result:** {'PASSED' if r['success'] else 'FAILED'}")
        lines.append(f"- **Reason:** {r['reason']}")
        if r["unintended_file_changes"]:
            lines.append(f"- **Unintended Changes:** {r['unintended_file_changes']}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


async def main_async(args: argparse.Namespace) -> int:
    mode = args.mode
    model_alias = args.model
    logger = logging.getLogger("eval_runner")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load model configuration if in live mode
    model_config: Optional[ModelConfig] = None
    if mode == "live":
        models_yaml = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
        all_models = load_models_config(models_yaml)
        if model_alias not in all_models:
            print(f"Error: Model alias '{model_alias}' not found in {models_yaml}. Available: {list(all_models.keys())}", file=sys.stderr)
            return 1
        model_config = all_models[model_alias]

    tasks_to_run = EVAL_TASKS
    if args.tasks:
        selected_ids = [t.strip() for t in args.tasks.split(",")]
        tasks_to_run = [t for t in EVAL_TASKS if any(t.task_id.startswith(sid) for sid in selected_ids)]
        if not tasks_to_run:
            print(f"Error: No matching tasks found for filter '{args.tasks}'.", file=sys.stderr)
            return 1

    print(f"\n=======================================================")
    print(f"Running Eval Suite v0 ({len(tasks_to_run)} tasks)")
    print(f"Mode:  {mode.upper()}")
    if mode == "live":
        print(f"Model: {model_alias} ({model_config.model}) [num_ctx={model_config.num_ctx}, budget={model_config.usable_budget}]")
    print(f"=======================================================\n")

    t_all_start = datetime.now()
    results: List[Dict[str, Any]] = []

    for idx, task in enumerate(tasks_to_run, 1):
        print(f"[{idx}/{len(tasks_to_run)}] Running {task.task_id}: '{task.name}'...", end=" ", flush=True)
        record = await run_single_task(task, mode=mode, model_config=model_config, logger=logger)
        results.append(record)
        status_text = "PASS" if record["success"] else "FAIL"
        print(f"[{status_text}] ({record['wall_time']}s)")
        if not record["success"]:
            print(f"      Reason: {record['reason']}")

    t_all_end = datetime.now()
    total_time = (t_all_end - t_all_start).total_seconds()

    timestamp_str = t_all_start.strftime("%Y%m%d_%H%M%S")
    out_base = resolve_eval_store_dir()
    output_dir = out_base / timestamp_str

    run_meta = {
        "timestamp": timestamp_str,
        "mode": mode,
        "model_alias": model_alias if mode == "live" else "scripted",
        "model_name": model_config.model if model_config else "scripted-fake-llm",
        "task_count": len(tasks_to_run),
        "total_wall_time": total_time,
    }

    json_path, md_path = write_eval_reports(run_meta, results, output_dir)

    passed_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    pass_rate = (passed_count / total_count * 100.0) if total_count > 0 else 0.0

    print(f"\n=======================================================")
    print(f"EVAL RUN COMPLETED: {passed_count}/{total_count} PASSED ({pass_rate:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Results JSON:     {json_path}")
    print(f"Summary Markdown: {md_path}")
    print(f"=======================================================\n")

    return 0 if (mode == "scripted" and passed_count == total_count) else 0


def main():
    parser = argparse.ArgumentParser(description="Agent Core Eval Suite v0 Runner")
    parser.add_argument(
        "--mode",
        choices=["scripted", "live"],
        default="scripted",
        help="Run mode: 'scripted' for deterministic CI, 'live' for Ollama model evaluation.",
    )
    parser.add_argument(
        "--model",
        default="local-coder",
        help="Model alias from models.yaml to use in live mode (e.g. 'local-coder', 'local-fast').",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated list of task IDs to run (defaults to all 8 tasks).",
    )

    args = parser.parse_args()
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
