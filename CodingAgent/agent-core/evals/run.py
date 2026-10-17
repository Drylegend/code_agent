"""CLI entry point for the agent-core eval harness.

Usage:
    python -m evals.run --mode scripted
    python -m evals.run --mode live --model local-coder --repeat 3
    python -m evals.run --mode live --model local-fast --repeat 3
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
from typing import Any, Dict, List, Optional, Tuple

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
        fallback = Path(__file__).resolve().parent.parent / "agent-store" / "evals"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def write_eval_reports(
    run_meta: Dict[str, Any],
    all_runs: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write results.json, per-task transcripts, and summary.md to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    md_path = output_dir / "summary.md"

    # Save transcripts separately to avoid bloating results.json
    clean_runs = []
    for r in all_runs:
        run_copy = dict(r)
        transcript = run_copy.pop("transcript", None)
        task_id = run_copy.get("task_id", "unknown")
        rep = run_copy.get("repeat", 1)
        if transcript:
            tr_file = transcripts_dir / f"{task_id}_rep{rep}.json"
            tr_file.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
        clean_runs.append(run_copy)

    full_payload = {
        "metadata": run_meta,
        "runs": clean_runs,
    }
    json_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")

    # Aggregate by task
    repeats = run_meta.get("repeats", 1)
    task_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in clean_runs:
        task_groups.setdefault(r["task_id"], []).append(r)

    total_runs = len(clean_runs)
    total_passes = sum(1 for r in clean_runs if r["success"])
    overall_pass_rate = (total_passes / total_runs * 100.0) if total_runs > 0 else 0.0

    lines = [
        f"# Eval Run Summary: {run_meta['mode'].upper()} mode",
        "",
        f"- **Timestamp:** {run_meta['timestamp']}",
        f"- **Mode:** `{run_meta['mode']}`",
        f"- **Model:** `{run_meta.get('model_alias', 'N/A')}` ({run_meta.get('model_name', 'scripted')})",
        f"- **Repeats:** {repeats}",
        f"- **Sampling Params:** `temperature={run_meta.get('temperature')}`, `seed={run_meta.get('seed')}`",
        f"- **Overall Pass Rate:** **{total_passes}/{total_runs} ({overall_pass_rate:.1f}%)**",
        f"- **Total Wall Time:** {run_meta['total_wall_time']:.2f}s",
        "",
        "## Per-Task Aggregate Results",
        "",
        "| Task ID | Name | Pass Rate | Success / Total | Avg Rounds | Avg Time | Max Ctx % |",
        "|---------|------|-----------|-----------------|------------|----------|-----------|",
    ]

    for task_id, runs in task_groups.items():
        name = runs[0]["name"]
        passes = sum(1 for r in runs if r["success"])
        t_count = len(runs)
        rate = (passes / t_count * 100.0)
        avg_rounds = sum(r["rounds"] for r in runs) / t_count
        avg_time = sum(r["wall_time"] for r in runs) / t_count
        max_ctx = max(r.get("max_context_percent", 0.0) for r in runs)
        status_bold = f"**{passes}/{t_count} ({rate:.1f}%)**"
        lines.append(
            f"| `{task_id}` | {name} | {status_bold} | {passes} / {t_count} | {avg_rounds:.1f} | {avg_time:.2f}s | {max_ctx:.1f}% |"
        )

    lines.append("")
    lines.append("## Details & Diagnostics (Latest Runs)")
    lines.append("")
    for task_id, runs in task_groups.items():
        latest = runs[-1]
        passes = sum(1 for r in runs if r["success"])
        lines.append(f"### {latest['name']} (`{task_id}`)")
        lines.append(f"- **Pass Rate:** {passes}/{len(runs)} ({passes/len(runs)*100:.1f}%)")
        lines.append(f"- **Latest Reason:** {latest['reason']}")
        if latest["unintended_file_changes"]:
            lines.append(f"- **Unintended Changes:** {latest['unintended_file_changes']}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


async def main_async(args: argparse.Namespace) -> int:
    mode = args.mode
    model_alias = args.model
    repeats = args.repeat
    temperature = args.temperature
    seed = args.seed
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
        if hasattr(model_config, "temperature") and temperature is not None:
            model_config.temperature = temperature
        if hasattr(model_config, "seed") and seed is not None:
            model_config.seed = seed

    tasks_to_run = EVAL_TASKS
    if args.tasks:
        selected_ids = [t.strip() for t in args.tasks.split(",")]
        tasks_to_run = [t for t in EVAL_TASKS if any(t.task_id.startswith(sid) for sid in selected_ids)]
        if not tasks_to_run:
            print(f"Error: No matching tasks found for filter '{args.tasks}'.", file=sys.stderr)
            return 1

    print(f"\n=======================================================")
    print(f"Running Eval Suite ({len(tasks_to_run)} tasks, {repeats} repeat{'s' if repeats > 1 else ''})")
    print(f"Mode:        {mode.upper()}")
    if mode == "live":
        print(f"Model:       {model_alias} ({model_config.model}) [num_ctx={model_config.num_ctx}, budget={model_config.usable_budget}]")
        print(f"Sampling:    temperature={temperature}, seed={seed}")
    print(f"=======================================================\n")

    t_all_start = datetime.now()
    all_runs: List[Dict[str, Any]] = []

    for rep in range(1, repeats + 1):
        if repeats > 1:
            print(f"\n--- Repeat {rep}/{repeats} ---")
        for idx, task in enumerate(tasks_to_run, 1):
            prefix = f"[{idx}/{len(tasks_to_run)}]" if repeats == 1 else f"[Rep {rep} | {idx}/{len(tasks_to_run)}]"
            print(f"{prefix} Running {task.task_id}: '{task.name}'...", end=" ", flush=True)
            record = await run_single_task(task, mode=mode, model_config=model_config, logger=logger)
            record["repeat"] = rep
            all_runs.append(record)
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
        "repeats": repeats,
        "temperature": temperature,
        "seed": seed,
        "total_wall_time": total_time,
    }

    json_path, md_path = write_eval_reports(run_meta, all_runs, output_dir)

    passed_count = sum(1 for r in all_runs if r["success"])
    total_count = len(all_runs)
    pass_rate = (passed_count / total_count * 100.0) if total_count > 0 else 0.0

    print(f"\n=======================================================")
    print(f"EVAL RUN COMPLETED: {passed_count}/{total_count} PASSED ({pass_rate:.1f}%)")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Results JSON:     {json_path}")
    print(f"Summary Markdown: {md_path}")
    print(f"Transcripts:      {output_dir / 'transcripts'}")
    print(f"=======================================================\n")

    return 0 if (mode == "scripted" and passed_count == total_count) else 0


def main():
    parser = argparse.ArgumentParser(description="Agent Core Eval Suite Runner")
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
        help="Comma-separated list of task IDs to run (defaults to all tasks).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of times to repeat each task (default: 3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LLM in live mode (default: 0.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model sampling (default: 42).",
    )

    args = parser.parse_args()
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
