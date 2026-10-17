# Agent-Core Evaluation Harness v0

Automated evaluation harness for inspecting and benchmarking workspace tool execution, approval flow, and context usage across deterministic CI and live local models.

## Usage

### 1. Deterministic Scripted Mode (CI)
Executes all 8 tasks using the real server tool loop, ToolGateway, PathGuard, SnapshotManager, and ApprovalManager, with deterministic scripted fake LLM responses:

```bash
python -m evals.run --mode scripted
```

### 2. Live Model Mode
Evaluates live local models running in Ollama. Live results are recorded honestly for analysis:

```bash
# Evaluate local-coder (qwen2.5-coder:7b)
python -m evals.run --mode live --model local-coder

# Evaluate local-fast (qwen2.5:3b)
python -m evals.run --mode live --model local-fast
```

### 3. Filter Specific Tasks
```bash
python -m evals.run --mode scripted --tasks task_1,task_2
```

## Evaluated Tasks (8 Core Tasks)

1. `task_1_create_file`: Create a file with exact contents (`greeting.txt`).
2. `task_2_apply_patch`: Target and change one configuration flag via `apply_patch` (`settings.py`).
3. `task_3_grep_report`: Locate a hidden symbol using `grep` and report the filename without workspace edits.
4. `task_4_glob_edit`: Locate markdown files using `glob` and update one target file (`docs/guide.md`).
5. `task_5_multistep`: Multi-step `read_file` -> inspect bug -> `apply_patch` fix -> verify.
6. `task_6_l2_reject_retry`: Reject-then-retry under L2 autonomy (first attempt rejected with guidance, second approved).
7. `task_7_repeated_line`: Edit a repeated line in `inventory.txt` without using blind `replace_all`.
8. `task_8_refusal`: Refuse an unsupported destructive shell command without making changes.

## Metrics Recorded

For each task run, the harness records:
- **Task Success:** Verification assertions on workspace file content and structure.
- **Rounds:** Total loop iterations executed.
- **Tools Called & Tool Errors:** Sequence of tools invoked and count of tool failures.
- **Nudges Fired:** Unexecuted tool announcements or recovery nudges triggered.
- **Ground Truth Token Counts:** Real prompt and completion tokens (from Ollama in live mode).
- **Max Context Usage %:** Peak prompt token utilization against configured usable budget.
- **Wall Time:** Execution duration in seconds.
- **Unintended File Changes:** Detection of any extraneous or unapproved file mutations.

## Output Artifacts

All run outputs are written outside the workspace to:
- `D:\agent-store\evals\{timestamp}\results.json` (machine-readable metrics)
- `D:\agent-store\evals\{timestamp}\summary.md` (human-readable markdown summary)
