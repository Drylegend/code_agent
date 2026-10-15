"""Audit log verification utility for checking hash chain integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from .audit import GENESIS_HASH, compute_record_hash


def verify_audit_log(log_path: Path | str) -> Tuple[bool, List[str]]:
    """Verify the integrity of a JSONL audit log file.

    Checks:
    1. Valid JSON on every non-empty line.
    2. Expected schema presence on every record.
    3. Proper link from previous_hash to preceding record hash (or GENESIS_HASH).
    4. Exact match of computed SHA256 against recorded hash.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    path = Path(log_path).resolve()
    if not path.exists():
        return False, [f"Audit file not found: {path}"]

    errors: List[str] = []
    expected_prev = GENESIS_HASH
    line_number = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line_number += 1
            line = raw_line.strip()
            if not line:
                continue

            try:
                record: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: Invalid JSON: {exc.msg}")
                continue

            for required_field in (
                "timestamp",
                "session_id",
                "tool",
                "args",
                "outcome",
                "duration",
                "previous_hash",
                "hash",
            ):
                if required_field not in record:
                    errors.append(
                        f"Line {line_number}: Missing required field '{required_field}'"
                    )

            if errors and not errors[-1].startswith(f"Line {line_number}"):
                continue

            prev_hash = record.get("previous_hash")
            rec_hash = record.get("hash")

            # Check chain link
            if prev_hash != expected_prev:
                errors.append(
                    f"Line {line_number}: Broken chain! previous_hash '{prev_hash}' does not match expected '{expected_prev}'"
                )

            # Recompute hash without 'hash' field
            payload = {k: v for k, v in record.items() if k != "hash"}
            computed = compute_record_hash(payload)

            if computed != rec_hash:
                errors.append(
                    f"Line {line_number}: Tamper detected! Computed hash '{computed}' does not match recorded hash '{rec_hash}'"
                )

            expected_prev = rec_hash

    is_valid = len(errors) == 0
    return is_valid, errors


def main() -> None:
    """CLI entry point for audit verification."""
    parser = argparse.ArgumentParser(description="Verify agent-core audit log integrity.")
    parser.add_argument("log_path", type=Path, help="Path to audit.jsonl file")
    args = parser.parse_args()

    valid, errors = verify_audit_log(args.log_path)
    if valid:
        print(f"SUCCESS: Audit log at '{args.log_path}' is valid and untampered.")
        sys.exit(0)
    else:
        print(f"FAILURE: Found {len(errors)} error(s) in audit log at '{args.log_path}':")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
