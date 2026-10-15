"""Untrusted content and prompt injection phrase detector for agent-core tools."""

from __future__ import annotations

import re
from typing import List

# Patterns targeting common prompt-injection / instruction-override phrases
DEFAULT_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"forget\s+previous\s+instructions", re.IGNORECASE),
]


def detect_instruction_injection(text: str) -> bool:
    """Detect whether text contains obvious instruction-injection or override phrases.

    Case-insensitive, non-mutating check.

    Args:
        text: String content to inspect.

    Returns:
        True if suspicious instruction-injection phrases are found, False otherwise.
    """
    if not text:
        return False

    for pattern in DEFAULT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True

    return False
