"""Explicit UTF-8 handling for Lila's interactive standard streams."""

from __future__ import annotations

import sys
from typing import TextIO


def configure_standard_streams(
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> None:
    """Configure real text streams to preserve Unicode without transliteration.

    Test doubles and embedded streams without ``reconfigure`` are left alone.
    A terminal/font still needs glyph support to visually render every emoji.
    """
    for stream in (stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")
