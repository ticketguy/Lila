"""Narrow private-file adapter used by vTalks.

The existing filesystem policy remains shared with Lila while other consumers
(OIA and Core logging) still depend on it. Keeping this adapter isolates that
legacy dependency from the voice session contract.
"""
from __future__ import annotations

from src.core.private_storage import (
    open_private_text,
    restrictive_umask,
    secure_private_directory,
    secure_private_file,
)

__all__ = [
    "open_private_text", "restrictive_umask",
    "secure_private_directory", "secure_private_file",
]
