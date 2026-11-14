"""Lila's optional foreground daemon-style runtime."""

__all__ = ["LilaDaemon"]


def __getattr__(name):
    if name == "LilaDaemon":
        from .service import LilaDaemon

        return LilaDaemon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
