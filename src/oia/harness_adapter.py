"""Deprecated compatibility import for the relocated Lila integration.

Harness composition now lives in :mod:`src.daemon.oia_harness`; reusable OIA
contracts and services do not import this module.
"""


def build_approval_service(owner_authentication, tool_registry):
    """Forward the historical helper name to the daemon integration module."""
    from src.daemon.oia_harness import build_approval_service as build

    return build(owner_authentication, tool_registry)
