"""Central risk classification and one-call tool permissions."""

from pathlib import Path
from typing import Callable, Optional
import re


# Modes: auto, approval, disabled.
POLICIES = {
    # Read-only operational tools
    "file_read": ("low", "auto"),
    "file_list": ("low", "auto"),
    "schedule_list": ("low", "auto"),
    "dns_lookup": ("low", "auto"),
    "ping": ("low", "auto"),
    "wifi_scan": ("low", "auto"),
    "usb_list": ("low", "auto"),
    "self_status": ("low", "auto"),
    "self_log": ("low", "auto"),
    "workstation_overview": ("low", "auto"),
    "workstation_list_categories": ("low", "auto"),
    "workstation_list_workspaces": ("low", "auto"),
    "workstation_get_workspace": ("low", "auto"),
    "workstation_intake_overview": ("low", "auto"),
    "workstation_software_overview": ("low", "auto"),

    # Ordinary changes and external requests
    "file_write": ("moderate", "approval"),
    "file_append": ("moderate", "approval"),
    "schedule_add": ("moderate", "approval"),
    "web_fetch": ("moderate", "approval"),
    "notify_user": ("moderate", "approval"),
    "set_volume": ("moderate", "approval"),
    "clipboard_set": ("moderate", "approval"),

    # Sensitive or potentially destructive operations
    "file_delete": ("high", "approval"),
    "bash": ("high", "approval"),
    "ssh_exec": ("high", "approval"),
    "wifi_connect": ("high", "approval"),
    "http_request": ("high", "approval"),
    "tcp_connect": ("high", "approval"),
    "udp_send": ("high", "approval"),
    "tcp_listen": ("high", "approval"),
    "gpio_write": ("high", "approval"),
    "gpio_read": ("high", "approval"),
    "i2c_transfer": ("high", "approval"),
    "spi_transfer": ("high", "approval"),
    "serial_write": ("high", "approval"),
    "serial_read": ("high", "approval"),
    "power_action": ("high", "approval"),
    "clipboard_get": ("high", "approval"),
    "screen_capture": ("high", "approval"),

    # Not implemented in the active runtime.
    "memory_store": ("restricted", "disabled"),
    "memory_recall": ("restricted", "disabled"),
    "memory_list": ("restricted", "disabled"),
}


def policy_for(name):
    """Unknown tools require approval by default."""
    return POLICIES.get(name, ("high", "approval"))


def redact_arguments(args, tool_name=None):
    """Avoid storing likely credentials and sensitive payloads in logs."""
    result = {}

    always_redact = {
        "bash": {"command"},
        "ssh_exec": {"command", "key"},
        "file_write": {"content"},
        "file_append": {"content"},
        "http_request": {"body"},
        "serial_write": {"data"},
        "clipboard_set": {"content"},
    }.get(tool_name, set())

    sensitive_names = (
        "password", "passwd", "secret", "token",
        "api_key", "private_key", "authorization",
        "cookie", "credential",
    )

    for name, value in args.items():
        lowered = name.lower()

        if lowered in always_redact or any(
            item in lowered for item in sensitive_names
        ):
            result[name] = "[REDACTED]"
        elif isinstance(value, dict):
            result[name] = redact_arguments(value)
        elif isinstance(value, str):
            result[name] = re.sub(
                r"(?i)(password|passwd|token|secret|api[_-]?key)=([^&\s]+)",
                r"\1=[REDACTED]",
                value,
            )
        else:
            result[name] = value

    return result


class PermissionManager:
    """Enforce automatic, approval-required and disabled tool policies."""

    SENSITIVE_PATH_PARTS = {
        ".ssh", ".aws", ".azure", ".gnupg",
        "secrets", "credentials",
    }

    def __init__(
        self,
        approval_callback: Optional[Callable] = None,
        workspace_root: Optional[str] = None,
    ):
        self.approval_callback = approval_callback
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root else None
        )

    def authorize(self, tool, args):
        if tool is None:
            return False, "Unknown tool."

        risk, mode = policy_for(tool.name)

        if mode == "disabled":
            return False, f"'{tool.name}' is disabled in the current runtime."

        requires_approval = mode == "approval"

        # File reads/lists outside Lila's workspace, including symlinks
        # resolving outside it, require explicit approval.
        if tool.name in {"file_read", "file_list"}:
            raw_path = Path(args.get("path", ".")).expanduser()

            if not raw_path.is_absolute():
                raw_path = Path.cwd() / raw_path

            try:
                candidate = raw_path.resolve(strict=False)
            except (OSError, RuntimeError):
                return False, "The requested path could not be resolved safely."
            sensitive = any(
                part.lower() in self.SENSITIVE_PATH_PARTS
                for part in candidate.parts
            )

            if candidate.name.lower().startswith(".env"):
                sensitive = True

            inside_workspace = False
            if self.workspace_root:
                try:
                    candidate.relative_to(self.workspace_root)
                    inside_workspace = True
                except ValueError:
                    pass

            if sensitive or (
                self.workspace_root is None or not inside_workspace
            ):
                requires_approval = True
                risk = "high"

        if not requires_approval:
            return True, "Automatically allowed by policy."

        if self.approval_callback is None:
            return False, (
                f"'{tool.name}' requires {risk}-risk approval, "
                "but no approval handler is configured."
            )

        try:
            approved = self.approval_callback(tool, args, risk)
        except Exception as exc:
            return False, f"Approval failed safely: {exc}"

        if approved is True:
            return True, "Approved for this exact call."

        return False, f"Permission not granted for '{tool.name}'."
