"""Central risk classification and one-call tool permissions."""

from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping
from typing import Callable, Optional
import re
import fnmatch
import os


class PolicyDisposition(str, Enum):
    """Trusted policy condition, separate from callback/approval outcomes."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    UNAVAILABLE = "unavailable"


class PolicyReasonCode(str, Enum):
    AUTOMATIC_POLICY = "policy.automatic"
    APPROVAL_POLICY = "policy.approval_required"
    DISABLED_POLICY = "policy.disabled"
    TOOL_UNAVAILABLE = "policy.tool_unavailable"
    INVALID_ARGUMENT_BINDING = "policy.argument_binding_unavailable"
    FILE_SCOPE_INVALID = "policy.file_scope_invalid"
    FILE_TARGET_UNAVAILABLE = "policy.file_target_unavailable"
    UNSUPPORTED_POLICY = "policy.unsupported_configuration"


_POLICY_RESULT_ISSUER = object()


class _FrozenPolicyList(tuple):
    """Immutable list snapshot that retains a distinct argument type tag."""


def _freeze_policy_value(value):
    if value is None or type(value) in (bool, int, float, str) or isinstance(value, Path):
        return value
    if hasattr(value, "items"):
        copied = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("Policy argument object keys must be strings.")
            copied[key] = _freeze_policy_value(item)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        values = tuple(_freeze_policy_value(item) for item in value)
        return _FrozenPolicyList(values) if isinstance(value, list) else values
    raise TypeError("Unsupported policy argument value.")


@dataclass(frozen=True, init=False)
class PolicyEvaluationResult:
    """Immutable, application-level result produced by trusted policy code.

    This describes the policy requirement. A legacy synchronous callback may
    still resolve REQUIRE_APPROVAL for the existing executor path, but that
    callback result is not a reusable or durable approval grant.
    """

    disposition: PolicyDisposition
    reason_code: PolicyReasonCode
    detail: str
    risk: str | None
    capability_id: str | None
    contract_version: str | None
    manifest_digest: str | None
    argument_digest: str | None
    arguments: object
    execution_argument_digest: str | None
    execution_arguments: object
    policy_revision: str | None

    def __init__(
        self, *, _issuer=None, disposition, reason_code, detail, risk=None,
        capability_id=None, contract_version=None, manifest_digest=None,
        argument_digest=None, arguments=None, execution_argument_digest=None,
        execution_arguments=None, policy_revision=None,
    ):
        if _issuer is not _POLICY_RESULT_ISSUER:
            raise TypeError("Policy results can only be issued by trusted policy code.")
        if not isinstance(disposition, PolicyDisposition):
            raise TypeError("disposition must be a PolicyDisposition.")
        if not isinstance(reason_code, PolicyReasonCode):
            raise TypeError("reason_code must be a PolicyReasonCode.")
        if not isinstance(detail, str) or not detail:
            raise ValueError("detail must be a non-empty string.")
        expected_codes = {
            PolicyDisposition.ALLOW: {PolicyReasonCode.AUTOMATIC_POLICY},
            PolicyDisposition.REQUIRE_APPROVAL: {PolicyReasonCode.APPROVAL_POLICY},
            PolicyDisposition.DENY: {
                PolicyReasonCode.DISABLED_POLICY,
                PolicyReasonCode.FILE_SCOPE_INVALID,
                PolicyReasonCode.FILE_TARGET_UNAVAILABLE,
            },
            PolicyDisposition.UNAVAILABLE: {
                PolicyReasonCode.TOOL_UNAVAILABLE,
                PolicyReasonCode.INVALID_ARGUMENT_BINDING,
                PolicyReasonCode.UNSUPPORTED_POLICY,
            },
        }
        if reason_code not in expected_codes[disposition]:
            raise ValueError("Policy disposition and reason code are inconsistent.")
        if risk is not None and not isinstance(risk, str):
            raise TypeError("risk must be a string or None.")
        identity = (capability_id, contract_version, manifest_digest)
        if any(value is not None for value in identity) and not all(
            isinstance(value, str) and value for value in identity
        ):
            raise ValueError("Capability identity fields must be all present or all absent.")
        if (argument_digest is None) != (arguments is None):
            raise ValueError("Argument snapshot and digest must be present together.")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise TypeError("arguments must be an immutable mapping snapshot or None.")
        if (execution_argument_digest is None) != (execution_arguments is None):
            raise ValueError("Execution argument snapshot and digest must be present together.")
        if execution_arguments is not None and not isinstance(execution_arguments, Mapping):
            raise TypeError("execution_arguments must be an immutable mapping snapshot or None.")
        for name, value in (("capability_id", capability_id),
                            ("contract_version", contract_version),
                            ("manifest_digest", manifest_digest),
                            ("argument_digest", argument_digest),
                            ("execution_argument_digest", execution_argument_digest)):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None.")
        if policy_revision is not None and not isinstance(policy_revision, str):
            raise TypeError("policy_revision must be a string or None.")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "detail", detail[:1000])
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "contract_version", contract_version)
        object.__setattr__(self, "manifest_digest", manifest_digest)
        object.__setattr__(self, "argument_digest", argument_digest)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "execution_argument_digest", execution_argument_digest)
        object.__setattr__(self, "execution_arguments", execution_arguments)
        object.__setattr__(self, "policy_revision", policy_revision)


def _policy_result(**values) -> PolicyEvaluationResult:
    """Private factory; model/tool data is never deserialized into this type."""
    return PolicyEvaluationResult(_issuer=_POLICY_RESULT_ISSUER, **values)


class FileListScopeError(ValueError):
    """A file-list pattern cannot be proven to stay in its search root."""


def resolve_file_list_scope(path, pattern):
    """Resolve and freeze the exact file-list result set for authorization."""
    if not isinstance(pattern, str) or not pattern or "\x00" in pattern:
        raise FileListScopeError("The file-list pattern is invalid.")
    if "\\" in pattern:
        raise FileListScopeError("Backslashes are not supported in file-list patterns.")
    posix_pattern = PurePosixPath(pattern)
    windows_pattern = PureWindowsPath(pattern)
    if posix_pattern.is_absolute() or windows_pattern.is_absolute() or windows_pattern.drive:
        raise FileListScopeError("Absolute file-list patterns are not allowed.")
    if ".." in posix_pattern.parts:
        raise FileListScopeError("Traversal in file-list patterns is not allowed.")

    try:
        root = Path(path).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise FileListScopeError("The file-list path is not a directory.")
        parts = posix_pattern.parts
        found = set()

        def walk(directory, index):
            if index == len(parts):
                found.add(Path(directory))
                return
            component = parts[index]
            if component == "**":
                walk(directory, index + 1)
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if (
                            not entry.name.startswith(".")
                            and entry.is_dir(follow_symlinks=False)
                        ):
                            walk(entry.path, index)
                return

            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name.startswith(".") and not component.startswith("."):
                        continue
                    if not fnmatch.fnmatchcase(entry.name, component):
                        continue
                    if index == len(parts) - 1:
                        found.add(Path(entry.path))
                    elif entry.is_symlink():
                        target = Path(entry.path).resolve(strict=False)
                        try:
                            target.relative_to(root)
                        except ValueError as exc:
                            raise FileListScopeError(
                                "The file-list pattern resolves outside its directory."
                            ) from exc
                    elif entry.is_dir(follow_symlinks=False):
                        walk(entry.path, index + 1)

        walk(root, 0)
        resolved = []
        for entry in found:
            target = Path(entry).resolve(strict=True)
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise FileListScopeError(
                    "The file-list pattern resolves outside its directory."
                ) from exc
            resolved.append(target)
    except FileListScopeError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise FileListScopeError("The requested file-list scope is unsafe.") from exc
    return root, tuple(sorted(set(resolved), key=str))


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
    "current_time": ("low", "auto"),
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
        allowed, reason, _ = self.authorize_with_context(tool, args)
        return allowed, reason

    @staticmethod
    def _make_policy_result(
        disposition, reason_code, detail, risk, tool, arguments,
        execution_arguments=None,
    ) -> PolicyEvaluationResult:
        capability_id = contract_version = manifest_digest = None
        if tool is not None:
            try:
                # Descriptive M1A identity is attached when a known legacy
                # definition has a reviewed mapping. It does not affect policy.
                from .capabilities import legacy_tool_to_manifest

                manifest = legacy_tool_to_manifest(tool)
                capability_id = manifest.capability_id
                contract_version = manifest.contract_version
                manifest_digest = manifest.manifest_digest
            except Exception:
                pass

        snapshot = None
        digest = None
        execution_snapshot = None
        execution_digest = None
        try:
            from .execution_context import argument_digest

            snapshot = _freeze_policy_value(arguments)
            digest = argument_digest(arguments)
            final_arguments = arguments if execution_arguments is None else execution_arguments
            execution_snapshot = _freeze_policy_value(final_arguments)
            execution_digest = argument_digest(final_arguments)
        except Exception:
            disposition = PolicyDisposition.UNAVAILABLE
            reason_code = PolicyReasonCode.INVALID_ARGUMENT_BINDING
            detail = "Validated arguments could not be safely bound to policy evaluation."
            snapshot = None
            digest = None
            execution_snapshot = None
            execution_digest = None

        return _policy_result(
            disposition=disposition,
            reason_code=reason_code,
            detail=detail,
            risk=risk,
            capability_id=capability_id,
            contract_version=contract_version,
            manifest_digest=manifest_digest,
            argument_digest=digest,
            arguments=snapshot,
            execution_argument_digest=execution_digest,
            execution_arguments=execution_snapshot,
            # No authoritative policy revision/epoch exists in the current code.
            policy_revision=None,
        )

    def _evaluate_policy_once(self, tool, args):
        """Evaluate trusted rules once; retain legacy execution args privately."""
        if tool is None:
            result = self._make_policy_result(
                PolicyDisposition.UNAVAILABLE,
                PolicyReasonCode.TOOL_UNAVAILABLE,
                "Unknown tool.", None, None, args if args is not None else {},
                args if args is not None else {},
            )
            # Preserve the pre-existing third tuple item for this legacy case.
            return result, args, args, None
        if not isinstance(args, Mapping):
            return (
                self._make_policy_result(
                    PolicyDisposition.UNAVAILABLE,
                    PolicyReasonCode.INVALID_ARGUMENT_BINDING,
                    "Validated arguments are unavailable.", None, tool, {},
                    {},
                ),
                {},
                args,
                None,
            )
        try:
            execution_args = dict(args) if args is not None else {}
        except (TypeError, ValueError):
            execution_args = {}
            return (
                self._make_policy_result(
                    PolicyDisposition.UNAVAILABLE,
                    PolicyReasonCode.INVALID_ARGUMENT_BINDING,
                    "Validated arguments are unavailable.", None, tool, {},
                    {},
                ),
                execution_args,
                args,
                None,
            )

        try:
            policy = policy_for(tool.name)
            if (
                not isinstance(policy, tuple) or len(policy) != 2
                or not all(isinstance(value, str) and value for value in policy)
            ):
                raise ValueError("Unsupported policy configuration.")
            risk, mode = policy
        except Exception:
            return (
                self._make_policy_result(
                    PolicyDisposition.UNAVAILABLE,
                    PolicyReasonCode.UNSUPPORTED_POLICY,
                    "Trusted policy configuration is unavailable.", None, tool, args,
                    execution_args,
                ),
                execution_args,
                args,
                None,
            )

        if mode == "disabled":
            return (
                self._make_policy_result(
                    PolicyDisposition.DENY,
                    PolicyReasonCode.DISABLED_POLICY,
                    f"'{tool.name}' is disabled in the current runtime.",
                    risk, tool, args, execution_args,
                ),
                execution_args,
                args,
                False,
            )
        if mode not in {"auto", "approval"}:
            return (
                self._make_policy_result(
                    PolicyDisposition.UNAVAILABLE,
                    PolicyReasonCode.UNSUPPORTED_POLICY,
                    "Trusted policy configuration is unavailable.", None, tool, args,
                    execution_args,
                ),
                execution_args,
                args,
                None,
            )

        requires_approval = mode == "approval"
        candidate = None
        sensitive = False
        if tool.name == "file_list":
            try:
                root, matches = resolve_file_list_scope(
                    args.get("path", "."), args.get("pattern", "*")
                )
            except FileListScopeError as exc:
                return (
                    self._make_policy_result(
                        PolicyDisposition.DENY,
                        PolicyReasonCode.FILE_SCOPE_INVALID,
                        str(exc), risk, tool, args, execution_args,
                    ),
                    execution_args,
                    args,
                    False,
                )
            candidate = root
            execution_args["_authorized_file_list_scope"] = (root, matches)
            sensitive = any(
                part.lower() in self.SENSITIVE_PATH_PARTS
                for entry in (root, *matches)
                for part in entry.parts
            ) or any(Path(match).name.lower().startswith(".env") for match in matches)
        elif tool.name == "file_read":
            raw_path = Path(args.get("path", ".")).expanduser()
            if not raw_path.is_absolute():
                raw_path = Path.cwd() / raw_path
            try:
                candidate = raw_path.resolve(strict=False)
            except (OSError, RuntimeError):
                return (
                    self._make_policy_result(
                        PolicyDisposition.DENY,
                        PolicyReasonCode.FILE_TARGET_UNAVAILABLE,
                        "The requested path could not be resolved safely.",
                        risk, tool, args, execution_args,
                    ),
                    execution_args,
                    args,
                    False,
                )
            sensitive = any(
                part.lower() in self.SENSITIVE_PATH_PARTS
                for part in candidate.parts
            ) or candidate.name.lower().startswith(".env")

        inside_workspace = False
        if candidate is not None and self.workspace_root:
            try:
                candidate.relative_to(self.workspace_root)
                inside_workspace = True
            except ValueError:
                pass
        if tool.name in {"file_read", "file_list"} and (
            sensitive or self.workspace_root is None or not inside_workspace
        ):
            requires_approval = True
            risk = "high"

        if requires_approval:
            disposition = PolicyDisposition.REQUIRE_APPROVAL
            code = PolicyReasonCode.APPROVAL_POLICY
            detail = f"'{tool.name}' requires {risk}-risk approval."
        else:
            disposition = PolicyDisposition.ALLOW
            code = PolicyReasonCode.AUTOMATIC_POLICY
            detail = "Automatically allowed by policy."
        return (
            self._make_policy_result(
                disposition, code, detail, risk, tool, args, execution_args
            ),
            execution_args,
            args,
            requires_approval,
        )

    def evaluate_policy(self, tool, validated_arguments) -> PolicyEvaluationResult:
        """Return policy requirements without invoking callbacks or tools."""
        result, _, _, _ = self._evaluate_policy_once(tool, validated_arguments)
        return result

    def evaluate_for_executor(self, tool, validated_arguments):
        """Evaluate once and retain the exact execution arguments for Harness.

        This is the executor's typed boundary: callers must route an explicit
        REQUIRE_APPROVAL result to the approval lifecycle and must not infer it
        from the legacy approval callback.
        """
        result, execution_args, _callback_args, _requires_approval = (
            self._evaluate_policy_once(tool, validated_arguments)
        )
        return result, execution_args

    @staticmethod
    def legacy_result_for_policy(result, execution_args):
        """Map non-approval typed outcomes to the existing dispatch tuple."""
        if not isinstance(result, PolicyEvaluationResult):
            return False, "Trusted policy result is unavailable.", execution_args
        if result.disposition is PolicyDisposition.ALLOW:
            return True, "Automatically allowed by policy.", execution_args
        if result.disposition is PolicyDisposition.DENY:
            return False, result.detail, execution_args
        if result.disposition is PolicyDisposition.UNAVAILABLE:
            return False, result.detail, execution_args
        return False, "Approval must be handled by the Harness continuation lifecycle.", execution_args

    def authorize_with_context(self, tool, args):
        """Adapt one typed policy result to the legacy authorization tuple."""
        policy_result, execution_args, callback_args, requires_approval = (
            self._evaluate_policy_once(tool, args)
        )
        if policy_result.disposition is PolicyDisposition.ALLOW:
            return True, "Automatically allowed by policy.", execution_args
        if policy_result.disposition is PolicyDisposition.DENY:
            return False, policy_result.detail, execution_args
        if policy_result.disposition is PolicyDisposition.UNAVAILABLE:
            return False, policy_result.detail, execution_args

        # Compatibility adapter for the existing synchronous caller. A True
        # return authorizes only this legacy invocation and is not persisted,
        # reusable, or represented as a typed ALLOW policy disposition.
        if not requires_approval:
            return False, "Trusted policy configuration is unavailable.", execution_args
        risk = policy_result.risk or "high"
        if self.approval_callback is None:
            return False, (
                f"'{tool.name}' requires {risk}-risk approval, "
                "but no approval handler is configured."
            ), execution_args
        try:
            approved = self.approval_callback(tool, callback_args, risk)
        except Exception as exc:
            return False, f"Approval failed safely: {exc}", execution_args
        if approved is True:
            return True, "Approved for this exact call.", execution_args
        return False, f"Permission not granted for '{tool.name}'.", execution_args
