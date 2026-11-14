"""Descriptive, non-authoritative capability manifests for M1A.

This module deliberately contains no executable provider loader or dispatch API.
Legacy tool projections are observations, not safety certifications.

Contract versions use semantic-version intent: major for incompatible input,
output, or externally observable meaning changes; minor for compatible contract
extensions; patch for compatible corrections or clarifications. Implementation
changes do not by themselves bump the contract version. Legacy contracts use
0.0.0-legacy as a provisional observation marker, not a released guarantee.
Any manifest payload change changes its digest and requires review for effect,
risk-relevant metadata, and bound changes. Kernel policy remains independent.

The legacy screen_capture record is historical only. It is not the future
user-initiated, selected-scope screen-perception contract and is prohibited
from future consumer exposure. Consumer-exposure metadata is descriptive only;
future consumers must apply an explicit review allowlist and must never expose
every registered catalog entry by default.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Mapping

from .tools import Tool, ToolArg

# Provisional namespace: owner confirmation is required before external use.
CAPABILITY_NAMESPACE = "org.lila.capability"  # PROVISIONAL; not an ownership claim.
LEGACY_CONTRACT_VERSION = "0.0.0-legacy"  # Observation identifier, not a guarantee.
CANONICALIZATION_PROFILE = "rfc8785-jcs-v1-i53"
MANIFEST_DIGEST_DOMAIN = b"lila-capability-manifest\0rfc8785-jcs-v1-i53\0"
PROVIDER_DIGEST_DOMAIN = b"lila-provider-artifact\0sha256-v1\0"
MAX_SAFE_INTEGER = (2**53) - 1


# Explicit, centralized mappings. Unknown tool names are rejected rather than
# converted into arbitrary identifiers from untrusted strings.
LEGACY_CAPABILITY_IDS = MappingProxyType({
    "file_read": f"{CAPABILITY_NAMESPACE}.files.read",
    "file_write": f"{CAPABILITY_NAMESPACE}.files.write",
    "file_append": f"{CAPABILITY_NAMESPACE}.files.append",
    "file_list": f"{CAPABILITY_NAMESPACE}.files.list",
    "file_delete": f"{CAPABILITY_NAMESPACE}.files.delete",
    "bash": f"{CAPABILITY_NAMESPACE}.process.shell",
    "memory_store": f"{CAPABILITY_NAMESPACE}.memory.store",
    "memory_recall": f"{CAPABILITY_NAMESPACE}.memory.recall",
    "memory_list": f"{CAPABILITY_NAMESPACE}.memory.list",
    "schedule_add": f"{CAPABILITY_NAMESPACE}.schedule.add",
    "schedule_list": f"{CAPABILITY_NAMESPACE}.schedule.list",
    "web_fetch": f"{CAPABILITY_NAMESPACE}.network.web-fetch",
    "self_status": f"{CAPABILITY_NAMESPACE}.lila.status",
    "self_log": f"{CAPABILITY_NAMESPACE}.lila.log",
    "tcp_connect": f"{CAPABILITY_NAMESPACE}.network.tcp-connect",
    "udp_send": f"{CAPABILITY_NAMESPACE}.network.udp-send",
    "tcp_listen": f"{CAPABILITY_NAMESPACE}.network.tcp-listen",
    "ssh_exec": f"{CAPABILITY_NAMESPACE}.network.ssh-exec",
    "wifi_scan": f"{CAPABILITY_NAMESPACE}.network.wifi-scan",
    "wifi_connect": f"{CAPABILITY_NAMESPACE}.network.wifi-connect",
    "dns_lookup": f"{CAPABILITY_NAMESPACE}.network.dns-lookup",
    "ping": f"{CAPABILITY_NAMESPACE}.network.ping",
    "http_request": f"{CAPABILITY_NAMESPACE}.network.http-request",
    "gpio_write": f"{CAPABILITY_NAMESPACE}.hardware.gpio-write",
    "gpio_read": f"{CAPABILITY_NAMESPACE}.hardware.gpio-read",
    "i2c_transfer": f"{CAPABILITY_NAMESPACE}.hardware.i2c-transfer",
    "spi_transfer": f"{CAPABILITY_NAMESPACE}.hardware.spi-transfer",
    "serial_write": f"{CAPABILITY_NAMESPACE}.hardware.serial-write",
    "serial_read": f"{CAPABILITY_NAMESPACE}.hardware.serial-read",
    "usb_list": f"{CAPABILITY_NAMESPACE}.hardware.usb-list",
    "set_volume": f"{CAPABILITY_NAMESPACE}.system.volume-set",
    "power_action": f"{CAPABILITY_NAMESPACE}.system.power-action",
    "notify_user": f"{CAPABILITY_NAMESPACE}.interaction.notify",
    "clipboard_get": f"{CAPABILITY_NAMESPACE}.system.clipboard-read",
    "clipboard_set": f"{CAPABILITY_NAMESPACE}.system.clipboard-write",
    "screen_capture": f"{CAPABILITY_NAMESPACE}.perception.legacy-screen-capture",
    "island_control": f"{CAPABILITY_NAMESPACE}.interaction.island-control",
    "screen_present": f"{CAPABILITY_NAMESPACE}.interaction.screen-present",
    "web_search": f"{CAPABILITY_NAMESPACE}.network.web-search",
    "file_search": f"{CAPABILITY_NAMESPACE}.filesystem.file-search",
    "workstation_overview": f"{CAPABILITY_NAMESPACE}.workstation.overview",
    "workstation_list_categories": f"{CAPABILITY_NAMESPACE}.workstation.categories.list",
    "workstation_list_workspaces": f"{CAPABILITY_NAMESPACE}.workstation.workspaces.list",
    "workstation_get_workspace": f"{CAPABILITY_NAMESPACE}.workstation.workspace.get",
    "workstation_intake_overview": f"{CAPABILITY_NAMESPACE}.workstation.intake.overview",
    "workstation_software_overview": f"{CAPABILITY_NAMESPACE}.workstation.software.overview",
    "current_time": f"{CAPABILITY_NAMESPACE}.system.current-time",
})

_FINDINGS: Mapping[str, tuple[tuple[str, str], ...]] = MappingProxyType({
    "memory_store": (("disabled_placeholder", "Registered memory handler is disabled by current policy."),),
    "memory_recall": (("disabled_placeholder", "Registered memory handler is disabled by current policy."),),
    "memory_list": (("disabled_placeholder", "Registered memory handler is disabled by current policy."),),
    "tcp_connect": (("schema_mismatch", "Handler accepts hex_mode, absent from ToolArg schema."),),
    "spi_transfer": (("schema_mismatch", "Declared speed argument is not used by the handler."),),
    "gpio_read": (("side_effecting_read", "Handler may export GPIO and set direction to input."),),
    "set_volume": (("schema_mismatch", "Windows handler sends a volume-down key and ignores requested level."),),
    "screen_capture": (("unsafe_legacy", "Legacy implementation uses shell command construction and lacks explicit selected-scope capture."),),
})

_EFFECTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "file_read": ("filesystem.read",), "file_write": ("filesystem.create_or_overwrite",),
    "file_append": ("filesystem.append",), "file_list": ("filesystem.enumerate",),
    "file_delete": ("filesystem.delete",), "bash": ("process.execute_arbitrary_command",),
    "memory_store": ("memory.write",), "memory_recall": ("memory.read",), "memory_list": ("memory.read",),
    "schedule_add": ("schedule.write",), "schedule_list": ("schedule.read",),
    "web_fetch": ("network.http_get",), "self_status": ("system.metadata_read",), "self_log": ("filesystem.append",),
    "tcp_connect": ("network.connect", "network.send", "network.receive"),
    "udp_send": ("network.send",), "tcp_listen": ("network.listen", "network.receive"),
    "ssh_exec": ("network.remote_process_execution",), "wifi_scan": ("network.scan",),
    "wifi_connect": ("network.configuration_change",), "dns_lookup": ("network.resolve",),
    "ping": ("network.probe",), "http_request": ("network.http_request",),
    "gpio_write": ("device.write",), "gpio_read": ("device.read", "device.configuration_change_possible"),
    "i2c_transfer": ("device.read_or_write",), "spi_transfer": ("device.write",),
    "serial_write": ("device.write", "device.configuration_change_possible"),
    "serial_read": ("device.read", "device.configuration_change_possible"), "usb_list": ("device.enumerate",),
    "set_volume": ("system.configuration_change",), "power_action": ("system.control",),
    "notify_user": ("user.notify",), "clipboard_get": ("clipboard.read",),
    "clipboard_set": ("clipboard.write",), "screen_capture": ("screen.capture", "filesystem.write"),
    "island_control": ("ui.control",),
    "screen_present": ("ui.present",),
    "web_search": ("network.http_get",),
    "file_search": ("filesystem.enumerate", "filesystem.read"),
    "workstation_overview": ("workstation.metadata_read",),
    "workstation_list_categories": ("workstation.metadata_read",),
    "workstation_list_workspaces": ("workstation.metadata_read",),
    "workstation_get_workspace": ("workstation.metadata_read",),
    "workstation_intake_overview": ("workstation.configuration_read",),
    "workstation_software_overview": ("workstation.configuration_read",),
    "current_time": ("system.metadata_read",),
})

# Only limits explicitly visible in current handler code are recorded.
_BOUNDS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "file_read": {"handler_output_chars": 50000},
    "file_list": {"handler_output_entries": 200},
    "bash": {"handler_timeout_arg_max_seconds": 300, "handler_output_chars": 20000},
    "web_fetch": {"handler_timeout_seconds": 10, "handler_output_chars": 30000},
    "tcp_connect": {"handler_receive_bytes": 65536},
    "tcp_listen": {"handler_receive_bytes": 65536},
    "ssh_exec": {"handler_timeout_seconds": 30},
    "wifi_scan": {"handler_timeout_seconds": 15},
    "http_request": {"handler_timeout_seconds": 10, "handler_output_chars": 100000},
    "i2c_transfer": {"handler_timeout_seconds": 5}, "spi_transfer": {"handler_timeout_seconds": 5},
    "usb_list": {"handler_timeout_seconds": 5}, "set_volume": {"handler_timeout_seconds": 5},
    "power_action": {"handler_timeout_seconds": 10}, "notify_user": {"handler_timeout_seconds": 5},
    "clipboard_get": {"handler_timeout_seconds": 5}, "clipboard_set": {"handler_timeout_seconds": 5},
    "screen_capture": {"handler_timeout_seconds": 10},
    "workstation_overview": {"handler_output_chars": 8000},
    "workstation_list_categories": {"handler_output_chars": 8000, "handler_output_items": 100},
    "workstation_list_workspaces": {"handler_output_chars": 8000, "handler_output_items": 100},
    "workstation_get_workspace": {"handler_output_chars": 8000},
    "workstation_intake_overview": {"handler_output_chars": 8000},
    "workstation_software_overview": {"handler_output_chars": 8000},
})

_SEMVER_NUMERIC = r"(?:0|[1-9][0-9]*)"
_SEMVER_PRERELEASE_ID = r"(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
_VERSION_RE = re.compile(
    rf"^{_SEMVER_NUMERIC}\.{_SEMVER_NUMERIC}\.{_SEMVER_NUMERIC}"
    rf"(?:-{_SEMVER_PRERELEASE_ID}(?:\.{_SEMVER_PRERELEASE_ID})*)?"
    rf"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_CANONICAL_DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")
_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)+$")
_FINDING_CODES = frozenset({"observed_legacy", "incomplete_output", "schema_mismatch", "side_effecting_read", "unsafe_legacy", "disabled_placeholder"})
CONSUMER_EXPOSURE_STATES = frozenset({
    "unreviewed",
    "approved_for_consumer_exposure",
    "prohibited_from_consumer_exposure",
})


class ManifestValidationError(ValueError):
    """Invalid or unsupported descriptive manifest data."""


class DuplicateCapabilityError(ValueError):
    """A registry already contains this exact capability identity/version."""


def _validate_string(value: str) -> str:
    if not isinstance(value, str):
        raise ManifestValidationError("Expected a string value.")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ManifestValidationError("Lone Unicode surrogates are unsupported.") from exc
    return value


def _freeze_json(value: Any) -> Any:
    """Copy and recursively freeze the restricted JCS value domain."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise ManifestValidationError("Integer exceeds the interoperable safe range.")
        return value
    if isinstance(value, float):
        raise ManifestValidationError("Floating-point values are unsupported; use a decimal string.")
    if isinstance(value, str):
        return _validate_string(value)
    if isinstance(value, Mapping):
        copied = {}
        for key, item in value.items():
            _validate_string(key)
            if key in copied:
                raise ManifestValidationError(f"Duplicate object key: {key}")
            copied[key] = _freeze_json(item)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise ManifestValidationError(f"Unsupported manifest value type: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _jcs_string(value: str) -> str:
    _validate_string(value)
    # For valid Unicode strings Python's non-ASCII-preserving JSON escaping
    # matches JCS string escaping; UTF-8 encoding is applied after serialization.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _utf16_sort_key(value: str) -> bytes:
    _validate_string(value)
    return value.encode("utf-16-be", errors="strict")


def canonicalize_restricted_jcs(value: Any) -> bytes:
    """Serialize the restricted JCS profile used by M1A.

    Accepted values are null, booleans, strings, arrays, objects and safe
    integers. Floats are rejected. This is JCS-compatible for this domain,
    not a general-purpose RFC 8785 implementation.
    """
    frozen = _freeze_json(value)

    def encode(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, int):
            return str(item)
        if isinstance(item, str):
            return _jcs_string(item)
        if isinstance(item, tuple):
            return "[" + ",".join(encode(element) for element in item) + "]"
        if isinstance(item, Mapping):
            keys = sorted(item.keys(), key=_utf16_sort_key)
            return "{" + ",".join(_jcs_string(key) + ":" + encode(item[key]) for key in keys) + "}"
        raise ManifestValidationError("Unsupported canonical JSON value.")

    return encode(frozen).encode("utf-8", errors="strict")


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_strict(text: str) -> Any:
    """Parse JSON while rejecting duplicate keys and unsupported values."""
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ManifestValidationError(f"Non-finite number is unsupported: {value}")
            ),
        )
    except ManifestValidationError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise ManifestValidationError("Malformed JSON manifest.") from exc
    return _freeze_json(parsed)


_MANIFEST_FIELDS = frozenset({
    "capability_id", "contract_version", "display_name", "summary",
    "input_arguments", "output_contract", "declared_effects",
    "resource_bounds", "findings", "availability_observations",
    "legacy_provenance", "consumer_exposure",
})
_ARGUMENT_FIELDS = frozenset({
    "name", "type", "description", "required", "default", "choices",
    "minimum", "maximum",
})


def manifest_from_mapping(value: Mapping[str, Any]) -> "CapabilityManifest":
    """Strictly validate an untrusted manifest payload and construct a value."""
    if not isinstance(value, Mapping):
        raise ManifestValidationError("Manifest payload must be an object.")
    unknown = set(value) - _MANIFEST_FIELDS
    missing = _MANIFEST_FIELDS - set(value)
    if unknown or missing:
        raise ManifestValidationError(
            f"Manifest fields invalid; unknown={sorted(unknown)}, missing={sorted(missing)}."
        )
    if not isinstance(value["input_arguments"], (list, tuple)):
        raise ManifestValidationError("input_arguments must be an array.")
    arguments = []
    seen_names = set()
    for raw_arg in value["input_arguments"]:
        if not isinstance(raw_arg, Mapping):
            raise ManifestValidationError("Each input argument must be an object.")
        if set(raw_arg) != _ARGUMENT_FIELDS:
            raise ManifestValidationError("Input argument has missing or unknown fields.")
        name, arg_type, description = raw_arg["name"], raw_arg["type"], raw_arg["description"]
        if not all(isinstance(item, str) for item in (name, arg_type, description)):
            raise ManifestValidationError("Argument name, type and description must be strings.")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in seen_names:
            raise ManifestValidationError("Argument name is invalid or duplicated.")
        if arg_type not in {"string", "int", "float", "bool", "path"}:
            raise ManifestValidationError("Unsupported legacy argument type.")
        if not isinstance(raw_arg["required"], bool):
            raise ManifestValidationError("Argument required must be boolean.")
        if raw_arg["choices"] is not None and not isinstance(raw_arg["choices"], (list, tuple)):
            raise ManifestValidationError("Argument choices must be an array or null.")
        frozen_arg = _freeze_json(dict(raw_arg))
        _validate_argument_contract(frozen_arg)
        arguments.append(frozen_arg)
        seen_names.add(name)
    if not isinstance(value["declared_effects"], (list, tuple)) or not all(isinstance(item, str) for item in value["declared_effects"]):
        raise ManifestValidationError("declared_effects must be an array of strings.")
    if not isinstance(value["findings"], (list, tuple)):
        raise ManifestValidationError("findings must be an array.")
    if not all(isinstance(value[field], Mapping) for field in ("output_contract", "resource_bounds", "availability_observations", "legacy_provenance")):
        raise ManifestValidationError("Manifest contract, bounds, availability and provenance must be objects.")
    return CapabilityManifest(
        capability_id=value["capability_id"],
        contract_version=value["contract_version"],
        display_name=value["display_name"],
        summary=value["summary"],
        input_arguments=tuple(arguments),
        output_contract=value["output_contract"],
        declared_effects=tuple(value["declared_effects"]),
        resource_bounds=value["resource_bounds"],
        findings=tuple(value["findings"]),
        availability_observations=value["availability_observations"],
        legacy_provenance=value["legacy_provenance"],
        consumer_exposure=value["consumer_exposure"],
    )


def manifest_from_json(text: str) -> "CapabilityManifest":
    parsed = parse_json_strict(text)
    return manifest_from_mapping(parsed)


def _decimal_string(value: float | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ManifestValidationError("Decimal source must be a non-boolean integer or float.")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ManifestValidationError("Invalid decimal bound.") from exc
    if not decimal.is_finite():
        raise ManifestValidationError("Non-finite decimal bound.")
    if decimal.is_zero():
        return "0"

    # Fixed-point formatting without normalize() preserves every source digit;
    # Decimal.normalize() applies the active context and can round large values.
    try:
        canonical = format(decimal, "f")
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise ManifestValidationError(
            "Decimal bound cannot be represented in canonical fixed-point form."
        ) from exc
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    if not _CANONICAL_DECIMAL_RE.fullmatch(canonical):
        raise ManifestValidationError(
            "Decimal bound cannot be represented by the supported canonical grammar."
        )
    return canonical


def _parse_canonical_decimal(value: Any, field_name: str) -> Decimal:
    """Parse exact decimal text in the canonical no-exponent manifest grammar.

    Grammar: an optional minus sign; integer part `0` or `[1-9][0-9]*`; an
    optional literal decimal point followed by digits ending in a nonzero digit.
    Negative zero, exponent notation, leading zeroes, and redundant fractional
    zeroes are rejected.
    """
    if not isinstance(value, str) or not _CANONICAL_DECIMAL_RE.fullmatch(value):
        raise ManifestValidationError(
            f"Argument {field_name} must use canonical decimal text."
        )
    if value.startswith("-") and Decimal(value).is_zero():
        raise ManifestValidationError("Negative zero is not canonical decimal text.")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ManifestValidationError("Invalid canonical decimal value.") from exc


def _validate_argument_value_type(value: Any, arg_type: str, field_name: str) -> None:
    """Validate a schema value without coercion, using the restricted JSON model."""
    valid = False
    if arg_type in {"string", "path"}:
        valid = isinstance(value, str) and (arg_type != "path" or bool(value.strip()))
    elif arg_type == "int":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif arg_type == "float":
        # Exact decimal values are strings because manifest JSON rejects floats.
        valid = isinstance(value, str)
        if valid:
            _parse_canonical_decimal(value, field_name)
    elif arg_type == "bool":
        valid = isinstance(value, bool)
    if not valid:
        raise ManifestValidationError(
            f"Argument {field_name} is incompatible with declared type {arg_type!r}."
        )


def _validate_argument_contract(argument: Mapping[str, Any]) -> None:
    """Validate a frozen argument schema for internally consistent semantics."""
    name, arg_type = argument["name"], argument["type"]
    required = argument["required"]
    default = argument["default"]
    choices = argument["choices"]
    minimum, maximum = argument["minimum"], argument["maximum"]

    if required and default is not None:
        raise ManifestValidationError(
            f"Required argument {name!r} cannot declare an applied default."
        )
    if default is not None:
        _validate_argument_value_type(default, arg_type, "default")

    if choices is not None:
        if not isinstance(choices, tuple) or not choices:
            raise ManifestValidationError("Argument choices must be a non-empty array or null.")
        seen_choices = set()
        for choice in choices:
            _validate_argument_value_type(choice, arg_type, "choice")
            # Include the type so values such as 1 and True cannot collide.
            identity = (type(choice), choice)
            if identity in seen_choices:
                raise ManifestValidationError("Argument choices must be unique.")
            seen_choices.add(identity)
        if default is not None and (type(default), default) not in seen_choices:
            raise ManifestValidationError("Argument default must be one of its declared choices.")

    if minimum is not None or maximum is not None:
        if arg_type not in {"int", "float"}:
            raise ManifestValidationError("Only numeric arguments may declare bounds.")
    parsed_minimum = (
        _parse_canonical_decimal(minimum, "minimum") if minimum is not None else None
    )
    parsed_maximum = (
        _parse_canonical_decimal(maximum, "maximum") if maximum is not None else None
    )
    if arg_type == "int":
        if any(bound is not None and bound != bound.to_integral_value()
               for bound in (parsed_minimum, parsed_maximum)):
            raise ManifestValidationError("Integer argument bounds must be integral decimals.")
    if parsed_minimum is not None and parsed_maximum is not None and parsed_minimum > parsed_maximum:
        raise ManifestValidationError("Argument minimum cannot exceed maximum.")

    def numeric_value(value: Any) -> Decimal:
        if arg_type == "int":
            return Decimal(value)
        return _parse_canonical_decimal(value, "numeric value")

    values_to_bound_check = []
    if default is not None and arg_type in {"int", "float"}:
        values_to_bound_check.append(("default", default))
    if choices is not None and arg_type in {"int", "float"}:
        values_to_bound_check.extend(("choice", choice) for choice in choices)
    for label, value in values_to_bound_check:
        numeric = numeric_value(value)
        if parsed_minimum is not None and numeric < parsed_minimum:
            raise ManifestValidationError(f"Argument {label} is below its minimum.")
        if parsed_maximum is not None and numeric > parsed_maximum:
            raise ManifestValidationError(f"Argument {label} is above its maximum.")


def _legacy_argument(arg: ToolArg) -> Mapping[str, Any]:
    choices = None
    if arg.choices is not None:
        choices = []
        for choice in arg.choices:
            if arg.type == "float" and isinstance(choice, (int, float)) and not isinstance(choice, bool):
                choices.append(_decimal_string(choice))
            else:
                choices.append(choice)
    # The legacy executor ignores defaults whenever required=True. Project the
    # effective contract rather than carrying an inactive declaration (file_list
    # currently has this legacy shape).
    default = None if arg.required else arg.default
    if arg.type == "float" and isinstance(default, (int, float)) and not isinstance(default, bool):
        default = _decimal_string(default)
    record = {
        "name": arg.name,
        "type": arg.type,
        "description": arg.description,
        "required": arg.required,
        "default": default,
        "choices": choices,
        "minimum": _decimal_string(arg.minimum),
        "maximum": _decimal_string(arg.maximum),
    }
    return _freeze_json(record)


def _manifest_payload(manifest: "CapabilityManifest") -> dict[str, Any]:
    return {
        "profile": CANONICALIZATION_PROFILE,
        "capability_id": manifest.capability_id,
        "contract_version": manifest.contract_version,
        "display_name": manifest.display_name,
        "summary": manifest.summary,
        "input_arguments": [_thaw_json(arg) for arg in manifest.input_arguments],
        "output_contract": _thaw_json(manifest.output_contract),
        "declared_effects": list(manifest.declared_effects),
        "resource_bounds": _thaw_json(manifest.resource_bounds),
        "findings": [_thaw_json(finding) for finding in manifest.findings],
        "availability_observations": _thaw_json(manifest.availability_observations),
        "legacy_provenance": _thaw_json(manifest.legacy_provenance),
        "consumer_exposure": manifest.consumer_exposure,
    }


@dataclass(frozen=True)
class CapabilityManifest:
    capability_id: str
    contract_version: str
    display_name: str
    summary: str
    input_arguments: tuple[Mapping[str, Any], ...]
    output_contract: Mapping[str, Any]
    availability_observations: Mapping[str, Any]
    legacy_provenance: Mapping[str, Any]
    # Descriptive review status only; never execution or authorization authority.
    consumer_exposure: str = "unreviewed"
    declared_effects: tuple[str, ...] = ()
    resource_bounds: Mapping[str, Any] = field(default_factory=dict)
    findings: tuple[Mapping[str, Any], ...] = ()
    manifest_digest: str = field(init=False)
    canonicalization_profile: str = field(default=CANONICALIZATION_PROFILE, init=False)

    def __post_init__(self):
        if not isinstance(self.capability_id, str) or not _ID_RE.fullmatch(self.capability_id):
            raise ManifestValidationError("Invalid capability ID.")
        if not isinstance(self.contract_version, str) or not _VERSION_RE.fullmatch(self.contract_version):
            raise ManifestValidationError("Invalid semantic contract version.")
        if not all(isinstance(value, str) for value in (self.display_name, self.summary)):
            raise ManifestValidationError("Display name and summary must be strings.")
        if not self.display_name or not isinstance(self.input_arguments, (tuple, list)):
            raise ManifestValidationError("Display name and input argument sequence are required.")
        _validate_string(self.display_name)
        _validate_string(self.summary)
        arguments = tuple(_freeze_json(arg) for arg in self.input_arguments)
        argument_names = set()
        for argument in arguments:
            if not isinstance(argument, Mapping) or set(argument) != _ARGUMENT_FIELDS:
                raise ManifestValidationError("Input argument has missing or unknown fields.")
            name, arg_type, description = argument["name"], argument["type"], argument["description"]
            if not all(isinstance(item, str) for item in (name, arg_type, description)):
                raise ManifestValidationError("Argument name, type and description must be strings.")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in argument_names:
                raise ManifestValidationError("Argument name is invalid or duplicated.")
            if arg_type not in {"string", "int", "float", "bool", "path"}:
                raise ManifestValidationError("Unsupported legacy argument type.")
            if not isinstance(argument["required"], bool):
                raise ManifestValidationError("Argument required must be boolean.")
            _validate_argument_contract(argument)
            argument_names.add(name)
        effects = tuple(_validate_string(effect) for effect in self.declared_effects)
        findings = tuple(_freeze_json(finding) for finding in self.findings)
        for finding in findings:
            if not isinstance(finding, Mapping) or set(finding) != {"code", "evidence"} or finding.get("code") not in _FINDING_CODES or not isinstance(finding.get("evidence"), str):
                raise ManifestValidationError("Unknown or malformed contract finding.")
        output_contract = _freeze_json(self.output_contract)
        bounds = _freeze_json(self.resource_bounds)
        availability = _freeze_json(self.availability_observations)
        provenance = _freeze_json(self.legacy_provenance)
        if not isinstance(output_contract, Mapping) or not isinstance(bounds, Mapping):
            raise ManifestValidationError("Output contract and resource bounds must be objects.")
        if not isinstance(availability, Mapping) or not isinstance(provenance, Mapping):
            raise ManifestValidationError("Availability and provenance must be objects.")
        if set(output_contract) != {"kind", "fields", "guarantees"} or not isinstance(output_contract["kind"], str) or not isinstance(output_contract["fields"], Mapping) or set(output_contract["fields"]) != {"success", "output", "error", "data"} or not all(isinstance(item, str) for item in output_contract["fields"].values()) or not isinstance(output_contract["guarantees"], tuple) or not all(isinstance(item, str) for item in output_contract["guarantees"]):
            raise ManifestValidationError("Unsupported output-contract shape.")
        if set(availability) != {"registered_observation", "legacy_prompt_visible", "legacy_risk_label", "legacy_permission_label", "daemon_without_approval_callback"}:
            raise ManifestValidationError("Unsupported availability observation fields.")
        if not isinstance(availability["registered_observation"], bool) or not isinstance(availability["legacy_prompt_visible"], bool) or any(not isinstance(availability[key], str) for key in set(availability) - {"registered_observation", "legacy_prompt_visible"}):
            raise ManifestValidationError("Malformed availability observations.")
        if set(provenance) != {"provider_module", "provider_digest_status", "provider_integrity_digest", "legacy_tool_name"} or provenance["provider_digest_status"] != "unverified" or provenance["provider_integrity_digest"] is not None:
            raise ManifestValidationError("Legacy provider integrity must remain explicitly unverified.")
        if provenance["provider_module"] is not None and not isinstance(provenance["provider_module"], str):
            raise ManifestValidationError("Provider module provenance must be text or unknown.")
        if not isinstance(provenance["legacy_tool_name"], str):
            raise ManifestValidationError("Legacy tool provenance must be text.")
        if (not isinstance(self.consumer_exposure, str)
                or self.consumer_exposure not in CONSUMER_EXPOSURE_STATES):
            raise ManifestValidationError("Unknown consumer-exposure classification.")
        prohibited_legacy_ids = {
            LEGACY_CAPABILITY_IDS[name]
            for name in ("memory_store", "memory_recall", "memory_list", "screen_capture")
        }
        prohibited_legacy_names = {"memory_store", "memory_recall", "memory_list", "screen_capture"}
        if (
            self.consumer_exposure != "prohibited_from_consumer_exposure"
            and (
                self.capability_id in prohibited_legacy_ids
                or provenance["legacy_tool_name"] in prohibited_legacy_names
            )
        ):
            raise ManifestValidationError(
                "This disabled or unsafe legacy identity is prohibited from consumer exposure."
            )
        object.__setattr__(self, "input_arguments", arguments)
        object.__setattr__(self, "declared_effects", effects)
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "output_contract", output_contract)
        object.__setattr__(self, "resource_bounds", bounds)
        object.__setattr__(self, "availability_observations", availability)
        object.__setattr__(self, "legacy_provenance", provenance)
        digest = hashlib.sha256(MANIFEST_DIGEST_DOMAIN + canonicalize_restricted_jcs(_manifest_payload(self))).hexdigest()
        object.__setattr__(self, "manifest_digest", digest)

    @property
    def identity(self) -> tuple[str, str]:
        return self.capability_id, self.contract_version

    def canonical_payload(self) -> bytes:
        return canonicalize_restricted_jcs(_manifest_payload(self))


def _argument_projection(arg: ToolArg) -> Mapping[str, Any]:
    return _legacy_argument(arg)


def legacy_tool_to_manifest(tool: Tool) -> CapabilityManifest:
    """Project legacy metadata to immutable descriptive data; retain no Tool."""
    if not isinstance(tool, Tool):
        raise TypeError("Expected a legacy Tool definition.")
    capability_id = LEGACY_CAPABILITY_IDS.get(tool.name)
    if capability_id is None:
        raise ManifestValidationError(f"No explicit capability ID mapping for {tool.name!r}.")

    findings = [{"code": "observed_legacy", "evidence": "Projected from the current legacy Tool/ToolArg definition; not an approved stable contract."},
                {"code": "incomplete_output", "evidence": "ToolResult.output is text and ToolResult.data is optional untyped data."}]
    findings.extend({"code": code, "evidence": evidence} for code, evidence in _FINDINGS.get(tool.name, ()))
    if tool.permission == "disabled" and not any(code == "disabled_placeholder" for code, _ in _FINDINGS.get(tool.name, ())):
        findings.append({"code": "disabled_placeholder", "evidence": "Current registry policy marks this definition disabled."})

    availability = {
        "registered_observation": True,
        "legacy_prompt_visible": tool.permission != "disabled",
        "legacy_risk_label": tool.risk,
        "legacy_permission_label": tool.permission,
        "daemon_without_approval_callback": "unavailable" if tool.permission == "approval" else "not_approval_gated",
    }
    provenance = {
        "provider_module": getattr(tool.execute, "__module__", None),
        "provider_digest_status": "unverified",
        "provider_integrity_digest": None,
        "legacy_tool_name": tool.name,
    }
    consumer_exposure = (
        "prohibited_from_consumer_exposure"
        if tool.name == "screen_capture" or tool.permission == "disabled"
        else "unreviewed"
    )
    return CapabilityManifest(
        capability_id=capability_id,
        contract_version=LEGACY_CONTRACT_VERSION,
        display_name=tool.name,
        summary=tool.description,
        input_arguments=tuple(_argument_projection(arg) for arg in tool.args),
        output_contract={
            "kind": "observed_legacy_tool_result",
            "fields": {"success": "boolean", "output": "text", "error": "optional text", "data": "optional untyped value"},
            "guarantees": [],
        },
        declared_effects=_EFFECTS.get(tool.name, ()),
        resource_bounds=_BOUNDS.get(tool.name, {}),
        findings=tuple(findings),
        availability_observations=availability,
        legacy_provenance=provenance,
        consumer_exposure=consumer_exposure,
    )


def provider_artifact_digest(artifact: bytes) -> str:
    """Digest fixed artifact bytes; this does not define current provider scope."""
    if not isinstance(artifact, bytes):
        raise TypeError("Provider artifact must be bytes.")
    return hashlib.sha256(PROVIDER_DIGEST_DOMAIN + artifact).hexdigest()


def verify_provider_artifact_digest(expected: str, artifact: bytes) -> bool:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ManifestValidationError("Expected provider digest must be lowercase SHA-256 hex.")
    return provider_artifact_digest(artifact) == expected


class CapabilityRegistry:
    """Instance-owned immutable-manifest catalog; never dispatches providers."""
    def __init__(self):
        self._manifests: dict[tuple[str, str], CapabilityManifest] = {}

    def register(self, manifest: CapabilityManifest) -> None:
        if not isinstance(manifest, CapabilityManifest):
            raise TypeError("CapabilityRegistry accepts CapabilityManifest values only.")
        key = manifest.identity
        if key in self._manifests:
            raise DuplicateCapabilityError(f"Capability {key[0]} version {key[1]} is already registered.")
        self._manifests[key] = manifest

    def get(self, capability_id: str, contract_version: str) -> CapabilityManifest:
        key = (capability_id, contract_version)
        try:
            return self._manifests[key]
        except KeyError as exc:
            raise KeyError(f"Unknown capability identity/version: {capability_id}@{contract_version}") from exc

    def all(self) -> tuple[CapabilityManifest, ...]:
        return tuple(self._manifests[key] for key in sorted(self._manifests))

    def snapshot(self) -> Mapping[tuple[str, str], CapabilityManifest]:
        return MappingProxyType(dict(self._manifests))


def build_legacy_capability_registry(tool_registry) -> CapabilityRegistry:
    """Copy a ToolRegistry inventory into a fresh manifest-only catalog."""
    catalog = CapabilityRegistry()
    for tool in tool_registry.all():
        catalog.register(legacy_tool_to_manifest(tool))
    return catalog
