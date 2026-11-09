"""Read-only, bounded Harness tools for the authoritative workstation registry."""

from __future__ import annotations

import json
from typing import Any

from src.harness.tools import Tool, ToolArg, ToolResult, register_tool
from .models import WorkstationCategory, WorkspaceProject
from .registry import RegistryError, WorkstationRegistry

MAX_OUTPUT_CHARS = 8_000
MAX_RETURNED_ITEMS = 100
MAX_FIELD_CHARS = 512


def _bounded_text(value: str) -> str:
    if len(value) <= MAX_FIELD_CHARS:
        return value
    return value[:MAX_FIELD_CHARS] + "... [truncated]"


def _encode_result(
    payload: dict[str, Any], collection_key: str | None = None
) -> ToolResult:
    """Serialize bounded JSON and trim collection responses if necessary."""
    items = payload.get(collection_key, []) if collection_key else []
    truncated = len(items) > MAX_RETURNED_ITEMS
    if collection_key:
        payload[collection_key] = items[:MAX_RETURNED_ITEMS]
        payload["returned_count"] = len(payload[collection_key])
        payload["truncated"] = truncated

    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    while (
        len(encoded) > MAX_OUTPUT_CHARS
        and collection_key
        and payload[collection_key]
    ):
        payload[collection_key].pop()
        payload["returned_count"] = len(payload[collection_key])
        payload["truncated"] = True
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    if len(encoded) > MAX_OUTPUT_CHARS:
        return ToolResult(
            success=False,
            output="",
            error="Workstation registry response exceeds the output limit.",
        )
    return ToolResult(success=True, output=encoded)


def _registry_or_error() -> tuple[WorkstationRegistry | None, ToolResult | None]:
    try:
        registry = WorkstationRegistry()
        registry.load()
    except RegistryError as exc:
        detail = str(exc)[:300]
        return None, ToolResult(
            success=False,
            output="",
            error=f"Workstation registry is unavailable or invalid: {detail}",
        )
    return registry, None


def _category_record(category: WorkstationCategory) -> dict[str, str]:
    return {
        "id": _bounded_text(category.id),
        "path": _bounded_text(category.path),
        "purpose": _bounded_text(category.purpose),
        "type": _bounded_text(category.type),
    }


def _workspace_record(
    registry: WorkstationRegistry, project: WorkspaceProject
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": _bounded_text(project.name),
        "category": _bounded_text(project.category),
        "canonical_path": _bounded_text(
            registry.resolve_project_path(project.name)
        ),
    }
    git_root = registry.resolve_git_root(project.name)
    if git_root is not None:
        record["git_root"] = _bounded_text(git_root)
    environment = registry.resolve_environment_path(project.name)
    if environment is not None:
        record["environment"] = _bounded_text(environment)
    if project.windows_scaffold is not None:
        record["windows_scaffold"] = _bounded_text(project.windows_scaffold)
    if project.windows_scaffold_is_canonical is not None:
        record["windows_scaffold_is_canonical"] = (
            project.windows_scaffold_is_canonical
        )
    if project.status is not None:
        record["status"] = _bounded_text(project.status)
    if project.aliases is not None:
        record["aliases"] = [
            _bounded_text(alias) for alias in project.aliases[:MAX_RETURNED_ITEMS]
        ]
    return record


def _workstation_overview() -> ToolResult:
    registry, error = _registry_or_error()
    if error is not None:
        return error
    assert registry is not None
    overview = registry.get_overview()
    payload = {
        "description": (
            "Registered Command Center organization. Registry entries describe "
            "declared configuration and do not confirm current filesystem or "
            "application presence."
        ),
        "last_inventory": _bounded_text(overview.last_inventory),
        "workstation": {
            key: _bounded_text(value)
            for key, value in overview.workstation.to_dict().items()
        },
        "category_count": len(overview.categories),
        "workspace_count": len(overview.workspaces),
        "categories": [
            _category_record(category) for category in registry.list_categories()
        ],
    }
    return _encode_result(payload, "categories")


def _workstation_list_categories() -> ToolResult:
    registry, error = _registry_or_error()
    if error is not None:
        return error
    assert registry is not None
    categories = registry.list_categories()
    payload = {
        "category_count": len(categories),
        "categories": [_category_record(category) for category in categories],
    }
    return _encode_result(payload, "categories")


def _workstation_list_workspaces() -> ToolResult:
    registry, error = _registry_or_error()
    if error is not None:
        return error
    assert registry is not None
    overview = registry.get_overview()
    payload = {
        "workspace_count": len(overview.workspaces),
        "workspaces": [
            _workspace_record(registry, project)
            for project in overview.workspaces
        ],
    }
    return _encode_result(payload, "workspaces")


def _workstation_get_workspace(identifier: str) -> ToolResult:
    if not identifier.strip() or len(identifier) > 128:
        return ToolResult(
            success=False,
            output="",
            error="Workspace identifier must be a non-empty registered name.",
        )
    registry, error = _registry_or_error()
    if error is not None:
        return error
    assert registry is not None
    try:
        project = registry.get_project(identifier)
        record = _workspace_record(registry, project)
    except RegistryError as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Unable to resolve registered workspace: {str(exc)[:300]}",
        )
    return _encode_result({"workspace": record})


def _workstation_intake_overview() -> ToolResult:
    registry, error = _registry_or_error()
    if error is not None:
        return error
    assert registry is not None
    intake = registry.get_overview().intake
    directories = [
        {
            "purpose": purpose,
            "configured_location": _bounded_text(value),
        }
        for purpose, value in (
            ("Incoming files awaiting classification", intake.inbox),
            ("Files awaiting human review", intake.needs_review),
            ("Files recorded as processed", intake.processed),
            ("Quarantined files", intake.quarantine),
        )
    ]
    return _encode_result(
        {
            "directories": directories,
            "initial_watch_sources": [
                _bounded_text(source)
                for source in intake.initial_watch_sources[:MAX_RETURNED_ITEMS]
            ],
            "auto_route_confidence": intake.auto_route_confidence,
            "review_confidence": intake.review_confidence,
            "overwrite_allowed": intake.overwrite_allowed,
            "move_log_required": intake.move_log_required,
            "watch_downloads": intake.watch_downloads,
            "watch_desktop": intake.watch_desktop,
            "note": "Configuration only; this tool does not inspect or process files.",
        }
    )


def _workstation_software_overview() -> ToolResult:
    registry, error = _registry_or_error()
    if error is not None:
        return error
    assert registry is not None
    software = registry.get_overview().software
    return _encode_result(
        {
            "locations": {
                "incoming": _bounded_text(software.incoming),
                "windows_installers": _bounded_text(
                    software.windows_installers
                ),
                "linux_installers": _bounded_text(software.linux_installers),
                "catalog": _bounded_text(software.catalog),
            },
            "automatic_installation": software.automatic_installation,
            "note": (
                "Configuration only; this tool does not inspect, install, or "
                "uninstall software."
            ),
        }
    )


def register_workstation_tools() -> None:
    """Register registry-backed metadata queries with the central harness."""
    definitions = (
        Tool(
            name="workstation_overview",
            description=(
                "Show registered workstation configuration, category structure, "
                "and category/workspace counts; does not verify live filesystem "
                "contents."
            ),
            args=[],
            execute=_workstation_overview,
            category="workstation",
        ),
        Tool(
            name="workstation_list_categories",
            description=(
                "List registered categories with their declared locations, "
                "purposes, and types."
            ),
            args=[],
            execute=_workstation_list_categories,
            category="workstation",
        ),
        Tool(
            name="workstation_list_workspaces",
            description=(
                "List registered workspace names, categories, canonical paths, "
                "and available metadata; does not assert that paths exist."
            ),
            args=[],
            execute=_workstation_list_workspaces,
            category="workstation",
        ),
        Tool(
            name="workstation_get_workspace",
            description=(
                "Get one registered workspace by its registry name (the schema "
                "has no separate workspace identifier); never accepts a path."
            ),
            args=[
                ToolArg(
                    "identifier",
                    "string",
                    "Registered workspace name, such as Lila",
                )
            ],
            execute=_workstation_get_workspace,
            category="workstation",
        ),
        Tool(
            name="workstation_intake_overview",
            description=(
                "Show configured file-intake locations and intake settings only."
            ),
            args=[],
            execute=_workstation_intake_overview,
            category="workstation",
        ),
        Tool(
            name="workstation_software_overview",
            description=(
                "Show configured software incoming, installer, and catalogue "
                "locations only; does not manage software."
            ),
            args=[],
            execute=_workstation_software_overview,
            category="workstation",
        ),
    )
    for tool in definitions:
        register_tool(tool)
