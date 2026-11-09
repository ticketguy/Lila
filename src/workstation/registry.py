"""Read-only reader for the authoritative Workstation Command Center registry."""

import json
import math
import os
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from .models import (
    IntakeSettings,
    SoftwareSettings,
    TaxonomySettings,
    WorkstationCategory,
    WorkstationConfiguration,
    WorkstationOverview,
    WorkspaceProject,
)

COMMAND_CENTER_ROOT = Path("/mnt/c/Workstation/00_Command_Center")
PRODUCTION_REGISTRY_PATH = COMMAND_CENTER_ROOT / "PROJECT_REGISTRY.json"
TRUSTED_UBUNTU_HOME = Path("/home/ticket")
TRUSTED_WORKSPACE_ROOT = TRUSTED_UBUNTU_HOME / "workspaces"
SUPPORTED_SCHEMA_VERSION = 1


class RegistryError(Exception):
    """Base exception for registry access and validation failures."""


class RegistryNotFoundError(RegistryError):
    """Raised when the configured registry file is missing."""


class RegistryPathError(RegistryError):
    """Raised when a registry or project path escapes its allowed boundary."""


class RegistryValidationError(RegistryError):
    """Raised when registry contents are malformed or inconsistent."""


class UnsupportedSchemaVersionError(RegistryValidationError):
    """Raised when the registry uses an unsupported schema version."""


class ProjectNotFoundError(RegistryError):
    """Raised when a requested workspace does not exist in the registry."""


class AmbiguousProjectError(RegistryError):
    """Raised when a name matches multiple workspaces."""


class RegistryNotLoadedError(RegistryError):
    """Raised when registry data is requested before load()."""


class WorkstationRegistry:
    """Load and query the fixed Command Center registry without modifying it.

    Production callers cannot provide a registry path. The private
    ``_for_testing`` factory is limited to registry fixtures in the system
    temporary directory and does not change the trusted deployment boundary.
    """

    def __init__(self) -> None:
        self._registry_path = PRODUCTION_REGISTRY_PATH
        self._overview: WorkstationOverview | None = None

    @classmethod
    def _for_testing(
        cls, fixture_path: str | os.PathLike[str]
    ) -> "WorkstationRegistry":
        temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
        candidate = Path(fixture_path)
        try:
            resolved_path = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RegistryNotFoundError(
                f"Fixture registry not found: {candidate}"
            ) from exc
        if not resolved_path.is_file() or not _is_within(resolved_path, temp_root):
            raise RegistryPathError(
                "Fixture registry must resolve to a file within the system "
                f"temporary directory: {candidate}"
            )
        registry = cls()
        registry._registry_path = resolved_path
        return registry

    def load(self) -> WorkstationOverview:
        registry_path = self._get_safe_registry_path()
        try:
            raw_bytes = registry_path.read_bytes()
        except FileNotFoundError as exc:
            raise RegistryNotFoundError(
                f"Workstation registry not found: {registry_path}"
            ) from exc
        except OSError as exc:
            raise RegistryError(
                f"Unable to read Workstation registry {registry_path}: {exc}"
            ) from exc

        try:
            document = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryValidationError(
                f"Workstation registry is not valid UTF-8 JSON: {registry_path}"
            ) from exc

        overview = _parse_overview(document)
        self._overview = overview
        return overview

    def get_overview(self) -> WorkstationOverview:
        return self._require_overview()

    def list_categories(self) -> tuple[WorkstationCategory, ...]:
        return self._require_overview().categories

    def find_project(self, name: str) -> WorkspaceProject | None:
        if not isinstance(name, str) or not name.strip():
            return None
        target = _normalise_name(name)
        matches = [
            project
            for project in self._require_overview().workspaces
            if target == _normalise_name(project.name)
            or any(target == _normalise_name(alias) for alias in project.aliases or ())
        ]
        if len(matches) > 1:
            names = ", ".join(project.name for project in matches)
            raise AmbiguousProjectError(
                f"Project name {name!r} matches multiple workspaces: {names}"
            )
        return matches[0] if matches else None

    def get_project(self, name: str) -> WorkspaceProject:
        project = self.find_project(name)
        if project is None:
            raise ProjectNotFoundError(f"No registered workspace matches {name!r}")
        return project

    def resolve_project_path(self, name: str) -> str:
        project = self.get_project(name)
        return _resolve_project_path(project.canonical_path)

    def resolve_git_root(self, name: str) -> str | None:
        project = self.get_project(name)
        if project.git_root is None:
            return None
        return _resolve_project_child_path(
            project.git_root, project.canonical_path
        )

    def resolve_environment_path(self, name: str) -> str | None:
        project = self.get_project(name)
        if project.environment is None:
            return None
        return _resolve_project_child_path(
            project.environment, project.canonical_path
        )

    def project_exists(self, name: str) -> bool:
        """Report directory existence separately from registry ownership."""
        return Path(self.resolve_project_path(name)).is_dir()

    def _get_safe_registry_path(self) -> Path:
        if self._registry_path != PRODUCTION_REGISTRY_PATH:
            temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
            try:
                resolved = self._registry_path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise RegistryNotFoundError(
                    f"Fixture registry not found: {self._registry_path}"
                ) from exc
            if not resolved.is_file() or not _is_within(resolved, temp_root):
                raise RegistryPathError(
                    "Injected registry path is not a file within the system "
                    "temporary directory"
                )
            return resolved

        try:
            root = COMMAND_CENTER_ROOT.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RegistryNotFoundError(
                f"Command Center directory not found: {COMMAND_CENTER_ROOT}"
            ) from exc
        try:
            resolved = PRODUCTION_REGISTRY_PATH.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RegistryNotFoundError(
                f"Workstation registry not found: {PRODUCTION_REGISTRY_PATH}"
            ) from exc
        if resolved.parent != root or not resolved.is_file():
            raise RegistryPathError(
                "The production registry must be a regular file directly within "
                f"{root}"
            )
        return resolved

    def _require_overview(self) -> WorkstationOverview:
        if self._overview is None:
            raise RegistryNotLoadedError("Call load() before querying the registry")
        return self._overview


def _parse_overview(document: Any) -> WorkstationOverview:
    root = _require_object(document, "registry")
    _reject_unknown_fields(
        root,
        {
            "schema_version",
            "last_inventory",
            "workstation",
            "categories",
            "workspaces",
            "intake",
            "software",
            "taxonomy",
        },
        "registry",
    )
    schema_version = _require_int(root, "schema_version", "registry")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Unsupported registry schema_version {schema_version}; "
            f"supported version is {SUPPORTED_SCHEMA_VERSION}"
        )
    last_inventory = _require_string(root, "last_inventory", "registry")

    workstation_data = _require_object(root.get("workstation"), "workstation")
    _reject_unknown_fields(
        workstation_data,
        {
            "windows_root",
            "ubuntu_distribution",
            "ubuntu_mount",
            "ubuntu_home",
            "ubuntu_workspace_root",
        },
        "workstation",
    )
    workstation = WorkstationConfiguration(
        windows_root=_require_string(workstation_data, "windows_root", "workstation"),
        ubuntu_distribution=_require_string(
            workstation_data, "ubuntu_distribution", "workstation"
        ),
        ubuntu_mount=_require_string(workstation_data, "ubuntu_mount", "workstation"),
        ubuntu_home=_require_string(workstation_data, "ubuntu_home", "workstation"),
        ubuntu_workspace_root=_require_string(
            workstation_data, "ubuntu_workspace_root", "workstation"
        ),
    )
    if workstation.ubuntu_home != TRUSTED_UBUNTU_HOME.as_posix():
        raise RegistryPathError(
            "workstation.ubuntu_home does not match the trusted deployment home"
        )
    if workstation.ubuntu_workspace_root != TRUSTED_WORKSPACE_ROOT.as_posix():
        raise RegistryPathError(
            "workstation.ubuntu_workspace_root does not match the trusted workspace root"
        )
    trusted_root = _get_trusted_workspace_root()

    category_data = _require_list(root, "categories", "registry")
    categories: list[WorkstationCategory] = []
    category_ids: set[str] = set()
    category_paths: set[str] = set()
    for index, value in enumerate(category_data):
        context = f"categories[{index}]"
        item = _require_object(value, context)
        _reject_unknown_fields(item, {"id", "path", "purpose", "type"}, context)
        category_id = _require_string(item, "id", context)
        category_path = _require_string(item, "path", context)
        purpose = _require_string(item, "purpose", context)
        category_type = _require_string(item, "type", context)
        if (
            "/" in category_path
            or "\\" in category_path
            or category_path in {".", ".."}
        ):
            raise RegistryValidationError(
                f"{context}.path must be a single relative category path"
            )
        if category_id.casefold() in category_ids:
            raise RegistryValidationError(f"Duplicate category id: {category_id!r}")
        normalised_category_path = category_path.casefold()
        if normalised_category_path in category_paths:
            raise RegistryValidationError(
                f"Duplicate category path: {category_path!r}"
            )
        category_ids.add(category_id.casefold())
        category_paths.add(normalised_category_path)
        categories.append(
            WorkstationCategory(category_id, category_path, purpose, category_type)
        )

    workspace_data = _require_list(root, "workspaces", "registry")
    workspaces: list[WorkspaceProject] = []
    for index, value in enumerate(workspace_data):
        context = f"workspaces[{index}]"
        item = _require_object(value, context)
        _reject_unknown_fields(
            item,
            {
                "name",
                "category",
                "canonical_path",
                "git_root",
                "environment",
                "windows_scaffold",
                "windows_scaffold_is_canonical",
                "status",
                "aliases",
            },
            context,
        )
        name = _require_string(item, "name", context)
        category = _require_string(item, "category", context)
        if category.casefold() not in category_ids:
            raise RegistryValidationError(
                f"{context}.category references unknown category {category!r}"
            )
        canonical_path = _require_string(item, "canonical_path", context)
        _resolve_project_path(canonical_path, trusted_root, context)

        git_root = _optional_string(item, "git_root", context)
        environment = _optional_string(item, "environment", context)
        if git_root is not None:
            _resolve_project_child_path(
                git_root, canonical_path, trusted_root, f"{context}.git_root"
            )
        if environment is not None:
            _resolve_project_child_path(
                environment,
                canonical_path,
                trusted_root,
                f"{context}.environment",
            )

        aliases_value = item.get("aliases")
        aliases: tuple[str, ...] | None = None
        if aliases_value is not None:
            if not isinstance(aliases_value, list) or not all(
                isinstance(alias, str) and alias.strip() for alias in aliases_value
            ):
                raise RegistryValidationError(
                    f"{context}.aliases must be a list of non-empty strings"
                )
            aliases = tuple(aliases_value)

        workspaces.append(
            WorkspaceProject(
                name=name,
                category=category,
                canonical_path=canonical_path,
                git_root=git_root,
                environment=environment,
                windows_scaffold=_optional_string(item, "windows_scaffold", context),
                windows_scaffold_is_canonical=_optional_bool(
                    item, "windows_scaffold_is_canonical", context
                ),
                status=_optional_string(item, "status", context),
                aliases=aliases,
            )
        )

    return WorkstationOverview(
        schema_version=schema_version,
        last_inventory=last_inventory,
        workstation=workstation,
        categories=tuple(categories),
        workspaces=tuple(workspaces),
        intake=_parse_intake(root.get("intake")),
        software=_parse_software(root.get("software")),
        taxonomy=_parse_taxonomy(root.get("taxonomy")),
    )


def _parse_intake(value: Any) -> IntakeSettings:
    data = _require_object(value, "intake")
    keys = {
        "inbox",
        "needs_review",
        "processed",
        "quarantine",
        "initial_watch_sources",
        "auto_route_confidence",
        "review_confidence",
        "overwrite_allowed",
        "move_log_required",
        "watch_downloads",
        "watch_desktop",
    }
    _reject_unknown_fields(data, keys, "intake")
    sources = _require_list(data, "initial_watch_sources", "intake")
    if not all(isinstance(source, str) and source.strip() for source in sources):
        raise RegistryValidationError(
            "intake.initial_watch_sources must contain non-empty strings"
        )
    auto_confidence = _require_probability(data, "auto_route_confidence", "intake")
    review_confidence = _require_probability(data, "review_confidence", "intake")
    return IntakeSettings(
        inbox=_require_string(data, "inbox", "intake"),
        needs_review=_require_string(data, "needs_review", "intake"),
        processed=_require_string(data, "processed", "intake"),
        quarantine=_require_string(data, "quarantine", "intake"),
        initial_watch_sources=tuple(sources),
        auto_route_confidence=auto_confidence,
        review_confidence=review_confidence,
        overwrite_allowed=_require_bool(data, "overwrite_allowed", "intake"),
        move_log_required=_require_bool(data, "move_log_required", "intake"),
        watch_downloads=_require_bool(data, "watch_downloads", "intake"),
        watch_desktop=_require_bool(data, "watch_desktop", "intake"),
    )


def _parse_software(value: Any) -> SoftwareSettings:
    data = _require_object(value, "software")
    _reject_unknown_fields(
        data,
        {
            "incoming",
            "windows_installers",
            "linux_installers",
            "catalog",
            "automatic_installation",
        },
        "software",
    )
    return SoftwareSettings(
        incoming=_require_string(data, "incoming", "software"),
        windows_installers=_require_string(data, "windows_installers", "software"),
        linux_installers=_require_string(data, "linux_installers", "software"),
        catalog=_require_string(data, "catalog", "software"),
        automatic_installation=_require_bool(
            data, "automatic_installation", "software"
        ),
    )


def _parse_taxonomy(value: Any) -> TaxonomySettings:
    data = _require_object(value, "taxonomy")
    _reject_unknown_fields(
        data,
        {
            "extensible",
            "next_available_category_number",
            "create_category_on_single_ambiguous_file",
            "record_every_new_category",
        },
        "taxonomy",
    )
    next_number = _require_int(data, "next_available_category_number", "taxonomy")
    if next_number < 0:
        raise RegistryValidationError(
            "taxonomy.next_available_category_number must not be negative"
        )
    return TaxonomySettings(
        extensible=_require_bool(data, "extensible", "taxonomy"),
        next_available_category_number=next_number,
        create_category_on_single_ambiguous_file=_require_bool(
            data, "create_category_on_single_ambiguous_file", "taxonomy"
        ),
        record_every_new_category=_require_bool(
            data, "record_every_new_category", "taxonomy"
        ),
    )


def _resolve_project_path(
    value: str,
    trusted_root: Path | None = None,
    context: str = "workspace.canonical_path",
) -> str:
    root = trusted_root if trusted_root is not None else _get_trusted_workspace_root()
    lexical_path = _validate_absolute_path(value, context)
    lexical_root = PurePosixPath(TRUSTED_WORKSPACE_ROOT.as_posix())
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise RegistryPathError(
            f"{context} is outside the trusted workspace root: {value!r}"
        ) from exc
    if not relative.parts:
        raise RegistryPathError(
            f"{context} must identify a project below the trusted workspace root"
        )

    resolved_path = _resolve_path(value, context)
    if not _is_within(resolved_path, root) or resolved_path == root:
        raise RegistryPathError(
            f"{context} resolves outside or to the trusted workspace root: {value!r}"
        )
    return str(resolved_path)


def _resolve_project_child_path(
    value: str,
    canonical_path: str,
    trusted_root: Path | None = None,
    context: str = "workspace child path",
) -> str:
    root = trusted_root if trusted_root is not None else _get_trusted_workspace_root()
    lexical_path = _validate_absolute_path(value, context)
    lexical_project = _validate_absolute_path(
        canonical_path, "workspace.canonical_path"
    )
    try:
        lexical_path.relative_to(lexical_project)
    except ValueError as exc:
        raise RegistryPathError(
            f"{context} is outside its owning canonical project: {value!r}"
        ) from exc

    resolved_project = _resolve_project_path(
        canonical_path, root, "workspace.canonical_path"
    )
    resolved_path = _resolve_path(value, context)
    if not _is_within(resolved_path, root) or not _is_within(
        resolved_path, Path(resolved_project)
    ):
        raise RegistryPathError(
            f"{context} resolves outside its owning canonical project: {value!r}"
        )
    return str(resolved_path)


def _validate_absolute_path(value: str, context: str) -> PurePosixPath:
    candidate = PurePosixPath(value)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise RegistryValidationError(
            f"{context} must be an absolute path without '..': {value!r}"
        )
    return candidate


def _get_trusted_workspace_root() -> Path:
    try:
        return TRUSTED_WORKSPACE_ROOT.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RegistryPathError(
            f"Trusted workspace root is unavailable: {TRUSTED_WORKSPACE_ROOT}"
        ) from exc


def _resolve_path(value: str, context: str) -> Path:
    try:
        return Path(value).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise RegistryPathError(f"Unable to safely resolve {context}: {value!r}") from exc


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalise_name(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def _reject_unknown_fields(
    data: dict[str, Any], allowed_fields: set[str], context: str
) -> None:
    unknown = set(data) - allowed_fields
    if unknown:
        fields = ", ".join(sorted(str(field) for field in unknown))
        raise RegistryValidationError(f"{context} has unsupported field(s): {fields}")


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegistryValidationError(f"{context} must be an object")
    return value


def _require_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _optional_string(data: dict[str, Any], key: str, context: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _require_bool(data: dict[str, Any], key: str, context: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise RegistryValidationError(f"{context}.{key} must be a boolean")
    return value


def _optional_bool(data: dict[str, Any], key: str, context: str) -> bool | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise RegistryValidationError(f"{context}.{key} must be a boolean")
    return value


def _require_int(data: dict[str, Any], key: str, context: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegistryValidationError(f"{context}.{key} must be an integer")
    return value


def _require_probability(data: dict[str, Any], key: str, context: str) -> float:
    value = data.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise RegistryValidationError(
            f"{context}.{key} must be a finite number between 0 and 1"
        )
    return float(value)


def _require_list(data: dict[str, Any], key: str, context: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise RegistryValidationError(f"{context}.{key} must be a list")
    return value
