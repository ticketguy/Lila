"""Typed, immutable, serializable models for Command Center registry data."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkstationConfiguration:
    windows_root: str
    ubuntu_distribution: str
    ubuntu_mount: str
    ubuntu_home: str
    ubuntu_workspace_root: str

    def to_dict(self) -> dict[str, str]:
        return {
            "windows_root": self.windows_root,
            "ubuntu_distribution": self.ubuntu_distribution,
            "ubuntu_mount": self.ubuntu_mount,
            "ubuntu_home": self.ubuntu_home,
            "ubuntu_workspace_root": self.ubuntu_workspace_root,
        }


@dataclass(frozen=True)
class WorkstationCategory:
    id: str
    path: str
    purpose: str
    type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "purpose": self.purpose,
            "type": self.type,
        }


@dataclass(frozen=True)
class WorkspaceProject:
    name: str
    category: str
    canonical_path: str
    git_root: str | None = None
    environment: str | None = None
    windows_scaffold: str | None = None
    windows_scaffold_is_canonical: bool | None = None
    status: str | None = None
    aliases: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "canonical_path": self.canonical_path,
        }
        optional_values = {
            "git_root": self.git_root,
            "environment": self.environment,
            "windows_scaffold": self.windows_scaffold,
            "windows_scaffold_is_canonical": self.windows_scaffold_is_canonical,
            "status": self.status,
            "aliases": list(self.aliases) if self.aliases is not None else None,
        }
        values.update(
            (key, value)
            for key, value in optional_values.items()
            if value is not None
        )
        return values


@dataclass(frozen=True)
class IntakeSettings:
    inbox: str
    needs_review: str
    processed: str
    quarantine: str
    initial_watch_sources: tuple[str, ...]
    auto_route_confidence: float
    review_confidence: float
    overwrite_allowed: bool
    move_log_required: bool
    watch_downloads: bool
    watch_desktop: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "inbox": self.inbox,
            "needs_review": self.needs_review,
            "processed": self.processed,
            "quarantine": self.quarantine,
            "initial_watch_sources": list(self.initial_watch_sources),
            "auto_route_confidence": self.auto_route_confidence,
            "review_confidence": self.review_confidence,
            "overwrite_allowed": self.overwrite_allowed,
            "move_log_required": self.move_log_required,
            "watch_downloads": self.watch_downloads,
            "watch_desktop": self.watch_desktop,
        }


@dataclass(frozen=True)
class SoftwareSettings:
    incoming: str
    windows_installers: str
    linux_installers: str
    catalog: str
    automatic_installation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "incoming": self.incoming,
            "windows_installers": self.windows_installers,
            "linux_installers": self.linux_installers,
            "catalog": self.catalog,
            "automatic_installation": self.automatic_installation,
        }


@dataclass(frozen=True)
class TaxonomySettings:
    extensible: bool
    next_available_category_number: int
    create_category_on_single_ambiguous_file: bool
    record_every_new_category: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "extensible": self.extensible,
            "next_available_category_number": self.next_available_category_number,
            "create_category_on_single_ambiguous_file": (
                self.create_category_on_single_ambiguous_file
            ),
            "record_every_new_category": self.record_every_new_category,
        }


@dataclass(frozen=True)
class WorkstationOverview:
    schema_version: int
    last_inventory: str
    workstation: WorkstationConfiguration
    categories: tuple[WorkstationCategory, ...]
    workspaces: tuple[WorkspaceProject, ...]
    intake: IntakeSettings
    software: SoftwareSettings
    taxonomy: TaxonomySettings

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "last_inventory": self.last_inventory,
            "workstation": self.workstation.to_dict(),
            "categories": [category.to_dict() for category in self.categories],
            "workspaces": [workspace.to_dict() for workspace in self.workspaces],
            "intake": self.intake.to_dict(),
            "software": self.software.to_dict(),
            "taxonomy": self.taxonomy.to_dict(),
        }
