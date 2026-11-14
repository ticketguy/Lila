import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from src.workstation import (
    AmbiguousProjectError,
    ProjectNotFoundError,
    RegistryNotFoundError,
    RegistryPathError,
    RegistryValidationError,
    UnsupportedSchemaVersionError,
    WorkstationRegistry,
)
from src.workstation.registry import PRODUCTION_REGISTRY_PATH
import src.workstation.registry as registry_module


def valid_document() -> dict:
    return {
        "schema_version": 1,
        "last_inventory": "2026-11-06",
        "workstation": {
            "windows_root": "C:\\Workstation",
            "ubuntu_distribution": "Ubuntu-24.04",
            "ubuntu_mount": "/mnt/c/Workstation",
            "ubuntu_home": "/home/ticket",
            "ubuntu_workspace_root": "/home/ticket/workspaces",
        },
        "categories": [
            {
                "id": "01_ai_operator",
                "path": "01_AI_Operator",
                "purpose": "Assistants",
                "type": "domain",
            }
        ],
        "workspaces": [
            {
                "name": "Lila",
                "category": "01_ai_operator",
                "canonical_path": "/home/ticket/workspaces/lila",
                "git_root": "/home/ticket/workspaces/lila/source",
                "environment": "/home/ticket/workspaces/lila/environment/.venv",
            }
        ],
        "intake": {
            "inbox": "00_Command_Center\\Intake\\Inbox",
            "needs_review": "00_Command_Center\\Intake\\Needs_Review",
            "processed": "00_Command_Center\\Intake\\Processed",
            "quarantine": "00_Command_Center\\Intake\\Quarantine",
            "initial_watch_sources": ["00_Command_Center\\Intake\\Inbox"],
            "auto_route_confidence": 0.90,
            "review_confidence": 0.65,
            "overwrite_allowed": False,
            "move_log_required": True,
            "watch_downloads": False,
            "watch_desktop": False,
        },
        "software": {
            "incoming": "00_Command_Center\\Software\\Incoming",
            "windows_installers": "00_Command_Center\\Software\\Installers\\Windows",
            "linux_installers": "00_Command_Center\\Software\\Installers\\Linux",
            "catalog": "00_Command_Center\\Software\\Catalog",
            "automatic_installation": False,
        },
        "taxonomy": {
            "extensible": True,
            "next_available_category_number": 12,
            "create_category_on_single_ambiguous_file": False,
            "record_every_new_category": True,
        },
    }


class WorkstationRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.fixture_path = Path(self.temp_dir.name) / "PROJECT_REGISTRY.json"

    def write_fixture(self, document: dict | str) -> WorkstationRegistry:
        content = document if isinstance(document, str) else json.dumps(document)
        self.fixture_path.write_text(content, encoding="utf-8")
        return WorkstationRegistry._for_testing(self.fixture_path)

    def test_valid_registry_loads_and_returns_categories(self) -> None:
        registry = self.write_fixture(valid_document())
        overview = registry.load()

        self.assertEqual(overview.schema_version, 1)
        self.assertEqual(len(registry.list_categories()), 1)
        self.assertEqual(overview.to_dict()["workspaces"][0]["name"], "Lila")

    def test_lila_paths_are_resolved(self) -> None:
        registry = self.write_fixture(valid_document())
        registry.load()

        self.assertEqual(
            registry.resolve_project_path("Lila"),
            "/home/ticket/workspaces/lila",
        )
        self.assertEqual(
            registry.resolve_git_root("lila"),
            "/home/ticket/workspaces/lila/source",
        )
        self.assertEqual(
            registry.resolve_environment_path("Lila"),
            "/home/ticket/workspaces/lila/environment/.venv",
        )

    def test_unknown_project_has_clear_result(self) -> None:
        registry = self.write_fixture(valid_document())
        registry.load()

        self.assertIsNone(registry.find_project("unknown"))
        with self.assertRaisesRegex(ProjectNotFoundError, "No registered workspace"):
            registry.get_project("unknown")

    def test_duplicate_category_ids_are_rejected(self) -> None:
        document = valid_document()
        document["categories"].append(dict(document["categories"][0], path="another"))
        with self.assertRaisesRegex(RegistryValidationError, "Duplicate category id"):
            self.write_fixture(document).load()

    def test_duplicate_category_paths_are_rejected_case_insensitively(self) -> None:
        document = valid_document()
        document["categories"].append(
            {
                "id": "another",
                "path": "01_ai_operator",
                "purpose": "Duplicate path",
                "type": "domain",
            }
        )
        with self.assertRaisesRegex(RegistryValidationError, "Duplicate category path"):
            self.write_fixture(document).load()

    def test_invalid_workspace_category_reference_is_rejected(self) -> None:
        document = valid_document()
        document["workspaces"][0]["category"] = "missing"
        with self.assertRaisesRegex(RegistryValidationError, "unknown category"):
            self.write_fixture(document).load()

    def test_malformed_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(RegistryValidationError, "valid UTF-8 JSON"):
            self.write_fixture("{malformed").load()

    def test_unsupported_schema_version_is_rejected(self) -> None:
        document = valid_document()
        document["schema_version"] = 2
        with self.assertRaises(UnsupportedSchemaVersionError):
            self.write_fixture(document).load()

    def test_missing_registry_is_explicit(self) -> None:
        missing_path = Path(self.temp_dir.name) / "missing.json"
        with self.assertRaises(RegistryNotFoundError):
            WorkstationRegistry._for_testing(missing_path)

    def test_runtime_constructor_does_not_accept_paths(self) -> None:
        with self.assertRaises(TypeError):
            WorkstationRegistry(self.fixture_path)

    def test_test_fixture_cannot_escape_temporary_directory(self) -> None:
        with self.assertRaises(RegistryPathError):
            WorkstationRegistry._for_testing(PRODUCTION_REGISTRY_PATH)

    def test_test_fixture_symlink_cannot_escape_temporary_directory(self) -> None:
        link = Path(self.temp_dir.name) / "linked.json"
        try:
            link.symlink_to(PRODUCTION_REGISTRY_PATH)
        except (NotImplementedError, OSError):
            self.skipTest("Symlinks are unavailable in this environment")

        with self.assertRaises(RegistryPathError):
            WorkstationRegistry._for_testing(link)

    def test_project_paths_cannot_escape_workspace_root(self) -> None:
        document = valid_document()
        document["workspaces"][0]["canonical_path"] = "/home/ticket/elsewhere"
        with self.assertRaises(RegistryPathError):
            self.write_fixture(document).load()

    def test_registry_cannot_replace_trusted_workspace_root(self) -> None:
        for root in ("/", "/etc", "/home"):
            with self.subTest(root=root):
                document = valid_document()
                document["workstation"]["ubuntu_workspace_root"] = root
                with self.assertRaises(RegistryPathError):
                    self.write_fixture(document).load()

    def test_canonical_path_outside_trusted_workspace_is_rejected(self) -> None:
        document = valid_document()
        document["workspaces"][0]["canonical_path"] = "/home/ticket/elsewhere"
        with self.assertRaises(RegistryPathError):
            self.write_fixture(document).load()

    def test_lexical_traversal_is_rejected(self) -> None:
        document = valid_document()
        document["workspaces"][0]["canonical_path"] = (
            "/home/ticket/workspaces/../elsewhere"
        )
        with self.assertRaises(RegistryValidationError):
            self.write_fixture(document).load()

    def test_git_root_must_be_inside_owning_project(self) -> None:
        document = valid_document()
        document["workspaces"][0]["git_root"] = "/home/ticket/workspaces/other/source"
        with self.assertRaises(RegistryPathError):
            self.write_fixture(document).load()

    def test_environment_must_be_inside_owning_project(self) -> None:
        document = valid_document()
        document["workspaces"][0]["environment"] = (
            "/home/ticket/workspaces/other/environment"
        )
        with self.assertRaises(RegistryPathError):
            self.write_fixture(document).load()

    def test_symlink_canonical_path_cannot_escape_trusted_root(self) -> None:
        trusted_root = Path(self.temp_dir.name) / "trusted"
        trusted_root.mkdir()
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        project_link = trusted_root / "lila"
        try:
            project_link.symlink_to(outside, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Symlinks are unavailable in this environment")

        document = valid_document()
        document["workstation"]["ubuntu_workspace_root"] = str(trusted_root)
        document["workspaces"][0].update(
            canonical_path=str(project_link),
            git_root=None,
            environment=None,
        )
        with patch.object(registry_module, "TRUSTED_WORKSPACE_ROOT", trusted_root):
            with self.assertRaises(RegistryPathError):
                self.write_fixture(document).load()

    def test_symlink_child_path_cannot_escape_owning_project(self) -> None:
        trusted_root = Path(self.temp_dir.name) / "trusted"
        project = trusted_root / "lila"
        project.mkdir(parents=True)
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        git_link = project / "source"
        try:
            git_link.symlink_to(outside, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Symlinks are unavailable in this environment")

        document = valid_document()
        document["workstation"]["ubuntu_workspace_root"] = str(trusted_root)
        document["workspaces"][0].update(
            canonical_path=str(project),
            git_root=str(git_link),
            environment=None,
        )
        with patch.object(registry_module, "TRUSTED_WORKSPACE_ROOT", trusted_root):
            with self.assertRaises(RegistryPathError):
                self.write_fixture(document).load()

    def test_ambiguous_names_are_reported(self) -> None:
        document = valid_document()
        document["workspaces"].append(
            {
                "name": "lila",
                "category": "01_ai_operator",
                "canonical_path": "/home/ticket/workspaces/other",
            }
        )
        registry = self.write_fixture(document)
        registry.load()
        with self.assertRaises(AmbiguousProjectError):
            registry.find_project("LILA")

    def test_no_write_operations_are_exposed(self) -> None:
        registry = WorkstationRegistry()
        exposed = [
            name
            for name in dir(registry)
            if any(word in name.lower() for word in ("write", "delete", "move", "rename"))
        ]
        self.assertEqual(exposed, [])

    def test_real_registry_loads_and_is_unchanged(self) -> None:
        if not PRODUCTION_REGISTRY_PATH.is_file():
            self.skipTest("Authoritative Windows Command Center registry is unavailable")
        before = hashlib.sha256(PRODUCTION_REGISTRY_PATH.read_bytes()).digest()
        registry = WorkstationRegistry()
        overview = registry.load()
        self.assertGreater(len(overview.categories), 0)
        self.assertEqual(registry.resolve_project_path("Lila"), "/home/ticket/workspaces/lila")
        after = hashlib.sha256(PRODUCTION_REGISTRY_PATH.read_bytes()).digest()
        self.assertEqual(before, after)

    def test_project_existence_is_separate_from_registry_validity(self) -> None:
        registry = self.write_fixture(valid_document())
        registry.load()
        with patch("src.workstation.registry.Path.is_dir", return_value=False):
            self.assertFalse(registry.project_exists("Lila"))
        self.assertEqual(registry.get_project("Lila").name, "Lila")

    def test_intake_configuration_is_preserved(self) -> None:
        overview = self.write_fixture(valid_document()).load()
        self.assertEqual(
            overview.to_dict()["intake"],
            valid_document()["intake"],
        )

    def test_software_configuration_is_preserved(self) -> None:
        overview = self.write_fixture(valid_document()).load()
        self.assertEqual(
            overview.to_dict()["software"],
            valid_document()["software"],
        )

    def test_taxonomy_configuration_is_preserved(self) -> None:
        overview = self.write_fixture(valid_document()).load()
        self.assertEqual(
            overview.to_dict()["taxonomy"],
            valid_document()["taxonomy"],
        )

    def test_unknown_registry_fields_are_rejected(self) -> None:
        document = valid_document()
        document["new_configuration"] = {"unrecognized": True}
        with self.assertRaisesRegex(RegistryValidationError, "unsupported field"):
            self.write_fixture(document).load()

    def test_returned_models_are_deeply_immutable(self) -> None:
        overview = self.write_fixture(valid_document()).load()
        with self.assertRaises(FrozenInstanceError):
            overview.workstation.ubuntu_workspace_root = "/"
        with self.assertRaises(FrozenInstanceError):
            overview.intake.inbox = "changed"
        with self.assertRaises(FrozenInstanceError):
            overview.categories[0].id = "changed"
        with self.assertRaises(TypeError):
            overview.intake.initial_watch_sources[0] = "changed"

    def test_to_dict_returns_defensive_copies(self) -> None:
        overview = self.write_fixture(valid_document()).load()
        serialized = overview.to_dict()
        serialized["workstation"]["ubuntu_workspace_root"] = "/"
        serialized["intake"]["initial_watch_sources"].append("changed")
        serialized["workspaces"][0]["name"] = "changed"
        self.assertEqual(
            overview.workstation.ubuntu_workspace_root,
            "/home/ticket/workspaces",
        )
        self.assertEqual(
            overview.intake.initial_watch_sources,
            ("00_Command_Center\\Intake\\Inbox",),
        )
        self.assertEqual(overview.workspaces[0].name, "Lila")


if __name__ == "__main__":
    unittest.main()
