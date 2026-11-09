import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.lilacore import LilaCore
from src.harness.executor import HarnessExecutor
from src.harness.permissions import policy_for
from src.harness.tools import get_registry
from src.workstation import WorkstationRegistry
from src.workstation.registry import PRODUCTION_REGISTRY_PATH
from src.workstation.tools import (
    MAX_OUTPUT_CHARS,
    register_workstation_tools,
)


WORKSTATION_TOOL_NAMES = {
    "workstation_overview",
    "workstation_list_categories",
    "workstation_list_workspaces",
    "workstation_get_workspace",
    "workstation_intake_overview",
    "workstation_software_overview",
}


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
                "purpose": "AI assistant workspaces",
                "type": "domain",
            },
            {
                "id": "02_development",
                "path": "02_Development",
                "purpose": "Development projects",
                "type": "workspace",
            },
        ],
        "workspaces": [
            {
                "name": "Lila",
                "category": "01_ai_operator",
                "canonical_path": "/home/ticket/workspaces/lila",
                "git_root": "/home/ticket/workspaces/lila/source",
                "environment": "/home/ticket/workspaces/lila/environment/.venv",
                "status": "active",
            },
            {
                "name": "Development workspace",
                "category": "02_development",
                "canonical_path": "/home/ticket/workspaces/development",
                "status": "empty_at_last_inventory",
            },
        ],
        "intake": {
            "inbox": "00_Command_Center\\Intake\\Inbox",
            "needs_review": "00_Command_Center\\Intake\\Needs_Review",
            "processed": "00_Command_Center\\Intake\\Processed",
            "quarantine": "00_Command_Center\\Intake\\Quarantine",
            "initial_watch_sources": [
                "00_Command_Center\\Intake\\Inbox"
            ],
            "auto_route_confidence": 0.9,
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


def json_tool_call(name: str, arguments: dict | None = None) -> str:
    return "<tool_call>" + json.dumps(
        {"name": name, "arguments": arguments or {}}
    ) + "</tool_call>"


class WorkstationToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool_registry = get_registry()
        self.previous_tools = dict(self.tool_registry._tools)
        register_workstation_tools()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.fixture_path = Path(self.temp_dir.name) / "PROJECT_REGISTRY.json"
        self.fixture_path.write_text(
            json.dumps(valid_document()), encoding="utf-8"
        )
        self.fixture_registry = WorkstationRegistry._for_testing(
            self.fixture_path
        )
        self.constructor_patch = patch(
            "src.workstation.tools.WorkstationRegistry",
            return_value=self.fixture_registry,
        )
        self.constructor_patch.start()
        self.addCleanup(self.constructor_patch.stop)
        self.addCleanup(self._restore_tools)

    def _restore_tools(self) -> None:
        self.tool_registry._tools.clear()
        self.tool_registry._tools.update(self.previous_tools)

    def executor(self) -> HarnessExecutor:
        return HarnessExecutor(
            log_dir=self.temp_dir.name,
            workspace_root=self.temp_dir.name,
        )

    def run_tool(self, name: str, arguments: dict | None = None):
        return self.executor().process(json_tool_call(name, arguments))[0]

    def decoded_output(self, result) -> dict:
        self.assertTrue(result["result"].success, result["result"].error)
        return json.loads(result["result"].output)

    def test_overview_returns_registered_structure_and_counts(self) -> None:
        result = self.run_tool("workstation_overview")
        payload = self.decoded_output(result)

        self.assertEqual(payload["category_count"], 2)
        self.assertEqual(payload["workspace_count"], 2)
        self.assertEqual(payload["categories"][0]["id"], "01_ai_operator")
        self.assertEqual(payload["workstation"]["ubuntu_home"], "/home/ticket")
        self.assertIn("do not confirm current filesystem", payload["description"])
        self.assertEqual(result["record"].permission_decision, "automatic")

    def test_category_listing_includes_declared_location_and_purpose(self) -> None:
        payload = self.decoded_output(
            self.run_tool("workstation_list_categories")
        )

        self.assertEqual(payload["category_count"], 2)
        self.assertEqual(payload["categories"][0]["path"], "01_AI_Operator")
        self.assertEqual(
            payload["categories"][0]["purpose"], "AI assistant workspaces"
        )

    def test_workspace_listing_includes_registered_project_metadata(self) -> None:
        payload = self.decoded_output(
            self.run_tool("workstation_list_workspaces")
        )
        lila = next(item for item in payload["workspaces"] if item["name"] == "Lila")

        self.assertEqual(payload["workspace_count"], 2)
        self.assertEqual(
            lila["canonical_path"], "/home/ticket/workspaces/lila"
        )
        self.assertEqual(
            lila["git_root"], "/home/ticket/workspaces/lila/source"
        )
        self.assertEqual(
            lila["environment"],
            "/home/ticket/workspaces/lila/environment/.venv",
        )
        self.assertNotIn("exists", lila)

    def test_workspace_lookup_uses_registered_name_not_a_path(self) -> None:
        payload = self.decoded_output(
            self.run_tool(
                "workstation_get_workspace",
                {"identifier": "Lila"},
            )
        )

        self.assertEqual(payload["workspace"]["name"], "Lila")
        self.assertEqual(
            payload["workspace"]["git_root"],
            "/home/ticket/workspaces/lila/source",
        )
        identifier_arg = get_registry().get(
            "workstation_get_workspace"
        ).args[0]
        self.assertEqual(identifier_arg.name, "identifier")

    def test_unknown_workspace_is_reported_without_reading_a_path(self) -> None:
        result = self.run_tool(
            "workstation_get_workspace",
            {"identifier": "not-registered"},
        )

        self.assertFalse(result["result"].success)
        self.assertIn("No registered workspace", result["result"].error)
        self.assertEqual(result["record"].permission_decision, "automatic")

    def test_empty_and_path_like_identifiers_are_rejected_as_names(self) -> None:
        for identifier in ("", "/etc/passwd", "../../etc/passwd"):
            with self.subTest(identifier=identifier):
                result = self.run_tool(
                    "workstation_get_workspace",
                    {"identifier": identifier},
                )
                self.assertFalse(result["result"].success)

    def test_missing_and_malformed_registry_fail_closed(self) -> None:
        missing_path = Path(self.temp_dir.name) / "missing.json"
        with patch(
            "src.workstation.tools.WorkstationRegistry",
            side_effect=lambda: WorkstationRegistry._for_testing(missing_path),
        ):
            missing = self.run_tool("workstation_overview")
        self.assertFalse(missing["result"].success)
        self.assertIn("unavailable or invalid", missing["result"].error)

        malformed_path = Path(self.temp_dir.name) / "malformed.json"
        malformed_path.write_text("{", encoding="utf-8")
        with patch(
            "src.workstation.tools.WorkstationRegistry",
            side_effect=lambda: WorkstationRegistry._for_testing(
                malformed_path
            ),
        ):
            malformed = self.run_tool("workstation_overview")
        self.assertFalse(malformed["result"].success)
        self.assertIn("unavailable or invalid", malformed["result"].error)

    def test_intake_configuration_is_reported_without_processing_files(self) -> None:
        payload = self.decoded_output(
            self.run_tool("workstation_intake_overview")
        )

        self.assertEqual(
            payload["directories"][0]["configured_location"],
            valid_document()["intake"]["inbox"],
        )
        self.assertFalse(payload["overwrite_allowed"])
        self.assertIn("does not inspect or process", payload["note"])

    def test_software_configuration_is_reported_without_installing(self) -> None:
        payload = self.decoded_output(
            self.run_tool("workstation_software_overview")
        )

        self.assertEqual(
            payload["locations"]["catalog"],
            valid_document()["software"]["catalog"],
        )
        self.assertFalse(payload["automatic_installation"])
        self.assertIn("does not inspect, install", payload["note"])

    def test_registry_fixture_is_unchanged_after_queries(self) -> None:
        before = self.fixture_path.read_bytes()
        self.run_tool("workstation_overview")
        self.run_tool("workstation_list_workspaces")
        self.assertEqual(self.fixture_path.read_bytes(), before)

    def test_large_registry_output_is_bounded_and_marks_truncation(self) -> None:
        document = valid_document()
        document["categories"] = [
            {
                "id": f"category_{index}",
                "path": f"Category_{index}",
                "purpose": "purpose " * 80,
                "type": "domain",
            }
            for index in range(150)
        ]
        document["workspaces"][0]["category"] = "category_0"
        document["workspaces"][1]["category"] = "category_1"
        self.fixture_path.write_text(json.dumps(document), encoding="utf-8")

        payload = self.decoded_output(
            self.run_tool("workstation_list_categories")
        )

        self.assertLessEqual(
            len(json.dumps(payload, separators=(",", ":"))),
            MAX_OUTPUT_CHARS,
        )
        self.assertTrue(payload["truncated"])
        self.assertLess(payload["returned_count"], payload["category_count"])

    def test_read_only_tools_are_automatic_and_other_policies_unchanged(self) -> None:
        for name in WORKSTATION_TOOL_NAMES:
            with self.subTest(name=name):
                self.assertEqual(policy_for(name), ("low", "auto"))
        self.assertEqual(policy_for("unknown_tool"), ("high", "approval"))
        self.assertEqual(policy_for("file_write"), ("moderate", "approval"))

    def test_unapproved_write_remains_denied(self) -> None:
        from src.harness.system_tools import register_all_system_tools

        register_all_system_tools()
        target = Path(self.temp_dir.name) / "must_not_be_written"
        result = self.executor().process(
            json_tool_call(
                "file_write",
                {"path": str(target), "content": "not allowed"},
            )
        )[0]

        self.assertFalse(result["result"].success)
        self.assertEqual(result["record"].permission_decision, "denied")
        self.assertFalse(target.exists())

    def test_lilacore_registers_tools_and_includes_awareness_guidance(self) -> None:
        core = LilaCore.__new__(LilaCore)
        core.workspace_root = Path(self.temp_dir.name)
        core.log_dir = Path(self.temp_dir.name) / "logs"
        core.approval_callback = None
        core._harness = None
        core._conversation_history = []
        core.workstation_enabled = True

        core._boot_harness()
        self.assertTrue(WORKSTATION_TOOL_NAMES.issubset(core.available_tools))
        system_message = core._build_messages("hello", None)[0]["content"]
        self.assertIn("Command Center registry describes registered", system_message)
        self.assertIn("clearly distinguish registry declarations", system_message)
        self.assertIn("never infer workspace names", system_message)
        self.assertIn("workstation_get_workspace", system_message)

    def test_lilacore_does_not_enable_personal_workstation_by_default(self) -> None:
        with (
            patch.dict("os.environ", {"LILA_WORKSTATION_ENABLED": ""}),
            patch("pathlib.Path.mkdir"),
            patch("src.core.lilacore.secure_private_directory"),
        ):
            core = LilaCore(api_mode=True)
        self.assertFalse(core.workstation_enabled)
        for name in WORKSTATION_TOOL_NAMES:
            self.tool_registry._tools.pop(name, None)
        core._harness = None
        core._conversation_history = []
        core._boot_harness()
        self.assertFalse(WORKSTATION_TOOL_NAMES.intersection(core.available_tools))
        system_message = core._build_messages("hello", None)[0]["content"]
        self.assertNotIn("Command Center registry", system_message)

    def test_real_registry_tool_query_does_not_change_authoritative_bytes(self) -> None:
        if not PRODUCTION_REGISTRY_PATH.is_file():
            self.skipTest("Authoritative Command Center registry is unavailable")
        before = hashlib.sha256(PRODUCTION_REGISTRY_PATH.read_bytes()).digest()

        with patch(
            "src.workstation.tools.WorkstationRegistry",
            WorkstationRegistry,
        ):
            result = self.run_tool("workstation_overview")

        self.assertTrue(result["result"].success, result["result"].error)
        after = hashlib.sha256(PRODUCTION_REGISTRY_PATH.read_bytes()).digest()
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
