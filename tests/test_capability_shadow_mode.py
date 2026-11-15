import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.core.lilacore import LilaCore
from src.harness.executor import HarnessExecutor
from src.harness.tools import Tool, ToolRegistry, ToolResult, get_registry, register_tool
from src.harness.permissions import PermissionManager


WORKSTATION_NAMES = {
    "workstation_overview",
    "workstation_list_categories",
    "workstation_list_workspaces",
    "workstation_get_workspace",
    "workstation_intake_overview",
    "workstation_software_overview",
}
DISABLED_NAMES = {"memory_store", "memory_recall", "memory_list"}


class CapabilityShadowModeTests(unittest.TestCase):
    def build_core_catalog(self, enabled):
        core = object.__new__(LilaCore)
        core.workstation_enabled = enabled
        core._capability_catalog = None
        core._capability_catalog_error = None
        core._capability_catalog = core._build_capability_catalog(enabled)
        return core

    def assert_opposite_order(self, first_setting, second_setting):
        global_registry = get_registry()
        prior_tools = global_registry.all()
        prior_ids = tuple(id(tool) for tool in prior_tools)
        prior_prompt = global_registry.prompt_block()
        prior_lookup = {tool.name: global_registry.get(tool.name) for tool in prior_tools}

        first = self.build_core_catalog(first_setting)
        second = self.build_core_catalog(second_setting)
        first_items = first._capability_catalog.all()
        second_items = second._capability_catalog.all()
        first_names = {item.display_name for item in first_items}
        second_names = {item.display_name for item in second_items}

        enabled_catalog = first if first_setting else second
        disabled_catalog = second if first_setting else first
        enabled_names = {item.display_name for item in enabled_catalog._capability_catalog.all()}
        disabled_names = {item.display_name for item in disabled_catalog._capability_catalog.all()}
        self.assertTrue(WORKSTATION_NAMES <= enabled_names)
        self.assertFalse(WORKSTATION_NAMES & disabled_names)
        self.assertEqual(enabled_names - WORKSTATION_NAMES, disabled_names)
        self.assertEqual(len(first_items), 47 if first_setting else 41)
        self.assertEqual(len(second_items), 47 if second_setting else 41)
        self.assertEqual(sum(item.availability_observations["legacy_prompt_visible"] for item in first_items), 44 if first_setting else 38)
        self.assertEqual(sum(item.availability_observations["legacy_prompt_visible"] for item in second_items), 44 if second_setting else 38)

        first_by_name = {item.display_name: item for item in first_items}
        second_by_name = {item.display_name: item for item in second_items}
        for name in first_by_name.keys() & second_by_name.keys():
            self.assertEqual(first_by_name[name], second_by_name[name])
        self.assertIsNot(first._capability_catalog, second._capability_catalog)
        self.assertIsNot(first._capability_catalog.snapshot(), second._capability_catalog.snapshot())

        disabled = {item.display_name: item for item in first_items + second_items if item.display_name in DISABLED_NAMES}
        self.assertEqual(set(disabled), DISABLED_NAMES)
        for item in disabled.values():
            self.assertIn("disabled_placeholder", {finding["code"] for finding in item.findings})
            self.assertFalse(item.availability_observations["legacy_prompt_visible"])
            self.assertEqual(item.consumer_exposure, "prohibited_from_consumer_exposure")

        by_name = {item.display_name: item for item in first_items}
        self.assertEqual(by_name["screen_capture"].consumer_exposure,
                         "prohibited_from_consumer_exposure")
        self.assertEqual(by_name["file_read"].consumer_exposure, "unreviewed")

        # Shadow registration is isolated from every live global object and prompt.
        self.assertEqual(tuple(id(tool) for tool in global_registry.all()), prior_ids)
        self.assertEqual(global_registry.prompt_block(), prior_prompt)
        for name, tool in prior_lookup.items():
            self.assertIs(global_registry.get(name), tool)

        # Catalog entries contain immutable values, never handler/provider objects.
        self.assertFalse(any(callable(value) for item in first_items for value in vars(item).values()))
        self.assertFalse(any(callable(value) for item in second_items for value in vars(item).values()))
        self.assertTrue(all(item.legacy_provenance["provider_digest_status"] == "unverified" for item in first_items))
        return first, second

    def test_workstation_on_then_off_are_isolated(self):
        self.assert_opposite_order(True, False)

    def test_workstation_off_then_on_are_isolated(self):
        self.assert_opposite_order(False, True)

    def test_explicit_registration_rejects_invalid_target_without_global_fallback(self):
        global_registry = get_registry()
        before = tuple(id(tool) for tool in global_registry.all())
        from src.harness.system_tools import register_all_system_tools
        from src.harness.extended_tools import register_extended_tools
        from src.workstation.tools import register_workstation_tools
        for register in (register_all_system_tools, register_extended_tools, register_workstation_tools):
            with self.subTest(register=register.__name__), self.assertRaises(TypeError):
                register(registry=object())
        self.assertEqual(tuple(id(tool) for tool in global_registry.all()), before)

    def test_explicit_registry_gets_fresh_tool_and_argument_objects(self):
        from src.harness.tools import ToolRegistry
        from src.harness.system_tools import register_all_system_tools
        from src.harness.extended_tools import register_extended_tools
        from src.workstation.tools import register_workstation_tools

        target = ToolRegistry()
        register_all_system_tools(target)
        register_extended_tools(target)
        register_workstation_tools(target)
        live = get_registry()
        for tool in target.all():
            live_tool = live.get(tool.name)
            if live_tool is None:
                continue
            self.assertIsNot(tool, live_tool)
            self.assertEqual(len(tool.args), len(live_tool.args))
            for arg, live_arg in zip(tool.args, live_tool.args):
                self.assertIsNot(arg, live_arg)

    def test_shadow_build_failure_is_recordable_without_marking_empty_success(self):
        core = object.__new__(LilaCore)
        core.workstation_enabled = False
        core._capability_catalog = None
        core._capability_catalog_error = None
        core.log_dir = Path("/tmp/lila-capability-shadow-test")
        core.workspace_root = Path("/tmp")
        core.approval_callback = None
        fake_executor = object()
        with patch.object(LilaCore, "_build_capability_catalog", side_effect=RuntimeError("test failure")), \
             patch("src.harness.system_tools.register_all_system_tools"), \
             patch("src.harness.extended_tools.register_extended_tools"), \
             patch("src.harness.executor.HarnessExecutor", return_value=fake_executor):
            core._boot_harness()
        self.assertIs(core._harness, fake_executor)
        self.assertIsNone(core._capability_catalog)
        self.assertEqual(
            core._capability_catalog_error,
            "capability.catalog_build_failed",
        )

    def test_shadow_registration_does_not_invoke_handlers_or_mutate_global_definitions(self):
        global_tools = get_registry().all()
        original = {tool.name: (tool, tool.risk, tool.permission, tuple(tool.args)) for tool in global_tools}
        with patch("src.harness.system_tools._file_read") as handler:
            catalog = LilaCore._build_capability_catalog(False)
        handler.assert_not_called()
        for name, (tool, risk, permission, args) in original.items():
            self.assertIs(get_registry().get(name), tool)
            self.assertEqual((tool.risk, tool.permission, tuple(tool.args)), (risk, permission, args))
        self.assertEqual(catalog.get("org.lila.capability.files.read", "0.0.0-legacy").display_name, "file_read")

    def test_live_executor_still_dispatches_through_original_global_registry(self):
        global_registry = get_registry()
        previous = dict(global_registry._tools)
        calls = []
        try:
            global_registry._tools.clear()

            def live_handler():
                calls.append("called")
                return ToolResult(success=True, output="live")

            original = Tool("self_status", "live probe", [], live_handler)
            register_tool(original)
            original_prompt = global_registry.prompt_block()
            catalog = LilaCore._build_capability_catalog(False)
            descriptive = catalog.get(
                "org.lila.capability.lila.status", "0.0.0-legacy"
            )
            approved_description = replace(
                descriptive, consumer_exposure="approved_for_consumer_exposure"
            )
            permission_before = PermissionManager().authorize(original, {})
            self.assertEqual(permission_before, (True, "Automatically allowed by policy."))
            self.assertEqual(calls, [])
            self.assertEqual(global_registry.prompt_block(), original_prompt)
            self.assertIs(global_registry.get("self_status"), original)
            self.assertNotEqual(descriptive.manifest_digest, approved_description.manifest_digest)
            self.assertEqual(PermissionManager().authorize(original, {}), permission_before)

            with tempfile.TemporaryDirectory() as temp_dir:
                executor = HarnessExecutor(log_dir=temp_dir)
                result = executor.process(
                    '<tool_call>{"name":"self_status","arguments":{}}</tool_call>'
                )[0]["result"]
            self.assertTrue(result.success)
            self.assertEqual(result.output, "live")
            self.assertEqual(calls, ["called"])
            self.assertEqual(catalog.get("org.lila.capability.lila.status", "0.0.0-legacy").display_name, "self_status")
        finally:
            global_registry._tools.clear()
            global_registry._tools.update(previous)


if __name__ == "__main__":
    unittest.main()
