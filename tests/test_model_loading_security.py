import sys
import tempfile
import unittest
import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from engine.asi.pack_asi import (
    _resolve_model_source as resolve_asi_model_source,
    build_tokenizer_section,
    pack_asi,
)
from engine.format.convert import (
    _resolve_model_source as resolve_convert_model_source,
    convert,
)
from src.core.lilacore import LilaCore


class ModelLoadingSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_missing_gguf_dependency_does_not_fall_back_to_transformers(self):
        gguf = self.root / "model.gguf"
        gguf.touch()
        core = LilaCore(gguf_path=str(gguf))
        fallback = Mock()
        core._boot_transformers = fallback

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with self.assertRaisesRegex(
                RuntimeError, "requires llama-cpp-python"
            ):
                core._boot_gguf()

        fallback.assert_not_called()
        self.assertIsNone(core._llm)

    def test_remote_identifier_is_rejected_without_explicit_opt_in(self):
        core = LilaCore(model_path="org/private-model")

        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            with self.assertRaisesRegex(RuntimeError, "Remote model identifiers are disabled"):
                core._boot_transformers()

    def test_remote_identifier_loads_only_after_explicit_opt_in(self):
        tokenizer = SimpleNamespace(pad_token=None, eos_token="eos")
        auto_tokenizer = Mock(return_value=tokenizer)
        model = object()
        auto_model = Mock(return_value=model)
        torch = SimpleNamespace(
            float16="float16",
            float32="float32",
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=auto_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=auto_model),
        )
        core = LilaCore(
            model_path="org/private-model",
            allow_remote_model=True,
        )

        with patch.dict(
            sys.modules,
            {"torch": torch, "transformers": transformers},
        ):
            core._boot_transformers()

        auto_tokenizer.assert_called_once_with(
            "org/private-model", local_files_only=False
        )
        self.assertEqual(
            auto_model.call_args.kwargs["local_files_only"], False
        )
        self.assertIs(core._model, model)
        self.assertEqual(tokenizer.pad_token, "eos")

    def test_local_transformers_model_never_requests_remote_files(self):
        model_dir = self.root / "local-model"
        model_dir.mkdir()
        tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")
        auto_tokenizer = Mock(return_value=tokenizer)
        auto_model = Mock(return_value=object())
        torch = SimpleNamespace(
            float16="float16",
            float32="float32",
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=auto_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=auto_model),
        )
        core = LilaCore(model_path=str(model_dir))

        with patch.dict(
            sys.modules,
            {"torch": torch, "transformers": transformers},
        ):
            core._boot_transformers()

        auto_tokenizer.assert_called_once_with(
            str(model_dir), local_files_only=True
        )
        self.assertTrue(auto_model.call_args.kwargs["local_files_only"])

    def test_missing_local_transformers_path_is_not_treated_as_remote(self):
        core = LilaCore(model_path=str(self.root / "missing-model"))
        with self.assertRaisesRegex(RuntimeError, "Remote model identifiers are disabled"):
            core._boot_transformers()

    def test_engine_conversion_tools_require_explicit_remote_opt_in(self):
        remote_id = "org/private-model"
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            with self.assertRaisesRegex(
                ValueError, "Remote model identifiers are disabled"
            ):
                convert(remote_id, str(self.root / "model.lila"))
            with self.assertRaisesRegex(
                ValueError, "Remote model identifiers are disabled"
            ):
                pack_asi(remote_id, str(self.root / "model.asi"))
            with self.assertRaisesRegex(
                ValueError, "Remote model identifiers are disabled"
            ):
                build_tokenizer_section(remote_id)

        self.assertEqual(
            resolve_convert_model_source(remote_id, True),
            (remote_id, False),
        )
        self.assertEqual(
            resolve_asi_model_source(remote_id, True),
            (remote_id, False),
        )

    def _run_conversion_loaders(self, model_path, local_files_only):
        model_dir = Path(model_path)
        if local_files_only:
            model_dir.mkdir()

        config = SimpleNamespace(
            num_hidden_layers=0,
            hidden_size=1,
            intermediate_size=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            vocab_size=1,
            max_position_embeddings=32,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
        )

        class FakeTensor:
            data = None

            def __init__(self):
                import numpy

                self.data = self
                self._array = numpy.array([1.0], dtype=numpy.float32)
                self.shape = self._array.shape
                self.nbytes = self._array.nbytes

            def numpy(self):
                return self._array

        model = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=FakeTensor()),
            get_output_embeddings=lambda: None,
            named_parameters=lambda: [],
        )
        class FakeTokenizer:
            vocab_size = 1
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 0

            def __len__(self):
                return 1

            def convert_ids_to_tokens(self, _index):
                return "x"

        tokenizer = FakeTokenizer()
        auto_config = Mock(return_value=config)
        auto_model = Mock(return_value=model)
        auto_tokenizer = Mock(return_value=tokenizer)
        transformers = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=auto_config),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=auto_model),
            AutoTokenizer=SimpleNamespace(from_pretrained=auto_tokenizer),
        )
        torch = SimpleNamespace(
            float32="float32",
            cuda=SimpleNamespace(is_available=lambda: False),
        )

        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            convert(
                model_path,
                str(self.root / "converted.lila"),
                allow_remote_model=not local_files_only,
            )
            with patch.multiple(
                "engine.asi.pack_asi",
                build_model_config=Mock(return_value=b"config"),
                build_weights_section=Mock(return_value=b"weights"),
                build_memory_fabric_section=Mock(return_value=b"memory"),
                build_bytecode_section=Mock(return_value=b"bytecode"),
                build_harness_section=Mock(return_value=b"harness"),
                build_personality_section=Mock(return_value=b"personality"),
                build_metadata_section=Mock(return_value=b"metadata"),
            ):
                pack_asi(
                    model_path,
                    str(self.root / "packed.asi"),
                    allow_remote_model=not local_files_only,
                )

        expected_path = str(model_dir) if local_files_only else model_path
        expected = {
            "local_files_only": local_files_only,
        }
        self.assertEqual(
            auto_config.call_args_list,
            [unittest.mock.call(expected_path, **expected)] * 2,
        )
        self.assertEqual(len(auto_model.call_args_list), 2)
        for call in auto_model.call_args_list:
            self.assertEqual(call.args[0], expected_path)
            self.assertEqual(call.kwargs["local_files_only"], local_files_only)
        self.assertEqual(
            auto_tokenizer.call_args_list,
            [unittest.mock.call(expected_path, **expected)] * 2,
        )
        self.assertTrue((self.root / "converted.lila").is_file())
        self.assertTrue((self.root / "packed.asi").is_file())

    def test_local_conversion_and_tokenizer_loads_are_local_only(self):
        self._run_conversion_loaders(str(self.root / "local-model"), True)

    def test_explicit_remote_conversion_and_tokenizer_loads_use_remote_opt_in(self):
        self._run_conversion_loaders("org/explicit-model", False)

    def test_default_gguf_boot_without_dependency_does_not_import_transformers(self):
        model_path = self.root / "models" / "qwen3-0.6b-q4_k_m.gguf"
        model_path.parent.mkdir()
        model_path.touch()
        core = LilaCore.__new__(LilaCore)
        core.model_path = None
        core.gguf_path = None
        core.api_mode = False
        core.workspace_root = self.root
        core._llm = None
        imports = []
        real_import = builtins.__import__

        def track_import(name, *args, **kwargs):
            imports.append(name)
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch("builtins.__import__", side_effect=track_import):
                with self.assertRaisesRegex(
                    RuntimeError, "requires llama-cpp-python"
                ):
                    core.boot()

        self.assertNotIn("transformers", imports)
        self.assertFalse(any(name.startswith("huggingface_hub") for name in imports))
        self.assertIsNone(core._llm)


if __name__ == "__main__":
    unittest.main()
