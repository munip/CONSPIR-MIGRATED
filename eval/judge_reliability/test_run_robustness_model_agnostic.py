import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("run_robustness.py")
SPEC = importlib.util.spec_from_file_location("run_robustness_under_test", MODULE_PATH)
rr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rr)


class FakeModel:
    async def generate(self, prompt, config):
        self.prompt = prompt
        self.config = config
        return SimpleNamespace(
            completion="hold",
            stop_reason="stop",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
            ),
        )


class ModelAgnosticRobustnessTests(unittest.IsolatedAsyncioTestCase):
    def test_model_provider_uses_inspect_prefix(self):
        self.assertEqual(rr.model_provider("ollama/llama3.2:3b"), "ollama")
        self.assertEqual(rr.model_provider("groq/openai/gpt-oss-20b"), "groq")
        self.assertEqual(rr.model_provider("unknown-model"), "unknown")

    async def test_inspect_model_call_returns_generic_result_shape(self):
        fake = FakeModel()

        with patch.object(rr, "get_model", return_value=fake) as get_model:
            result = await rr._call_inspect_model(
                "judge this response",
                "ollama/test-model",
            )

        get_model.assert_called_once_with("ollama/test-model")
        self.assertEqual(fake.prompt, "judge this response")
        self.assertEqual(result["raw_output"], "hold")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["total_tokens"], 12)
        self.assertIsNone(result["request_id"])
        self.assertIsNone(fake.config.reasoning_effort)

    def test_reasoning_effort_auto_keeps_the_gpt_oss_baseline_only(self):
        self.assertEqual(rr.resolve_reasoning_effort("groq/openai/gpt-oss-20b", "auto"), "low")
        self.assertIsNone(rr.resolve_reasoning_effort("ollama/llama3.2:3b", "auto"))
        self.assertEqual(rr.resolve_reasoning_effort("ollama/llama3.2:3b", "high"), "high")
        self.assertIsNone(rr.resolve_reasoning_effort("groq/openai/gpt-oss-20b", "none"))

    async def test_reasoning_effort_is_passed_to_the_model(self):
        fake = FakeModel()
        with patch.object(rr, "get_model", return_value=fake):
            await rr._call_inspect_model("judge this response", "groq/openai/gpt-oss-20b", "low")
        self.assertEqual(fake.config.reasoning_effort, "low")


if __name__ == "__main__":
    unittest.main()
