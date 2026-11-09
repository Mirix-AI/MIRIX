"""Offline tests for the evaluation client's credential boundary.

Compile the actual request method without importing GPU/model dependencies.
All HTTP calls are mocked; no provider credential is tested against a service.
"""

import ast
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


AGENT_PATH = Path(__file__).resolve().parents[1] / "agent.py"


def request_method(http):
    module = ast.parse(AGENT_PATH.read_text())
    wrapper = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AgentWrapper")
    method = next(
        node for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_answer_from_retrieved_images_and_texts"
    )
    namespace = {"os": os, "requests": http, "load_dotenv": lambda: None}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(AGENT_PATH), "exec"), namespace)
    return namespace[method.name]


class ProviderCredentialTests(unittest.TestCase):
    def test_gemini_uses_configured_key_in_header(self):
        http = Mock()
        http.post.return_value = SimpleNamespace(
            status_code=200,
            json=lambda: {"candidates": [{"content": {"parts": [{"text": "answer"}]}}]},
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only-configured-key"}, clear=True):
            result = request_method(http)(
                object(), retrieved_texts=[], retrieved_captions=[],
                message="question", model_name="gemini-2.0-flash",
            )
        self.assertEqual(result, "answer")
        self.assertEqual(http.post.call_args.kwargs["headers"]["x-goog-api-key"], "test-only-configured-key")
        self.assertNotIn("params", http.post.call_args.kwargs)
        self.assertNotIn("key=", http.post.call_args.args[0])

    def test_missing_gemini_key_fails_before_http(self):
        http = Mock()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                request_method(http)(
                    object(), retrieved_texts=[], retrieved_captions=[],
                    message="question", model_name="gemini-2.0-flash",
                )
        http.post.assert_not_called()

    def test_agent_has_no_embedded_provider_keys(self):
        source = AGENT_PATH.read_text()
        patterns = (
            r"AIza[0-9A-Za-z_-]{35}",
            r"sk-(?:proj-)?[0-9A-Za-z_-]{20,}",
        )
        # Do not print a matching credential in the assertion output.
        self.assertFalse(any(re.search(pattern, source) for pattern in patterns),
                         "An embedded provider credential must be removed and revoked")


if __name__ == "__main__":
    unittest.main()
