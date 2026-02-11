"""
Unit tests for _filter_function_args function.

Tests that unexpected/hallucinated arguments from LLMs (like 'internal_monologue')
are properly filtered out before function execution.
"""

import pytest

from mirix.agent.agent import _filter_function_args
from mirix.schemas.enums import ToolType
from mirix.schemas.tool import Tool


def make_tool(name: str, tool_type: ToolType) -> Tool:
    """Create a minimal Tool object for testing."""
    return Tool(
        name=name,
        tool_type=tool_type,
        tags=[],
        source_type="python",
        source_code="",
        json_schema={
            "name": name,
            "description": "Test tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )


class TestFilterFunctionArgs:
    """Tests for _filter_function_args function."""

    def test_filters_internal_monologue_from_trigger_memory_update(self):
        """
        Test that 'internal_monologue' is filtered from trigger_memory_update.
        This is the exact scenario from the bug report.
        """
        tool = make_tool("trigger_memory_update", ToolType.MIRIX_MEMORY_CORE)

        # Args as they might come from ChatGPT with hallucinated internal_monologue
        input_args = {
            "memory_types": ["episodic", "procedural", "semantic", "knowledge_vault"],
            "internal_monologue": "This interaction details a complex QuickBooks Desktop Payroll support session...",
            "user_message": {"message": "test message"},
        }

        filtered = _filter_function_args("trigger_memory_update", input_args, tool)

        # internal_monologue should be removed
        assert "internal_monologue" not in filtered
        # Valid args should remain
        assert "memory_types" in filtered
        assert "user_message" in filtered
        assert filtered["memory_types"] == ["episodic", "procedural", "semantic", "knowledge_vault"]

    def test_filters_multiple_unexpected_args(self):
        """Test that multiple unexpected args are all filtered."""
        tool = make_tool("trigger_memory_update", ToolType.MIRIX_MEMORY_CORE)

        input_args = {
            "memory_types": ["episodic"],
            "internal_monologue": "Some thoughts...",
            "reasoning": "Because...",
            "extra_field": "should be removed",
            "user_message": {"message": "test"},
        }

        filtered = _filter_function_args("trigger_memory_update", input_args, tool)

        assert "internal_monologue" not in filtered
        assert "reasoning" not in filtered
        assert "extra_field" not in filtered
        assert "memory_types" in filtered
        assert "user_message" in filtered

    def test_preserves_all_valid_args(self):
        """Test that all valid arguments are preserved."""
        tool = make_tool("core_memory_append", ToolType.MIRIX_MEMORY_CORE)

        input_args = {
            "label": "persona",
            "content": "New information about the user",
        }

        filtered = _filter_function_args("core_memory_append", input_args, tool)

        assert filtered == input_args

    def test_returns_empty_dict_when_all_args_invalid(self):
        """Test behavior when all provided args are invalid."""
        tool = make_tool("finish_memory_update", ToolType.MIRIX_MEMORY_CORE)

        # finish_memory_update takes no args (except self which is added later)
        input_args = {
            "internal_monologue": "Some thoughts...",
            "random_field": "value",
        }

        filtered = _filter_function_args("finish_memory_update", input_args, tool)

        # All args should be filtered out
        assert "internal_monologue" not in filtered
        assert "random_field" not in filtered

    def test_does_not_filter_user_defined_tools(self):
        """Test that USER_DEFINED tools are not filtered (returned as-is)."""
        tool = make_tool("custom_tool", ToolType.USER_DEFINED)

        input_args = {
            "some_arg": "value",
            "internal_monologue": "Should NOT be filtered for user tools",
        }

        filtered = _filter_function_args("custom_tool", input_args, tool)

        # Should return args unchanged
        assert filtered == input_args
        assert "internal_monologue" in filtered

    def test_does_not_filter_mcp_tools(self):
        """Test that MCP tools are not filtered (returned as-is)."""
        tool = make_tool("mcp_tool", ToolType.MIRIX_MCP)

        input_args = {
            "query": "test",
            "internal_monologue": "Should NOT be filtered for MCP tools",
        }

        filtered = _filter_function_args("mcp_tool", input_args, tool)

        # Should return args unchanged
        assert filtered == input_args
        assert "internal_monologue" in filtered

    def test_filters_mirix_core_tools(self):
        """Test that MIRIX_CORE tools are filtered."""
        tool = make_tool("send_message", ToolType.MIRIX_CORE)

        input_args = {
            "message": "Hello!",
            "internal_monologue": "Should be filtered",
        }

        filtered = _filter_function_args("send_message", input_args, tool)

        assert "message" in filtered
        assert "internal_monologue" not in filtered

    def test_filters_mirix_extra_tools(self):
        """Test that MIRIX_EXTRA tools are filtered."""
        tool = make_tool("web_search", ToolType.MIRIX_EXTRA)

        input_args = {
            "query": "test query",
            "num_results": 5,
            "internal_monologue": "Should be filtered",
        }

        filtered = _filter_function_args("web_search", input_args, tool)

        assert "query" in filtered
        assert "num_results" in filtered
        assert "internal_monologue" not in filtered

    def test_handles_empty_args(self):
        """Test that empty args dict is handled correctly."""
        tool = make_tool("finish_memory_update", ToolType.MIRIX_MEMORY_CORE)

        filtered = _filter_function_args("finish_memory_update", {}, tool)

        assert filtered == {}

    def test_preserves_arg_values_exactly(self):
        """Test that argument values are preserved exactly (not modified)."""
        tool = make_tool("episodic_memory_insert", ToolType.MIRIX_MEMORY_CORE)

        complex_items = [
            {
                "occurred_at": "2024-01-15T10:30:00Z",
                "event_type": "action",
                "actor": "user",
                "summary": "User did something",
                "details": "Detailed description",
            }
        ]

        input_args = {
            "items": complex_items,
            "internal_monologue": "filter this",
        }

        filtered = _filter_function_args("episodic_memory_insert", input_args, tool)

        assert "items" in filtered
        assert filtered["items"] is complex_items  # Same object reference
        assert filtered["items"][0]["summary"] == "User did something"
