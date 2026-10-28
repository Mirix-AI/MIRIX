"""The retrieval renderer: snapshot tests per memory type.

``mirix.client.render.render_retrieved_memories`` is what the eval harness's
MIRIX method imports (``methods/mirix.py``: ``RENDER_MODULE`` /
``RENDER_FUNCTION``) and injects unchanged; ``MirixClient.render_memories`` is
the same function.
"""

import pytest

from mirix.client import MirixClient
from mirix.client.render import RENDER_ORDER, render_retrieved_memories, render_skill

SKILL_A = {
    "id": "proc-a",
    "memory_type": "procedural",
    "name": "export-report",
    "description": "Export a report as CSV.",
    "instructions": "1. Open the report.\n2. Click Export.\n3. Choose CSV.",
    "triggers": ["user wants a CSV", "report export"],
    "examples": [{"input": "export Q3", "output": "q3.csv", "note": "äö"}],
    "retrieval_scores": {"bm25": 1.5, "cosine": 0.8, "rrf": 0.03},
}
SKILL_B = {
    "id": "proc-b",
    "name": "retry-upload",
    "description": "Retry a failed upload.",
    "instructions": "Wait, then upload again.",
    "triggers": [],
    "examples": [],
    "retrieval_scores": {"bm25": 0.0, "cosine": 0.7, "rrf": 0.016},
}


def _response(**memories):
    return {"success": True, "retrieval_id": "rlog-1", "memories": memories}


class TestProcedural:
    def test_exact_spec_format(self):
        text = render_retrieved_memories(
            _response(procedural={"items": [SKILL_A, SKILL_B]})
        )
        assert text == (
            "## Relevant skills from memory\n"
            "### export-report\n"
            "Export a report as CSV.\n"
            "When to use: user wants a CSV; report export\n"
            "Instructions:\n"
            "1. Open the report.\n2. Click Export.\n3. Choose CSV.\n"
            "Examples:\n"
            "- input: export Q3; output: q3.csv; note: äö\n"
            "### retry-upload\n"
            "Retry a failed upload.\n"
            # No trigger: no "When to use:" line (it was "When to use: ").
            "Instructions:\n"
            "Wait, then upload again."
        )

    def test_a_normal_skill_without_examples_is_unchanged(self):
        """The renderer changes only content-free fields and the examples: a skill
        with a description, triggers and no examples renders as before."""
        skill = {**SKILL_A, "examples": []}
        assert render_skill(skill) == (
            "### export-report\n"
            "Export a report as CSV.\n"
            "When to use: user wants a CSV; report export\n"
            "Instructions:\n"
            "1. Open the report.\n2. Click Export.\n3. Choose CSV."
        )

    def test_name_description_and_instructions_are_literal(self):
        """Nothing with content is stripped or rewritten."""
        skill = {
            "name": "pad",
            "description": "  Leading and trailing spaces.  ",
            "instructions": "\n  indented step\n",
            "triggers": [" spaced trigger "],
            "examples": [],
        }
        assert render_skill(skill) == (
            "### pad\n"
            "  Leading and trailing spaces.  \n"
            "When to use:  spaced trigger \n"
            "Instructions:\n"
            "\n  indented step\n"
        )

    @pytest.mark.parametrize("description", ["...", "-", "", "  ", "…", None, "?!"])
    def test_a_content_free_description_has_no_line(self, description):
        """A placeholder description ("...") was injected verbatim."""
        text = render_skill({**SKILL_A, "description": description, "examples": []})
        assert text.splitlines()[:2] == [
            "### export-report",
            "When to use: user wants a CSV; report export",
        ]

    @pytest.mark.parametrize("triggers", [[], None, ["", "  "], ["...", "-"], "..."])
    def test_no_trigger_with_content_has_no_when_to_use_line(self, triggers):
        """The "When to use: " line was printed with no trigger at all."""
        text = render_skill({**SKILL_A, "triggers": triggers, "examples": []})
        assert "When to use" not in text
        assert text.splitlines()[:3] == [
            "### export-report",
            "Export a report as CSV.",
            "Instructions:",
        ]

    def test_content_free_triggers_are_left_out(self):
        text = render_skill({**SKILL_A, "triggers": ["...", "report export", "-"]})
        assert "When to use: report export\n" in text

    def test_examples_line_is_omitted_when_empty(self):
        for examples in ([], None, {}, [{}], [{}, {"input": "", "output": None}], [""]):
            text = render_skill({**SKILL_A, "examples": examples})
            assert "Examples:" not in text, examples
            assert text.endswith("3. Choose CSV.")

    def test_examples_are_one_line_each(self):
        """Examples were one compact-JSON line (613 characters for one
        skill); now each is a "- key: value; ..." line in its key order."""
        text = render_skill(
            {
                **SKILL_A,
                "examples": [
                    {"input": "export Q3", "output": "q3.csv"},
                    {},
                    {"output": "done", "input": "retry"},
                    "free-text example",
                ],
            }
        )
        assert text.split("Examples:\n", 1)[1] == (
            "- input: export Q3; output: q3.csv\n"
            "- output: done; input: retry\n"
            "- free-text example"
        )

    def test_a_nested_example_value_is_compact_json(self):
        text = render_skill(
            {**SKILL_A, "examples": [{"a": 1, "b": [1, 2], "c": {"d": "é"}}]}
        )
        assert text.splitlines()[-2:] == [
            "Examples:",
            '- a: 1; b: [1,2]; c: {"d":"é"}',
        ]

    def test_a_multi_line_example_value_keeps_its_lines_under_its_example(self):
        """A newline inside a value broke the one-example-per-"- "-block layout,
        and a value line starting "### " read as a new skill heading; collapsing
        the value onto one line lost the structure of a code or a
        table example. Each line of a multi-line value is indented under its
        key, its relative indentation kept; the other pairs of the example get
        a line each."""
        text = render_skill(
            {
                **SKILL_A,
                "examples": [
                    {
                        "input": "Sum column B",
                        "output": "```python\ndf['B'].sum()\n```",
                    },
                    {"input": "| a | b |\n|---|---|\n| 1 | 2 |", "output": "3"},
                    {"input": "q", "output": "done\n### fake-skill\nInstructions:"},
                    {"input": "\n  def f():\n      return 1\n\n", "output": "ok"},
                    "free\r\n  text",
                    {"input": "a\n\n\n\nb", "output": "one\u2028two"},
                ],
            }
        )
        examples = text.split("Examples:\n", 1)[1]
        assert examples == (
            "- input: Sum column B\n"
            "  output:\n"
            "    ```python\n"
            "    df['B'].sum()\n"
            "    ```\n"
            "- input:\n"
            "    | a | b |\n"
            "    |---|---|\n"
            "    | 1 | 2 |\n"
            "  output: 3\n"
            "- input: q\n"
            "  output:\n"
            "    done\n"
            "    ### fake-skill\n"
            "    Instructions:\n"
            "- input:\n"
            "    def f():\n"
            "        return 1\n"
            "  output: ok\n"
            "- free\n"
            "    text\n"
            "- input:\n"
            "    a\n"
            "\n"
            "    b\n"
            "  output:\n"
            "    one\n"
            "    two"
        )
        # No line of a value reads as a heading or a field of the skill block.
        for line in examples.splitlines():
            assert not line.startswith(("#", "Instructions:", "When to use:")), line
        # Blank lines only around a value's text: it stays on its example's line.
        text = render_skill(
            {**SKILL_A, "examples": [{"input": "\n  q  \n", "output": "a  b"}]}
        )
        assert text.splitlines()[-1] == "- input: q; output: a  b"
        # Only the indentation the lines share is removed: a tab and two
        # spaces share none, so neither line loses a character of its own.
        text = render_skill(
            {
                **SKILL_A,
                "examples": [{"input": "\tdef f():\n  return 1", "output": "ok"}],
            }
        )
        assert text.split("Examples:\n", 1)[1] == (
            "- input:\n    \tdef f():\n      return 1\n  output: ok"
        )
        text = render_skill({**SKILL_A, "examples": ["\t\tif x:\n\t\t\ty()\n\t\tz()"]})
        assert text.split("Examples:\n", 1)[1] == "- if x:\n  \ty()\n  z()"

    def test_an_example_key_is_one_line(self):
        text = render_skill(
            {**SKILL_A, "examples": [{"in\n### put": "q", "output": "a"}]}
        )
        assert text.splitlines()[-1] == "- in ### put: q; output: a"

    def test_a_punctuation_only_value_of_a_real_example_is_kept(self):
        """An output such as "==" or "-" is a real value of an
        example that says something; an elision ("...", "…") is a placeholder,
        and an example with no letter or digit at all says nothing."""
        text = render_skill(
            {
                **SKILL_A,
                "examples": [
                    {"input": "Which operator tests equality?", "output": "=="},
                    {"input": "Negate the sign", "output": "-"},
                    {"input": "Shorten the quote", "output": "…"},
                    {"input": "Summarize the report", "output": ". . ."},
                    {"input": "==", "output": "!="},
                    "?",
                    {"input": "Filter", "output": ["?"]},
                ],
            }
        )
        assert text.split("Examples:\n", 1)[1] == (
            "- input: Which operator tests equality?; output: ==\n"
            "- input: Negate the sign; output: -\n"
            "- input: Shorten the quote\n"
            "- input: Summarize the report\n"
            '- input: Filter; output: ["?"]'
        )

    def test_content_free_example_parts_are_left_out(self):
        for examples in (
            [[]],
            [{"input": {}, "output": []}],
            [{"input": "...", "output": " "}],
        ):
            text = render_skill({**SKILL_A, "examples": examples})
            assert "Examples:" not in text, examples
        text = render_skill(
            {**SKILL_A, "examples": [{"input": {}, "output": "q3.csv", "note": [[]]}]}
        )
        assert text.splitlines()[-2:] == ["Examples:", "- output: q3.csv"]
        # Falsy values with content stay.
        text = render_skill({**SKILL_A, "examples": [{"input": 0, "output": False}]})
        assert text.splitlines()[-1] == "- input: 0; output: false"

    def test_a_single_mapping_is_one_example(self):
        text = render_skill({**SKILL_A, "examples": {"input": "q", "output": "a"}})
        assert text.splitlines()[-2:] == ["Examples:", "- input: q; output: a"]

    def test_no_skill_is_the_empty_string(self):
        """The gate can return no skill: nothing is injected."""
        assert render_retrieved_memories(_response(procedural={"items": []})) == ""
        assert render_retrieved_memories(_response(procedural={"total_count": 3})) == ""

    def test_order_is_the_retrieval_order(self):
        text = render_retrieved_memories(
            _response(procedural={"items": [SKILL_B, SKILL_A]})
        )
        assert text.index("### retry-upload") < text.index("### export-report")


class TestOtherTypes:
    def test_core(self):
        core = {
            "total_count": 3,
            "scopes": {
                "default": {
                    "items": [
                        {
                            "id": "b1",
                            "label": "persona",
                            "value": "I am a helpful assistant.",
                        },
                        {"id": "b2", "label": "human", "value": "Name: Ada."},
                        {"id": "b3", "label": "empty", "value": "  "},
                    ]
                }
            },
        }
        assert render_retrieved_memories(_response(core=core)) == (
            "## Core memory\n"
            "### persona\n"
            "I am a helpful assistant.\n"
            "### human\n"
            "Name: Ada."
        )

    def test_core_from_several_scopes_names_the_scope(self):
        core = {
            "scopes": {
                "a": {"items": [{"label": "human", "value": "A"}]},
                "b": {"items": [{"label": "human", "value": "B"}]},
            }
        }
        assert render_retrieved_memories(_response(core=core)) == (
            "## Core memory\n### human (scope: a)\nA\n### human (scope: b)\nB"
        )

    def test_episodic_relevant_then_recent_without_repeats(self):
        episodic = {
            "total_count": 5,
            "relevant": [
                {
                    "id": "e1",
                    "timestamp": "2026-09-01T10:00:00",
                    "summary": "Met Bob",
                    "details": "Talked about Q3.",
                },
            ],
            "recent": [
                {
                    "id": "e1",
                    "timestamp": "2026-09-01T10:00:00",
                    "summary": "Met Bob",
                    "details": "Talked about Q3.",
                },
                {"id": "e2", "timestamp": None, "summary": "Lunch", "details": ""},
            ],
        }
        assert render_retrieved_memories(_response(episodic=episodic)) == (
            "## Relevant events from memory\n"
            "### 2026-09-01T10:00:00 Met Bob\n"
            "Talked about Q3.\n"
            "\n"
            "## Recent events from memory\n"
            "### Lunch"
        )

    def test_semantic(self):
        semantic = {
            "items": [
                {
                    "id": "s1",
                    "name": "Paris",
                    "summary": "Capital of France.",
                    "details": "Population 2.1M.",
                },
            ]
        }
        assert render_retrieved_memories(_response(semantic=semantic)) == (
            "## Relevant knowledge from memory\n### Paris\nCapital of France.\nPopulation 2.1M."
        )

    def test_resource(self):
        resource = {
            "items": [
                {
                    "id": "r1",
                    "title": "Q3 plan",
                    "summary": "The plan.",
                    "resource_type": "doc",
                },
                {"id": "r2", "title": "Notes", "summary": "", "resource_type": None},
            ]
        }
        assert render_retrieved_memories(_response(resource=resource)) == (
            "## Relevant resources from memory\n### Q3 plan (doc)\nThe plan.\n### Notes"
        )

    def test_knowledge_vault_shows_captions_only(self):
        vault = {
            "items": [
                {"id": "k1", "caption": "GitHub token"},
                {"id": "k2", "caption": ""},
            ]
        }
        assert render_retrieved_memories(_response(knowledge_vault=vault)) == (
            "## Relevant knowledge vault entries\n- GitHub token"
        )

    def test_empty_sections_render_nothing(self):
        response = _response(
            core={"total_count": 0, "scopes": {}},
            episodic={"total_count": 0, "recent": [], "relevant": []},
            semantic={"total_count": 0, "items": []},
            procedural={"total_count": 0, "items": []},
            resource={"total_count": 0, "items": []},
            knowledge_vault={"total_count": 0, "items": []},
        )
        assert render_retrieved_memories(response) == ""


class TestSelectionAndShape:
    def _full(self):
        return _response(
            procedural={"items": [SKILL_B]},
            semantic={
                "items": [{"name": "Paris", "summary": "Capital.", "details": ""}]
            },
            core={
                "scopes": {"default": {"items": [{"label": "human", "value": "Ada"}]}}
            },
        )

    def test_default_order_and_blank_line_between_sections(self):
        text = render_retrieved_memories(self._full())
        assert RENDER_ORDER == (
            "core",
            "episodic",
            "semantic",
            "procedural",
            "resource",
            "knowledge_vault",
        )
        assert text.split("\n\n") == [
            "## Core memory\n### human\nAda",
            "## Relevant knowledge from memory\n### Paris\nCapital.",
            render_retrieved_memories(_response(procedural={"items": [SKILL_B]})),
        ]

    def test_memory_types_selects_and_orders(self):
        text = render_retrieved_memories(self._full(), ["procedural", "core"])
        assert text.startswith("## Relevant skills from memory")
        assert text.endswith("## Core memory\n### human\nAda")
        assert "Paris" not in text
        assert render_retrieved_memories(self._full(), "procedural").startswith(
            "## Relevant skills"
        )

    def test_unknown_memory_type_is_an_error(self):
        with pytest.raises(ValueError, match="unknown memory_types"):
            render_retrieved_memories(self._full(), ["skills"])

    @pytest.mark.parametrize(
        "response",
        [
            {},
            {"success": False, "error": "No agents found"},
            {"memories": None},
            None,
            "x",
        ],
    )
    def test_nothing_to_render(self, response):
        assert render_retrieved_memories(response) == ""

    def test_tolerates_missing_and_odd_fields(self):
        odd = {
            "items": [
                {
                    "name": None,
                    "description": None,
                    "instructions": None,
                    "triggers": "one trigger",
                    "examples": None,
                },
                "not a dict",
            ]
        }
        # No description (no empty line for it), one trigger.
        assert render_retrieved_memories(_response(procedural=odd)) == (
            "## Relevant skills from memory\n### \nWhen to use: one trigger\nInstructions:\n"
        )

    def test_the_client_method_is_the_same_function(self):
        response = self._full()
        assert MirixClient.render_memories(response) == render_retrieved_memories(
            response
        )
        assert MirixClient.render_memories(response, ["core"]) == (
            "## Core memory\n### human\nAda"
        )

    def test_the_harness_call_shape(self):
        """methods/mirix.py calls render(response) with the whole response of
        retrieve_with_conversation(memory_types=["procedural"])."""
        response = {
            "success": True,
            "retrieval_id": "rlog-x",
            "topics": "export; csv",
            "temporal_expression": None,
            "extraction_status": "ok",
            "date_range": None,
            "memories": {"procedural": {"total_count": 2, "items": [SKILL_A]}},
            "errors": {},
        }
        text = render_retrieved_memories(response)
        assert isinstance(text, str) and text.startswith(
            "## Relevant skills from memory\n### export-report\n"
        )
