"""Guard: no synthetic ``[System Message]`` literals in production code.

GenSRF flags the instruction-after-user-content pattern when a trailing user
turn is prefixed with ``[System Message]``. Kickoff directives belong in leading
system prompts for the meta agent and all child memory agents. This test fails
if the literal reappears outside the small, documented allowlist (image-deletion
placeholders in LLM client adapters).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MIRIX_ROOT = _REPO_ROOT / "mirix"

# The only permitted production use: replaced-image placeholder copy in LLM
# client adapters (not the GenSRF injection pattern).
_ALLOWED_IMAGE_DELETION_SUFFIX = (
    "There was an image here but now the image has been deleted to save space."
)
_ALLOWED_LLMAPI_FILES = {
    "openai_client.py",
    "google_ai_client.py",
    "anthropic_client.py",
}


def _scan_paths() -> list[Path]:
    paths: list[Path] = []
    paths.extend(_MIRIX_ROOT.rglob("*.py"))
    paths.extend((_MIRIX_ROOT / "prompts").rglob("*.txt"))
    return sorted(set(paths))


def _is_allowed_occurrence(path: Path, line: str) -> bool:
    if "[System Message]" not in line:
        return True
    if path.parent.name == "llm_api" and path.name in _ALLOWED_LLMAPI_FILES:
        return _ALLOWED_IMAGE_DELETION_SUFFIX in line
    return False


def test_production_code_has_no_system_message_literal_outside_allowlist():
    """Fail if ``[System Message]`` appears anywhere under mirix/ except the
    image-deletion placeholders in llm_api client adapters."""
    violations: list[str] = []
    for path in _scan_paths():
        rel = path.relative_to(_REPO_ROOT)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "[System Message]" in line and not _is_allowed_occurrence(path, line):
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert not violations, (
        "Unexpected '[System Message]' literal(s) in production code. "
        "Relocate instructions into system prompts instead:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "memory_type,agent_type_str",
    [
        ("core", "core_memory_agent"),
        ("episodic", "episodic_memory_agent"),
        ("procedural", "procedural_memory_agent"),
        ("resource", "resource_memory_agent"),
        ("semantic", "semantic_memory_agent"),
        ("knowledge_vault", "knowledge_vault_memory_agent"),
    ],
)
def test_child_memory_agent_system_prompt_has_no_system_message_prefix(
    memory_type, agent_type_str
):
    """Each child memory agent kickoff lives in the leading system prompt
    without a ``[System Message]`` prefix (base + screen_monitor variants)."""
    from mirix.prompts import gpt_system

    for variant in ("base", "screen_monitor"):
        system_prompt = gpt_system.get_system_text(f"{variant}/{agent_type_str}")
        assert "update the corresponding memory" in system_prompt, (
            f"{variant}/{agent_type_str} missing kickoff directive"
        )
        assert "[System Message]" not in system_prompt, (
            f"{variant}/{agent_type_str} still contains '[System Message]'"
        )
