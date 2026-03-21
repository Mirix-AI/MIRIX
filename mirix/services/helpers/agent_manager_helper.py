import datetime
from typing import List, Literal, Optional

from sqlalchemy import select

from mirix.constants import IN_CONTEXT_MEMORY_KEYWORD, STRUCTURED_OUTPUT_MODELS
from mirix.helpers import ToolRulesSolver
from mirix.orm.agent import Agent as AgentModel
from mirix.orm.errors import NoResultFound
from mirix.prompts import gpt_system
from mirix.schemas.agent import AgentType
from mirix.schemas.memory import Memory
from mirix.schemas.tool_rule import ToolRule
from mirix.utils import get_local_time


async def _process_relationship(
    session,
    agent: AgentModel,
    relationship_name: str,
    model_class,
    item_ids: List[str],
    allow_partial=False,
    replace=True,
):
    """
    Generalized async function to handle relationships like tools, sources, and blocks using item IDs.

    Args:
        session: The async database session.
        agent: The AgentModel instance.
        relationship_name: The name of the relationship attribute (e.g., 'tools', 'sources').
        model_class: The ORM class corresponding to the related items.
        item_ids: List of IDs to set or update.
        allow_partial: If True, allows missing items without raising errors.
        replace: If True, replaces the entire relationship; otherwise, extends it.

    Raises:
        ValueError: If `allow_partial` is False and some IDs are missing.
    """
    current_relationship = getattr(agent, relationship_name, [])
    if not item_ids:
        if replace:
            setattr(agent, relationship_name, [])
        return

    result = await session.execute(select(model_class).where(model_class.id.in_(item_ids)))
    found_items = result.scalars().all()

    if not allow_partial and len(found_items) != len(item_ids):
        missing = set(item_ids) - {item.id for item in found_items}
        raise NoResultFound(f"Items not found in {relationship_name}: {missing}")

    if replace:
        setattr(agent, relationship_name, found_items)
    else:
        current_ids = {item.id for item in current_relationship}
        new_items = [item for item in found_items if item.id not in current_ids]
        current_relationship.extend(new_items)


def derive_system_message(agent_type: AgentType, system: Optional[str] = None):
    if system is None:
        # Map agent types to their corresponding system prompt paths
        if agent_type == AgentType.chat_agent:
            system = gpt_system.get_system_text("base/chat_agent")
        elif agent_type == AgentType.episodic_memory_agent:
            system = gpt_system.get_system_text("base/episodic_memory_agent")
        elif agent_type == AgentType.procedural_memory_agent:
            system = gpt_system.get_system_text("base/procedural_memory_agent")
        elif agent_type == AgentType.knowledge_vault_memory_agent:
            system = gpt_system.get_system_text("base/knowledge_vault_memory_agent")
        elif agent_type == AgentType.meta_memory_agent:
            system = gpt_system.get_system_text("base/meta_memory_agent")
        elif agent_type == AgentType.semantic_memory_agent:
            system = gpt_system.get_system_text("base/semantic_memory_agent")
        elif agent_type == AgentType.core_memory_agent:
            system = gpt_system.get_system_text("base/core_memory_agent")
        elif agent_type == AgentType.resource_memory_agent:
            system = gpt_system.get_system_text("base/resource_memory_agent")
        elif agent_type == AgentType.reflexion_agent:
            system = gpt_system.get_system_text("base/reflexion_agent")
        elif agent_type == AgentType.background_agent:
            system = gpt_system.get_system_text("base/background_agent")
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

    return system


# TODO: This code is kind of wonky and deserves a rewrite
def compile_memory_metadata_block(
    memory_edit_timestamp: datetime.datetime,
    previous_message_count: int = 0,
    archival_memory_size: int = 0,
) -> str:
    # Put the timestamp in the local timezone (mimicking get_local_time())
    timestamp_str = memory_edit_timestamp.astimezone().strftime("%Y-%m-%d %I:%M:%S %p %Z%z").strip()

    # Create a metadata block of info so the agent knows about the metadata of out-of-context memories
    memory_metadata_block = "\n".join(
        [
            f"### Memory [last modified: {timestamp_str}]",
            f"{previous_message_count} previous messages between you and the user are stored in recall memory (use functions to access them)",
            f"{archival_memory_size} total memories you created are stored in archival memory (use functions to access them)",
            "\nCore memory shown below (limited in size, additional information stored in archival / recall memory):",
        ]
    )
    return memory_metadata_block


def compile_system_message(
    system_prompt: str,
    in_context_memory: Memory,
    in_context_memory_last_edit: datetime.datetime,  # TODO move this inside of BaseMemory?
    user_defined_variables: Optional[dict] = None,
    append_icm_if_missing: bool = True,
    template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
    previous_message_count: int = 0,
    archival_memory_size: int = 0,
) -> str:
    """Prepare the final/full system message that will be fed into the LLM API

    The base system message may be templated, in which case we need to render the variables.

    The following are reserved variables:
      - CORE_MEMORY: the in-context memory of the LLM
    """

    if user_defined_variables is not None:
        # TODO eventually support the user defining their own variables to inject
        raise NotImplementedError
    else:
        variables = {}

    # Add the protected memory variable
    if IN_CONTEXT_MEMORY_KEYWORD in variables:
        raise ValueError(
            f"Found protected variable '{IN_CONTEXT_MEMORY_KEYWORD}' in user-defined vars: {str(user_defined_variables)}"
        )
    else:
        # TODO should this all put into the memory.__repr__ function?
        memory_metadata_string = compile_memory_metadata_block(
            memory_edit_timestamp=in_context_memory_last_edit,
            previous_message_count=previous_message_count,
            archival_memory_size=archival_memory_size,
        )
        full_memory_string = memory_metadata_string + "\n" + in_context_memory.compile()

        # Add to the variables list to inject
        variables[IN_CONTEXT_MEMORY_KEYWORD] = full_memory_string

    if template_format == "f-string":
        # Catch the special case where the system prompt is unformatted
        if append_icm_if_missing:
            memory_variable_string = "{" + IN_CONTEXT_MEMORY_KEYWORD + "}"
            if memory_variable_string not in system_prompt:
                # In this case, append it to the end to make sure memory is still injected
                # warnings.warn(f"{IN_CONTEXT_MEMORY_KEYWORD} variable was missing from system prompt, appending instead")
                system_prompt += "\n" + memory_variable_string

        # render the variables using the built-in templater
        try:
            formatted_prompt = system_prompt.format_map(variables)
        except Exception as e:
            raise ValueError(f"Failed to format system prompt - {str(e)}. System prompt value:\n{system_prompt}")

    else:
        # TODO support for mustache and jinja2
        raise NotImplementedError(template_format)

    return formatted_prompt


def check_supports_structured_output(model: str, tool_rules: List[ToolRule]) -> bool:
    if model not in STRUCTURED_OUTPUT_MODELS:
        if len(ToolRulesSolver(tool_rules=tool_rules).init_tool_rules) > 1:
            raise ValueError(
                "Multiple initial tools are not supported for non-structured models. Please use only one initial tool rule."
            )
        return False
    else:
        return True
