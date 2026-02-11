from typing import TYPE_CHECKING, List, Optional

from jinja2 import Environment, Template, TemplateSyntaxError
from pydantic import BaseModel, Field

# Forward referencing to avoid circular import with Agent -> Memory -> Agent
if TYPE_CHECKING:
    from mirix.schemas.agent import AgentState

from mirix.constants import CORE_MEMORY_BLOCK_CHAR_LIMIT
from mirix.schemas.block import Block
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_request import Tool
from mirix.schemas.user import User as PydanticUser


class ContextWindowOverview(BaseModel):
    """
    Overview of the context window, including the number of messages and tokens.
    """

    # top-level information
    context_window_size_max: int = Field(..., description="The maximum amount of tokens the context window can hold.")
    context_window_size_current: int = Field(..., description="The current number of tokens in the context window.")

    # context window breakdown (in messages)
    # (technically not in the context window, but useful to know)
    num_messages: int = Field(..., description="The number of messages in the context window.")
    num_archival_memory: int = Field(..., description="The number of messages in the archival memory.")
    num_recall_memory: int = Field(..., description="The number of messages in the recall memory.")
    num_tokens_external_memory_summary: int = Field(
        ...,
        description="The number of tokens in the external memory summary (archival + recall metadata).",
    )
    external_memory_summary: str = Field(
        ...,
        description="The metadata summary of the external memory sources (archival + recall metadata).",
    )

    # context window breakdown (in tokens)
    # this should all add up to context_window_size_current

    num_tokens_system: int = Field(..., description="The number of tokens in the system prompt.")
    system_prompt: str = Field(..., description="The content of the system prompt.")

    num_tokens_core_memory: int = Field(..., description="The number of tokens in the core memory.")
    core_memory: str = Field(..., description="The content of the core memory.")

    num_tokens_summary_memory: int = Field(..., description="The number of tokens in the summary memory.")
    summary_memory: Optional[str] = Field(None, description="The content of the summary memory.")

    num_tokens_functions_definitions: int = Field(..., description="The number of tokens in the functions definitions.")
    functions_definitions: Optional[List[Tool]] = Field(..., description="The content of the functions definitions.")

    num_tokens_messages: int = Field(..., description="The number of tokens in the messages list.")
    # TODO make list of messages?
    # messages: List[dict] = Field(..., description="The messages in the context window.")
    messages: List[Message] = Field(..., description="The messages in the context window.")


def line_numbers(value: str, prefix: str = "Line ") -> str:
    """
    Turn
        "a\nb"
    into
        "Line 1:\ta\nLine 2:\tb"
    """
    return "\n".join(f"{prefix}{idx + 1}:\t{line}" for idx, line in enumerate(value.splitlines()))


# Build an environment and add a custom filter
env = Environment()
env.filters["line_numbers"] = line_numbers


class Memory(BaseModel, validate_assignment=True):
    """

    Represents the in-context memory (i.e. Core memory) of the agent. This includes both the `Block` objects (labelled by sections), as well as tools to edit the blocks.

    """

    # Memory.block contains the list of memory blocks in the core memory
    blocks: List[Block] = Field(..., description="Memory blocks contained in the agent's in-context memory")

    # Memory.template is a Jinja2 template for compiling memory module into a prompt string.
    prompt_template: str = Field(
        default="{% for block in blocks %}"
        "{% set percentage = ((block.value|length / block.limit) * 100)|int %}"
        "{% set status = '' %}"
        "{% if percentage >= 90 %}"
        "{% set status = ' WARNING: NEARLY FULL - USE core_memory_rewrite TO CONDENSE' %}"
        "{% elif percentage >= 75 %}"
        "{% set status = ' Getting Full - Consider Rewriting Soon' %}"
        "{% endif %}"
        '<{{ block.label }} characters="{{ block.value|length }}/{{ block.limit }}" ({{ percentage }}% full){{ status }}>\n'
        "{{ block.value|line_numbers }}\n"
        "</{{ block.label }}>"
        "{% if not loop.last %}\n{% endif %}"
        "{% endfor %}",
        description="Jinja2 template for compiling memory blocks into a prompt string with usage statistics",
    )

    def get_prompt_template(self) -> str:
        """Return the current Jinja2 template string."""
        return str(self.prompt_template)

    def set_prompt_template(self, prompt_template: str):
        """
        Set a new Jinja2 template string.
        Validates the template syntax and compatibility with current memory structure.
        """
        try:
            # Validate Jinja2 syntax
            Template(prompt_template)

            # Validate compatibility with current memory structure
            # Defensive: Handle None blocks
            blocks = self.blocks if self.blocks is not None else []
            Template(prompt_template).render(blocks=blocks)

            # If we get here, the template is valid and compatible
            self.prompt_template = prompt_template
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid Jinja2 template syntax: {str(e)}")
        except Exception as e:
            raise ValueError(f"Prompt template is not compatible with current memory structure: {str(e)}")

    def get_block_usage_stats(self, label: str) -> dict:
        """Get usage statistics for a specific block.

        Args:
            label: The label of the block to get stats for

        Returns:
            dict: Dictionary containing current_size, limit, percentage, and status
        """
        block = self.get_block(label)
        current_size = len(block.value)
        limit = block.limit
        percentage = int((current_size / limit) * 100)

        if percentage >= 90:
            status = "critical"
            recommendation = "USE core_memory_rewrite NOW"
        elif percentage >= 75:
            status = "warning"
            recommendation = "Consider using core_memory_rewrite soon"
        elif percentage >= 50:
            status = "moderate"
            recommendation = "Healthy usage"
        else:
            status = "good"
            recommendation = "Plenty of space available"

        return {
            "current_size": current_size,
            "limit": limit,
            "percentage": percentage,
            "status": status,
            "recommendation": recommendation,
        }

    def compile(self) -> str:
        """Generate a string representation of the memory in-context using the Jinja2 template"""
        template = env.from_string(self.prompt_template)
        # Defensive: Handle None blocks (backward compatibility with old cached agents)
        blocks = self.blocks if self.blocks is not None else []
        return template.render(blocks=blocks)

    def list_block_labels(self) -> List[str]:
        """Return a list of the block names held inside the memory object"""
        # return list(self.memory.keys())
        # Defensive: Handle None blocks
        if self.blocks is None:
            return []
        return [block.label for block in self.blocks]

    # TODO: these should actually be label, not name
    def get_block(self, label: str) -> Block:
        """Correct way to index into the memory.memory field, returns a Block"""
        keys = []
        # Defensive: Handle None blocks
        blocks = self.blocks if self.blocks is not None else []
        for block in blocks:
            if block.label == label:
                return block
            keys.append(block.label)
        raise KeyError(f"Block field {label} does not exist (available sections = {', '.join(keys)})")

    def get_blocks(self) -> List[Block]:
        """Return a list of the blocks held inside the memory object"""
        # return list(self.memory.values())
        # Defensive: Handle None blocks
        return self.blocks if self.blocks is not None else []

    def set_block(self, block: Block):
        """Set a block in the memory object"""
        # Defensive: Handle None blocks - initialize if needed
        if self.blocks is None:
            self.blocks = []
        for i, b in enumerate(self.blocks):
            if b.label == block.label:
                self.blocks[i] = block
                return
        self.blocks.append(block)

    def update_block_value(self, label: str, value: str):
        """Update the value of a block"""
        if not isinstance(value, str):
            raise ValueError("Provided value must be a string")

        # Defensive: Handle None blocks - initialize if needed
        if self.blocks is None:
            self.blocks = []

        for block in self.blocks:
            if block.label == label:
                block.value = value
                return
        raise ValueError(f"Block with label {label} does not exist")


# TODO: ideally this is refactored into ChatMemory and the subclasses are given more specific names.
class BasicBlockMemory(Memory):
    """
    BasicBlockMemory is a basic implemention of the Memory class, which takes in a list of blocks and links them to the memory object. These are editable by the agent via the core memory functions.

    Attributes:
        memory (Dict[str, Block]): Mapping from memory block section to memory block.

    Methods:
        core_memory_append: Append to the contents of core memory.
        core_memory_rewrite: Rewrite the contents of core memory.
    """

    def __init__(self, blocks: List[Block] = []):
        """
        Initialize the BasicBlockMemory object with a list of pre-defined blocks.

        Args:
            blocks (List[Block]): List of blocks to be linked to the memory object.
        """
        super().__init__(blocks=blocks)

    def core_memory_append(agent_state: "AgentState", label: str, content: str) -> Optional[str]:  # type: ignore
        """
        Append to the contents of core memory.

        Args:
            label (str): Section of the memory to be edited (persona or human).
            content (str): Content to write to the memory. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        current_value = str(agent_state.memory.get_block(label).value)
        new_value = current_value + "\n" + str(content)
        agent_state.memory.update_block_value(label=label, value=new_value)
        return None

    # def core_memory_replace(agent_state: "AgentState", label: str, old_content: str, new_content: str) -> Optional[str]:  # type: ignore
    #     """
    #     Replace the contents of core memory. To delete memories, use an empty string for new_content.

    #     Args:
    #         label (str): Section of the memory to be edited (persona or human).
    #         old_content (str): String to replace. Must be an exact match.
    #         new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    #     Returns:
    #         Optional[str]: None is always returned as this function does not produce a response.
    #     """
    #     current_value = str(agent_state.memory.get_block(label).value)
    #     if old_content not in current_value:
    #         raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
    #     new_value = current_value.replace(str(old_content), str(new_content))
    #     agent_state.memory.update_block_value(label=label, value=new_value)
    #     return None


class ChatMemory(BasicBlockMemory):
    """
    ChatMemory initializes a BaseChatMemory with two default blocks, `human` and `persona`.
    """

    def __init__(
        self,
        persona: str,
        human: str,
        user: PydanticUser,
        limit: int = CORE_MEMORY_BLOCK_CHAR_LIMIT,
    ):
        """
        Initialize the ChatMemory object with a persona and human string.

        Args:
            persona (str): The starter value for the persona block.
            human (str): The starter value for the human block.
            user (PydanticUser): The user who owns this memory (used for user_id on blocks).
            limit (int): The character limit for each block.
        """
        # TODO: Should these be CreateBlocks?
        super().__init__(
            blocks=[
                Block(value=persona, limit=limit, label="persona", user_id=user.id),
                Block(value=human, limit=limit, label="human", user_id=user.id),
            ]
        )


class UpdateMemory(BaseModel):
    """Update the memory of the agent"""


class ArchivalMemorySummary(BaseModel):
    size: int = Field(..., description="Number of rows in archival memory")


class RecallMemorySummary(BaseModel):
    size: int = Field(..., description="Number of rows in recall memory")


class CreateArchivalMemory(BaseModel):
    text: str = Field(..., description="Text to write to archival memory.")
