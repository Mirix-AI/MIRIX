from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    try:
        from composio import ActionType
    except ImportError:
        ActionType = Any  # type: ignore
    try:
        from crewai_tools import BaseTool as CrewAIBaseTool
    except ImportError:
        CrewAIBaseTool = Any  # type: ignore
    try:
        from langchain_core.tools import BaseTool as LangChainBaseTool
    except ImportError:
        LangChainBaseTool = Any  # type: ignore
from mirix.constants import FUNCTION_RETURN_CHAR_LIMIT
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.block import Human, Persona
from mirix.schemas.embedding_config import EmbeddingConfig

# new schemas
from mirix.schemas.environment_variables import SandboxEnvironmentVariable
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary, Memory, RecallMemorySummary
from mirix.schemas.message import Message
from mirix.schemas.mirix_response import MirixResponse
from mirix.schemas.organization import Organization
from mirix.schemas.sandbox_config import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfig,
)
from mirix.schemas.tool import Tool
from mirix.schemas.tool_rule import BaseToolRule


class AbstractClient(object):
    def __init__(
        self,
        debug: bool = False,
    ):
        self.debug = debug

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        raise NotImplementedError

    def create_agent(
        self,
        name: Optional[str] = None,
        agent_type: Optional[AgentType] = AgentType.chat_agent,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        memory: Optional[Memory] = None,
        block_ids: Optional[List[str]] = None,
        system: Optional[str] = None,
        tool_ids: Optional[List[str]] = None,
        tool_rules: Optional[List[BaseToolRule]] = None,
        include_base_tools: Optional[bool] = True,
        include_meta_memory_tools: Optional[bool] = False,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
        initial_message_sequence: Optional[List[Message]] = None,
        tags: Optional[List[str]] = None,
    ) -> AgentState:
        raise NotImplementedError

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system: Optional[str] = None,
        tool_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        message_ids: Optional[List[str]] = None,
        memory: Optional[Memory] = None,
        tags: Optional[List[str]] = None,
    ):
        raise NotImplementedError

    def get_tools_from_agent(self, agent_id: str):
        raise NotImplementedError

    def add_tool_to_agent(self, agent_id: str, tool_id: str):
        raise NotImplementedError

    def remove_tool_from_agent(self, agent_id: str, tool_id: str):
        raise NotImplementedError

    def rename_agent(self, agent_id: str, new_name: str):
        raise NotImplementedError

    def delete_agent(self, agent_id: str):
        raise NotImplementedError

    def get_agent(self, agent_id: str) -> AgentState:
        raise NotImplementedError

    def get_agent_id(self, agent_name: str) -> AgentState:
        raise NotImplementedError

    def get_in_context_memory(self, agent_id: str) -> Memory:
        raise NotImplementedError

    def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        raise NotImplementedError

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        raise NotImplementedError

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        raise NotImplementedError

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        raise NotImplementedError

    def send_message(
        self,
        message: str,
        role: str,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        user_id: Optional[str] = None,  # End-user ID for message attribution
        name: Optional[str] = None,
        stream_steps: bool = False,
        stream_tokens: bool = False,
        chaining: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ) -> MirixResponse:
        raise NotImplementedError

    def user_message(self, agent_id: str, message: str) -> MirixResponse:
        raise NotImplementedError

    def create_human(self, name: str, text: str) -> Human:
        raise NotImplementedError

    def create_persona(self, name: str, text: str) -> Persona:
        raise NotImplementedError

    def list_humans(self) -> List[Human]:
        raise NotImplementedError

    def list_personas(self) -> List[Persona]:
        raise NotImplementedError

    def update_human(self, human_id: str, text: str) -> Human:
        raise NotImplementedError

    def update_persona(self, persona_id: str, text: str) -> Persona:
        raise NotImplementedError

    def get_persona(self, id: str) -> Persona:
        raise NotImplementedError

    def get_human(self, id: str) -> Human:
        raise NotImplementedError

    def get_persona_id(self, name: str) -> str:
        raise NotImplementedError

    def get_human_id(self, name: str) -> str:
        raise NotImplementedError

    def delete_persona(self, id: str):
        raise NotImplementedError

    def delete_human(self, id: str):
        raise NotImplementedError

    def load_langchain_tool(
        self,
        langchain_tool: "LangChainBaseTool",
        additional_imports_module_attr_map: dict[str, str] = None,
    ) -> Tool:
        raise NotImplementedError

    def load_composio_tool(self, action: "ActionType") -> Tool:
        raise NotImplementedError

    def create_tool(
        self,
        func,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        raise NotImplementedError

    def create_or_update_tool(
        self,
        func,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        raise NotImplementedError

    def update_tool(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        func: Optional[Callable] = None,
        tags: Optional[List[str]] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        raise NotImplementedError

    def list_tools(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[Tool]:
        raise NotImplementedError

    def get_tool(self, id: str) -> Tool:
        raise NotImplementedError

    def delete_tool(self, id: str):
        raise NotImplementedError

    def get_tool_id(self, name: str) -> Optional[str]:
        raise NotImplementedError

    def upsert_base_tools(self) -> List[Tool]:
        raise NotImplementedError

    def get_messages(
        self,
        agent_id: str,
        before: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = 1000,
    ) -> List[Message]:
        raise NotImplementedError

    def list_model_configs(self) -> List[LLMConfig]:
        raise NotImplementedError

    def list_embedding_configs(self) -> List[EmbeddingConfig]:
        raise NotImplementedError

    def create_org(self, name: Optional[str] = None) -> Organization:
        raise NotImplementedError

    def list_orgs(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[Organization]:
        raise NotImplementedError

    def delete_org(self, org_id: str) -> Organization:
        raise NotImplementedError

    def create_sandbox_config(self, config: Union[LocalSandboxConfig, E2BSandboxConfig]) -> SandboxConfig:
        """
        Create a new sandbox configuration.

        Args:
            config (Union[LocalSandboxConfig, E2BSandboxConfig]): The sandbox settings.

        Returns:
            SandboxConfig: The created sandbox configuration.
        """
        raise NotImplementedError

    def update_sandbox_config(
        self,
        sandbox_config_id: str,
        config: Union[LocalSandboxConfig, E2BSandboxConfig],
    ) -> SandboxConfig:
        """
        Update an existing sandbox configuration.

        Args:
            sandbox_config_id (str): The ID of the sandbox configuration to update.
            config (Union[LocalSandboxConfig, E2BSandboxConfig]): The updated sandbox settings.

        Returns:
            SandboxConfig: The updated sandbox configuration.
        """
        raise NotImplementedError

    def delete_sandbox_config(self, sandbox_config_id: str) -> None:
        """
        Delete a sandbox configuration.

        Args:
            sandbox_config_id (str): The ID of the sandbox configuration to delete.
        """
        raise NotImplementedError

    def list_sandbox_configs(self, limit: int = 50, cursor: Optional[str] = None) -> List[SandboxConfig]:
        """
        List all sandbox configurations.

        Args:
            limit (int, optional): The maximum number of sandbox configurations to return. Defaults to 50.
            cursor (Optional[str], optional): The pagination cursor for retrieving the next set of results.

        Returns:
            List[SandboxConfig]: A list of sandbox configurations.
        """
        raise NotImplementedError

    def create_sandbox_env_var(
        self,
        sandbox_config_id: str,
        key: str,
        value: str,
        description: Optional[str] = None,
    ) -> SandboxEnvironmentVariable:
        """
        Create a new environment variable for a sandbox configuration.

        Args:
            sandbox_config_id (str): The ID of the sandbox configuration to associate the environment variable with.
            key (str): The name of the environment variable.
            value (str): The value of the environment variable.
            description (Optional[str], optional): A description of the environment variable. Defaults to None.

        Returns:
            SandboxEnvironmentVariable: The created environment variable.
        """
        raise NotImplementedError

    def update_sandbox_env_var(
        self,
        env_var_id: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SandboxEnvironmentVariable:
        """
        Update an existing environment variable.

        Args:
            env_var_id (str): The ID of the environment variable to update.
            key (Optional[str], optional): The updated name of the environment variable. Defaults to None.
            value (Optional[str], optional): The updated value of the environment variable. Defaults to None.
            description (Optional[str], optional): The updated description of the environment variable. Defaults to None.

        Returns:
            SandboxEnvironmentVariable: The updated environment variable.
        """
        raise NotImplementedError

    def delete_sandbox_env_var(self, env_var_id: str) -> None:
        """
        Delete an environment variable by its ID.

        Args:
            env_var_id (str): The ID of the environment variable to delete.
        """
        raise NotImplementedError

    def list_sandbox_env_vars(
        self, sandbox_config_id: str, limit: int = 50, cursor: Optional[str] = None
    ) -> List[SandboxEnvironmentVariable]:
        """
        List all environment variables associated with a sandbox configuration.

        Args:
            sandbox_config_id (str): The ID of the sandbox configuration to retrieve environment variables for.
            limit (int, optional): The maximum number of environment variables to return. Defaults to 50.
            cursor (Optional[str], optional): The pagination cursor for retrieving the next set of results.

        Returns:
            List[SandboxEnvironmentVariable]: A list of environment variables.
        """
        raise NotImplementedError
