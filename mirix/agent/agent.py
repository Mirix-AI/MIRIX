import asyncio
import copy
import json
import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import httpx
import numpy as np
import pytz

from mirix.agent.tool_validators import validate_tool_args
from mirix.constants import (
    CHAINING_FOR_MEMORY_UPDATE,
    CLI_WARNING_PREFIX,
    ERROR_MESSAGE_PREFIX,
    FIRST_MESSAGE_ATTEMPTS,
    FUNC_FAILED_HEARTBEAT_MESSAGE,
    LLM_MAX_TOKENS,
    MAX_CHAINING_STEPS,
    MAX_EMBEDDING_DIM,
    MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
    MIRIX_CORE_TOOL_MODULE_NAME,
    MIRIX_EXTRA_TOOL_MODULE_NAME,
    MIRIX_MEMORY_TOOL_MODULE_NAME,
    REQ_HEARTBEAT_MESSAGE,
)
from mirix.embeddings import embedding_model
from mirix.errors import ContextWindowExceededError, LLMError
from mirix.functions.functions import get_function_from_module
from mirix.helpers import ToolRulesSolver
from mirix.helpers.message_helpers import prepare_input_message_create
from mirix.interface import AgentInterface
from mirix.llm_api.helpers import get_token_counts_for_messages, is_context_overflow_error
from mirix.llm_api.llm_api_tools import create
from mirix.llm_api.llm_client import LLMClient
from mirix.log import get_logger
from mirix.memory import summarize_messages
from mirix.observability.context import get_trace_context, mark_observation_as_child
from mirix.observability.langfuse_client import get_langfuse_client
from mirix.schemas.agent import AgentState, AgentStepResponse
from mirix.schemas.block import BlockUpdate
from mirix.schemas.client import Client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import MessageRole, ToolType
from mirix.schemas.memory import Memory
from mirix.schemas.message import Message, MessageCreate
from mirix.schemas.mirix_message_content import CloudFileContent, FileContent, ImageContent, TextContent
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse
from mirix.schemas.openai.chat_completion_response import Message as ChatCompletionMessage
from mirix.schemas.openai.chat_completion_response import UsageStatistics
from mirix.schemas.tool import Tool
from mirix.schemas.tool_rule import TerminalToolRule
from mirix.schemas.usage import MirixUsageStatistics
from mirix.schemas.user import User
from mirix.services.agent_manager import AgentManager
from mirix.services.block_manager import BlockManager
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.helpers.agent_manager_helper import check_supports_structured_output
from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
from mirix.services.message_manager import MessageManager
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.resource_memory_manager import ResourceMemoryManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.services.step_manager import StepManager
from mirix.services.tool_execution_sandbox import ToolExecutionSandbox
from mirix.services.user_manager import UserManager
from mirix.settings import settings
from mirix.system import get_contine_chaining, get_token_limit_warning, package_function_response, package_user_message
from mirix.tracing import trace_method
from mirix.utils import (
    convert_timezone_to_utc,
    get_friendly_error_msg,
    get_tool_call_id,
    get_utc_time,
    json_dumps,
    json_loads,
    log_telemetry,
    parse_json,
    printv,
    validate_function_response,
)

# Initialize module-level logger
logger = get_logger(__name__)


def _filter_function_args(
    function_name: str,
    function_args: dict,
    tool: Tool,
) -> dict:
    """
    Filter function arguments to only include parameters accepted by the function.
    Strips hallucinated args like 'internal_monologue' that LLMs sometimes add.

    Args:
        function_name: Name of the function being called
        function_args: Dictionary of arguments from the LLM
        tool: The Tool object containing tool type information

    Returns:
        Filtered dictionary containing only valid arguments
    """
    import inspect

    # Only filter MIRIX internal tools - don't filter USER_DEFINED or MCP tools
    if tool.tool_type == ToolType.MIRIX_CORE:
        callable_func = get_function_from_module(MIRIX_CORE_TOOL_MODULE_NAME, function_name)
    elif tool.tool_type == ToolType.MIRIX_MEMORY_CORE:
        callable_func = get_function_from_module(MIRIX_MEMORY_TOOL_MODULE_NAME, function_name)
    elif tool.tool_type == ToolType.MIRIX_EXTRA:
        callable_func = get_function_from_module(MIRIX_EXTRA_TOOL_MODULE_NAME, function_name)
    else:
        return function_args  # Don't filter USER_DEFINED or MCP tools

    sig = inspect.signature(callable_func)
    valid_params = set(sig.parameters.keys())

    filtered = {}
    removed = []

    for key, value in function_args.items():
        if key in valid_params:
            filtered[key] = value
        else:
            removed.append(key)

    if removed:
        logger.debug(f"Filtered unexpected args from {function_name}: {removed}")

    return filtered


class BaseAgent(ABC):
    """
    Abstract class for all agents.
    Only one interface is required: step.
    """

    @abstractmethod
    def step(
        self,
        messages: Union[Message, List[Message]],
    ) -> MirixUsageStatistics:
        """
        Top-level event message handler for the agent.
        """
        raise NotImplementedError


class Agent(BaseAgent):
    def __init__(
        self,
        interface: Optional[AgentInterface],
        agent_state: AgentState,  # in-memory representation of the agent state (read from multiple tables)
        actor: Client,
        # extras
        first_message_verify_mono: bool = True,  # TODO move to config?
        filter_tags: Optional[dict] = None,  # Filter tags for memory operations
        block_filter_tags: Optional[dict] = None,  # Applied to block filter_tags when core memory agent runs
        block_filter_tags_update_mode: Optional[str] = "merge",  # "merge" or "replace"
        use_cache: bool = True,  # Control Redis cache behavior for this request
        user: Optional[User] = None,  # End-user user
    ):
        # Hold a copy of the state that was used to init the agent
        self.agent_state = agent_state

        # Runtime scratch pad for core memory blocks, populated during step()
        self.blocks_in_memory: Optional[Memory] = None

        self.actor = actor
        # Store filter_tags as a COPY to prevent mutation across agent instances
        from copy import deepcopy

        # Keep None as None, don't convert to empty dict - they have different meanings
        self.filter_tags = deepcopy(filter_tags) if filter_tags is not None else None
        self.block_filter_tags = deepcopy(block_filter_tags) if block_filter_tags is not None else None
        self.block_filter_tags_update_mode = block_filter_tags_update_mode or "merge"
        self.use_cache = use_cache  # Store use_cache for memory operations
        self.user = user  # Store user for end-user tracking
        self.occurred_at = None  # Optional timestamp for episodic memory, set by server if provided

        # Derive block scopes from filter_tags for block_manager.get_blocks() calls.
        # filter_tags["scope"] is the client's write_scope, set by the server when queuing work.
        scope = self.filter_tags.get("scope") if self.filter_tags else None
        self._block_scopes: list[str] | None = [scope] if scope else None

        # Initialize logger early in constructor
        self.logger = logging.getLogger(f"Mirix.Agent.{self.agent_state.name}")
        self.logger.setLevel(logging.INFO)

        if user:
            self.user_id = user.id
        else:
            from mirix.services.user_manager import UserManager

            self.user_id = UserManager().ADMIN_USER_ID

        if actor:
            self.client_id = actor.id
        else:
            from mirix.services.client_manager import ClientManager

            self.client_id = ClientManager().DEFAULT_CLIENT_ID

        # initialize a tool rules solver
        if agent_state.tool_rules:
            # if there are tool rules, log a warning
            for rule in agent_state.tool_rules:
                if not isinstance(rule, TerminalToolRule):
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] WARNING: Tool rules only work reliably for the latest OpenAI models that support structured outputs."
                    )
                    break
        # add default rule for having send_message be a terminal tool
        if agent_state.tool_rules is None:
            agent_state.tool_rules = []

        self.tool_rules_solver = ToolRulesSolver(tool_rules=agent_state.tool_rules)

        # gpt-4, gpt-3.5-turbo, ...
        self.model = self.agent_state.llm_config.model
        self.supports_structured_output = check_supports_structured_output(
            model=self.model, tool_rules=agent_state.tool_rules
        )

        # state managers
        self.block_manager = BlockManager()
        self.agent_manager = AgentManager()

        # Interface must implement:
        # - internal_monologue
        # - assistant_message
        # - function_message
        # ...
        # Different interfaces can handle events differently
        # e.g., print in CLI vs send a discord message with a discord bot
        self.interface = interface

        # Create the persistence manager object based on the AgentState info
        self.message_manager = MessageManager()
        self.agent_manager = AgentManager()
        self.step_manager = StepManager()

        # Create the memory managers
        self.episodic_memory_manager = EpisodicMemoryManager()
        self.knowledge_vault_manager = KnowledgeVaultManager()
        self.procedural_memory_manager = ProceduralMemoryManager()
        self.resource_memory_manager = ResourceMemoryManager()
        self.semantic_memory_manager = SemanticMemoryManager()

        # State needed for contine_chaining pausing

        self.first_message_verify_mono = first_message_verify_mono

        # Controls if the convo memory pressure warning is triggered
        # When an alert is sent in the message queue, set this to True (to avoid repeat alerts)
        # When the summarizer is run, set this back to False (to reset)
        self.agent_alerted_about_memory_pressure = False

        # Load last function response from message history (deferred to first step())
        self.last_function_response = None

        # Logger that the Agent specifically can use, will also report the agent_state ID with the logs
        # Note: Logger is already initialized earlier in constructor

    async def update_memory_if_changed(self, new_memory: Memory) -> bool:
        """
        Update internal memory object and system prompt if there have been modifications.

        Args:
            new_memory (Memory): the new memory object to compare to the current memory object

        Returns:
            modified (bool): whether the memory was updated
        """
        if self.blocks_in_memory is None:
            return False
        if self.blocks_in_memory.compile() != new_memory.compile():
            # update the blocks (LRW) in the DB
            for label in self.blocks_in_memory.list_block_labels():
                updated_value = new_memory.get_block(label).value
                if updated_value != self.blocks_in_memory.get_block(label).value:
                    # update the block if it's changed
                    block_id = self.blocks_in_memory.get_block(label).id
                    block = await self.block_manager.update_block(
                        block_id=block_id,
                        block_update=BlockUpdate(value=updated_value),
                        actor=self.actor,
                        user=self.user,
                    )
                    assert block.user_id == self.user.id
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] INFO: Updated block {label} with value {updated_value} and user {self.user.id}"
                    )

            # refresh memory from DB (using block ids)
            blocks_result = await self.block_manager.get_blocks(
                user=self.user,
                auto_create_from_default=False,  # Don't auto-create here, only in step()
            )
            self.blocks_in_memory = Memory(
                blocks=[await self.block_manager.get_block_by_id(block.id, user=self.user) for block in blocks_result]
            )

            # NOTE: don't do this since re-buildin the memory is handled at the start of the step
            # rebuild memory - this records the last edited timestamp of the memory
            # TODO: pass in update timestamp from block edit time
            return True

        return False

    async def _apply_block_filter_tags(self, blocks: list) -> list:
        """Apply self.block_filter_tags to loaded blocks using the configured update mode.

        Mutates blocks in-place and persists only those whose filter_tags actually
        changed.  ``scope`` is never overwritten — it is always preserved from the
        existing block.

        Returns the same list (for convenience).
        """
        safe_tags = {k: v for k, v in self.block_filter_tags.items() if k != "scope"}
        for block in blocks:
            existing_tags = block.filter_tags or {}
            scope = existing_tags.get("scope")

            if self.block_filter_tags_update_mode == "replace":
                desired = {**safe_tags}
                if scope is not None:
                    desired["scope"] = scope
            else:
                desired = {**existing_tags, **safe_tags}

            if desired != existing_tags:
                block.filter_tags = desired
                await self.block_manager.update_block_filter_tags(
                    block_id=block.id,
                    new_filter_tags=desired,
                    actor=self.actor,
                    user=self.user,
                )
        return blocks

    async def _execute_mcp_tool(
        self,
        function_name: str,
        function_args: dict,
        target_mirix_tool: Tool,
        request_user_confirmation: Optional[Callable] = None,
    ) -> str:
        """Execute MCP tool using the auto-generated async source code."""
        try:
            if function_name == "gmail_native_gmail_send_email" and request_user_confirmation:
                email_details = {
                    "to": function_args.get("to", ""),
                    "subject": function_args.get("subject", ""),
                    "body": function_args.get("body", ""),
                    "cc": function_args.get("cc", []),
                    "bcc": function_args.get("bcc", []),
                    "attachments": function_args.get("attachments", []),
                }

                confirmed = request_user_confirmation("gmail_send", email_details)

                if not confirmed:
                    return "Email send cancelled by user"

            source_code = target_mirix_tool.source_code
            if not source_code:
                return f"Error: MCP tool '{function_name}' has no source code"

            local_namespace = {
                "self": self,
                "agent_state": self.agent_state,
                "Optional": Optional,
            }

            exec(source_code, globals(), local_namespace)

            func_name = function_name.replace(".", "_").replace("-", "_")

            if func_name not in local_namespace:
                return f"Error: Function '{func_name}' not found in MCP tool source code"

            callable_func = local_namespace[func_name]
            function_args["self"] = self
            function_args["agent_state"] = self.agent_state

            result = await callable_func(**function_args)
            return str(result)

        except Exception as e:
            error_msg = f"Error executing MCP tool '{function_name}': {str(e)}"
            printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: {error_msg}")
            return error_msg

    async def execute_tool_and_persist_state(
        self,
        function_name: str,
        function_args: dict,
        target_mirix_tool: Tool,
        display_intermediate_message: Optional[Callable] = None,
        request_user_confirmation: Optional[Callable] = None,
    ) -> str:
        """
        Execute tool modifications and persist the state of the agent.
        Note: only some agent state modifications will be persisted, such as data in the AgentState ORM and block data
        """
        blocks_result = await self.block_manager.get_blocks(
            user=self.user,
            any_scopes=self._block_scopes,
            auto_create_from_default=False,  # Don't auto-create here, only in step()
        )
        self.blocks_in_memory = Memory(blocks=blocks_result)

        # Get Langfuse client for tracing tool executions
        langfuse = get_langfuse_client()
        trace_context = get_trace_context() if langfuse else {}
        trace_id = trace_context.get("trace_id") if trace_context else None
        parent_span_id = trace_context.get("observation_id") if trace_context else None

        # Sanitize args for tracing (exclude 'self')
        args_for_trace = {}
        for key, value in function_args.items():
            if key == "self":
                continue  # Don't include 'self' in trace
            args_for_trace[key] = str(value)

        async def _execute_tool_inner() -> Tuple[str, bool]:
            """Inner function to execute tool. Returns (response, is_error)."""
            nonlocal function_args  # Allow modification of outer function_args
            function_response = ""
            is_error = False

            try:
                if function_name in [
                    "episodic_memory_insert",
                    "episodic_memory_replace",
                    "list_memory_within_timerange",
                ]:
                    key = "items" if function_name == "episodic_memory_insert" else "new_items"
                    if key in function_args:
                        # Need to change the timezone into UTC timezone
                        for item in function_args[key]:
                            if "occurred_at" in item:
                                item["occurred_at"] = convert_timezone_to_utc(
                                    item["occurred_at"],
                                    self.user.timezone,
                                )

                if function_name in [
                    "search_in_memory",
                    "list_memory_within_timerange",
                ]:
                    function_args["timezone_str"] = self.user.timezone

                if target_mirix_tool.tool_type == ToolType.MIRIX_CORE:
                    # base tools are allowed to access the `Agent` object and run on the database
                    callable_func = get_function_from_module(MIRIX_CORE_TOOL_MODULE_NAME, function_name)
                    function_args["self"] = self  # need to attach self to arg since it's dynamically linked
                    if function_name in ["send_message", "send_intermediate_message"]:
                        agent_state_copy = self.agent_state.__deepcopy__()
                        function_args["agent_state"] = (
                            agent_state_copy  # need to attach self to arg since it's dynamically linked
                        )
                    function_response = await callable_func(**function_args)
                    if function_name == "send_intermediate_message":
                        # send intermediate message to the user
                        if display_intermediate_message:
                            display_intermediate_message("response", function_args["message"])

                elif target_mirix_tool.tool_type == ToolType.MIRIX_MEMORY_CORE:
                    callable_func = get_function_from_module(MIRIX_MEMORY_TOOL_MODULE_NAME, function_name)
                    if function_name in ["core_memory_append", "core_memory_rewrite"]:
                        from copy import deepcopy

                        memory_copy = deepcopy(self.blocks_in_memory)
                        function_args["blocks_in_memory"] = memory_copy
                    if function_name in [
                        "check_episodic_memory",
                        "check_semantic_memory",
                    ]:
                        function_args["timezone_str"] = self.user.timezone
                    function_args["self"] = self

                    function_response = await callable_func(**function_args)
                    if function_name in ["core_memory_append", "core_memory_rewrite"]:
                        await self.update_memory_if_changed(memory_copy)

                elif target_mirix_tool.tool_type == ToolType.MIRIX_EXTRA:
                    callable_func = get_function_from_module(MIRIX_EXTRA_TOOL_MODULE_NAME, function_name)
                    function_args["self"] = self  # need to attach self to arg since it's dynamically linked
                    function_response = await callable_func(**function_args)

                elif target_mirix_tool.tool_type == ToolType.USER_DEFINED:
                    agent_state_copy = self.agent_state.__deepcopy__()

                    # Execute user-defined tool in sandbox for security
                    sandbox = ToolExecutionSandbox(
                        tool_name=function_name,
                        args=function_args,
                        actor=self.actor,
                        tool_object=target_mirix_tool,
                    )
                    sandbox_result = await sandbox.run(agent_state=agent_state_copy)
                    function_response = sandbox_result.func_return

                elif target_mirix_tool.tool_type == ToolType.MIRIX_MCP:
                    function_response = await self._execute_mcp_tool(
                        function_name,
                        function_args,
                        target_mirix_tool,
                        request_user_confirmation,
                    )

                else:
                    raise ValueError(f"Tool type {target_mirix_tool.tool_type} not supported")

            except Exception as e:
                # Need to catch error here, or else truncation wont happen
                is_error = True
                function_response = get_friendly_error_msg(
                    function_name=function_name,
                    exception_name=type(e).__name__,
                    exception_message=str(e),
                )

            return function_response, is_error

        # Execute with Langfuse tracing if available
        if langfuse and trace_id:
            from typing import cast

            from langfuse.types import TraceContext

            # Build trace context
            trace_context_dict: dict = {"trace_id": trace_id}
            if parent_span_id:
                trace_context_dict["parent_span_id"] = parent_span_id

            try:
                with langfuse.start_as_current_observation(
                    name=f"tool: {function_name}",
                    as_type="tool",
                    trace_context=cast(TraceContext, trace_context_dict),
                    input={"tool_name": function_name, "args": args_for_trace},
                    metadata={
                        "tool_type": str(target_mirix_tool.tool_type),
                        "tool_name": function_name,
                        "agent_name": self.agent_state.name,
                    },
                ) as span:
                    mark_observation_as_child(span)
                    function_response, is_error = await _execute_tool_inner()

                    span.update(
                        output={
                            "response": str(function_response),
                            "is_error": is_error,
                        },
                        metadata={
                            "tool_type": str(target_mirix_tool.tool_type),
                            "tool_name": function_name,
                            "is_error": is_error,
                        },
                        level="ERROR" if is_error else "DEFAULT",
                    )
            except Exception as e:
                self.logger.debug(f"Langfuse tool execution trace failed: {e}")
                function_response, _ = await _execute_tool_inner()
        else:
            function_response, _ = await _execute_tool_inner()

        return function_response

    @trace_method
    async def _get_ai_reply(
        self,
        message_sequence: List[Message],
        function_call: Optional[str] = None,
        first_message: bool = False,
        stream: bool = False,  # TODO move to config?
        empty_response_retry_limit: Optional[int] = None,  # Uses settings.llm_retry_limit if None
        backoff_factor: Optional[float] = None,  # Uses settings.llm_retry_backoff_factor if None
        max_delay: Optional[float] = None,  # Uses settings.llm_retry_max_delay if None
        step_count: Optional[int] = None,
        last_function_failed: bool = False,
        get_input_data_for_debugging: bool = False,
        existing_file_uris: Optional[List[str]] = None,
        second_try: bool = False,
        llm_client: Optional[LLMClient] = None,
    ) -> ChatCompletionResponse:
        """Get response from LLM API with robust retry mechanism.

        Retry settings can be configured via environment variables:
        - MIRIX_LLM_RETRY_LIMIT: Max retry attempts (default: 3)
        - MIRIX_LLM_RETRY_BACKOFF_FACTOR: Exponential backoff multiplier (default: 0.5)
        - MIRIX_LLM_RETRY_MAX_DELAY: Max delay between retries in seconds (default: 10.0)
        """
        # Apply defaults from settings if not explicitly provided
        if empty_response_retry_limit is None:
            empty_response_retry_limit = settings.llm_retry_limit
        if backoff_factor is None:
            backoff_factor = settings.llm_retry_backoff_factor
        if max_delay is None:
            max_delay = settings.llm_retry_max_delay

        log_telemetry(self.logger, "_get_ai_reply start")
        allowed_tool_names = self.tool_rules_solver.get_allowed_tool_names(
            last_function_response=self.last_function_response
        )
        agent_state_tool_jsons = [t.json_schema for t in self.agent_state.tools]

        allowed_functions = (
            agent_state_tool_jsons
            if not allowed_tool_names
            else [func for func in agent_state_tool_jsons if func["name"] in allowed_tool_names]
        )

        for func in allowed_functions:
            assert func

        # Don't allow a tool to be called if it failed last time
        if last_function_failed and self.tool_rules_solver.tool_call_history:
            allowed_functions = [
                f for f in allowed_functions if f["name"] != self.tool_rules_solver.tool_call_history[-1]
            ]
            if not allowed_functions:
                return None

        # For the first message, force the initial tool if one is specified
        force_tool_call = None
        if (
            step_count is not None
            and step_count == 0
            and not self.supports_structured_output
            and len(self.tool_rules_solver.init_tool_rules) > 0
        ):
            # TODO: This just seems wrong? What if there are more than 1 init tool rules?
            force_tool_call = self.tool_rules_solver.init_tool_rules[0].tool_name
        # Force a tool call if exactly one tool is specified
        elif step_count is not None and step_count > 0 and len(allowed_tool_names) == 1:
            force_tool_call = allowed_tool_names[0]

        active_llm_client = llm_client or LLMClient.create(
            llm_config=self.agent_state.llm_config,
        )

        for attempt in range(1, empty_response_retry_limit + 1):
            try:
                log_telemetry(self.logger, "_get_ai_reply create start")

                # New LLM client flow
                if active_llm_client and not stream:
                    response = await active_llm_client.send_llm_request(
                        messages=message_sequence,
                        tools=allowed_functions,
                        stream=stream,
                        force_tool_call=force_tool_call,
                        get_input_data_for_debugging=get_input_data_for_debugging,
                        existing_file_uris=existing_file_uris,
                    )

                    if get_input_data_for_debugging:
                        return response

                else:
                    # Fallback to existing flow
                    response = await create(
                        llm_config=self.agent_state.llm_config,
                        messages=message_sequence,
                        user_id=self.agent_state.created_by_id,
                        functions=allowed_functions,
                        # functions_python=self.functions_python, do we need this?
                        function_call=function_call,
                        first_message=first_message,
                        force_tool_call=force_tool_call,
                        stream=stream,
                        stream_interface=self.interface,
                        name=self.agent_state.name,
                    )
                log_telemetry(self.logger, "_get_ai_reply create finish")

                # These bottom two are retryable
                if len(response.choices) == 0 or response.choices[0] is None:
                    raise ValueError(f"API call returned an empty message: {response}")

                for choice in response.choices:
                    if choice.message.content == "" and len(choice.message.tool_calls) == 0:
                        raise ValueError(f"API call returned an empty message: {response}")

                if response.choices[0].finish_reason not in [
                    "stop",
                    "function_call",
                    "tool_calls",
                ]:
                    if response.choices[0].finish_reason == "length":
                        if attempt >= empty_response_retry_limit:
                            raise RuntimeError(
                                "Retries exhausted and no valid response received. Final error: maximum context length exceeded or generated content is too long"
                            )
                        else:
                            delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                            printv(
                                f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {response.choices[0].finish_reason}. Retrying in {delay} seconds..."
                            )
                            await asyncio.sleep(delay)
                            continue
                    else:
                        raise ValueError(f"Bad finish reason from API: {response.choices[0].finish_reason}")
                log_telemetry(self.logger, "_handle_ai_response finish")

            except ValueError as ve:
                # Some upstream libraries raise ValueError() with an empty message, which
                # makes retry logs unhelpful. Always include type + repr for visibility.
                ve_desc = f"{type(ve).__name__}: {ve!r}"
                if attempt >= empty_response_retry_limit:
                    printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: Retry limit reached. Final error: {ve_desc}")
                    log_telemetry(self.logger, "_handle_ai_response finish ValueError")
                    # Log traceback once at the final attempt for actionable debugging.
                    self.logger.exception(
                        "[Mirix.Agent.%s] Retry limit reached (ValueError).",
                        self.agent_state.name,
                    )
                    raise Exception(f"Retries exhausted and no valid response received. Final error: {ve_desc}")
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {ve_desc}. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    continue

            except KeyError as ke:
                # Gemini api sometimes can yield empty response
                # This is a retryable error
                ke_desc = f"{type(ke).__name__}: {ke!r}"
                if attempt >= empty_response_retry_limit:
                    printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: Retry limit reached. Final error: {ke_desc}")
                    log_telemetry(self.logger, "_handle_ai_response finish KeyError")
                    self.logger.exception(
                        "[Mirix.Agent.%s] Retry limit reached (KeyError).",
                        self.agent_state.name,
                    )
                    raise Exception(f"Retries exhausted and no valid response received. Final error: {ke_desc}")
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {ke_desc}. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    continue

            except LLMError as llm_error:
                llm_error_desc = f"{type(llm_error).__name__}: {llm_error!r}"
                if attempt >= empty_response_retry_limit:
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] ERROR: Retry limit reached. Final error: {llm_error_desc}"
                    )
                    log_telemetry(self.logger, "_handle_ai_response finish LLMError")
                    log_telemetry(self.logger, "_get_ai_reply_last_message_hacking start")
                    self.logger.exception(
                        "[Mirix.Agent.%s] Retry limit reached (LLMError).",
                        self.agent_state.name,
                    )
                    if second_try:
                        raise Exception(
                            f"Retries exhausted and no valid response received. Final error: {llm_error_desc}"
                        )
                    return await self._get_ai_reply(
                        [message_sequence[-1]],
                        function_call,
                        first_message,
                        stream,
                        empty_response_retry_limit,
                        backoff_factor,
                        max_delay,
                        step_count,
                        last_function_failed,
                        get_input_data_for_debugging,
                        second_try=True,
                        llm_client=active_llm_client,
                    )

                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {llm_error_desc}. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    continue

            except AssertionError as ae:
                tb_str = traceback.format_exc()
                ae_desc = f"{type(ae).__name__}: {ae!r}\nTraceback:\n{tb_str}"
                if attempt >= empty_response_retry_limit:
                    printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: Retry limit reached. Final error: {ae_desc}")
                    self.logger.exception(
                        "[Mirix.Agent.%s] Retry limit reached (AssertionError).",
                        self.agent_state.name,
                    )
                    raise Exception(f"Retries exhausted and no valid response received. Final error: {ae_desc}")
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {ae_desc}. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    continue

            except httpx.HTTPStatusError as he:
                he_desc = f"{type(he).__name__}: {he!r}"
                if attempt >= empty_response_retry_limit:
                    printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: Retry limit reached. Final error: {he_desc}")
                    self.logger.exception(
                        "[Mirix.Agent.%s] Retry limit reached (HTTPError).",
                        self.agent_state.name,
                    )
                    raise Exception(f"Retries exhausted and no valid response received. Final error: {he_desc}")
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {he_desc}. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    continue

            except Exception as e:
                log_telemetry(self.logger, "_handle_ai_response finish generic Exception")
                # For non-retryable errors, exit immediately
                log_telemetry(self.logger, "_handle_ai_response finish generic Exception")
                raise e

            # return the response
            return response

        log_telemetry(self.logger, "_handle_ai_response finish catch-all exception")
        raise Exception("Retries exhausted and no valid response received.")

    async def _handle_ai_response(
        self,
        input_message: Message,
        response_message: ChatCompletionMessage,  # TODO should we eventually move the Message creation outside of this function?
        existing_file_uris: Optional[List[str]] = None,
        override_tool_call_id: bool = False,
        # If we are streaming, we needed to create a Message ID ahead of time,
        # and now we want to use it in the creation of the Message object
        # TODO figure out a cleaner way to do this
        response_message_id: Optional[str] = None,
        force_response: bool = False,
        retrieved_memories: str = None,
        display_intermediate_message: Optional[Callable] = None,
        request_user_confirmation: Optional[Callable] = None,
        return_memory_types_without_update: bool = False,
        message_queue: Optional[any] = None,
        chaining: bool = True,
    ) -> Tuple[List[Message], bool, bool]:
        """Handles parsing and function execution"""

        # Hacky failsafe for now to make sure we didn't implement the streaming Message ID creation incorrectly
        if response_message_id is not None:
            assert response_message_id.startswith("message-"), response_message_id

        messages = []  # append these to the history when done
        function_name = None

        # Step 2: check if LLM wanted to call a function
        if response_message.function_call or (
            response_message.tool_calls is not None and len(response_message.tool_calls) > 0
        ):
            if response_message.function_call:
                raise DeprecationWarning(response_message)

            assert response_message.tool_calls is not None and len(response_message.tool_calls) > 0

            # Generate UUIDs for tool calls if needed
            if override_tool_call_id or response_message.function_call:
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] WARNING: Overriding the tool call can result in inconsistent tool call IDs during streaming"
                )
                for tool_call in response_message.tool_calls:
                    tool_call.id = get_tool_call_id()  # needs to be a string for JSON
            else:
                for tool_call in response_message.tool_calls:
                    assert tool_call.id is not None  # should be defined

            # Memory agents are instructed to emit only ONE tool call per step.
            # In practice, the LLM can occasionally return multiple tool calls (often duplicates),
            # which can cause non-idempotent operations to fail (e.g., double deletes).
            # To match the prompt contract and keep behavior predictable, truncate to the first.
            from mirix.schemas.agent import AgentType

            memory_agent_types = {
                AgentType.core_memory_agent,
                AgentType.episodic_memory_agent,
                AgentType.procedural_memory_agent,
                AgentType.resource_memory_agent,
                AgentType.knowledge_vault_memory_agent,
                AgentType.semantic_memory_agent,
            }

            if (
                self.agent_state.agent_type in memory_agent_types
                and response_message.tool_calls is not None
                and len(response_message.tool_calls) > 1
            ):
                kept = response_message.tool_calls[0]
                dropped = response_message.tool_calls[1:]
                dropped_desc = [f"{tc.function.name}:{tc.id}" for tc in dropped if tc and tc.function]
                self.logger.warning(
                    "Truncating %d extra tool call(s) for memory agent %s (keeping %s:%s, dropping %s)",
                    len(dropped),
                    self.agent_state.agent_type,
                    kept.function.name if kept and kept.function else None,
                    kept.id if kept else None,
                    dropped_desc,
                )
                response_message.tool_calls = [kept]

            # role: assistant (requesting tool call, set tool call ID)
            messages.append(
                # NOTE: we're recreating the message here
                # TODO should probably just overwrite the fields?
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply

            nonnull_content = False
            if response_message.content:
                # The content if then internal monologue, not chat
                self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])
                # Log inner thoughts for debugging and analysis
                printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Inner thoughts: {response_message.content}")
                # Flag to avoid printing a duplicate if inner thoughts get popped from the function call
                nonnull_content = True

            # Step 3: Process each tool call
            continue_chaining = True
            overall_function_failed = False
            executed_function_names = []  # Track which functions were executed

            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: Processing {len(response_message.tool_calls)} tool call(s)"
            )

            for tool_call_idx, tool_call in enumerate(response_message.tool_calls):
                tool_call_id = tool_call.id
                function_call = tool_call.function
                function_name = function_call.name

                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Processing tool call {tool_call_idx + 1}/{len(response_message.tool_calls)}: {function_name} with tool_call_id: {tool_call_id}"
                )

                # Failure case 1: function name is wrong (not in agent_state.tools)
                target_mirix_tool = None
                for t in self.agent_state.tools:
                    if t.name == function_name:
                        target_mirix_tool = t

                if not target_mirix_tool:
                    error_msg = f"No function named {function_name}"
                    function_response = package_function_response(False, error_msg)
                    messages.append(
                        Message.dict_to_message(
                            agent_id=self.agent_state.id,
                            model=self.model,
                            openai_message_dict={
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                                "tool_call_id": tool_call_id,
                            },
                        )
                    )  # extend conversation with function response
                    self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                    overall_function_failed = True
                    continue  # Continue with next tool call

                # Failure case 2: function name is OK, but function args are bad JSON
                try:
                    raw_function_args = function_call.arguments
                    function_args = parse_json(raw_function_args)
                except Exception:
                    error_msg = (
                        f"Error parsing JSON for function '{function_name}' arguments: {function_call.arguments}"
                    )
                    function_response = package_function_response(False, error_msg)
                    messages.append(
                        Message.dict_to_message(
                            agent_id=self.agent_state.id,
                            model=self.model,
                            openai_message_dict={
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                                "tool_call_id": tool_call_id,
                            },
                        )
                    )  # extend conversation with function response
                    self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                    overall_function_failed = True
                    continue  # Continue with next tool call

                # Filter out unexpected arguments that LLMs sometimes hallucinate
                # (e.g., 'internal_monologue'). This must run BEFORE validators.
                function_args = _filter_function_args(function_name, function_args, target_mirix_tool)

                if function_name == "trigger_memory_update":
                    function_args["user_message"] = {
                        "message": input_message,
                        "existing_file_uris": existing_file_uris,
                        "retrieved_memories": retrieved_memories,
                        "chaining": CHAINING_FOR_MEMORY_UPDATE,
                    }
                    if message_queue is not None:
                        function_args["user_message"]["message_queue"] = message_queue

                elif function_name == "trigger_memory_update_with_instruction":
                    function_args["user_message"] = {
                        "existing_file_uris": existing_file_uris,
                        "retrieved_memories": retrieved_memories,
                    }

                # The content if then internal monologue, not chat
                if response_message.content and not nonnull_content:
                    self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] INFO: Inner thoughts (from function call): {response_message.content}"
                    )

                continue_chaining = True

                # Failure case 3: function arguments fail validation
                validation_error = validate_tool_args(function_name, function_args)
                if validation_error:
                    function_response = package_function_response(False, validation_error)
                    messages.append(
                        Message.dict_to_message(
                            agent_id=self.agent_state.id,
                            model=self.model,
                            openai_message_dict={
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                                "tool_call_id": tool_call_id,
                            },
                        )
                    )
                    self.interface.function_message(f"Validation Error: {validation_error}", msg_obj=messages[-1])
                    overall_function_failed = True
                    continue  # Skip execution, let LLM retry

                # Failure case 5: function failed during execution
                # NOTE: the msg_obj associated with the "Running " message is the prior assistant message, not the function/tool role message
                #       this is because the function/tool role message is only created once the function/tool has executed/returned
                self.interface.function_message(f"Running {function_name}()", msg_obj=messages[-1])

                try:
                    if display_intermediate_message:
                        # send intermediate message to the user
                        display_intermediate_message("internal_monologue", response_message.content)

                    function_response = await self.execute_tool_and_persist_state(
                        function_name,
                        function_args,
                        target_mirix_tool,
                        display_intermediate_message=display_intermediate_message,
                        request_user_confirmation=request_user_confirmation,
                    )

                    if function_name == "send_message" or function_name == "finish_memory_update":
                        assert (
                            tool_call_idx == len(response_message.tool_calls) - 1
                        ), f"{function_name} must be the last tool call"

                    if tool_call_idx == len(response_message.tool_calls) - 1:
                        if function_name == "send_message":
                            continue_chaining = False
                        elif function_name == "finish_memory_update":
                            continue_chaining = False
                        else:
                            continue_chaining = True

                    # handle trunction
                    if function_name in [
                        "conversation_search",
                        "conversation_search_date",
                        "archival_memory_search",
                    ]:
                        # with certain functions we rely on the paging mechanism to handle overflow
                        truncate = False
                    else:
                        # but by default, we add a truncation safeguard to prevent bad functions from
                        # overflow the agent context window
                        truncate = True

                    # get the function response limit
                    return_char_limit = target_mirix_tool.return_char_limit
                    function_response_string = validate_function_response(
                        function_response,
                        return_char_limit=return_char_limit,
                        truncate=truncate,
                    )

                    function_args.pop("self", None)
                    function_response = package_function_response(True, function_response_string)
                    function_failed = False

                except Exception as e:
                    function_args.pop("self", None)
                    # error_msg = f"Error calling function {function_name} with args {function_args}: {str(e)}"
                    # Less detailed - don't provide full args, idea is that it should be in recent context so no need (just adds noise)
                    error_msg = get_friendly_error_msg(
                        function_name=function_name,
                        exception_name=type(e).__name__,
                        exception_message=str(e),
                    )
                    error_msg_user = f"{error_msg}\n{traceback.format_exc()}"
                    printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: {error_msg_user}")
                    function_response = package_function_response(False, error_msg)
                    self.last_function_response = function_response
                    # TODO: truncate error message somehow
                    messages.append(
                        Message.dict_to_message(
                            agent_id=self.agent_state.id,
                            model=self.model,
                            openai_message_dict={
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                                "tool_call_id": tool_call_id,
                            },
                        )
                    )  # extend conversation with function response
                    self.interface.function_message(f"Ran {function_name}()", msg_obj=messages[-1])
                    self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                    overall_function_failed = True
                    continue  # Continue with next tool call

                # Step 4: check if function response is an error
                if function_response_string.startswith(ERROR_MESSAGE_PREFIX):
                    function_response = package_function_response(False, function_response_string)
                    # TODO: truncate error message somehow
                    messages.append(
                        Message.dict_to_message(
                            agent_id=self.agent_state.id,
                            model=self.model,
                            openai_message_dict={
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                                "tool_call_id": tool_call_id,
                            },
                        )
                    )  # extend conversation with function response
                    self.interface.function_message(f"Ran {function_name}()", msg_obj=messages[-1])
                    self.interface.function_message(f"Error: {function_response_string}", msg_obj=messages[-1])
                    overall_function_failed = True
                    continue  # Continue with next tool call

                # If no failures happened along the way: ...
                # Step 5: send the info on the function call and function response to GPT
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(f"Ran {function_name}()", msg_obj=messages[-1])
                self.interface.function_message(f"Success: {function_response_string}", msg_obj=messages[-1])
                self.last_function_response = function_response

                # Track successfully executed function names
                executed_function_names.append(function_name)

            function_failed = overall_function_failed

        else:
            # Standard non-function reply
            # Validate that we have content - LLM returned neither tool_calls nor content
            if not response_message.content:
                raise ValueError(
                    f"LLM returned empty response, " f"no tool_calls and no content. Response: {response_message}"
                )
            messages.append(
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])
            # Log inner thoughts for debugging and analysis
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: Inner thoughts (no function call): {response_message.content}"
            )
            continue_chaining = True
            function_failed = False
            if display_intermediate_message:
                display_intermediate_message("internal_monologue", response_message.content)

        # Update ToolRulesSolver state with last called function
        if function_name is not None:
            self.tool_rules_solver.update_tool_usage(function_name)
            # Update contine_chaining request according to provided tool rules
            if self.tool_rules_solver.has_children_tools(function_name):
                continue_chaining = True
            elif self.tool_rules_solver.is_terminal_tool(function_name):
                continue_chaining = False

        return messages, continue_chaining, function_failed

    async def step(
        self,
        input_messages: Union[Message, MessageCreate, List[Union[Message, MessageCreate]]],
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        actor: Optional["Client"] = None,  # Client
        user: Optional[User] = None,
        **kwargs,
    ) -> MirixUsageStatistics:
        """A "step" is one full invocation of an agent.

        Run Agent.inner_step in a loop, handling chaining via continue_chaining requests and function failures

        Args:
            actor: Client object for write operations (updating messages, agent state) - audit trail
            user: User object for read operations (loading blocks, memory filtering) - data scope
        """

        from mirix.schemas.agent import AgentType

        # chat_agent is deprecated - raise immediately
        if self.agent_state.is_type(AgentType.chat_agent):
            raise NotImplementedError(
                "AgentType.chat_agent is deprecated and no longer supported. " "Use a memory agent type instead."
            )

        if actor is None or user is None:
            raise ValueError("Agent.step requires non-null actor and user.")

        # Store actor/user context for this step invocation.
        self.actor = actor
        self.user = user

        # Special case for Core Memory Agent: load blocks to use later in the step
        if self.agent_state.is_type(AgentType.core_memory_agent):
            # Load existing blocks for this user, scoped by the client's write_scope.
            # auto_create_from_default=True will create blocks from template if they don't exist for this scope.
            # filter_tags_set_on_create is applied only when new blocks are created (e.g. from default template).
            existing_blocks = await self.block_manager.get_blocks(
                user=self.user,
                any_scopes=self._block_scopes,
                filter_tags_set_on_create=self.block_filter_tags,
            )

            # Apply block_filter_tags to existing blocks (merge or replace).
            # Skips blocks whose filter_tags already match the desired state
            # (e.g. blocks just created from template with the same tags).
            if self.block_filter_tags and existing_blocks:
                existing_blocks = await self._apply_block_filter_tags(existing_blocks)

            # Load blocks into memory for core_memory_agent
            self.blocks_in_memory = Memory(blocks=existing_blocks)

        # Reset last function response for this step
        self.last_function_response = None

        max_chaining_steps = max_chaining_steps or MAX_CHAINING_STEPS

        # Normalize to runtime Message objects for downstream prompt assembly.
        raw_input_messages = input_messages
        if not isinstance(raw_input_messages, list):
            raw_input_messages = [raw_input_messages]

        # At the end of this normalization step we will end up with a list containing only one Message object
        # (multiple messages are packed into a single Message object in the upstream caller)
        # The step also converts it from a MessageCreate to a Message object
        # to match compatability with the downstream prompt assembly.
        normalized_input_messages: List[Message] = []
        for m in raw_input_messages:
            if isinstance(m, Message):
                normalized_input_messages.append(m)
            elif isinstance(m, MessageCreate):
                normalized_input_messages.append(
                    prepare_input_message_create(
                        m,
                        self.agent_state.id,
                        wrap_user_message=False,
                        wrap_system_message=True,
                    )
                )
            else:
                raise ValueError("input_messages items must be Message or MessageCreate, " f"got {type(m)}")

        # Read retained history from the parent scope (for sub-agents) or from this
        # agent's scope (for top-level agents/meta). This keeps sub-agent inputs as a
        # single packed message while still providing parent retained context.
        retention = (self.actor.message_set_retention_count or 0) if self.actor else 0
        retention_agent_id = (
            self.agent_state.parent_id or self.agent_state.id
        )  # Retained messages in the DB are associated with the meta agent
        should_read_retention = retention > 0 and self.actor and self.user_id
        is_meta_agent = self.agent_state.is_type(AgentType.meta_memory_agent)
        should_write_retention = retention > 0 and is_meta_agent and self.actor and self.user_id
        retained_input_sets: List[Message] = []
        if should_read_retention:
            retained_input_sets = await self.message_manager.get_messages_for_agent_user(
                agent_id=retention_agent_id,
                user_id=self.user_id,
                actor=self.actor,
                limit=retention,
            )

        # Chaining accumulator for the active agent loop only.
        accumulated: List[Message] = list(retained_input_sets)
        # Persist only the original input payload, never synthetic helper messages
        # appended to iteration messages during meta-agent processing.
        input_messages_for_persistence: List[Message] = list(normalized_input_messages)

        # Initialize the LLM client once per step to reuse across retries.
        llm_client = LLMClient.create(
            llm_config=self.agent_state.llm_config,
        )

        if self.agent_state.is_type(AgentType.meta_memory_agent):
            # Extract topics from retained context + current input messages.
            try:
                # make sure to include both retained context and current input messages in the search topic extraction
                topics = await self._extract_topics_from_messages(retained_input_sets + normalized_input_messages)

                if topics is not None:
                    kwargs["topics"] = topics
                else:
                    printv(f"[Mirix.Agent.{self.agent_state.name}] WARNING: No topics extracted from input")

            except Exception as e:
                printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Error in extracting the topic from the input: {e}")
                pass

        # Main loop:ing
        # Each iteration calls inner_step and then makes a decision about whether to continue chaining
        # or to terminate the step. When chaining, the curren_input_messages are updated to reference
        # a heartbeat message (e.g. "function failed", "continue chaining", etc.) and the previous input messages
        # are added to the in-memory accumulator.
        counter = 0
        total_usage = UsageStatistics()
        step_count = 0
        loop_input_messages: List[Message] = list(normalized_input_messages)
        while True:
            kwargs["first_message"] = False
            kwargs["step_count"] = step_count

            loop_iteration_messages = list(loop_input_messages)
            if self.agent_state.is_type(AgentType.meta_memory_agent) and step_count == 0:
                meta_message = prepare_input_message_create(
                    MessageCreate(
                        role="user",
                        content="[System Message] As the meta memory manager, analyze the provided content and perform your function.",
                        filter_tags=self.filter_tags,
                    ),
                    self.agent_state.id,
                    wrap_user_message=False,
                    wrap_system_message=True,
                )
                loop_iteration_messages.append(meta_message)

            step_response = await self.inner_step(
                messages=loop_iteration_messages,
                accumulated=accumulated,
                chaining=chaining,
                llm_client=llm_client,
                retained_count=len(retained_input_sets),
                **kwargs,
            )

            continue_chaining = step_response.continue_chaining
            function_failed = step_response.function_failed
            usage = step_response.usage

            # Accumulate step messages for next chaining iteration
            accumulated = accumulated + step_response.messages

            step_count += 1
            total_usage += usage
            counter += 1
            self.interface.step_complete()

            # Chain stops
            if not chaining and (not function_failed):
                printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: No chaining, stopping after one step")
                break
            elif max_chaining_steps is not None and counter == max_chaining_steps:
                # Add warning message based on agent type
                if self.agent_state.is_type(AgentType.chat_agent):
                    warning_content = "[System Message] You have reached the maximum chaining steps. Please call 'send_message' to send your response to the user."
                else:
                    warning_content = "[System Message] You have reached the maximum chaining steps. Please call 'finish_memory_update' to end the chaining."
                loop_input_messages = [
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        model=self.model,
                        openai_message_dict={
                            "role": "user",
                            "content": warning_content,
                        },
                    )
                ]
                continue  # give agent one more chance to respond
            elif max_chaining_steps is not None and counter > max_chaining_steps:
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Hit max chaining steps, stopping after {counter} steps"
                )
                break
            elif function_failed:
                assert self.agent_state.created_by_id is not None
                loop_input_messages = [
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        model=self.model,
                        openai_message_dict={
                            "role": "user",  # TODO: change to system?
                            "content": get_contine_chaining(FUNC_FAILED_HEARTBEAT_MESSAGE),
                        },
                    )
                ]
                continue  # always chain
            elif continue_chaining:
                assert self.agent_state.created_by_id is not None
                loop_input_messages = [
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        model=self.model,
                        openai_message_dict={
                            "role": "user",  # TODO: change to system?
                            "content": get_contine_chaining(REQ_HEARTBEAT_MESSAGE),
                        },
                    )
                ]
                continue  # always chain
            # Mirix no-op / yield
            else:
                break

        # Retention write-back: persist input messages and prune old ones if configured
        if should_write_retention and input_messages_for_persistence:
            await self.message_manager.create_many_messages(
                input_messages_for_persistence,
                actor=self.actor,
                client_id=self.client_id,
                user_id=self.user_id,
            )
            await self.message_manager.hard_delete_user_messages_for_agent(
                agent_id=self.agent_state.id,
                user_id=self.user_id,
                actor=self.actor,
                keep_newest_n=retention,
            )

        return MirixUsageStatistics(**total_usage.model_dump(), step_count=step_count)

    async def build_system_prompt_with_memories(
        self,
        raw_system: str,
        topics: Optional[str] = None,
        retrieved_memories: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        """
        Build the complete system prompt by retrieving memories and combining with the raw system prompt.

        Args:
            raw_system (str): The base system prompt
            topics (Optional[str]): Topics to use for memory retrieval
            retrieved_memories (Optional[dict]): Pre-retrieved memories to use instead of fetching new ones

        Returns:
            Tuple[str, dict]: The complete system prompt and the retrieved memories dict
        """
        from mirix.schemas.agent import AgentType

        timezone_str = self.user.timezone

        if retrieved_memories is None:
            retrieved_memories = {}

        if "key_words" in retrieved_memories:
            key_words = retrieved_memories["key_words"]
        else:
            key_words = topics if topics is not None else ""
            retrieved_memories["key_words"] = key_words

        search_method = "bm25"

        # Prepare embedding for semantic search
        if key_words != "" and search_method == "embedding":
            embedded_text = await (await embedding_model(self.agent_state.embedding_config)).get_text_embedding(
                key_words
            )
            embedded_text = np.array(embedded_text)
            embedded_text = np.pad(
                embedded_text,
                (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]),
                mode="constant",
            ).tolist()
        else:
            embedded_text = None

        # Retrieve core memory
        if self.agent_state.is_type(AgentType.core_memory_agent) or "core" not in retrieved_memories:
            blocks_result = await self.block_manager.get_blocks(
                user=self.user,
                auto_create_from_default=False,  # Don't auto-create here, only in step()
            )
            current_persisted_memory = Memory(
                blocks=[
                    b
                    for block in blocks_result
                    if (b := await self.block_manager.get_block_by_id(block.id, user=self.user)) is not None
                ]
            )
            core_memory = current_persisted_memory.compile()
            retrieved_memories["core"] = core_memory

        if (
            self.agent_state.is_type(AgentType.knowledge_vault_memory_agent)
            or "knowledge_vault" not in retrieved_memories
        ):
            if self.agent_state.is_type(AgentType.knowledge_vault_memory_agent, AgentType.reflexion_agent):
                current_knowledge_vault = await self.knowledge_vault_manager.list_knowledge(
                    agent_state=self.agent_state,
                    user=self.user,
                    embedded_text=embedded_text,
                    query=key_words,
                    search_field="caption",
                    search_method=search_method,
                    limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                    timezone_str=timezone_str,
                )
            else:
                current_knowledge_vault = await self.knowledge_vault_manager.list_knowledge(
                    agent_state=self.agent_state,
                    user=self.user,
                    embedded_text=embedded_text,
                    query=key_words,
                    search_field="caption",
                    search_method=search_method,
                    limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                    timezone_str=timezone_str,
                    sensitivity=["low", "medium"],
                )

            knowledge_vault_memory = ""
            if len(current_knowledge_vault) > 0:
                for idx, knowledge_vault_item in enumerate(current_knowledge_vault):
                    knowledge_vault_memory += f"[{idx}] Knowledge Vault Item ID: {knowledge_vault_item.id}; Caption: {knowledge_vault_item.caption}\n"
            retrieved_memories["knowledge_vault"] = {
                "total_number_of_items": await self.knowledge_vault_manager.get_total_number_of_items(user=self.user),
                "current_count": len(current_knowledge_vault),
                "text": knowledge_vault_memory,
            }

        # Retrieve episodic memory
        is_owning_agent = self.agent_state.is_type(AgentType.episodic_memory_agent, AgentType.reflexion_agent)
        if is_owning_agent or "episodic" not in retrieved_memories:
            current_episodic_memory = await self.episodic_memory_manager.list_episodic_memory(
                agent_state=self.agent_state,
                user=self.user,
                limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                timezone_str=timezone_str,
            )
            episodic_memory = ""
            if len(current_episodic_memory) > 0:
                for idx, event in enumerate(current_episodic_memory):
                    if is_owning_agent:
                        episodic_memory += f"[Event ID: {event.id}] Timestamp: {event.occurred_at.strftime('%Y-%m-%d %H:%M:%S')} - {event.summary} (Details: {len(event.details)} Characters)\n"
                    else:
                        episodic_memory += f"[{idx}] Timestamp: {event.occurred_at.strftime('%Y-%m-%d %H:%M:%S')} - {event.summary} (Details: {len(event.details)} Characters)\n"

            recent_episodic_memory = episodic_memory.strip()

            most_relevant_episodic_memory = await self.episodic_memory_manager.list_episodic_memory(
                agent_state=self.agent_state,
                user=self.user,
                embedded_text=embedded_text,
                query=key_words,
                search_field="details",
                search_method=search_method,
                limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                timezone_str=timezone_str,
            )
            most_relevant_episodic_memory_str = ""
            if len(most_relevant_episodic_memory) > 0:
                for idx, event in enumerate(most_relevant_episodic_memory):
                    if is_owning_agent:
                        most_relevant_episodic_memory_str += f"[Event ID: {event.id}] Timestamp: {event.occurred_at.strftime('%Y-%m-%d %H:%M:%S')} - {event.summary}  (Details: {len(event.details)} Characters)\n"
                    else:
                        most_relevant_episodic_memory_str += f"[{idx}] Timestamp: {event.occurred_at.strftime('%Y-%m-%d %H:%M:%S')} - {event.summary}  (Details: {len(event.details)} Characters)\n"
            relevant_episodic_memory = most_relevant_episodic_memory_str.strip()
            retrieved_memories["episodic"] = {
                "total_number_of_items": await self.episodic_memory_manager.get_total_number_of_items(user=self.user),
                "recent_count": len(current_episodic_memory),
                "relevant_count": len(most_relevant_episodic_memory),
                "recent_episodic_memory": recent_episodic_memory,
                "relevant_episodic_memory": relevant_episodic_memory,
            }

        # Retrieve resource memory
        # Owning agents need IDs for merge/update operations, so always retrieve fresh
        is_owning_agent = self.agent_state.is_type(AgentType.resource_memory_agent, AgentType.reflexion_agent)
        if is_owning_agent or "resource" not in retrieved_memories:
            current_resource_memory = await self.resource_memory_manager.list_resources(
                agent_state=self.agent_state,
                user=self.user,
                query=key_words,
                embedded_text=embedded_text,
                search_field="summary",
                search_method=search_method,
                limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                timezone_str=timezone_str,
            )
            resource_memory = ""
            if len(current_resource_memory) > 0:
                for idx, resource in enumerate(current_resource_memory):
                    if is_owning_agent:
                        resource_memory += f"[Resource ID: {resource.id}] Resource Title: {resource.title}; Resource Summary: {resource.summary} Resource Type: {resource.resource_type}\n"
                    else:
                        resource_memory += f"[{idx}] Resource Title: {resource.title}; Resource Summary: {resource.summary} Resource Type: {resource.resource_type}\n"
            resource_memory = resource_memory.strip()
            retrieved_memories["resource"] = {
                "total_number_of_items": await self.resource_memory_manager.get_total_number_of_items(user=self.user),
                "current_count": len(current_resource_memory),
                "text": resource_memory,
            }

        # Retrieve procedural memory
        # Owning agents need IDs for merge/update operations, so always retrieve fresh
        is_owning_agent = self.agent_state.is_type(AgentType.procedural_memory_agent, AgentType.reflexion_agent)
        if is_owning_agent or "procedural" not in retrieved_memories:
            current_procedural_memory = await self.procedural_memory_manager.list_procedures(
                agent_state=self.agent_state,
                user=self.user,
                query=key_words,
                embedded_text=embedded_text,
                search_field="summary",
                search_method=search_method,
                limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                timezone_str=timezone_str,
            )
            procedural_memory = ""
            if len(current_procedural_memory) > 0:
                for idx, procedure in enumerate(current_procedural_memory):
                    if is_owning_agent:
                        procedural_memory += f"[Procedure ID: {procedure.id}] Entry Type: {procedure.entry_type}; Summary: {procedure.summary}\n"
                    else:
                        procedural_memory += (
                            f"[{idx}] Entry Type: {procedure.entry_type}; Summary: {procedure.summary}\n"
                        )
            procedural_memory = procedural_memory.strip()
            retrieved_memories["procedural"] = {
                "total_number_of_items": await self.procedural_memory_manager.get_total_number_of_items(user=self.user),
                "current_count": len(current_procedural_memory),
                "text": procedural_memory,
            }

        # Retrieve semantic memory
        # Owning agents need IDs for merge/update operations, so always retrieve fresh
        is_owning_agent = self.agent_state.is_type(AgentType.semantic_memory_agent, AgentType.reflexion_agent)
        if is_owning_agent or "semantic" not in retrieved_memories:
            current_semantic_memory = await self.semantic_memory_manager.list_semantic_items(
                agent_state=self.agent_state,
                user=self.user,
                query=key_words,
                embedded_text=embedded_text,
                search_field="details",
                search_method=search_method,
                limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                timezone_str=timezone_str,
            )
            semantic_memory = ""
            if len(current_semantic_memory) > 0:
                for idx, semantic_memory_item in enumerate(current_semantic_memory):
                    if is_owning_agent:
                        semantic_memory += f"[Semantic Memory ID: {semantic_memory_item.id}] Name: {semantic_memory_item.name}; Summary: {semantic_memory_item.summary}\n"
                    else:
                        semantic_memory += (
                            f"[{idx}] Name: {semantic_memory_item.name}; Summary: {semantic_memory_item.summary}\n"
                        )

            semantic_memory = semantic_memory.strip()
            retrieved_memories["semantic"] = {
                "total_number_of_items": await self.semantic_memory_manager.get_total_number_of_items(user=self.user),
                "current_count": len(current_semantic_memory),
                "text": semantic_memory,
            }

        # Build the complete system prompt
        memory_system_prompt = self.build_system_prompt(retrieved_memories)

        complete_system_prompt = raw_system + "\n\n" + memory_system_prompt

        if key_words:
            complete_system_prompt += "\n\nThe above memories were retrieved based on the previously listed keywords. If some memories are empty or do not contain the content related to the keywords, it is highly likely that memory does not contain any relevant information."

        return complete_system_prompt, retrieved_memories

    def build_system_prompt(self, retrieved_memories: dict) -> str:
        """Build the system prompt for the LLM API"""
        template = """Current Time: {current_time}

User Focus:
<keywords>
{keywords}
</keywords>
These keywords have been used to retrieve relevant memories from the database.

<core_memory>
{core_memory}
</core_memory>

<episodic_memory> Most Recent Events (Orderred by Timestamp):
{episodic_memory}
</episodic_memory>
"""
        user_timezone_str = self.user.timezone
        user_tz = pytz.timezone(user_timezone_str.split(" (")[0])
        current_time = datetime.now(user_tz).strftime("%Y-%m-%d %H:%M:%S")

        keywords = retrieved_memories["key_words"]
        core_memory = retrieved_memories["core"]
        episodic_memory = retrieved_memories["episodic"]
        resource_memory = retrieved_memories["resource"]
        semantic_memory = retrieved_memories["semantic"]
        procedural_memory = retrieved_memories["procedural"]
        knowledge_vault = retrieved_memories["knowledge_vault"]

        system_prompt = template.format(
            current_time=current_time,
            keywords=keywords,
            core_memory=core_memory if core_memory else "Empty",
            episodic_memory=(episodic_memory["recent_episodic_memory"] if episodic_memory else "Empty"),
        )

        if keywords is not None:
            episodic_total = episodic_memory["total_number_of_items"] if episodic_memory else 0
            relevant_episodic_text = episodic_memory["relevant_episodic_memory"] if episodic_memory else ""
            relevant_count = episodic_memory["relevant_count"] if episodic_memory else 0

            system_prompt += (
                f"\n<episodic_memory> Most Relevant Events ({relevant_count} out of {episodic_total} Events Orderred by Relevance to Keywords):\n"
                + (relevant_episodic_text if relevant_episodic_text else "Empty")
                + "\n</episodic_memory>\n"
            )

        # Add knowledge vault with counts
        knowledge_vault_total = knowledge_vault["total_number_of_items"] if knowledge_vault else 0
        knowledge_vault_text = knowledge_vault["text"] if knowledge_vault else ""
        knowledge_vault_count = knowledge_vault["current_count"] if knowledge_vault else 0
        system_prompt += (
            f"\n<knowledge_vault> ({knowledge_vault_count} out of {knowledge_vault_total} Items):\n"
            + (knowledge_vault_text if knowledge_vault_text else "Empty")
            + "\n</knowledge_vault>\n"
        )

        # Add semantic memory with counts
        semantic_total = semantic_memory["total_number_of_items"] if semantic_memory else 0
        semantic_text = semantic_memory["text"] if semantic_memory else ""
        semantic_count = semantic_memory["current_count"] if semantic_memory else 0
        system_prompt += (
            f"\n<semantic_memory> ({semantic_count} out of {semantic_total} Items):\n"
            + (semantic_text if semantic_text else "Empty")
            + "\n</semantic_memory>\n"
        )

        # Add resource memory with counts
        resource_total = resource_memory["total_number_of_items"] if resource_memory else 0
        resource_text = resource_memory["text"] if resource_memory else ""
        resource_count = resource_memory["current_count"] if resource_memory else 0
        system_prompt += (
            f"\n<resource_memory> ({resource_count} out of {resource_total} Items):\n"
            + (resource_text if resource_text else "Empty")
            + "\n</resource_memory>\n"
        )

        # Add procedural memory with counts
        procedural_total = procedural_memory["total_number_of_items"] if procedural_memory else 0
        procedural_text = procedural_memory["text"] if procedural_memory else ""
        procedural_count = procedural_memory["current_count"] if procedural_memory else 0
        system_prompt += (
            f"\n<procedural_memory> ({procedural_count} out of {procedural_total} Items):\n"
            + (procedural_text if procedural_text else "Empty")
            + "\n</procedural_memory>"
        )

        return system_prompt

    async def extract_memory_for_system_prompt(self, message: str) -> str:
        """
        Extract topics from the message and build the memory system prompt without raw_system.
        This is similar to construct_system_message but returns only the memory portion.

        Args:
            message (str): The message to extract topics from

        Returns:
            str: The memory system prompt (without raw_system prefix)
        """
        topics = await self._extract_topics_from_message(message)

        retrieved_memories = await self._retrieve_memories_for_topics(topics)
        memory_system_prompt = self.build_system_prompt(retrieved_memories)

        return memory_system_prompt

    async def _extract_topics_from_message(self, message: str) -> Optional[str]:
        """
        Extract topics from a message using LLM.

        Args:
            message (str): The message to extract topics from

        Returns:
            Optional[str]: Extracted topics or None if extraction fails
        """
        temporary_messages = [
            prepare_input_message_create(
                MessageCreate(
                    role=MessageRole.user,
                    content=message,
                ),
                self.agent_state.id,
                wrap_user_message=False,
                wrap_system_message=True,
            )
        ]

        return await self._extract_topics_from_messages(temporary_messages)

    async def _extract_topics_from_messages(self, messages: List[Message]) -> Optional[str]:
        """
        Extract topics from a list of messages using LLM.

        Args:
            messages (List[Message]): The messages to extract topics from

        Returns:
            Optional[str]: Extracted topics or None if extraction fails
        """
        try:
            # Add instruction message for topic extraction
            temporary_messages = copy.deepcopy(messages)
            temporary_messages.append(
                prepare_input_message_create(
                    MessageCreate(
                        role=MessageRole.user,
                        content="The above are the inputs from the user, please look at these content and extract the topic (brief description of what the user is focusing on) from these content. If there are multiple focuses in these content, then extract them all and put them into one string separated by ';'. Call the function `update_topic` to update the topic with the extracted topics.",
                    ),
                    self.agent_state.id,
                    wrap_user_message=False,
                    wrap_system_message=True,
                )
            )

            temporary_messages = [
                prepare_input_message_create(
                    MessageCreate(
                        role=MessageRole.system,
                        content="You are a helpful assistant that extracts the topic from the user's input.",
                    ),
                    self.agent_state.id,
                    wrap_user_message=False,
                    wrap_system_message=True,
                ),
            ] + temporary_messages

            # Define the function for topic extraction
            functions = [
                {
                    "name": "update_topic",
                    "description": "Update the topic of the conversation/content. The topic will be used for retrieving relevant information from the database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": 'The topic of the current conversation/content. If there are multiple topics then separate them with ";".',
                            }
                        },
                        "required": ["topic"],
                    },
                }
            ]

            # Use LLMClient to extract topics (run async in event loop from sync context)
            llm_client = LLMClient.create(
                llm_config=self.agent_state.llm_config,
            )

            if llm_client:
                response = await llm_client.send_llm_request(
                    messages=temporary_messages,
                    tools=functions,
                    stream=False,
                    force_tool_call="update_topic",
                )
            else:
                response = await create(
                    llm_config=self.agent_state.llm_config,
                    messages=temporary_messages,
                    functions=functions,
                    force_tool_call="update_topic",
                )

            # Extract topics from the response
            for choice in response.choices:
                if (
                    hasattr(choice.message, "tool_calls")
                    and choice.message.tool_calls is not None
                    and len(choice.message.tool_calls) > 0
                ):
                    try:
                        function_args = json.loads(choice.message.tool_calls[0].function.arguments)
                        topics = function_args.get("topic")
                        printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Extracted topics: {topics}")
                        return topics
                    except (json.JSONDecodeError, KeyError) as parse_error:
                        printv(
                            f"[Mirix.Agent.{self.agent_state.name}] WARNING: Failed to parse topic extraction response: {parse_error}"
                        )
                        continue

        except Exception as e:
            printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Error in extracting the topic from the messages: {e}")

        return None

    async def _retrieve_memories_for_topics(self, topics: Optional[str]) -> dict:
        """
        Retrieve memories based on topics. This is extracted from build_system_prompt_with_memories
        to avoid code duplication.

        Args:
            topics (Optional[str]): Topics to use for memory retrieval

        Returns:
            dict: Retrieved memories dictionary
        """
        # Use the existing memory retrieval logic from build_system_prompt_with_memories
        # but without the raw_system combination
        _, retrieved_memories = await self.build_system_prompt_with_memories(
            raw_system="",  # Empty since we only want memories
            topics=topics,
        )
        return retrieved_memories

    async def construct_system_message(self, message: str) -> str:
        """
        Construct a complete system message by extracting topics from the message and
        combining with the raw system prompt and memories.

        Args:
            message (str): The message to extract topics from

        Returns:
            str: The complete system prompt including raw system and memories
        """
        topics = await self._extract_topics_from_message(message)

        # Use system prompt directly from agent state (no longer stored as a DB message)
        raw_system = self.agent_state.system or ""

        # Build the complete system prompt with memories
        complete_system_prompt, _ = await self.build_system_prompt_with_memories(raw_system=raw_system, topics=topics)

        return complete_system_prompt

    async def summarize_and_replace_retained_messages(
        self,
        retained_messages: List[Message],
        existing_file_uris: Optional[List[str]] = None,
    ) -> Message:
        """Summarize retained input-set messages and replace them in the DB.

        Calls the LLM to produce a summary of the retained messages, persists
        the summary as a single ``message_type='summary'`` row, then hard-deletes
        the original retained rows.

        Returns the new summary ``Message`` for use in the in-memory accumulator.
        """
        printv(
            f"[Mirix.Agent.{self.agent_state.name}] INFO: "
            f"Summarizing {len(retained_messages)} retained messages to recover from context overflow"
        )

        summary_text = await summarize_messages(
            agent_state=self.agent_state,
            message_sequence_to_summarize=retained_messages,
            existing_file_uris=existing_file_uris,
        )

        retention_agent_id = self.agent_state.parent_id or self.agent_state.id
        summary_msg = Message(
            agent_id=retention_agent_id,
            role=MessageRole.user,
            content=[TextContent(text=summary_text)],
            user_id=self.user_id,
            message_type="summary",
        )

        await self.message_manager.create_message(
            summary_msg,
            actor=self.actor,
            client_id=self.client_id,
            user_id=self.user_id,
        )

        for msg in retained_messages:
            await self.message_manager.delete_message_by_id(
                message_id=msg.id,
                actor=self.actor,
            )

        printv(
            f"[Mirix.Agent.{self.agent_state.name}] INFO: "
            f"Replaced {len(retained_messages)} retained messages with summary (id={summary_msg.id})"
        )

        return summary_msg

    async def inner_step(
        self,
        messages: Union[Message, List[Message]],
        accumulated: Optional[List[Message]] = None,
        stream: bool = False,  # TODO move to config?
        step_count: Optional[int] = None,
        force_response: bool = False,
        topics: Optional[str] = None,
        retrieved_memories: Optional[dict] = None,
        display_intermediate_message: any = None,
        request_user_confirmation: Optional[Callable] = None,
        existing_file_uris: Optional[List[str]] = None,
        return_memory_types_without_update: bool = False,
        message_queue: Optional[any] = None,
        chaining: bool = True,
        llm_client: Optional[LLMClient] = None,
        retained_count: int = 0,
        _summarization_attempted: bool = False,
        **kwargs,
    ) -> AgentStepResponse:
        """Runs a single step in the agent loop (generates at most one LLM call)"""

        if accumulated is None:
            accumulated = []

        try:
            # Log the start of each reasoning step
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: Starting agent step - step_count: {step_count}, chaining: {chaining}"
            )
            if topics:
                printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Step topics: {topics}")

            # Step 0: build the system message on-the-fly from agent_state.system + memories
            raw_system = self.agent_state.system or ""

            # Build the complete system prompt with memories
            complete_system_prompt, retrieved_memories = await self.build_system_prompt_with_memories(
                raw_system=raw_system,
                topics=topics,
                retrieved_memories=retrieved_memories,
            )

            system_msg = Message.dict_to_message(
                agent_id=self.agent_state.id,
                model=self.model,
                openai_message_dict={"role": "system", "content": complete_system_prompt},
            )

            # Step 1: add user message
            if isinstance(messages, Message):
                messages = [messages]

            if not all(isinstance(m, Message) for m in messages):
                message_types = [type(m).__name__ for m in messages]
                raise ValueError(
                    "messages should be a Message or a list of Message, "
                    f"got container={type(messages)}, elements={message_types}"
                )

            # Build sequence: [system] + accumulated (prior chaining steps) + current messages
            input_message_sequence = [system_msg] + accumulated + messages

            if len(input_message_sequence) > 1 and input_message_sequence[-1].role != "user":
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] WARNING: {CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue"
                )

            # Step 2: send the conversation and available functions to the LLM
            response = await self._get_ai_reply(
                message_sequence=input_message_sequence,
                stream=stream,
                step_count=step_count,
                existing_file_uris=existing_file_uris,
                llm_client=llm_client,
            )

            # Log the raw AI response for debugging and analysis
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: AI response received - choices: {len(response.choices)}"
            )
            for i, choice in enumerate(response.choices):
                if choice.message.content:
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] INFO: Choice {i} reasoning content: {choice.message.content}"
                    )
                if choice.message.tool_calls:
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] INFO: Choice {i} has {len(choice.message.tool_calls)} tool calls"
                    )
                    for j, tool_call in enumerate(choice.message.tool_calls):
                        printv(
                            f"[Mirix.Agent.{self.agent_state.name}] INFO: Tool call {j}: {tool_call.function.name} with args: {tool_call.function.arguments}"
                        )

            # Step 3: check if LLM wanted to call a function
            # (if yes) Step 4: call the function
            # (if yes) Step 5: send the info on the function call and function response to LLM
            all_response_messages = []
            for response_choice in response.choices:
                response_message = response_choice.message
                tmp_response_messages, continue_chaining, function_failed = await self._handle_ai_response(
                    messages[0],  # Input messages are always packed into a single MessageCreate object
                    response_message,
                    existing_file_uris=existing_file_uris,
                    # TODO this is kind of hacky, find a better way to handle this
                    # the only time we set up message creation ahead of time is when streaming is on
                    response_message_id=response.id if stream else None,
                    force_response=force_response,
                    retrieved_memories=retrieved_memories,
                    display_intermediate_message=display_intermediate_message,
                    request_user_confirmation=request_user_confirmation,
                    return_memory_types_without_update=return_memory_types_without_update,
                    message_queue=message_queue,
                    chaining=chaining,
                )
                all_response_messages.extend(tmp_response_messages)

            if function_failed:
                # Find the actual failed message(s) to log
                failed_messages = []
                for msg in all_response_messages:
                    if msg.role == "tool" and msg.content:
                        try:
                            content = msg.content[0].text if isinstance(msg.content, list) else msg.content
                            response_data = json.loads(content)
                            if response_data.get("status") == "Failed":
                                failed_messages.append(f"{msg.name}: {content}")
                        except (json.JSONDecodeError, AttributeError, KeyError):
                            pass

                if failed_messages:
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] ERROR: One or more functions failed:\n"
                        + "\n".join(failed_messages)
                    )
                else:
                    # Fallback if we can't parse the messages
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] ERROR: Function execution encountered errors (see logs above for details)"
                    )

            # Step 6: extend the message history
            if len(messages) > 0:
                all_new_messages = messages + all_response_messages
            else:
                all_new_messages = all_response_messages

            # Log step
            step = await self.step_manager.log_step(
                actor=self.actor,
                provider_name=self.agent_state.llm_config.model_endpoint_type,
                model=self.agent_state.llm_config.model,
                context_window_limit=self.agent_state.llm_config.context_window,
                usage=response.usage,
            )
            for message in all_new_messages:
                message.step_id = step.id

            # Log step completion and results
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: Agent step completed - continue_chaining: {continue_chaining}, function_failed: {function_failed}, messages_generated: {len(all_new_messages)}"
            )

            return AgentStepResponse(
                messages=all_new_messages,
                continue_chaining=continue_chaining,
                function_failed=function_failed,
                usage=response.usage,
            )

        except Exception as e:
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] ERROR: inner_step() failed\nmessages = {messages}\nerror = {e}"
            )
            if is_context_overflow_error(e):
                num_accumulated = len(accumulated) + len(messages)

                # Attempt summarization recovery: summarize retained DB messages
                # and retry once with a smaller context.
                retained = accumulated[:retained_count] if retained_count > 0 else []
                if retained and not _summarization_attempted:
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] INFO: "
                        f"Context overflow with {num_accumulated} messages — "
                        f"attempting summarization of {len(retained)} retained messages"
                    )
                    try:
                        summary_msg = await self.summarize_and_replace_retained_messages(retained, existing_file_uris)
                    except Exception as summarize_err:
                        printv(
                            f"[Mirix.Agent.{self.agent_state.name}] ERROR: " f"Summarization failed: {summarize_err}"
                        )
                        raise ContextWindowExceededError(
                            f"Context window exceeded for agent id={self.agent_state.id} "
                            f"and summarization recovery failed: {summarize_err}",
                            details={"num_in_context_messages": num_accumulated},
                        ) from e

                    chaining_outputs = accumulated[retained_count:]
                    new_accumulated = [summary_msg] + chaining_outputs

                    return await self.inner_step(
                        messages=messages,
                        accumulated=new_accumulated,
                        stream=stream,
                        step_count=step_count,
                        force_response=force_response,
                        topics=topics,
                        retrieved_memories=retrieved_memories,
                        display_intermediate_message=display_intermediate_message,
                        request_user_confirmation=request_user_confirmation,
                        existing_file_uris=existing_file_uris,
                        return_memory_types_without_update=return_memory_types_without_update,
                        message_queue=message_queue,
                        chaining=chaining,
                        llm_client=llm_client,
                        retained_count=1,
                        _summarization_attempted=True,
                        **kwargs,
                    )

                err_msg = (
                    f"Context window exceeded for agent id={self.agent_state.id} "
                    f"with {num_accumulated} in-context messages."
                )
                printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: {err_msg}")
                raise ContextWindowExceededError(
                    err_msg,
                    details={"num_in_context_messages": num_accumulated},
                )
            else:
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] ERROR: inner_step() failed with an unrecognized exception: '{str(e)}'"
                )
                raise e

    async def step_user_message(self, user_message_str: str, **kwargs) -> AgentStepResponse:
        """Takes a basic user message string, turns it into a stringified JSON with extra metadata, then sends it to the agent

        Example:
        -> user_message_str = 'hi'
        -> {'message': 'hi', 'type': 'user_message', ...}
        -> json.dumps(...)
        -> agent.step(messages=[Message(role='user', text=...)])
        """
        # Wrap with metadata, dumps to JSON
        assert user_message_str and isinstance(
            user_message_str, str
        ), f"user_message_str should be a non-empty string, got {type(user_message_str)}"
        user_message_json_str = package_user_message(user_message_str)

        # Validate JSON via save/load
        user_message = validate_json(user_message_json_str)
        cleaned_user_message_text, name = strip_name_field_from_user_message(user_message)

        # Turn into a dict
        openai_message_dict = {
            "role": "user",
            "content": cleaned_user_message_text,
            "name": name,
        }

        # Create the associated Message object (in the database)
        assert self.agent_state.created_by_id is not None, "User ID is not set"
        user_message = Message.dict_to_message(
            agent_id=self.agent_state.id,
            model=self.model,
            openai_message_dict=openai_message_dict,
            # created_at=timestamp,
        )

        return await self.inner_step(messages=[user_message], **kwargs)

    def add_function(self, function_name: str) -> str:
        # TODO: refactor
        raise NotImplementedError

    def remove_function(self, function_name: str) -> str:
        # TODO: refactor
        raise NotImplementedError

    def migrate_embedding(self, embedding_config: EmbeddingConfig):
        """Migrate the agent to a new embedding"""
        # TODO: archival memory

        # TODO: recall memory
        raise NotImplementedError()


def strip_name_field_from_user_message(
    user_message_text: str,
) -> Tuple[str, Optional[str]]:
    """If 'name' exists in the JSON string, remove it and return the cleaned text + name value"""
    try:
        user_message_json = dict(json_loads(user_message_text))
        # Special handling for AutoGen messages with 'name' field
        # Treat 'name' as a special field
        # If it exists in the input message, elevate it to the 'message' level
        name = user_message_json.pop("name", None)
        clean_message = json_dumps(user_message_json)
        return clean_message, name

    except Exception as e:
        # Note: This is a static function, so we'll use a module-level logger
        logger = logging.getLogger("Mirix.Agent.Utils")
        logger.error("Handling of 'name' field failed with: %s", e)
        raise e


def validate_json(user_message_text: str) -> str:
    """Make sure that the user input message is valid JSON"""
    try:
        user_message_json = dict(json_loads(user_message_text))
        user_message_json_val = json_dumps(user_message_json)
        return user_message_json_val
    except Exception as e:
        logger.debug("%scouldn't parse user input message as JSON: %s", CLI_WARNING_PREFIX, e)
        raise e


def convert_message_to_input_message(message: Message) -> Union[str, List[dict]]:
    """
    Convert a Message object back to the input format expected by client.send_message().

    Args:
        message (Message): The Message object to convert

    Returns:
        Union[str, List[dict]]: Either a string (for simple text messages) or a list of
                               dictionaries (for multi-modal messages)
    """
    if not message.content:
        return ""

    # TODO: this might cause duplicated files and images as these images will be recreated.
    # TODO: we need to set a tag or something to avoid duplicated files and images.

    # If it's a single text content, return as string
    if len(message.content) == 1 and isinstance(message.content[0], TextContent):
        return message.content[0].text

    # For multi-modal content, convert to list of dictionaries
    result = []

    for content_part in message.content:
        if isinstance(content_part, TextContent):
            result.append({"type": "text", "text": content_part.text})
        elif isinstance(content_part, ImageContent):
            result.append({"type": "database_image_id", "image_id": content_part.image_id})
        elif isinstance(content_part, FileContent):
            result.append(
                {
                    "type": "database_file_id",
                    "file_id": content_part.file_id,
                }
            )

        elif isinstance(content_part, CloudFileContent):
            result.append(
                {
                    "type": "database_google_cloud_file_uri",
                    "cloud_file_uri": content_part.cloud_file_uri,
                }
            )
        else:
            # For any other content types, skip them or handle as text
            # This includes tool calls, tool returns, reasoning content, etc.
            # These are internal message types that shouldn't be converted back
            continue

    return result
