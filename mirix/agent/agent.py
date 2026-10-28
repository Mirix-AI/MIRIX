import asyncio
import copy
import json
import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from fractions import Fraction
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pytz
import httpx
from opentelemetry import trace

from mirix.agent.tool_call_plan import (
    TERMINAL_ALONE_REMINDERS,
    TERMINAL_TOOLS,
    held_terminal_text,
    plan_tool_calls,
)
from mirix.agent.tool_failures import ToolFailureLog
from mirix.agent.tool_validators import validate_tool_args
from mirix.constants import (
    CHAINING_FOR_MEMORY_UPDATE,
    CLEAR_HISTORY_AFTER_MEMORY_UPDATE,
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
    UNFINISHED_TURN_HEARTBEAT_MESSAGE,
)
from mirix.embeddings import embedding_model
from mirix.errors import (
    AbortAgentStep,
    AgentContextLimitReached,
    ContextWindowExceededError,
    LLMError,
    LLMInvalidResponseError,
    LLMOutputTruncatedError,
    ToolArgumentError,
    ToolCallRefused,
)
from mirix.functions.functions import get_function_from_module
from mirix.helpers import ToolRulesSolver
from mirix.helpers.message_helpers import prepare_input_message_create
from mirix.interface import AgentInterface
from mirix.llm_api.helpers import calculate_summarizer_cutoff, get_token_counts_for_messages, is_context_overflow_error
from mirix.llm_api.llm_api_tools import create
from mirix.llm_api.llm_client import LLMClient
from mirix.log import get_logger
from mirix.memory import summarize_messages
from mirix.observability.context import get_trace_context, mark_observation_as_child
from mirix.observability.langfuse_client import get_langfuse_client
from mirix.schemas.agent import AgentState, AgentStepResponse, AgentType, UpdateAgent
from mirix.schemas.block import BlockUpdate
from mirix.schemas.client import Client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import MessageRole, ToolType
from mirix.schemas.memory import ContextWindowOverview, Memory
from mirix.schemas.message import Message, MessageCreate
from mirix.schemas.mirix_message_content import CloudFileContent, FileContent, ImageContent, TextContent
from mirix.schemas.openai.chat_completion_request import Tool as ChatCompletionRequestTool
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
from mirix.services.helpers.agent_manager_helper import check_supports_structured_output, compile_memory_metadata_block
from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
from mirix.services.message_manager import MessageManager
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.resource_memory_manager import ResourceMemoryManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.services.step_manager import StepManager
from mirix.services.user_manager import UserManager
from mirix.services.tool_execution_sandbox import ToolExecutionSandbox
from mirix.settings import settings, summarizer_settings
from mirix.system import (
    get_contine_chaining,
    get_token_limit_warning,
    package_function_response,
    package_summarize_message,
    package_user_message,
)
from mirix.tracing import trace_method
from mirix.utils import (
    convert_timezone_to_utc,
    count_tokens,
    get_friendly_error_msg,
    get_tool_call_id,
    get_utc_time,
    json_dumps,
    json_loads,
    log_telemetry,
    num_tokens_from_functions,
    num_tokens_from_messages,
    parse_json,
    printv,
    validate_function_response,
)

# Initialize module-level logger
logger = get_logger(__name__)
otel_tracer = trace.get_tracer(__name__)


def _bind_tool_arguments(
    function_name: str, func: Callable, function_args: dict
) -> None:
    """Raise :class:`ToolArgumentError` when ``function_args`` do not fit
    ``func``'s signature (a required argument the model left out): the call
    would raise a ``TypeError`` indistinguishable from one of the tool's own
    code, which is MIRIX's failure, not the model's."""
    import inspect

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # not introspectable: call it as it is
        return
    try:
        signature.bind(**function_args)
    except TypeError as exc:
        raise ToolArgumentError(f"{function_name}(): {exc}") from exc


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

        # The tools a step of this instance may offer and run (None: every tool
        # of the agent). Kept on the instance, never persisted: the agent state
        # is replaced by the stored one after every LLM call (inner_step), so a
        # per-run restriction on a copy of the state would not last.
        self.tool_allowlist: Optional[frozenset] = None

        # Logger that the Agent specifically can use, will also report the agent_state ID with the logs
        # Note: Logger is already initialized earlier in constructor

    def _visible_tools(self) -> List[Tool]:
        """The agent's tools this step may offer and run: those named in
        ``tool_allowlist``, or every tool when it is None."""
        tools = list(self.agent_state.tools or [])
        allowlist = getattr(self, "tool_allowlist", None)
        if allowlist is None:
            return tools
        return [t for t in tools if t.name in allowlist]

    async def load_last_function_response(self):
        """Load the last function response from message history."""
        # Skip if actor not set yet (during __init__)
        if self.actor is None:
            return None

        in_context_messages = await self.agent_manager.get_in_context_messages(
            agent_state=self.agent_state, actor=self.actor, user=self.user
        )
        for i in range(len(in_context_messages) - 1, -1, -1):
            msg = in_context_messages[i]
            if msg.role == MessageRole.tool and msg.content[0].text:
                try:
                    response_json = json.loads(msg.content[0].text)
                    if response_json.get("message"):
                        return response_json["message"]
                except (json.JSONDecodeError, KeyError):
                    raise ValueError(f"Invalid JSON format in message: {msg.content[0].text}")
        return None

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
                blocks=[
                    await self.block_manager.get_block_by_id(block.id, user=self.user)
                    for block in blocks_result
                ]
            )

            # NOTE: don't do this since re-buildin the memory is handled at the start of the step
            # rebuild memory - this records the last edited timestamp of the memory
            # TODO: pass in update timestamp from block edit time
            # self.agent_state = self.agent_manager.rebuild_system_prompt(agent_id=self.agent_state.id, actor=self.user)
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
            self.last_tool_exception = None

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
                    _bind_tool_arguments(function_name, callable_func, function_args)
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

                    _bind_tool_arguments(function_name, callable_func, function_args)
                    function_response = await callable_func(**function_args)
                    if function_name in ["core_memory_append", "core_memory_rewrite"]:
                        await self.update_memory_if_changed(memory_copy)

                elif target_mirix_tool.tool_type == ToolType.MIRIX_EXTRA:
                    callable_func = get_function_from_module(MIRIX_EXTRA_TOOL_MODULE_NAME, function_name)
                    function_args["self"] = self  # need to attach self to arg since it's dynamically linked
                    _bind_tool_arguments(function_name, callable_func, function_args)
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

            except (AbortAgentStep, ToolCallRefused):
                # The tool ends the whole step (e.g. its learning run is gone):
                # never turn it into a tool result the model could react to.
                # A refused call is answered by the step loop (not an error).
                raise
            except Exception as e:
                # Need to catch error here, or else truncation wont happen
                is_error = True
                # The step loop records which exception the tool raised (the
                # error text alone would not say, tool_failures).
                self.last_tool_exception = e
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

            executed = False
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
                    executed = True

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
            except (AbortAgentStep, ToolCallRefused):
                raise
            except Exception as e:
                self.logger.debug(f"Langfuse tool execution trace failed: {e}")
                # Only a tracing failure BEFORE the tool ran may run it here: a
                # tool that already ran (a skill write, say) must never run twice.
                if not executed:
                    function_response, _ = await _execute_tool_inner()
        else:
            function_response, _ = await _execute_tool_inner()

        return function_response

    async def _wait_before_llm_retry(
        self, delay: float, attempt: int, error: BaseException
    ) -> None:
        """Record an agent-level retry of an LLM call (every retry
        lands in the learning run's ``retry_events`` when the step runs inside a
        ``recording_retries`` block), then wait ``delay`` seconds."""
        from mirix.services.retry_policy import ErrorKind, classify_error, record_retry

        try:
            kind = classify_error(error)
        except TypeError:  # not an Exception (never expected here)
            kind = ErrorKind.OTHER
        await record_retry(kind, attempt, delay, stage="agent", error=error)
        await asyncio.sleep(delay)

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
        # Only the tools this step may use (``tool_allowlist``) are offered.
        agent_state_tool_jsons = [t.json_schema for t in self._visible_tools()]

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
        configured_model = self.agent_state.llm_config.model
        if active_llm_client and active_llm_client.llm_config:
            configured_model = active_llm_client.llm_config.model

        def _serialize_for_span(value: object, max_len: int = 8000) -> str:
            """Serialize complex values safely for OpenTelemetry span attributes."""
            try:
                if hasattr(value, "model_dump"):
                    text = json_dumps(value.model_dump())
                else:
                    text = json_dumps(value)
            except Exception:
                text = str(value)
            if len(text) > max_len:
                return f"{text[:max_len]}...[truncated]"
            return text

        # A strict step (the skill curator) takes a provider error
        # as it comes: no agent-level retry on top of the SDK's transport
        # retries, and never the last-message-only fallback, which would send
        # the model a heartbeat without the system prompt or its input. Only
        # an ANSWER that stays unusable (no choices, an empty message, a bad
        # finish reason, one the client cannot convert) becomes
        # LLMInvalidResponseError below; a failure before anything was sent —
        # building the request, calling the SDK wrongly — reaches this loop as
        # LLMRequestBuildError (LLMClientBase.send_llm_request), which no
        # handler here catches: it propagates as MIRIX's own error, never as
        # the model's.
        strict = bool(getattr(self, "_strict_llm_errors", False))
        for attempt in range(1, empty_response_retry_limit + 1):
            with otel_tracer.start_as_current_span("Agent._get_ai_reply.llm_attempt") as llm_span:
                llm_span.set_attribute("llm.attempt", attempt)
                llm_span.set_attribute("llm.stream", stream)
                llm_span.set_attribute("llm.agent_name", self.agent_state.name)
                llm_span.set_attribute("llm.model.configured", configured_model or "unknown")
                llm_span.set_attribute("llm.input.messages", _serialize_for_span(message_sequence))
                llm_span.set_attribute("llm.input.allowed_functions", _serialize_for_span(allowed_functions))
                if force_tool_call:
                    llm_span.set_attribute("llm.input.force_tool_call", force_tool_call)

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
                    llm_span.set_attribute("llm.model.response", response.model or configured_model or "unknown")
                    llm_span.set_attribute("llm.output.response", _serialize_for_span(response))
                    if response.usage:
                        llm_span.set_attribute("llm.usage.prompt_tokens", response.usage.prompt_tokens)
                        llm_span.set_attribute("llm.usage.completion_tokens", response.usage.completion_tokens)
                        llm_span.set_attribute("llm.usage.total_tokens", response.usage.total_tokens)

                    # These bottom two are retryable
                    if len(response.choices) == 0 or response.choices[0] is None:
                        raise ValueError(f"API call returned an empty message: {response}")

                    for choice in response.choices:
                        # ``tool_calls`` is None when the answer has none: an
                        # empty answer is the model's (retried, then PARSE), not
                        # a TypeError of ours.
                        if choice.message.content == "" and not choice.message.tool_calls:
                            raise ValueError(f"API call returned an empty message: {response}")

                    if response.choices[0].finish_reason not in [
                        "stop",
                        "function_call",
                        "tool_calls",
                    ]:
                        if response.choices[0].finish_reason == "length":
                            if strict:
                                # A truncated answer is never used and the
                                # same input would be cut again: the caller
                                # decides (the curator counts an attempt).
                                raise LLMOutputTruncatedError()
                            if attempt >= empty_response_retry_limit:
                                raise RuntimeError(
                                    "Retries exhausted and no valid response received. Final error: maximum context length exceeded or generated content is too long"
                                )
                            else:
                                delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                                printv(
                                    f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {response.choices[0].finish_reason}. Retrying in {delay} seconds..."
                                )
                                await self._wait_before_llm_retry(
                                    delay, attempt, LLMOutputTruncatedError()
                                )
                                continue
                        else:
                            raise ValueError(f"Bad finish reason from API: {response.choices[0].finish_reason}")
                    log_telemetry(self.logger, "_handle_ai_response finish")

                except ValueError as ve:
                    llm_span.record_exception(ve)
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
                        if strict:
                            raise LLMInvalidResponseError(
                                f"Retries exhausted and no valid response received. Final error: {ve_desc}"
                            ) from ve
                        raise Exception(f"Retries exhausted and no valid response received. Final error: {ve_desc}")
                    else:
                        delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                        printv(
                            f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {ve_desc}. Retrying in {delay} seconds..."
                        )
                        await self._wait_before_llm_retry(delay, attempt, ve)
                        continue

                except KeyError as ke:
                    llm_span.record_exception(ke)
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
                        if strict:
                            raise LLMInvalidResponseError(
                                f"Retries exhausted and no valid response received. Final error: {ke_desc}"
                            ) from ke
                        raise Exception(f"Retries exhausted and no valid response received. Final error: {ke_desc}")
                    else:
                        delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                        printv(
                            f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {ke_desc}. Retrying in {delay} seconds..."
                        )
                        await self._wait_before_llm_retry(delay, attempt, ke)
                        continue

                except LLMError as llm_error:
                    llm_span.record_exception(llm_error)
                    if strict:
                        # The SDK already retried this call at the transport
                        # layer (MIRIX_LLM_MAX_RETRIES): no second layer of
                        # retries, and never the context-less fallback below.
                        raise
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
                        await self._wait_before_llm_retry(delay, attempt, llm_error)
                        continue

                except AssertionError as ae:
                    llm_span.record_exception(ae)
                    tb_str = traceback.format_exc()
                    ae_desc = f"{type(ae).__name__}: {ae!r}\nTraceback:\n{tb_str}"
                    if attempt >= empty_response_retry_limit:
                        printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: Retry limit reached. Final error: {ae_desc}")
                        self.logger.exception(
                            "[Mirix.Agent.%s] Retry limit reached (AssertionError).",
                            self.agent_state.name,
                        )
                        if strict:
                            raise LLMInvalidResponseError(
                                f"Retries exhausted and no valid response received. Final error: {ae_desc}"
                            ) from ae
                        raise Exception(f"Retries exhausted and no valid response received. Final error: {ae_desc}")
                    else:
                        delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                        printv(
                            f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt} failed: {ae_desc}. Retrying in {delay} seconds..."
                        )
                        await self._wait_before_llm_retry(delay, attempt, ae)
                        continue

                except httpx.HTTPStatusError as he:
                    llm_span.record_exception(he)
                    if strict:
                        raise  # transport-level already retried (see LLMError)
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
                        await self._wait_before_llm_retry(delay, attempt, he)
                        continue

                except Exception as e:
                    llm_span.record_exception(e)
                    log_telemetry(self.logger, "_handle_ai_response finish generic Exception")
                    # For non-retryable errors, exit immediately
                    log_telemetry(self.logger, "_handle_ai_response finish generic Exception")
                    raise e

            # check if we are going over the context window: this allows for articifial constraints
            if (
                getattr(self, "_auto_summarize", True)
                and response.usage.total_tokens > self.agent_state.llm_config.context_window
            ):
                # trigger summarization
                log_telemetry(self.logger, "_get_ai_reply summarize_messages_inplace")
                await self.summarize_messages_inplace(existing_file_uris=existing_file_uris)

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

        # Only the chat_agent inherits the triggering input's session_id so its
        # synthesized assistant/tool messages stay in the same session. For all
        # non-chat agents (meta_memory_agent + the memory sub-agents) the
        # synthesized messages carry NO session_id: session_id identifies an
        # external conversation, never the memory-production machinery. Context
        # reconstruction uses message_ids (not session_id), so nulling this is safe.
        input_session_id = (
            getattr(input_message, "session_id", None)
            if self.agent_state.is_type(AgentType.chat_agent)
            else None
        )

        messages = []  # append these to the history when done
        function_name = None
        must_continue = False  # a call of this turn was left out or refused
        # The tools this reply ran successfully (step() reads it to tell a
        # finish_memory_update apart from any other end of the chain).
        self.last_executed_functions = []

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

            tool_calls = response_message.tool_calls
            # Which calls run, in what order, before anything runs
            # (tool_call_plan): a finish (finish_memory_update) runs last and
            # once; each distinct reply (send_message) runs at the end of the
            # turn, or waits for a later turn when another call follows it (and
            # then every reply of the turn waits); a step with a per-turn cap
            # runs only its first calls, and a terminal call only alone. Every
            # call left out is answered "Not run" below. A model told
            # TERMINAL_ALONE_REMINDERS turn(s) in a row to make its terminal
            # call alone that pairs it again gets it run after its other calls
            # (plan.guarded: only when they went through): it is never held
            # until the step cap. Only a tool this step can call (after
            # tool_allowlist) is terminal: a send_message the agent lacks is
            # answered "not available" below, like any unknown name.
            alone_turns = getattr(self, "_terminal_alone_turns", 0)
            plan = plan_tool_calls(
                [tool_call.function.name for tool_call in tool_calls],
                max_calls=getattr(self, "_max_tool_calls_per_turn", None),
                terminal_after_others=alone_turns >= TERMINAL_ALONE_REMINDERS,
                arguments=[tool_call.function.arguments for tool_call in tool_calls],
                available=[tool.name for tool in self._visible_tools()],
            )
            self._terminal_alone_turns = (
                alone_turns + 1 if plan.terminal_deferred else 0
            )
            if len(tool_calls) > 1:
                self.multi_call_turns = getattr(self, "multi_call_turns", 0) + 1
                self.logger.info(
                    "Agent %s returned %d tool call(s); running %d of them sequentially",
                    self.agent_state.agent_type,
                    len(tool_calls),
                    len(plan.order),
                )
            self.skipped_tool_calls = getattr(self, "skipped_tool_calls", 0) + len(
                plan.skipped
            )

            # role: assistant (requesting tool call, set tool call ID)
            messages.append(
                # NOTE: we're recreating the message here
                # TODO should probably just overwrite the fields?
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                    session_id=input_session_id,
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

            # Step 3: Process each tool call the plan runs, in its order
            continue_chaining = True
            overall_function_failed = False
            failed_tool_calls = 0  # tool calls answered with an error result
            tool_failures = ToolFailureLog()  # why they failed (same calls)
            executed_function_names = []  # Track which functions were executed
            refused_tool_calls = 0  # calls a tool declined (ToolCallRefused)
            # Calls a tool rejected with an answer (nothing done, no error):
            # tools count them on the agent (declined_tool_calls).
            declined_at_start = getattr(self, "declined_tool_calls", 0)
            held_terminal = False  # a guarded terminal call was held
            answered = []  # the call each tool message below answers, in run order

            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: Processing {len(tool_calls)} tool call(s)"
            )

            for tool_call_idx in plan.order:
                tool_call = tool_calls[tool_call_idx]
                answered.append(tool_call_idx)
                tool_call_id = tool_call.id
                function_call = tool_call.function

                # A terminal call run after its turn's other calls on a repeat
                # (plan.guarded) runs only when they all went through: a
                # rejected, refused or failed call must be read first. A held
                # call did not run (function_name keeps the last call that did).
                if tool_call_idx in plan.guarded and (
                    held_terminal
                    or failed_tool_calls
                    or refused_tool_calls
                    or getattr(self, "declined_tool_calls", 0) != declined_at_start
                ):
                    held_terminal = True
                    held_text = held_terminal_text(function_call.name)
                    messages.append(
                        Message.dict_to_message(
                            agent_id=self.agent_state.id,
                            model=self.model,
                            openai_message_dict={
                                "role": "tool",
                                "name": function_call.name,
                                "content": package_function_response(False, held_text),
                                "tool_call_id": tool_call_id,
                            },
                            session_id=input_session_id,
                        )
                    )
                    self.skipped_tool_calls = getattr(self, "skipped_tool_calls", 0) + 1
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] INFO: Tool call {tool_call_idx + 1}/{len(tool_calls)} "
                        f"({function_call.name}) not run: {held_text}"
                    )
                    continue
                function_name = function_call.name

                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Processing tool call {tool_call_idx + 1}/{len(tool_calls)}: {function_name} with tool_call_id: {tool_call_id}"
                )

                # Failure case 1: function name is wrong (not a tool this step
                # may use: not in agent_state.tools, or outside tool_allowlist)
                visible_tools = self._visible_tools()
                target_mirix_tool = None
                for t in visible_tools:
                    if t.name == function_name:
                        target_mirix_tool = t

                if not target_mirix_tool:
                    if getattr(self, "tool_allowlist", None) is not None:
                        error_msg = (
                            f"Tool {function_name} is not available in this step; "
                            f"available: {', '.join(t.name for t in visible_tools)}."
                        )
                    else:
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
                            session_id=input_session_id,
                        )
                    )  # extend conversation with function response
                    self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                    overall_function_failed = True
                    failed_tool_calls += 1
                    tool_failures.record("unknown_tool", function_name, error=error_msg)
                    continue  # Continue with next tool call

                # Failure case 2: function name is OK, but function args are bad JSON
                try:
                    raw_function_args = function_call.arguments
                    function_args = parse_json(raw_function_args)
                    if not isinstance(function_args, dict):
                        # A JSON repair of garbage can yield a list or a string:
                        # still invalid arguments, answered with an error result
                        # (never a crash of the whole step).
                        raise ValueError("the arguments are not a JSON object")
                except Exception as parse_error:
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
                            session_id=input_session_id,
                        )
                    )  # extend conversation with function response
                    self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                    overall_function_failed = True
                    failed_tool_calls += 1
                    tool_failures.record("arguments", function_name, exc=parse_error)
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
                            session_id=input_session_id,
                        )
                    )
                    # The validator's text starts "Validation error: ...": with the
                    # "Error: " prefix every interface renders it as an error.
                    self.interface.function_message(f"Error: {validation_error}", msg_obj=messages[-1])
                    overall_function_failed = True
                    failed_tool_calls += 1
                    tool_failures.record("validation", function_name, error=validation_error)
                    continue  # Skip execution, let LLM retry

                # Failure case 5: function failed during execution
                # NOTE: the msg_obj associated with the "Running " message is the prior assistant message, not the function/tool role message
                #       this is because the function/tool role message is only created once the function/tool has executed/returned
                self.interface.function_message(f"Running {function_name}()", msg_obj=messages[-1])

                try:
                    if display_intermediate_message:
                        # send intermediate message to the user
                        display_intermediate_message("internal_monologue", response_message.content)

                    # Set again by this call's tool run only: never an earlier call's.
                    self.last_tool_exception = None
                    function_response = await self.execute_tool_and_persist_state(
                        function_name,
                        function_args,
                        target_mirix_tool,
                        display_intermediate_message=display_intermediate_message,
                        request_user_confirmation=request_user_confirmation,
                    )

                    # The plan runs a terminal call last, so the chain ends
                    # (or goes on) as the last call that ran says.
                    continue_chaining = function_name not in TERMINAL_TOOLS

                    # handle trunction
                    if function_name in [
                        "conversation_search",
                        "conversation_search_date",
                        "archival_memory_search",
                        # skill_list pages and skill_read caps its output head +
                        # tail with a marker; a head-only cut here
                        # would silently drop the tail.
                        "skill_list",
                        "skill_read",
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

                except AbortAgentStep:
                    raise
                except ToolCallRefused as refused:
                    # The tool declined this call (it did not run): its message
                    # is the answer, the call is neither executed nor failed, and
                    # the chain goes on so the model can act on it.
                    function_args.pop("self", None)
                    function_response = package_function_response(
                        False, refused.message
                    )
                    self.last_function_response = function_response
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
                            session_id=input_session_id,
                        )
                    )
                    self.interface.function_message(
                        f"Ran {function_name}()", msg_obj=messages[-1]
                    )
                    self.interface.function_message(
                        f"Error: {refused.message}", msg_obj=messages[-1]
                    )
                    refused_tool_calls += 1
                    continue_chaining = True
                    continue
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
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] ERROR: {error_msg_user}"
                    )
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
                            session_id=input_session_id,
                        )
                    )  # extend conversation with function response
                    self.interface.function_message(
                        f"Ran {function_name}()", msg_obj=messages[-1]
                    )
                    self.interface.function_message(
                        f"Error: {error_msg}", msg_obj=messages[-1]
                    )
                    overall_function_failed = True
                    failed_tool_calls += 1
                    # Raised around the tool's run: preparing it, the ordering
                    # guard above, or checking its result.
                    tool_failures.record("execution", function_name, exc=e)
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
                            session_id=input_session_id,
                        )
                    )  # extend conversation with function response
                    self.interface.function_message(f"Ran {function_name}()", msg_obj=messages[-1])
                    self.interface.function_message(f"Error: {function_response_string}", msg_obj=messages[-1])
                    overall_function_failed = True
                    failed_tool_calls += 1
                    raised = getattr(self, "last_tool_exception", None)
                    if isinstance(raised, ToolArgumentError):
                        # Its arguments do not fit the tool's signature: found
                        # before it ran, the model's mistake.
                        tool_failures.record("validation", function_name, exc=raised)
                    elif raised is not None:
                        # The tool raised while it ran; execute_tool_and_persist_state
                        # answered with its text.
                        tool_failures.record("execution", function_name, exc=raised)
                    else:
                        tool_failures.record("error_result", function_name, error=function_response_string)
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
                        session_id=input_session_id,
                    )
                )  # extend conversation with function response
                self.interface.function_message(f"Ran {function_name}()", msg_obj=messages[-1])
                self.interface.function_message(f"Success: {function_response_string}", msg_obj=messages[-1])
                self.last_function_response = function_response

                # Track successfully executed function names
                executed_function_names.append(function_name)

            # Every call the plan left out is answered too, and the tool
            # messages go out in the answer's order: each tool_call_id gets its
            # result where the provider expects it.
            not_run = {}
            for index, text in plan.skipped.items():
                skipped_call = tool_calls[index]
                not_run[index] = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict={
                        "role": "tool",
                        "name": skipped_call.function.name,
                        "content": package_function_response(False, text),
                        "tool_call_id": skipped_call.id,
                    },
                    session_id=input_session_id,
                )
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Tool call {index + 1}/{len(tool_calls)} "
                    f"({skipped_call.function.name}) not run: {text}"
                )
            if len(messages) - 1 == len(answered):
                by_index = dict(zip(answered, messages[1:]))
                by_index.update(not_run)
                messages = messages[:1] + [by_index[i] for i in sorted(by_index)]
            else:  # never expected: every call that ran added one tool message
                self.logger.warning(
                    "Agent %s: %d tool message(s) for %d call(s) run; kept in run order",
                    self.agent_state.agent_type,
                    len(messages) - 1,
                    len(answered),
                )
                messages.extend(not_run[i] for i in sorted(not_run))

            function_failed = overall_function_failed
            must_continue = plan.deferred or refused_tool_calls > 0 or held_terminal
            if held_terminal:
                # The model still pairs its terminal call: the next pairing is
                # a repeat again (run after the others when they go through).
                self._terminal_alone_turns = TERMINAL_ALONE_REMINDERS
            self.failed_tool_calls = getattr(self, "failed_tool_calls", 0) + failed_tool_calls
            if getattr(self, "tool_failures", None) is None:
                self.tool_failures = ToolFailureLog()
            self.tool_failures.merge(tool_failures)
            self.last_executed_functions = list(executed_function_names)

            # Handle context message clearing only if ALL functions succeeded
            if not overall_function_failed:
                should_clear_history = False

                # Clear history for all non-chat agents when:
                # 1. chaining=False (clear regardless of function calls), OR
                # 2. finish_memory_update was called (clear when chaining completes)
                if CLEAR_HISTORY_AFTER_MEMORY_UPDATE and not self.agent_state.is_type(AgentType.chat_agent):
                    if not chaining:
                        should_clear_history = True
                        self.logger.info(f"should_clear_history=True (chaining=False)")
                    else:
                        for func_name in executed_function_names:
                            if func_name == "finish_memory_update":
                                should_clear_history = True
                                self.logger.info(f"should_clear_history=True (finish_memory_update called)")
                                break

                else:
                    self.logger.debug(
                        f"Clearing skipped - CLEAR_HISTORY_AFTER_MEMORY_UPDATE={CLEAR_HISTORY_AFTER_MEMORY_UPDATE}, is_chat_agent={self.agent_state.is_type(AgentType.chat_agent)}"
                    )

                if should_clear_history:
                    continue_chaining = False

                    in_context_messages = await self.agent_manager.get_in_context_messages(
                        agent_state=self.agent_state, actor=self.actor, user=self.user
                    )
                    self.logger.info(
                        f"Clearing history - {len(in_context_messages)} messages -> keeping only system message"
                    )
                    message_ids = [message.id for message in in_context_messages]
                    message_ids = [message_ids[0]]

                    # show the last edited memory item
                    memory_item = None
                    memory_item_str = None

                    if self.user is None:
                        raise ValueError("User is required to clear history")

                    if self.agent_state.name.endswith("episodic_memory_agent"):
                        memory_item = await self.episodic_memory_manager.get_most_recently_updated_event(
                            user=self.user,
                            timezone_str=self.user.timezone,
                        )
                        if memory_item:
                            memory_item = memory_item[0]
                            memory_item_str = ""
                            memory_item_str += "[Episodic Event ID]: " + memory_item.id + "\n"
                            memory_item_str += (
                                "[Event Occurred At]: " + memory_item.occurred_at.strftime("%Y-%m-%d %H:%M:%S") + "\n"
                            )
                            memory_item_str += "[Summary]: " + memory_item.summary + "\n"
                            memory_item_str += "[Details]: " + memory_item.details + "\n"
                            memory_item_str += (
                                "[Last Modified]: "
                                + memory_item.last_modify["operation"]
                                + " at "
                                + memory_item.last_modify["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                                + "\n"
                            )
                            memory_item_str = memory_item_str.strip()

                    elif self.agent_state.name.endswith("procedural_memory_agent"):
                        memory_item = await self.procedural_memory_manager.get_most_recently_updated_item(
                            user=self.user,
                            timezone_str=self.user.timezone,
                        )
                        if memory_item:
                            memory_item = memory_item[0]
                            # Phase 1 of skill-evolve replaced the legacy
                            # summary+steps shape with name/description/
                            # instructions/version. Surface the new fields.
                            memory_item_str = ""
                            memory_item_str += "[Skill ID]: " + memory_item.id + "\n"
                            memory_item_str += "[Name]: " + memory_item.name + "\n"
                            memory_item_str += "[Entry Type]: " + memory_item.entry_type + "\n"
                            memory_item_str += "[Description]: " + (memory_item.description or "N/A") + "\n"
                            memory_item_str += "[Instructions]: " + (memory_item.instructions or "N/A") + "\n"
                            memory_item_str += "[Version]: " + str(getattr(memory_item, "version", "0.1.0")) + "\n"
                            ts = memory_item.last_modify.get("timestamp") if isinstance(memory_item.last_modify, dict) else None
                            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
                            op = memory_item.last_modify.get("operation") if isinstance(memory_item.last_modify, dict) else "?"
                            memory_item_str += (
                                "[Last Modified]: " + str(op) + " at " + ts_str + "\n"
                            )
                            memory_item_str = memory_item_str.strip()

                    elif self.agent_state.name.endswith("resource_memory_agent"):
                        memory_item = await self.resource_memory_manager.get_most_recently_updated_item(
                            user=self.user,
                            timezone_str=self.user.timezone,
                        )
                        if memory_item:
                            memory_item = memory_item[0]
                            memory_item_str = ""
                            memory_item_str += "[Resource Memory ID]: " + memory_item.id + "\n"
                            memory_item_str += "[Title]: " + memory_item.title + "\n"
                            memory_item_str += "[Summary]: " + (memory_item.summary or "N/A") + "\n"
                            memory_item_str += "[Resource Type]: " + memory_item.resource_type + "\n"
                            memory_item_str += "[Content]: " + memory_item.content + "\n"
                            memory_item_str += (
                                "[Last Modified]: "
                                + memory_item.last_modify["operation"]
                                + " at "
                                + memory_item.last_modify["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                                + "\n"
                            )
                            memory_item_str = memory_item_str.strip()

                    elif self.agent_state.name.endswith("knowledge_vault_memory_agent"):
                        memory_item = await self.knowledge_vault_manager.get_most_recently_updated_item(
                            user=self.user,
                            timezone_str=self.user.timezone,
                        )

                        # Check if finish_memory_update was one of the executed functions
                        if "finish_memory_update" in executed_function_names and memory_item is None:
                            memory_item_str = "No new knowledge vault items were added."

                        if memory_item:
                            memory_item = memory_item[0]
                            memory_item_str = ""
                            memory_item_str += "[Knowledge Vault ID]: " + memory_item.id + "\n"
                            memory_item_str += "[Entry Type]: " + memory_item.entry_type + "\n"
                            memory_item_str += "[Caption]: " + memory_item.caption + "\n"
                            memory_item_str += "[Source]: " + memory_item.source + "\n"
                            memory_item_str += "[Sensitivity]: " + memory_item.sensitivity + "\n"
                            memory_item_str += "[Secret Value]: " + memory_item.secret_value + "\n"
                            memory_item_str += (
                                "[Last Modified]: "
                                + memory_item.last_modify["operation"]
                                + " at "
                                + memory_item.last_modify["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                                + "\n"
                            )
                            memory_item_str = memory_item_str.strip()

                    elif self.agent_state.name.endswith("semantic_memory_agent"):
                        memory_item = await self.semantic_memory_manager.get_most_recently_updated_item(
                            user=self.user,
                            timezone_str=self.user.timezone,
                        )
                        if memory_item:
                            memory_item = memory_item[0]
                            memory_item_str = ""
                            memory_item_str += "[Semantic Memory ID]: " + memory_item.id + "\n"
                            memory_item_str += "[Name]: " + memory_item.name + "\n"
                            memory_item_str += "[Summary]: " + memory_item.summary + "\n"
                            memory_item_str += "[Details]: " + (memory_item.details or "N/A") + "\n"
                            memory_item_str += "[Source]: " + (memory_item.source or "N/A") + "\n"
                            memory_item_str += (
                                "[Last Modified]: "
                                + memory_item.last_modify["operation"]
                                + " at "
                                + memory_item.last_modify["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                                + "\n"
                            )
                            memory_item_str = memory_item_str.strip()

                    elif self.agent_state.name.endswith("core_memory_agent"):
                        memory_item_str = self.blocks_in_memory.compile() if self.blocks_in_memory else ""

                        # Optionally create a summary message showing last edited memory item
                        if memory_item_str:
                            if self.agent_state.name.endswith("core_memory_agent"):
                                message_content = "Current Full Core Memory:\n\n" + memory_item_str
                            else:
                                message_content = "Last edited memory item:\n\n" + memory_item_str

                            # create a new message
                            new_message = Message.dict_to_message(
                                agent_id=self.agent_state.id,
                                model=self.model,
                                openai_message_dict={
                                    "role": "user",
                                    "content": message_content,
                                },
                                session_id=input_session_id,
                            )

                            # persist the message to the database
                            persisted_message = await self.message_manager.create_message(
                                new_message,
                                actor=self.actor,  # Client for write operations (audit trail)
                                client_id=self.client_id,  # From actor (Client)
                                user_id=(
                                    self.user_id if self.user_id else UserManager.ADMIN_USER_ID
                                ),  # Fallback to default user
                            )

                            # append the persisted message ID to the message list
                            message_ids.append(persisted_message.id)

                    # Clear history for all non-chat agents when should_clear_history is True
                    # This applies to meta_memory_agent and all memory sub-agents
                    await self.agent_manager.set_in_context_messages(
                        agent_id=self.agent_state.id,
                        message_ids=message_ids,
                        actor=self.actor,
                    )
                    # Retention: the memory-production machinery (meta_memory_agent
                    # + the 6 memory sub-agents) is a pure transient producer again.
                    # The canonical learnable conversation lives in the dedicated
                    # Conversation Message Store, so the meta agent no longer needs to
                    # retain a transcript for the distiller. All non-chat agents keep
                    # the legacy full delete (retain=0) of detached extraction scratch.
                    await self.message_manager.delete_detached_messages_for_agent(
                        agent_id=self.agent_state.id,
                        actor=self.actor,
                        retain_last_n_sessions=0,
                        user_id=self.user_id,
                    )

                    # Clear all messages since they were manually added to the conversation history
                    messages = []

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
                    session_id=input_session_id,
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

        if must_continue:
            # A call left for a later turn or declined by its tool: the model
            # must get the turn back to make it (again).
            continue_chaining = True

        return messages, continue_chaining, function_failed

    async def _estimate_input_tokens(self, message_sequence: List[Message]) -> int:
        """Estimated input tokens of an LLM call with ``message_sequence`` and
        this agent's tool schemas (the context-budget token counter)."""
        from mirix.services.context_budget import get_token_counter

        messages = []
        for message in message_sequence:
            try:
                messages.append(message.to_openai_dict())
            except Exception:  # noqa: BLE001 — an odd message still counts
                messages.append(message.model_dump(exclude_none=True))
        tools = [t.json_schema for t in self._visible_tools() if t.json_schema]
        return await get_token_counter().acount_request(
            {"messages": messages, "tools": tools}
        )

    async def _check_context_limit(self, message_sequence: List[Message]) -> None:
        """Raise :class:`AgentContextLimitReached` when this step has a context
        limit (``step(context_stop_fraction=...)``) and the LLM call about to be
        made would pass it. Also tracks the step's peak estimated input."""
        limit = getattr(self, "_context_stop_limit", None)
        if limit is None:
            return
        estimated = await self._estimate_input_tokens(message_sequence)
        self._context_peak_tokens = max(getattr(self, "_context_peak_tokens", 0), estimated)
        if estimated > limit:
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: stopping before an LLM call: "
                f"estimated input {estimated} > context limit {limit}"
            )
            raise AgentContextLimitReached(estimated, limit)

    async def step(
        self,
        input_messages: Union[Message, MessageCreate, List[Union[Message, MessageCreate]]],
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        extra_messages: Optional[List[dict]] = None,
        actor: Optional["Client"] = None,  # Client for write operations (audit trail)
        user: Optional[User] = None,  # User for read operations (data scope)
        context_stop_fraction: Optional[float] = None,
        auto_summarize: bool = True,
        hard_step_cap: bool = False,
        strict_llm_errors: Optional[bool] = None,
        max_tool_calls_per_turn: Optional[int] = None,
        **kwargs,
    ) -> MirixUsageStatistics:
        """Run Agent.step in a loop, handling chaining via continue_chaining requests and function failures

        Args:
            actor: Client object for write operations (updating messages, agent state) - audit trail
            user: User object for read operations (loading blocks, memory filtering) - data scope
            context_stop_fraction: when set, the step ends (stop reason
                ``context_limit``) before any LLM call whose estimated input is
                above ``floor(context_window * fraction) - planning output
                reserve`` (the curator passes 0.8).
            auto_summarize: False turns the summarizer off for this step: no
                memory-pressure summary, and a context overflow ends the step
                (``context_limit``) instead of summarizing the input away.
            hard_step_cap: True makes ``max_chaining_steps`` a hard cap: after
                that many LLM calls without finish_memory_update the step stops
                (``step_cap``) instead of granting one more call with a warning.
            strict_llm_errors: True makes a provider error end the step at
                once (it propagates, e.g. an ``LLMServerError`` after the SDK's
                own transport retries): no agent-level retry of it and never
                the fallback that resends only the last message without the
                system prompt. A truncated answer raises
                ``LLMOutputTruncatedError`` and an answer still unusable after
                the agent's retries ``LLMInvalidResponseError``. Default: on
                for a step with ``hard_step_cap`` or without auto-summary (the
                skill curator), off otherwise.
            max_tool_calls_per_turn: at most this many tool calls of one model
                turn run (the first ones); the rest, and a
                send_message / finish_memory_update that is not the turn's
                only call, are answered "Not run" and the chain goes on
                (``tool_call_plan``); a terminal call paired again in the turn
                right after such an answer runs after the turn's other calls
                when they all fit the cap. None (the default): every call
                runs, a terminal call last.

        The returned usage carries ``stop_reason`` (``finish``, ``ended``,
        ``chaining_off``, ``step_cap`` or ``context_limit``; with
        ``hard_step_cap`` only ``finish``, ``step_cap`` or ``context_limit``: a
        turn ended without ``finish_memory_update`` is answered with a reminder
        and the step goes on), the step's peak estimated input (with
        ``context_stop_fraction``), its failed tool calls, the calls left
        unrun (``skipped_tool_calls``) and its turns of several calls
        (``multi_call_turns``).
        """
        if max_tool_calls_per_turn is not None and max_tool_calls_per_turn < 1:
            raise ValueError("max_tool_calls_per_turn must be at least 1")
        previous_auto_summarize = getattr(self, "_auto_summarize", True)
        previous_stop_limit = getattr(self, "_context_stop_limit", None)
        previous_strict = getattr(self, "_strict_llm_errors", False)
        previous_max_calls = getattr(self, "_max_tool_calls_per_turn", None)
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._auto_summarize = auto_summarize
        self._strict_llm_errors = (
            (hard_step_cap or not auto_summarize)
            if strict_llm_errors is None
            else bool(strict_llm_errors)
        )
        self._context_peak_tokens = 0
        self.failed_tool_calls = 0
        self.tool_failures = ToolFailureLog()
        self.skipped_tool_calls = 0
        self.multi_call_turns = 0
        self._terminal_alone_turns = 0
        if context_stop_fraction is not None:
            from mirix.services.context_budget import ModelLimits

            limits = ModelLimits.from_llm_config(self.agent_state.llm_config)
            # floor(window * fraction), exactly (Fraction("0.8") is 4/5).
            fraction = Fraction(str(context_stop_fraction))
            self._context_stop_limit = (
                int(limits.context_window * fraction) - limits.planning_output_reserve
            )
        else:
            self._context_stop_limit = None
        try:
            return await self._step_loop(
                input_messages,
                chaining=chaining,
                max_chaining_steps=max_chaining_steps,
                extra_messages=extra_messages,
                actor=actor,
                user=user,
                hard_step_cap=hard_step_cap,
                **kwargs,
            )
        finally:
            self._auto_summarize = previous_auto_summarize
            self._context_stop_limit = previous_stop_limit
            self._strict_llm_errors = previous_strict
            self._max_tool_calls_per_turn = previous_max_calls

    async def _step_loop(
        self,
        input_messages: Union[Message, MessageCreate, List[Union[Message, MessageCreate]]],
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        extra_messages: Optional[List[dict]] = None,
        actor: Optional["Client"] = None,
        user: Optional[User] = None,
        hard_step_cap: bool = False,
        **kwargs,
    ) -> MirixUsageStatistics:
        """The body of :meth:`step` (see there)."""

        # Store actor for write operations
        if actor:
            self.actor = actor

        # Store user and load user's memory blocks
        if user:
            self.user = user

            # Load blocks for agents that can directly edit core memory.
            if self.agent_state.is_type(AgentType.core_memory_agent, AgentType.auto_dream_agent):
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

        # Load last function response from message history (deferred from __init__)
        if self.actor is not None and self.last_function_response is None:
            self.last_function_response = await self.load_last_function_response()

        max_chaining_steps = max_chaining_steps or MAX_CHAINING_STEPS

        first_input_message = input_messages[0] if isinstance(input_messages, list) else input_messages

        # Derive the session_id for this step once so all synthesized messages
        # (heartbeats, warnings, meta-memory bootstrap, summaries) inherit it.
        # Only the chat_agent inherits it; for every non-chat agent the
        # synthesized messages carry NO session_id (session_id identifies an
        # external conversation, never the meta agent's own bookkeeping).
        step_session_id = (
            getattr(first_input_message, "session_id", None)
            if self.agent_state.is_type(AgentType.chat_agent)
            else None
        )
        # Expose on self so helpers called without an explicit session_id
        # (e.g. summarize_messages_inplace) can pick it up. Other entrypoints
        # that skip step() (e.g. step_user_message) must set/reset this
        # themselves — see step_user_message for the save/restore pattern.
        self._current_step_session_id = step_session_id

        # Convert MessageCreate objects to Message objects
        if not isinstance(input_messages, list):
            input_messages = [input_messages]
        message_objects = [
            (
                m
                if isinstance(m, Message)
                else prepare_input_message_create(
                    m,
                    self.agent_state.id,
                    wrap_user_message=False,
                    wrap_system_message=True,
                )
            )
            for m in input_messages
        ]

        extra_message_objects = (
            [
                prepare_input_message_create(
                    m,
                    self.agent_state.id,
                    wrap_user_message=False,
                    wrap_system_message=True,
                )
                for m in extra_messages
            ]
            if extra_messages is not None
            else None
        )
        next_input_message = message_objects
        counter = 0
        total_usage = UsageStatistics()
        step_count = 0
        stop_reason: Optional[str] = None

        initial_message_count = len(
            await self.agent_manager.get_in_context_messages(
                agent_state=self.agent_state, actor=self.actor, user=self.user
            )
        )

        if self.agent_state.is_type(AgentType.reflexion_agent):
            # clear previous messages
            in_context_messages = await self.agent_manager.get_in_context_messages(
                agent_state=self.agent_state, actor=self.actor, user=self.user
            )
            in_context_messages = in_context_messages[:1]
            await self.agent_manager.set_in_context_messages(
                agent_id=self.agent_state.id,
                message_ids=[message.id for message in in_context_messages],
                actor=self.actor,
            )

        # Initialize the LLM client once per step to reuse across retries.
        llm_client = LLMClient.create(
            llm_config=self.agent_state.llm_config,
        )

        while True:
            kwargs["first_message"] = False
            kwargs["step_count"] = step_count

            if self.agent_state.is_type(AgentType.meta_memory_agent, AgentType.chat_agent) and step_count == 0:
                # When the agent first gets the screenshots, we need to extract the topic to search the query.
                try:
                    topics = await self._extract_topics_from_messages(next_input_message)

                    if topics is not None:
                        kwargs["topics"] = topics
                    else:
                        printv(f"[Mirix.Agent.{self.agent_state.name}] WARNING: No topics extracted from screenshots")

                except Exception as e:
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] INFO: Error in extracting the topic from the screenshots: {e}"
                    )
                    pass

            if self.agent_state.is_type(AgentType.meta_memory_agent) and step_count == 0:
                meta_message = prepare_input_message_create(
                    MessageCreate(
                        role=MessageRole.user,
                        content="[System Message] As the meta memory manager, analyze the provided content and trigger the appropriate memory updates.",
                        name=None,
                        otid=None,
                        sender_id=None,
                        group_id=None,
                        filter_tags=self.filter_tags,
                        session_id=step_session_id,
                    ),
                    self.agent_state.id,
                    wrap_user_message=False,
                    wrap_system_message=True,
                )
                next_input_message.append(meta_message)

            try:
                step_response = await self.inner_step(
                    first_input_messge=first_input_message,
                    messages=next_input_message,
                    extra_messages=extra_message_objects,
                    initial_message_count=initial_message_count,
                    chaining=chaining,
                    llm_client=llm_client,
                    **kwargs,
                )
            except AgentContextLimitReached:
                stop_reason = "context_limit"
                break

            continue_chaining = step_response.continue_chaining
            function_failed = step_response.function_failed
            token_warning = step_response.in_context_memory_warning
            usage = step_response.usage

            step_count += 1
            total_usage += usage
            counter += 1
            self.interface.step_complete()

            # logger.debug("Saving agent state")
            # save updated state
            await save_agent(self)

            natural_end = not continue_chaining and not function_failed
            finished = "finish_memory_update" in (getattr(self, "last_executed_functions", None) or [])

            # Chain stops
            if not chaining and (not function_failed):
                printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: No chaining, stopping after one step")
                stop_reason = "finish" if finished else "chaining_off"
                break
            elif hard_step_cap and natural_end and finished:
                stop_reason = "finish"
                break
            elif hard_step_cap and max_chaining_steps is not None and counter >= max_chaining_steps:
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Hit the step cap, stopping after {counter} steps"
                )
                stop_reason = "step_cap"
                break
            elif hard_step_cap and natural_end:
                # The model ended its turn without finish_memory_update. With a
                # hard step cap (the skill curator) only the finish
                # call, the context limit or the step cap end the step: remind
                # it and go on (the reminder's call counts toward the cap).
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",
                        "content": get_contine_chaining(UNFINISHED_TURN_HEARTBEAT_MESSAGE),
                    },
                    session_id=step_session_id,
                )
                continue
            elif max_chaining_steps is not None and counter == max_chaining_steps:
                # Add warning message based on agent type
                if self.agent_state.is_type(AgentType.chat_agent):
                    warning_content = "[System Message] You have reached the maximum chaining steps. Please call 'send_message' to send your response to the user."
                else:
                    warning_content = "[System Message] You have reached the maximum chaining steps. Please call 'finish_memory_update' to end the chaining."
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",
                        "content": warning_content,
                    },
                    session_id=step_session_id,
                )
                continue  # give agent one more chance to respond
            elif max_chaining_steps is not None and counter > max_chaining_steps:
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Hit max chaining steps, stopping after {counter} steps"
                )
                stop_reason = "finish" if (natural_end and finished) else "step_cap"
                break
            # Chain handlers
            elif (
                token_warning
                and summarizer_settings.send_memory_warning_message
                and getattr(self, "_auto_summarize", True)
            ):
                assert self.agent_state.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_token_limit_warning(),
                    },
                    session_id=step_session_id,
                )
                continue  # always chain
            elif function_failed:
                assert self.agent_state.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_contine_chaining(FUNC_FAILED_HEARTBEAT_MESSAGE),
                    },
                    session_id=step_session_id,
                )
                continue  # always chain
            elif continue_chaining:
                assert self.agent_state.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_contine_chaining(REQ_HEARTBEAT_MESSAGE),
                    },
                    session_id=step_session_id,
                )
                continue  # always chain
            # Mirix no-op / yield
            else:
                stop_reason = "finish" if finished else "ended"
                break

        # Save the message_ids
        await save_agent(self)

        failures = getattr(self, "tool_failures", None)
        return MirixUsageStatistics(
            **total_usage.model_dump(),
            step_count=step_count,
            stop_reason=stop_reason,
            context_peak_tokens=getattr(self, "_context_peak_tokens", 0),
            failed_tool_calls=getattr(self, "failed_tool_calls", 0),
            tool_failures=failures.as_list() if failures is not None else [],
            tool_failures_omitted=failures.omitted if failures is not None else 0,
            skipped_tool_calls=getattr(self, "skipped_tool_calls", 0),
            multi_call_turns=getattr(self, "multi_call_turns", 0),
        )

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
            embedded_text = await (await embedding_model(self.agent_state.embedding_config)).get_text_embedding(key_words)
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

        # Procedural memory (skills) is intentionally NOT injected into any
        # agent's system prompt. Skills are a RETRIEVAL TARGET, not ambient
        # context:
        #   * the procedural_memory_agent surveys them ON DEMAND via its
        #     `skill_list` / `skill_read` tools (its system prompt explicitly
        #     instructs "Survey existing skills: Call skill_list()"), and
        #   * task-executing / external agents fetch them through the unified
        #     search interface (GET /memory/search?memory_type=procedural, or
        #     the search_in_memory tool).
        # Passively broadcasting the whole skill list into every agent's prompt
        # was redundant for the owner (it uses the tool) and pure token noise
        # for the pure-extractor agents (episodic/semantic/resource/...), which
        # never act on skills. Removing it also drops a per-prompt-build
        # retrieval from the hot path.

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

        # NOTE: no <procedural_memory> block — skills are retrieved on demand
        # (procedural_memory_agent via skill_list; consumers via /memory/search),
        # not broadcast into every prompt. See the procedural-retrieval removal
        # note earlier in this method.

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

        in_context_messages = await self.agent_manager.get_in_context_messages(
            agent_state=self.agent_state, actor=self.actor, user=self.user
        )
        raw_system = (
            in_context_messages[0].content[0].text
            if in_context_messages and in_context_messages[0].role == MessageRole.system
            else ""
        )

        # Build the complete system prompt with memories
        complete_system_prompt, _ = await self.build_system_prompt_with_memories(raw_system=raw_system, topics=topics)

        return complete_system_prompt

    async def inner_step(
        self,
        first_input_messge: Message,
        messages: Union[Message, List[Message]],
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        stream: bool = False,  # TODO move to config?
        step_count: Optional[int] = None,
        metadata: Optional[dict] = None,
        summarize_attempt_count: int = 0,
        force_response: bool = False,
        topics: Optional[str] = None,
        retrieved_memories: Optional[dict] = None,
        display_intermediate_message: any = None,
        request_user_confirmation: Optional[Callable] = None,
        existing_file_uris: Optional[List[str]] = None,
        extra_messages: Optional[List[dict]] = None,
        initial_message_count: Optional[int] = None,
        return_memory_types_without_update: bool = False,
        message_queue: Optional[any] = None,
        chaining: bool = True,
        llm_client: Optional[LLMClient] = None,
        **kwargs,
    ) -> AgentStepResponse:
        """Runs a single step in the agent loop (generates at most one LLM call)"""

        try:
            # Log the start of each reasoning step
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: Starting agent step - step_count: {step_count}, chaining: {chaining}"
            )
            if topics:
                printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Step topics: {topics}")

            # previous_in_context_messages = self.agent_state.message_ids
            # new_message_ids = self.agent_manager.get_agent_by_id(agent_id=self.agent_state.id, actor=self.user).message_ids

            # Step 0: get in-context messages and get the raw system prompt
            in_context_messages = await self.agent_manager.get_in_context_messages(
                agent_state=self.agent_state, actor=self.actor, user=self.user
            )

            assert in_context_messages[0].role == MessageRole.system
            raw_system = in_context_messages[0].content[0].text

            # Build the complete system prompt with memories
            complete_system_prompt, retrieved_memories = await self.build_system_prompt_with_memories(
                raw_system=raw_system,
                topics=topics,
                retrieved_memories=retrieved_memories,
            )

            in_context_messages[0].content[0].text = complete_system_prompt

            # Step 1: add user message
            if isinstance(messages, Message):
                messages = [messages]

            if not all(isinstance(m, Message) for m in messages):
                raise ValueError(f"messages should be a Message or a list of Message, got {type(messages)}")

            input_message_sequence = in_context_messages + messages

            if extra_messages is not None:
                input_message_sequence = (
                    input_message_sequence[:initial_message_count]
                    + extra_messages
                    + input_message_sequence[initial_message_count:]
                )

            if len(input_message_sequence) > 1 and input_message_sequence[-1].role != "user":
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] WARNING: {CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue"
                )

            # Before the LLM call: a step with a context limit stops
            # here when the call's estimated input would pass it.
            await self._check_context_limit(input_message_sequence)

            # Step 2: send the conversation and available functions to the LLM
            response = await self._get_ai_reply(
                message_sequence=input_message_sequence,
                first_message=first_message,
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
                    first_input_messge,  # give the last message to the function so that other agents can see this message through funciton_calls
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

            # if function_failed:

            #     inputs = self._get_ai_reply(
            #         message_sequence=input_message_sequence,
            #         first_message=first_message,
            #         stream=stream,
            #         step_count=step_count,
            #         # extra_messages=extra_messages,
            #         get_input_data_for_debugging=True
            #     )

            #     try:
            #         error = json.loads(all_response_messages[-1].content[0].text)
            #     except:
            #         error = 'Not Known'

            #     response_json = response.model_dump()
            #     response_json.pop('created', None)
            #     results_to_log = {
            #         'input': inputs,
            #         'output': response_json,
            #         'error': error
            #     }

            #     if not os.path.exists("debug"):
            #         os.makedirs("debug")
            #     count = 0
            #     while os.path.exists(f"debug/debug_{count}.json"):
            #         count += 1
            #     with open(f"debug/debug_{count}.json", "w") as f:
            #         json.dump(results_to_log, f, indent=2)

            # Step 6: extend the message history
            if len(messages) > 0:
                all_new_messages = messages + all_response_messages
            else:
                all_new_messages = all_response_messages

            # Check the memory pressure and potentially issue a memory pressure warning
            current_total_tokens = response.usage.total_tokens
            active_memory_warning = False

            # We can't do summarize logic properly if context_window is undefined
            if self.agent_state.llm_config.context_window is None:
                # Fallback if for some reason context_window is missing, just set to the default
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] WARNING: Could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}"
                )
                printv(f"[Mirix.Agent.{self.agent_state.name}] DEBUG: Agent state: {self.agent_state}")
                self.agent_state.llm_config.context_window = (
                    LLM_MAX_TOKENS[self.model]
                    if (self.model is not None and self.model in LLM_MAX_TOKENS)
                    else LLM_MAX_TOKENS["DEFAULT"]
                )

            # Log step - this must happen before messages are persisted
            step = await self.step_manager.log_step(
                actor=self.actor,
                provider_name=self.agent_state.llm_config.model_endpoint_type,
                model=self.agent_state.llm_config.model,
                context_window_limit=self.agent_state.llm_config.context_window,
                usage=response.usage,
            )
            for message in all_new_messages:
                message.step_id = step.id

            # Persisting into Messages - MUST happen before summarization
            # so that summarize_messages_inplace can see all messages
            self.agent_state = await self.agent_manager.append_to_in_context_messages(
                all_new_messages,
                agent_id=self.agent_state.id,
                actor=self.actor,
                user_id=self.user_id,
            )

            # Check memory pressure AFTER messages are persisted. A step with
            # auto-summary off (the skill curator) never summarizes:
            # it would summarize away the input it is working on.
            if getattr(self, "_auto_summarize", True) and current_total_tokens > (
                summarizer_settings.memory_warning_threshold
                * int(self.agent_state.llm_config.context_window)
            ):
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Memory pressure detected: last response total_tokens ({current_total_tokens}) > {summarizer_settings.memory_warning_threshold * int(self.agent_state.llm_config.context_window)}"
                )

                # Only deliver the alert if we haven't already (this period)
                if not self.agent_alerted_about_memory_pressure:
                    active_memory_warning = True
                    self.agent_alerted_about_memory_pressure = True  # it's up to the outer loop to handle this

                # if it is too long then run summarization here.
                await self.summarize_messages_inplace(existing_file_uris=existing_file_uris)

            else:
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] DEBUG: Memory usage acceptable: last response total_tokens ({current_total_tokens}) < {summarizer_settings.memory_warning_threshold * int(self.agent_state.llm_config.context_window)}"
                )

            # Log step completion and results
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: Agent step completed - continue_chaining: {continue_chaining}, function_failed: {function_failed}, messages_generated: {len(all_new_messages)}"
            )

            return AgentStepResponse(
                messages=all_new_messages,
                continue_chaining=continue_chaining,
                function_failed=function_failed,
                in_context_memory_warning=active_memory_warning,
                usage=response.usage,
            )

        except (AbortAgentStep, AgentContextLimitReached):
            raise
        except Exception as e:
            printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: step() failed\nmessages = {messages}\nerror = {e}")

            if is_context_overflow_error(e) and not getattr(self, "_auto_summarize", True):
                # No summarizer here: the input does not fit, stop the step.
                raise AgentContextLimitReached(
                    getattr(e, "estimated_input_tokens", None),
                    getattr(self, "_context_stop_limit", None),
                ) from e

            # If we got a context alert, try trimming the messages length, then try again
            if is_context_overflow_error(e):
                in_context_messages = await self.agent_manager.get_in_context_messages(
                    agent_state=self.agent_state, actor=self.actor, user=self.user
                )

                if summarize_attempt_count <= summarizer_settings.max_summarizer_retries:
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] WARNING: context window exceeded with limit {self.agent_state.llm_config.context_window}, attempting to summarize ({summarize_attempt_count}/{summarizer_settings.max_summarizer_retries}"
                    )
                    # A separate API call to run a summarizer
                    await self.summarize_messages_inplace(existing_file_uris=existing_file_uris)

                    # Try step again
                    return await self.inner_step(
                        messages=messages,
                        first_message=first_message,
                        first_input_messge=first_input_messge,
                        first_message_retry_limit=first_message_retry_limit,
                        skip_verify=skip_verify,
                        stream=stream,
                        metadata=metadata,
                        summarize_attempt_count=summarize_attempt_count + 1,
                        force_response=force_response,
                        extra_messages=extra_messages,
                        topics=topics,
                        retrieved_memories=retrieved_memories,
                        chaining=chaining,
                        message_queue=message_queue,
                        initial_message_count=initial_message_count,
                        return_memory_types_without_update=return_memory_types_without_update,
                        display_intermediate_message=display_intermediate_message,
                        request_user_confirmation=request_user_confirmation,
                        existing_file_uris=existing_file_uris,
                        llm_client=llm_client,
                    )
                else:
                    err_msg = f"Ran summarizer {summarize_attempt_count - 1} times for agent id={self.agent_state.id}, but messages are still overflowing the context window."
                    token_counts = (get_token_counts_for_messages(in_context_messages),)
                    printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: {err_msg}")
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] ERROR: num_in_context_messages: {len(self.agent_state.message_ids)}"
                    )
                    printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: token_counts: {token_counts}")
                    raise ContextWindowExceededError(
                        err_msg,
                        details={
                            "num_in_context_messages": len(self.agent_state.message_ids),
                            "in_context_messages_text": [m.text for m in in_context_messages],
                            "token_counts": token_counts,
                        },
                    )

            else:
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] ERROR: step() failed with an unrecognized exception: '{str(e)}'"
                )
                raise e

    async def step_user_message(
        self,
        user_message_str: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AgentStepResponse:
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

        # session_id identifies an external conversation and is only stamped on the
        # chat_agent's messages. For any non-chat agent reaching this entrypoint the
        # message (and the step-level session context below) must carry NO session_id,
        # matching the guard applied to input_session_id/step_session_id in step().
        effective_session_id = (
            session_id if self.agent_state.is_type(AgentType.chat_agent) else None
        )

        # Create the associated Message object (in the database)
        assert self.agent_state.created_by_id is not None, "User ID is not set"
        user_message = Message.dict_to_message(
            agent_id=self.agent_state.id,
            model=self.model,
            openai_message_dict=openai_message_dict,
            session_id=effective_session_id,
            # created_at=timestamp,
        )

        # Seed the step-level session context so any pre-persist summarization
        # triggered inside inner_step() / _get_ai_reply() stamps the correct
        # session_id on the summary message.
        prev_session_id = getattr(self, "_current_step_session_id", None)
        self._current_step_session_id = effective_session_id
        try:
            return await self.inner_step(messages=[user_message], **kwargs)
        finally:
            self._current_step_session_id = prev_session_id

    async def summarize_messages_inplace(
        self,
        existing_file_uris: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ):
        in_context_messages = await self.agent_manager.get_in_context_messages(
            agent_state=self.agent_state, actor=self.actor, user=self.user
        )
        in_context_messages_openai = [m.to_openai_dict() for m in in_context_messages]
        in_context_messages_openai_no_system = in_context_messages_openai[1:]
        token_counts = get_token_counts_for_messages(in_context_messages)
        logger.info("System message token count=%s", token_counts[0])
        logger.info("token_counts_no_system=%s", token_counts[1:])

        if in_context_messages_openai[0]["role"] != "system":
            raise RuntimeError(
                f"in_context_messages_openai[0] should be system (instead got {in_context_messages_openai[0]})"
            )

        # If at this point there's nothing to summarize, throw an error
        if len(in_context_messages_openai_no_system) == 0:
            raise ContextWindowExceededError(
                "Not enough messages to compress for summarization",
                details={
                    "num_candidate_messages": len(in_context_messages_openai_no_system),
                    "num_total_messages": len(in_context_messages_openai),
                },
            )

        cutoff = calculate_summarizer_cutoff(
            in_context_messages=in_context_messages,
            token_counts=token_counts,
            logger=self.logger,
        )

        message_sequence_to_summarize = in_context_messages[1:cutoff]  # do NOT get rid of the system message
        self.logger.info(
            f"Attempting to summarize {len(message_sequence_to_summarize)} messages of {len(in_context_messages)}"
        )

        # We can't do summarize logic properly if context_window is undefined
        if self.agent_state.llm_config.context_window is None:
            # Fallback if for some reason context_window is missing, just set to the default
            self.logger.warning(
                f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}"
            )
            self.agent_state.llm_config.context_window = (
                LLM_MAX_TOKENS[self.model]
                if (self.model is not None and self.model in LLM_MAX_TOKENS)
                else LLM_MAX_TOKENS["DEFAULT"]
            )

        summary = await summarize_messages(
            agent_state=self.agent_state,
            message_sequence_to_summarize=message_sequence_to_summarize,
            existing_file_uris=existing_file_uris,
        )
        logger.info("Got summary: %s", summary)

        # Metadata that's useful for the agent to see
        all_time_message_count = await self.message_manager.size(
            agent_id=self.agent_state.id, actor=self.actor, user_id=self.user_id
        )
        remaining_message_count = 1 + len(in_context_messages) - cutoff  # System + remaining
        hidden_message_count = all_time_message_count - remaining_message_count
        summary_message_count = len(message_sequence_to_summarize)
        summary_message = package_summarize_message(
            summary, summary_message_count, hidden_message_count, all_time_message_count
        )
        logger.info("Packaged into message: %s", summary_message)

        prior_len = len(in_context_messages_openai)
        self.agent_state = await self.agent_manager.trim_older_in_context_messages(
            num=cutoff,
            agent_id=self.agent_state.id,
            actor=self.actor,
            user_id=self.user_id,
        )
        packed_summary_message = {"role": "user", "content": summary_message}

        # Prepend the summary. The summary is a synthesized message, so it follows
        # the same rule as every other synthesized message: only the chat_agent
        # carries a session_id; for any non-chat (memory-production) agent the
        # summary carries NONE. Within the chat_agent, preference order is:
        #   1. explicit `session_id` argument (caller-provided, e.g. step_user_message)
        #   2. _current_step_session_id stashed by Agent.step() for the current step
        #   3. the latest in-context message's session_id (inherits whatever session
        #      the conversation being summarized is already in)
        summary_session_id = None
        if self.agent_state.is_type(AgentType.chat_agent):
            summary_session_id = session_id
            if summary_session_id is None:
                summary_session_id = getattr(self, "_current_step_session_id", None)
            if summary_session_id is None:
                for m in reversed(in_context_messages):
                    if getattr(m, "session_id", None):
                        summary_session_id = m.session_id
                        break
        self.agent_state = await self.agent_manager.prepend_to_in_context_messages(
            messages=[
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict=packed_summary_message,
                    session_id=summary_session_id,
                )
            ],
            agent_id=self.agent_state.id,
            actor=self.actor,
            user_id=self.user_id,
        )

        # reset alert
        self.agent_alerted_about_memory_pressure = False
        curr_in_context_messages = await self.agent_manager.get_in_context_messages(
            agent_state=self.agent_state, actor=self.actor, user=self.user
        )

        self.logger.info(f"Ran summarizer, messages length {prior_len} -> {len(curr_in_context_messages)}")
        self.logger.info(
            f"Summarizer brought down total token count from {sum(token_counts)} -> {sum(get_token_counts_for_messages(curr_in_context_messages))}"
        )

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

    async def get_context_window(self) -> ContextWindowOverview:
        """Get the context window of the agent"""

        system_prompt = self.agent_state.system  # TODO is this the current system or the initial system?
        num_tokens_system = count_tokens(system_prompt)
        core_memory = self.blocks_in_memory.compile() if self.blocks_in_memory else ""
        num_tokens_core_memory = count_tokens(core_memory)

        # Grab the in-context messages
        # conversion of messages to OpenAI dict format, which is passed to the token counter
        in_context_messages = await self.agent_manager.get_in_context_messages(
            agent_state=self.agent_state, actor=self.actor, user=self.user
        )
        in_context_messages_openai = [m.to_openai_dict() for m in in_context_messages]

        # Check if there's a summary message in the message queue
        if (
            len(in_context_messages) > 1
            and in_context_messages[1].role == MessageRole.user
            and isinstance(in_context_messages[1].text, str)
            # TODO remove hardcoding
            and "The following is a summary of the previous " in in_context_messages[1].text
        ):
            # Summary message exists
            assert in_context_messages[1].text is not None
            summary_memory = in_context_messages[1].text
            num_tokens_summary_memory = count_tokens(in_context_messages[1].text)
            # with a summary message, the real messages start at index 2
            num_tokens_messages = (
                num_tokens_from_messages(messages=in_context_messages_openai[2:], model=self.model)
                if len(in_context_messages_openai) > 2
                else 0
            )

        else:
            summary_memory = None
            num_tokens_summary_memory = 0
            # with no summary message, the real messages start at index 1
            num_tokens_messages = (
                num_tokens_from_messages(messages=in_context_messages_openai[1:], model=self.model)
                if len(in_context_messages_openai) > 1
                else 0
            )

        message_manager_size = await self.message_manager.size(
            actor=self.actor, agent_id=self.agent_state.id, user_id=self.user_id
        )
        external_memory_summary = compile_memory_metadata_block(
            memory_edit_timestamp=get_utc_time(),
            previous_message_count=await self.message_manager.size(
                actor=self.actor, agent_id=self.agent_state.id, user_id=self.user_id
            ),
        )
        num_tokens_external_memory_summary = count_tokens(external_memory_summary)

        # tokens taken up by function definitions
        agent_state_tool_jsons = [t.json_schema for t in self._visible_tools()]
        if agent_state_tool_jsons:
            available_functions_definitions = [
                ChatCompletionRequestTool(type="function", function=f) for f in agent_state_tool_jsons
            ]
            num_tokens_available_functions_definitions = num_tokens_from_functions(
                functions=agent_state_tool_jsons, model=self.model
            )
        else:
            available_functions_definitions = []
            num_tokens_available_functions_definitions = 0

        num_tokens_used_total = (
            num_tokens_system  # system prompt
            + num_tokens_available_functions_definitions  # function definitions
            + num_tokens_core_memory  # core memory
            + num_tokens_external_memory_summary  # metadata (statistics) about recall/archival
            + num_tokens_summary_memory  # summary of ongoing conversation
            + num_tokens_messages  # tokens taken by messages
        )
        assert isinstance(num_tokens_used_total, int)

        return ContextWindowOverview(
            # context window breakdown (in messages)
            num_messages=len(in_context_messages),
            num_recall_memory=message_manager_size,
            num_tokens_external_memory_summary=num_tokens_external_memory_summary,
            external_memory_summary=external_memory_summary,
            # top-level information
            context_window_size_max=self.agent_state.llm_config.context_window,
            context_window_size_current=num_tokens_used_total,
            # context window breakdown (in tokens)
            num_tokens_system=num_tokens_system,
            system_prompt=system_prompt,
            num_tokens_core_memory=num_tokens_core_memory,
            core_memory=core_memory,
            num_tokens_summary_memory=num_tokens_summary_memory,
            summary_memory=summary_memory,
            num_tokens_messages=num_tokens_messages,
            messages=in_context_messages,
            # related to functions
            num_tokens_functions_definitions=num_tokens_available_functions_definitions,
            functions_definitions=available_functions_definitions,
        )

    async def count_tokens(self) -> int:
        """Count the tokens in the current context window"""
        context_window_breakdown = await self.get_context_window()
        return context_window_breakdown.context_window_size_current


async def save_agent(agent: Agent):
    """Save agent to metadata store"""
    agent_state = agent.agent_state

    # TODO: move this to agent manager
    # TODO: Completely strip out metadata
    # convert to persisted model
    agent_manager = AgentManager()
    update_agent = UpdateAgent(
        message_ids=agent_state.message_ids,
        # TODO: Add this back in later
        # tool_exec_environment_variables=agent_state.get_agent_env_vars_as_dict(),
    )
    await agent_manager.update_agent(agent_id=agent_state.id, agent_update=update_agent, actor=agent.actor)


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
