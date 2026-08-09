import asyncio
import copy
import json
import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytz

from mirix.agent.tool_normalizers import normalize_tool_args
from mirix.agent.tool_validators import validate_tool_args
from mirix.constants import (
    CHAINING_FOR_MEMORY_UPDATE,
    CLI_WARNING_PREFIX,
    FIRST_MESSAGE_ATTEMPTS,
    FUNC_FAILED_HEARTBEAT_MESSAGE,
    HYBRID_READ_WINDOW_SECONDS,
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
from mirix.errors import (
    ContextWindowExceededError,
    CorrectableToolError,
    LLMBadResponseShapeError,
    LLMChainingExhaustedError,
)
from mirix.functions.functions import get_function_from_module
from mirix.helpers import ToolRulesSolver
from mirix.helpers.message_helpers import prepare_input_message_create
from mirix.interface import AgentInterface
from mirix.llm_api.helpers import get_token_counts_for_messages, is_context_overflow_error
from mirix.llm_api.llm_api_tools import create
from mirix.llm_api.llm_client import LLMClient
from mirix.log import get_logger
from mirix.memory import summarize_messages
from mirix.observability.context import (
    get_trace_context,
    mark_observation_as_child,
)
from mirix.observability.context import stamp_tid as _tid_stamped
from mirix.observability.langfuse_client import get_langfuse_client
from mirix.observability.skip_spans import emit_idempotency_skip_span
from mirix.queue.error_policy import Bucket, classify
from mirix.schemas.agent import AgentState, AgentStepResponse
from mirix.schemas.block import BlockUpdate
from mirix.schemas.client import Client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import MessageRole, ToolType
from mirix.schemas.knowledge_vault import KnowledgeVaultItem as PydanticKnowledgeVaultItem
from mirix.schemas.memory import Memory
from mirix.schemas.message import Message, MessageCreate
from mirix.schemas.mirix_message_content import CloudFileContent, FileContent, ImageContent, TextContent
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse
from mirix.schemas.openai.chat_completion_response import Message as ChatCompletionMessage
from mirix.schemas.openai.chat_completion_response import UsageStatistics
from mirix.schemas.procedural_memory import ProceduralMemoryItem as PydanticProceduralMemoryItem
from mirix.schemas.resource_memory import ResourceMemoryItem as PydanticResourceMemoryItem
from mirix.schemas.semantic_memory import SemanticMemoryItem as PydanticSemanticMemoryItem
from mirix.schemas.tool import Tool
from mirix.schemas.tool_rule import TerminalToolRule
from mirix.schemas.usage import MirixUsageStatistics
from mirix.schemas.user import User
from mirix.services.agent_manager import AgentManager
from mirix.services.block_manager import BlockManager
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.helpers.agent_manager_helper import check_supports_structured_output
from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
from mirix.services.memory_source_manager import MemorySourceManager
from mirix.services.message_manager import MessageManager
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.resource_memory_manager import ResourceMemoryManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.services.source_message_manager import SourceMessageManager
from mirix.services.step_manager import StepManager
from mirix.services.tool_execution_sandbox import ToolExecutionSandbox
from mirix.services.user_manager import UserManager
from mirix.settings import settings
from mirix.system import get_contine_chaining, get_token_limit_warning, package_function_response, package_user_message
from mirix.testing import fault_injection
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


_EPISODIC_ITEM_KEYS = {
    "episodic_memory_insert": "items",
    "episodic_memory_replace": "new_items",
}


def _preprocess_episodic_tool_args(
    function_name: str,
    function_args: dict,
    timezone_str: str,
    occurred_at_override,
) -> None:
    """Normalize LLM-emitted ``occurred_at`` values before dispatch.

    When the API supplies an ``occurred_at`` override, the LLM's value is
    discarded downstream (see memory_tools.episodic_memory_insert), so we
    skip parsing — otherwise a malformed LLM timestamp would crash a turn
    whose real timestamp is already known-good.
    """
    if occurred_at_override is not None:
        return

    key = _EPISODIC_ITEM_KEYS.get(function_name)
    if key is None or key not in function_args:
        return

    for item in function_args[key]:
        if "occurred_at" in item:
            item["occurred_at"] = convert_timezone_to_utc(item["occurred_at"], timezone_str)


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

        # Memory source fields — set by _step() when memory_source_id is present
        self.memory_source_id = None
        self.direct_writes: Optional[List[Dict[str, Any]]] = None
        self.external_id = None
        self.external_thread_id = None
        self.source_type = None
        self.source_system = None
        self.source_metadata = None
        self.source_summary = None
        self.source_summary_source = None
        self.summarize = False
        # Original per-turn messages (as plain dicts) for source_message persistence.
        # These exist separately from the packed input_messages because the add_memory
        # handler flattens all turns into a single MessageCreate with [USER]/[ASSISTANT]
        # markers for agent processing, which loses per-message identity. source_messages
        # preserves the original role, external_message_id, and occurred_at per turn.
        # See the comment in queue_util.py put_messages() for the full explanation.
        self.source_messages = None

        # The save's write scope, derived from filter_tags["scope"] (the client's
        # write_scope, set by the server when queuing work).  Used to scope all
        # memory retrieval — blocks, episodic, semantic, procedural, resource,
        # knowledge_vault — so the LLM prompt only sees the current scope's data.
        scope = self.filter_tags.get("scope") if self.filter_tags else None
        self._save_scopes: list[str] | None = [scope] if scope else None

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

        # Memory source managers
        self.memory_source_manager = MemorySourceManager()
        self.source_message_manager = SourceMessageManager()

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

            # refresh memory from DB (using block ids), scoped to this save's
            # scope so the reloaded set never pulls in another scope's blocks
            # (mirrors step()/_retrieve_core; see VEPAGE-1474).
            blocks_result = await self.block_manager.get_blocks(
                user=self.user,
                any_scopes=self._save_scopes,
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
            any_scopes=self._save_scopes,
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

        async def _execute_tool_inner() -> str:
            """Inner function to execute tool. Returns the tool's response string.

            Exceptions are NOT caught here. The outer `_handle_ai_response`
            catches `CorrectableToolError` exclusively for the bounded
            re-prompt path (LLM produced bad args / malformed JSON / failed
            validation). Everything else (DB / provider / code bug)
            propagates out so `process_with_policy` can classify it.
            """
            nonlocal function_args  # Allow modification of outer function_args

            _preprocess_episodic_tool_args(
                function_name,
                function_args,
                timezone_str=self.user.timezone,
                occurred_at_override=getattr(self, "occurred_at", None),
            )

            # Test-only fault injection (inert in prod): inject a fault in the
            # tool body for sources whose directive targets "tool_body"
            # (optionally scoped to this tool name). Raising here — inside the
            # tool body, in pure agent code with no provider frame — is what
            # makes an `attribute_error` shape classify PERMANENT via the
            # error_policy origin-split, and a `correctable` shape feed the
            # bounded LLM re-prompt. Sub-agents inherit memory_source_id from the
            # meta-agent, so the per-source directive matches across the fan-out.
            fault_injection.maybe_raise(
                "tool_body",
                source_key=getattr(self, "memory_source_id", None),
                tool=function_name,
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

            return function_response

        # Execute with Langfuse tracing if available
        if langfuse and trace_id:
            from typing import cast

            from langfuse.types import TraceContext

            # Build trace context
            trace_context_dict: dict = {"trace_id": trace_id}
            if parent_span_id:
                trace_context_dict["parent_span_id"] = parent_span_id

            from mirix.observability.context import current_observation_id, set_trace_context

            # The fallback below is for the case where the langfuse SDK
            # itself fails *before* the tool body executes (so the tool
            # never ran). Once we are about to invoke the body, set
            # `inner_attempted=True` so any exception from the body — or
            # from the langfuse machinery after the body — does NOT trigger
            # a silent re-run of the tool. Non-correctable exceptions
            # raised by the body propagate out so `process_with_policy`
            # can classify them rather than being silently swallowed.
            inner_attempted = False
            try:
                with langfuse.start_as_current_observation(
                    name=f"tool: {function_name}",
                    as_type="tool",
                    trace_context=cast(TraceContext, trace_context_dict),
                    input={"tool_name": function_name, "args": args_for_trace},
                    metadata=_tid_stamped(
                        {
                            "tool_type": str(target_mirix_tool.tool_type),
                            "tool_name": function_name,
                            "agent_name": self.agent_state.name,
                        }
                    ),
                ) as span:
                    mark_observation_as_child(span)

                    # Publish this tool span as the current observation while the
                    # tool body runs so any child observation opened during
                    # execution (e.g. "Resolve Child Agents" or the memory
                    # sub-agent spans spawned by trigger_memory_update) nests under
                    # the tool span rather than the agent-level span above it.
                    # Restore the prior observation id afterward so the next
                    # sibling tool span parents back to the agent span.
                    span_observation_id = getattr(span, "id", None)
                    if span_observation_id:
                        set_trace_context(
                            trace_id=trace_id,
                            observation_id=span_observation_id,
                            user_id=trace_context.get("user_id"),
                            session_id=trace_context.get("session_id"),
                        )
                    inner_attempted = True
                    try:
                        function_response = await _execute_tool_inner()
                    finally:
                        # Restore the prior observation id (set_trace_context ignores
                        # a falsy observation_id, so set the ContextVar directly to
                        # also handle the None / no-parent case correctly).
                        current_observation_id.set(parent_span_id)

                    try:
                        # We only reach here on a clean run — _execute_tool_inner
                        # either returns the response string or raises (which
                        # propagates past the span.update call).
                        span.update(
                            output={"response": str(function_response)},
                            metadata={
                                "tool_type": str(target_mirix_tool.tool_type),
                                "tool_name": function_name,
                            },
                        )
                    except Exception as span_exc:
                        # span.update failure is observability-only — log at
                        # WARNING so a sustained tracing problem is visible
                        # without taking the save path down.
                        self.logger.warning("Langfuse span.update failed: %s", span_exc)
            except Exception as e:
                if inner_attempted:
                    # Tool body already ran (and may have raised). Do NOT
                    # re-execute; propagate the original exception unchanged.
                    raise
                # Langfuse SDK failed before the tool body ran — fall back
                # to running without tracing. Sustained breakage here means
                # we're losing observability across many saves; log at
                # WARNING so it's not silent.
                self.logger.warning("Langfuse tool execution trace failed before body ran: %s", e)
                function_response = await _execute_tool_inner()
        else:
            function_response = await _execute_tool_inner()

        return function_response

    @trace_method
    async def _get_ai_reply(
        self,
        message_sequence: List[Message],
        function_call: Optional[str] = None,
        first_message: bool = False,
        stream: bool = False,  # TODO move to config?
        step_count: Optional[int] = None,
        last_function_failed: bool = False,
        get_input_data_for_debugging: bool = False,
        existing_file_uris: Optional[List[str]] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> ChatCompletionResponse:
        """Call the LLM once, with a bounded retry for Transient errors only.

        Transient errors (rate limits, 5xx, connection errors) are retried a
        few times with short exponential backoff. Permanent errors (422, 400,
        401, 403) propagate on the first occurrence so the caller's policy
        wrapper can classify and decide what to do.

        Budget settings (mirix/settings.py):
            llm_inline_retry_max_attempts, llm_inline_retry_base_seconds,
            llm_inline_retry_max_delay.
        """
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

        max_attempts = max(1, settings.llm_inline_retry_max_attempts)
        base = settings.llm_inline_retry_base_seconds
        cap = settings.llm_inline_retry_max_delay
        # Total tries = 1 initial + max_attempts retries.
        for attempt in range(max_attempts + 1):
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
                        function_call=function_call,
                        first_message=first_message,
                        force_tool_call=force_tool_call,
                        stream=stream,
                        stream_interface=self.interface,
                        name=self.agent_state.name,
                    )
                log_telemetry(self.logger, "_get_ai_reply create finish")

                # Validate the response shape. LLMBadResponseShapeError is in
                # _TRANSIENT_TYPES, so the catch below + classify() treat
                # these as provider quirks worth retrying (not as code bugs).
                if len(response.choices) == 0 or response.choices[0] is None:
                    raise LLMBadResponseShapeError(f"API call returned an empty message: {response}")

                for choice in response.choices:
                    if choice.message.content == "" and len(choice.message.tool_calls) == 0:
                        raise LLMBadResponseShapeError(f"API call returned an empty message: {response}")

                if response.choices[0].finish_reason not in [
                    "stop",
                    "function_call",
                    "tool_calls",
                ]:
                    # "length" finish_reason historically retried — same
                    # classification path as the other shape failures.
                    raise LLMBadResponseShapeError(f"Bad finish reason from API: {response.choices[0].finish_reason}")

                log_telemetry(self.logger, "_handle_ai_response finish")
                return response

            except Exception as exc:
                bucket = classify(exc)
                if bucket is Bucket.PERMANENT:
                    # Permanent errors are not retryable. Propagate to the caller
                    # immediately so the outer policy wrapper can classify and
                    # decide whether to ack or surface the failure.
                    log_telemetry(self.logger, "_get_ai_reply permanent")
                    self.logger.info(
                        "[Mirix.Agent.%s] _get_ai_reply: permanent error (%s) — propagating immediately",
                        self.agent_state.name,
                        type(exc).__name__,
                    )
                    raise
                if attempt >= max_attempts:
                    # Budget exhausted on a Transient. Tag with the inner-
                    # exhausted marker so process_with_policy doesn't add
                    # another whole-step retry cycle on top — that would
                    # multiply inner × whole-step attempts (an exhausted
                    # 3-tier inner × 3-tier whole-step = 9 LLM calls per
                    # save under a sustained 429).
                    log_telemetry(self.logger, "_get_ai_reply transient exhausted")
                    self.logger.warning(
                        "[Mirix.Agent.%s] _get_ai_reply: transient retries exhausted after %d attempts: %s",
                        self.agent_state.name,
                        max_attempts + 1,
                        f"{type(exc).__name__}: {exc!r}",
                    )
                    from mirix.queue.error_policy import mark_inner_exhausted

                    raise mark_inner_exhausted(exc)
                delay = min(base * (2**attempt), cap)
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] WARNING: Attempt {attempt + 1} failed: "
                    f"{type(exc).__name__}: {exc!r}. Retrying in {delay} seconds..."
                )
                await asyncio.sleep(delay)
                continue

        # Unreachable: the loop either returns or raises. Defensive only.
        log_telemetry(self.logger, "_get_ai_reply unreachable")
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
                # Do not log raw LLM completion content to Splunk — content
                # itself rides through the masked Langfuse trace. Length is
                # enough to confirm "model produced something" at the log
                # level.
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Inner thoughts emitted: "
                    f"len={len(response_message.content or '')}"
                )
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
                function_args = {}  # Populated below; ensured set for the except cleanup.

                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Processing tool call {tool_call_idx + 1}/{len(response_message.tool_calls)}: {function_name} with tool_call_id: {tool_call_id}"
                )

                # Every per-tool-call failure mode that the LLM can self-correct
                # on a re-prompt is funneled through `CorrectableToolError`:
                # unknown tool name, malformed JSON args, validation failure,
                # and anything raised by the tool body that subclasses this
                # type. The single `except` below converts the exception to
                # the tool-role message the LLM reads next iteration and flags
                # function_failed so the meta-loop re-prompts.
                #
                # Anything else (DB blip, provider 503, AttributeError, code
                # bug) is intentionally NOT caught here — it propagates out of
                # `_handle_ai_response` -> `step()` so `process_with_policy`
                # sees the typed exception and classifies it. Without this
                # rule, real bugs get turned into friendly strings that make
                # the source appear "complete" (silent data loss).
                try:
                    # Failure case 1: function name is wrong (not in agent_state.tools)
                    target_mirix_tool = None
                    for t in self.agent_state.tools:
                        if t.name == function_name:
                            target_mirix_tool = t

                    if not target_mirix_tool:
                        raise CorrectableToolError(f"No function named {function_name}")

                    # Failure case 2: function name is OK, but function args are bad JSON
                    try:
                        raw_function_args = function_call.arguments
                        function_args = parse_json(raw_function_args)
                    except Exception as parse_exc:
                        raise CorrectableToolError(
                            f"Error parsing JSON for function '{function_name}' "
                            f"arguments: {function_call.arguments}"
                        ) from parse_exc
                    # parse_json may coerce malformed input (e.g. "{invalid")
                    # into the wrong shape (a list) without raising. Treat
                    # anything not-a-dict as LLM-correctable so the LLM can
                    # re-emit the call rather than crashing downstream.
                    if not isinstance(function_args, dict):
                        raise CorrectableToolError(
                            f"Function '{function_name}' arguments did not "
                            f"parse to a JSON object: {function_call.arguments}"
                        )

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
                            f"[Mirix.Agent.{self.agent_state.name}] INFO: Inner thoughts (from function call) emitted: "
                            f"len={len(response_message.content or '')}"
                        )

                    continue_chaining = True

                    # Fill in omitted prompt-mandated constant fields (e.g.
                    # actor, event_type, source) before validation, so a
                    # missing key degrades to its default instead of either
                    # failing validation or crashing later via bare item[...]
                    # access in memory_tools.py (ECMS-534).
                    normalize_tool_args(function_name, function_args)

                    # Failure case 3: function arguments fail validation
                    validation_error = validate_tool_args(function_name, function_args)
                    if validation_error:
                        raise CorrectableToolError(f"Validation Error: {validation_error}")

                    # Failure case 4: function body raises CorrectableToolError
                    # (or any non-correctable exception, which propagates).
                    # NOTE: the msg_obj associated with the "Running " message
                    # is the prior assistant message, not the function/tool
                    # role message — the function/tool role message is only
                    # created once the function/tool has executed/returned.
                    self.interface.function_message(f"Running {function_name}()", msg_obj=messages[-1])

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

                except CorrectableToolError as e:
                    # function_args might not be a dict (parse_json can coerce
                    # malformed input to a list before we shape-check it).
                    if isinstance(function_args, dict):
                        function_args.pop("self", None)
                    error_msg = get_friendly_error_msg(
                        function_name=function_name,
                        exception_name=type(e).__name__,
                        exception_message=str(e),
                    )
                    printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: {error_msg}\n{traceback.format_exc()}")
                    function_response = package_function_response(False, error_msg)
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
                        )
                    )  # extend conversation with function response
                    self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
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
                    f"LLM returned empty response, no tool_calls and no content. Response: {response_message}"
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
            # Content rides through the masked Langfuse trace; log shape only.
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: Inner thoughts (no function call) emitted: "
                f"len={len(response_message.content or '')}"
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
                "AgentType.chat_agent is deprecated and no longer supported. Use a memory agent type instead."
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
                any_scopes=self._save_scopes,
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
                raise ValueError(f"input_messages items must be Message or MessageCreate, got {type(m)}")

        from mirix.observability.timed import timedspan

        async with timedspan(
            "Agent Step",
            metadata={"agent_type": str(self.agent_state.agent_type)},
        ):
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
                async with timedspan(
                    "Load Retained History",
                    metadata={"agent_id": retention_agent_id, "limit": retention},
                ) as rec:
                    retained_input_sets = await self.message_manager.get_messages_for_agent_user(
                        agent_id=retention_agent_id,
                        user_id=self.user_id,
                        actor=self.actor,
                        limit=retention,
                    )
                    rec["span_output"] = {"loaded_count": len(retained_input_sets)}

            logger.info(
                "[RETENTION] agent=%s retention=%d should_read=%s loaded=%d",
                self.agent_state.id,
                retention,
                should_read_retention,
                len(retained_input_sets),
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

            # Persist memory source and messages before sub-agent dispatch
            # If memory_source_id is set on this agent instance, persist the source
            # and its messages before running any memory extraction. This is gated on
            # the meta_memory_agent type so sub-agents don't re-persist.
            summary_task: Optional[asyncio.Task] = None
            if self.agent_state.is_type(AgentType.meta_memory_agent) and self.memory_source_id:
                # Skip already-processed sources (redelivery)
                # Note: this catches scenarios where the same source (same id) is being retried.
                # It does not catch scenarios where the same source with a different id was queued as a separate message/
                # It also ONLY short circuits if the matched result has been marked as complete.
                async with timedspan(
                    "Check Source Processing State",
                    metadata={"memory_source_id": self.memory_source_id},
                ) as rec:
                    source = await self.memory_source_manager.get_by_id(self.memory_source_id)
                    # The decision this step reached: does the source row exist,
                    # and is it already marked complete (-> idempotency skip)?
                    rec["span_output"] = {
                        "found": source is not None,
                        "processing_complete": bool(source and source.processing_complete),
                    }
                if source and source.processing_complete:
                    logger.info("Source %s already processed, skipping", self.memory_source_id)
                    emit_idempotency_skip_span(
                        name="Idempotency Skip: processing complete",
                        reason="processing-complete",
                        metadata={"memory_source_id": self.memory_source_id},
                    )
                    return MirixUsageStatistics(step_count=0)

                # Test-only fault injection (inert in prod). Resolved BEFORE
                # persist so a directive scoped to the source_messages write (in
                # persist, incl. the conflict path) can fire. Only registers;
                # idempotent; no-op when disabled.
                # Placed after the check above so a persist-memory-source
                # fault directive is not resolved before the persist actually
                # runs.
                fault_injection.resolve_directives(self.memory_source_id, getattr(self, "source_metadata", None))

                # Thread-level message dedup (ECMS-513): for incremental threads,
                # filter out messages already processed in prior saves so the LLM
                # only extracts from genuinely new turns. Engaged only when:
                # - external_thread_id is set (incremental thread)
                # - no explicit external_id (save-once sources keep all-or-nothing)
                # On Kafka retry (source exists), exclude this source's own
                # persisted messages from the "seen" set so the filter correctly
                # returns the same delta as the first attempt.
                if (
                    getattr(self, "external_thread_id", None)
                    and not getattr(self, "external_id", None)
                    and getattr(self, "source_messages", None)
                ):
                    from mirix.services.source_message_manager import filter_new_messages
                    from mirix.utils import flatten_messages_for_agent

                    incoming_count = len(self.source_messages)
                    async with timedspan(
                        "Thread Message Dedup",
                        metadata={
                            "memory_source_id": self.memory_source_id,
                            "external_thread_id": self.external_thread_id,
                            "incoming_count": incoming_count,
                        },
                    ) as dedup_rec:
                        seen_ext_ids, seen_hashes = await self.source_message_manager.get_seen_keys_for_thread(
                            external_thread_id=self.external_thread_id,
                            exclude_source_id=self.memory_source_id if source is not None else None,
                        )
                        new_msgs = filter_new_messages(self.source_messages, seen_ext_ids, seen_hashes)
                        dedup_rec["span_output"] = {
                            "seen_ext_ids": len(seen_ext_ids),
                            "seen_hashes": len(seen_hashes),
                            "new_count": len(new_msgs),
                            "filtered_count": incoming_count - len(new_msgs),
                        }

                    if not new_msgs:
                        logger.info(
                            "All %d messages already seen for thread %s, skipping",
                            incoming_count,
                            self.external_thread_id,
                        )
                        emit_idempotency_skip_span(
                            name="Idempotency Skip: thread messages all seen",
                            reason="thread-message-dedup-all-seen",
                            metadata={
                                "memory_source_id": self.memory_source_id,
                                "external_thread_id": self.external_thread_id,
                                "incoming_count": incoming_count,
                            },
                        )
                        return MirixUsageStatistics(step_count=0)

                    # Keep self.source_messages intact for _persist_memory_source
                    # (stores ALL messages under this source for full provenance;
                    # bulk_insert ON CONFLICT handles per-source uniqueness).
                    # Only re-pack the LLM input to the filtered set.
                    raw_input_messages = flatten_messages_for_agent(new_msgs)
                    normalized_input_messages = []
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
                    input_messages_for_persistence = list(normalized_input_messages)

                # Persist the memory source and its messages before we process it.
                async with timedspan(
                    "Persist Memory Source",
                    metadata={"memory_source_id": self.memory_source_id},
                ) as rec:
                    # In a race condition scenario (kafka redelivery while still processing),
                    # this relies on DB unique constraints to avoid writing duplicate rows.
                    # Note: should_continue will be False in a scenario where a source with the
                    # same externalId or batchHash, but a different primary Id, was found in the
                    # database. In this scenario, we short circuit (even if the row has not been
                    # marked as complete, because it indicates that the source is handled by a different
                    # worker process (different kafka message)
                    should_continue = await self._persist_memory_source(
                        memory_source_id=self.memory_source_id,
                        input_messages=raw_input_messages,
                    )
                    # persisted=False means the content is owned by a different
                    # submission (deduped-elsewhere) and processing short-circuits.
                    rec["span_output"] = {
                        "persisted": should_continue,
                        "message_count": len(raw_input_messages),
                    }
                if not should_continue:
                    logger.info("Source %s deduped, skipping agent processing", self.memory_source_id)
                    emit_idempotency_skip_span(
                        name="Idempotency Skip: source deduped",
                        reason="source-deduped",
                        metadata={"memory_source_id": self.memory_source_id},
                    )
                    return MirixUsageStatistics(step_count=0)

                # Dispatch summary generation in parallel with the memory sub-agents.
                # Awaited before mark_processing_complete; on failure the exception
                # propagates out of step() so the Kafka worker redelivers the message.
                # Source-level idempotency (external_id / batch_hash + processing_complete)
                # makes redelivery a safe full retry.
                # Skipped on direct-write requests because the caller provides the
                # per-item summary directly, so there's nothing to generate.
                if self.summarize and not self.source_summary and not self.direct_writes:
                    summary_task = asyncio.create_task(self._generate_source_summary_traced())

            # Direct-write branch: bypass LLM dispatch and call registered handlers.
            # Placed AFTER _persist_memory_source + dedup/processing_complete checks so
            # deduped or already-processed sources short-circuit before this runs.
            if self.agent_state.is_type(AgentType.meta_memory_agent) and self.direct_writes:
                await self._apply_direct_writes_traced()
                # step() does not finalize. On clean return, dispatch_save
                # calls finalize_source(SUCCESS) in the post-policy handler.
                # If _apply_direct_writes_traced raises, the exception
                # propagates and dispatch_save records the appropriate
                # failure outcome.
                return MirixUsageStatistics(step_count=0)

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
                    # Don't interpolate `e` — its __str__ may carry user
                    # content from the LLM/parsing error. Type alone is
                    # enough at info level.
                    printv(
                        f"[Mirix.Agent.{self.agent_state.name}] INFO: Error in extracting the topic "
                        f"from the input: error_type={type(e).__name__}"
                    )

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

                # The meta-memory agent's kickoff instruction lives in its leading
                # system prompt (prompts/system/*/meta_memory_agent.txt), not as a
                # trailing user turn. Keeping instructions colocated in the single
                # leading system message preserves a clean trust boundary: everything
                # after the system prompt is untrusted user/retrieved content.
                loop_iteration_messages = list(loop_input_messages)

                async with timedspan("Inner Step", metadata={"step_count": step_count}) as rec:
                    step_response = await self.inner_step(
                        messages=loop_iteration_messages,
                        accumulated=accumulated,
                        chaining=chaining,
                        llm_client=llm_client,
                        retained_count=len(retained_input_sets),
                        **kwargs,
                    )
                    # What this iteration decided: keep chaining? did a tool fail?
                    rec["span_output"] = {
                        "continue_chaining": step_response.continue_chaining,
                        "function_failed": step_response.function_failed,
                    }

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
                        warning_content = "You have reached the maximum chaining steps. Please call 'send_message' to send your response to the user."
                    else:
                        warning_content = "You have reached the maximum chaining steps. Please call 'finish_memory_update' to end the chaining."
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
                logger.info(
                    "[RETENTION] agent=%s wrote=%d kept_newest=%d",
                    self.agent_state.id,
                    len(input_messages_for_persistence),
                    retention,
                )

            # Await the parallel summary task (dispatched before sub-agents ran).
            # Raises on failure so the worker redelivers the message — processing_complete
            # stays False and the retry gets a clean full reprocess.
            if summary_task is not None:
                try:
                    await summary_task
                except Exception as e:
                    # The summary task runs CONCURRENTLY with the sub-agent
                    # gather via asyncio.create_task. Its exceptions surface
                    # here only at await time. Log the full cause chain so
                    # wrapped exceptions (e.g. asyncpg / MissingGreenlet
                    # signatures) stay visible.
                    try:
                        from mirix.queue.error_policy import format_exc_chain

                        chain = format_exc_chain(e)
                    except Exception:
                        chain = "<chain-format-failed>"
                    logger.error(
                        "summary_task failed: source=%s exc_type=%s — chain: %s",
                        self.memory_source_id,
                        type(e).__name__,
                        chain,
                        exc_info=True,
                    )
                    raise

            # Test-only fault injection (inert in prod): a `llm` directive forces
            # the malformed-tool-calls-past-budget path deterministically,
            # without having to make a real LLM emit broken output. The shape
            # llm_chaining_exhausted raises LLMChainingExhaustedError, taking the
            # exact same propagation path as the natural raise below.
            if self.agent_state.is_type(AgentType.meta_memory_agent):
                fault_injection.maybe_raise("llm", source_key=getattr(self, "memory_source_id", None))

            # If the LLM loop exited with `function_failed=True`, the meta-agent
            # LLM emitted a malformed tool call (CorrectableToolError-shape) on
            # the last iteration AND the bounded re-prompt didn't recover.
            # Raise a typed exception so process_with_policy classifies this as
            # PERMANENT (no value in retrying — the LLM already had its
            # budget) and the post-policy handler records the right outcome.
            #
            # Non-correctable failures (DB / provider / code bug) never reach
            # this point: they raised out of `inner_step` earlier.
            #
            # step() does NOT finalize on the success path either — that's
            # `dispatch_save`'s job after process_with_policy returns SUCCESS.
            if self.agent_state.is_type(AgentType.meta_memory_agent) and function_failed:
                raise LLMChainingExhaustedError(
                    f"meta-agent LLM produced malformed tool calls past chaining budget "
                    f"(counter={counter}, max_chaining_steps={max_chaining_steps})"
                )

            return MirixUsageStatistics(**total_usage.model_dump(), step_count=step_count)

    async def _apply_direct_write(self, memory_type: str, payload: Dict[str, Any]) -> None:
        """Dispatch a direct write to the registered handler for memory_type.

        Handlers write the memory row via the appropriate manager and call
        the shared _write_citation helper. No LLM involvement.
        """
        from mirix.functions.direct_write_handlers import DIRECT_WRITE_HANDLERS

        handler = DIRECT_WRITE_HANDLERS.get(memory_type)
        if handler is None:
            raise ValueError(f"No direct-write handler registered for memory_type: {memory_type}")
        await handler(self, **payload)

    async def _apply_direct_writes_traced(self) -> None:
        """Apply all self.direct_writes under Langfuse so the path matches the
        meta-agent tree: ``Direct Writes`` (span) → per-type insert span → embeddings.

        Uses ``as_type="span"`` (not nested ``agent``) so observations stay under
        the queue worker's ``Meta Agent`` span. Each insert span becomes the
        ``observation_id`` in trace context while its handler runs so embedding
        observations parent correctly.

        Restores trace context to the Direct Writes span only *after* each insert
        observation context manager exits, so Langfuse can close child spans
        before the parent id is switched back (avoids broken / empty subtrees).
        """
        parent_trace_context = get_trace_context()
        langfuse = get_langfuse_client()
        trace_id = parent_trace_context.get("trace_id") if parent_trace_context else None
        parent_span_id = parent_trace_context.get("observation_id") if parent_trace_context else None

        async def _run_all_untraced():
            for write in self.direct_writes:
                await self._apply_direct_write(write["memory_type"], write["payload"])

        if not (langfuse and trace_id):
            await _run_all_untraced()
            return

        from typing import cast

        from langfuse.types import TraceContext

        from mirix.functions.direct_write_handlers import DIRECT_WRITE_HANDLERS
        from mirix.observability.context import set_trace_context

        trace_context_dict: dict = {"trace_id": trace_id}
        if parent_span_id:
            trace_context_dict["parent_span_id"] = parent_span_id

        direct_writes_io = {
            "memory_source_id": self.memory_source_id,
            "agent_name": self.agent_state.name,
            "direct_write_count": len(self.direct_writes),
            "memory_types": [w["memory_type"] for w in self.direct_writes],
        }
        with langfuse.start_as_current_observation(
            name="Direct Writes",
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            # Metadata mirror as input: counts + type enums only (payloads are
            # caller-authored content and never reach the span).
            input=direct_writes_io,
            metadata=_tid_stamped(direct_writes_io),
        ) as span:
            mark_observation_as_child(span)
            span_observation_id = getattr(span, "id", None)
            if span_observation_id:
                set_trace_context(
                    trace_id=trace_id,
                    observation_id=span_observation_id,
                    user_id=parent_trace_context.get("user_id"),
                    session_id=parent_trace_context.get("session_id"),
                )

            for write in self.direct_writes:
                memory_type = write["memory_type"]
                payload = write["payload"]
                handler = DIRECT_WRITE_HANDLERS.get(memory_type)
                function_name = handler.__name__ if handler else str(memory_type)
                insert_span_name = f"insert_{memory_type}_memory"

                trace_input: dict = {
                    "memory_type": memory_type,
                    "function": function_name,
                    "payload_keys": list(payload.keys()),
                }
                items = payload.get("items")
                if isinstance(items, list):
                    trace_input["items_count"] = len(items)

                if not span_observation_id:
                    await self._apply_direct_write(memory_type, payload)
                    continue

                insert_trace_dict: dict = {"trace_id": trace_id, "parent_span_id": span_observation_id}
                try:
                    insert_observation_cm = langfuse.start_as_current_observation(
                        name=insert_span_name,
                        as_type="span",
                        trace_context=cast(TraceContext, insert_trace_dict),
                        input=trace_input,
                        metadata=_tid_stamped(
                            {
                                "memory_source_id": self.memory_source_id,
                                "memory_type": memory_type,
                                "function": function_name,
                            }
                        ),
                    )
                except Exception as e:
                    logger.debug(
                        "Langfuse direct_write insert observation failed: %s; running handler without insert span",
                        e,
                    )
                    await self._apply_direct_write(memory_type, payload)
                    continue

                try:
                    with insert_observation_cm as insert_span:
                        mark_observation_as_child(insert_span)
                        insert_observation_id = getattr(insert_span, "id", None)
                        if insert_observation_id:
                            set_trace_context(
                                trace_id=trace_id,
                                observation_id=insert_observation_id,
                                user_id=parent_trace_context.get("user_id"),
                                session_id=parent_trace_context.get("session_id"),
                            )
                        await self._apply_direct_write(memory_type, payload)
                        try:
                            insert_span.update(
                                output={"status": "completed"},
                                metadata={"memory_type": memory_type},
                            )
                        except Exception as e:
                            logger.debug("Langfuse direct_write insert span update failed: %s", e)
                finally:
                    if span_observation_id:
                        set_trace_context(
                            trace_id=trace_id,
                            observation_id=span_observation_id,
                            user_id=parent_trace_context.get("user_id"),
                            session_id=parent_trace_context.get("session_id"),
                        )

            try:
                span.update(output={"status": "completed"})
            except Exception as e:
                logger.debug("Langfuse Direct Writes span output update failed: %s", e)

    async def _persist_memory_source(
        self,
        memory_source_id: str,
        input_messages: list,
    ) -> bool:
        """Persist a MemorySource and its SourceMessages at the start of meta-agent processing.

        Returns True if processing should continue, False if this submission
        should be skipped (the content is owned by a different submission — a
        true duplicate, or one another worker is finishing). A same-id conflict
        (retry / redelivery of this submission) returns True and resumes.

        Uses INSERT ON CONFLICT DO NOTHING for idempotent redelivery. Called only
        by meta_memory_agent before sub-agent dispatch. Computes batch_hash and
        auto-derives external_id so duplicate submissions hit the partial unique
        indexes on memory_sources.
        """
        from mirix.services.source_message_manager import (
            compute_batch_hash,
            derive_external_id_from_message_ids,
            normalize_message,
        )

        try:
            # Prefer source_messages (original per-turn dicts with role, external_message_id,
            # occurred_at intact) over the packed input_messages which lost per-message
            # fields when the add_memory handler flattened turns into [USER]/[ASSISTANT]
            # markers. Falls back to input_messages for callers that pass memory_source_id
            # without source_messages (nobody does today, but the two params aren't coupled).
            #
            # Direct-write jobs: REST enqueues messages=[] which becomes a single placeholder
            # MessageCreate with empty content — not a real conversation. Persist the
            # memory_sources row only; source_messages stay empty (provenance is payload +
            # citations). See memory-sources architecture (direct-write flow).
            if self.direct_writes:
                messages_for_persistence = []
            elif self.source_messages:
                messages_for_persistence = self.source_messages
            else:
                messages_for_persistence = input_messages

            # Normalize messages once — reused for hash computation and persistence
            msg_dicts = [normalize_message(msg) for msg in messages_for_persistence] if messages_for_persistence else []

            # Compute dedup keys for source-level idempotency
            external_id = self.external_id
            batch_hash = None

            if not external_id and msg_dicts:
                # Auto-derive external_id if all messages have external_message_ids
                ext_msg_ids = [m["external_message_id"] for m in msg_dicts if m.get("external_message_id")]
                if len(ext_msg_ids) == len(msg_dicts):
                    external_id = derive_external_id_from_message_ids(ext_msg_ids)

            if not external_id and msg_dicts:
                # Fallback: compute batch_hash for content-based dedup
                batch_hash = compute_batch_hash(
                    external_thread_id=self.external_thread_id,
                    occurred_at=self.occurred_at,
                    messages=msg_dicts,
                )

            source = await self.memory_source_manager.create(
                memory_source_id=memory_source_id,
                actor=self.actor,
                user_id=self.user_id,
                organization_id=self.agent_state.organization_id,
                source_type=self.source_type or "conversation",
                external_id=external_id,
                external_thread_id=self.external_thread_id,
                source_system=self.source_system,
                source_metadata=self.source_metadata,
                occurred_at=self.occurred_at,
                summary=self.source_summary,
                summary_source=self.source_summary_source,
                batch_hash=batch_hash,
                filter_tags=self.filter_tags,
            )

            if source is None:
                # Content owned by a different submission/ kafaka message (true duplicate, or one
                # another worker is finishing) — skip without persisting messages.
                return False

            if msg_dicts:
                await self.source_message_manager.bulk_insert(
                    messages=msg_dicts,
                    memory_source_id=memory_source_id,
                    external_thread_id=self.external_thread_id,
                    fallback_occurred_at=self.occurred_at,
                    created_by_id=self.actor.id,
                )

            logger.info(
                "Persisted memory source %s with %d source messages",
                memory_source_id,
                len(msg_dicts),
            )
            return True
        except Exception as e:
            logger.error("Failed to persist memory source %s: %s", memory_source_id, e)
            raise

    async def _generate_source_summary_traced(self) -> None:
        """Wrap _generate_source_summary in a LangFuse child span.

        Captures the trace context at dispatch time so the span lands as a
        sibling of the memory sub-agent spans under the meta_memory_agent trace,
        not as an orphan created by the background task.
        """
        parent_trace_context = get_trace_context()
        langfuse = get_langfuse_client()
        trace_id = parent_trace_context.get("trace_id") if parent_trace_context else None
        parent_span_id = parent_trace_context.get("observation_id") if parent_trace_context else None

        if not (langfuse and trace_id):
            await self._generate_source_summary()
            return

        from typing import cast

        from langfuse.types import TraceContext

        from mirix.observability.context import set_trace_context

        trace_context_dict: dict = {"trace_id": trace_id}
        if parent_span_id:
            trace_context_dict["parent_span_id"] = parent_span_id

        summary_agent_io = {
            "memory_source_id": self.memory_source_id,
            "agent_name": self.agent_state.name,
        }
        with langfuse.start_as_current_observation(
            name="Summary Agent",
            as_type="agent",
            trace_context=cast(TraceContext, trace_context_dict),
            # Metadata mirror as input — ids only; the transcript this agent
            # reads is conversation content and never reaches the span.
            input=summary_agent_io,
            metadata=_tid_stamped(summary_agent_io),
        ) as span:
            mark_observation_as_child(span)
            span_observation_id = getattr(span, "id", None)
            if span_observation_id:
                set_trace_context(
                    trace_id=trace_id,
                    observation_id=span_observation_id,
                    user_id=parent_trace_context.get("user_id"),
                    session_id=parent_trace_context.get("session_id"),
                )
            summary_text = await self._generate_source_summary()
            try:
                # Shape only (chars count) — the summary text is LLM output
                # and stays inside the masked generation pathway.
                span.update(
                    output={
                        "summary_generated": bool(summary_text),
                        "summary_chars": len(summary_text) if summary_text else 0,
                    }
                )
            except Exception as e:
                logger.debug("Failed to set Summary Agent span output: %s", e)

    async def _generate_source_summary(self) -> Optional[str]:
        """Generate a summary for the memory source using the agent's LLM.

        Retrieves source messages, formats them into a prompt, and calls the LLM.
        The generated summary is written to memory_sources.summary with
        summary_source="generated".

        Returns the generated summary text (or None when skipped/empty) — used
        by the traced wrapper for span-shape reporting only (chars, never
        content). Raises on failure — caller is responsible for error handling.
        """
        from mirix.prompts.gpt_summarize_source_messages import SYSTEM as SUMMARY_PROMPT_SYSTEM
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.mirix_message_content import TextContent
        from mirix.utils import count_tokens

        # Retrieve source messages
        result = await self.source_message_manager.get_messages_by_source_id(
            memory_source_id=self.memory_source_id,
            limit=2000,
        )
        if not result.items:
            logger.warning("No source messages found for %s, skipping summary", self.memory_source_id)
            return

        # Format messages into a conversation transcript
        transcript_lines = []
        for msg in result.items:
            content = msg.content
            if isinstance(content, dict):
                text = content.get("text", "") or content.get("content", "")
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            transcript_lines.append(f"{msg.role}: {text}")
        transcript = "\n\n".join(transcript_lines)

        # Truncate to fit within ~90% of the context window
        from mirix.observability.cpu_timing import time_cpu

        context_window = self.agent_state.llm_config.context_window
        max_input_tokens = int(context_window * 0.9)
        with time_cpu("count_tokens.transcript"):
            transcript_tokens = count_tokens(transcript)
        if transcript_tokens > max_input_tokens:
            ratio = max_input_tokens / transcript_tokens * 0.8
            keep = max(1, int(len(result.items) * ratio))
            transcript = "\n\n".join(transcript_lines[-keep:])

        llm_messages = [
            Message(
                agent_id=self.agent_state.id,
                role=MessageRole.system,
                content=[TextContent(text=SUMMARY_PROMPT_SYSTEM)],
            ),
            Message(
                agent_id=self.agent_state.id,
                role=MessageRole.user,
                content=[TextContent(text=transcript)],
            ),
        ]

        from mirix.llm_api.llm_api_tools import retry_with_exponential_backoff
        from mirix.settings import settings

        llm_client = LLMClient.create(
            llm_config=self.agent_state.llm_config.model_copy(deep=True),
        )

        async def _send_request():
            return await llm_client.send_llm_request(messages=llm_messages)

        send_with_retry = retry_with_exponential_backoff(
            _send_request,
            initial_delay=settings.llm_retry_backoff_factor,
            max_retries=settings.llm_retry_limit,
            error_codes=(429, 500, 502, 503, 504),
        )
        response = await send_with_retry()
        summary_text = response.choices[0].message.content

        if summary_text:
            await self.memory_source_manager.update_summary(
                memory_source_id=self.memory_source_id,
                summary=summary_text,
                summary_source="generated",
            )
            logger.info("Generated summary for memory source %s", self.memory_source_id)
        else:
            logger.warning("LLM returned empty summary for source %s", self.memory_source_id)
        # Returned ONLY so the Summary Agent span can report shape (chars),
        # never content — the summary text itself is LLM output (sensitive).
        return summary_text

    async def _fetch_recent_indexing_lag_window(
        self,
        table: str,
        pydantic_cls: Any,
    ) -> List[Any]:
        """Fetch rows written in the recent indexing-lag window for an owning sub-agent's dedup decision.

        Queries the Relational DB provider for rows whose ``created_at`` or
        ``updated_at`` falls within ``HYBRID_READ_WINDOW_SECONDS``. These are
        the just-written candidates the Search provider may not have indexed
        yet. The ranked ("relevant") bucket is obtained separately by the
        prompt builder via the manager ``list_*`` call (which preserves
        ``@update_timezone`` and the manager's post-processing); the caller
        unions these recent rows into that list. Returns the recent rows as
        Pydantic instances, capped at ``MAX_RETRIEVAL_LIMIT_IN_SYSTEM``.

        Returns ``[]`` when either provider is unregistered (PG-only path):
        in that mode reads come from the canonical SQL store, so the
        indexing-lag distinction does not exist. Fail-closed: a raising
        Relational call propagates.

        Scoped to ``self._save_scopes`` so this save's recent-window read
        never surfaces the user's rows from another scope (see ECMS-58).
        """
        from mirix.database.relational_provider import get_relational_provider
        from mirix.database.search_provider import get_search_provider

        sp = get_search_provider()
        rp = get_relational_provider()
        if not (sp and rp):
            return []

        from mirix.observability.timed import timedspan

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=HYBRID_READ_WINDOW_SECONDS)
        # Child span so the IPS-R recent-window leg shows up distinctly under the
        # parent "Retrieve <type>" span, separate from the IPS-S "IPS Search" leg.
        async with timedspan("Recent window fetch", metadata={"backend": "ipsr", "table": table}) as rec:
            recent_records = await rp.list(
                table,
                user_id=self.user.id,
                organization_id=self.user.organization_id,
                scopes=self._save_scopes,
                time_range={
                    "updated_at__gte": cutoff.isoformat(),
                    "created_at__gte": cutoff.isoformat(),
                },
                time_range_or_null_updated=True,
                limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
            )
            rec["span_output"] = {"records_count": len(recent_records)}
        return [pydantic_cls(**r) for r in recent_records]

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
        from mirix.observability.timed import timedspan
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

        # Each memory type is retrieved by its own coroutine below; the six are
        # then run concurrently via asyncio.gather. The blocks
        # write disjoint keys in ``retrieved_memories`` and read only the
        # inputs computed once above (key_words, embedded_text, timezone_str,
        # search_method), so there is no inter-block data dependency. Each
        # coroutine keeps its own per-type gate, timedspan, and recent+relevant
        # merge, and computes its own owning-agent flag locally (no shared
        # mutable owning-flag variable across the concurrent blocks).

        async def _retrieve_core():
            if self.agent_state.is_type(AgentType.core_memory_agent) or "core" not in retrieved_memories:
                async with timedspan("Retrieve core", metadata={"backend": "ipsr", "memory_type": "core"}) as rec:
                    # Scope the core memory fed into the prompt to the current
                    # save's scope. Without any_scopes this returns the user's
                    # blocks across ALL scopes, leaking another scope's core
                    # memory into this scope's LLM context (and thus into the
                    # block written here). See VEPAGE-1474.
                    blocks_result = await self.block_manager.get_blocks(
                        user=self.user,
                        any_scopes=self._save_scopes,
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
                    rec["span_output"] = {"block_count": len(current_persisted_memory.blocks)}

        async def _retrieve_knowledge_vault():
            is_owning_kv_agent = self.agent_state.is_type(
                AgentType.knowledge_vault_memory_agent, AgentType.reflexion_agent
            )
            if (
                self.agent_state.is_type(AgentType.knowledge_vault_memory_agent)
                or "knowledge_vault" not in retrieved_memories
            ):
                async with timedspan(
                    "Retrieve knowledge_vault",
                    metadata={"backend": "ipss+ipsr", "memory_type": "knowledge_vault"},
                ) as rec:
                    current_knowledge_vault = await self.knowledge_vault_manager.list_knowledge(
                        agent_state=self.agent_state,
                        user=self.user,
                        embedded_text=embedded_text,
                        query=key_words,
                        search_field="caption",
                        search_method=search_method,
                        limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                        timezone_str=timezone_str,
                        sensitivity=None if is_owning_kv_agent else ["low", "medium"],
                        scopes=self._save_scopes,
                    )
                    recent_knowledge_vault = await self._fetch_recent_indexing_lag_window(
                        table="knowledge_vault",
                        pydantic_cls=PydanticKnowledgeVaultItem,
                    )
                    merged_knowledge_vault = self._merge_recent_into_relevant(
                        current_knowledge_vault, recent_knowledge_vault
                    )

                    knowledge_vault_memory = ""
                    if len(merged_knowledge_vault) > 0:
                        for idx, knowledge_vault_item in enumerate(merged_knowledge_vault):
                            knowledge_vault_memory += f"[{idx}] Knowledge Vault Item ID: {knowledge_vault_item.id}; Caption: {knowledge_vault_item.caption}\n"
                    retrieved_memories["knowledge_vault"] = {
                        "total_number_of_items": await self.knowledge_vault_manager.get_total_number_of_items(
                            user=self.user, scopes=self._save_scopes,
                        ),
                        "current_count": len(merged_knowledge_vault),
                        "text": knowledge_vault_memory.strip(),
                    }
                    rec["span_output"] = {
                        "merged_count": len(merged_knowledge_vault),
                        "total_items": retrieved_memories["knowledge_vault"]["total_number_of_items"],
                    }

        async def _retrieve_episodic():
            is_owning_agent = self.agent_state.is_type(AgentType.episodic_memory_agent, AgentType.reflexion_agent)
            if is_owning_agent or "episodic" not in retrieved_memories:
                async with timedspan(
                    "Retrieve episodic",
                    metadata={"backend": "ipss+ipsr", "memory_type": "episodic"},
                ) as rec:
                    current_episodic_memory = await self.episodic_memory_manager.list_episodic_memory(
                        agent_state=self.agent_state,
                        user=self.user,
                        limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                        timezone_str=timezone_str,
                        scopes=self._save_scopes,
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
                        scopes=self._save_scopes,
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
                        "total_number_of_items": await self.episodic_memory_manager.get_total_number_of_items(
                            user=self.user, scopes=self._save_scopes,
                        ),
                        "recent_count": len(current_episodic_memory),
                        "relevant_count": len(most_relevant_episodic_memory),
                        "recent_episodic_memory": recent_episodic_memory,
                        "relevant_episodic_memory": relevant_episodic_memory,
                    }
                    rec["span_output"] = {
                        "recent_count": len(current_episodic_memory),
                        "relevant_count": len(most_relevant_episodic_memory),
                        "total_items": retrieved_memories["episodic"]["total_number_of_items"],
                    }

        async def _retrieve_resource():
            # Owning agents need IDs for merge/update operations, so always retrieve fresh
            is_owning_agent = self.agent_state.is_type(AgentType.resource_memory_agent, AgentType.reflexion_agent)
            if is_owning_agent or "resource" not in retrieved_memories:
                async with timedspan(
                    "Retrieve resource",
                    metadata={"backend": "ipss+ipsr", "memory_type": "resource"},
                ) as rec:
                    current_resource_memory = await self.resource_memory_manager.list_resources(
                        agent_state=self.agent_state,
                        user=self.user,
                        query=key_words,
                        embedded_text=embedded_text,
                        search_field="summary",
                        search_method=search_method,
                        limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                        timezone_str=timezone_str,
                        scopes=self._save_scopes,
                    )
                    recent_resource_memory_items = await self._fetch_recent_indexing_lag_window(
                        table="resource_memory",
                        pydantic_cls=PydanticResourceMemoryItem,
                    )
                    merged_resource_memory = self._merge_recent_into_relevant(
                        current_resource_memory, recent_resource_memory_items
                    )
                    resource_memory = ""
                    if len(merged_resource_memory) > 0:
                        for idx, resource in enumerate(merged_resource_memory):
                            if is_owning_agent:
                                resource_memory += f"[Resource ID: {resource.id}] Resource Title: {resource.title}; Resource Summary: {resource.summary} Resource Type: {resource.resource_type}\n"
                            else:
                                resource_memory += f"[{idx}] Resource Title: {resource.title}; Resource Summary: {resource.summary} Resource Type: {resource.resource_type}\n"
                    resource_memory = resource_memory.strip()
                    retrieved_memories["resource"] = {
                        "total_number_of_items": await self.resource_memory_manager.get_total_number_of_items(
                            user=self.user, scopes=self._save_scopes,
                        ),
                        "current_count": len(merged_resource_memory),
                        "text": resource_memory,
                    }
                    rec["span_output"] = {
                        "merged_count": len(merged_resource_memory),
                        "total_items": retrieved_memories["resource"]["total_number_of_items"],
                    }

        async def _retrieve_procedural():
            # Owning agents need IDs for merge/update operations, so always retrieve fresh
            is_owning_agent = self.agent_state.is_type(AgentType.procedural_memory_agent, AgentType.reflexion_agent)
            if is_owning_agent or "procedural" not in retrieved_memories:
                async with timedspan(
                    "Retrieve procedural",
                    metadata={"backend": "ipss+ipsr", "memory_type": "procedural"},
                ) as rec:
                    current_procedural_memory = await self.procedural_memory_manager.list_procedures(
                        agent_state=self.agent_state,
                        user=self.user,
                        query=key_words,
                        embedded_text=embedded_text,
                        search_field="summary",
                        search_method=search_method,
                        limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                        timezone_str=timezone_str,
                        scopes=self._save_scopes,
                    )
                    recent_procedural_memory_items = await self._fetch_recent_indexing_lag_window(
                        table="procedural_memory",
                        pydantic_cls=PydanticProceduralMemoryItem,
                    )
                    merged_procedural_memory = self._merge_recent_into_relevant(
                        current_procedural_memory, recent_procedural_memory_items
                    )
                    procedural_memory = ""
                    if len(merged_procedural_memory) > 0:
                        for idx, procedure in enumerate(merged_procedural_memory):
                            if is_owning_agent:
                                procedural_memory += f"[Procedure ID: {procedure.id}] Entry Type: {procedure.entry_type}; Summary: {procedure.summary}\n"
                            else:
                                procedural_memory += (
                                    f"[{idx}] Entry Type: {procedure.entry_type}; Summary: {procedure.summary}\n"
                                )
                    procedural_memory = procedural_memory.strip()
                    retrieved_memories["procedural"] = {
                        "total_number_of_items": await self.procedural_memory_manager.get_total_number_of_items(
                            user=self.user, scopes=self._save_scopes,
                        ),
                        "current_count": len(merged_procedural_memory),
                        "text": procedural_memory,
                    }
                    rec["span_output"] = {
                        "merged_count": len(merged_procedural_memory),
                        "total_items": retrieved_memories["procedural"]["total_number_of_items"],
                    }

        async def _retrieve_semantic():
            # Owning agents need IDs for merge/update operations, so always retrieve fresh
            is_owning_agent = self.agent_state.is_type(AgentType.semantic_memory_agent, AgentType.reflexion_agent)
            if is_owning_agent or "semantic" not in retrieved_memories:
                async with timedspan(
                    "Retrieve semantic",
                    metadata={"backend": "ipss+ipsr", "memory_type": "semantic"},
                ) as rec:
                    current_semantic_memory = await self.semantic_memory_manager.list_semantic_items(
                        agent_state=self.agent_state,
                        user=self.user,
                        query=key_words,
                        embedded_text=embedded_text,
                        search_field="details",
                        search_method=search_method,
                        limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                        timezone_str=timezone_str,
                        scopes=self._save_scopes,
                    )
                    recent_semantic_memory_items = await self._fetch_recent_indexing_lag_window(
                        table="semantic_memory",
                        pydantic_cls=PydanticSemanticMemoryItem,
                    )
                    merged_semantic_memory = self._merge_recent_into_relevant(
                        current_semantic_memory, recent_semantic_memory_items
                    )
                    semantic_memory = ""
                    if len(merged_semantic_memory) > 0:
                        for idx, semantic_memory_item in enumerate(merged_semantic_memory):
                            if is_owning_agent:
                                semantic_memory += f"[Semantic Memory ID: {semantic_memory_item.id}] Name: {semantic_memory_item.name}; Summary: {semantic_memory_item.summary}\n"
                            else:
                                semantic_memory += f"[{idx}] Name: {semantic_memory_item.name}; Summary: {semantic_memory_item.summary}\n"

                    semantic_memory = semantic_memory.strip()
                    retrieved_memories["semantic"] = {
                        "total_number_of_items": await self.semantic_memory_manager.get_total_number_of_items(
                            user=self.user, scopes=self._save_scopes,
                        ),
                        "current_count": len(merged_semantic_memory),
                        "text": semantic_memory,
                    }
                    rec["span_output"] = {
                        "merged_count": len(merged_semantic_memory),
                        "total_items": retrieved_memories["semantic"]["total_number_of_items"],
                    }

        # Run the six retrievals concurrently. gather (return_exceptions=False)
        # propagates the first failure rather than silently swallowing it.
        await asyncio.gather(
            _retrieve_core(),
            _retrieve_knowledge_vault(),
            _retrieve_episodic(),
            _retrieve_resource(),
            _retrieve_procedural(),
            _retrieve_semantic(),
        )

        # Build the complete system prompt (synchronous string assembly of all
        # retrieved memories into the prompt template -- a CPU block on the loop).
        from mirix.observability.cpu_timing import time_cpu

        with time_cpu("build_system_prompt.assembly"):
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

        knowledge_vault_total = knowledge_vault["total_number_of_items"] if knowledge_vault else 0
        knowledge_vault_text = knowledge_vault["text"] if knowledge_vault else ""
        knowledge_vault_count = knowledge_vault["current_count"] if knowledge_vault else 0
        system_prompt += (
            f"\n<knowledge_vault> ({knowledge_vault_count} out of {knowledge_vault_total} Items):\n"
            + (knowledge_vault_text if knowledge_vault_text else "Empty")
            + "\n</knowledge_vault>\n"
        )

        semantic_total = semantic_memory["total_number_of_items"] if semantic_memory else 0
        semantic_text = semantic_memory["text"] if semantic_memory else ""
        semantic_count = semantic_memory["current_count"] if semantic_memory else 0
        system_prompt += (
            f"\n<semantic_memory> ({semantic_count} out of {semantic_total} Items):\n"
            + (semantic_text if semantic_text else "Empty")
            + "\n</semantic_memory>\n"
        )

        resource_total = resource_memory["total_number_of_items"] if resource_memory else 0
        resource_text = resource_memory["text"] if resource_memory else ""
        resource_count = resource_memory["current_count"] if resource_memory else 0
        system_prompt += (
            f"\n<resource_memory> ({resource_count} out of {resource_total} Items):\n"
            + (resource_text if resource_text else "Empty")
            + "\n</resource_memory>\n"
        )

        procedural_total = procedural_memory["total_number_of_items"] if procedural_memory else 0
        procedural_text = procedural_memory["text"] if procedural_memory else ""
        procedural_count = procedural_memory["current_count"] if procedural_memory else 0
        system_prompt += (
            f"\n<procedural_memory> ({procedural_count} out of {procedural_total} Items):\n"
            + (procedural_text if procedural_text else "Empty")
            + "\n</procedural_memory>"
        )

        return system_prompt

    @staticmethod
    def _merge_recent_into_relevant(relevant: List[Any], recent: List[Any]) -> List[Any]:
        """Union just-written ``recent`` rows into the ranked ``relevant`` list.

        Preserves the Search-provider ranking of ``relevant`` and appends only
        the ``recent`` rows whose ``id`` is not already present (de-dup by id).
        The appended rows are the just-written candidates the Search provider
        may not have indexed yet, so the dedup-deciding owning sub-agent sees
        them alongside the ranked results in a single list.
        """
        relevant_ids = {item.id for item in relevant}
        appended = [item for item in recent if item.id not in relevant_ids]
        return relevant + appended

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

        The returned string is consumed downstream as a ``;``-joined
        list of search topics — each segment becomes its own clause in
        the OpenSearch query. The IPS Search backend caps text-field
        values at 200 characters, so individual topics must stay under
        that limit; the prompt and ``update_topic`` tool description
        below set that expectation with the LLM, and
        ``_enforce_topic_length`` truncates anything that slips
        through with a logged warning so the prompt can be tuned.
        """
        try:
            # Add instruction message for topic extraction
            temporary_messages = copy.deepcopy(messages)
            temporary_messages.append(
                prepare_input_message_create(
                    MessageCreate(
                        role=MessageRole.user,
                        content=(
                            "The above are the inputs from the user, please "
                            "look at these content and extract the topic "
                            "(brief description of what the user is focusing "
                            "on) from these content. If there are multiple "
                            "focuses in these content, then extract them all "
                            "and put them into one string separated by ';'. "
                            "Each individual topic must be a short noun "
                            "phrase (NOT a full sentence) and STRICTLY UNDER "
                            "200 characters — preferably much shorter. "
                            "Topics longer than 200 characters will be "
                            "rejected. Call the function `update_topic` to "
                            "update the topic with the extracted topics."
                        ),
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
                                "description": (
                                    "The topic of the current conversation/"
                                    "content. If there are multiple topics "
                                    'then separate them with ";". Each '
                                    "individual topic (each ;-delimited "
                                    "segment) MUST be a short noun phrase "
                                    "strictly under 200 characters. "
                                    "Downstream search will reject longer "
                                    "values."
                                ),
                            }
                        },
                        "required": ["topic"],
                    },
                }
            ]

            # Resolve the llm_config to use for topic extraction: the client's
            # independently configurable topic_extraction_agent row, falling back
            # to this agent's own config (meta_memory_agent's) unconditionally on
            # a miss -- preserving today's exact save-path fallback identity
            # (R3.3), not the shared helper's own all_agents[0] default.
            from mirix.services.agent_manager import get_or_create_topic_extraction_agent

            all_agents = await self.agent_manager.list_agents(actor=self.actor, include_tools=False)

            if all_agents:
                extraction_llm_config = await get_or_create_topic_extraction_agent(
                    agent_manager=self.agent_manager,
                    actor=self.actor,
                    all_agents=all_agents,
                    fallback_llm_config=self.agent_state.llm_config,
                )
            else:
                # No agents at all for this client (shouldn't happen in practice —
                # this Agent instance itself is one of the client's agents — but
                # fail safe exactly like today: use this agent's own config,
                # unconditionally).
                extraction_llm_config = self.agent_state.llm_config

            # Use LLMClient to extract topics (run async in event loop from sync context)
            llm_client = LLMClient.create(
                llm_config=extraction_llm_config,
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
                    llm_config=extraction_llm_config,
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
                        # Guard against the LLM ignoring the per-topic
                        # length constraint in the prompt — truncate
                        # any individual ;-delimited segment that
                        # would exceed the downstream IPS Search
                        # field limit. Failing closed here would abort
                        # the entire memory-extraction pipeline, so we
                        # truncate and warn instead.
                        topics = self._enforce_topic_length(topics)
                        # Topics derive from user messages; log shape only.
                        printv(
                            f"[Mirix.Agent.{self.agent_state.name}] INFO: Extracted topics: "
                            f"count={len(topics) if isinstance(topics, list) else 0}"
                        )
                        return topics
                    except (json.JSONDecodeError, KeyError) as parse_error:
                        printv(
                            f"[Mirix.Agent.{self.agent_state.name}] WARNING: Failed to parse topic "
                            f"extraction response: error_type={type(parse_error).__name__}"
                        )
                        continue

        except Exception as e:
            # Don't interpolate `e` — LLM error path may carry user content.
            printv(
                f"[Mirix.Agent.{self.agent_state.name}] INFO: Error in extracting the topic "
                f"from the messages: error_type={type(e).__name__}"
            )

        return None

    # Maximum length, in characters, of a single ``;``-delimited topic
    # forwarded to memory search. Set to the IPS Search backend's
    # 200-character text-field cap (values over this yield a 400 from
    # IPS Search that would otherwise abort the whole inner_step).
    # Topics flow into ``query=`` on the four memory-list calls in
    # ``build_system_prompt_with_memories``.
    _TOPIC_MAX_LEN = 200

    def _enforce_topic_length(self, topics: Optional[str]) -> Optional[str]:
        """Truncate individual ``;``-delimited topics that exceed
        ``_TOPIC_MAX_LEN``. Returns the (possibly modified) joined
        string. None / empty inputs pass through unchanged.

        The downstream search backend rejects text-field values over
        the limit. We truncate rather than reject the whole string:
        the memory-extraction pipeline depends on this value being
        present, and failing closed reproduces the exact crash this
        guard is here to prevent. A warning is logged per truncation
        so the upstream extraction prompt can be tuned over time.
        """
        if not topics:
            return topics
        parts = [p.strip() for p in topics.split(";")]
        out: List[str] = []
        for p in parts:
            if not p:
                continue
            if len(p) > self._TOPIC_MAX_LEN:
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] WARNING: "
                    f"truncating oversized topic "
                    f"({len(p)} chars > {self._TOPIC_MAX_LEN}); "
                    f"the topic-extraction prompt should keep this "
                    f"under the limit: {p[:80]!r}..."
                )
                p = p[: self._TOPIC_MAX_LEN]
            out.append(p)
        return ";".join(out) if out else None

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
                # Topics derive from user messages; log shape only.
                printv(
                    f"[Mirix.Agent.{self.agent_state.name}] INFO: Step topics: "
                    f"count={len(topics) if isinstance(topics, list) else 0}"
                )

            # Step 0: build the system message on-the-fly from agent_state.system + memories
            raw_system = self.agent_state.system or ""

            # Build the complete system prompt with memories
            from mirix.observability.timed import timedspan

            async with timedspan(
                "Build System Prompt With Memories",
                metadata={
                    "agent_type": str(self.agent_state.agent_type),
                    "step_count": step_count,
                },
            ) as build_prompt_rec:
                complete_system_prompt, retrieved_memories = await self.build_system_prompt_with_memories(
                    raw_system=raw_system,
                    topics=topics,
                    retrieved_memories=retrieved_memories,
                )
                # Counts only (the retrieved text is memory content). Core is a
                # compiled string here — its item count lives on the "Retrieve
                # core" child span; this reports the countable dict entries.
                _counts_by_type = {
                    k: v["current_count"] if "current_count" in v else v.get("recent_count", 0)
                    for k, v in retrieved_memories.items()
                    if isinstance(v, dict) and ("current_count" in v or "recent_count" in v)
                }
                build_prompt_rec["span_output"] = {
                    "counts_by_memory_type": _counts_by_type,
                    "prompt_chars": len(complete_system_prompt),
                }

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

            # Test-only fault injection (inert in prod): a `llm_request` directive
            # raises a synthetic "maximum context length" error here — inside the
            # try whose except runs summarize-and-retry recovery — so the
            # context-overflow recovery path can be exercised deterministically
            # without a real >context-window payload. AgentType is imported
            # locally (matching this module's pattern; inner_step has no scope-
            # level import) — omitting it raised NameError on every save.
            from mirix.schemas.agent import AgentType

            if self.agent_state.is_type(AgentType.meta_memory_agent):
                fault_injection.maybe_raise("llm_request", source_key=getattr(self, "memory_source_id", None))

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
                    # A memory function genuinely failed — log at ERROR (not the
                    # prior printv->INFO, which hid real failures from error-level
                    # alerting/grep and only *said* "ERROR" in the message text).
                    self.logger.error("One or more functions failed:\n" + "\n".join(failed_messages))
                else:
                    # Fallback if we can't parse the messages
                    self.logger.error("Function execution encountered errors (see logs above for details)")

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
            # str(e) for an LLMError wraps the upstream provider's 4xx
            # response which can include the user's prompt. Redact via
            # ispy-pii so the error REASON ("prompt is too long: …",
            # "content_policy_violation: …") stays debuggable in Splunk
            # with PII tokens scrubbed.
            #
            # `messages` is typed Union[Message, List[Message]]; normalize
            # so the error logger itself doesn't raise (Message is a
            # Pydantic BaseModel, so .get() is unavailable, and len()
            # rejects scalars).
            from mirix.pii import log_error_strip_pii

            msgs_list = messages if isinstance(messages, list) else [messages]
            await log_error_strip_pii(
                logger,
                f"[Mirix.Agent.{self.agent_state.name}] inner_step() failed: " "num_messages=%d message_roles=%s",
                len(msgs_list),
                [getattr(m, "role", None) for m in msgs_list],
                exc=e,
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
                        printv(f"[Mirix.Agent.{self.agent_state.name}] ERROR: Summarization failed: {summarize_err}")
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
                # Redact str(e) via ispy-pii so the unrecognized error
                # message is preserved (with PII scrubbed) for debugging.
                from mirix.pii import log_error_strip_pii

                await log_error_strip_pii(
                    logger,
                    f"[Mirix.Agent.{self.agent_state.name}] inner_step() failed with " "an unrecognized exception:",
                    exc=e,
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
