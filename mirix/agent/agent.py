import re
import os
import copy
import json
import time
import pytz
import base64
import shutil
import traceback
import warnings
import requests
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Callable

from mirix.constants import (
    CLI_WARNING_PREFIX,
    ERROR_MESSAGE_PREFIX,
    FIRST_MESSAGE_ATTEMPTS,
    FUNC_FAILED_HEARTBEAT_MESSAGE,
    MIRIX_CORE_TOOL_MODULE_NAME,
    MIRIX_MEMORY_TOOL_MODULE_NAME,
    MIRIX_EXTRA_TOOL_MODULE_NAME,
    LLM_MAX_TOKENS,
    REQ_HEARTBEAT_MESSAGE,
    CLEAR_HISTORY_AFTER_MEMORY_UPDATE,
    CHAINING_FOR_MEMORY_UPDATE,
    MAX_EMBEDDING_DIM,
    MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
    MAX_CHAINING_STEPS
)
import logging
from mirix import LLMConfig
from mirix.errors import ContextWindowExceededError, LLMError
from mirix.functions.ast_parsers import coerce_dict_args_by_annotations, get_function_annotations_from_source
from mirix.functions.functions import get_function_from_module
from mirix.helpers import ToolRulesSolver
from mirix.helpers.message_helpers import prepare_input_message_create
from mirix.interface import AgentInterface
from mirix.llm_api.helpers import calculate_summarizer_cutoff, get_token_counts_for_messages, is_context_overflow_error
from mirix.llm_api.llm_api_tools import create
from mirix.utils import num_tokens_from_functions, num_tokens_from_messages
from mirix.memory import summarize_messages
from mirix.orm import User
from mirix.orm.enums import ToolType
from mirix.schemas.agent import AgentState, AgentStepResponse, UpdateAgent
from mirix.schemas.block import BlockUpdate
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import MessageRole
from mirix.schemas.memory import ContextWindowOverview, Memory
from mirix.schemas.message import Message, MessageCreate
from mirix.schemas.openai.chat_completion_request import Tool as ChatCompletionRequestTool
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse
from mirix.schemas.openai.chat_completion_response import Message as ChatCompletionMessage
from mirix.schemas.openai.chat_completion_response import UsageStatistics
from mirix.schemas.tool import Tool
from mirix.schemas.tool_rule import TerminalToolRule
from mirix.schemas.usage import MirixUsageStatistics
from mirix.schemas.mirix_message_content import TextContent, ImageContent, FileContent, CloudFileContent
from mirix.services.agent_manager import AgentManager
from mirix.services.block_manager import BlockManager
from mirix.services.helpers.agent_manager_helper import check_supports_structured_output, compile_memory_metadata_block
from mirix.services.message_manager import MessageManager
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.resource_memory_manager import ResourceMemoryManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.services.step_manager import StepManager
from mirix.services.user_manager import UserManager
from mirix.services.tool_execution_sandbox import ToolExecutionSandbox
from mirix.settings import summarizer_settings
from mirix.embeddings import embedding_model
from mirix.system import get_contine_chaining, get_token_limit_warning, package_function_response, package_summarize_message, package_user_message
from mirix.tracing import log_event, trace_method
from mirix.llm_api.llm_client import LLMClient
from mirix.utils import (
    count_tokens,
    get_friendly_error_msg,
    get_tool_call_id,
    get_utc_time,
    json_dumps,
    json_loads,
    parse_json,
    printd,
    validate_function_response,
    convert_timezone_to_utc,
    log_telemetry,
)
from mirix.services.file_manager import FileManager


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
        user: User,
        # extras
        first_message_verify_mono: bool = True,  # TODO move to config?
    ):
        assert isinstance(agent_state.memory, Memory), f"Memory object is not of type Memory: {type(agent_state.memory)}"
        # Hold a copy of the state that was used to init the agent
        self.agent_state = agent_state
        assert isinstance(self.agent_state.memory, Memory), f"Memory object is not of type Memory: {type(self.agent_state.memory)}"

        self.user = user

        # Initialize logger early in constructor
        self.logger = logging.getLogger(f"Mirix.Agent.{agent_state.name}")
        self.logger.setLevel(logging.INFO)

        # initialize a tool rules solver
        if agent_state.tool_rules:
            # if there are tool rules, log a warning
            for rule in agent_state.tool_rules:
                if not isinstance(rule, TerminalToolRule):
                    self.logger.warning("Tool rules only work reliably for the latest OpenAI models that support structured outputs.")
                    break
        # add default rule for having send_message be a terminal tool
        if agent_state.tool_rules is None:
            agent_state.tool_rules = []

        self.tool_rules_solver = ToolRulesSolver(tool_rules=agent_state.tool_rules)

        # gpt-4, gpt-3.5-turbo, ...
        self.model = self.agent_state.llm_config.model
        self.supports_structured_output = check_supports_structured_output(model=self.model, tool_rules=agent_state.tool_rules)

        # state managers
        self.block_manager = BlockManager()

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
        self.user_manager = UserManager()

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

        # Load last function response from message history
        self.last_function_response = self.load_last_function_response()

        # Logger that the Agent specifically can use, will also report the agent_state ID with the logs
        # Note: Logger is already initialized earlier in constructor

    def load_last_function_response(self):
        """Load the last function response from message history"""
        in_context_messages = self.agent_manager.get_in_context_messages(agent_id=self.agent_state.id, actor=self.user)
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

    def update_topic_if_changed(self, topic: str) -> bool:
        """
        Update the agent's topic if it has changed.

        Args:
            topic (str): the new topic

        Returns:
            modified (bool): whether the topic was updated
        """
        if self.agent_state.topic != topic:
            self.agent_manager.update_topic(
                agent_id=self.agent_state.id,
                topic=topic,
                actor=self.user,
            )
            self.agent_state.topic = topic
            return True
        return False

    def update_memory_if_changed(self, new_memory: Memory) -> bool:
        """
        Update internal memory object and system prompt if there have been modifications.

        Args:
            new_memory (Memory): the new memory object to compare to the current memory object

        Returns:
            modified (bool): whether the memory was updated
        """
        if self.agent_state.memory.compile() != new_memory.compile():
            # update the blocks (LRW) in the DB
            for label in self.agent_state.memory.list_block_labels():
                updated_value = new_memory.get_block(label).value
                if updated_value != self.agent_state.memory.get_block(label).value:
                    # update the block if it's changed
                    block_id = self.agent_state.memory.get_block(label).id
                    block = self.block_manager.update_block(
                        block_id=block_id, block_update=BlockUpdate(value=updated_value), actor=self.user
                    )

            # refresh memory from DB (using block ids)
            self.agent_state.memory = Memory(
                blocks=[self.block_manager.get_block_by_id(block.id, actor=self.user) for block in self.agent_state.memory.get_blocks()]
            )

            # NOTE: don't do this since re-buildin the memory is handled at the start of the step
            # rebuild memory - this records the last edited timestamp of the memory
            # TODO: pass in update timestamp from block edit time
            # self.agent_state = self.agent_manager.rebuild_system_prompt(agent_id=self.agent_state.id, actor=self.user)
            return True
        
        return False

    def execute_tool_and_persist_state(self, function_name: str, function_args: dict, target_mirix_tool: Tool, 
                                       display_intermediate_message: Optional[Callable] = None) -> str:
        """
        Execute tool modifications and persist the state of the agent.
        Note: only some agent state modifications will be persisted, such as data in the AgentState ORM and block data
        """
        # TODO: add agent manager here
        orig_memory_str = self.agent_state.memory.compile()

        # TODO: need to have an AgentState object that actually has full access to the block data
        # this is because the sandbox tools need to be able to access block.value to edit this data
        try:

            if function_name in ['episodic_memory_insert', 'episodic_memory_replace', 'list_memory_within_timerange']:
                key = "items" if function_name == 'episodic_memory_insert' else 'new_items'
                if key in function_args:
                    # Need to change the timezone into UTC timezone
                    for item in function_args[key]:
                        if 'occurred_at' in item:
                            item['occurred_at'] = convert_timezone_to_utc(item['occurred_at'], self.user_manager.get_user_by_id(self.user.id).timezone)

            if function_name in ['search_in_memory', 'list_memory_within_timerange']:
                function_args['timezone_str'] = self.user_manager.get_user_by_id(self.user.id).timezone

            if target_mirix_tool.tool_type == ToolType.MIRIX_CORE:
                # base tools are allowed to access the `Agent` object and run on the database
                callable_func = get_function_from_module(MIRIX_CORE_TOOL_MODULE_NAME, function_name)
                function_args["self"] = self  # need to attach self to arg since it's dynamically linked
                if function_name in ['send_message', 'send_intermediate_message']:
                    agent_state_copy = self.agent_state.__deepcopy__()
                    function_args["agent_state"] = agent_state_copy  # need to attach self to arg since it's dynamically linked
                function_response = callable_func(**function_args)
                if function_name in ['send_message', 'send_intermediate_message']:
                    self.update_topic_if_changed(agent_state_copy.topic)
                if function_name == 'send_intermediate_message':
                    # send intermediate message to the user
                    if display_intermediate_message:
                        display_intermediate_message("response", function_args['message'])
            
            elif target_mirix_tool.tool_type == ToolType.MIRIX_MEMORY_CORE:
                callable_func = get_function_from_module(MIRIX_MEMORY_TOOL_MODULE_NAME, function_name)
                if function_name in ['core_memory_append', 'core_memory_rewrite']:
                    agent_state_copy = self.agent_state.__deepcopy__()
                    function_args["agent_state"] = agent_state_copy  # need to attach self to arg since it's dynamically linked
                if function_name in ['check_episodic_memory', 'check_semantic_memory']:
                    function_args['timezone_str'] = self.user_manager.get_user_by_id(self.user.id).timezone
                function_args["self"] = self
                function_response = callable_func(**function_args)
                if function_name in ['core_memory_append', 'core_memory_rewrite']:
                    self.update_memory_if_changed(agent_state_copy.memory)

            elif target_mirix_tool.tool_type == ToolType.MIRIX_EXTRA:
                callable_func = get_function_from_module(MIRIX_EXTRA_TOOL_MODULE_NAME, function_name)
                function_args["self"] = self  # need to attach self to arg since it's dynamically linked
                function_response = callable_func(**function_args)

            else:
                raise ValueError(f"Tool type {target_mirix_tool.tool_type} not supported")

        except Exception as e:
            # Need to catch error here, or else trunction wont happen
            # TODO: modify to function execution error
            function_response = get_friendly_error_msg(
                function_name=function_name, exception_name=type(e).__name__, exception_message=str(e)
            )

        return function_response

    @trace_method
    def _get_ai_reply(
        self,
        message_sequence: List[Message],
        function_call: Optional[str] = None,
        first_message: bool = False,
        stream: bool = False,  # TODO move to config?
        empty_response_retry_limit: int = 3,
        backoff_factor: float = 0.5,  # delay multiplier for exponential backoff
        max_delay: float = 10.0,  # max delay between retries
        step_count: Optional[int] = None,
        last_function_failed: bool = False,
        put_inner_thoughts_first: bool = True,
        get_input_data_for_debugging: bool = False,
        existing_file_uris: Optional[List[str]] = None,
        second_try: bool = False,
    ) -> ChatCompletionResponse:
        """Get response from LLM API with robust retry mechanism."""
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

        # Don't allow a tool to be called if it failed last time
        if last_function_failed and self.tool_rules_solver.tool_call_history:
            allowed_functions = [f for f in allowed_functions if f["name"] != self.tool_rules_solver.tool_call_history[-1]]
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

        for attempt in range(1, empty_response_retry_limit + 1):
            try:
                log_telemetry(self.logger, "_get_ai_reply create start")

                # New LLM client flow
                llm_client = LLMClient.create(
                    llm_config=self.agent_state.llm_config,
                    put_inner_thoughts_first=put_inner_thoughts_first,
                )

                if llm_client and not stream:
                    response = llm_client.send_llm_request(
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
                    response = create(
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
                        put_inner_thoughts_first=put_inner_thoughts_first,
                        name=self.agent_state.name,
                    )
                log_telemetry(self.logger, "_get_ai_reply create finish")

                # These bottom two are retryable
                if len(response.choices) == 0 or response.choices[0] is None:
                    raise ValueError(f"API call returned an empty message: {response}")

                for choice in response.choices:
                    if choice.message.content == '' and len(choice.message.tool_calls) == 0:
                        raise ValueError(f"API call returned an empty message: {response}")

                if response.choices[0].finish_reason not in ["stop", "function_call", "tool_calls"]:
                    if response.choices[0].finish_reason == "length":
                        if attempt >= empty_response_retry_limit:
                            raise RuntimeError("Retries exhausted and no valid response received. Final error: maximum context length exceeded or generated content is too long")
                        else:
                            delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                            self.logger.warning(f"Attempt {attempt} failed: {response.choices[0].finish_reason}. Retrying in {delay} seconds...")
                            time.sleep(delay)
                            continue
                    else:
                        raise ValueError(f"Bad finish reason from API: {response.choices[0].finish_reason}")
                log_telemetry(self.logger, "_handle_ai_response finish")

            except ValueError as ve:
                if attempt >= empty_response_retry_limit:
                    self.logger.error(f"Retry limit reached. Final error: {ve}")
                    log_telemetry(self.logger, "_handle_ai_response finish ValueError")
                    raise Exception(f"Retries exhausted and no valid response received. Final error: {ve}")
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    self.logger.warning(f"Attempt {attempt} failed: {ve}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

            except KeyError as ke:
                # Gemini api sometimes can yield empty response
                # This is a retryable error
                if attempt >= empty_response_retry_limit:
                    self.logger.error(f"Retry limit reached. Final error: {ke}")
                    log_telemetry(self.logger, "_handle_ai_response finish KeyError")
                    raise Exception(f"Retries exhausted and no valid response received. Final error: {ke}")
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    self.logger.warning(f"Attempt {attempt} failed: {ke}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

            except LLMError as llm_error:
                if attempt >= empty_response_retry_limit:
                    self.logger.error(f"Retry limit reached. Final error: {llm_error}")
                    log_telemetry(self.logger, "_handle_ai_response finish LLMError")
                    log_telemetry(self.logger, "_get_ai_reply_last_message_hacking start")
                    if second_try:
                        raise Exception(f"Retries exhausted and no valid response received. Final error: {llm_error}")
                    return self._get_ai_reply([message_sequence[-1]], function_call, first_message, stream, empty_response_retry_limit, backoff_factor, max_delay, step_count, last_function_failed, put_inner_thoughts_first, get_input_data_for_debugging, second_try=True)
                
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    self.logger.warning(f"Attempt {attempt} failed: {llm_error}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
            
            except AssertionError as ae:
                if attempt >= empty_response_retry_limit:
                    self.logger.error(f"Retry limit reached. Final error: {ae}")
                    raise Exception(f"Retries exhausted and no valid response received. Final error: {ae}")
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    self.logger.warning(f"Attempt {attempt} failed: {ae}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

            except requests.exceptions.HTTPError as he:
                if attempt >= empty_response_retry_limit:
                    self.logger.error(f"Retry limit reached. Final error: {he}")
                    raise Exception(f"Retries exhausted and no valid response received. Final error: {he}")
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    self.logger.warning(f"Attempt {attempt} failed: {he}. Retrying in {delay} seconds...")
                    time.sleep(delay)

            except Exception as e:
                log_telemetry(self.logger, "_handle_ai_response finish generic Exception")
                # For non-retryable errors, exit immediately
                log_telemetry(self.logger, "_handle_ai_response finish generic Exception")
                raise e

            # check if we are going over the context window: this allows for articifial constraints
            if response.usage.total_tokens > self.agent_state.llm_config.context_window:
                # trigger summarization
                log_telemetry(self.logger, "_get_ai_reply summarize_messages_inplace")
                self.summarize_messages_inplace()

            # return the response
            return response

        log_telemetry(self.logger, "_handle_ai_response finish catch-all exception")
        raise Exception("Retries exhausted and no valid response received.")

    def _handle_ai_response(
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
        message_added = False

        # Step 2: check if LLM wanted to call a function
        if response_message.function_call or (response_message.tool_calls is not None and len(response_message.tool_calls) > 0):
            if response_message.function_call:
                raise DeprecationWarning(response_message)
            
            assert response_message.tool_calls is not None and len(response_message.tool_calls) > 0

            # Generate UUIDs for tool calls if needed
            if override_tool_call_id or response_message.function_call:
                self.logger.warning("Overriding the tool call can result in inconsistent tool call IDs during streaming")
                for tool_call in response_message.tool_calls:
                    tool_call.id = get_tool_call_id()  # needs to be a string for JSON
            else:
                for tool_call in response_message.tool_calls:
                    assert tool_call.id is not None  # should be defined

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
                # Flag to avoid printing a duplicate if inner thoughts get popped from the function call
                nonnull_content = True

            # Step 3: Process each tool call
            continue_chaining = True
            overall_function_failed = False
            any_message_added = False
            executed_function_names = []  # Track which functions were executed
            
            self.logger.info(f"Processing {len(response_message.tool_calls)} tool call(s)")
            
            for tool_call_idx, tool_call in enumerate(response_message.tool_calls):
                tool_call_id = tool_call.id
                function_call = tool_call.function
                function_name = function_call.name
                
                self.logger.info(f"Processing tool call {tool_call_idx + 1}/{len(response_message.tool_calls)}: {function_name} with tool_call_id: {tool_call_id}")

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
                    error_msg = f"Error parsing JSON for function '{function_name}' arguments: {function_call.arguments}"
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

                if function_name == 'trigger_memory_update':
                    function_args["user_message"] = {'message': convert_message_to_input_message(input_message), 
                                                     'existing_file_uris': existing_file_uris,
                                                     'retrieved_memories': retrieved_memories,
                                                     'chaining': CHAINING_FOR_MEMORY_UPDATE}
                    if message_queue is not None:
                        function_args["user_message"]['message_queue'] = message_queue
                
                elif function_name == 'trigger_memory_update_with_instruction':
                    function_args["user_message"] = {'existing_file_uris': existing_file_uris,
                                                     'retrieved_memories': retrieved_memories}

                # Check if inner thoughts is in the function call arguments (possible apparently if you are using Azure)
                if "inner_thoughts" in function_args:
                    response_message.content = function_args.pop("inner_thoughts")
                # The content if then internal monologue, not chat
                if response_message.content and not nonnull_content:
                    self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])

                continue_chaining = True

                # Failure case 5: function failed during execution
                # NOTE: the msg_obj associated with the "Running " message is the prior assistant message, not the function/tool role message
                #       this is because the function/tool role message is only created once the function/tool has executed/returned
                self.interface.function_message(f"Running {function_name}()", msg_obj=messages[-1])

                try:
                    if display_intermediate_message:
                        # send intermediate message to the user
                        display_intermediate_message("internal_monologue", response_message.content)

                    function_response = self.execute_tool_and_persist_state(function_name, function_args, 
                                                                            target_mirix_tool, 
                                                                            display_intermediate_message=display_intermediate_message)
                    
                    if function_name == 'send_message' or function_name == 'finish_memory_update':
                        assert tool_call_idx == len(response_message.tool_calls) - 1, f"{function_name} must be the last tool call"

                    if tool_call_idx == len(response_message.tool_calls) - 1:
                        if function_name == 'send_message':
                            continue_chaining = False
                        elif function_name == 'finish_memory_update':
                            continue_chaining = False
                        else:
                            continue_chaining = True

                    # handle trunction
                    if function_name in ["conversation_search", "conversation_search_date", "archival_memory_search"]:
                        # with certain functions we rely on the paging mechanism to handle overflow
                        truncate = False
                    else:
                        # but by default, we add a truncation safeguard to prevent bad functions from
                        # overflow the agent context window
                        truncate = True

                    # get the function response limit
                    return_char_limit = target_mirix_tool.return_char_limit
                    function_response_string = validate_function_response(
                        function_response, return_char_limit=return_char_limit, truncate=truncate
                    )

                    function_args.pop("self", None)
                    function_response = package_function_response(True, function_response_string)
                    function_failed = False
                
                except Exception as e:
                    function_args.pop("self", None)
                    # error_msg = f"Error calling function {function_name} with args {function_args}: {str(e)}"
                    # Less detailed - don't provide full args, idea is that it should be in recent context so no need (just adds noise)
                    error_msg = get_friendly_error_msg(function_name=function_name, exception_name=type(e).__name__, exception_message=str(e))
                    error_msg_user = f"{error_msg}\n{traceback.format_exc()}"
                    self.logger.error(error_msg_user)
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

            # Handle context message clearing only if ALL functions succeeded
            if not overall_function_failed:
                # Check if any executed function should trigger history clearing
                should_clear_history = False
                for func_name in executed_function_names:
                    if CLEAR_HISTORY_AFTER_MEMORY_UPDATE:
                        if self.agent_state.name == 'reflexion_agent':
                            if func_name == 'finish_memory_update':
                                should_clear_history = True
                                break
                        elif self.agent_state.name == 'meta_memory_agent' and (func_name == 'finish_memory_update' or not chaining):
                            should_clear_history = True
                            break
                        elif self.agent_state.name not in ['meta_memory_agent', 'chat_agent'] and (func_name == 'finish_memory_update' or not chaining):
                            should_clear_history = True
                            break
                
                if should_clear_history:

                    # It means one of the following conditions is met:
                    # (1) meta_memory_agent, finish_memory_update -> continue_chaining = False
                    # (2) non-meta_memory_agent, finish_memory_update -> continue_chaining = False
                    # (3) non-meta_memory_agent, CHAINING_FOR_MEMORY_UPDATE = False -> continue_chaining = False
                    continue_chaining = False
                    
                    in_context_messages = self.agent_manager.get_in_context_messages(agent_id=self.agent_state.id, actor=self.user)
                    message_ids = [message.id for message in in_context_messages]
                    message_ids = [message_ids[0]]

                    # show the last edited memory item
                    memory_item = None
                    memory_item_str = None

                    if self.agent_state.name == 'episodic_memory_agent':
                        memory_item = self.episodic_memory_manager.get_most_recently_updated_event(timezone_str=self.user_manager.get_user_by_id(self.user.id).timezone)
                        if memory_item:
                            memory_item = memory_item[0]
                            memory_item_str = ''
                            memory_item_str += '[Episodic Event ID]: ' + memory_item.id + '\n'
                            memory_item_str += '[Event Occurred At]: ' + memory_item.occurred_at.strftime('%Y-%m-%d %H:%M:%S') + '\n'
                            memory_item_str += '[Summary]: ' + memory_item.summary + '\n'
                            memory_item_str += '[Details]: ' + memory_item.details + '\n'
                            memory_item_str += '[Tree Path]: ' + (' > '.join(memory_item.tree_path) if memory_item.tree_path else 'N/A') + '\n'
                            memory_item_str += '[Last Modified]: ' + memory_item.last_modify['operation']  + ' at ' + memory_item.last_modify['timestamp'].strftime('%Y-%m-%d %H:%M:%S') + '\n'
                            memory_item_str = memory_item_str.strip()
                            
                    elif self.agent_state.name == 'procedural_memory_agent':
                        memory_item = self.procedural_memory_manager.get_most_recently_updated_item(timezone_str=self.user_manager.get_user_by_id(self.user.id).timezone)
                        if memory_item:
                            memory_item = memory_item[0]
                            memory_item_str = ''
                            memory_item_str += '[Procedural Memory ID]: ' + memory_item.id + '\n'
                            memory_item_str += '[Entry Type]: ' + memory_item.entry_type + '\n'
                            memory_item_str += '[Summary]: ' + (memory_item.summary or 'N/A') + '\n'
                            memory_item_str += '[Steps]: ' + "; ".join(memory_item.steps) + '\n'
                            memory_item_str += '[Tree Path]: ' + (' > '.join(memory_item.tree_path) if memory_item.tree_path else 'N/A') + '\n'
                            memory_item_str += '[Last Modified]: ' + memory_item.last_modify['operation']  + ' at ' + memory_item.last_modify['timestamp'].strftime('%Y-%m-%d %H:%M:%S') + '\n'
                            memory_item_str = memory_item_str.strip()
                            
                    elif self.agent_state.name == 'resource_memory_agent':
                        memory_item = self.resource_memory_manager.get_most_recently_updated_item(timezone_str=self.user_manager.get_user_by_id(self.user.id).timezone)
                        if memory_item:
                            memory_item = memory_item[0]
                            memory_item_str = ''
                            memory_item_str += '[Resource Memory ID]: ' + memory_item.id + '\n'
                            memory_item_str += '[Title]: ' + memory_item.title + '\n'
                            memory_item_str += '[Summary]: ' + (memory_item.summary or 'N/A') + '\n'
                            memory_item_str += '[Resource Type]: ' + memory_item.resource_type + '\n'
                            memory_item_str += '[Content]: ' + memory_item.content + '\n'
                            memory_item_str += '[Tree Path]: ' + (' > '.join(memory_item.tree_path) if memory_item.tree_path else 'N/A') + '\n'
                            memory_item_str += '[Last Modified]: ' + memory_item.last_modify['operation']  + ' at ' + memory_item.last_modify['timestamp'].strftime('%Y-%m-%d %H:%M:%S') + '\n'
                            memory_item_str = memory_item_str.strip()
                            
                    elif self.agent_state.name == 'knowledge_vault_agent':
                        memory_item = self.knowledge_vault_manager.get_most_recently_updated_item(timezone_str=self.user_manager.get_user_by_id(self.user.id).timezone)
                        
                        # Check if finish_memory_update was one of the executed functions
                        if 'finish_memory_update' in executed_function_names and memory_item is None:
                            memory_item_str = "No new knowledge vault items were added."

                        if memory_item:
                            memory_item = memory_item[0]
                            memory_item_str = ''
                            memory_item_str += '[Knowledge Vault ID]: ' + memory_item.id + '\n'
                            memory_item_str += '[Entry Type]: ' + memory_item.entry_type + '\n'
                            memory_item_str += '[Caption]: ' + memory_item.caption + '\n'
                            memory_item_str += '[Source]: ' + memory_item.source + '\n'
                            memory_item_str += '[Sensitivity]: ' + memory_item.sensitivity + '\n'
                            memory_item_str += '[Secret Value]: ' + memory_item.secret_value + '\n'
                            memory_item_str += '[Last Modified]: ' + memory_item.last_modify['operation']  + ' at ' + memory_item.last_modify['timestamp'].strftime('%Y-%m-%d %H:%M:%S') + '\n'
                            memory_item_str = memory_item_str.strip()
                            
                    elif self.agent_state.name == 'semantic_memory_agent':
                        memory_item = self.semantic_memory_manager.get_most_recently_updated_item(timezone_str=self.user_manager.get_user_by_id(self.user.id).timezone)
                        if memory_item:
                            memory_item = memory_item[0]
                            memory_item_str = ''
                            memory_item_str += '[Semantic Memory ID]: ' + memory_item.id + '\n'
                            memory_item_str += '[Name]: ' + memory_item.name + '\n'
                            memory_item_str += '[Summary]: ' + memory_item.summary + '\n'
                            memory_item_str += '[Details]: ' + (memory_item.details or 'N/A') + '\n'
                            memory_item_str += '[Source]: ' + (memory_item.source or 'N/A') + '\n'
                            memory_item_str += '[Tree Path]: ' + (' > '.join(memory_item.tree_path) if memory_item.tree_path else 'N/A') + '\n'
                            memory_item_str += '[Last Modified]: ' + memory_item.last_modify['operation']  + ' at ' + memory_item.last_modify['timestamp'].strftime('%Y-%m-%d %H:%M:%S') + '\n'
                            memory_item_str = memory_item_str.strip()

                    elif self.agent_state.name == 'core_memory_agent':
                        memory_item_str = self.agent_state.memory.compile()

                    # create a new message for this:
                    if memory_item_str:
                        if self.agent_state.name == 'core_memory_agent':
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
                        )
                        
                        # persist the message to the database
                        persisted_message = self.message_manager.create_message(new_message, actor=self.user)
                        
                        # append the persisted message ID to the message list
                        message_ids.append(persisted_message.id)
                        self.agent_manager.set_in_context_messages(agent_id=self.agent_state.id, message_ids=message_ids, actor=self.user)

                        # delete the detached messages
                        deleted_count = self.message_manager.delete_detached_messages_for_agent(agent_id=self.agent_state.id, actor=self.user)

                    if self.agent_state.name == 'meta_memory_agent':
                        self.agent_manager.set_in_context_messages(agent_id=self.agent_state.id, message_ids=message_ids, actor=self.user)
                        deleted_count = self.message_manager.delete_detached_messages_for_agent(agent_id=self.agent_state.id, actor=self.user)

                    if self.agent_state.name == 'reflexion_agent':
                        self.agent_manager.set_in_context_messages(agent_id=self.agent_state.id, message_ids=message_ids, actor=self.user)
                        deleted_count = self.message_manager.delete_detached_messages_for_agent(agent_id=self.agent_state.id, actor=self.user)

                    # Clear all messages since they were manually added to the conversation history
                    messages = []

        else:
            # Standard non-function reply
            messages.append(
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])
            continue_chaining = True
            function_failed = False
            if display_intermediate_message:
                display_intermediate_message("internal_monologue", response_message.content)

        # Update ToolRulesSolver state with last called function
        self.tool_rules_solver.update_tool_usage(function_name)
        # Update contine_chaining request according to provided tool rules
        if self.tool_rules_solver.has_children_tools(function_name):
            continue_chaining = True
        elif self.tool_rules_solver.is_terminal_tool(function_name):
            continue_chaining = False

        return messages, continue_chaining, function_failed

    def step(
        self,
        input_messages: Union[Message, List[Message]],
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        extra_messages: Optional[List[dict]] = None,
        **kwargs,
    ) -> MirixUsageStatistics:
        """Run Agent.step in a loop, handling chaining via contine_chaining requests and function failures"""

        max_chaining_steps = max_chaining_steps or MAX_CHAINING_STEPS

        first_input_message = input_messages[0]

        # Convert MessageCreate objects to Message objects
        message_objects = [prepare_input_message_create(m, self.agent_state.id, wrap_user_message=False, wrap_system_message=True) for m in input_messages]
        
        extra_message_objects = [prepare_input_message_create(m, self.agent_state.id, wrap_user_message=False, wrap_system_message=True) for m in extra_messages] if extra_messages is not None else None
        next_input_message = message_objects
        counter = 0
        total_usage = UsageStatistics()
        step_count = 0

        initial_message_count = len(self.agent_manager.get_in_context_messages(agent_id=self.agent_state.id, actor=self.user))

        if self.agent_state.name == 'reflexion_agent':
            # clear previous messages
            in_context_messages = self.agent_manager.get_in_context_messages(agent_id=self.agent_state.id, actor=self.user)
            in_context_messages = in_context_messages[:1]
            self.agent_manager.set_in_context_messages(agent_id=self.agent_state.id, message_ids=[message.id for message in in_context_messages], actor=self.user)

        while True:

            kwargs["first_message"] = False
            kwargs["step_count"] = step_count

            if self.agent_state.name in ['meta_memory_agent', 'chat_agent'] and step_count == 0:
                # When the agent first gets the screenshots, we need to extract the topic to search the query.

                try:
                    topics = None

                    temporary_messages = copy.deepcopy(next_input_message)

                    temporary_messages.append(prepare_input_message_create(MessageCreate(
                        role=MessageRole.user,
                        content="The above are the inputs from the user, please look at these content and extract the topic (brief description of what the user is focusing on) from these content. If there are multiple focuses in these content, then extract them all and put them into one string separated by ';'. Call the function `update_topic` to update the topic with the extracted topics.",
                    ), self.agent_state.id, wrap_user_message=False, wrap_system_message=True))

                    temporary_messages = [
                        prepare_input_message_create(MessageCreate(
                            role=MessageRole.system,
                            content="You are a helpful assistant that extracts the topic from the user's input.",
                        ), self.agent_state.id, wrap_user_message=False, wrap_system_message=True),
                    ] + temporary_messages
                    
                    # Define the function for topic extraction
                    functions = [{
                        'name': 'update_topic',
                        'description': "Update the topic of the conversation/content. The topic will be used for retrieving relevant information from the database",
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'topic': {
                                    'type': 'string', 
                                    'description': 'The topic of the current conversation/content. If there are multiple topics then separate them with ";".'}
                            },
                            'required': ['topic']
                        },
                    }]
                    
                    # Use LLMClient to extract topics
                    llm_client = LLMClient.create(
                        llm_config=self.agent_state.llm_config,
                        put_inner_thoughts_first=True,
                    )
                    
                    if llm_client:
                        response = llm_client.send_llm_request(
                            messages=temporary_messages,
                            tools=functions,
                            stream=False,
                            force_tool_call='update_topic',
                        )
                    else:
                        # Fallback to existing create function
                        response = create(
                            llm_config=self.agent_state.llm_config,
                            messages=temporary_messages,
                            functions=functions,
                            force_tool_call='update_topic',
                        )

                    # Extract topics from the response
                    for choice in response.choices:
                        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0:
                            try:
                                function_args = json.loads(choice.message.tool_calls[0].function.arguments)
                                topics = function_args.get('topic')
                                break
                            except (json.JSONDecodeError, KeyError) as parse_error:
                                self.logger.warning(f"Failed to parse topic extraction response: {parse_error}")
                                continue
        
                    if topics is not None:
                        kwargs['topics'] = topics
                        self.update_topic_if_changed(topics)
                    else:
                        self.logger.warning("No topics extracted from screenshots")

                except Exception as e:
                    self.logger.info(f"Error in extracting the topic from the screenshots: {e}")
                    pass

            step_response = self.inner_step(
                first_input_messge=first_input_message,
                messages=next_input_message,
                extra_messages=extra_message_objects,
                initial_message_count=initial_message_count,
                chaining=chaining,
                **kwargs,
            )

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
            save_agent(self)

            # Chain stops
            if not chaining and (not function_failed):
                self.logger.info("No chaining, stopping after one step")
                break
            elif max_chaining_steps is not None and counter == max_chaining_steps:
                # Add warning message based on agent type
                if self.agent_state.name == "chat_agent":
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
                )
                continue  # give agent one more chance to respond
            elif max_chaining_steps is not None and counter > max_chaining_steps:
                self.logger.info(f"Hit max chaining steps, stopping after {counter} steps")
                break
            # Chain handlers
            elif token_warning and summarizer_settings.send_memory_warning_message:
                assert self.agent_state.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_token_limit_warning(),
                    },
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
                )
                continue  # always chain
            # Mirix no-op / yield
            else:
                break

        return MirixUsageStatistics(**total_usage.model_dump(), step_count=step_count)

    def build_system_prompt_with_memories(self, raw_system: str, topics: Optional[str] = None, retrieved_memories: Optional[dict] = None) -> Tuple[str, dict]:
        """
        Build the complete system prompt by retrieving memories and combining with the raw system prompt.
        
        Args:
            raw_system (str): The base system prompt
            topics (Optional[str]): Topics to use for memory retrieval
            retrieved_memories (Optional[dict]): Pre-retrieved memories to use instead of fetching new ones
            
        Returns:
            Tuple[str, dict]: The complete system prompt and the retrieved memories dict
        """
        timezone_str = self.user_manager.get_user_by_id(self.user.id).timezone
        
        if retrieved_memories is None:
            retrieved_memories = {}

        key_words = topics if topics is not None else self.agent_state.topic

        if "key_words" in retrieved_memories:
            key_words = retrieved_memories["key_words"]
        else:
            retrieved_memories["key_words"] = key_words

        search_method = 'bm25'

        # Prepare embedding for semantic search
        if key_words != '' and search_method == 'embedding':
            embedded_text = embedding_model(self.agent_state.embedding_config).get_text_embedding(key_words)
            embedded_text = np.array(embedded_text)
            embedded_text = np.pad(embedded_text, (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]), mode="constant").tolist()
        else:
            embedded_text = None

        # Retrieve core memory
        if self.agent_state.name == 'core_memory_agent' or "core" not in retrieved_memories:
            current_persisted_memory = Memory(
                blocks=[self.block_manager.get_block_by_id(block.id, actor=self.user) for block in self.agent_state.memory.get_blocks()]
            )
            core_memory = current_persisted_memory.compile()
            retrieved_memories['core'] = core_memory
        
        if self.agent_state.name == 'knowledge_vault' or 'knowledge_vault' not in retrieved_memories:
            if self.agent_state.name == 'knowledge_vault' or self.agent_state.name == 'reflexion_agent':
                current_knowledge_vault = self.knowledge_vault_manager.list_knowledge(agent_state=self.agent_state, embedded_text=embedded_text, query=key_words, search_field='caption', search_method=search_method, limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM, timezone_str=timezone_str)
            else:
                current_knowledge_vault = self.knowledge_vault_manager.list_knowledge(agent_state=self.agent_state, embedded_text=embedded_text, query=key_words, search_field='caption', search_method=search_method, limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM, timezone_str=timezone_str, sensitivity=['low', 'medium'])
            
            knowledge_vault_memory = ''
            if len(current_knowledge_vault) > 0:
                for idx, knowledge_vault_item in enumerate(current_knowledge_vault):
                    knowledge_vault_memory += f"[{idx}] Knowledge Vault Item ID: {knowledge_vault_item.id}; Caption: {knowledge_vault_item.caption}\n"
            retrieved_memories['knowledge_vault'] = {
                'total_number_of_items': self.knowledge_vault_manager.get_total_number_of_items(),
                'current_count': len(current_knowledge_vault),
                'text': knowledge_vault_memory
            }

        # Retrieve episodic memory
        if self.agent_state.name == 'episodic_memory_agent' or 'episodic' not in retrieved_memories:
            current_episodic_memory = self.episodic_memory_manager.list_episodic_memory(agent_state=self.agent_state, limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM, timezone_str=timezone_str)
            episodic_memory = ''
            if len(current_episodic_memory) > 0:
                for idx, event in enumerate(current_episodic_memory):
                    tree_path_str = f" - Path: {' > '.join(event.tree_path)}" if event.tree_path else ""
                    if self.agent_state.name == 'episodic_memory_agent' or self.agent_state.name == 'reflexion_agent':
                        episodic_memory += f"[Event ID: {event.id}] Timestamp: {event.occurred_at.strftime('%Y-%m-%d %H:%M:%S')} - {event.summary}{tree_path_str} (Details: {len(event.details)} Characters)\n"
                    else:
                        episodic_memory += f"[{idx}] Timestamp: {event.occurred_at.strftime('%Y-%m-%d %H:%M:%S')} - {event.summary}{tree_path_str} (Details: {len(event.details)} Characters)\n"
                        
            recent_episodic_memory = episodic_memory.strip()
        
            most_relevant_episodic_memory = self.episodic_memory_manager.list_episodic_memory(agent_state=self.agent_state, embedded_text=embedded_text, query=key_words, search_field='details', search_method=search_method, limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM, timezone_str=timezone_str)
            most_relevant_episodic_memory_str = ''
            if len(most_relevant_episodic_memory) > 0:
                for idx, event in enumerate(most_relevant_episodic_memory):
                    tree_path_str = f" - Path: {' > '.join(event.tree_path)}" if event.tree_path else ""
                    if self.agent_state.name == 'episodic_memory_agent' or self.agent_state.name == 'reflexion_agent':
                        most_relevant_episodic_memory_str += f"[Event ID: {event.id}] Timestamp: {event.occurred_at.strftime('%Y-%m-%d %H:%M:%S')} - {event.summary}{tree_path_str}  (Details: {len(event.details)} Characters)\n"
                    else:
                        most_relevant_episodic_memory_str += f"[{idx}] Timestamp: {event.occurred_at.strftime('%Y-%m-%d %H:%M:%S')} - {event.summary}{tree_path_str}  (Details: {len(event.details)} Characters)\n"
            relevant_episodic_memory = most_relevant_episodic_memory_str.strip()
            retrieved_memories['episodic'] = {
                'total_number_of_items': self.episodic_memory_manager.get_total_number_of_items(),
                'recent_count': len(current_episodic_memory),
                'relevant_count': len(most_relevant_episodic_memory),
                'recent_episodic_memory': recent_episodic_memory,
                'relevant_episodic_memory': relevant_episodic_memory
            }

        # Retrieve resource memory
        if self.agent_state.name == 'resource_memory_agent' or 'resource' not in retrieved_memories:
            current_resource_memory = self.resource_memory_manager.list_resources(agent_state=self.agent_state, query=key_words, embedded_text=embedded_text, search_field='summary', search_method=search_method, limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM, timezone_str=timezone_str)
            resource_memory = ''
            if len(current_resource_memory) > 0:
                for idx, resource in enumerate(current_resource_memory):
                    tree_path_str = f"; Path: {' > '.join(resource.tree_path)}" if resource.tree_path else ""
                    if self.agent_state.name == 'resource_memory_agent' or self.agent_state.name == 'reflexion_agent':
                        resource_memory += f"[Resource ID: {resource.id}] Resource Title: {resource.title}; Resource Summary: {resource.summary} Resource Type: {resource.resource_type}{tree_path_str}\n"
                    else:
                        resource_memory += f"[{idx}] Resource Title: {resource.title}; Resource Summary: {resource.summary} Resource Type: {resource.resource_type}{tree_path_str}\n"
            resource_memory = resource_memory.strip()
            retrieved_memories['resource'] = {
                'total_number_of_items': self.resource_memory_manager.get_total_number_of_items(),
                'current_count': len(current_resource_memory),
                'text': resource_memory
            }

        # Retrieve procedural memory
        if self.agent_state.name == 'procedural_memory_agent' or 'procedural' not in retrieved_memories:
            current_procedural_memory = self.procedural_memory_manager.list_procedures(agent_state=self.agent_state, query=key_words, embedded_text=embedded_text, search_field="summary", search_method=search_method,limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM, timezone_str=timezone_str)
            procedural_memory = ''
            if len(current_procedural_memory) > 0:
                for idx, procedure in enumerate(current_procedural_memory):
                    tree_path_str = f"; Path: {' > '.join(procedure.tree_path)}" if procedure.tree_path else ""
                    if self.agent_state.name == 'procedural_memory_agent' or self.agent_state.name == 'reflexion_agent':
                        procedural_memory += f"[Procedure ID: {procedure.id}] Entry Type: {procedure.entry_type}; Summary: {procedure.summary}{tree_path_str}\n"
                    else:
                        procedural_memory += f"[{idx}] Entry Type: {procedure.entry_type}; Summary: {procedure.summary}{tree_path_str}\n"
            procedural_memory = procedural_memory.strip()
            retrieved_memories['procedural'] = {
                'total_number_of_items': self.procedural_memory_manager.get_total_number_of_items(),
                'current_count': len(current_procedural_memory),
                'text': procedural_memory
            }
        
        # Retrieve semantic memory
        if self.agent_state.name == 'semantic_memory_agent' or 'semantic' not in retrieved_memories:
            current_semantic_memory = self.semantic_memory_manager.list_semantic_items(agent_state=self.agent_state, query=key_words, embedded_text=embedded_text, search_field="details", search_method=search_method,limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM, timezone_str=timezone_str)
            semantic_memory = ''
            if len(current_semantic_memory) > 0:
                for idx, semantic_memory_item in enumerate(current_semantic_memory):
                    tree_path_str = f"; Path: {' > '.join(semantic_memory_item.tree_path)}" if semantic_memory_item.tree_path else ""
                    if self.agent_state.name == 'semantic_memory_agent' or self.agent_state.name == 'reflexion_agent':
                        semantic_memory += f"[Semantic Memory ID: {semantic_memory_item.id}] Name: {semantic_memory_item.name}; Summary: {semantic_memory_item.summary}{tree_path_str}\n"
                    else:
                        semantic_memory += f"[{idx}] Name: {semantic_memory_item.name}; Summary: {semantic_memory_item.summary}{tree_path_str}\n"
                        
            semantic_memory = semantic_memory.strip()
            retrieved_memories['semantic'] = {
                'total_number_of_items': self.semantic_memory_manager.get_total_number_of_items(),
                'current_count': len(current_semantic_memory),
                'text': semantic_memory
            }

        # Build the complete system prompt
        memory_system_prompt = self.build_system_prompt(retrieved_memories)
        
        complete_system_prompt = raw_system + "\n\n" + memory_system_prompt

        if key_words:
            complete_system_prompt += "\n\nThe above memories are retrieved based on the following keywords. If some memories are empty or does not contain the content related to the keywords, it is highly likely that memory does not contain any relevant information."
        
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
        user_timezone_str = self.user_manager.get_user_by_id(self.user.id).timezone
        user_tz = pytz.timezone(user_timezone_str.split(" (")[0])
        # current_time = datetime.now(user_tz).strftime('%Y-%m-%d %H:%M:%S')
        current_time = "Not Specified"
        
        keywords = retrieved_memories['key_words']
        core_memory = retrieved_memories['core']
        episodic_memory = retrieved_memories['episodic']
        resource_memory = retrieved_memories['resource']
        semantic_memory = retrieved_memories['semantic']
        procedural_memory = retrieved_memories['procedural']
        knowledge_vault = retrieved_memories['knowledge_vault']
        
        system_prompt = template.format(
            current_time=current_time,
            keywords=keywords,
            core_memory=core_memory if core_memory else "Empty",
            episodic_memory=episodic_memory['recent_episodic_memory'] if episodic_memory else "Empty",
        )

        if keywords is not None:
            episodic_total = episodic_memory['total_number_of_items'] if episodic_memory else 0
            relevant_episodic_text = episodic_memory['relevant_episodic_memory'] if episodic_memory else ""
            relevant_count = episodic_memory['relevant_count'] if episodic_memory else 0
            
            system_prompt += f"\n<episodic_memory> Most Relevant Events ({relevant_count} out of {episodic_total} Events Orderred by Relevance to Keywords):\n" + (relevant_episodic_text if relevant_episodic_text else "Empty") + "\n</episodic_memory>\n"
        
        # Add knowledge vault with counts
        knowledge_vault_total = knowledge_vault['total_number_of_items'] if knowledge_vault else 0
        knowledge_vault_text = knowledge_vault['text'] if knowledge_vault else ""
        knowledge_vault_count = knowledge_vault['current_count'] if knowledge_vault else 0
        system_prompt += f"\n<knowledge_vault> ({knowledge_vault_count} out of {knowledge_vault_total} Items):\n" + (knowledge_vault_text if knowledge_vault_text else "Empty") + "\n</knowledge_vault>\n"
        
        # Add semantic memory with counts
        semantic_total = semantic_memory['total_number_of_items'] if semantic_memory else 0
        semantic_text = semantic_memory['text'] if semantic_memory else ""
        semantic_count = semantic_memory['current_count'] if semantic_memory else 0
        system_prompt += f"\n<semantic_memory> ({semantic_count} out of {semantic_total} Items):\n" + (semantic_text if semantic_text else "Empty") + "\n</semantic_memory>\n"
        
        # Add resource memory with counts
        resource_total = resource_memory['total_number_of_items'] if resource_memory else 0
        resource_text = resource_memory['text'] if resource_memory else ""
        resource_count = resource_memory['current_count'] if resource_memory else 0
        system_prompt += f"\n<resource_memory> ({resource_count} out of {resource_total} Items):\n" + (resource_text if resource_text else "Empty") + "\n</resource_memory>\n"
        
        # Add procedural memory with counts
        procedural_total = procedural_memory['total_number_of_items'] if procedural_memory else 0
        procedural_text = procedural_memory['text'] if procedural_memory else ""
        procedural_count = procedural_memory['current_count'] if procedural_memory else 0
        system_prompt += f"\n<procedural_memory> ({procedural_count} out of {procedural_total} Items):\n" + (procedural_text if procedural_text else "Empty") + "\n</procedural_memory>"

        return system_prompt

    def inner_step(
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
        put_inner_thoughts_first: bool = True,
        existing_file_uris: Optional[List[str]] = None,
        extra_messages: Optional[List[dict]] = None,
        initial_message_count: Optional[int] = None,
        return_memory_types_without_update: bool = False,
        message_queue: Optional[any] = None,
        chaining: bool = True,
        **kwargs,
    ) -> AgentStepResponse:
        """Runs a single step in the agent loop (generates at most one LLM call)"""

        try:

            # Step 0: get in-context messages and get the raw system prompt
            in_context_messages = self.agent_manager.get_in_context_messages(agent_id=self.agent_state.id, actor=self.user)
            assert in_context_messages[0].role == MessageRole.system
            raw_system = in_context_messages[0].content[0].text

            # Build the complete system prompt with memories
            complete_system_prompt, retrieved_memories = self.build_system_prompt_with_memories(
                raw_system=raw_system,
                topics=topics,
                retrieved_memories=retrieved_memories
            )

            in_context_messages[0].content[0].text = complete_system_prompt

            # Step 1: add user message
            if isinstance(messages, Message):
                messages = [messages]

            if not all(isinstance(m, Message) for m in messages):
                raise ValueError(f"messages should be a Message or a list of Message, got {type(messages)}")

            input_message_sequence = in_context_messages + messages

            if extra_messages is not None:
                input_message_sequence = input_message_sequence[:initial_message_count] + extra_messages + input_message_sequence[initial_message_count:]

            if len(input_message_sequence) > 1 and input_message_sequence[-1].role != "user":
                self.logger.warning(f"{CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue")

            # Step 2: send the conversation and available functions to the LLM
            response = self._get_ai_reply(
                message_sequence=input_message_sequence,
                first_message=first_message,
                stream=stream,
                step_count=step_count,
                put_inner_thoughts_first=put_inner_thoughts_first,
                existing_file_uris=existing_file_uris,
            )

            # Step 3: check if LLM wanted to call a function
            # (if yes) Step 4: call the function
            # (if yes) Step 5: send the info on the function call and function response to LLM
            all_response_messages = []
            for response_choice in response.choices:
                response_message = response_choice.message
                tmp_response_messages, continue_chaining, function_failed = self._handle_ai_response(
                    first_input_messge, # give the last message to the function so that other agents can see this message through funciton_calls
                    response_message,
                    existing_file_uris=existing_file_uris,
                    # TODO this is kind of hacky, find a better way to handle this
                    # the only time we set up message creation ahead of time is when streaming is on
                    response_message_id=response.id if stream else None,
                    force_response=force_response,
                    retrieved_memories=retrieved_memories,
                    display_intermediate_message=display_intermediate_message,
                    return_memory_types_without_update=return_memory_types_without_update,
                    message_queue=message_queue,
                    chaining=chaining
                )
                all_response_messages.extend(tmp_response_messages)

            if function_failed:
                self.logger.info(f"Function failed with error: {all_response_messages[-1].content[0].text if all_response_messages else 'Unknown error'}")

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
                self.logger.warning(f"Could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
                self.logger.debug(f"Agent state: {self.agent_state}")
                self.agent_state.llm_config.context_window = (
                    LLM_MAX_TOKENS[self.model] if (self.model is not None and self.model in LLM_MAX_TOKENS) else LLM_MAX_TOKENS["DEFAULT"]
                )

            if current_total_tokens > summarizer_settings.memory_warning_threshold * int(self.agent_state.llm_config.context_window):
                self.logger.info(
                    f"Memory pressure detected: last response total_tokens ({current_total_tokens}) > {summarizer_settings.memory_warning_threshold * int(self.agent_state.llm_config.context_window)}"
                )

                # Only deliver the alert if we haven't already (this period)
                if not self.agent_alerted_about_memory_pressure:
                    active_memory_warning = True
                    self.agent_alerted_about_memory_pressure = True  # it's up to the outer loop to handle this

                # if it is too long then run summarization here.
                self.summarize_messages_inplace()

            else:
                self.logger.debug(
                    f"Memory usage acceptable: last response total_tokens ({current_total_tokens}) < {summarizer_settings.memory_warning_threshold * int(self.agent_state.llm_config.context_window)}"
                )

            # Log step - this must happen before messages are persisted
            step = self.step_manager.log_step(
                actor=self.user,
                provider_name=self.agent_state.llm_config.model_endpoint_type,
                model=self.agent_state.llm_config.model,
                context_window_limit=self.agent_state.llm_config.context_window,
                usage=response.usage,
            )
            for message in all_new_messages:
                message.step_id = step.id

            # Persisting into Messages
            self.agent_state = self.agent_manager.append_to_in_context_messages(
                all_new_messages, agent_id=self.agent_state.id, actor=self.user
            )

            return AgentStepResponse(
                messages=all_new_messages,
                continue_chaining=continue_chaining,
                function_failed=function_failed,
                in_context_memory_warning=active_memory_warning,
                usage=response.usage,
            )

        except Exception as e:
            self.logger.error(f"step() failed\nmessages = {messages}\nerror = {e}")

            # If we got a context alert, try trimming the messages length, then try again
            if is_context_overflow_error(e):
                in_context_messages = self.agent_manager.get_in_context_messages(agent_id=self.agent_state.id, actor=self.user)

                if summarize_attempt_count <= summarizer_settings.max_summarizer_retries:
                    self.logger.warning(
                        f"context window exceeded with limit {self.agent_state.llm_config.context_window}, attempting to summarize ({summarize_attempt_count}/{summarizer_settings.max_summarizer_retries}"
                    )
                    # A separate API call to run a summarizer
                    self.summarize_messages_inplace()

                    # Try step again
                    return self.inner_step(
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
                        chaining=chaining
                    )
                else:
                    err_msg = f"Ran summarizer {summarize_attempt_count - 1} times for agent id={self.agent_state.id}, but messages are still overflowing the context window."
                    token_counts = (get_token_counts_for_messages(in_context_messages),)
                    self.logger.error(err_msg)
                    self.logger.error(f"num_in_context_messages: {len(self.agent_state.message_ids)}")
                    self.logger.error(f"token_counts: {token_counts}")
                    raise ContextWindowExceededError(
                        err_msg,
                        details={
                            "num_in_context_messages": len(self.agent_state.message_ids),
                            "in_context_messages_text": [m.text for m in in_context_messages],
                            "token_counts": token_counts,
                        },
                    )

            else:
                self.logger.error(f"step() failed with an unrecognized exception: '{str(e)}'")
                raise e

    def step_user_message(self, user_message_str: str, **kwargs) -> AgentStepResponse:
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
        openai_message_dict = {"role": "user", "content": cleaned_user_message_text, "name": name}

        # Create the associated Message object (in the database)
        assert self.agent_state.created_by_id is not None, "User ID is not set"
        user_message = Message.dict_to_message(
            agent_id=self.agent_state.id,
            model=self.model,
            openai_message_dict=openai_message_dict,
            # created_at=timestamp,
        )

        return self.inner_step(messages=[user_message], **kwargs)

    def summarize_messages_inplace(self):
        in_context_messages = self.agent_manager.get_in_context_messages(agent_id=self.agent_state.id, actor=self.user)
        in_context_messages_openai = [m.to_openai_dict() for m in in_context_messages]
        in_context_messages_openai_no_system = in_context_messages_openai[1:]
        token_counts = get_token_counts_for_messages(in_context_messages)
        self.logger.info(f"System message token count={token_counts[0]}")
        self.logger.info(f"token_counts_no_system={token_counts[1:]}")

        if in_context_messages_openai[0]["role"] != "system":
            raise RuntimeError(f"in_context_messages_openai[0] should be system (instead got {in_context_messages_openai[0]})")

        # If at this point there's nothing to summarize, throw an error
        if len(in_context_messages_openai_no_system) == 0:
            raise ContextWindowExceededError(
                "Not enough messages to compress for summarization",
                details={
                    "num_candidate_messages": len(in_context_messages_openai_no_system),
                    "num_total_messages": len(in_context_messages_openai),
                },
            )

        cutoff = calculate_summarizer_cutoff(in_context_messages=in_context_messages, token_counts=token_counts, logger=self.logger)

        message_sequence_to_summarize = in_context_messages[1:cutoff]  # do NOT get rid of the system message
        self.logger.info(f"Attempting to summarize {len(message_sequence_to_summarize)} messages of {len(in_context_messages)}")

        # We can't do summarize logic properly if context_window is undefined
        if self.agent_state.llm_config.context_window is None:
            # Fallback if for some reason context_window is missing, just set to the default
            self.logger.warning(f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
            self.agent_state.llm_config.context_window = (
                LLM_MAX_TOKENS[self.model] if (self.model is not None and self.model in LLM_MAX_TOKENS) else LLM_MAX_TOKENS["DEFAULT"]
            )

        summary = summarize_messages(agent_state=self.agent_state, message_sequence_to_summarize=message_sequence_to_summarize)
        self.logger.info(f"Got summary: {summary}")

        # Metadata that's useful for the agent to see
        all_time_message_count = self.message_manager.size(agent_id=self.agent_state.id, actor=self.user)
        remaining_message_count = 1 + len(in_context_messages) - cutoff  # System + remaining
        hidden_message_count = all_time_message_count - remaining_message_count
        summary_message_count = len(message_sequence_to_summarize)
        summary_message = package_summarize_message(summary, summary_message_count, hidden_message_count, all_time_message_count)
        self.logger.info(f"Packaged into message: {summary_message}")

        prior_len = len(in_context_messages_openai)
        self.agent_state = self.agent_manager.trim_older_in_context_messages(num=cutoff, agent_id=self.agent_state.id, actor=self.user)
        packed_summary_message = {"role": "user", "content": summary_message}

        # Prepend the summary
        self.agent_state = self.agent_manager.prepend_to_in_context_messages(
            messages=[
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    model=self.model,
                    openai_message_dict=packed_summary_message,
                )
            ],
            agent_id=self.agent_state.id,
            actor=self.user,
        )

        # reset alert
        self.agent_alerted_about_memory_pressure = False
        curr_in_context_messages = self.agent_manager.get_in_context_messages(agent_id=self.agent_state.id, actor=self.user)

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

    def get_context_window(self) -> ContextWindowOverview:
        """Get the context window of the agent"""

        system_prompt = self.agent_state.system  # TODO is this the current system or the initial system?
        num_tokens_system = count_tokens(system_prompt)
        core_memory = self.agent_state.memory.compile()
        num_tokens_core_memory = count_tokens(core_memory)

        # Grab the in-context messages
        # conversion of messages to OpenAI dict format, which is passed to the token counter
        in_context_messages = self.agent_manager.get_in_context_messages(agent_id=self.agent_state.id, actor=self.user)
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

        message_manager_size = self.message_manager.size(actor=self.user, agent_id=self.agent_state.id)
        external_memory_summary = compile_memory_metadata_block(
            memory_edit_timestamp=get_utc_time(),
            previous_message_count=self.message_manager.size(actor=self.user, agent_id=self.agent_state.id),
        )
        num_tokens_external_memory_summary = count_tokens(external_memory_summary)

        # tokens taken up by function definitions
        agent_state_tool_jsons = [t.json_schema for t in self.agent_state.tools]
        if agent_state_tool_jsons:
            available_functions_definitions = [ChatCompletionRequestTool(type="function", function=f) for f in agent_state_tool_jsons]
            num_tokens_available_functions_definitions = num_tokens_from_functions(functions=agent_state_tool_jsons, model=self.model)
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

    def count_tokens(self) -> int:
        """Count the tokens in the current context window"""
        context_window_breakdown = self.get_context_window()
        return context_window_breakdown.context_window_size_current


def save_agent(agent: Agent):
    """Save agent to metadata store"""
    agent_state = agent.agent_state
    assert isinstance(agent_state.memory, Memory), f"Memory is not a Memory object: {type(agent_state.memory)}"

    # TODO: move this to agent manager
    # TODO: Completely strip out metadata
    # convert to persisted model
    agent_manager = AgentManager()
    update_agent = UpdateAgent(
        name=agent_state.name,
        tool_ids=[t.id for t in agent_state.tools],
        block_ids=[b.id for b in agent_state.memory.blocks],
        tags=agent_state.tags,
        system=agent_state.system,
        tool_rules=agent_state.tool_rules,
        llm_config=agent_state.llm_config,
        embedding_config=agent_state.embedding_config,
        message_ids=agent_state.message_ids,
        description=agent_state.description,
        metadata_=agent_state.metadata_,
        # TODO: Add this back in later
        # tool_exec_environment_variables=agent_state.get_agent_env_vars_as_dict(),
    )
    agent_manager.update_agent(agent_id=agent_state.id, agent_update=update_agent, actor=agent.user)


def strip_name_field_from_user_message(user_message_text: str) -> Tuple[str, Optional[str]]:
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
        logger.error(f"Handling of 'name' field failed with: {e}")
        raise e


def validate_json(user_message_text: str) -> str:
    """Make sure that the user input message is valid JSON"""
    try:
        user_message_json = dict(json_loads(user_message_text))
        user_message_json_val = json_dumps(user_message_json)
        return user_message_json_val
    except Exception as e:
        print(f"{CLI_WARNING_PREFIX}couldn't parse user input message as JSON: {e}")
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
    file_manager = FileManager()
    result = []
    
    for content_part in message.content:
        if isinstance(content_part, TextContent):
            result.append({
                'type': 'text',
                'text': content_part.text
            })
        elif isinstance(content_part, ImageContent):
            result.append({
                'type': 'database_image_id',
                'image_id': content_part.image_id
            })
        elif isinstance(content_part, FileContent):
            result.append({
                'type': 'database_file_id',
                'file_id': content_part.file_id,
            })

        elif isinstance(content_part, CloudFileContent):
            result.append({
                'type': 'database_google_cloud_file_uri',
                'cloud_file_uri': content_part.cloud_file_uri,
            })
        else:
            # For any other content types, skip them or handle as text
            # This includes tool calls, tool returns, reasoning content, etc.
            # These are internal message types that shouldn't be converted back
            continue
    
    return result
