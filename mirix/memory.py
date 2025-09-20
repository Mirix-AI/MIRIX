from typing import Callable, Dict, List, Optional

from mirix.constants import MESSAGE_SUMMARY_REQUEST_ACK
from mirix.llm_api.llm_api_tools import create
from mirix.prompts.gpt_summarize import SYSTEM as SUMMARY_PROMPT_SYSTEM
from mirix.schemas.agent import AgentState
from mirix.schemas.enums import MessageRole
from mirix.schemas.memory import Memory
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent
from mirix.settings import summarizer_settings
from mirix.utils import count_tokens, printd
from mirix.llm_api.llm_client import LLMClient

def get_memory_functions(cls: Memory) -> Dict[str, Callable]:
    """Get memory functions for a memory class"""
    functions = {}

    # collect base memory functions (should not be included)
    base_functions = []
    for func_name in dir(Memory):
        funct = getattr(Memory, func_name)
        if callable(funct):
            base_functions.append(func_name)

    for func_name in dir(cls):
        if func_name.startswith("_") or func_name in ["load", "to_dict"]:  # skip base functions
            continue
        if func_name in base_functions:  # dont use BaseMemory functions
            continue
        func = getattr(cls, func_name)
        if not callable(func):  # not a function
            continue
        functions[func_name] = func
    return functions


def _format_summary_history(message_history: List[Message]):
    # TODO use existing prompt formatters for this (eg ChatML)
    def format_message(m: Message):
        content_str = ''
        for content in m.content:
            if content.type == 'text':
                content_str += content.text + "\n"
            elif content.type == 'image_url':
                content_str += f"[Image: {content.image_id}]" + "\n"
            elif content.type == 'file_uri':
                content_str += f"[File: {content.file_id}]" + "\n"
            elif content.type == 'google_cloud_file_uri':
                content_str += f"[Cloud File: {content.cloud_file_uri}]" + "\n"
            else:
                content_str += f"[Unknown content type: {content.type}]" + "\n"
        return content_str.strip()
    return "\n\n".join([f"{m.role}: {format_message(m)}" for m in message_history])


def summarize_messages(
    agent_state: AgentState,
    message_sequence_to_summarize: List[Message],
    existing_file_uris: Optional[List[str]] = None,
):
    """Summarize a message sequence using GPT"""
    # we need the context_window
    context_window = agent_state.llm_config.context_window

    summary_prompt = SUMMARY_PROMPT_SYSTEM
    summary_input = _format_summary_history(message_sequence_to_summarize)
    summary_input_tkns = count_tokens(summary_input)
    if summary_input_tkns > summarizer_settings.memory_warning_threshold * context_window:
        trunc_ratio = (summarizer_settings.memory_warning_threshold * context_window / summary_input_tkns) * 0.8  # For good measure...
        cutoff = int(len(message_sequence_to_summarize) * trunc_ratio)
        summary_input = str(
            [summarize_messages(agent_state, message_sequence_to_summarize=message_sequence_to_summarize[:cutoff])]
            + message_sequence_to_summarize[cutoff:]
        )

    dummy_agent_id = agent_state.id
    message_sequence = [
        Message(agent_id=dummy_agent_id, role=MessageRole.system, content=[TextContent(text=summary_prompt)]),
        Message(agent_id=dummy_agent_id, role=MessageRole.assistant, content=[TextContent(text=MESSAGE_SUMMARY_REQUEST_ACK)]),
        Message(agent_id=dummy_agent_id, role=MessageRole.user, content=[TextContent(text=summary_input)]),
    ]

    # TODO: We need to eventually have a separate LLM config for the summarizer LLM
    llm_config_no_inner_thoughts = agent_state.llm_config.model_copy(deep=True)
    llm_config_no_inner_thoughts.put_inner_thoughts_in_kwargs = False
    # response = create(
    #     llm_config=llm_config_no_inner_thoughts,
    #     messages=message_sequence,
    #     stream=False,
    #     summarizing=True
    # )
    llm_client = LLMClient.create(
        llm_config=llm_config_no_inner_thoughts,
    )
    response = llm_client.send_llm_request(
        messages=message_sequence,
        existing_file_uris=existing_file_uris,
    )

    printd(f"summarize_messages gpt reply: {response.choices[0]}")
    reply = response.choices[0].message.content
    return reply
