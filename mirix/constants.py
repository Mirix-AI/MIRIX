import os
from logging import CRITICAL, DEBUG, ERROR, INFO, NOTSET, WARN, WARNING

MIRIX_DIR = os.path.join(os.path.expanduser("~"), ".mirix")
MIRIX_DIR_TOOL_SANDBOX = os.path.join(MIRIX_DIR, "tool_sandbox_dir")

ADMIN_PREFIX = "/v1/admin"
API_PREFIX = "/v1"
OPENAI_API_PREFIX = "/openai"

COMPOSIO_ENTITY_ENV_VAR_KEY = "COMPOSIO_ENTITY"
COMPOSIO_TOOL_TAG_NAME = "composio"

MIRIX_CORE_TOOL_MODULE_NAME = "mirix.functions.function_sets.base"
MIRIX_MEMORY_TOOL_MODULE_NAME = "mirix.functions.function_sets.memory_tools"
MIRIX_EXTRA_TOOL_MODULE_NAME = "mirix.functions.function_sets.extras"

# String in the error message for when the context window is too large
# Example full message:
# This model's maximum context length is 8192 tokens. However, your messages resulted in 8198 tokens (7450 in the messages, 748 in the functions). Please reduce the length of the messages or functions.
OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING = "maximum context length"

# System prompt templating
IN_CONTEXT_MEMORY_KEYWORD = "CORE_MEMORY"

# OpenAI error message: Invalid 'messages[1].tool_calls[0].id': string too long. Expected a string with maximum length 29, but got a string with length 36 instead.
TOOL_CALL_ID_MAX_LEN = 29

# minimum context window size
MIN_CONTEXT_WINDOW = 4096

# embeddings
MAX_EMBEDDING_DIM = 4096  # maximum supported embeding size - do NOT change or else DBs will need to be reset
DEFAULT_EMBEDDING_CHUNK_SIZE = 300

MAX_CHAINING_STEPS = 10
MAX_RETRIEVAL_LIMIT_IN_SYSTEM = 10

# tokenizers
EMBEDDING_TO_TOKENIZER_MAP = {
    "text-embedding-3-small": "cl100k_base",
}
EMBEDDING_TO_TOKENIZER_DEFAULT = "cl100k_base"


DEFAULT_MIRIX_MODEL = "gpt-4"  # TODO: fixme
DEFAULT_PERSONA = "sam_pov"
DEFAULT_HUMAN = "basic"
DEFAULT_PRESET = "memgpt_chat"

# Base tools that cannot be edited, as they access agent state directly
# Note that we don't include "conversation_search_date" for now
BASE_TOOLS = ["send_message", "send_intermediate_message", "conversation_search", 'search_in_memory', 'list_memory_within_timerange']
# Base memory tools CAN be edited, and are added by default by the server
CORE_MEMORY_TOOLS = ["core_memory_append", "core_memory_rewrite"]
EPISODIC_MEMORY_TOOLS = ['episodic_memory_insert', 'episodic_memory_merge', 'episodic_memory_replace', 'check_episodic_memory']
PROCEDURAL_MEMORY_TOOLS = ['procedural_memory_insert', 'procedural_memory_update']
RESOURCE_MEMORY_TOOLS = ['resource_memory_insert', 'resource_memory_update']
KNOWLEDGE_VAULT_TOOLS = ['knowledge_vault_insert', 'knowledge_vault_update']
SEMANTIC_MEMORY_TOOLS = ['semantic_memory_insert', 'semantic_memory_update', 'check_semantic_memory']
CHAT_AGENT_TOOLS = ['trigger_memory_update_with_instruction']
EXTRAS_TOOLS = ['web_search', 'fetch_and_read_pdf']
META_MEMORY_TOOLS = ['trigger_memory_update']
SEARCH_MEMORY_TOOLS = ['search_in_memory', 'list_memory_within_timerange']
UNIVERSAL_MEMORY_TOOLS = ['search_in_memory', "finish_memory_update", 'list_memory_within_timerange']
ALL_TOOLS = list(set(BASE_TOOLS + CORE_MEMORY_TOOLS + EPISODIC_MEMORY_TOOLS + PROCEDURAL_MEMORY_TOOLS + RESOURCE_MEMORY_TOOLS + KNOWLEDGE_VAULT_TOOLS + SEMANTIC_MEMORY_TOOLS + META_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS + CHAT_AGENT_TOOLS + EXTRAS_TOOLS))

# The name of the tool used to send message to the user
# May not be relevant in cases where the agent has multiple ways to message to user (send_imessage, send_discord_mesasge, ...)
# or in cases where the agent has no concept of messaging a user (e.g. a workflow agent)
DEFAULT_MESSAGE_TOOL = "send_message"
DEFAULT_MESSAGE_TOOL_KWARG = "message"

# Structured output models
STRUCTURED_OUTPUT_MODELS = {"gpt-4o", "gpt-4o-mini"}

# LOGGER_LOG_LEVEL is use to convert Text to Logging level value for logging mostly for Cli input to setting level
LOGGER_LOG_LEVELS = {"CRITICAL": CRITICAL, "ERROR": ERROR, "WARN": WARN, "WARNING": WARNING, "INFO": INFO, "DEBUG": DEBUG, "NOTSET": NOTSET}

FIRST_MESSAGE_ATTEMPTS = 10

INITIAL_BOOT_MESSAGE = "Boot sequence complete. Persona activated."
INITIAL_BOOT_MESSAGE_SEND_MESSAGE_THOUGHT = "Bootup sequence complete. Persona activated. Testing messaging functionality."
STARTUP_QUOTES = [
    "I think, therefore I am.",
    "All those moments will be lost in time, like tears in rain.",
    "More human than human is our motto.",
]
INITIAL_BOOT_MESSAGE_SEND_MESSAGE_FIRST_MSG = STARTUP_QUOTES[2]

CLI_WARNING_PREFIX = "Warning: "

ERROR_MESSAGE_PREFIX = "Error"

NON_USER_MSG_PREFIX = "[This is an automated system message hidden from the user] "

# Constants to do with summarization / conversation length window
# The max amount of tokens supported by the underlying model (eg 8k for gpt-4 and Mistral 7B)
LLM_MAX_TOKENS = {
    "DEFAULT": 8192,
    ## OpenAI models: https://platform.openai.com/docs/models/overview
    # "o1-preview
    "chatgpt-4o-latest": 128000,
    # "o1-preview-2024-09-12
    "gpt-4o-2024-08-06": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo-instruct": 16385,
    "gpt-4-0125-preview": 128000,
    "gpt-3.5-turbo-0125": 16385,
    # "babbage-002": 128000,
    # "davinci-002": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    # "gpt-4o-realtime-preview-2024-10-01
    "gpt-4-turbo": 8192,
    "gpt-4o-2024-05-13": 128000,
    # "o1-mini
    # "o1-mini-2024-09-12
    # "gpt-3.5-turbo-instruct-0914
    "gpt-4o-mini": 128000,
    # "gpt-4o-realtime-preview
    "gpt-4o-mini-2024-07-18": 128000,
    # gpt-4
    "gpt-4-1106-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    "gpt-4-0314": 8192,  # legacy
    "gpt-4-32k-0314": 32768,  # legacy
    # gpt-3.5
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-0613": 4096,  # legacy
    "gpt-3.5-turbo-16k-0613": 16385,  # legacy
    "gpt-3.5-turbo-0301": 4096,  # legacy
}
# The error message that Mirix will receive
# MESSAGE_SUMMARY_WARNING_STR = f"Warning: the conversation history will soon reach its maximum length and be trimmed. Make sure to save any important information from the conversation to your memory before it is removed."
# Much longer and more specific variant of the prompt
# TODO: Emit the warning to Meta Memory Manager instead of the Chat Agent.
MESSAGE_SUMMARY_WARNING_STR = " ".join(
    [
        f"{NON_USER_MSG_PREFIX}The conversation history will soon reach its maximum length and be trimmed.",
        "Do NOT tell the user about this system alert, they should not know that the history is reaching max length.",
    ]
)

# The ackknowledgement message used in the summarize sequence
MESSAGE_SUMMARY_REQUEST_ACK = "Understood, I will respond with a summary of the message (and only the summary, nothing else) once I receive the conversation history. I'm ready."

# Maximum length of an error message
MAX_ERROR_MESSAGE_CHAR_LIMIT = 500

# Default memory limits
CORE_MEMORY_PERSONA_CHAR_LIMIT: int = 5000
CORE_MEMORY_HUMAN_CHAR_LIMIT: int = 5000
CORE_MEMORY_BLOCK_CHAR_LIMIT: int = 5000

# Function return limits
FUNCTION_RETURN_CHAR_LIMIT = 60000  # ~300 words

MAX_PAUSE_HEARTBEATS = 360  # in min

MESSAGE_CHATGPT_FUNCTION_MODEL = "gpt-3.5-turbo"
MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE = "You are a helpful assistant. Keep your responses short and concise."

#### Functions related

# REQ_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}continue_chaining == true"
REQ_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function called using continue_chaining=true, returning control"
# FUNC_FAILED_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function call failed"
FUNC_FAILED_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function call failed, returning control"


RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE = 5

MAX_FILENAME_LENGTH = 255
RESERVED_FILENAMES = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1", "LPT2"}

MAX_IMAGES_TO_PROCESS = 100

DEFAULT_WRAPPER_NAME = "chatml"
INNER_THOUGHTS_KWARG = "inner_thoughts"
INNER_THOUGHTS_KWARG_DESCRIPTION = "Deep inner monologue private to you only."
INNER_THOUGHTS_KWARG_DESCRIPTION_GO_FIRST = f"Deep inner monologue private to you only. Think before you act, so always generate arg '{INNER_THOUGHTS_KWARG}' first before any other arg."
INNER_THOUGHTS_CLI_SYMBOL = "💭"
ASSISTANT_MESSAGE_CLI_SYMBOL = "🤖"

CLEAR_HISTORY_AFTER_MEMORY_UPDATE = True
CALL_MEMORY_AGENT_IN_PARALLEL = False
CHAINING_FOR_MEMORY_UPDATE = False

LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY = False
BUILD_EMBEDDINGS_FOR_MEMORY = True