import logging
import os
from logging import CRITICAL, DEBUG, ERROR, INFO, NOTSET, WARN, WARNING
from typing import Optional

# ============================================================================
# Client Constants - Used by both client and server
# ============================================================================

# Default organization and admin user IDs (needed by schemas)
DEFAULT_ORG_ID = "org-00000000-0000-4000-8000-000000000000"
ADMIN_USER_ID = "user-00000000-0000-4000-8000-000000000000"

# Embedding constants
MAX_EMBEDDING_DIM = 4096  # maximum supported embedding size - do NOT change or else DBs will need to be reset
DEFAULT_EMBEDDING_CHUNK_SIZE = 300
MIN_CONTEXT_WINDOW = 4096

# Memory limits
CORE_MEMORY_BLOCK_CHAR_LIMIT: int = 5000

# Function/Tool constants
FUNCTION_RETURN_CHAR_LIMIT = 60000  # ~300 words
TOOL_CALL_ID_MAX_LEN = 29

# Tool module names
COMPOSIO_TOOL_TAG_NAME = "composio"
MIRIX_CORE_TOOL_MODULE_NAME = "mirix.functions.function_sets.base"
MIRIX_MEMORY_TOOL_MODULE_NAME = "mirix.functions.function_sets.memory_tools"
MIRIX_EXTRA_TOOL_MODULE_NAME = "mirix.functions.function_sets.extras"

# Message defaults
DEFAULT_MESSAGE_TOOL = "send_message"
DEFAULT_MESSAGE_TOOL_KWARG = "message"

# LLM model token limits
LLM_MAX_TOKENS = {
    "DEFAULT": 8192,
    ## OpenAI models: https://platform.openai.com/docs/models/overview
    "chatgpt-4o-latest": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo-instruct": 16385,
    "gpt-4-0125-preview": 128000,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo": 8192,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    "gpt-4-0314": 8192,  # legacy
    "gpt-4-32k-0314": 32768,  # legacy
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-0613": 4096,  # legacy
    "gpt-3.5-turbo-16k-0613": 16385,  # legacy
    "gpt-3.5-turbo-0301": 4096,  # legacy
}

# ============================================================================
# Server-Only Constants
# ============================================================================

MIRIX_DIR = os.path.join(os.path.expanduser("~"), ".mirix")
MIRIX_DIR_TOOL_SANDBOX = os.path.join(MIRIX_DIR, "tool_sandbox_dir")

ADMIN_PREFIX = "/v1/admin"
API_PREFIX = "/v1"
OPENAI_API_PREFIX = "/openai"

COMPOSIO_ENTITY_ENV_VAR_KEY = "COMPOSIO_ENTITY"

# String in the error message for when the context window is too large
# Example full message:
# This model's maximum context length is 8192 tokens. However, your messages resulted in 8198 tokens (7450 in the messages, 748 in the functions). Please reduce the length of the messages or functions.
OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING = "maximum context length"

# System prompt templating
IN_CONTEXT_MEMORY_KEYWORD = "CORE_MEMORY"

MAX_CHAINING_STEPS = int(os.getenv("MAX_CHAINING_STEPS", "10"))
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
BASE_TOOLS = [
    "send_intermediate_message",
    "conversation_search",
    "search_in_memory",
    "list_memory_within_timerange",
]
# Base memory tools CAN be edited, and are added by default by the server
CORE_MEMORY_TOOLS = ["core_memory_append", "core_memory_rewrite"]
EPISODIC_MEMORY_TOOLS = [
    "episodic_memory_insert",
    "episodic_memory_merge",
    "episodic_memory_replace",
    "check_episodic_memory",
]
SKILL_TOOLS = ["skill_list", "skill_read", "skill_create", "skill_edit", "skill_delete"]
# Tools of the procedural agent that work only inside a skill curation batch.
# Not in SKILL_TOOLS: the auto-dream agent never curates a batch.
SKILL_CURATION_TOOLS = ["experience_triage"]
# The only tools a skill curation batch offers and runs (the curator agent's
# tool_allowlist): the skill tools, triage and the finish call. The procedural
# agent's other tools (search_in_memory, list_memory_within_timerange) search
# memories a curation batch has no use for.
SKILL_CURATOR_TOOLS = SKILL_TOOLS + SKILL_CURATION_TOOLS + ["finish_memory_update"]
RESOURCE_MEMORY_TOOLS = ["resource_memory_insert", "resource_memory_update"]
KNOWLEDGE_VAULT_TOOLS = ["knowledge_vault_insert", "knowledge_vault_update"]
SEMANTIC_MEMORY_TOOLS = [
    "semantic_memory_insert",
    "semantic_memory_update",
    "check_semantic_memory",
]
CHAT_AGENT_TOOLS = []
EXTRAS_TOOLS = ["web_search", "fetch_and_read_pdf"]
MCP_TOOLS = []
META_MEMORY_TOOLS = ["trigger_memory_update"]
SEARCH_MEMORY_TOOLS = ["search_in_memory", "list_memory_within_timerange"]
UNIVERSAL_MEMORY_TOOLS = [
    "search_in_memory",
    "finish_memory_update",
    "list_memory_within_timerange",
]
AUTO_DREAM_TOOLS = list(
    set(
        CORE_MEMORY_TOOLS
        + EPISODIC_MEMORY_TOOLS
        # auto-dream consolidates procedural memory via our skill paradigm
        # (skill_* tools), not the removed procedural_memory_insert/update.
        + SKILL_TOOLS
        + RESOURCE_MEMORY_TOOLS
        + KNOWLEDGE_VAULT_TOOLS
        + SEMANTIC_MEMORY_TOOLS
        + UNIVERSAL_MEMORY_TOOLS
    )
)
ALL_TOOLS = list(
    set(
        BASE_TOOLS
        + CORE_MEMORY_TOOLS
        + EPISODIC_MEMORY_TOOLS
        + SKILL_TOOLS
        + SKILL_CURATION_TOOLS
        + RESOURCE_MEMORY_TOOLS
        + KNOWLEDGE_VAULT_TOOLS
        + SEMANTIC_MEMORY_TOOLS
        + META_MEMORY_TOOLS
        + UNIVERSAL_MEMORY_TOOLS
        + CHAT_AGENT_TOOLS
        + EXTRAS_TOOLS
        + MCP_TOOLS
    )
)

# The name of the tool used to send message to the user
# Structured output models
STRUCTURED_OUTPUT_MODELS = {"gpt-4o", "gpt-4o-mini"}

# LOGGER_LOG_LEVEL is use to convert Text to Logging level value for logging mostly for Cli input to setting level
LOGGER_LOG_LEVELS = {
    "CRITICAL": CRITICAL,
    "ERROR": ERROR,
    "WARN": WARN,
    "WARNING": WARNING,
    "INFO": INFO,
    "DEBUG": DEBUG,
    "NOTSET": NOTSET,
}

FIRST_MESSAGE_ATTEMPTS = 10

INITIAL_BOOT_MESSAGE = "Boot sequence complete. Persona activated."
INITIAL_BOOT_MESSAGE_SEND_MESSAGE_THOUGHT = (
    "Bootup sequence complete. Persona activated. Testing messaging functionality."
)
STARTUP_QUOTES = [
    "I think, therefore I am.",
    "All those moments will be lost in time, like tears in rain.",
    "More human than human is our motto.",
]
INITIAL_BOOT_MESSAGE_SEND_MESSAGE_FIRST_MSG = STARTUP_QUOTES[2]

CLI_WARNING_PREFIX = "Warning: "

ERROR_MESSAGE_PREFIX = "Error"

NON_USER_MSG_PREFIX = "[This is an automated system message hidden from the user] "

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

MAX_PAUSE_HEARTBEATS = 360  # in min

MESSAGE_CHATGPT_FUNCTION_MODEL = "gpt-3.5-turbo"
MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Keep your responses short and concise."
)

#### Functions related

# REQ_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}continue_chaining == true"
REQ_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function called using continue_chaining=true, returning control"
# FUNC_FAILED_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function call failed"
FUNC_FAILED_HEARTBEAT_MESSAGE = (
    f"{NON_USER_MSG_PREFIX}Function call failed, returning control"
)
# A step with a hard step cap (the skill curator) ends only through
# finish_memory_update, the context limit or the step cap.
UNFINISHED_TURN_HEARTBEAT_MESSAGE = (
    f"{NON_USER_MSG_PREFIX}You ended your turn without calling "
    "finish_memory_update(). The work is not finished until you call it: "
    "continue with the remaining items, then call finish_memory_update()."
)


RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE = 5

MAX_FILENAME_LENGTH = 255
RESERVED_FILENAMES = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1", "LPT2"}

MAX_IMAGES_TO_PROCESS = 100

DEFAULT_WRAPPER_NAME = "chatml"
INNER_THOUGHTS_KWARG_DESCRIPTION = "Deep inner monologue private to you only."
INNER_THOUGHTS_CLI_SYMBOL = "💭"
ASSISTANT_MESSAGE_CLI_SYMBOL = "🤖"

CLEAR_HISTORY_AFTER_MEMORY_UPDATE = os.getenv(
    "CLEAR_HISTORY_AFTER_MEMORY_UPDATE", "true"
).lower() in (
    "true",
    "1",
    "yes",
)
# When clearing history after a memory update, retain the raw conversation
# messages of the most-recent N sessions (per agent+user) instead of hard-deleting
# them. These retained rows stay DETACHED from the agent's message_ids (NOT re-added
# to any in-context list — token economy preserved); they exist only so a later
# distiller / auto-dream pass can read the raw transcript. The window is a bounded
# rolling window: when a new session arrives the oldest retained session ages out
# and becomes eligible for deletion on the next clear. Default 5; 0 disables
# retention entirely (legacy hard-delete-everything behavior).
MESSAGE_RETAIN_LAST_N_SESSIONS = int(os.getenv("MESSAGE_RETAIN_LAST_N_SESSIONS", "5"))
CALL_MEMORY_AGENT_IN_PARALLEL = os.getenv(
    "CALL_MEMORY_AGENT_IN_PARALLEL", "false"
).lower() in ("true", "1", "yes")
CHAINING_FOR_MEMORY_UPDATE = os.getenv(
    "CHAINING_FOR_MEMORY_UPDATE", "false"
).lower() in ("true", "1", "yes")
CHAINING_FOR_META_AGENT = os.getenv("CHAINING_FOR_META_AGENT", "true").lower() in (
    "true",
    "1",
    "yes",
)

LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY = os.getenv(
    "LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY", "false"
).lower() in (
    "true",
    "1",
    "yes",
)
BUILD_EMBEDDINGS_FOR_MEMORY = os.getenv(
    "BUILD_EMBEDDINGS_FOR_MEMORY", "true"
).lower() in ("true", "1", "yes")
SKILL_TRIGGER_MESSAGE_THRESHOLD = int(
    os.getenv("SKILL_TRIGGER_MESSAGE_THRESHOLD", "10")
)
# Number of ENDED, not-yet-distilled task sessions that must be pending in one
# (organization, user, meta agent) scope before a skill learning run fires
# automatically. Strict: below the threshold nothing fires, no
# matter how long the sessions wait — use POST /memory/skills/flush to learn
# from fewer. Sessions already claimed by a queued/running learning run do not
# count.
SKILL_TRIGGER_SESSION_THRESHOLD = int(os.getenv("SKILL_TRIGGER_SESSION_THRESHOLD", "5"))

# ---------------------------------------------------------------------------
# Conversation-session lifecycle and skill learning runs.
# ---------------------------------------------------------------------------
DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS = 172800
# How often a live learning run refreshes its heartbeat (every 30 s).
LEARNING_RUN_HEARTBEAT_SECONDS = 30
# A lease must outlast several heartbeats, or every sweep would reclaim runs
# that are alive (and their sessions would be claimed twice).
MIN_LEASE_HEARTBEATS = 3


def _positive_int_setting(name: str, raw: Optional[str], default: int) -> int:
    """``raw`` as a positive int; the default (with a warning) otherwise."""
    try:
        value = int(raw) if raw is not None and str(raw).strip() else default
    except ValueError:
        value = 0
    if value <= 0:
        logging.getLogger(__name__).warning(
            "%s=%r is not a positive integer; using the default %d", name, raw, default
        )
        return default
    return value


def _lease_seconds_setting(raw: Optional[str], heartbeat_seconds: int) -> int:
    """The learning-run lease, at least MIN_LEASE_HEARTBEATS heartbeats long.

    A shorter lease (e.g. MIRIX_LEARNING_RUN_LEASE_SECONDS=20 to speed up
    recovery) would let a sweep reclaim runs that are alive, so it is raised to
    the minimum with a warning.
    """
    lease = _positive_int_setting("MIRIX_LEARNING_RUN_LEASE_SECONDS", raw, 300)
    minimum = MIN_LEASE_HEARTBEATS * heartbeat_seconds
    if lease < minimum:
        logging.getLogger(__name__).warning(
            "MIRIX_LEARNING_RUN_LEASE_SECONDS=%d is shorter than %d heartbeats of "
            "%d s; using %d s so live runs are never reclaimed",
            lease,
            MIN_LEASE_HEARTBEATS,
            heartbeat_seconds,
            minimum,
        )
        return minimum
    return lease


# A session with no message for this long counts as ended ("idle"); the first
# check that sees it writes closed_at = last_message_at + timeout. Must
# be positive: with 0 every session would end the moment it is written.
SESSION_IDLE_TIMEOUT_SECONDS = _positive_int_setting(
    "MIRIX_SESSION_IDLE_TIMEOUT_SECONDS",
    os.getenv("MIRIX_SESSION_IDLE_TIMEOUT_SECONDS"),
    DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
)
# Period of the background sweeper started in the FastAPI lifespan: it runs the
# trigger check for every scope that has pending sessions. <= 0 disables the
# periodic sweep (lease recovery keeps its own cadence, see below).
SESSION_SWEEP_INTERVAL_SECONDS = int(
    os.getenv("MIRIX_SESSION_SWEEP_INTERVAL_SECONDS", "3600")
)
# A queued/running learning run whose heartbeat is older than this is marked
# failed ('lease_expired'), which releases the sessions it claimed.
# Recovery runs at startup and then every half lease, independently of the
# sweep interval, so a crashed run is released about one lease after it died.
LEARNING_RUN_LEASE_SECONDS = _lease_seconds_setting(
    os.getenv("MIRIX_LEARNING_RUN_LEASE_SECONDS"), LEARNING_RUN_HEARTBEAT_SECONDS
)
# Cross-run backoff after an infrastructure failure while distilling a session:
# the session stays pending but is neither counted nor claimed
# before conversation_session.distill_retry_after = now + min * 2**(n - 1),
# capped at max, n = its consecutive infrastructure failures.
DISTILL_RETRY_BACKOFF_MIN_SECONDS = _positive_int_setting(
    "MIRIX_DISTILL_RETRY_BACKOFF_MIN_SECONDS",
    os.getenv("MIRIX_DISTILL_RETRY_BACKOFF_MIN_SECONDS"),
    60,
)
DISTILL_RETRY_BACKOFF_MAX_SECONDS = max(
    DISTILL_RETRY_BACKOFF_MIN_SECONDS,
    _positive_int_setting(
        "MIRIX_DISTILL_RETRY_BACKOFF_MAX_SECONDS",
        os.getenv("MIRIX_DISTILL_RETRY_BACKOFF_MAX_SECONDS"),
        3600,
    ),
)


# ---------------------------------------------------------------------------
# LLM transport.
# ---------------------------------------------------------------------------
def _non_negative_int_setting(name: str, raw: Optional[str], default: int) -> int:
    """``raw`` as an int >= 0; the default (with a warning) otherwise."""
    try:
        value = int(raw) if raw is not None and str(raw).strip() else default
    except ValueError:
        value = -1
    if value < 0:
        logging.getLogger(__name__).warning(
            "%s=%r is not a non-negative integer; using the default %d",
            name,
            raw,
            default,
        )
        return default
    return value


def _positive_float_setting(name: str, raw: Optional[str], default: float) -> float:
    """``raw`` as a float > 0; the default (with a warning) otherwise."""
    try:
        value = float(raw) if raw is not None and str(raw).strip() else default
    except ValueError:
        value = 0.0
    if not value > 0:
        logging.getLogger(__name__).warning(
            "%s=%r is not a positive number; using the default %s", name, raw, default
        )
        return default
    return value


# Applied explicitly to every model SDK client (OpenAI-compatible, Azure,
# Anthropic) and to the Google client's HTTP calls: the per-attempt timeout and
# how many times the TRANSPORT layer retries one call (connection errors,
# timeouts, 408/409/429/5xx, honoring Retry-After). The eval sets retries to 0
# and lets its gateway retry within one 10-minute deadline; the
# application-level retry policy is separate.
LLM_TIMEOUT_SECONDS = _positive_float_setting(
    "MIRIX_LLM_TIMEOUT_SECONDS", os.getenv("MIRIX_LLM_TIMEOUT_SECONDS"), 600.0
)
LLM_MAX_RETRIES = _non_negative_int_setting(
    "MIRIX_LLM_MAX_RETRIES", os.getenv("MIRIX_LLM_MAX_RETRIES"), 2
)
# The per-attempt timeout of an embedding call, for every embedding type: how
# long MIRIX waits for one answer of the embedding endpoint (the connect at
# most 5 s) before the embedding retry policy retries it. The eval sets it
# above its gateway's call deadline, like MIRIX_LLM_TIMEOUT_SECONDS, so MIRIX
# always receives the gateway's classified answer instead of abandoning the
# call and sending it again.
EMBEDDING_TIMEOUT_SECONDS = _positive_float_setting(
    "MIRIX_EMBEDDING_TIMEOUT_SECONDS",
    os.getenv("MIRIX_EMBEDDING_TIMEOUT_SECONDS"),
    60.0,
)

# ---------------------------------------------------------------------------
# Image intake bounds (memory-add payloads, remote image fetches).
# ---------------------------------------------------------------------------
# Formats every supported model provider accepts; anything else is a 422 at
# ingest instead of a provider error later in the worker or the distiller.
ALLOWED_IMAGE_FORMATS = ("PNG", "JPEG", "GIF", "WEBP")
MAX_IMAGE_BYTES = _positive_int_setting(
    "MIRIX_MAX_IMAGE_BYTES", os.getenv("MIRIX_MAX_IMAGE_BYTES"), 20 * 1024 * 1024
)
MAX_IMAGE_BYTES_PER_REQUEST = _positive_int_setting(
    "MIRIX_MAX_IMAGE_BYTES_PER_REQUEST",
    os.getenv("MIRIX_MAX_IMAGE_BYTES_PER_REQUEST"),
    64 * 1024 * 1024,
)
MAX_IMAGE_PIXELS = _positive_int_setting(
    "MIRIX_MAX_IMAGE_PIXELS", os.getenv("MIRIX_MAX_IMAGE_PIXELS"), 50_000_000
)
IMAGE_DETAIL_VALUES = ("auto", "low", "high")
# Server-side fetch of a caller-supplied https image URL (only the Google client
# needs the bytes; OpenAI and Anthropic fetch URLs themselves).
REMOTE_IMAGE_FETCH_TIMEOUT_SECONDS = float(
    os.getenv("MIRIX_REMOTE_IMAGE_FETCH_TIMEOUT_SECONDS", "10")
)
# Deadline for the WHOLE fetch (resolution + download): the per-operation
# timeout above resets on every byte, so a server trickling bytes would
# otherwise hold the request open indefinitely.
REMOTE_IMAGE_FETCH_TOTAL_TIMEOUT_SECONDS = float(
    os.getenv("MIRIX_REMOTE_IMAGE_FETCH_TOTAL_TIMEOUT_SECONDS", "30")
)

# ---------------------------------------------------------------------------
# Adaptive per-session routing (session_tag).
# ---------------------------------------------------------------------------
# A session is either a "task" (an agentic rollout: the agent acts with tools or
# in an environment) or a "conversation" (chat / dialogue). The tag decides which memory
# machinery engages:
#   task         -> skill learning only: /memory/add(_sync)
#                   store the turns in the conversation store and never call the
#                   meta memory agent (a session_id is required); once N task
#                   sessions have ended, a skill learning run fires
#                   (SKILL_TRIGGER_SESSION_THRESHOLD).
#   conversation -> the meta memory agent extracts episodic/semantic/... memory
#                   as usual; the procedural batch trigger does NOT fire (no
#                   skills), and experience consolidation stays operator-invoked
#                   (auto_dream mode="experience"), as for any dialogue memory.
# Callers declare the tag on ingest (AddMemoryRequest.session_tag). A session's
# first write fixes its tag. Unset falls back to DEFAULT_SESSION_TAG so
# historical/untagged callers keep dialogue-only behaviour (no spurious skill
# distillation).
SESSION_TAG_TASK = "task"
SESSION_TAG_CONVERSATION = "conversation"
VALID_SESSION_TAGS = (SESSION_TAG_TASK, SESSION_TAG_CONVERSATION)
DEFAULT_SESSION_TAG = os.getenv("MIRIX_DEFAULT_SESSION_TAG", SESSION_TAG_CONVERSATION)

# ---------------------------------------------------------------------------
# Skill curator.
#
# A learning run curates ALL pending experiences of its scope, oldest first, in
# batches of at most MIRIX_SKILL_CURATOR_BATCH_FRACTION (0.5) of the curator's
# input budget. Each batch may make one create/edit per experience (its
# edit budget); a learning run may soft-delete at most one skill across
# all its batches. A batch ends when the curator calls
# finish_memory_update, when the next call's estimated input would pass
# floor(window * 0.8) - planning output reserve ("context_limit"), or after
# MIRIX_SKILL_CURATOR_MAX_STEPS LLM calls ("step_cap"); the last two return the
# batch to pending with curation_attempts + 1, and after
# SKILL_CURATION_MAX_ATTEMPTS it is closed out as consumed (unaddressed).
# ---------------------------------------------------------------------------
def _fraction_setting(name: str, raw: Optional[str], default: float) -> float:
    """``raw`` as a float in (0, 1]; the default (with a warning) otherwise."""
    try:
        value = float(raw) if raw is not None and str(raw).strip() else default
    except ValueError:
        value = -1.0
    if not 0.0 < value <= 1.0:
        logging.getLogger(__name__).warning(
            "%s=%r is not a fraction in (0, 1]; using the default %s", name, raw, default
        )
        return default
    return value


SKILL_CURATOR_MAX_STEPS = _positive_int_setting(
    "MIRIX_SKILL_CURATOR_MAX_STEPS", os.getenv("MIRIX_SKILL_CURATOR_MAX_STEPS"), 100
)
SKILL_CURATOR_BATCH_FRACTION = _fraction_setting(
    "MIRIX_SKILL_CURATOR_BATCH_FRACTION",
    os.getenv("MIRIX_SKILL_CURATOR_BATCH_FRACTION"),
    0.5,
)
# The curator stops before a call whose estimated input would pass
# floor(window * this) - planning output reserve (the usable window).
SKILL_CURATOR_CONTEXT_STOP_FRACTION = 0.8
# Batches that ended without the curator finishing (context_limit, step_cap,
# a non-infrastructure curator failure) before an experience is closed out as
# consumed (unaddressed).
SKILL_CURATION_MAX_ATTEMPTS = 3
# Tool calls of one model turn a curation batch runs (Agent.step
# max_tool_calls_per_turn): the rest are answered "Not run", and
# finish_memory_update runs only as a turn's only call. 1 is the serial
# tool-call contract (parallel_tool_calls=false) that MIRIX's Anthropic path
# already sends: the model reads every result before its next call, so a
# burst of blind calls (a create echoing the tool schema, a finish before the
# triage was read) cannot run.
SKILL_CURATOR_TOOL_CALLS_PER_TURN = _positive_int_setting(
    "MIRIX_SKILL_CURATOR_TOOL_CALLS_PER_TURN",
    os.getenv("MIRIX_SKILL_CURATOR_TOOL_CALLS_PER_TURN"),
    1,
)
# How often finish_memory_update refuses to end a curation batch while some of
# its experiences are neither cited nor triaged (the refusal lists them). One
# reminder turns a silent default into an explicit decision; the next finish
# goes through (they count as deferred), so a model that cannot triage never
# spins until the step cap.
SKILL_CURATOR_FINISH_REMINDERS = 1

# Hard ceiling on a skill's `instructions` length. It bounds the
# size of a skill (skills are injected into the actor's prompt), not the size
# of one change: a create or an edit whose RESULT exceeds it is rejected with
# a request to condense, and the rejection costs no budget.
SKILL_MAX_INSTRUCTIONS_CHARS = int(os.getenv("SKILL_MAX_INSTRUCTIONS_CHARS", "12000"))


def _skill_target_instructions_chars(raw: Optional[str], cap: int) -> int:
    """``MIRIX_SKILL_TARGET_INSTRUCTIONS_CHARS`` as a positive int, at most the
    hard cap; half the cap by default (with a warning when invalid)."""
    value = _positive_int_setting(
        "MIRIX_SKILL_TARGET_INSTRUCTIONS_CHARS", raw, max(1, cap // 2)
    )
    if value > cap:
        logging.getLogger(__name__).warning(
            "MIRIX_SKILL_TARGET_INSTRUCTIONS_CHARS=%r is above the hard cap "
            "SKILL_MAX_INSTRUCTIONS_CHARS (%d); using the cap",
            raw,
            cap,
        )
        return cap
    return value


# A skill's soft size target: a create or an instructions edit whose
# result is longer goes through, and its answer (and skill_read, skill_list)
# asks the curator to condense the skill or split out a separate kind of task
# in a later change. A retrieved skill is injected whole, and find-and-replace
# patches only ever add text: without a target short of the hard cap, skills
# grow to it. Half the cap: in logged learning runs skills of 1.4k-4.4k
# characters helped the agents that received them, and harm co-occurred only
# with skills of 5.9k-12k (confounded with other causes), so the target flags
# that range and leaves room to condense before the hard cap.
SKILL_TARGET_INSTRUCTIONS_CHARS = _skill_target_instructions_chars(
    os.getenv("MIRIX_SKILL_TARGET_INSTRUCTIONS_CHARS"), SKILL_MAX_INSTRUCTIONS_CHARS
)

# ---------------------------------------------------------------------------
# Procedural (skill) RETRIEVAL defaults.
#
# The single authoritative default for SEARCH-SURFACE procedural retrieval
# (the REST /memory/search endpoint, the eval client, and the documented
# recommendation for the search_in_memory agent tool). When a search surface
# does NOT explicitly choose a method, procedural memory is retrieved with the
# EverOS-aligned "hybrid" lane: BM25 (lexical) + embedding (dense) recall fused
# with Reciprocal Rank Fusion.
#
# Override to "bm25" or "embedding" via env for latency-sensitive deployments
# or A/B experiments. Internal pipelines that PIN a method explicitly (evolve
# before/after snapshots, the per-step system-context prefetch) keep their
# explicit choice and are not governed by this default.
PROCEDURAL_DEFAULT_SEARCH_METHOD = os.getenv("MIRIX_SKILL_SEARCH_METHOD", "hybrid")

# The distiller reads a session whole, sized by the context budget:
# there is no character cap on its transcript any more. At most this many
# distiller LLM calls run at once within one learning run, across its sessions
# AND the chunks of a long session; at most this many sessions
# are being distilled at once, too.
DISTILLER_MAX_CONCURRENCY = max(1, int(os.getenv("MIRIX_DISTILLER_MAX_CONCURRENCY", "4")))
# A session whose distillation failed this many times (truncated after every
# halving, unparseable after every repair prompt, timed out after its retry,
# ...) gets distill_failed_at and is never selected again.
# Infrastructure failures never count (they back off instead).
DISTILL_MAX_FAILURES = 3

# Retention window (days) for VERBATIM conversation turns in the
# conversation_message store AFTER they have been distilled. Once distilled,
# a turn's learning value lives in skill_experience rows; keeping the raw
# transcript indefinitely is a pure PII liability. Pruning runs at the end of
# each procedural distillation pass. Set to 0 (or negative) to keep forever.
CONVERSATION_RETENTION_DAYS = int(os.getenv("MIRIX_CONVERSATION_RETENTION_DAYS", "30"))

# Reciprocal Rank Fusion constant for procedural retrieval (k=60).
# k=60 is the canonical RRF constant (everalgo.rank.fusion.rrf /
# RankConfig.rrf_k=60); the fusion is rank-based, 1-based, and unweighted —
# raw BM25/cosine scores are discarded, only rank positions feed the sum
# 1/(k+rank).
SKILL_HYBRID_RRF_K = int(os.getenv("MIRIX_SKILL_HYBRID_RRF_K", "60"))

# Over-fetch multiplier for each lane of the ORG-WIDE hybrid search
# (list_procedures_by_org, /memory/search_all_users; EverOS
# DEFAULT_RECALL_MULTIPLIER=2): each lane recalls limit*multiplier candidates so
# fusion picks from a larger, higher-recall pool, then the result is sliced to
# the requested limit. Per-user retrieval (retrieve_skills) ranks
# every candidate instead and does not use it.
SKILL_HYBRID_RECALL_MULTIPLIER = int(
    os.getenv("MIRIX_SKILL_HYBRID_RECALL_MULTIPLIER", "2")
)

# ---------------------------------------------------------------------------
# Procedural retrieval pipeline: retrieve_skills in
# mirix/services/skill_retrieval.py.
# ---------------------------------------------------------------------------
# Okapi BM25 over a skill's retrieval_tsv (name and triggers at weight A,
# description at weight B): a position of weight A counts 1.0 towards its
# term's frequency and the document length, a position of weight B counts 0.4.
SKILL_BM25_K1 = 1.2
SKILL_BM25_B = 0.75
SKILL_BM25_WEIGHT_A = 1.0
SKILL_BM25_WEIGHT_B = 0.4
# Above this many candidates only the skills that contain at least one query
# term (GIN prefilter on Postgres) are scored; N, df and the average document
# length are still aggregated over every candidate, so IDF is not distorted.
SKILL_BM25_MAX_DOCS = _positive_int_setting(
    "MIRIX_SKILL_BM25_MAX_DOCS", os.getenv("MIRIX_SKILL_BM25_MAX_DOCS"), 20000
)
# At most this many distinct query terms are scored: the lexemes in the order
# they first appear in the query. An extracted topic has a handful; the raw
# conversation used when extraction fails (up to
# RETRIEVAL_FALLBACK_QUERY_MAX_CHARS) has thousands, and scoring every one of
# them against every candidate made one retrieval cost seconds of database and
# Python time at a few thousand skills. The terms dropped are counted in
# the ranking's diagnostics.
SKILL_BM25_MAX_QUERY_TERMS = _positive_int_setting(
    "MIRIX_SKILL_BM25_MAX_QUERY_TERMS",
    os.getenv("MIRIX_SKILL_BM25_MAX_QUERY_TERMS"),
    64,
)


def _cosine_setting(name: str, raw: Optional[str], default: float) -> float:
    """``raw`` as a cosine similarity in [-1, 1]; the default (with a warning)
    otherwise."""
    try:
        value = float(raw) if raw is not None and str(raw).strip() else default
    except ValueError:
        value = float("nan")
    if not -1.0 <= value <= 1.0:
        logging.getLogger(__name__).warning(
            "%s=%r is not a cosine similarity in [-1, 1]; using the default %s",
            name,
            raw,
            default,
        )
        return default
    return value


# The retrieval gate, applied only where retrieved skills are injected
# into a context (apply_gate=True): a skill is kept only when its cosine
# similarity to the query is at least MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY AND
# at least MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO x the best cosine among all
# candidates.
#
# Defaults calibrated for gemini-embedding-001 (synthetic calibration set in
# scripts/calibration/q10/): on 30 neutral skills x 60 queries + 16 queries
# no skill matches, the relevant skill's cosine was >= 0.656 (p5 0.718), a
# non-relevant one's <= 0.672 (p99 0.604), and an unmatched query's best
# <= 0.6015. A 0.62 floor keeps every relevant skill and lets no skill through
# for any unmatched query (0.60 let one through).
#
# The ratio was calibrated only where it is easy: every calibration query has
# exactly ONE relevant skill, in a set of unrelated domains. There it decides
# nothing in 0.85-0.95 (a non-relevant skill's ratio to the best is <= 0.776
# at p99; 0.80 lets 0.05 per query through), so 0.90 stays: it drops
# near-duplicates of a clearly better skill. Its effect when several skills of
# ONE domain are relevant was never calibrated. There it can also drop a less
# specific relevant skill: a domain's general skill may score a lower fraction
# of the best cosine than specific skills for another kind of task of the same
# domain, and no ratio or floor separates the two, because cosine over the
# retrieval text cannot tell the task types of one domain apart. That is fixed
# where it arises (the skills' retrieval text, the task-level query of
# topic_extraction), not with this number, which stays at the only value the
# calibrated evidence supports.
#
# The calibration holds for gemini-embedding-001 only: another model's
# cosines may sit on another scale. SKILL_RETRIEVAL_GATE_CALIBRATIONS lists
# the models whose gate was calibrated; skill_retrieval.skill_retrieval_gate
# gives any other model these defaults as "uncalibrated" (with one warning
# per model), and every retrieval's gate trace names the calibration used.
SKILL_RETRIEVAL_MIN_SIMILARITY_DEFAULT = 0.62
SKILL_RETRIEVAL_RELATIVE_RATIO_DEFAULT = 0.90
SKILL_RETRIEVAL_MIN_SIMILARITY = _cosine_setting(
    "MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY",
    os.getenv("MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY"),
    SKILL_RETRIEVAL_MIN_SIMILARITY_DEFAULT,
)
SKILL_RETRIEVAL_RELATIVE_RATIO = _fraction_setting(
    "MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO",
    os.getenv("MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO"),
    SKILL_RETRIEVAL_RELATIVE_RATIO_DEFAULT,
)
# (min_similarity, relative_ratio) of each embedding model whose gate was
# calibrated, by model name without a provider prefix (looked up like
# EMBEDDING_TOKEN_LIMITS: "google/gemini-embedding-001" is
# "gemini-embedding-001").
SKILL_RETRIEVAL_GATE_CALIBRATIONS = {
    "gemini-embedding-001": (
        SKILL_RETRIEVAL_MIN_SIMILARITY_DEFAULT,
        SKILL_RETRIEVAL_RELATIVE_RATIO_DEFAULT,
    ),
}


def _skill_retrieval_gate_from_env(
    raw_min_similarity: Optional[str], raw_relative_ratio: Optional[str]
) -> bool:
    """Whether the environment sets the gate: a valid value in
    MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY or MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO
    (an invalid one is replaced by its default, above)."""

    def number(raw: Optional[str]) -> Optional[float]:
        try:
            return float(raw) if raw is not None and str(raw).strip() else None
        except ValueError:
            return None

    min_similarity = number(raw_min_similarity)
    relative_ratio = number(raw_relative_ratio)
    return (min_similarity is not None and -1.0 <= min_similarity <= 1.0) or (
        relative_ratio is not None and 0.0 < relative_ratio <= 1.0
    )


# An operator's MIRIX_SKILL_RETRIEVAL_* setting wins for every embedding
# model: the two settings above then apply as they are ("env").
SKILL_RETRIEVAL_GATE_FROM_ENV = _skill_retrieval_gate_from_env(
    os.getenv("MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY"),
    os.getenv("MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO"),
)
# At most this many skills are returned by default.
SKILL_RETRIEVAL_DEFAULT_LIMIT = 10

# ---------------------------------------------------------------------------
# The experience pool of the skill curator.
#
# A curator batch that finishes keeps its experiences that no skill used and
# that it found deferred or not worth a skill (disposition) in the pool of
# their (organization, user, meta agent) and write scope. When a later batch
# is assembled, the pool's experiences whose max cosine to one of the batch's
# new experiences is >= MIRIX_SKILL_EXPERIENCE_HISTORY_MIN_SIMILARITY are
# ranked by that cosine x SKILL_EXPERIENCE_POOL_DECAY ** ignored_pushes and
# the first MIRIX_SKILL_EXPERIENCE_HISTORY_TOP_K are shown as "earlier
# lessons" (they cost context, never edit budget). A pool experience whose
# cosine is >= MIRIX_SKILL_EXPERIENCE_RECURRENCE_SIMILARITY recurred: its
# recurrence_count grows and its ignored_pushes resets. A shown one the batch
# neither cites nor covers is ignored (+1, +2 when listed not worth, +1 when
# listed deferred); at SKILL_EXPERIENCE_POOL_RETIRE_IGNORED it leaves the pool.
# A new experience that recurred a lesson still in the pool is merged into it
# instead of entering the pool twice, and a pool experience whose embedding
# failed in SKILL_EXPERIENCE_POOL_EMBED_MAX_FAILURES finished batches (while
# their own experiences embedded) leaves it as unembeddable.
#
# Both similarity defaults are PROVISIONAL: the retrieval-gate calibration of
# gemini-embedding-001 (queries vs skills across unrelated domains: relevant
# median 0.787, p75 0.827; non-relevant p99 0.604, max 0.672) is only a weak
# anchor for lesson-vs-lesson similarity within one domain. Every batch logs
# its candidates' max cosines (curator_metrics[].history) for calibration.
# ---------------------------------------------------------------------------
SKILL_EXPERIENCE_POOL_ENABLED = os.getenv(
    "MIRIX_SKILL_EXPERIENCE_POOL_ENABLED", "true"
).strip().lower() in ("true", "1", "yes")
SKILL_EXPERIENCE_HISTORY_MIN_SIMILARITY = _cosine_setting(
    "MIRIX_SKILL_EXPERIENCE_HISTORY_MIN_SIMILARITY",
    os.getenv("MIRIX_SKILL_EXPERIENCE_HISTORY_MIN_SIMILARITY"),
    0.60,
)
SKILL_EXPERIENCE_RECURRENCE_SIMILARITY = _cosine_setting(
    "MIRIX_SKILL_EXPERIENCE_RECURRENCE_SIMILARITY",
    os.getenv("MIRIX_SKILL_EXPERIENCE_RECURRENCE_SIMILARITY"),
    0.80,
)
if SKILL_EXPERIENCE_RECURRENCE_SIMILARITY < SKILL_EXPERIENCE_HISTORY_MIN_SIMILARITY:
    # Allowed, but a lesson that recurred may then not be shown.
    logging.getLogger(__name__).warning(
        "MIRIX_SKILL_EXPERIENCE_RECURRENCE_SIMILARITY (%s) is below "
        "MIRIX_SKILL_EXPERIENCE_HISTORY_MIN_SIMILARITY (%s): a lesson that "
        "recurred may fall below the threshold for being shown",
        SKILL_EXPERIENCE_RECURRENCE_SIMILARITY,
        SKILL_EXPERIENCE_HISTORY_MIN_SIMILARITY,
    )
SKILL_EXPERIENCE_HISTORY_TOP_K = _positive_int_setting(
    "MIRIX_SKILL_EXPERIENCE_HISTORY_TOP_K",
    os.getenv("MIRIX_SKILL_EXPERIENCE_HISTORY_TOP_K"),
    10,
)
# Active pool experiences read per batch, newest first (a bound on the vectors
# loaded; the cut is counted as pool_scan_truncated).
SKILL_EXPERIENCE_POOL_SCAN_LIMIT = _positive_int_setting(
    "MIRIX_SKILL_EXPERIENCE_POOL_SCAN_LIMIT",
    os.getenv("MIRIX_SKILL_EXPERIENCE_POOL_SCAN_LIMIT"),
    2000,
)
SKILL_EXPERIENCE_POOL_DECAY = 0.5
SKILL_EXPERIENCE_POOL_RETIRE_IGNORED = 3
# A pool experience listed as not worth a skill counts as this many ignored
# pushes; a new one enters the pool already decayed once.
SKILL_EXPERIENCE_NOT_WORTH_PENALTY = 2
SKILL_EXPERIENCE_NOT_WORTH_START = 1
# Finished batches in which a pool experience's own embedding failed (while
# the batch's new experiences embedded, so the model was up) before it leaves
# the pool as unembeddable: a text the embedding model keeps refusing would
# otherwise be retried, and left out of the ranking, in every later batch.
SKILL_EXPERIENCE_POOL_EMBED_MAX_FAILURES = 3
# Candidates whose max cosine a batch's history metrics keep (the rest are
# summarized in quantiles); also the most ids any list of them names.
SKILL_EXPERIENCE_HISTORY_LOG_LIMIT = 50

# ---------------------------------------------------------------------------
# Embedding length guard.
# ---------------------------------------------------------------------------
# Input token limit per embedding model. A model missing from the table is
# treated as DEFAULT_EMBEDDING_TOKEN_LIMIT (with a warning). The model's own
# limit, never EmbeddingConfig.embedding_chunk_size (300 by default: far too
# small). MIRIX_EMBEDDING_TOKEN_LIMITS (a JSON object, e.g.
# '{"gemini-embedding-001": 2048, "text-embedding-3-small": 8191}') replaces
# the table. A provider prefix is ignored when looking a model up
# ("google/gemini-embedding-001" is "gemini-embedding-001").
DEFAULT_EMBEDDING_TOKEN_LIMITS = {"gemini-embedding-001": 2048}
DEFAULT_EMBEDDING_TOKEN_LIMIT = 2048
# A chunk holds at most this fraction of the model's limit, rounded down to a
# whole hundred tokens (gemini-embedding-001: 2,048 -> 1,800): counts are
# o200k_base approximations of the provider's own tokenizer.
EMBEDDING_CHUNK_FRACTION = 0.88


def _embedding_token_limits_setting(raw: Optional[str]) -> dict:
    """``MIRIX_EMBEDDING_TOKEN_LIMITS`` as {model: positive int}; the default
    table when it is unset. Invalid JSON keeps the default; an invalid entry
    is dropped; each with a warning."""
    import json

    log = logging.getLogger(__name__)
    if raw is None or not str(raw).strip():
        return dict(DEFAULT_EMBEDDING_TOKEN_LIMITS)
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    if not isinstance(value, dict):
        log.warning(
            "MIRIX_EMBEDDING_TOKEN_LIMITS=%r is not a JSON object of "
            "{model: token limit}; using the default %s",
            raw,
            DEFAULT_EMBEDDING_TOKEN_LIMITS,
        )
        return dict(DEFAULT_EMBEDDING_TOKEN_LIMITS)
    limits = {}
    for model, limit in value.items():
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            log.warning(
                "MIRIX_EMBEDDING_TOKEN_LIMITS: %r needs a positive integer token "
                "limit, not %r; entry ignored",
                model,
                limit,
            )
            continue
        limits[str(model)] = limit
    return limits


EMBEDDING_TOKEN_LIMITS = _embedding_token_limits_setting(
    os.getenv("MIRIX_EMBEDDING_TOKEN_LIMITS")
)
# At most this many chunks are embedded for one text (one provider call each):
# a longer text is represented by its first and last chunks (half each), with
# a warning, so one request can never cost thousands of sequential paid
# embedding calls (a 10 MB input would be ~1,400 chunks).
EMBEDDING_MAX_CHUNKS = _positive_int_setting(
    "MIRIX_EMBEDDING_MAX_CHUNKS", os.getenv("MIRIX_EMBEDDING_MAX_CHUNKS"), 16
)
# Skills re-embedded at once by the startup pass and
# scripts/reembed_skill_retrieval.py (bounded concurrency).
SKILL_REEMBED_CONCURRENCY = _positive_int_setting(
    "MIRIX_SKILL_REEMBED_CONCURRENCY", os.getenv("MIRIX_SKILL_REEMBED_CONCURRENCY"), 4
)
# The re-embedding pass at the end of a learning run stops after this many
# skills in a row failed to embed (each already retried 3 times): during
# a sustained embedding outage it would otherwise spend 4 calls and ~7 s of
# backoff on every skill of the user still missing a vector, holding the run
# (and its lease) for minutes. The skills it did not try are counted as
# "skipped" on the run's reembed entry and left for the next pass. The
# startup pass and scripts/reembed_skill_retrieval.py try
# every skill.
SKILL_REEMBED_MAX_CONSECUTIVE_FAILURES = _positive_int_setting(
    "MIRIX_SKILL_REEMBED_MAX_CONSECUTIVE_FAILURES",
    os.getenv("MIRIX_SKILL_REEMBED_MAX_CONSECUTIVE_FAILURES"),
    3,
)

# ---------------------------------------------------------------------------
# Conversation retrieval and its log.
# ---------------------------------------------------------------------------
# One retrieval_log row per /memory/retrieve/* call (its id is returned as
# retrieval_id), unless MIRIX_RETRIEVAL_LOG_ENABLED is false. Rows older than
# MIRIX_RETRIEVAL_LOG_RETENTION_DAYS are purged by a background task at startup
# and every RETRIEVAL_LOG_PURGE_INTERVAL_SECONDS; <= 0 days keeps them forever.
# Erasing a user's (or a client's) memories deletes their rows at once.
RETRIEVAL_LOG_ENABLED = os.getenv("MIRIX_RETRIEVAL_LOG_ENABLED", "true").strip().lower() in (
    "true",
    "1",
    "yes",
)


def _int_setting(name: str, raw: Optional[str], default: int) -> int:
    """``raw`` as an int (any sign); the default (with a warning) otherwise."""
    try:
        return int(raw) if raw is not None and str(raw).strip() else default
    except ValueError:
        logging.getLogger(__name__).warning(
            "%s=%r is not an integer; using the default %d", name, raw, default
        )
        return default


RETRIEVAL_LOG_RETENTION_DAYS = _int_setting(
    "MIRIX_RETRIEVAL_LOG_RETENTION_DAYS",
    os.getenv("MIRIX_RETRIEVAL_LOG_RETENTION_DAYS"),
    30,
)
RETRIEVAL_LOG_PURGE_INTERVAL_SECONDS = 3600
# Rows deleted per statement by the retention purge.
RETRIEVAL_LOG_PURGE_BATCH = 5000
# When topic extraction fails, the conversation's own text is the query
# (extraction_status "fallback"). A text longer than this keeps its head
# and tail: the query feeds full-text search and one embedding call, and a
# Postgres tsvector cannot exceed 1 MB.
RETRIEVAL_FALLBACK_QUERY_MAX_CHARS = 50_000
# The other memory types (episodic, semantic, resource, knowledge vault) search
# the fallback text by keywords only: a few hundred characters (head and tail)
# are a query; tens of thousands build a huge AND / prefix-OR tsquery.
RETRIEVAL_FALLBACK_KEYWORD_QUERY_MAX_CHARS = 500
# The text a local (Ollama) extraction model receives, head and tail kept:
# its context window is unknown to MIRIX.
RETRIEVAL_LOCAL_EXTRACTION_MAX_CHARS = _positive_int_setting(
    "MIRIX_RETRIEVAL_LOCAL_EXTRACTION_MAX_CHARS",
    os.getenv("MIRIX_RETRIEVAL_LOCAL_EXTRACTION_MAX_CHARS"),
    12_000,
)
# A retrieval log row keeps the raw query (head and tail, with a marker) up to
# this many characters, and the procedural trace keeps every returned skill
# plus the next RETRIEVAL_LOG_TRACE_CANDIDATES candidates in fused order —
# not every candidate — with the counts of what was left out: one row stays a
# few kilobytes however many skills a user has.
RETRIEVAL_LOG_QUERY_MAX_CHARS = 50_000
RETRIEVAL_LOG_TRACE_CANDIDATES = 50
