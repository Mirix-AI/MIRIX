import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file if it exists before initializing settings
# This ensures environment variables from .env are available when settings are instantiated
load_dotenv()


class ToolSettings(BaseSettings):
    composio_api_key: Optional[str] = None

    # E2B Sandbox configurations
    e2b_api_key: Optional[str] = None
    e2b_sandbox_template_id: Optional[str] = None  # Updated manually

    # Local Sandbox configurations
    local_sandbox_dir: Optional[str] = None


class SummarizerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="mirix_summarizer_", extra="ignore")

    # Controls if we should evict all messages
    # TODO: Can refactor this into an enum if we have a bunch of different kinds of summarizers
    evict_all_messages: bool = False

    # The maximum number of retries for the summarizer
    # If we reach this cutoff, it probably means that the summarizer is not compressing down the in-context messages any further
    # And we throw a fatal error
    max_summarizer_retries: int = 3

    # When to warn the model that a summarize command will happen soon
    # The amount of tokens before a system warning about upcoming truncation is sent to Mirix
    memory_warning_threshold: float = 0.75

    # Whether to send the system memory warning message
    send_memory_warning_message: bool = False

    # The desired memory pressure to summarize down to
    desired_memory_token_pressure: float = 0.1

    # The number of messages at the end to keep
    # Even when summarizing, we may want to keep a handful of recent messages
    # These serve as in-context examples of how to use functions / what user messages look like
    keep_last_n_messages: int = 5


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # env_prefix='my_prefix_'

    # openai
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"

    # groq
    groq_api_key: Optional[str] = None

    # Bedrock
    aws_access_key: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    bedrock_anthropic_version: Optional[str] = "bedrock-2023-05-31"

    # anthropic
    anthropic_api_key: Optional[str] = None

    # ollama
    ollama_base_url: Optional[str] = None

    # azure
    azure_api_key: Optional[str] = None
    azure_base_url: Optional[str] = None
    # We provide a default here, since usually people will want to be on the latest API version.
    azure_api_version: Optional[str] = (
        "2024-09-01-preview"  # https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation
    )

    # google ai
    gemini_api_key: Optional[str] = None

    # together
    together_api_key: Optional[str] = None

    # vLLM
    vllm_api_base: Optional[str] = None

    # openllm
    openllm_auth_type: Optional[str] = None
    openllm_api_key: Optional[str] = None

    # disable openapi schema generation
    disable_schema_generation: bool = False


cors_origins = [
    "http://mirix.localhost",
    "http://localhost:8283",
    "http://localhost:8083",
    "http://localhost:3000",
    "http://localhost:4200",
]

# read pg_uri from ~/.mirix/pg_uri or set to none, this is to support Mirix Desktop
default_pg_uri = None

## check if --use-file-pg-uri is passed
if "--use-file-pg-uri" in sys.argv:
    try:
        with open(Path.home() / ".mirix/pg_uri", "r") as f:
            default_pg_uri = f.read()
            # Note: Using print instead of logger to avoid circular import with mirix.log
            print("Read pg_uri from ~/.mirix/pg_uri")
    except FileNotFoundError:
        pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="mirix_", extra="ignore")

    mirix_dir: Optional[Path] = Field(Path.home() / ".mirix", env="MIRIX_DIR")
    # Directory where uploaded/processed images are stored
    # Can be overridden with MIRIX_IMAGES_DIR environment variable
    images_dir: Optional[Path] = Field(Path.home() / ".mirix" / "images", env="MIRIX_IMAGES_DIR")
    debug: Optional[bool] = False
    cors_origins: Optional[list] = cors_origins

    # database configuration
    pg_db: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    pg_host: Optional[str] = None
    pg_port: Optional[int] = None
    pg_uri: Optional[str] = Field(default_pg_uri, env="MIRIX_PG_URI")  # option to specify full uri
    pg_pool_size: int = 80  # Concurrent connections
    pg_max_overflow: int = 30  # Overflow limit
    pg_pool_timeout: int = 30  # Seconds to wait for a connection
    pg_pool_recycle: int = 1800  # When to recycle connections
    pg_echo: bool = False  # Logging

    # Redis configuration (optional - for caching and search acceleration)
    redis_enabled: bool = Field(False, env="MIRIX_REDIS_ENABLED")  # Master switch
    redis_host: Optional[str] = Field(None, env="MIRIX_REDIS_HOST")
    redis_port: int = Field(6379, env="MIRIX_REDIS_PORT")
    redis_db: int = Field(0, env="MIRIX_REDIS_DB")
    redis_password: Optional[str] = Field(None, env="MIRIX_REDIS_PASSWORD")
    redis_uri: Optional[str] = Field(None, env="MIRIX_REDIS_URI")  # Full URI override

    # Redis connection pool settings (optimized for production)
    redis_max_connections: int = Field(50, env="MIRIX_REDIS_MAX_CONNECTIONS")  # Per container
    redis_socket_timeout: int = Field(5, env="MIRIX_REDIS_SOCKET_TIMEOUT")  # Read/write timeout (seconds)
    redis_socket_connect_timeout: int = Field(5, env="MIRIX_REDIS_SOCKET_CONNECT_TIMEOUT")  # Connect timeout (seconds)
    redis_socket_keepalive: bool = Field(True, env="MIRIX_REDIS_SOCKET_KEEPALIVE")  # Enable TCP keepalive
    redis_retry_on_timeout: bool = Field(True, env="MIRIX_REDIS_RETRY_ON_TIMEOUT")  # Retry on timeout errors

    # Redis TTL settings (cache expiration times in seconds)
    redis_ttl_default: int = Field(3600, env="MIRIX_REDIS_TTL_DEFAULT")  # 1 hour default TTL
    redis_ttl_blocks: int = Field(7200, env="MIRIX_REDIS_TTL_BLOCKS")  # 2 hours for hot data (blocks)
    redis_ttl_messages: int = Field(7200, env="MIRIX_REDIS_TTL_MESSAGES")  # 2 hours for messages
    redis_ttl_organizations: int = Field(43200, env="MIRIX_REDIS_TTL_ORGANIZATIONS")  # 12 hours for organizations
    redis_ttl_users: int = Field(43200, env="MIRIX_REDIS_TTL_USERS")  # 12 hours for users
    redis_ttl_clients: int = Field(43200, env="MIRIX_REDIS_TTL_CLIENTS")  # 12 hours for clients
    redis_ttl_agents: int = Field(43200, env="MIRIX_REDIS_TTL_AGENTS")  # 12 hours for agents
    redis_ttl_tools: int = Field(43200, env="MIRIX_REDIS_TTL_TOOLS")  # 12 hours for tools

    @property
    def mirix_redis_uri(self) -> Optional[str]:
        """Construct Redis URI from components or return explicit URI."""
        if not self.redis_enabled:
            return None

        if self.redis_uri:
            return self.redis_uri
        elif self.redis_host:
            auth = f":{self.redis_password}@" if self.redis_password else ""
            return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
        else:
            return None

    # multi agent settings
    multi_agent_send_message_max_retries: int = 3
    multi_agent_send_message_timeout: int = 20 * 60
    multi_agent_concurrent_sends: int = 50

    # telemetry logging
    verbose_telemetry_logging: bool = False
    otel_exporter_otlp_endpoint: Optional[str] = None  # otel default: "http://localhost:4317"
    disable_tracing: bool = False

    # uvicorn settings
    uvicorn_workers: int = 1
    uvicorn_reload: bool = False
    uvicorn_timeout_keep_alive: int = 5

    # event loop parallelism
    event_loop_threadpool_max_workers: int = 43

    # experimental toggle
    use_experimental: bool = False

    # logging configuration
    log_level: str = Field("INFO", env="MIRIX_LOG_LEVEL")
    log_file: Optional[Path] = Field(None, env="MIRIX_LOG_FILE")  # If set, enables file logging
    log_to_console: bool = Field(True, env="MIRIX_LOG_TO_CONSOLE")  # Console logging is default
    log_max_bytes: int = Field(10 * 1024 * 1024, env="MIRIX_LOG_MAX_BYTES")  # 10 MB
    log_backup_count: int = Field(5, env="MIRIX_LOG_BACKUP_COUNT")

    # LLM provider client settings
    httpx_max_retries: int = 5
    httpx_timeout_connect: float = 10.0
    httpx_timeout_read: float = 60.0
    httpx_timeout_write: float = 30.0
    httpx_timeout_pool: float = 10.0
    httpx_max_connections: int = 500
    httpx_max_keepalive_connections: int = 500
    httpx_keepalive_expiry: float = 120.0

    # LLM retry settings (for agent-level retries on transient errors)
    llm_retry_limit: int = Field(3, env="MIRIX_LLM_RETRY_LIMIT")  # Max retry attempts for LLM calls
    llm_retry_backoff_factor: float = Field(0.5, env="MIRIX_LLM_RETRY_BACKOFF_FACTOR")  # Exponential backoff multiplier
    llm_retry_max_delay: float = Field(10.0, env="MIRIX_LLM_RETRY_MAX_DELAY")  # Max delay between retries (seconds)

    # cron job parameters
    enable_batch_job_polling: bool = False
    poll_running_llm_batches_interval_seconds: int = 5 * 60

    # LangFuse observability settings (for distributed tracing)
    langfuse_enabled: bool = Field(False, env="MIRIX_LANGFUSE_ENABLED")
    langfuse_public_key: Optional[str] = Field(None, env="MIRIX_LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(None, env="MIRIX_LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", env="MIRIX_LANGFUSE_HOST")
    langfuse_flush_interval: float = Field(1.0, env="MIRIX_LANGFUSE_FLUSH_INTERVAL")  # seconds
    langfuse_flush_at: int = Field(512, env="MIRIX_LANGFUSE_FLUSH_AT")  # spans per batch
    langfuse_debug: bool = Field(False, env="MIRIX_LANGFUSE_DEBUG")
    langfuse_flush_timeout: float = Field(10.0, env="MIRIX_LANGFUSE_FLUSH_TIMEOUT")  # seconds
    # Environment identifier for filtering traces in shared Langfuse projects
    # Common values: "dev", "e2e", "qal", "prf", "prod"
    # Must match regex: ^(?!langfuse)[a-z0-9-_]+$ with max 40 chars
    langfuse_environment: str = Field("dev", env="MIRIX_LANGFUSE_ENVIRONMENT")

    # JWT settings for dashboard authentication
    jwt_secret_key: Optional[str] = Field(None, env="MIRIX_JWT_SECRET_KEY")
    jwt_expiration_hours: int = Field(24, env="MIRIX_JWT_EXPIRATION_HOURS")

    @property
    def mirix_pg_uri(self) -> str:
        if self.pg_uri:
            return self.pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return "postgresql+pg8000://mirix:mirix@localhost:5432/mirix"

    # add this property to avoid being returned the default
    # reference: https://github.com/mirix-ai/mirix/issues/1362
    @property
    def mirix_pg_uri_no_default(self) -> str:
        if self.pg_uri:
            return self.pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return None


class TestSettings(Settings):
    model_config = SettingsConfigDict(env_prefix="mirix_test_", extra="ignore")

    mirix_dir: Optional[Path] = Field(Path.home() / ".mirix/test", env="MIRIX_TEST_DIR")
    images_dir: Optional[Path] = Field(Path.home() / ".mirix/test" / "images", env="MIRIX_TEST_IMAGES_DIR")


# singleton
settings = Settings(_env_parse_none_str="None")
test_settings = TestSettings()
model_settings = ModelSettings()
tool_settings = ToolSettings()
summarizer_settings = SummarizerSettings()
