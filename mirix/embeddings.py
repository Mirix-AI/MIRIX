import uuid
from typing import Any, List, Optional

import numpy as np
import tiktoken

from mirix.constants import (
    EMBEDDING_TO_TOKENIZER_DEFAULT,
    EMBEDDING_TO_TOKENIZER_MAP,
    MAX_EMBEDDING_DIM,
)
from mirix.llm_api.llm_api_tools import retry_with_exponential_backoff
from mirix.log import get_logger
from mirix.observability.context import get_trace_context, mark_observation_as_child
from mirix.observability.langfuse_client import get_langfuse_client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.utils import is_valid_url, printd

logger = get_logger(__name__)


def is_embedding_tracing_enabled() -> bool:
    """Check if Langfuse tracing is available and active for embeddings."""
    langfuse = get_langfuse_client()
    if not langfuse:
        return False
    trace_context = get_trace_context()
    return trace_context is not None and trace_context.get("trace_id") is not None


def embedding_with_retry(embedding_func) -> List[float]:
    """Execute embedding function with retry logic (no tracing)."""
    from mirix.settings import settings

    wrapped = retry_with_exponential_backoff(
        embedding_func,
        initial_delay=settings.llm_retry_backoff_factor,
        max_retries=settings.llm_retry_limit,
        error_codes=(429, 500, 502, 503, 504),
    )
    return wrapped()


def traced_embedding_with_retry(
    model: str, provider: str, text: str, embedding_func, endpoint: Optional[str] = None
) -> List[float]:
    """
    Execute embedding with Langfuse tracing and retry logic.

    Each retry attempt is traced individually so failed attempts are visible.
    Langfuse failures are handled gracefully - they won't break embedding functionality.

    Only call this when tracing is enabled (check with is_embedding_tracing_enabled()).
    """
    from typing import cast

    from langfuse.types import TraceContext

    from mirix.settings import settings

    langfuse = get_langfuse_client()
    trace_context = get_trace_context()
    trace_id = trace_context.get("trace_id")
    parent_span_id = trace_context.get("observation_id")

    # Optional:runcate input text for trace (avoid huge payloads)
    # text_preview = text[:500] + "..." if len(text) > 500 else text

    # Or, use the full text for the trace
    text_preview = text

    # Build trace context
    trace_context_dict: dict = {"trace_id": trace_id}
    if parent_span_id:
        trace_context_dict["parent_span_id"] = parent_span_id

    def traced_call() -> List[float]:
        """Execute a single embedding attempt with tracing."""
        # Try to start Langfuse observation - if this fails, execute without tracing
        try:
            observation_context = langfuse.start_as_current_observation(
                name="embedding",
                as_type="embedding",
                trace_context=cast(TraceContext, trace_context_dict),
                model=model,
                input={"text": text_preview},
                metadata={
                    "provider": provider,
                    "endpoint": endpoint,
                    "input_length": len(text),
                },
            )
        except Exception as e:
            # Langfuse failed to start observation - execute without tracing
            logger.error(f"Langfuse failed to start observation. Continuing without tracing: {e}")
            return embedding_func()

        # Now execute with tracing - embedding errors propagate normally
        with observation_context as generation:
            mark_observation_as_child(generation)

            try:
                result = embedding_func()
                # Try to update trace with success - don't fail if this errors
                try:
                    generation.update(
                        output={"embedding_dim": len(result) if result else 0},
                        metadata={
                            "provider": provider,
                            "endpoint": endpoint,
                            "input_length": len(text),
                            "output_dim": len(result) if result else 0,
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to update LangFuse trace with success. Continuing without tracing: {e}")
                return result
            except Exception as e:
                # Update trace with error before re-raising
                try:
                    generation.update(
                        level="ERROR",
                        status_message=str(e),
                        metadata={
                            "provider": provider,
                            "endpoint": endpoint,
                            "input_length": len(text),
                            "error_type": type(e).__name__,
                        },
                    )
                except Exception as langfuse_err:
                    logger.error(f"Failed to update LangFuse trace with error. Continuing: {langfuse_err}")
                    pass  # Don't let Langfuse errors mask the real error
                raise  # Re-raise the embedding error

    # Wrap traced call with retry logic
    wrapped = retry_with_exponential_backoff(
        traced_call,
        initial_delay=settings.llm_retry_backoff_factor,
        max_retries=settings.llm_retry_limit,
        error_codes=(429, 500, 502, 503, 504),
    )
    return wrapped()


def parse_and_chunk_text(text: str, chunk_size: int) -> List[str]:
    from llama_index.core import Document as LlamaIndexDocument
    from llama_index.core.node_parser import SentenceSplitter

    parser = SentenceSplitter(chunk_size=chunk_size)
    llama_index_docs = [LlamaIndexDocument(text=text)]
    nodes = parser.get_nodes_from_documents(llama_index_docs)
    return [n.text for n in nodes]


def truncate_text(text: str, max_length: int, encoding) -> str:
    # truncate the text based on max_length and encoding
    encoded_text = encoding.encode(text)[:max_length]
    return encoding.decode(encoded_text)


def check_and_split_text(text: str, embedding_model: str) -> List[str]:
    """Split text into chunks of max_length tokens or less"""

    if embedding_model in EMBEDDING_TO_TOKENIZER_MAP:
        encoding = tiktoken.get_encoding(EMBEDDING_TO_TOKENIZER_MAP[embedding_model])
    else:
        logger.debug(
            f"Warning: couldn't find tokenizer for model {embedding_model}, using default tokenizer {EMBEDDING_TO_TOKENIZER_DEFAULT}"
        )
        encoding = tiktoken.get_encoding(EMBEDDING_TO_TOKENIZER_DEFAULT)

    num_tokens = len(encoding.encode(text))

    # determine max length
    if hasattr(encoding, "max_length"):
        # TODO(fix) this is broken
        max_length = encoding.max_length
    else:
        # TODO: figure out the real number
        printd(f"Warning: couldn't find max_length for tokenizer {embedding_model}, using default max_length 8191")
        max_length = 8191

    # truncate text if too long
    if num_tokens > max_length:
        logger.debug(f"Warning: text is too long ({num_tokens} tokens), truncating to {max_length} tokens.")
        # Truncate the text
        text = truncate_text(text, max_length, encoding)

    return [text]


class OpenAIEmbeddingWithCustomAuth:
    """OpenAI embedding client with auth provider support using sync OpenAI SDK."""

    def __init__(self, config: EmbeddingConfig, auth_provider: str):
        """
        Initialize OpenAI embedding client with auth provider support.

        Args:
            config: EmbeddingConfig with auth_provider field set
            api_key: OpenAI API key
        """
        self.config = config
        self.model = config.embedding_model
        self.auth_provider = auth_provider

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding with dynamic auth headers from auth provider.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If auth provider is not found in registry
        """

        def _do_embedding():
            from openai import OpenAI

            from mirix.llm_api.auth_provider import get_auth_provider

            # Get auth headers from provider (sync)
            auth_provider = get_auth_provider(self.auth_provider)
            if not auth_provider:
                raise ValueError(
                    f"Auth provider '{self.config.auth_provider}' not found in registry. "
                    "Make sure to register it before using."
                )

            try:
                auth_headers = auth_provider.get_auth_headers()  # Sync call
                logger.info(
                    f"OpenAI Embedding - Using auth provider '{self.config.auth_provider}' "
                    f"to inject {len(auth_headers)} header(s)"
                )
            except Exception as e:
                logger.error(f"Failed to get auth headers from provider '{self.config.auth_provider}': {e}")
                raise

            # Create OpenAI client with auth headers
            client = OpenAI(
                api_key="DUMMY_API_KEY",  # This is not used, but the SDK will thow an error if it is not set
                base_url=self.config.embedding_endpoint,
                default_headers=auth_headers,
            )

            # Call embeddings API
            response = client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding

        if is_embedding_tracing_enabled():
            return traced_embedding_with_retry(
                model=self.model,
                provider="openai_custom_auth",
                text=text,
                embedding_func=_do_embedding,
                endpoint=self.config.embedding_endpoint,
            )
        else:
            return embedding_with_retry(_do_embedding)


class EmbeddingEndpoint:
    """Implementation for OpenAI compatible endpoint"""

    # """ Based off llama index https://github.com/run-llama/llama_index/blob/a98bdb8ecee513dc2e880f56674e7fd157d1dc3a/llama_index/embeddings/text_embeddings_inference.py """

    # _user: str = PrivateAttr()
    # _timeout: float = PrivateAttr()
    # _base_url: str = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str,
        user: str,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        if not is_valid_url(base_url):
            raise ValueError(
                f"Embeddings endpoint was provided an invalid URL (set to: '{base_url}'). Make sure embedding_endpoint is set correctly in your Mirix config."
            )
        # TODO: find a neater solution - re-mapping for mirix endpoint
        if model == "mirix-free":
            model = "BAAI/bge-large-en-v1.5"
        self.model_name = model
        self._user = user
        self._base_url = base_url
        self._timeout = timeout

    def _call_api(self, text: str) -> List[float]:
        if not is_valid_url(self._base_url):
            raise ValueError(
                f"Embeddings endpoint does not have a valid URL (set to: '{self._base_url}'). Make sure embedding_endpoint is set correctly in your Mirix config."
            )
        import httpx

        headers = {"Content-Type": "application/json"}
        json_data = {"input": text, "model": self.model_name, "user": self._user}

        with httpx.Client() as client:
            response = client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json=json_data,
                timeout=self._timeout,
            )

        response_json = response.json()

        if isinstance(response_json, list):
            # embedding directly in response
            embedding = response_json
        elif isinstance(response_json, dict):
            # TEI embedding packaged inside openai-style response
            try:
                embedding = response_json["data"][0]["embedding"]
            except (KeyError, IndexError):
                raise TypeError(
                    f"Got back an unexpected payload from text embedding function, response=\n{response_json}"
                )
        else:
            # unknown response, can't parse
            raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")

        return embedding

    def get_text_embedding(self, text: str) -> List[float]:
        if is_embedding_tracing_enabled():
            return traced_embedding_with_retry(
                model=self.model_name,
                provider="hugging-face",
                text=text,
                embedding_func=lambda: self._call_api(text),
                endpoint=self._base_url,
            )
        else:
            return embedding_with_retry(lambda: self._call_api(text))


class AzureOpenAIEmbedding:
    def __init__(self, api_endpoint: str, api_key: str, api_version: str, model: str):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=api_endpoint)
        self.model = model

    def get_text_embedding(self, text: str):

        def _do_embedding():
            return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

        if is_embedding_tracing_enabled():
            return traced_embedding_with_retry(
                model=self.model,
                provider="azure",
                text=text,
                embedding_func=_do_embedding,
            )
        else:
            return embedding_with_retry(_do_embedding)


class OllamaEmbeddings:
    # Format:
    # curl http://localhost:11434/api/embeddings -d '{
    #   "model": "mxbai-embed-large",
    #   "prompt": "Llamas are members of the camelid family"
    # }'

    def __init__(self, model: str, base_url: str, ollama_additional_kwargs: dict):
        self.model = model
        self.base_url = base_url
        self.ollama_additional_kwargs = ollama_additional_kwargs

    def get_text_embedding(self, text: str):

        def _do_embedding():
            import httpx

            headers = {"Content-Type": "application/json"}
            json_data = {"model": self.model, "prompt": text}
            json_data.update(self.ollama_additional_kwargs)

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    headers=headers,
                    json=json_data,
                )

            response_json = response.json()
            return response_json["embedding"]

        if is_embedding_tracing_enabled():
            return traced_embedding_with_retry(
                model=self.model,
                provider="ollama",
                text=text,
                embedding_func=_do_embedding,
                endpoint=self.base_url,
            )
        else:
            return embedding_with_retry(_do_embedding)


def query_embedding(embedding_model, query_text: str):
    """Generate padded embedding for querying database"""
    query_vec = embedding_model.get_text_embedding(query_text)
    query_vec = np.array(query_vec)
    query_vec = np.pad(query_vec, (0, MAX_EMBEDDING_DIM - query_vec.shape[0]), mode="constant").tolist()
    return query_vec


def embedding_model(config: EmbeddingConfig, user_id: Optional[uuid.UUID] = None):
    """Return LlamaIndex embedding model to use for embeddings"""

    endpoint_type = config.embedding_endpoint_type

    # TODO: refactor to pass in settings from server
    from mirix.settings import model_settings

    if endpoint_type == "openai":
        from mirix.services.provider_manager import ProviderManager

        # Check for database-stored API key first, fall back to model_settings
        override_key = ProviderManager().get_openai_override_key()
        api_key = override_key if override_key else model_settings.openai_api_key

        # Use direct OpenAI SDK if auth_provider is configured
        if hasattr(config, "auth_provider") and config.auth_provider:
            logger.info(f"Using OpenAI embedding with auth provider: {config.auth_provider}")
            return OpenAIEmbeddingWithCustomAuth(config=config, auth_provider=config.auth_provider)

        # Otherwise use llama_index (backwards compatible)
        from llama_index.embeddings.openai import OpenAIEmbedding

        additional_kwargs = {"user_id": user_id} if user_id else {}
        model = OpenAIEmbedding(
            api_base=config.embedding_endpoint,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
        )
        return model

    elif endpoint_type == "google_ai":
        # Use Google AI (Gemini) for embeddings
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        from mirix.services.provider_manager import ProviderManager

        # Check for database-stored API key first, fall back to model_settings

        override_key = ProviderManager().get_gemini_override_key()
        api_key = override_key if override_key else model_settings.gemini_api_key

        model = GoogleGenAIEmbedding(
            model_name=config.embedding_model,
            api_key=api_key,
            api_base=config.embedding_endpoint,
        )
        return model

    elif endpoint_type == "azure":
        assert all(
            [
                model_settings.azure_api_key is not None,
                model_settings.azure_base_url is not None,
                model_settings.azure_api_version is not None,
            ]
        )
        # from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

        ## https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
        # model = "text-embedding-3-small"
        # deployment = credentials.azure_embedding_deployment if credentials.azure_embedding_deployment is not None else model
        # return AzureOpenAIEmbedding(
        #    model=model,
        #    deployment_name=deployment,
        #    api_key=credentials.azure_key,
        #    azure_endpoint=credentials.azure_endpoint,
        #    api_version=credentials.azure_version,
        # )

        return AzureOpenAIEmbedding(
            api_endpoint=model_settings.azure_base_url,
            api_key=model_settings.azure_api_key,
            api_version=model_settings.azure_api_version,
            model=config.embedding_model,
        )

    elif endpoint_type == "hugging-face":
        return EmbeddingEndpoint(
            model=config.embedding_model,
            base_url=config.embedding_endpoint,
            user=user_id,
        )
    elif endpoint_type == "ollama":
        model = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.embedding_endpoint,
            ollama_additional_kwargs={},
        )
        return model

    else:
        raise ValueError(f"Unknown endpoint type {endpoint_type}")
