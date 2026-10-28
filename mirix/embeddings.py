"""Embedding models (every one is length-guarded).

:func:`embedding_model` is the one factory every call site uses. It builds
the provider's model for an :class:`EmbeddingConfig` and wraps it in
:class:`~mirix.embedding_guard.GuardedEmbedding`, so an input above the
model's token limit is embedded in chunks and merged, and every vector comes
back ``MAX_EMBEDDING_DIM`` wide.

Every embedding call has ONE retry layer, the embedding retry policy (3
backoff retries, 1 s doubling to 8 s; ``retry_call(operation="embedding")``):
the provider SDKs' own retries are switched off (``max_retries=0`` for the
OpenAI / Azure clients and llama_index's ``OpenAIEmbedding``, one attempt for
llama_index's ``GoogleGenAIEmbedding``), and a call made inside an enclosing
``retry_call(operation="embedding")`` is made once, that layer retrying it
(one query embedding under a sustained 429 used to cost up to 440
provider calls through the ``openai`` type).

Every attempt waits at most ``MIRIX_EMBEDDING_TIMEOUT_SECONDS`` (default 60 s;
the connect at most 5 s) for its answer, whatever the embedding type
(:func:`embedding_http_timeout`). A deployment behind a proxy that enforces its
own call deadline (the eval gateway) sets it above that deadline, so a
slow call is answered by the proxy instead of being abandoned and sent again.
"""

import uuid
from typing import Any, List, Optional

import httpx
import numpy as np

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.embedding_guard import GuardedEmbedding
from mirix.log import get_logger
from mirix.observability.context import (
    get_trace_context,
    mark_observation_as_child,
    outbound_session_headers,
)
from mirix.observability.langfuse_client import get_langfuse_client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.utils import is_valid_url

logger = get_logger(__name__)


def is_embedding_tracing_enabled() -> bool:
    """Check if Langfuse tracing is available and active for embeddings."""
    langfuse = get_langfuse_client()
    if not langfuse:
        return False
    trace_context = get_trace_context()
    return trace_context is not None and trace_context.get("trace_id") is not None


def raise_for_embedding_status(response: httpx.Response, provider: str) -> None:
    """Raise ``httpx.HTTPStatusError`` unless the embedding endpoint answered
    200: an error body must never be parsed as an embedding. 429 / 5xx
    are then retried by :func:`embedding_with_retry`; the message carries the
    status and the start of the body, never the request's credentials."""
    if response.status_code == 200:
        return
    body = response.text[:500] if response.content else ""
    raise httpx.HTTPStatusError(
        f"{provider} embeddings endpoint answered HTTP {response.status_code}"
        f" for {str(response.request.url).split('?', 1)[0]}: {body}",
        request=response.request,
        response=response,
    )


#: Connecting to an embedding endpoint never waits longer than this: an
#: unreachable host is a network error in seconds, not after the whole
#: per-attempt timeout.
EMBEDDING_CONNECT_TIMEOUT_SECONDS = 5.0


def embedding_timeout_seconds() -> float:
    """The per-attempt timeout of one embedding call, for every embedding type
    (``MIRIX_EMBEDDING_TIMEOUT_SECONDS``, default 60 s), read at call time."""
    from mirix import constants

    return float(constants.EMBEDDING_TIMEOUT_SECONDS)


def embedding_http_timeout(seconds: Optional[float] = None) -> httpx.Timeout:
    """The HTTP timeout of one embedding call: read, write and pool waits up to
    ``seconds`` (default :func:`embedding_timeout_seconds`), the connect at
    most :data:`EMBEDDING_CONNECT_TIMEOUT_SECONDS`. ``httpx`` and the OpenAI
    SDKs (``AsyncOpenAI`` / ``AsyncAzureOpenAI``) take it as is."""
    seconds = embedding_timeout_seconds() if seconds is None else float(seconds)
    return httpx.Timeout(
        seconds, connect=min(EMBEDDING_CONNECT_TIMEOUT_SECONDS, seconds)
    )


async def embedding_with_retry(embedding_func) -> List[float]:
    """Await one embedding call under the single embedding retry policy.

    Inside ``retry_call(operation="embedding")`` (skill writes and queries,
    the re-embedding pass, a guarded chunked call) that layer owns the
    retries: the call is made once here. Anywhere else this IS that layer:
    ``retry_call(operation="embedding")`` retries a transient failure (a
    network error, 408/429/5xx — as ``httpx`` or as an OpenAI / Google SDK
    error —, a timeout) 3 times with backoff and re-raises anything else at
    once. Either way retries are never stacked."""
    from mirix.services.retry_policy import embedding_retries_owned, retry_call

    if embedding_retries_owned():
        return await embedding_func()
    # No stage of its own: a retry is recorded under the enclosing block's
    # (a learning run's stage, a retrieval), if any.
    return await retry_call(embedding_func, operation="embedding")


async def traced_embedding_with_retry(
    model: str, provider: str, text: str, embedding_func, endpoint: Optional[str] = None
) -> List[float]:
    """
    Execute async embedding with Langfuse tracing and the single retry
    layer (see :func:`embedding_with_retry`).

    Each retry attempt is traced individually so failed attempts are visible.
    Langfuse failures are handled gracefully - they won't break embedding functionality.

    Only call this when tracing is enabled (check with is_embedding_tracing_enabled()).
    """
    from typing import cast

    from langfuse.types import TraceContext

    langfuse = get_langfuse_client()
    trace_context = get_trace_context()
    trace_id = trace_context.get("trace_id")
    parent_span_id = trace_context.get("observation_id")

    text_preview = text

    trace_context_dict: dict = {"trace_id": trace_id}
    if parent_span_id:
        trace_context_dict["parent_span_id"] = parent_span_id

    async def traced_call() -> List[float]:
        """Execute a single embedding attempt with tracing."""
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
            logger.error(f"Langfuse failed to start observation. Continuing without tracing: {e}")
            return await embedding_func()

        # Langfuse returns a sync context manager; use "with" not "async with"
        with observation_context as generation:
            mark_observation_as_child(generation)

            try:
                result = await embedding_func()
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
                raise

    from mirix.services.retry_policy import embedding_retries_owned, retry_call

    if embedding_retries_owned():  # the caller's retry_call retries
        return await traced_call()
    return await retry_call(traced_call, operation="embedding")


def _session_headers() -> dict:
    """``{"extra_headers": ...}`` for an OpenAI SDK call, with the served conversation
    session (``MIRIX_LLM_SESSION_HEADER``, off by default), or ``{}``."""
    headers = outbound_session_headers()
    return {"extra_headers": headers} if headers else {}


class OpenAIEmbeddingWithCustomAuth:
    """OpenAI embedding client with auth provider support using async OpenAI SDK."""

    def __init__(self, config: EmbeddingConfig, auth_provider: str):
        """
        Initialize OpenAI embedding client with auth provider support.

        Args:
            config: EmbeddingConfig with auth_provider field set
        """
        self.config = config
        self.model = config.embedding_model
        self.auth_provider = auth_provider

    async def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding with dynamic auth headers from auth provider.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If auth provider is not found in registry
        """

        async def _do_embedding():
            from openai import AsyncOpenAI

            from mirix.llm_api.auth_provider import get_auth_provider

            auth_provider = get_auth_provider(self.auth_provider)
            if not auth_provider:
                raise ValueError(
                    f"Auth provider '{self.config.auth_provider}' not found in registry. "
                    "Make sure to register it before using."
                )

            try:
                auth_headers = auth_provider.get_auth_headers()
                logger.info(
                    f"OpenAI Embedding - Using auth provider '{self.config.auth_provider}' "
                    f"to inject {len(auth_headers)} header(s)"
                )
            except Exception as e:
                logger.error(f"Failed to get auth headers from provider '{self.config.auth_provider}': {e}")
                raise

            client = AsyncOpenAI(
                api_key="DUMMY_API_KEY",
                base_url=self.config.embedding_endpoint,
                default_headers=auth_headers,
                max_retries=0,  # one retry layer: embedding_with_retry's
                timeout=embedding_http_timeout(),
            )

            response = await client.embeddings.create(
                model=self.model, input=text, **_session_headers()
            )
            return response.data[0].embedding

        if is_embedding_tracing_enabled():
            return await traced_embedding_with_retry(
                model=self.config.langfuse_model or self.model,
                provider="openai_custom_auth",
                text=text,
                embedding_func=_do_embedding,
                endpoint=self.config.embedding_endpoint,
            )
        else:
            return await embedding_with_retry(_do_embedding)


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
        timeout: Optional[float] = None,
        langfuse_model: Optional[str] = None,
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
        self._langfuse_model = langfuse_model
        self._user = user
        self._base_url = base_url
        # None: MIRIX_EMBEDDING_TIMEOUT_SECONDS, read at call time.
        self._timeout = timeout

    async def _call_api(self, text: str) -> List[float]:
        if not is_valid_url(self._base_url):
            raise ValueError(
                f"Embeddings endpoint does not have a valid URL (set to: '{self._base_url}'). Make sure embedding_endpoint is set correctly in your Mirix config."
            )
        headers = {**outbound_session_headers(), "Content-Type": "application/json"}
        json_data = {"input": text, "model": self.model_name, "user": self._user}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json=json_data,
                timeout=embedding_http_timeout(self._timeout),
            )

        raise_for_embedding_status(response, "hugging-face")
        response_json = response.json()

        if isinstance(response_json, list):
            embedding = response_json
        elif isinstance(response_json, dict):
            try:
                embedding = response_json["data"][0]["embedding"]
            except (KeyError, IndexError):
                raise TypeError(
                    f"Got back an unexpected payload from text embedding function, response=\n{response_json}"
                )
        else:
            raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")

        return embedding

    async def get_text_embedding(self, text: str) -> List[float]:
        if is_embedding_tracing_enabled():
            return await traced_embedding_with_retry(
                model=self._langfuse_model or self.model_name,
                provider="hugging-face",
                text=text,
                embedding_func=lambda: self._call_api(text),
                endpoint=self._base_url,
            )
        else:
            return await embedding_with_retry(lambda: self._call_api(text))


class OpenRouterEmbedding(EmbeddingEndpoint):
    """EmbeddingEndpoint subclass that adds Bearer auth for OpenRouter."""

    def __init__(self, api_key: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._api_key = api_key

    async def _call_api(self, text: str) -> List[float]:
        headers = {
            **outbound_session_headers(),
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        json_data = {"input": text, "model": self.model_name}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json=json_data,
                timeout=embedding_http_timeout(self._timeout),
            )

        # A non-200 answer (an error body) is an error, never parsed.
        raise_for_embedding_status(response, "OpenRouter")
        response_json = response.json()
        if isinstance(response_json, dict):
            try:
                return response_json["data"][0]["embedding"]
            except (KeyError, IndexError):
                raise TypeError(f"Unexpected embedding response: {response_json}")
        raise TypeError(f"Unexpected embedding response type: {response_json}")


class AzureOpenAIEmbedding:
    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        api_version: str,
        model: str,
        langfuse_model: Optional[str] = None,
    ):
        from openai import AsyncAzureOpenAI

        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint,
            max_retries=0,  # one retry layer: embedding_with_retry's
            timeout=embedding_http_timeout(),
        )
        self.model = model
        self._langfuse_model = langfuse_model

    async def get_text_embedding(self, text: str):

        async def _do_embedding():
            response = await self.client.embeddings.create(
                input=[text], model=self.model, **_session_headers()
            )
            return response.data[0].embedding

        if is_embedding_tracing_enabled():
            return await traced_embedding_with_retry(
                model=self._langfuse_model or self.model,
                provider="azure",
                text=text,
                embedding_func=_do_embedding,
            )
        else:
            return await embedding_with_retry(_do_embedding)


class OllamaEmbeddings:
    # Format:
    # curl http://localhost:11434/api/embeddings -d '{
    #   "model": "mxbai-embed-large",
    #   "prompt": "Llamas are members of the camelid family"
    # }'

    def __init__(
        self,
        model: str,
        base_url: str,
        ollama_additional_kwargs: dict,
        langfuse_model: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.ollama_additional_kwargs = ollama_additional_kwargs
        self._langfuse_model = langfuse_model

    async def get_text_embedding(self, text: str):

        async def _do_embedding():
            headers = {**outbound_session_headers(), "Content-Type": "application/json"}
            json_data = {"model": self.model, "prompt": text}
            json_data.update(self.ollama_additional_kwargs)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    headers=headers,
                    json=json_data,
                    timeout=embedding_http_timeout(),
                )

            raise_for_embedding_status(response, "Ollama")
            response_json = response.json()
            return response_json["embedding"]

        if is_embedding_tracing_enabled():
            return await traced_embedding_with_retry(
                model=self._langfuse_model or self.model,
                provider="ollama",
                text=text,
                embedding_func=_do_embedding,
                endpoint=self.base_url,
            )
        else:
            return await embedding_with_retry(_do_embedding)


class AsyncLlamaIndexEmbeddingAdapter:
    """Adapter that wraps LlamaIndex embedding models to provide a consistent
    async ``get_text_embedding`` interface via ``aget_text_embedding``.

    The wrapped model must make ONE attempt per call (its own retries off,
    see :func:`_provider_embedding_model`): the call goes through
    :func:`embedding_with_retry`, the single retry layer."""

    def __init__(
        self,
        llama_index_model,
        *,
        provider: str = "llama_index",
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        self._model = llama_index_model
        self._provider = provider
        self._model_label = model or getattr(llama_index_model, "model_name", None) or provider
        self._endpoint = endpoint

    async def get_text_embedding(self, text: str) -> List[float]:
        async def _do_embedding() -> List[float]:
            return await self._model.aget_text_embedding(text)

        if is_embedding_tracing_enabled():
            return await traced_embedding_with_retry(
                model=self._model_label,
                provider=self._provider,
                text=text,
                embedding_func=_do_embedding,
                endpoint=self._endpoint,
            )
        return await embedding_with_retry(_do_embedding)


async def query_embedding(embedding_model, query_text: str):
    """Generate padded embedding for querying database"""
    query_vec = await embedding_model.get_text_embedding(query_text)
    query_vec = np.array(query_vec)
    query_vec = np.pad(query_vec, (0, MAX_EMBEDDING_DIM - query_vec.shape[0]), mode="constant").tolist()
    return query_vec


async def embedding_model(
    config: EmbeddingConfig, user_id: Optional[uuid.UUID] = None
) -> GuardedEmbedding:
    """The embedding model of ``config``, behind the length guard.

    ``get_text_embedding(text)`` returns a ``MAX_EMBEDDING_DIM``-wide vector;
    a text above the model's token limit (``MIRIX_EMBEDDING_TOKEN_LIMITS``) is
    embedded in chunks and merged (see :mod:`mirix.embedding_guard`). Every
    call site goes through here, so none needs its own guard.
    """
    provider_model = await _provider_embedding_model(config, user_id)
    return GuardedEmbedding(provider_model, model_name=config.embedding_model)


async def _provider_embedding_model(
    config: EmbeddingConfig, user_id: Optional[uuid.UUID] = None
):
    """The provider's own embedding model for ``config`` (unguarded)."""

    endpoint_type = config.embedding_endpoint_type

    # TODO: refactor to pass in settings from server
    from mirix.settings import model_settings

    if endpoint_type == "openai":
        from mirix.services.provider_manager import ProviderManager

        if config.api_key:
            api_key = config.api_key
        else:
            # Check for database-stored API key first, fall back to model_settings
            override_key = await ProviderManager().get_openai_override_key()
            api_key = override_key if override_key else model_settings.openai_api_key

        # Use direct OpenAI SDK if auth_provider is configured
        if hasattr(config, "auth_provider") and config.auth_provider:
            logger.info(f"Using OpenAI embedding with auth provider: {config.auth_provider}")
            return OpenAIEmbeddingWithCustomAuth(config=config, auth_provider=config.auth_provider)

        # Otherwise use llama_index (backwards compatible)
        from llama_index.embeddings.openai import OpenAIEmbedding

        additional_kwargs = {"user_id": user_id} if user_id else {}
        # max_retries=0: one attempt per call — no tenacity loop (10 attempts
        # within 60 s) and no SDK retries (its AsyncOpenAI gets the same
        # max_retries); embedding_with_retry is the only retry layer.
        model = OpenAIEmbedding(
            api_base=config.embedding_endpoint,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=0,
            timeout=embedding_timeout_seconds(),
        )
        return AsyncLlamaIndexEmbeddingAdapter(
            model,
            provider="openai",
            model=config.langfuse_model or getattr(model, "model_name", None),
            endpoint=config.embedding_endpoint,
        )

    elif endpoint_type == "google_ai":
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        from mirix.services.provider_manager import ProviderManager

        if config.api_key:
            api_key = config.api_key
        else:
            override_key = await ProviderManager().get_gemini_override_key()
            api_key = override_key if override_key else model_settings.gemini_api_key

        from google.genai import types as genai_types

        # retries=1: one attempt per call (its tenacity loop makes 3 by
        # default; the google-genai client itself retries only when given
        # retry_options); embedding_with_retry is the only retry layer.
        # llama_index never hands its own ``timeout`` to the google-genai
        # client, which then waits without bound: the per-attempt timeout goes
        # in the client's HTTP options (milliseconds).
        seconds = embedding_timeout_seconds()
        model = GoogleGenAIEmbedding(
            model_name=config.embedding_model,
            api_key=api_key,
            api_base=config.embedding_endpoint,
            retries=1,
            timeout=max(1, int(seconds)),
            http_options=genai_types.HttpOptions(timeout=max(1, int(seconds * 1000))),
        )
        return AsyncLlamaIndexEmbeddingAdapter(
            model,
            provider="google_ai",
            model=config.langfuse_model or config.embedding_model,
            endpoint=config.embedding_endpoint,
        )

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
            langfuse_model=config.langfuse_model,
        )

    elif endpoint_type == "hugging-face":
        return EmbeddingEndpoint(
            model=config.embedding_model,
            base_url=config.embedding_endpoint,
            user=user_id,
            langfuse_model=config.langfuse_model,
        )
    elif endpoint_type == "openrouter":
        # OpenRouter's OpenAI-compatible /embeddings endpoint (Bearer auth).
        # Key: the config's, else OPENROUTER_API_KEY. An OpenAI key is never
        # sent to openrouter.ai by default: OPENAI_API_KEY is used only when
        # the operator set the embedding endpoint explicitly (the historical
        # setup of deployments that put an OpenRouter key there).
        import os

        from mirix.errors import MirixConfigurationError

        api_key = (
            config.api_key
            or model_settings.openrouter_api_key
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if not api_key and config.embedding_endpoint:
            api_key = model_settings.openai_api_key
            if api_key:
                logger.warning(
                    "openrouter embeddings: no OpenRouter key; using OPENAI_API_KEY "
                    "for the explicitly configured endpoint %s",
                    config.embedding_endpoint,
                )
        if not api_key:
            raise MirixConfigurationError(
                "OpenRouter embeddings need an OpenRouter API key: set "
                "OPENROUTER_API_KEY or the embedding config's api_key",
                missing_fields=["openrouter_api_key"],
            )
        from mirix.llm_api.openai_client import OPENROUTER_DEFAULT_BASE_URL

        return OpenRouterEmbedding(
            api_key=api_key,
            model=config.embedding_model,
            base_url=config.embedding_endpoint or OPENROUTER_DEFAULT_BASE_URL,
            user=str(user_id) if user_id else "",
            langfuse_model=config.langfuse_model,
        )

    elif endpoint_type == "ollama":
        model = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.embedding_endpoint,
            ollama_additional_kwargs={},
            langfuse_model=config.langfuse_model,
        )
        return model

    else:
        raise ValueError(f"Unknown endpoint type {endpoint_type}")
