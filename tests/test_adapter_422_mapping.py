"""End-to-end test of the 422 -> Permanent classification chain.

Exercises the full path that fires when an upstream provider returns HTTP
422: the SDK's UnprocessableEntityError, the adapter's handle_llm_error
mapping to LLMUnprocessableEntityError, the classifier's PERMANENT
bucket assignment, and process_with_policy's PERMANENT_FAILURE outcome.

Uses the actual prod LXS Risk Screening response body so the test reflects
real upstream shape, not a synthetic one.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from mirix.errors import LLMUnprocessableEntityError
from mirix.queue.error_policy import Bucket, SaveOutcome, classify, process_with_policy
from mirix.schemas.llm_config import LLMConfig

# Real prod LXS 422 body, captured by Lucas from Langfuse traces.
LXS_422_BODY = {
    "error_message": "Risk Screening failed for input",
    "cause": "A suspicious language is detected",
}


def _make_openai_422_exception() -> openai.UnprocessableEntityError:
    """Construct an openai.UnprocessableEntityError matching a real LXS 422."""
    fake_response = httpx.Response(
        status_code=422,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json=LXS_422_BODY,
    )
    return openai.UnprocessableEntityError(
        message="Risk Screening failed for input",
        response=fake_response,
        body=LXS_422_BODY,
    )


@pytest.mark.asyncio
async def test_openai_client_maps_422_to_llm_unprocessable_entity_error():
    """The OpenAI adapter's handle_llm_error must convert the SDK's
    UnprocessableEntityError into MIRIX's LLMUnprocessableEntityError.

    This is the link the classifier depends on. If the adapter ever stops
    catching this case, the classifier silently defaults the error to
    Transient and we re-introduce the cascade bug.
    """
    from mirix.llm_api.openai_client import OpenAIClient

    client = OpenAIClient(
        llm_config=LLMConfig(
            model="gpt-4o",
            model_endpoint_type="openai",
            model_endpoint="https://api.openai.com",
            context_window=128000,
        ),
    )

    sdk_exc = _make_openai_422_exception()
    mapped = await client.handle_llm_error(sdk_exc)

    assert isinstance(
        mapped, LLMUnprocessableEntityError
    ), f"Expected LLMUnprocessableEntityError, got {type(mapped).__name__}"


@pytest.mark.asyncio
async def test_full_chain_openai_422_classifies_as_permanent_and_acks():
    """End-to-end: openai 422 -> adapter -> typed exception -> classifier -> PERMANENT.

    process_with_policy must return PERMANENT_FAILURE so the runtime ack
    path runs (the cascade-fix outcome).
    """
    from mirix.llm_api.openai_client import OpenAIClient

    client = OpenAIClient(
        llm_config=LLMConfig(
            model="gpt-4o",
            model_endpoint_type="openai",
            model_endpoint="https://api.openai.com",
            context_window=128000,
        ),
    )

    async def run_step():
        # Stand-in for the agent step that would invoke the LLM and
        # receive a 422 from the upstream provider.
        sdk_exc = _make_openai_422_exception()
        raise await client.handle_llm_error(sdk_exc)

    outcome = await process_with_policy(run_step, memory_source_id="src-test")

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    assert outcome.bucket is Bucket.PERMANENT
    assert isinstance(outcome.cause, LLMUnprocessableEntityError)
    # The mapped error message should reference the upstream LXS reason.
    assert "Risk Screening" in str(outcome.cause)


def test_classify_directly_on_mapped_exception():
    """Sanity: the result of the adapter mapping classifies as PERMANENT.

    Smaller-scope twin of the end-to-end test above; useful for debugging
    when the chain breaks.
    """
    mapped = LLMUnprocessableEntityError(message="Invalid request content for OpenAI: Risk Screening failed for input")
    assert classify(mapped) is Bucket.PERMANENT
