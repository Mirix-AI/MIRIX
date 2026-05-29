"""
AWS Bedrock client wrapper — drop-in replacement for ``openai.OpenAI``.

Provides a duck-type-compatible interface so that both ``PRMScorer`` and
``SkillEvolver`` can use Bedrock without code changes::

    from metaclaw.bedrock_client import BedrockChatClient

    client = BedrockChatClient(model_id="us.anthropic.claude-sonnet-4-6")
    # Works exactly like openai.OpenAI():
    resp = client.chat.completions.create(
        model="ignored",   # model_id from constructor is used
        messages=[...],
        temperature=0.6,
        max_completion_tokens=1024,
    )
    print(resp.choices[0].message.content)

Also implements ``chat_complete(prompt) -> str`` for SkillEvolver's
``llm_client`` interface.

No API key needed — uses IAM role credentials from the instance profile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Response dataclasses (mimic openai.ChatCompletion)                   #
# ------------------------------------------------------------------ #

@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _ChatCompletion:
    choices: list[_Choice] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)
    model: str = ""


# ------------------------------------------------------------------ #
# Completions / Chat namespace (duck-type openai.OpenAI().chat)        #
# ------------------------------------------------------------------ #

class _Completions:
    """Mimics ``openai.resources.chat.Completions``."""

    def __init__(self, bedrock_client, model_id: str, region: str):
        self._model_id = model_id
        self._region = region
        self._client = bedrock_client

    def create(
        self,
        *,
        model: str = "",
        messages: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_completion_tokens: int = 1024,
        max_tokens: int | None = None,
        **kwargs,
    ) -> _ChatCompletion:
        """Synchronous Bedrock Converse call, same signature as OpenAI."""
        max_tok = max_completion_tokens or max_tokens or 1024
        messages = messages or []

        # Convert OpenAI message format → Bedrock Converse format
        system_parts = []
        converse_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": content})
            else:
                converse_messages.append({
                    "role": role,
                    "content": [{"text": content}],
                })

        # Bedrock requires at least one user message
        if not converse_messages:
            converse_messages = [{"role": "user", "content": [{"text": ""}]}]

        converse_kwargs: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tok,
                "temperature": temperature,
            },
        }
        if system_parts:
            converse_kwargs["system"] = system_parts

        response = self._client.converse(**converse_kwargs)

        # Extract response
        output_content = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                output_content += block["text"]

        usage = response.get("usage", {})
        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=output_content))],
            usage=_Usage(
                prompt_tokens=usage.get("inputTokens", 0),
                completion_tokens=usage.get("outputTokens", 0),
                total_tokens=usage.get("totalTokens", 0),
            ),
            model=self._model_id,
        )


class _Chat:
    """Mimics ``openai.resources.Chat``."""

    def __init__(self, completions: _Completions):
        self.completions = completions


# ------------------------------------------------------------------ #
# Public class                                                         #
# ------------------------------------------------------------------ #

class BedrockChatClient:
    """
    Drop-in replacement for ``openai.OpenAI()`` that calls AWS Bedrock.

    Parameters
    ----------
    model_id:
        Bedrock inference profile ID, e.g. ``"us.anthropic.claude-sonnet-4-6"``.
    region:
        AWS region (default ``"us-east-1"``).

    Usage (OpenAI-compatible)::

        client = BedrockChatClient()
        resp = client.chat.completions.create(
            model="anything",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(resp.choices[0].message.content)

    Usage (SkillEvolver llm_client)::

        client = BedrockChatClient()
        text = client.chat_complete("Analyze these failures...")
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-6",
        region: str = "us-east-1",
    ):
        import boto3
        self._boto_client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self._completions = _Completions(self._boto_client, model_id, region)
        self.chat = _Chat(self._completions)

    def chat_complete(self, prompt: str) -> str:
        """SkillEvolver ``llm_client`` interface: prompt in → text out."""
        resp = self.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=3000,
        )
        return resp.choices[0].message.content
