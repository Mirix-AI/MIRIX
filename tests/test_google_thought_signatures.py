"""Gemini tool-loop signatures survive response parsing, storage and replay."""

import json

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, select

from mirix.llm_api.google_ai_client import GoogleAIClient
from mirix.orm.custom_columns import ToolCallColumn
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent


def response(parts):
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 7,
            "totalTokenCount": 19,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("signature", ["EpoGCpcG+/opaque==", None])
async def test_tool_loop_preserves_signature_through_storage(signature, monkeypatch):
    config = LLMConfig(
        model="gemini-3.6-flash",
        model_endpoint_type="google_ai",
        context_window=1_000_000,
    )
    client = GoogleAIClient(config)
    part = {"functionCall": {"name": "open_drawer", "args": {"number": 3}}}
    if signature is not None:
        part["thoughtSignature"] = signature
    requests = []

    async def request(data):
        requests.append(data)
        if len(requests) == 1:
            return response([part])
        [replayed] = [c for c in data["contents"] if c["role"] == "model"]
        assert replayed["parts"] == [part]
        return response([{"text": "Drawer opened."}])

    monkeypatch.setattr(client, "request", request)
    history = [
        Message(role="system", content=[TextContent(text="You are an agent.")]),
        Message(role="user", content=[TextContent(text="Open drawer 3.")]),
    ]
    tools = [
        {
            "name": "open_drawer",
            "description": "Open a drawer.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        }
    ]
    reply = (
        (await client.send_llm_request(messages=history, tools=tools))
        .choices[0]
        .message
    )
    assert reply.tool_calls[0].thought_signature == signature
    stored = Message.dict_to_message(
        agent_id="a1", openai_message_dict=reply.model_dump()
    )

    # Exercise the real JSON column, including deserialization into the internal schema.
    engine = create_engine("sqlite://")
    metadata = MetaData()
    table = Table(
        "messages",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("calls", ToolCallColumn),
    )
    metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            connection.execute(table.insert().values(id=1, calls=stored.tool_calls))
            stored.tool_calls = connection.execute(select(table.c.calls)).scalar_one()
    finally:
        engine.dispose()
    assert stored.tool_calls[0].thought_signature == signature
    assert "thought_signature" not in stored.to_openai_dict()["tool_calls"][0]
    assert "thoughtSignature" not in json.dumps(stored.to_openai_dict())
    history.extend(
        [
            stored,
            Message.dict_to_message(
                agent_id="a1",
                openai_message_dict={
                    "role": "tool",
                    "name": "open_drawer",
                    "content": '{"status": "OK"}',
                    "tool_call_id": reply.tool_calls[0].id,
                },
                name="open_drawer",
            ),
        ]
    )
    final = await client.send_llm_request(messages=history, tools=tools)
    assert final.choices[0].message.content == "Drawer opened."
    assert len(requests) == 2


def test_parallel_calls_keep_each_signature_on_its_own_part():
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "thought_signature": signature,
            "function": {"name": "open_drawer", "arguments": json.dumps({"number": i})},
        }
        for i, signature in enumerate(["first+/==", None, "third+/=="])
    ]
    message = Message.dict_to_message(
        agent_id="a1",
        openai_message_dict={
            "role": "assistant",
            "content": None,
            "tool_calls": calls,
        },
    )
    parts = message.to_google_ai_dict()["parts"]
    assert [p.get("thoughtSignature") for p in parts] == [
        "first+/==",
        None,
        "third+/==",
    ]
    assert "thoughtSignature" not in parts[1]
    assert [p["functionCall"]["args"]["number"] for p in parts] == [0, 1, 2]
