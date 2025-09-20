from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class MessageContentType(str, Enum):
    text = "text"
    image_url = "image_url"
    file_uri = "file_uri"
    google_cloud_file_uri = "google_cloud_file_uri"
    tool_call = "tool_call"
    tool_return = "tool_return"
    reasoning = "reasoning"
    redacted_reasoning = "redacted_reasoning"
    omitted_reasoning = "omitted_reasoning"


class MessageContent(BaseModel):
    type: MessageContentType = Field(..., description="The type of the message.")


# -------------------------------
# User Content Types
# -------------------------------


class TextContent(MessageContent):
    type: Literal[MessageContentType.text] = Field(MessageContentType.text, description="The type of the message.")
    text: str = Field(..., description="The text content of the message.")

class ImageContent(MessageContent):
    type: Literal[MessageContentType.image_url] = Field(MessageContentType.image_url, description="The type of the message.")
    image_id: str = Field(..., description="The id of the image in the database")
    detail: Optional[str] = Field(None, description="The detail of the image.")

class FileContent(MessageContent):
    type: Literal[MessageContentType.file_uri] = Field(MessageContentType.file_uri, description="The type of the message.")
    file_id: str = Field(..., description="The id of the file in the database")

class CloudFileContent(MessageContent):
    type: Literal[MessageContentType.google_cloud_file_uri] = Field(MessageContentType.google_cloud_file_uri, description="The type of the message.")
    cloud_file_uri: str = Field(..., description="The URI of the file in the database")

MirixUserMessageContentUnion = Annotated[
    Union[TextContent, ImageContent, FileContent, CloudFileContent],
    Field(discriminator="type"),
]

def create_mirix_user_message_content_union_schema():
    return {
        "oneOf": [
            {"$ref": "#/components/schemas/TextContent"},
        ],
        "discriminator": {
            "propertyName": "type",
            "mapping": {
                "text": "#/components/schemas/TextContent",
                "image": "#/components/schemas/ImageContent", # TODO: (yu) Not sure about this
            },
        },
    }


def get_mirix_user_message_content_union_str_json_schema():
    return {
        "anyOf": [
            {
                "type": "array",
                "items": {
                    "$ref": "#/components/schemas/MirixUserMessageContentUnion",
                },
            },
            {"type": "string"},
        ],
    }


# -------------------------------
# Assistant Content Types
# -------------------------------


MirixAssistantMessageContentUnion = Annotated[
    Union[TextContent],
    Field(discriminator="type"),
]


def create_mirix_assistant_message_content_union_schema():
    return {
        "oneOf": [
            {"$ref": "#/components/schemas/TextContent"},
        ],
        "discriminator": {
            "propertyName": "type",
            "mapping": {
                "text": "#/components/schemas/TextContent",
            },
        },
    }


def get_mirix_assistant_message_content_union_str_json_schema():
    return {
        "anyOf": [
            {
                "type": "array",
                "items": {
                    "$ref": "#/components/schemas/MirixAssistantMessageContentUnion",
                },
            },
            {"type": "string"},
        ],
    }


# -------------------------------
# Intermediate Step Content Types
# -------------------------------


class ToolCallContent(MessageContent):
    type: Literal[MessageContentType.tool_call] = Field(
        MessageContentType.tool_call, description="Indicates this content represents a tool call event."
    )
    id: str = Field(..., description="A unique identifier for this specific tool call instance.")
    name: str = Field(..., description="The name of the tool being called.")
    input: dict = Field(
        ..., description="The parameters being passed to the tool, structured as a dictionary of parameter names to values."
    )


class ToolReturnContent(MessageContent):
    type: Literal[MessageContentType.tool_return] = Field(
        MessageContentType.tool_return, description="Indicates this content represents a tool return event."
    )
    tool_call_id: str = Field(..., description="References the ID of the ToolCallContent that initiated this tool call.")
    content: str = Field(..., description="The content returned by the tool execution.")
    is_error: bool = Field(..., description="Indicates whether the tool execution resulted in an error.")


class ReasoningContent(MessageContent):
    type: Literal[MessageContentType.reasoning] = Field(
        MessageContentType.reasoning, description="Indicates this is a reasoning/intermediate step."
    )
    is_native: bool = Field(..., description="Whether the reasoning content was generated by a reasoner model that processed this step.")
    reasoning: str = Field(..., description="The intermediate reasoning or thought process content.")
    signature: Optional[str] = Field(None, description="A unique identifier for this reasoning step.")


class RedactedReasoningContent(MessageContent):
    type: Literal[MessageContentType.redacted_reasoning] = Field(
        MessageContentType.redacted_reasoning, description="Indicates this is a redacted thinking step."
    )
    data: str = Field(..., description="The redacted or filtered intermediate reasoning content.")


class OmittedReasoningContent(MessageContent):
    type: Literal[MessageContentType.omitted_reasoning] = Field(
        MessageContentType.omitted_reasoning, description="Indicates this is an omitted reasoning step."
    )
    tokens: int = Field(..., description="The reasoning token count for intermediate reasoning content.")


MirixMessageContentUnion = Annotated[
    Union[TextContent, ImageContent, FileContent, CloudFileContent, ToolCallContent, ToolReturnContent, ReasoningContent, RedactedReasoningContent, OmittedReasoningContent],
    Field(discriminator="type"),
]



def get_mirix_message_content_union_str_json_schema():
    return {
        "anyOf": [
            {
                "type": "array",
                "items": {
                    "$ref": "#/components/schemas/MiriMessageContentUnion",
                },
            },
            {"type": "string"},
        ],
    }

