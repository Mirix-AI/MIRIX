import html
import json
import re
from typing import List, Union

from pydantic import BaseModel, Field

from mirix.schemas.enums import MessageStreamStatus
from mirix.schemas.mirix_message import MirixMessage, MirixMessageUnion
from mirix.schemas.usage import MirixUsageStatistics
from mirix.utils import json_dumps

# TODO: consider moving into own file


class MirixResponse(BaseModel):
    """
    Response object from an agent interaction, consisting of the new messages generated by the agent and usage statistics.
    The type of the returned messages can be either `Message` or `MirixMessage`, depending on what was specified in the request.

    Attributes:
        messages (List[Union[Message, MirixMessage]]): The messages returned by the agent.
        usage (MirixUsageStatistics): The usage statistics
    """

    messages: List[MirixMessageUnion] = Field(
        ...,
        description="The messages returned by the agent.",
        json_schema_extra={
            "items": {
                "$ref": "#/components/schemas/MirixMessageUnion",
            }
        },
    )
    usage: MirixUsageStatistics = Field(
        ...,
        description="The usage statistics of the agent.",
    )

    def __str__(self):
        return json_dumps(
            {
                "messages": [message.model_dump() for message in self.messages],
                # Assume `Message` and `MirixMessage` have a `dict()` method
                "usage": self.usage.model_dump(),  # Assume `MirixUsageStatistics` has a `dict()` method
            },
            indent=4,
        )

    def _repr_html_(self):
        def get_formatted_content(msg):
            if msg.message_type == "internal_monologue":
                return f'<div class="content"><span class="internal-monologue">{html.escape(msg.internal_monologue)}</span></div>'
            if msg.message_type == "reasoning_message":
                return f'<div class="content"><span class="internal-monologue">{html.escape(msg.reasoning)}</span></div>'
            elif msg.message_type == "function_call":
                args = format_json(msg.function_call.arguments)
                return f'<div class="content"><span class="function-name">{html.escape(msg.function_call.name)}</span>({args})</div>'
            elif msg.message_type == "tool_call_message":
                args = format_json(msg.tool_call.arguments)
                return f'<div class="content"><span class="function-name">{html.escape(msg.tool_call.name)}</span>({args})</div>'
            elif msg.message_type == "function_return":
                return_value = format_json(msg.function_return)
                # return f'<div class="status-line">Status: {html.escape(msg.status)}</div><div class="content">{return_value}</div>'
                return f'<div class="content">{return_value}</div>'
            elif msg.message_type == "tool_return_message":
                return_value = format_json(msg.tool_return)
                # return f'<div class="status-line">Status: {html.escape(msg.status)}</div><div class="content">{return_value}</div>'
                return f'<div class="content">{return_value}</div>'
            elif msg.message_type == "user_message":
                if is_json(msg.message):
                    return f'<div class="content">{format_json(msg.message)}</div>'
                else:
                    return f'<div class="content">{html.escape(msg.message)}</div>'
            elif msg.message_type in ["assistant_message", "system_message"]:
                return f'<div class="content">{html.escape(msg.message)}</div>'
            else:
                return f'<div class="content">{html.escape(str(msg))}</div>'

        def is_json(string):
            try:
                json.loads(string)
                return True
            except ValueError:
                return False

        def format_json(json_str):
            try:
                parsed = json.loads(json_str)
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                formatted = formatted.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                formatted = formatted.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
                formatted = re.sub(r'(".*?"):', r'<span class="json-key">\1</span>:', formatted)
                formatted = re.sub(r': (".*?")', r': <span class="json-string">\1</span>', formatted)
                formatted = re.sub(r": (\d+)", r': <span class="json-number">\1</span>', formatted)
                formatted = re.sub(r": (true|false)", r': <span class="json-boolean">\1</span>', formatted)
                return formatted
            except json.JSONDecodeError:
                return html.escape(json_str)

        html_output = """
        <style>
            .message-container, .usage-container {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 20px auto;
                background-color: #1e1e1e;
                border-radius: 8px;
                overflow: hidden;
                color: #d4d4d4;
            }
            .message, .usage-stats {
                padding: 10px 15px;
                border-bottom: 1px solid #3a3a3a;
            }
            .message:last-child, .usage-stats:last-child {
                border-bottom: none;
            }
            .title {
                font-weight: bold;
                margin-bottom: 5px;
                color: #ffffff;
                text-transform: uppercase;
                font-size: 0.9em;
            }
            .content {
                background-color: #2d2d2d;
                border-radius: 4px;
                padding: 5px 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                white-space: pre-wrap;
            }
            .json-key, .function-name, .json-boolean { color: #9cdcfe; }
            .json-string { color: #ce9178; }
            .json-number { color: #b5cea8; }
            .internal-monologue { font-style: italic; }
        </style>
        <div class="message-container">
        """

        for msg in self.messages:
            content = get_formatted_content(msg)
            title = msg.message_type.replace("_", " ").upper()
            html_output += f"""
            <div class="message">
                <div class="title">{title}</div>
                {content}
            </div>
            """
        html_output += "</div>"

        # Formatting the usage statistics
        usage_html = json.dumps(self.usage.model_dump(), indent=2)
        html_output += f"""
        <div class="usage-container">
            <div class="usage-stats">
                <div class="title">USAGE STATISTICS</div>
                <div class="content">{format_json(usage_html)}</div>
            </div>
        </div>
        """

        return html_output


# The streaming response is either [DONE], [DONE_STEP], [DONE], an error, or a MirixMessage
MirixStreamingResponse = Union[MirixMessage, MessageStreamStatus, MirixUsageStatistics]
