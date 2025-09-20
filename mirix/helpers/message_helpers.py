from mirix import system
from mirix.schemas.enums import MessageRole
from mirix.schemas.mirix_message_content import TextContent, ImageContent, FileContent, CloudFileContent
from mirix.schemas.message import Message, MessageCreate


def extract_and_wrap_message_content(
                                     role: MessageRole,
                                     content: str | TextContent | ImageContent | FileContent | CloudFileContent,
                                     wrap_user_message: bool = False,
                                     wrap_system_message: bool = True,):
    
    # Extract message content
    if isinstance(content, str):
        message_content = content
    elif content and isinstance(content, TextContent):
        message_content = content.text
    elif content and isinstance(content, ImageContent):
        # Skip wrapping if the content is an image
        return content
    elif content and isinstance(content, FileContent):
        # Skip wrapping if the content is a file
        return content
    elif content and isinstance(content, CloudFileContent):
        return content
    else:
        raise ValueError("Message content is empty or invalid")

    # Apply wrapping if needed
    if role == MessageRole.user and wrap_user_message:
        message_content = system.package_user_message(user_message=message_content)
    elif role == MessageRole.system and wrap_system_message:
        message_content = system.package_system_message(system_message=message_content)
    elif role not in {MessageRole.user, MessageRole.system}:
        raise ValueError(f"Invalid message role: {role}")
    return TextContent(text=message_content)

def prepare_input_message_create(
    message: MessageCreate,
    agent_id: str,
    **kwargs
) -> Message:
    """Converts a MessageCreate object into a Message object, applying wrapping if needed."""
    # TODO: This seems like extra boilerplate with little benefit
    assert isinstance(message, MessageCreate)

    if isinstance(message.content, list):
        content = [extract_and_wrap_message_content(role=message.role, content=c, **kwargs) for c in message.content]
    else:
        content = [extract_and_wrap_message_content(role=message.role, content=message.content, **kwargs)]

    return Message(
        agent_id=agent_id,
        role=message.role,
        content=content,
        name=message.name,
        model=None,  # assigned later?
        tool_calls=None,  # irrelevant
        tool_call_id=None,
        otid=message.otid,
        sender_id=message.sender_id,
        group_id=message.group_id,
    )
