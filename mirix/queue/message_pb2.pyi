import datetime
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class QueueMessage(_message.Message):
    __slots__ = ()
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    CHAINING_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    VERBOSE_FIELD_NUMBER: _ClassVar[int]
    FILTER_TAGS_FIELD_NUMBER: _ClassVar[int]
    USE_CACHE_FIELD_NUMBER: _ClassVar[int]
    OCCURRED_AT_FIELD_NUMBER: _ClassVar[int]
    LANGFUSE_TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    LANGFUSE_OBSERVATION_ID_FIELD_NUMBER: _ClassVar[int]
    LANGFUSE_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    LANGFUSE_USER_ID_FIELD_NUMBER: _ClassVar[int]
    actor: User
    agent_id: str
    input_messages: _containers.RepeatedCompositeFieldContainer[MessageCreate]
    chaining: bool
    user_id: str
    verbose: bool
    filter_tags: _struct_pb2.Struct
    use_cache: bool
    occurred_at: str
    langfuse_trace_id: str
    langfuse_observation_id: str
    langfuse_session_id: str
    langfuse_user_id: str
    def __init__(
        self,
        actor: _Optional[_Union[User, _Mapping]] = ...,
        agent_id: _Optional[str] = ...,
        input_messages: _Optional[_Iterable[_Union[MessageCreate, _Mapping]]] = ...,
        chaining: _Optional[bool] = ...,
        user_id: _Optional[str] = ...,
        verbose: _Optional[bool] = ...,
        filter_tags: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        use_cache: _Optional[bool] = ...,
        occurred_at: _Optional[str] = ...,
        langfuse_trace_id: _Optional[str] = ...,
        langfuse_observation_id: _Optional[str] = ...,
        langfuse_session_id: _Optional[str] = ...,
        langfuse_user_id: _Optional[str] = ...,
    ) -> None: ...

class User(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    ORGANIZATION_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TIMEZONE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    IS_DELETED_FIELD_NUMBER: _ClassVar[int]
    id: str
    organization_id: str
    name: str
    status: str
    timezone: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    is_deleted: bool
    def __init__(
        self,
        id: _Optional[str] = ...,
        organization_id: _Optional[str] = ...,
        name: _Optional[str] = ...,
        status: _Optional[str] = ...,
        timezone: _Optional[str] = ...,
        created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...,
        updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...,
        is_deleted: _Optional[bool] = ...,
    ) -> None: ...

class MessageCreate(_message.Message):
    __slots__ = ()

    class Role(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ROLE_UNSPECIFIED: _ClassVar[MessageCreate.Role]
        ROLE_USER: _ClassVar[MessageCreate.Role]
        ROLE_SYSTEM: _ClassVar[MessageCreate.Role]

    ROLE_UNSPECIFIED: MessageCreate.Role
    ROLE_USER: MessageCreate.Role
    ROLE_SYSTEM: MessageCreate.Role
    ROLE_FIELD_NUMBER: _ClassVar[int]
    TEXT_CONTENT_FIELD_NUMBER: _ClassVar[int]
    STRUCTURED_CONTENT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OTID_FIELD_NUMBER: _ClassVar[int]
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    role: MessageCreate.Role
    text_content: str
    structured_content: MessageContentList
    name: str
    otid: str
    sender_id: str
    group_id: str
    def __init__(
        self,
        role: _Optional[_Union[MessageCreate.Role, str]] = ...,
        text_content: _Optional[str] = ...,
        structured_content: _Optional[_Union[MessageContentList, _Mapping]] = ...,
        name: _Optional[str] = ...,
        otid: _Optional[str] = ...,
        sender_id: _Optional[str] = ...,
        group_id: _Optional[str] = ...,
    ) -> None: ...

class MessageContentList(_message.Message):
    __slots__ = ()
    PARTS_FIELD_NUMBER: _ClassVar[int]
    parts: _containers.RepeatedCompositeFieldContainer[MessageContentPart]
    def __init__(self, parts: _Optional[_Iterable[_Union[MessageContentPart, _Mapping]]] = ...) -> None: ...

class MessageContentPart(_message.Message):
    __slots__ = ()
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    CLOUD_FILE_FIELD_NUMBER: _ClassVar[int]
    text: TextContent
    image: ImageContent
    file: FileContent
    cloud_file: CloudFileContent
    def __init__(
        self,
        text: _Optional[_Union[TextContent, _Mapping]] = ...,
        image: _Optional[_Union[ImageContent, _Mapping]] = ...,
        file: _Optional[_Union[FileContent, _Mapping]] = ...,
        cloud_file: _Optional[_Union[CloudFileContent, _Mapping]] = ...,
    ) -> None: ...

class TextContent(_message.Message):
    __slots__ = ()
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class ImageContent(_message.Message):
    __slots__ = ()
    IMAGE_ID_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    image_id: str
    detail: str
    def __init__(self, image_id: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class FileContent(_message.Message):
    __slots__ = ()
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    file_id: str
    def __init__(self, file_id: _Optional[str] = ...) -> None: ...

class CloudFileContent(_message.Message):
    __slots__ = ()
    CLOUD_FILE_URI_FIELD_NUMBER: _ClassVar[int]
    cloud_file_uri: str
    def __init__(self, cloud_file_uri: _Optional[str] = ...) -> None: ...
