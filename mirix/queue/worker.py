"""
Background worker that consumes messages from the queue.
Runs as an asyncio.Task in the main event loop (async-native).

Save dispatch is unified across all 3 run modes (numaflow / kafka /
in-memory) via `dispatch_incoming_message` (this file) →
`error_policy.dispatch_save`. Each mode does the same thing: receive a
message, normalize/validate it (filter_tags shape, memory_source_id/tid
backfill, malformed → permanent refusal), run the save under
process_with_policy, route the verdict through the single finalize
chokepoint.

* numaflow (external) — process_external_message in queue/__init__.py.
* internal kafka manual / in-memory sim — this file's BatchQueueWorker.

The only mode-specific behavior is HOW messages are obtained (broker
delivery vs in-process queue.get()) and what happens on
TRANSIENT_EXHAUSTED — numaflow could in principle let the broker
redeliver, but we treat exhausted-transient as dead-letter in all modes.
Broker-level redelivery is reserved for *process-death* cases (message
not ack'd), not for failures we classified.

ONE classifier (error_policy.classify), ONE policy
(error_policy.process_with_policy), ONE save dispatcher
(error_policy.dispatch_save), ONE finalize chokepoint
(memory_source_manager.finalize_source).

NOTE — boolean schema today. finalize_source records the SaveOutcome
in the log line but writes `processing_complete=True` for every outcome
(SUCCESS / PERMANENT_FAILURE / TRANSIENT_EXHAUSTED). A future
status-column migration will diversify column writes per outcome; this
file does not enforce "complete = success only" at the DB level.

Consumer topology: the in-memory / internal-kafka-manual consumer is the
BatchQueueWorker. Per loop iteration it accumulates up to
`READ_BATCH_SIZE` messages from its queue/partition (or flushes a partial
batch after `FLUSH_INTERVAL_MS`), then runs them through the shared
`mirix.queue.batch.process_batch` core (group-by-user, gather across groups
under `Semaphore(MAX_IN_FLIGHT_USERS)`, serial within a user). The
per-message work is the SAME `dispatch_save(_process_message_async)` chokepoint
the old serial loop used, so `READ_BATCH_SIZE=1` is behaviorally identical to
the deleted serial consumer (one message → one user-group → one dispatch_save
→ one finalize). There is no separate "serial mode".
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from google.protobuf.json_format import MessageToDict

from mirix.log import get_logger
from mirix.observability import (
    get_langfuse_client,
    mark_observation_as_child,
    restore_trace_from_queue_message,
)
from mirix.observability.context import (
    get_tid,
    get_trace_context,
)
from mirix.queue import config
from mirix.queue.batch import process_batch
from mirix.queue.error_policy import dispatch_save
from mirix.queue.message_pb2 import QueueMessage
from mirix.services.user_manager import UserManager
from mirix.utils import flatten_messages_for_agent

if TYPE_CHECKING:
    from mirix.schemas.client import Client
    from mirix.schemas.message import MessageCreate

    from .queue_interface import QueueInterface


logger = get_logger(__name__)


def derive_write_kind(message: QueueMessage) -> str:
    """Categorize a save by its payload shape (R2 trace tag ``write_kind:``).

    Fully determined by the message: a conversation payload (unified
    ``messages`` field, or the legacy packed ``input_messages``) means LLM
    extraction; ``direct_writes`` bypass the LLM pipeline. Both present →
    ``"mixed"`` — never one of the pure kinds, so eval tooling can select
    pure-extraction traces with a single exact-match filter.

    The no-payload combination is rejected upstream (ECMS validates that at
    least one is present before enqueueing); the helper stays total and maps
    it to ``"extraction"``.
    """
    has_messages = bool(message.messages or message.input_messages)
    has_direct = bool(message.direct_writes)
    if has_direct and has_messages:
        return "mixed"
    if has_direct:
        return "direct"
    return "extraction"


def reconcile_user_org_to_actor(user, actor):
    """Return ``user`` with its org corrected to the actor's (client's) org.

    The ``users`` row is a global, id-only-PK shared stub: its
    ``organization_id`` records whichever org first created the user, and
    per-org isolation lives on the child tables (blocks, memories). On the save
    path the resolved user therefore carries its *first-seen* org, not the org
    this save is for. Block scoping (``block_manager.get_blocks`` keys off
    ``user.organization_id``) and the temporal guard then operate against the
    stale org. The save's authoritative org is the client/actor's org, so
    correct it here, at the single point where the stub becomes the live user
    for this save.

    Returns a copy (never mutates the input). No-op when ``user`` is None, the
    actor has no org, or the orgs already match.
    """
    if user is None:
        return None
    actor_org = getattr(actor, "organization_id", None)
    if not actor_org or user.organization_id == actor_org:
        return user
    return user.model_copy(update={"organization_id": actor_org})


async def dispatch_incoming_message(worker: "QueueWorker", message: QueueMessage) -> None:
    """Shared consume-side entry for ALL THREE run modes (numaflow-external,
    internal kafka manual, in-memory sim).

    Normalizes/validates the incoming message (filter_tags/block_filter_tags
    canonicalization, memory_source_id + tid backfill — the work that used to
    exist only at the ECMS HTTP layer) and then runs it under the
    `dispatch_save` chokepoint. A message that fails validation is refused:
    a refused-to-process Langfuse span is emitted and a
    `QueueMessageRejectedError` (classified PERMANENT) dead-letters it via the
    same finalize path as any other permanent failure.

    Living here (not in the external consumer) is the point: the internal
    kafka-manual worker consumes the SAME topic a direct producer writes to,
    so the normalization contract must hold regardless of which consumer
    topology is deployed.
    """
    from mirix.queue.queue_util import normalize_and_validate_incoming_message

    normalization_error: Optional[ValueError] = None
    try:
        normalize_and_validate_incoming_message(message)
    except ValueError as e:
        normalization_error = e
        logger.error(
            "Rejecting malformed queue message: agent_id=%s, memory_source_id=%s: %s",
            message.agent_id,
            message.memory_source_id if message.HasField("memory_source_id") else None,
            e,
        )

    memory_source_id = message.memory_source_id if message.HasField("memory_source_id") else None

    async def _run_step() -> None:
        if normalization_error is not None:
            # Raised HERE (inside the dispatch_save-wrapped step) rather than
            # before it, so a deterministically-malformed message still goes
            # through classify() -> PERMANENT -> finalize_source, and the
            # consumer acks it instead of redelivering it forever.
            from mirix.errors import QueueMessageRejectedError
            from mirix.observability import restore_trace_from_queue_message
            from mirix.observability.skip_spans import emit_refused_to_process_span

            # The per-message processing (which normally restores trace
            # context) never runs on this path, so restore it here first —
            # otherwise the refusal span can't attach to the message's trace.
            # dispatch_save clears the context after finalize, as usual.
            restore_trace_from_queue_message(message)
            emit_refused_to_process_span(
                reason="malformed-message",
                metadata={
                    "agent_id": message.agent_id,
                    "memory_source_id": (message.memory_source_id if message.HasField("memory_source_id") else None),
                    "error": str(normalization_error),
                },
            )
            raise QueueMessageRejectedError(f"Malformed queue message: {normalization_error}") from normalization_error
        await worker.process_external_message(message)

    await dispatch_save(_run_step, memory_source_id=memory_source_id)


class QueueWorker:
    """Background worker that processes messages from the queue as an asyncio.Task"""

    def __init__(
        self,
        queue: "QueueInterface",
        server: Optional[Any] = None,
        partition_id: Optional[int] = None,
    ):
        """
        Initialize the queue worker.

        Args:
            queue: Async queue implementation to consume from
            server: Optional server instance to invoke APIs on
            partition_id: Optional partition ID for partitioned queues.
                         If set, worker will only consume from this partition.
        """
        logger.debug(
            "Initializing queue worker: server=%s, partition_id=%s",
            "provided" if server else "None",
            partition_id,
        )

        self.queue = queue
        self._server = server
        self._partition_id = partition_id
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def _convert_proto_user_to_pydantic(self, proto_user) -> "Client":
        """
        Convert protobuf User to Pydantic Client.

        The protobuf schema still uses "User" for historical reasons,
        but it represents a Client in the new architecture.
        """
        from mirix.schemas.client import Client

        return Client(
            id=proto_user.id,
            organization_id=(proto_user.organization_id if proto_user.organization_id else None),
            name=proto_user.name,
            status=proto_user.status,
            write_scope=None,
            read_scopes=[],
            created_at=(proto_user.created_at.ToDatetime() if proto_user.HasField("created_at") else datetime.now()),
            updated_at=(proto_user.updated_at.ToDatetime() if proto_user.HasField("updated_at") else datetime.now()),
            is_deleted=proto_user.is_deleted,
        )

    def _convert_proto_message_to_pydantic(self, proto_msg) -> "MessageCreate":
        """Convert protobuf MessageCreate to Pydantic MessageCreate."""
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import MessageCreate

        if proto_msg.role == proto_msg.ROLE_USER:
            role = MessageRole.user
        elif proto_msg.role == proto_msg.ROLE_SYSTEM:
            role = MessageRole.system
        else:
            role = MessageRole.user

        content = proto_msg.text_content if proto_msg.HasField("text_content") else ""

        kwargs = {}
        if hasattr(proto_msg, "external_message_id") and proto_msg.HasField("external_message_id"):
            kwargs["external_message_id"] = proto_msg.external_message_id
        if hasattr(proto_msg, "message_occurred_at") and proto_msg.HasField("message_occurred_at"):
            kwargs["message_occurred_at"] = proto_msg.message_occurred_at

        return MessageCreate(
            role=role,
            content=content,
            name=proto_msg.name if proto_msg.HasField("name") else None,
            otid=proto_msg.otid if proto_msg.HasField("otid") else None,
            sender_id=proto_msg.sender_id if proto_msg.HasField("sender_id") else None,
            group_id=proto_msg.group_id if proto_msg.HasField("group_id") else None,
            filter_tags=None,
            **kwargs,
        )

    @staticmethod
    def _convert_proto_source_message_to_dict(proto_msg) -> dict:
        """Convert a protobuf MessageCreate from source_messages to a plain dict.

        Unlike _convert_proto_message_to_pydantic (which produces Pydantic
        MessageCreate objects for agent processing), this returns a plain dict
        because source_messages need to preserve the "assistant" role which
        Pydantic MessageCreate doesn't allow (its role field is
        Literal["user", "system"]).

        The returned dict is consumed by _persist_memory_source() →
        normalize_message() which accepts dicts with any string role.
        """
        _ROLE_MAP = {
            proto_msg.ROLE_USER: "user",
            proto_msg.ROLE_SYSTEM: "system",
            proto_msg.ROLE_ASSISTANT: "assistant",
        }
        role = _ROLE_MAP.get(proto_msg.role, "user")
        content = proto_msg.text_content if proto_msg.HasField("text_content") else ""

        result = {"role": role, "content": content}

        if hasattr(proto_msg, "external_message_id") and proto_msg.HasField("external_message_id"):
            result["external_message_id"] = proto_msg.external_message_id
        if hasattr(proto_msg, "message_occurred_at") and proto_msg.HasField("message_occurred_at"):
            result["message_occurred_at"] = proto_msg.message_occurred_at
        if hasattr(proto_msg, "message_metadata") and proto_msg.message_metadata:
            result["metadata"] = MessageToDict(proto_msg.message_metadata)

        return result

    def set_server(self, server: Any) -> None:
        """Set or update the server instance."""
        self._server = server
        logger.info("Updated worker server instance")

    async def process_external_message(self, message: QueueMessage) -> None:
        """
        Process one already-deserialized QueueMessage.

        Named for its original (Numaflow/external-consumer) call site, but it
        is the per-message processing entry for every run mode — the shared
        `dispatch_incoming_message` funnel calls it for external, internal
        kafka manual, and in-memory messages alike.

        Args:
            message: QueueMessage protobuf already consumed from the transport
        """
        logger.debug(
            "Processing consumed message: agent_id=%s, user_id=%s",
            message.agent_id,
            message.user_id if message.HasField("user_id") else "None",
        )
        await self._process_message_async(message)

    async def _process_message_async(self, message: QueueMessage) -> None:
        """Process a queue message by calling server.send_messages()."""
        try:
            trace_restored = restore_trace_from_queue_message(message)
            if trace_restored:
                logger.debug("Restored trace context from queue message for processing")

            server = self._server

            if server is None:
                logger.warning(
                    "No server available - skipping message: agent_id=%s, message_count=%s",
                    message.agent_id,
                    len(message.messages) or len(message.input_messages),
                )
                return

            langfuse = get_langfuse_client()
            trace_context = get_trace_context()
            trace_id = trace_context.get("trace_id") if trace_context else None
            parent_span_id = trace_context.get("observation_id") if trace_context else None
            logger.debug(f"Queue worker trace context: trace_id={trace_id}, parent_span_id={parent_span_id}")

            client_id = message.client_id if message.client_id else None
            if not client_id:
                from mirix.errors import QueueMessageRejectedError
                from mirix.observability.skip_spans import emit_refused_to_process_span

                # Missing client_id is a deterministic producer bug (never
                # resolvable by retrying) — refuse and dead-letter, same
                # pattern as the no-write-scope refusal below, rather than a
                # bare ValueError which error_policy.classify() would default
                # to Transient and burn a full retry cycle first.
                emit_refused_to_process_span(
                    reason="missing-client-id",
                    metadata={
                        "agent_id": message.agent_id,
                        "memory_source_id": (
                            message.memory_source_id if message.HasField("memory_source_id") else None
                        ),
                    },
                )
                raise QueueMessageRejectedError(
                    f"Queue message for agent {message.agent_id} missing required client_id"
                )

            # Prefer the unified `messages` field (single per-turn wire copy,
            # see message.proto). The worker derives BOTH the packed
            # agent-input and the source_messages provenance records from it.
            # Falls back to the legacy input_messages (already packed) +
            # source_messages (original per-turn) pair for producers that
            # haven't migrated / messages already in flight during rollout.
            if message.messages:
                source_message_dicts = [self._convert_proto_source_message_to_dict(msg) for msg in message.messages]
                input_messages = flatten_messages_for_agent(source_message_dicts)
            else:
                input_messages = [self._convert_proto_message_to_pydantic(msg) for msg in message.input_messages]
                source_message_dicts = (
                    [self._convert_proto_source_message_to_dict(msg) for msg in message.source_messages]
                    if message.source_messages
                    else None
                )

            chaining = message.chaining if message.HasField("chaining") else True
            user_id = message.user_id if message.HasField("user_id") else None

            async def _resolve_actor_and_user():
                from mirix.errors import QueueMessageRejectedError
                from mirix.observability.skip_spans import emit_refused_to_process_span
                from mirix.orm.errors import NoResultFound

                try:
                    actor = await server.client_manager.get_client_by_id(client_id)
                except NoResultFound:
                    # A client_id that doesn't resolve is deterministic
                    # (retrying the same lookup won't make the row appear) —
                    # refuse and dead-letter immediately instead of defaulting
                    # to Transient and burning a retry cycle.
                    emit_refused_to_process_span(
                        reason="client-not-found",
                        metadata={
                            "client_id": client_id,
                            "memory_source_id": (
                                message.memory_source_id if message.HasField("memory_source_id") else None
                            ),
                        },
                    )
                    raise QueueMessageRejectedError(f"Client with id={client_id} not found in database")

                user_manager = UserManager()
                if user_id:
                    try:
                        user = await user_manager.get_user_by_id(user_id)
                    except Exception:
                        logger.info(
                            "User with id=%s not found, auto-creating with organization_id=%s",
                            user_id,
                            actor.organization_id,
                        )
                        from mirix.schemas.user import User as PydanticUser

                        try:
                            user = await user_manager.create_user(
                                pydantic_user=PydanticUser(
                                    id=user_id,
                                    name=user_id,
                                    organization_id=actor.organization_id,
                                    timezone=user_manager.DEFAULT_TIME_ZONE,
                                    status="active",
                                    is_deleted=False,
                                    is_admin=False,
                                )
                            )
                            logger.info(
                                "Auto-created user: %s in organization: %s",
                                user_id,
                                actor.organization_id,
                            )
                        except Exception as create_error:
                            logger.error(
                                "Failed to auto-create user with id=%s: %s. Falling back to admin user.",
                                user_id,
                                create_error,
                            )
                            user = await user_manager.get_admin_user()
                    return actor, reconcile_user_org_to_actor(user, actor)
                user = await user_manager.get_admin_user()
                return actor, reconcile_user_org_to_actor(user, actor)

            actor, user = await _resolve_actor_and_user()

            # Extract filter_tags from protobuf Struct (deep conversion to avoid ListValue/Value remnants)
            filter_tags = None
            if message.HasField("filter_tags") and message.filter_tags:
                filter_tags = MessageToDict(message.filter_tags)

            # The worker is the authority for write scope: "scope" is derived from
            # the client (actor) resolved by client_id, and any scope present on the
            # queue message is ignored and overwritten. A client with no write_scope
            # cannot create memories; this is a deterministic misconfiguration, so
            # raise a permanent error to dead-letter the message rather than burning
            # transient retries.
            if actor.write_scope is None:
                from mirix.errors import QueueMessageRejectedError
                from mirix.observability.skip_spans import (
                    emit_refused_to_process_span,
                )

                # Make the refusal explicit in the trace (parallel to the
                # idempotency-skip spans) so a read-only client's dropped save is
                # visible in Langfuse rather than looking like a silent failure.
                emit_refused_to_process_span(
                    reason="no-write-scope",
                    metadata={
                        "client_id": actor.id,
                        "memory_source_id": (
                            message.memory_source_id if message.HasField("memory_source_id") else None
                        ),
                    },
                )
                logger.warning(
                    "Refused to process: client %s has no write_scope - "
                    "cannot create memories (memory_source_id=%s)",
                    actor.id,
                    message.memory_source_id if message.HasField("memory_source_id") else None,
                )
                raise QueueMessageRejectedError(f"Client {actor.id} has no write_scope - cannot create memories")
            if filter_tags is None:
                filter_tags = {}
            filter_tags["scope"] = actor.write_scope

            use_cache = message.use_cache if message.HasField("use_cache") else True
            occurred_at = message.occurred_at if message.HasField("occurred_at") else None

            # Extract block_filter_tags (deep conversion to native Python types)
            block_filter_tags = None
            if hasattr(message, "block_filter_tags") and message.block_filter_tags:
                try:
                    block_filter_tags = MessageToDict(message.block_filter_tags)
                except Exception as e:
                    raise ValueError("block_filter_tags was provided but could not be parsed as a dict") from e

            block_filter_tags_update_mode = (
                message.block_filter_tags_update_mode if message.HasField("block_filter_tags_update_mode") else "merge"
            )

            # Extract memory source fields if present on the protobuf message
            memory_source_id = (
                message.memory_source_id
                if hasattr(message, "memory_source_id") and message.HasField("memory_source_id")
                else None
            )
            external_id = (
                message.external_id if hasattr(message, "external_id") and message.HasField("external_id") else None
            )
            external_thread_id = (
                message.external_thread_id
                if hasattr(message, "external_thread_id") and message.HasField("external_thread_id")
                else None
            )
            source_type = (
                message.source_type if hasattr(message, "source_type") and message.HasField("source_type") else None
            )
            source_system = (
                message.source_system
                if hasattr(message, "source_system") and message.HasField("source_system")
                else None
            )
            source_metadata = None
            if hasattr(message, "source_metadata") and message.source_metadata:
                try:
                    source_metadata = MessageToDict(message.source_metadata)
                except Exception:
                    pass
            summary = message.summary if hasattr(message, "summary") and message.HasField("summary") else None
            summarize = message.summarize if hasattr(message, "summarize") and message.HasField("summarize") else False

            # source_message_dicts was already derived up front (from the unified
            # `messages` field, or the legacy source_messages field as a fallback)
            # alongside input_messages — see the comment there. These dicts go
            # straight to _persist_memory_source() → normalize_message(), which
            # accepts dicts with any string role (including "assistant", which
            # Pydantic MessageCreate.role can't hold).

            # Extract direct_writes — each entry tells the meta-agent to call
            # the registered handler for memory_type instead of dispatching
            # sub-agents via the LLM. Payloads are opaque JSON blobs here;
            # the handler's signature validates their shape at call time.
            direct_writes = None
            if hasattr(message, "direct_writes") and message.direct_writes:
                import json as _json

                direct_writes = [
                    {
                        "memory_type": w.memory_type,
                        "payload": _json.loads(w.payload_json),
                    }
                    for w in message.direct_writes
                ]

            # Log the processing
            logger.info(
                "Processing message via server: agent_id=%s, client_id=%s (from actor), user_id=%s, input_messages_count=%s, use_cache=%s, filter_tags=%s, occurred_at=%s",
                message.agent_id,
                actor.id,
                user_id,
                len(input_messages),
                use_cache,
                filter_tags,
                occurred_at,
            )

            async def _do_send_messages():
                return await server.send_messages(
                    actor=actor,
                    agent_id=message.agent_id,
                    input_messages=input_messages,
                    chaining=chaining,
                    user=user,
                    filter_tags=filter_tags,
                    block_filter_tags=block_filter_tags,
                    block_filter_tags_update_mode=block_filter_tags_update_mode,
                    use_cache=use_cache,
                    occurred_at=occurred_at,
                    memory_source_id=memory_source_id,
                    external_id=external_id,
                    external_thread_id=external_thread_id,
                    source_type=source_type,
                    source_system=source_system,
                    source_metadata=source_metadata,
                    summary=summary,
                    summarize=summarize,
                    source_messages=source_message_dicts,
                    direct_writes=direct_writes,
                )

            # NOTE: this core does NOT classify errors or mark the source
            # complete. It runs the agent step and RAISES on any failure. Every
            # caller wraps it in dispatch_save (process_with_policy + the single
            # finalize chokepoint), so the raised failure is classified and the
            # source finalized exactly once:
            #   * external consumer path (queue/__init__.process_external_message).
            #   * internal batch consumer (BatchQueueWorker._run_one_iteration),
            #     per message inside the shared process_batch core.
            if langfuse and trace_id:
                from typing import cast

                from langfuse.types import TraceContext

                from mirix.observability.context import set_trace_context
                from mirix.observability.trace_attrs import (
                    get_write_counts,
                    update_trace_attributes,
                )

                # Write kind (R2) is fully determined by the message; derived
                # before the span opens so both the trace tag and any future
                # span field agree.
                write_kind = derive_write_kind(message)

                trace_context_dict: dict = {"trace_id": trace_id}

                # Root-span input (R3 AC3): the request's non-sensitive
                # parameters — ids, enums, and counts only. Conversation
                # content never appears here (pre-mask design).
                root_input = {
                    "message_count": len(input_messages),
                    "direct_write_count": len(direct_writes) if direct_writes else 0,
                    "memory_source_id": memory_source_id,
                    "source_type": source_type,
                    "source_system": source_system,
                    "external_thread_id": external_thread_id,
                    "filter_tag_keys": sorted(filter_tags.keys()),
                    "scope": actor.write_scope,
                    "summarize": summarize,
                    "has_caller_summary": bool(summary),
                    "occurred_at": str(occurred_at) if occurred_at else None,
                    "agent_id": message.agent_id,
                    "user_id": user_id,
                }

                with langfuse.start_as_current_observation(
                    name="Meta Agent",
                    as_type="agent",
                    trace_context=cast(TraceContext, trace_context_dict),
                    input=root_input,
                    metadata={
                        "agent_id": message.agent_id,
                        "message_count": len(input_messages),
                        "source": "queue_worker",
                        # TID on the root worker span so every span in this
                        # save's tree can be correlated back to the request.
                        "tid": get_tid(),
                    },
                ) as span:
                    mark_observation_as_child(span)

                    # Surface TID / client / write-kind at the TRACE level
                    # (tags = filterable in the Langfuse dashboard, metadata =
                    # visible). This is the worker trace, decoupled from the
                    # HTTP-entry trace, so it needs its own trace-level tags.
                    # Written through the accumulate-and-rewrite helper so this
                    # write is a strict superset of whatever the HTTP entry
                    # wrote on a stitched trace (last-writer-wins safety); the
                    # helper never raises.
                    _worker_tid = get_tid()
                    _trace_tags = []
                    _trace_metadata: dict = {}
                    if _worker_tid:
                        _trace_tags.append(f"tid:{_worker_tid}")
                        _trace_metadata["tid"] = _worker_tid
                    # Client tag from the resolved actor's registered name —
                    # omitted when falsy (R1 AC2: no placeholder values).
                    if actor.name:
                        _trace_tags.append(f"client:{actor.name}")
                        _trace_metadata["client"] = actor.name
                    _trace_tags.append(f"write_kind:{write_kind}")
                    _trace_metadata["write_kind"] = write_kind
                    update_trace_attributes(tags=_trace_tags, metadata=_trace_metadata)

                    span_observation_id = getattr(span, "id", None)
                    if span_observation_id:
                        set_trace_context(
                            trace_id=trace_id,
                            observation_id=span_observation_id,
                            user_id=trace_context.get("user_id"),
                            session_id=trace_context.get("session_id"),
                        )
                    usage = await _do_send_messages()

                    # Root-span output (R3 AC4): what the save produced. An
                    # idempotency-skipped save reads step_count=0 with {}
                    # writes — the explicit "skips read as successes" shape.
                    try:
                        span.update(
                            output={
                                "step_count": usage.step_count if usage else 0,
                                "writes_by_memory_type": get_write_counts(),
                            }
                        )
                    except Exception as e:
                        logger.debug("Failed to set Meta Agent span output: %s", e)
            else:
                usage = await _do_send_messages()

            logger.debug(
                "Successfully processed message: agent_id=%s, usage=%s",
                message.agent_id,
                usage.model_dump() if usage else "None",
            )

        except Exception as e:
            # Log here for context, then RE-RAISE. This core is the shared
            # execution path for both the external consumer and the internal
            # BatchQueueWorker; both wrap it in dispatch_save's
            # process_with_policy, which needs to SEE the exception to classify
            # it. Previously this block swallowed without re-raising, which
            # meant process_with_policy NEVER saw failures — its
            # classify/mark-permanent/redeliver logic was effectively dead.
            # Re-raising lets the shared policy apply.
            logger.error(
                "Error processing message for agent_id=%s: %s",
                message.agent_id,
                e,
                exc_info=True,
            )
            raise
        # NOTE: the TID/trace context restored at the top of this method is
        # deliberately NOT cleared here. dispatch_save (the per-save boundary
        # for every run mode) clears it after finalize_source runs, so the
        # "Finalized memory_source=... outcome=..." log line still carries the
        # TID. Clearing here — before finalize — is what used to leave that
        # line stamped tid=-.


class BatchQueueWorker(QueueWorker):
    """In-memory / internal-kafka-manual consumer that pulls messages in
    batches and runs them through the shared :func:`process_batch` core.

    Replaces the deleted serial ``QueueWorker._consume_loop``. It reuses every
    per-message helper from :class:`QueueWorker` (``_process_message_async``,
    the ``_convert_proto_*`` converters, ``set_server``,
    ``process_external_message``) — only HOW messages are obtained and grouped
    changes here.

    Per loop iteration (:meth:`_run_one_iteration`):

      1. :meth:`_collect_batch` accumulates up to ``config.READ_BATCH_SIZE``
         messages from this worker's queue/partition, OR returns the partial
         batch once ``config.FLUSH_INTERVAL_MS`` has elapsed since the first
         message arrived. An empty queue yields an empty batch (idle no-op).
      2. The batch runs through :func:`process_batch`, grouped by
         :meth:`_user_key` (the message's ``user_id`` if present, else None),
         with ``max_in_flight_users=config.MAX_IN_FLIGHT_USERS``.
      3. The per-message ``process`` wraps ``_process_message_async`` in
         ``dispatch_save`` with the message's ``memory_source_id`` — the SAME
         chokepoint the serial loop used.

    ``config.*`` is read at call time (not bound at import) so env/monkeypatch
    changes take effect without re-importing.

    ``READ_BATCH_SIZE=1`` ⇒ one message per pull ⇒ one user-group ⇒ one
    ``dispatch_save`` ⇒ one finalize: behaviorally identical to the old serial
    consumer.

    Config values are snapshotted ONCE at the start of each iteration so a mid
    iteration env change cannot desync the collect cap from the dispatch cap.
    """

    # How long a single queue.get poll blocks while accumulating a batch. The
    # flush deadline (FLUSH_INTERVAL_MS) governs when a partial batch is
    # released; this just bounds how often we wake to re-check the deadline /
    # the running flag.
    _POLL_TIMEOUT_SECONDS = 0.05

    @staticmethod
    def _user_key(message: QueueMessage) -> Optional[str]:
        """Grouping key for :func:`process_batch`: the message's ``user_id`` if
        set, else None (None-keyed messages share one serial 'unknown' group)."""
        return message.user_id if message.HasField("user_id") else None

    async def _get_one(self, timeout: float) -> QueueMessage:
        """Pull a single message from this worker's queue or partition."""
        if self._partition_id is not None and hasattr(self.queue, "get_from_partition"):
            return await self.queue.get_from_partition(self._partition_id, timeout=timeout)
        return await self.queue.get(timeout=timeout)

    async def _collect_batch(self) -> List[QueueMessage]:
        """Accumulate up to ``READ_BATCH_SIZE`` messages, or flush the partial
        batch after ``FLUSH_INTERVAL_MS`` since the first message arrived.

        Returns an empty list if no message arrives within the flush window
        (idle). The flush timer starts only once the FIRST message of a batch
        has been pulled — an idle worker is bounded by a single flush-window
        wait, then loops.
        """
        read_batch_size = max(1, config.READ_BATCH_SIZE)
        flush_interval_s = max(0.0, config.FLUSH_INTERVAL_MS / 1000.0)

        batch: List[QueueMessage] = []
        deadline: Optional[float] = None

        # Accumulation is NOT gated on self._running: a stop() cancels the
        # consume task, so a CancelledError propagates out of _get_one and
        # unwinds this collect cleanly. Gating here would make a directly
        # invoked _collect_batch (and the first iteration before the loop sets
        # the flag) return empty.
        loop = asyncio.get_running_loop()
        while len(batch) < read_batch_size:
            if deadline is None:
                # No message yet — wait up to one flush window for the first.
                poll = max(self._POLL_TIMEOUT_SECONDS, flush_interval_s)
            else:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break  # flush window elapsed → release the partial batch
                poll = min(self._POLL_TIMEOUT_SECONDS, remaining)

            try:
                message = await self._get_one(poll)
            except asyncio.TimeoutError:
                if deadline is None and not batch:
                    # Still idle after a full flush window with nothing pulled.
                    break
                continue

            batch.append(message)
            if deadline is None:
                deadline = loop.time() + flush_interval_s

        return batch

    async def _run_one_iteration(self) -> None:
        """Collect one batch and run it through the shared batch core.

        Snapshots ``MAX_IN_FLIGHT_USERS`` once so the dispatch cap matches the
        batch that was just collected. No-op (no ``process_batch`` call) when
        the batch is empty.
        """
        batch = await self._collect_batch()
        if not batch:
            return

        max_in_flight_users = max(1, config.MAX_IN_FLIGHT_USERS)

        distinct_users = len({self._user_key(m) for m in batch})
        partition_info = f", partition={self._partition_id}" if self._partition_id is not None else ""
        logger.debug(
            "[BATCH] processing%s: batch_size=%d, distinct_users=%d, max_in_flight_users=%d",
            partition_info,
            len(batch),
            distinct_users,
            max_in_flight_users,
        )

        async def _process(message: QueueMessage) -> None:
            # SAME funnel as the external consumer: normalize/validate, then
            # run the agent step under dispatch_save (classify + bounded retry
            # + single finalize). See dispatch_incoming_message for why the
            # internal paths must normalize too.
            await dispatch_incoming_message(self, message)

        await process_batch(
            batch,
            user_key=self._user_key,
            process=_process,
            max_in_flight_users=max_in_flight_users,
        )

    async def _batch_consume_loop(self) -> None:
        """Async consume loop running as an asyncio.Task in the main event loop.

        Each iteration collects a batch (:meth:`_collect_batch`) and runs it
        through :func:`process_batch`. A genuinely unexpected error in the loop
        body is swallowed (logged) so the worker task keeps running — same
        defense-in-depth as the old serial loop. ``dispatch_save`` already
        finalizes its own classified failures; we do NOT finalize here because
        a protocol error gives us no per-save outcome.
        """
        partition_info = f", partition={self._partition_id}" if self._partition_id is not None else ""
        logger.info("Batch queue worker task started%s", partition_info)

        while self._running:
            try:
                await self._run_one_iteration()
            except asyncio.CancelledError:
                logger.info("Batch queue worker task cancelled%s", partition_info)
                break
            except Exception as e:
                logger.error("Error in batch consumption loop: %s", e, exc_info=True)

    async def start(self) -> None:
        """Start the background batch worker as an asyncio.Task."""
        if self._running:
            logger.warning("Batch queue worker already running")
            return

        partition_info = f" (partition {self._partition_id})" if self._partition_id is not None else ""
        logger.info(
            "Starting batch queue worker task%s (read_batch_size=%d, max_in_flight_users=%d, flush_interval_ms=%d)...",
            partition_info,
            config.READ_BATCH_SIZE,
            config.MAX_IN_FLIGHT_USERS,
            config.FLUSH_INTERVAL_MS,
        )
        self._running = True

        task_name = f"BatchQueueWorker-{self._partition_id}" if self._partition_id is not None else "BatchQueueWorker"
        self._task = asyncio.create_task(self._batch_consume_loop(), name=task_name)

        logger.info("Batch queue worker task%s started successfully", partition_info)

    async def stop(self, close_queue: bool = True) -> None:
        """
        Stop the background worker task.

        Args:
            close_queue: Whether to close the queue resources. Set to False
                        when multiple workers share the same queue.
        """
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if close_queue:
            await self.queue.close()
