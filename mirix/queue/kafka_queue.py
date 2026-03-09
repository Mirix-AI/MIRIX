"""
Kafka queue implementation using aiokafka (async-native).
Supports both Protocol Buffers and JSON serialization.
"""

import asyncio
import logging
import ssl
from typing import Optional

from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.queue_interface import QueueInterface
from mirix.queue.queue_util import deserialize_queue_message, serialize_queue_message

logger = logging.getLogger(__name__)


class KafkaQueue(QueueInterface):
    """Async-native Kafka queue using aiokafka"""

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        serialization_format: str = "protobuf",
        security_protocol: str = "PLAINTEXT",
        ssl_cafile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
        auto_offset_reset: str = "earliest",
        consumer_timeout_ms: int = 1000,
        max_poll_interval_ms: int = 900000,
        session_timeout_ms: int = 30000,
    ):
        """
        Store Kafka configuration. Actual connection happens in start().

        Args:
            bootstrap_servers: Kafka broker address(es)
            topic: Kafka topic name
            group_id: Consumer group ID
            serialization_format: 'protobuf' or 'json' (default: 'protobuf')
            security_protocol: 'PLAINTEXT', 'SSL', 'SASL_PLAINTEXT', 'SASL_SSL'
            ssl_cafile: Path to CA certificate file
            ssl_certfile: Path to client certificate file for mTLS
            ssl_keyfile: Path to client private key file for mTLS
            auto_offset_reset: 'earliest' (default) or 'latest'
            consumer_timeout_ms: Kept for config compatibility (used as getone timeout)
            max_poll_interval_ms: Max time between poll() calls. Default: 900000 (15 min)
            session_timeout_ms: Timeout for detecting consumer failures. Default: 30000
        """
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError:
            raise ImportError(
                "aiokafka is required for Kafka support. "
                "Install it with: pip install aiokafka"
            )

        logger.info(
            "Initializing Kafka queue: servers=%s, topic=%s, group=%s, format=%s, security=%s",
            bootstrap_servers,
            topic,
            group_id,
            serialization_format,
            security_protocol,
        )

        self.topic = topic
        self.serialization_format = serialization_format.lower()
        self._consumer_timeout_s = consumer_timeout_ms / 1000.0

        value_serializer = lambda msg: serialize_queue_message(msg, format=self.serialization_format)
        value_deserializer = lambda data: deserialize_queue_message(data, format=self.serialization_format)

        logger.info(
            "Using %s serialization for Kafka messages",
            self.serialization_format.upper(),
        )

        ssl_context = None
        if security_protocol.upper() in ["SSL", "SASL_SSL"]:
            ssl_context = ssl.create_default_context(cafile=ssl_cafile)
            if ssl_certfile and ssl_keyfile:
                ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)
            logger.info("Kafka SSL/TLS configured: protocol=%s", security_protocol)

        self.producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            security_protocol=security_protocol.upper(),
            ssl_context=ssl_context,
            key_serializer=lambda k: k.encode("utf-8"),
            value_serializer=value_serializer,
        )

        self.consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            security_protocol=security_protocol.upper(),
            ssl_context=ssl_context,
            value_deserializer=value_deserializer,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=True,
            max_poll_interval_ms=max_poll_interval_ms,
            session_timeout_ms=session_timeout_ms,
        )

        logger.info(
            "Kafka consumer configured: auto_offset_reset=%s, max_poll_interval=%dms (%.1f min), session_timeout=%dms",
            auto_offset_reset,
            max_poll_interval_ms,
            max_poll_interval_ms / 60000,
            session_timeout_ms,
        )

    async def start(self) -> None:
        """Connect producer and consumer to Kafka brokers."""
        logger.info("Starting aiokafka producer and consumer...")
        await self.producer.start()
        await self.consumer.start()
        logger.info("Kafka producer and consumer started")

    async def put(self, message: QueueMessage) -> None:
        """
        Send a message to Kafka topic with user_id as partition key.

        Ensures all messages for the same user go to the same partition,
        guaranteeing single-worker processing and message ordering per user.
        """
        if message.user_id:
            partition_key = message.user_id
        elif message.client_id:
            partition_key = message.client_id
        else:
            raise ValueError("Queue message missing partition key: must have user_id or client_id")

        logger.debug(
            "Sending message to Kafka topic %s: agent_id=%s, partition_key=%s",
            self.topic,
            message.agent_id,
            partition_key,
        )

        await self.producer.send_and_wait(
            self.topic,
            key=partition_key,
            value=message,
        )

        logger.debug("Message sent to Kafka successfully with partition key: %s", partition_key)

    async def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from Kafka.

        Args:
            timeout: Timeout in seconds (defaults to consumer_timeout_ms from config)

        Raises:
            asyncio.TimeoutError: If no message available within timeout
        """
        effective_timeout = timeout if timeout is not None else self._consumer_timeout_s
        logger.debug("Polling Kafka topic %s for messages (timeout=%.1fs)", self.topic, effective_timeout)

        try:
            record = await asyncio.wait_for(
                self.consumer.getone(), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            logger.debug("No message available from Kafka within timeout")
            raise

        logger.debug("Retrieved message from Kafka: agent_id=%s", record.value.agent_id)
        return record.value

    async def close(self) -> None:
        """Stop Kafka producer and consumer gracefully."""
        logger.info("Closing Kafka connections")
        try:
            await self.producer.stop()
            logger.debug("Kafka producer stopped")
        except Exception as e:
            logger.warning("Error stopping Kafka producer: %s", e)
        try:
            await self.consumer.stop()
            logger.debug("Kafka consumer stopped")
        except Exception as e:
            logger.warning("Error stopping Kafka consumer: %s", e)
