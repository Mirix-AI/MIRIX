"""
Kafka queue implementation
Requires kafka-python and protobuf libraries to be installed
Supports both Protocol Buffers and JSON serialization
"""

import logging
from typing import Optional

from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.queue_interface import QueueInterface
from mirix.queue.queue_util import deserialize_queue_message, serialize_queue_message

logger = logging.getLogger(__name__)


class KafkaQueue(QueueInterface):
    """Kafka-based queue implementation supporting Protobuf and JSON serialization"""

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
        Initialize Kafka producer and consumer with configurable serialization

        Args:
            bootstrap_servers: Kafka broker address(es)
            topic: Kafka topic name
            group_id: Consumer group ID
            serialization_format: 'protobuf' or 'json' (default: 'protobuf')
            security_protocol: Kafka security protocol - 'PLAINTEXT', 'SSL', 'SASL_PLAINTEXT', 'SASL_SSL'
            ssl_cafile: Path to CA certificate file for SSL/TLS verification
            ssl_certfile: Path to client certificate file for mTLS
            ssl_keyfile: Path to client private key file for mTLS
            auto_offset_reset: Where to start consuming if no offset exists ('earliest' or 'latest')
                              - 'earliest': Start from beginning (default - ensures no messages missed)
                              - 'latest': Start from end (only new messages) - useful for "clearing queue"
                              Note: Changing group_id + setting to 'latest' can effectively clear the queue
            consumer_timeout_ms: Timeout for polling messages (milliseconds). Default: 1000
            max_poll_interval_ms: Max time between poll() calls before consumer is considered failed.
                                  Default: 900000 (15 minutes) to accommodate long-running memory agent ops
            session_timeout_ms: Timeout for detecting consumer failures. Default: 30000 (30 seconds)

        Note:
            enable_auto_commit is hardcoded to True and not configurable. This is coupled with the
            code's message processing logic - changing it would require manual commit implementation.
            (We should change it though.)
        """
        logger.info(
            "Initializing Kafka queue: servers=%s, topic=%s, group=%s, format=%s, security=%s",
            bootstrap_servers,
            topic,
            group_id,
            serialization_format,
            security_protocol,
        )
        logger.info(
            "Kafka consumer config: auto_offset_reset=%s, consumer_timeout_ms=%d, max_poll_interval_ms=%d, session_timeout_ms=%d",
            auto_offset_reset,
            consumer_timeout_ms,
            max_poll_interval_ms,
            session_timeout_ms,
        )

        try:
            from kafka import KafkaConsumer, KafkaProducer
        except ImportError:
            logger.error("kafka-python not installed")
            raise ImportError(
                "kafka-python is required for Kafka support. " "Install it with: pip install queue-sample[kafka]"
            )

        self.topic = topic
        self.serialization_format = serialization_format.lower()

        # Configure message serialization format
        value_serializer = lambda msg: serialize_queue_message(msg, format=self.serialization_format)
        value_deserializer = lambda data: deserialize_queue_message(data, format=self.serialization_format)

        logger.info(
            "Using %s serialization for Kafka messages",
            self.serialization_format.upper(),
        )

        # Build Kafka producer/consumer config with optional SSL
        kafka_config = {
            "bootstrap_servers": bootstrap_servers,
        }

        # Add SSL configuration if security protocol is SSL
        if security_protocol.upper() in ["SSL", "SASL_SSL"]:
            kafka_config["security_protocol"] = security_protocol.upper()
            if ssl_cafile:
                kafka_config["ssl_cafile"] = ssl_cafile
            if ssl_certfile:
                kafka_config["ssl_certfile"] = ssl_certfile
            if ssl_keyfile:
                kafka_config["ssl_keyfile"] = ssl_keyfile
            logger.info("Kafka SSL/TLS configured: protocol=%s", security_protocol)

        # Initialize Kafka producer with selected serializer and key serializer
        # Key serializer enables partition key routing for consistent message ordering per user
        self.producer = KafkaProducer(
            **kafka_config,
            key_serializer=lambda k: k.encode("utf-8"),  # Encode partition key to bytes
            value_serializer=value_serializer,
        )

        # Initialize Kafka consumer with selected deserializer
        # Use configurable timeouts for long-running operations (memory agent chains)
        self.consumer = KafkaConsumer(
            topic,
            **kafka_config,
            group_id=group_id,
            value_deserializer=value_deserializer,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=True,
            max_poll_interval_ms=max_poll_interval_ms,
            session_timeout_ms=session_timeout_ms,
            consumer_timeout_ms=consumer_timeout_ms,
        )

        logger.info(
            "Kafka consumer configured: auto_offset_reset=%s, max_poll_interval=%dms (%.1f min), session_timeout=%dms, consumer_timeout=%dms",
            auto_offset_reset,
            max_poll_interval_ms,
            max_poll_interval_ms / 60000,
            session_timeout_ms,
            consumer_timeout_ms,
        )

    def put(self, message: QueueMessage) -> None:
        """
        Send a message to Kafka topic with user_id as partition key.

        This ensures all messages for the same user go to the same partition,
        guaranteeing single-worker processing and message ordering per user.

        Implementation:
        - Uses user_id (or actor.id as fallback) as partition key
        - Kafka assigns partition via: hash(key) % num_partitions
        - Consumer group ensures only one worker per partition
        - Result: Same user always processed by same worker (no race conditions)

        Args:
            message: QueueMessage protobuf message to send
        """
        # Extract partition key: prefer user_id, then client_id
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

        # Send message with partition key - ensures consistent partitioning
        # Kafka will route this to: partition = hash(partition_key) % num_partitions
        future = self.producer.send(
            self.topic,
            key=partition_key,  # Partition key for consistent routing
            value=message,
        )
        future.get(timeout=10)  # Wait up to 10 seconds for confirmation

        logger.debug("Message sent to Kafka successfully with partition key: %s", partition_key)

    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from Kafka

        Args:
            timeout: Not used for Kafka (uses consumer_timeout_ms instead)

        Returns:
            QueueMessage protobuf message from Kafka

        Raises:
            StopIteration: If no message available
        """
        logger.debug("Polling Kafka topic %s for messages", self.topic)

        # Poll for messages
        for message in self.consumer:
            logger.debug("Retrieved message from Kafka: agent_id=%s", message.value.agent_id)
            return message.value

        # If no message received, raise exception (similar to queue.Empty)
        logger.debug("No message available from Kafka")
        raise StopIteration("No message available")

    def close(self) -> None:
        """Close Kafka producer and consumer connections"""
        logger.info("Closing Kafka connections")

        if hasattr(self, "producer"):
            self.producer.close()
            logger.debug("Kafka producer closed")
        if hasattr(self, "consumer"):
            self.consumer.close()
            logger.debug("Kafka consumer closed")
