"""
Configuration module for queue-sample
Reads settings from environment variables
"""

import os

# Queue type: 'memory' or 'kafka'
QUEUE_TYPE = os.environ.get("QUEUE_TYPE", "memory").lower()

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "queue-sample-topic")
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "queue-sample-consumer-group")

# Kafka serialization format: 'protobuf' or 'json'
KAFKA_SERIALIZATION_FORMAT = os.environ.get("KAFKA_SERIALIZATION_FORMAT", "protobuf").lower()

# Kafka SSL/TLS configuration (optional)
KAFKA_SECURITY_PROTOCOL = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SSL_CAFILE = os.environ.get("KAFKA_SSL_CAFILE")
KAFKA_SSL_CERTFILE = os.environ.get("KAFKA_SSL_CERTFILE")
KAFKA_SSL_KEYFILE = os.environ.get("KAFKA_SSL_KEYFILE")

# Kafka consumer configuration
# auto_offset_reset: Where to start consuming if no offset exists
#   - 'earliest': Start from the beginning of the topic (default - ensures no messages are missed)
#   - 'latest': Start from the end (only new messages) - useful when "clearing the queue"
# Note: Changing group_id + setting auto_offset_reset='latest' can effectively "clear the queue"
KAFKA_AUTO_OFFSET_RESET = os.environ.get("KAFKA_AUTO_OFFSET_RESET", "earliest")

# consumer_timeout_ms: Default timeout for consumer.getone() (milliseconds)
# Used as the default timeout when polling for messages via aiokafka
KAFKA_CONSUMER_TIMEOUT_MS = int(os.environ.get("KAFKA_CONSUMER_TIMEOUT_MS", "1000"))

# max_poll_interval_ms: Maximum time between poll() calls before consumer is considered failed (milliseconds)
# Increased to 15 minutes to accommodate long-running memory agent operations
# Default aiokafka/Kafka value is 5 minutes (300000)
KAFKA_MAX_POLL_INTERVAL_MS = int(os.environ.get("KAFKA_MAX_POLL_INTERVAL_MS", "900000"))

# session_timeout_ms: Timeout for detecting consumer failures (milliseconds)
# If no heartbeat is received within this time, the consumer is removed from the group
# Increased to 30 seconds from default of 10 seconds
KAFKA_SESSION_TIMEOUT_MS = int(os.environ.get("KAFKA_SESSION_TIMEOUT_MS", "30000"))

NUM_WORKERS = int(os.environ.get("MIRIX_MEMORY_QUEUE_NUM_WORKERS", 1))
ROUND_ROBIN = os.environ.get("MIRIX_MEMORY_QUEUE_ROUND_ROBIN", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Batch-consumer knobs. The in-memory consumer is ALWAYS the BatchQueueWorker;
# there is no separate serial mode. READ_BATCH_SIZE=1 is the degenerate
# serial-equivalent (one message per pull, one user-group, through the same
# dispatch_save -> finalize chokepoint). These env-var names are deliberately
# shared with any external batch consumer's config so one set of values can
# tune both the in-memory consumer and the external one identically.
#
# - READ_BATCH_SIZE: max messages a worker accumulates per pull before running
#   the batch through process_batch. 1 == serial-equivalent.
# - MAX_IN_FLIGHT_USERS: cap on concurrently-processed user groups within a
#   batch (asyncio.Semaphore in process_batch). Per-user work stays serial.
# - FLUSH_INTERVAL_MS: how long to wait for the batch to fill before flushing a
#   partial batch. Kept small so a lone message (READ_BATCH_SIZE=1, or a
#   "save then observe" flow) dispatches promptly instead of stalling.
READ_BATCH_SIZE = int(os.environ.get("MIRIX_MEMORY_QUEUE_READ_BATCH_SIZE", 1))
MAX_IN_FLIGHT_USERS = int(os.environ.get("MIRIX_MEMORY_QUEUE_MAX_IN_FLIGHT_USERS", 15))
FLUSH_INTERVAL_MS = int(os.environ.get("MIRIX_MEMORY_QUEUE_FLUSH_INTERVAL_MS", 50))


# Worker auto-start configuration
# Set to 'false' or '0' to disable automatic worker startup when using external consumers
# When disabled, workers are created but not started - you must call process_external_message()
AUTO_START_WORKERS = os.environ.get("MIRIX_QUEUE_AUTO_START_WORKERS", "true").lower() in (
    "true",
    "1",
    "yes",
)
