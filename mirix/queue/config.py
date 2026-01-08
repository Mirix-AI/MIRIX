"""
Configuration module for queue-sample
Reads settings from environment variables
"""
import os

# Queue type: 'memory' or 'kafka'
QUEUE_TYPE = os.environ.get('QUEUE_TYPE', 'memory').lower()

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.environ.get('KAFKA_TOPIC', 'queue-sample-topic')
KAFKA_GROUP_ID = os.environ.get('KAFKA_GROUP_ID', 'queue-sample-consumer-group')

# Kafka serialization format: 'protobuf' or 'json'
KAFKA_SERIALIZATION_FORMAT = os.environ.get('KAFKA_SERIALIZATION_FORMAT', 'protobuf').lower()

# Kafka SSL/TLS configuration (optional)
KAFKA_SECURITY_PROTOCOL = os.environ.get('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')
KAFKA_SSL_CAFILE = os.environ.get('KAFKA_SSL_CAFILE')
KAFKA_SSL_CERTFILE = os.environ.get('KAFKA_SSL_CERTFILE')
KAFKA_SSL_KEYFILE = os.environ.get('KAFKA_SSL_KEYFILE')


