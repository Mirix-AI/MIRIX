# Add JSON Serialization and SSL/TLS Support for Kafka Queue

## 📋 Summary

This PR adds **JSON serialization** and **SSL/TLS** support to MIRIX's Kafka queue implementation, enabling integration with Intuit Event Bus while maintaining 100% backward compatibility with existing Protobuf serialization.

## 🎯 Motivation

**Context and Memory Service (ECMS)** is integrating with **Intuit Event Bus**, which requires:
1. ✅ **JSON message format** (Event Bus standard)
2. ✅ **mTLS authentication** (production security requirement)

**Current State:** MIRIX Kafka queue only supports:
- ❌ Protobuf binary serialization (incompatible with Event Bus)
- ❌ PLAINTEXT connections (no production-ready security)

**This PR enables:**
- ✅ JSON serialization (Event Bus compatible)
- ✅ SSL/TLS with mTLS (production ready)
- ✅ Configurable via environment variables
- ✅ 100% backward compatible (Protobuf + PLAINTEXT remain defaults)

## 📁 Files Changed (3 files)

### 1. `mirix/queue/config.py`
**Added 5 new environment variables:**

```python
# Serialization format
KAFKA_SERIALIZATION_FORMAT = os.environ.get("KAFKA_SERIALIZATION_FORMAT", "protobuf")

# SSL/TLS configuration
KAFKA_SECURITY_PROTOCOL = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SSL_CAFILE = os.environ.get("KAFKA_SSL_CAFILE")
KAFKA_SSL_CERTFILE = os.environ.get("KAFKA_SSL_CERTFILE")
KAFKA_SSL_KEYFILE = os.environ.get("KAFKA_SSL_KEYFILE")
```

**Why:** Provides configuration interface for JSON and SSL features

---

### 2. `mirix/queue/kafka_queue.py`
**Added:**
- `json_serializer()` - Converts Protobuf → JSON → bytes
- `json_deserializer()` - Converts bytes → JSON → Protobuf
- New constructor parameters:
  - `serialization_format: str = 'protobuf'`
  - `security_protocol: str = 'PLAINTEXT'`
  - `ssl_cafile`, `ssl_certfile`, `ssl_keyfile` (Optional)
- Format selection logic (JSON vs Protobuf)
- SSL configuration for KafkaProducer/KafkaConsumer

**Why:** Implements JSON serialization and SSL/TLS support

**Example Flow:**
```
QueueMessage (Protobuf) 
  → json_serializer() 
  → JSON string 
  → bytes 
  → Kafka
  → bytes 
  → json_deserializer() 
  → QueueMessage (Protobuf)
```

---

### 3. `mirix/queue/manager.py`
**Updated `_create_queue()` to:**
- Build kwargs dict with serialization format and SSL parameters
- Pass all config to `KafkaQueue(**kwargs)`
- Conditionally add SSL params only if provided

**Why:** Connects config → KafkaQueue initialization

---

## 🚀 Usage Examples

### Scenario 1: Local Development (PLAINTEXT + JSON)
```bash
export QUEUE_TYPE=kafka
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_SERIALIZATION_FORMAT=json
export KAFKA_SECURITY_PROTOCOL=PLAINTEXT
```

**Result:** JSON messages on local Docker Kafka

---

### Scenario 2: E2E/Prod Event Bus (SSL + JSON)
```bash
export QUEUE_TYPE=kafka
export KAFKA_BOOTSTRAP_SERVERS=broker1:9094,broker2:9094
export KAFKA_SERIALIZATION_FORMAT=json
export KAFKA_SECURITY_PROTOCOL=SSL
export KAFKA_SSL_CAFILE=/tmp/ca.pem
export KAFKA_SSL_CERTFILE=/tmp/cert.pem
export KAFKA_SSL_KEYFILE=/tmp/key.pem
```

**Result:** JSON messages on secure Kafka with mTLS

---

### Scenario 3: Default (Unchanged Behavior)
```bash
export QUEUE_TYPE=kafka
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
# No KAFKA_SERIALIZATION_FORMAT → defaults to protobuf
# No KAFKA_SECURITY_PROTOCOL → defaults to PLAINTEXT
```

**Result:** Protobuf messages on PLAINTEXT Kafka (existing behavior)

---

## 📊 Message Format Comparison

### Protobuf (Default)
```
Binary: \x0a\x10client-uuid\x12\x05agent\x1a\x04user...
Size: ~250 bytes (compact)
Human-readable: ❌ No
Event Bus compatible: ❌ No
```

### JSON (New)
```json
{
  "client_id": "client-d18089e4-3b28-43ed-b45b-53bdfeac82cb",
  "agent_id": "agent-3e229396-d7d1-4b0f-81d6-12f7ce2eb4d5",
  "user_id": "9411863178471709",
  "actor": {
    "id": "9411863178471709",
    "type": "USER"
  },
  "messages": [
    {
      "role": "USER",
      "content": "What is QuickBooks?"
    }
  ],
  "timestamp": "2026-01-08T08:31:03.020080"
}
```
```
Size: ~400 bytes (larger but readable)
Human-readable: ✅ Yes
Event Bus compatible: ✅ Yes
Kafka UI compatible: ✅ Yes (can view in browser)
```

---

## ✅ Testing

**Tested locally with ECMS:**
- ✅ Local Kafka (Docker) with JSON + PLAINTEXT
- ✅ Messages successfully produced and consumed
- ✅ Verified in Kafka UI (JSON visible in browser)
- ✅ ECMS `/v1/memories` endpoint → Kafka → MIRIX worker processing

**Backward Compatibility:**
- ✅ Default Protobuf behavior unchanged
- ✅ Existing deployments continue to work
- ✅ No breaking changes

---

## 🔄 Backward Compatibility

| Aspect | Before | After | Breaking? |
|--------|--------|-------|-----------|
| Default serialization | Protobuf | Protobuf | ❌ No |
| Default security | PLAINTEXT | PLAINTEXT | ❌ No |
| Config required? | Yes | Yes | ❌ No |
| JSON support | ❌ No | ✅ Yes | ➕ Feature |
| SSL support | ❌ No | ✅ Yes | ➕ Feature |

**Conclusion:** 100% backward compatible - all changes are opt-in via environment variables.

---

## 🎯 ECMS Integration

This PR enables **ECMS** to integrate with **Intuit Event Bus** by:

1. **ECMS sets environment variables** (`common/mirix_adapter.py`):
   ```python
   os.environ["KAFKA_SERIALIZATION_FORMAT"] = "json"
   os.environ["KAFKA_SECURITY_PROTOCOL"] = "SSL"
   # ... SSL cert paths from IDPS
   ```

2. **MIRIX reads config** (`mirix/queue/config.py`):
   ```python
   KAFKA_SERIALIZATION_FORMAT = "json"
   KAFKA_SECURITY_PROTOCOL = "SSL"
   ```

3. **MIRIX creates Kafka queue** (`mirix/queue/manager.py`):
   ```python
   KafkaQueue(serialization_format="json", security_protocol="SSL", ...)
   ```

4. **Messages flow to Event Bus** in JSON format with mTLS!

---

## 📦 Versioning

**Version will be set at publish time:**
```bash
python scripts/packaging/setup_server.py \
    --package-name jl-ecms-server \
    --version 0.20.0 \
    sdist bdist_wheel
```

**Targeting:** `0.20.x` (per Lucas's guidance to avoid conflicts with his `0.19.x` work)

**Package:** `jl-ecms-server==0.20.0`

---

## 🔗 Related Links

- **ECMS Kafka Integration PR:** https://github.intuit.com/expertise-help/context-and-memory-service/pull/43
- **MIRIX Repo:** https://github.com/LiaoJianhe/MIRIX_Intuit (re-org branch)
- **Event Bus Docs:** [Intuit Event Bus Guide]

---

## 👥 Reviewers

@LucasParzych @LiaoJianhe

---

## 💡 Questions?

Feel free to ask any questions or suggest improvements!

---

## 📝 Checklist

- [x] Code changes implemented
- [x] Backward compatibility maintained
- [x] Tested locally with ECMS
- [x] No version bump in code (will be set at publish time)
- [x] PR description written
- [x] Ready for review
