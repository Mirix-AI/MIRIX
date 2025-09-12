# LLM 调试日志记录功能

这个功能提供了详细的 LLM 调用和响应日志记录，帮助诊断和解决 LLM API 调用中的问题。

## 功能特性

- 📝 **详细的请求记录**: 记录发送给 LLM 的完整请求内容
- 📥 **完整的响应记录**: 记录 LLM 返回的完整响应内容
- ⏱️ **性能监控**: 记录请求响应时间
- ❌ **错误跟踪**: 详细记录错误信息和上下文
- 🔧 **JSON 解析错误诊断**: 专门记录 JSON 解析失败的情况
- 📁 **文件日志**: 将日志保存到文件中，便于后续分析
- 🖥️ **控制台输出**: 实时显示关键信息

## 使用方法

### 方法 1: 使用环境变量

```bash
# 启用调试日志记录
source enable_llm_debug.sh

# 运行测试
python -m tests.run_specific_memory_test episodic_memory_indirect --config mirix/configs/mirix_qwen3.yaml
```

### 方法 2: 在代码中启用

```python
from mirix.llm_api.llm_debug_logger import enable_llm_debug_logging

# 启用调试日志记录
enable_llm_debug_logging(
    log_dir="./logs/llm_debug",
    enable_file_logging=True
)

# 运行你的测试代码
```

## 日志文件说明

调试日志会保存到 `./logs/llm_debug/` 目录下：

- `llm_requests_YYYYMMDD.log`: 所有 LLM 请求的详细记录
- `llm_responses_YYYYMMDD.log`: 所有 LLM 响应的详细记录
- `llm_errors_YYYYMMDD.log`: 所有错误和异常的详细记录
- `session_*.json`: 完整的调试会话记录（如果使用会话功能）

## 日志内容示例

### 请求日志
```json
{
  "timestamp": "2025-01-08T12:30:00.000Z",
  "request_id": "req_1704715800000",
  "model_name": "qwen-plus",
  "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "request_data": {
    "model": "qwen-plus",
    "messages": [...],
    "tools": [...],
    "temperature": 0.6,
    "max_tokens": 4096
  },
  "additional_info": {
    "tools_count": 5,
    "messages_count": 3,
    "stream": false
  }
}
```

### 响应日志
```json
{
  "timestamp": "2025-01-08T12:30:01.500Z",
  "request_id": "req_1704715800000",
  "response_data": {
    "choices": [...],
    "usage": {
      "prompt_tokens": 1000,
      "completion_tokens": 200,
      "total_tokens": 1200
    }
  },
  "response_time_ms": 1500.25,
  "additional_info": {
    "success": true
  }
}
```

### 错误日志
```json
{
  "timestamp": "2025-01-08T12:30:01.500Z",
  "request_id": "req_1704715800000",
  "error_type": "JSONDecodeError",
  "error_message": "Extra data: line 1 column 473 (char 472)",
  "error_context": {
    "stage": "response_conversion",
    "response_data_keys": ["choices", "usage"]
  }
}
```

## 控制台输出示例

```
🚀 LLM Request [req_1704715800000]
   Model: qwen-plus
   Endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1
   Messages Count: 3
   Message 1: user - This is a test memory about going to the grocery store...
   Message 2: assistant - I understand you want to store this information...
   Message 3: user - Please add this to episodic memory
   Tools: 5 tools available
     - episodic_memory_insert
     - episodic_memory_search
     - episodic_memory_update
     - episodic_memory_delete
     - finish_memory_update

📥 LLM Response [req_1704715800000]
   Response Time: 1500.25ms
   Choices Count: 1
   Choice 1: assistant
     Content: I'll help you store this information in episodic memory...
     Tool Calls: 1
       Tool 1: episodic_memory_insert
         Args: {"items": [{"actor": "user", "details": "The user mentioned going to the grocery store..."}]}
   Usage: 1000 prompt + 200 completion = 1200 total
```

## 故障排除

### 常见问题

1. **JSON 解析错误**
   - 查看 `llm_errors_*.log` 文件中的 JSON 解析错误详情
   - 检查 LLM 返回的 JSON 格式是否正确

2. **API 调用失败**
   - 查看请求日志确认发送的数据格式
   - 查看错误日志了解具体的错误原因

3. **响应时间过长**
   - 查看响应日志中的 `response_time_ms` 字段
   - 分析是否有网络或模型性能问题

### 调试技巧

1. **关联请求和响应**: 使用 `request_id` 字段关联请求和响应
2. **分析错误模式**: 查看错误日志中的重复错误模式
3. **性能分析**: 使用响应时间数据分析性能瓶颈
4. **内容验证**: 检查请求和响应内容是否符合预期

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_DEBUG_ENABLE_FILE` | `true` | 是否启用文件日志记录 |
| `LLM_DEBUG_LOG_DIR` | `./logs/llm_debug` | 日志文件保存目录 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 注意事项

- 调试日志记录会增加一些性能开销
- 日志文件可能会变得很大，建议定期清理
- 敏感信息（如 API 密钥）会被自动过滤
- 在生产环境中建议关闭详细的调试日志记录
