# Health Check 调试指南

## 问题描述
前端报错: `❌ Backend health check failed: The user aborted a request.`

## 已添加的调试日志

### 1. App.js 增强日志
现在health check会输出：
```
🔍 Starting health check: http://localhost:47283
📡 Sending fetch request...
📥 Response received in XXms, status: 200
✅ Backend is healthy
```

如果失败会显示：
```
❌ Backend health check failed: [error message]
   Error name: [error name]
   Error type: [error type]
   → Request was aborted (timeout or cancelled)
```

### 2. Timeout调整
- **旧值**: 5秒
- **新值**: 30秒
- **原因**: 排除超时可能性

## 调试步骤

### Step 1: 刷新前端页面
1. 打开Electron应用
2. 按 `Cmd+R` 或 `Ctrl+R` 刷新
3. 打开开发者工具: `Cmd+Option+I`
4. 查看Console标签

### Step 2: 观察日志输出
查找以下内容：

**成功的health check**:
```
🔍 Starting health check: http://localhost:47283
📡 Sending fetch request...
📥 Response received in 2.50ms, status: 200
✅ Backend is healthy - hiding loading modal
```

**失败的health check**:
```
🔍 Starting health check: http://localhost:47283
📡 Sending fetch request...
❌ Backend health check failed: The user aborted a request.
   Error name: AbortError
   Error type: DOMException
   → Request was aborted (timeout or cancelled)
```

### Step 3: 检查并发调用
查看是否有多个health check同时运行：
```
Health check already in progress, skipping...
```

如果看到这个，说明有并发调用。

### Step 4: 检查网络请求
1. 打开开发者工具的 **Network** 标签
2. 过滤请求: 输入 `health`
3. 观察 `/health` 请求:
   - Status: 应该是 `200`
   - Time: 应该小于50ms
   - Preview: 应该显示 `{"status":"healthy",...}`

### Step 5: 测试服务器直接响应
在终端运行：
```bash
curl -w "\nTime: %{time_total}s\n" http://localhost:47283/health
```

应该看到：
```json
{"status":"healthy","agent_initialized":true,"timestamp":"..."}
Time: 0.001s
```

## 可能的原因

### 原因 1: 真实超时
- **症状**: 日志显示 "Health check timeout (30s)"
- **解决**: 检查服务器是否卡死

### 原因 2: React严格模式
- **症状**: 组件挂载两次，第一次请求被取消
- **解决**: 暂时禁用严格模式（仅用于调试）

### 原因 3: Electron窗口事件
- **症状**: 窗口切换时多次调用health check
- **解决**: 添加debounce或更严格的并发控制

### 原因 4: AbortController被意外调用
- **症状**: 请求立即被abort，没有timeout警告
- **解决**: 检查是否有其他代码调用了abort()

## 预期行为

### 正常情况
```
🔍 Starting health check: http://localhost:47283
📡 Sending fetch request...
📥 Response received in 2.50ms, status: 200
✅ Backend is healthy - hiding loading modal
```

### 服务器停止
```
🔍 Starting health check: http://localhost:47283
📡 Sending fetch request...
⏱️  Health check timeout (30s) - aborting
❌ Backend health check failed: The user aborted a request.
   Error name: AbortError
   → Request was aborted (timeout or cancelled)
```

## 下一步

1. **刷新页面** 查看新日志
2. **截图控制台** 发送给我看
3. **检查Network标签** 确认请求状态
4. **观察时间戳** 看是否有并发调用

如果问题仍然存在，我们可以：
- 完全移除AbortController看是否还有问题
- 添加请求ID来追踪每个请求
- 使用防抖(debounce)来减少并发调用
