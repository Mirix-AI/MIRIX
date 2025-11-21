# UAT Phase 1 关键问题分析与修复方案

**创建时间**: 2025-11-19
**最后更新**: 2025-11-19
**状态**: 🟡 Issue #1 Fixed, #2 & #3 In Progress

---

## 📋 问题总览

用户在 UAT 测试中发现了 3 个关键问题：

1. ✅ **Raw Memory 展示无用** - 显示文件路径而不是截图预览 → **FIXED & VALIDATED**
2. ⏳ **Memory References 看不到** - 前端没有显示引用徽章 → **需要用户验证**
3. ⏳ **新记忆未生成** - 实时截图不产生新的记忆 → **需要诊断**

---

## 🔍 问题 1: Raw Memory 展示无意义

### 当前行为
```
📸 /fake/screenshots/github_mirix_20251218_103000.png
```

用户看到的是：
- 文件路径的文本 ✗
- 没有截图预览 ✗
- Google Cloud 元数据（name='files/gw7my1j5wsrc'...） ✗

### 应该展示
```
[截图缩略图]
🌐 Chrome
🔗 https://github.com/user/mirix
📅 2025-12-18 10:30
📄 MIRIX Repository - Phase 1 Implementation...
```

### 根本原因

**前端代码** (`ExistingMemory.js:869-873`):
```javascript
{item.screenshot_path && (
  <div className="memory-screenshot-path">
    📸 {item.screenshot_path}  // ❌ 只显示路径文本
  </div>
)}
```

**后端 API** (`fastapi_server.py:1921-1930`):
```python
raw_items.append({
    "id": item.id,
    "screenshot_path": item.screenshot_path,  # ❌ 本地文件路径，前端无法访问
    "source_app": item.source_app,
    "source_url": item.source_url,
    "captured_at": item.captured_at.isoformat() if item.captured_at else None,
    "ocr_text": item.ocr_text,  # ✓ 完整文本，但应该是摘要
    "processed": item.processed,
    "created_at": item.created_at.isoformat() if item.created_at else None,
})
```

**问题**:
1. `screenshot_path` 是本地文件系统路径（如 `/fake/screenshots/...`）
2. 前端运行在浏览器中，无法访问本地文件系统
3. 即使是真实路径，浏览器也无法通过 `file://` 访问（安全限制）
4. Mock data 的路径是假的，文件根本不存在

### 修复方案

#### 方案 A: Screenshot Serve API（推荐）

**后端添加端点**:
```python
@app.get("/raw_memory/{raw_memory_id}/screenshot")
async def get_raw_memory_screenshot(raw_memory_id: str):
    """Serve screenshot image for a raw_memory"""
    with db_context() as session:
        item = session.get(RawMemoryItem, raw_memory_id)
        if not item or not item.screenshot_path:
            raise HTTPException(status_code=404, detail="Screenshot not found")

        # Check if file exists
        if not os.path.exists(item.screenshot_path):
            raise HTTPException(status_code=404, detail="Screenshot file not found")

        # Return image file
        from fastapi.responses import FileResponse
        return FileResponse(
            item.screenshot_path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"}
        )
```

**前端修改**:
```javascript
{item.screenshot_path && (
  <div className="memory-screenshot-preview">
    <img
      src={`${settings.serverUrl}/raw_memory/${item.id}/screenshot`}
      alt={`Screenshot from ${item.source_app}`}
      className="screenshot-thumbnail"
      onError={(e) => {
        e.target.style.display = 'none';
        e.target.nextSibling.style.display = 'block';
      }}
    />
    <div className="screenshot-fallback" style={{display: 'none'}}>
      📸 Screenshot unavailable
    </div>
  </div>
)}
```

**优点**:
- 不增加 API 响应大小
- 支持浏览器缓存
- 支持懒加载

#### 方案 B: Base64 Encoding

**后端修改**:
```python
import base64
from PIL import Image
import io

def get_screenshot_thumbnail(screenshot_path: str, max_width: int = 400) -> str:
    """Generate base64 encoded thumbnail"""
    if not os.path.exists(screenshot_path):
        return None

    # Open and resize image
    img = Image.open(screenshot_path)
    img.thumbnail((max_width, max_width * 2))

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_base64}"

# In API:
raw_items.append({
    ...
    "screenshot_thumbnail": get_screenshot_thumbnail(item.screenshot_path),
})
```

**缺点**:
- 增加 API 响应大小
- 无法利用浏览器缓存
- 每次请求都重新编码

#### 推荐: 方案 A + OCR 摘要优化

```python
raw_items.append({
    "id": item.id,
    "screenshot_url": f"/raw_memory/{item.id}/screenshot",  # ✓ 可访问的 URL
    "source_app": item.source_app,
    "source_url": item.source_url,
    "captured_at": item.captured_at.isoformat() if item.captured_at else None,
    "ocr_preview": item.ocr_text[:200] if item.ocr_text else "",  # ✓ 摘要
    "ocr_text": item.ocr_text,  # ✓ 完整文本（仅展开时使用）
    "processed": item.processed,
    "created_at": item.created_at.isoformat() if item.created_at else None,
})
```

### ✅ 修复验证结果 (2025-11-19)

**实施方案**: 方案 A (Screenshot Serve API)

**修改文件**:
1. `mirix/server/fastapi_server.py` (lines 1941-1990, 1918-1943)
2. `frontend/src/components/ExistingMemory.js` (lines 860-908)
3. `frontend/src/components/ExistingMemory.css` (lines 765-831)

**验证测试**:

✅ **Mock Data 测试**:
- 前端显示: "📸 Screenshot unavailable"
- API 返回: HTTP 404 (预期行为)
- 用户截图验证: 通过

✅ **真实文件测试**:
- 测试文件: `/Users/power/.mirix/tmp/images/screenshot-2025-09-05T06-30-37-992Z.png`
- 文件大小: 1.6 MB
- API 测试: HTTP 200, Content-Type: image/png, Size: 1,724,554 bytes
- 缓存头: Cache-Control: public, max-age=3600

**成功标准**:
- ✅ 截图通过 HTTP 正确返回
- ✅ 错误处理正常（文件不存在时显示 fallback）
- ✅ OCR 预览和展开功能正常
- ✅ 响应式样式正常
- ✅ 缓存优化正常

**详细报告**: 见 `UAT_FIX_VALIDATION.md`

**状态**: 🎉 **ISSUE #1 FULLY RESOLVED**

---

## 🔍 问题 2: Memory References 看不到

### 当前状态

**数据库检查结果**:
```sql
# ✓ 数据存在
SELECT id, name, raw_memory_references FROM semantic_memory
WHERE raw_memory_references IS NOT NULL;

-- 返回 6 条记录，包括:
-- "Cursor (AI Code Editor)" 引用了 20 个 raw_memory
-- "Python Async/Await Patterns" 引用了 1 个 raw_memory
```

**API 检查结果**:
- ✓ API 已修复（Task 21）
- ✓ `/memory/semantic` 返回 `raw_memory_references` 详情
- ✓ `/memory/episodic`, `/memory/procedural` 等也都返回

### 可能原因

1. **前端未刷新** - 用户可能在看旧的前端缓存
2. **Mock Data 问题** - Mock data 的 references 可能格式不对
3. **前端未展开** - References 只在"显示详情"后才显示

### 验证步骤

1. 刷新浏览器（Ctrl+Shift+R 强制刷新）
2. 打开 Memory Library → Semantic
3. 找到 "Cursor (AI Code Editor)" 或 "Python Async/Await Patterns"
4. 点击 "显示详情"
5. 查看是否有紫色的 Memory References 徽章

### 如果还是看不到

**检查浏览器控制台**:
```javascript
// F12 打开控制台，查看:
// 1. Network 面板 - /memory/semantic 请求是否成功
// 2. Console - 是否有 JavaScript 错误
// 3. Response - raw_memory_references 字段是否存在
```

**检查 API 返回**:
```bash
curl http://localhost:47283/memory/semantic | jq '.[0].raw_memory_references'
```

---

## 🔍 问题 3: 新记忆未生成

### 完整流程应该是

```
用户活动 (在 Chrome/Safari 等)
    ↓
Electron 截图监控 (每 N 秒截图)
    ↓
OCR 提取 (tesseract.js)
    ↓
Raw Memory 存储 (insert_raw_memory)
    ↓
发送给 Memory Agents (meta memory → specific agents)
    ↓
Semantic/Episodic Memory 创建 (带 raw_memory_references)
```

### 可能的断点

#### 1. 截图未触发
**检查**:
```bash
# 查看最新的 raw_memory
psql -U power -d mirix -c "SELECT id, source_app, captured_at FROM raw_memory ORDER BY captured_at DESC LIMIT 3;"
```

**原因**:
- Electron 截图监控未启动
- 截图间隔太长
- 截图保存失败

#### 2. Raw Memory 未创建
**检查**:
```bash
# 检查 raw_memory 数量增长
psql -U power -d mirix -c "SELECT COUNT(*) FROM raw_memory;"
# 等 1 分钟后再检查
```

**原因**:
- `TemporaryMessageAccumulator` 未调用 `insert_raw_memory`
- OCR 提取失败
- 数据库连接问题

#### 3. Memory Agents 未处理
**检查**:
```bash
# 查看未处理的 raw_memory
psql -U power -d mirix -c "SELECT COUNT(*) FROM raw_memory WHERE processed = false;"
```

**原因**:
- `SKIP_META_MEMORY_MANAGER` 配置问题
- Memory agents 未响应
- LLM API 配额耗尽

#### 4. Semantic Memory 未创建
**检查**:
```bash
# 查看最新的 semantic_memory
psql -U power -d mirix -c "SELECT id, name, created_at FROM semantic_memory ORDER BY created_at DESC LIMIT 3;"
```

**原因**:
- Memory tools 未调用
- raw_memory_references 未传递
- Agent 判断不需要创建记忆

### 诊断脚本

```bash
#!/bin/bash
# diagnose_memory_pipeline.sh

echo "=== 诊断记忆生成流程 ==="

echo ""
echo "1. Raw Memory 数量:"
psql -U power -d mirix -c "SELECT COUNT(*) as total, COUNT(*) FILTER (WHERE processed = true) as processed, COUNT(*) FILTER (WHERE processed = false) as pending FROM raw_memory;"

echo ""
echo "2. 最新的 3 条 Raw Memory:"
psql -U power -d mirix -c "SELECT id, source_app, captured_at, processed FROM raw_memory ORDER BY captured_at DESC LIMIT 3;"

echo ""
echo "3. Semantic Memory 数量:"
psql -U power -d mirix -c "SELECT COUNT(*) as total, COUNT(*) FILTER (WHERE raw_memory_references IS NOT NULL AND raw_memory_references != '[]') as with_references FROM semantic_memory;"

echo ""
echo "4. 最新的 3 条 Semantic Memory:"
psql -U power -d mirix -c "SELECT id, name, created_at, (raw_memory_references::text != '[]') as has_refs FROM semantic_memory ORDER BY created_at DESC LIMIT 3;"

echo ""
echo "5. SKIP_META_MEMORY_MANAGER 配置:"
grep "SKIP_META_MEMORY_MANAGER" mirix/agent/app_constants.py

echo ""
echo "6. 检查后端日志:"
tail -50 /tmp/mirix_server.log | grep -i "raw_memory\|memory_agent\|screenshot"
```

---

## 🛠️ 修复优先级

### P0 - Critical (立即修复)

1. **Raw Memory 截图展示**
   - 添加 `/raw_memory/{id}/screenshot` API
   - 修改前端显示 `<img>` 而不是路径文本
   - 添加 OCR 预览（200 字符）

### P1 - High (本周内)

2. **验证 Memory References 显示**
   - 用户强制刷新浏览器
   - 检查 API 返回数据
   - 如需要，添加调试日志

3. **诊断新记忆生成问题**
   - 运行诊断脚本
   - 检查每个流程断点
   - 记录发现的问题

### P2 - Medium (下周)

4. **优化 Mock Data**
   - 创建真实的截图文件
   - 确保所有 references 格式正确

---

## 📝 实施计划

### Step 1: 修复 Raw Memory 展示 (2小时)

**后端**:
```python
# 文件: mirix/server/fastapi_server.py

@app.get("/raw_memory/{raw_memory_id}/screenshot")
async def get_raw_memory_screenshot(raw_memory_id: str):
    """Serve screenshot for raw_memory"""
    # Implementation...

# 修改 /memory/raw API
# 返回 screenshot_url 而不是 screenshot_path
```

**前端**:
```javascript
// 文件: frontend/src/components/ExistingMemory.js
// Line 869-873

// 替换为:
{item.screenshot_url && (
  <div className="memory-screenshot-preview">
    <img
      src={`${settings.serverUrl}${item.screenshot_url}`}
      alt="Screenshot"
      className="screenshot-thumbnail"
    />
  </div>
)}
{item.ocr_preview && (
  <div className="memory-ocr-preview">
    {highlightText(item.ocr_preview, searchQuery)}
    {item.ocr_text && item.ocr_text.length > 200 && (
      <span className="read-more">... (点击展开查看完整内容)</span>
    )}
  </div>
)}
```

**CSS**:
```css
.screenshot-thumbnail {
  max-width: 100%;
  max-height: 300px;
  border-radius: 8px;
  margin: 10px 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.memory-ocr-preview {
  color: #666;
  font-size: 14px;
  line-height: 1.6;
  margin: 10px 0;
}
```

### Step 2: 验证 References 显示 (30分钟)

1. 强制刷新前端
2. 打开浏览器 DevTools
3. 检查 Network 和 Console
4. 验证 API 返回的数据格式
5. 如有问题，记录并修复

### Step 3: 诊断记忆生成 (1小时)

1. 运行诊断脚本
2. 检查每个流程节点
3. 查看后端日志
4. 记录发现的问题
5. 制定修复方案

---

## 🎯 成功标准

### Raw Memory 展示
- ✅ 能看到截图缩略图
- ✅ 能看到 source_app 图标
- ✅ 能看到 source_url 链接
- ✅ 能看到 OCR 文本摘要
- ✅ 点击展开能看到完整 OCR 文本

### Memory References
- ✅ Semantic Memory 中能看到紫色徽章
- ✅ 徽章显示 app 图标、URL、日期
- ✅ 点击徽章能跳转到 Raw Memory
- ✅ 跳转后能高亮目标项

### 新记忆生成
- ✅ 截图能自动触发
- ✅ Raw Memory 能自动创建
- ✅ Semantic Memory 能自动生成
- ✅ References 关系正确建立

---

**下一步**: 执行 Step 1 - 修复 Raw Memory 展示
