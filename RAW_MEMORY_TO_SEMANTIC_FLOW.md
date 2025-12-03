# 从截图到语义记忆的完整数据流

**文档日期**: 2025-11-19
**目的**: 详细解释 MIRIX 如何将用户的截图转化为结构化的语义记忆

---

## 📊 流程总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        用户活动监控层                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    1. 截图捕获 (Electron/系统监控)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        OCR 处理层                                         │
│  - tesseract.js 提取文本                                                 │
│  - OCRUrlExtractor 提取 URLs                                             │
│  - 生成 embedding (text-embedding-3-small)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    2. Raw Memory 存储
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL: raw_memory 表                              │
│  ✓ screenshot_path (本地文件路径)                                        │
│  ✓ source_app (应用名称，如 Chrome)                                      │
│  ✓ source_url (提取的 URL)                                               │
│  ✓ captured_at (时间戳)                                                  │
│  ✓ ocr_text (OCR 提取的完整文本)                                         │
│  ✓ ocr_text_embedding (向量，用于语义搜索)                               │
│  ✓ google_cloud_url (云存储 URI，用于 Gemini 多模态)                     │
│  ✓ processed = false (标记未处理)                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    3. Message Queue 分发
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Meta Memory Agent                                     │
│  🤖 LLM: Gemini 2.0 Flash                                                │
│  📋 作用: 分析截图内容，决定发送给哪些 Memory Agent                       │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    4. 分发到专门的 Memory Agents
                                  │
        ┌─────────────┬───────────┴───────────┬────────────┐
        │             │                       │            │
        ▼             ▼                       ▼            ▼
┌──────────────┐ ┌──────────────┐  ┌──────────────┐ ┌──────────────┐
│  Semantic    │ │  Episodic    │  │ Procedural   │ │  Resource    │
│  Memory      │ │  Memory      │  │  Memory      │ │  Memory      │
│  Agent       │ │  Agent       │  │  Agent       │ │  Agent       │
│              │ │              │  │              │ │              │
│ 🤖 LLM       │ │ 🤖 LLM       │  │ 🤖 LLM       │ │ 🤖 LLM       │
│ 📝 概念/知识 │ │ 📅 事件      │  │ 🔧 流程/步骤 │ │ 📚 资源/链接 │
└──────────────┘ └──────────────┘  ┌──────────────┘ └──────────────┘
        │             │                       │            │
        │         5. LLM 分析截图，调用工具函数            │
        │             │                       │            │
        └─────────────┴───────────┬───────────┴────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Memory Tools 工具函数                               │
│  • semantic_memory_insert(items, raw_memory_references)                 │
│  • episodic_memory_insert(items, raw_memory_references)                 │
│  • procedural_memory_insert(items, raw_memory_references)               │
│  • resource_memory_insert(items, raw_memory_references)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    6. 写入数据库 (带 raw_memory_references)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL: semantic_memory 表                         │
│  ✓ name (名称，如 "Cursor (AI Code Editor)")                            │
│  ✓ summary (摘要)                                                        │
│  ✓ details (详细信息)                                                    │
│  ✓ source (来源)                                                         │
│  ✓ raw_memory_references = [rawmem-uuid1, rawmem-uuid2, ...]           │
│  ✓ embedding (语义向量)                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    7. 标记 Raw Memory 为已处理
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              更新 raw_memory.processed = true                             │
│              processing_count += 1                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    8. 前端展示
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         用户界面                                          │
│  📚 Semantic Memory: "Cursor (AI Code Editor)"                          │
│     └─ References: [📸 Chrome | 🔗 cursor.com | 📅 Nov 19]              │
│     └─ 点击 → 跳转到 Raw Memory 查看原始截图                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 详细流程说明

### Phase 1: 截图捕获 (Electron 层)

**触发方式**:
- 定时截图（每 N 秒）
- 应用切换时截图
- 用户手动触发

**捕获信息**:
```javascript
{
  image_path: "/Users/power/.mirix/tmp/images/screenshot-2025-11-19.png",
  source_app: "Chrome",  // 当前活跃应用
  timestamp: "2025-11-19T10:24:49",
  screen_info: {...}
}
```

**代码位置**: Electron main process

---

### Phase 2: OCR 处理

**处理组件**: `OCRUrlExtractor` (`mirix/helpers/ocr_url_extractor.py`)

**处理步骤**:
1. **文本提取** - tesseract.js OCR
   ```python
   ocr_text, urls = OCRUrlExtractor.extract_urls_and_text(image_path)
   # ocr_text: "@ Chrome Xx #8 ... youtube.com/watch?v=..."
   # urls: ["youtube.com/watch?v=VDREHIOd80k"]
   ```

2. **URL 提取** - 正则表达式匹配
   ```python
   source_url = urls[0] if urls else None
   # source_url: "youtube.com/watch?v=VDREHIOd80k"
   ```

3. **Embedding 生成** (如果启用)
   ```python
   from mirix.embeddings import embedding_model
   embed_model = embedding_model(EmbeddingConfig.default_config("text-embedding-3-small"))
   ocr_text_embedding = embed_model.get_text_embedding(ocr_text)
   # ocr_text_embedding: [0.123, -0.456, ...] (1536 维向量)
   ```

**代码位置**: `mirix/agent/temporary_message_accumulator.py:609-643`

---

### Phase 3: Raw Memory 存储

**存储组件**: `RawMemoryManager` (`mirix/services/raw_memory_manager.py`)

**调用代码** (`temporary_message_accumulator.py:623-636`):
```python
raw_memory = raw_memory_manager.insert_raw_memory(
    actor=self.client.user,
    screenshot_path=image_path,
    source_app=source_app,
    captured_at=captured_at,
    ocr_text=ocr_text if ocr_text else None,
    source_url=source_url,
    google_cloud_url=google_cloud_url,
    metadata={
        "batch_index": idx,
        "total_in_batch": len(image_uris),
    },
    organization_id=self.client.user.organization_id,
)

raw_memory_ids.append(raw_memory.id)
# raw_memory_ids: ["rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028", ...]
```

**数据库写入** (`raw_memory_manager.py:82-103`):
```python
raw_memory = RawMemoryItem(
    id=f"rawmem-{uuid.uuid4()}",
    screenshot_path=screenshot_path,
    source_app=source_app,
    captured_at=captured_at,
    ocr_text=ocr_text,
    source_url=source_url,
    google_cloud_url=google_cloud_url,
    metadata_=metadata or {},
    ocr_text_embedding=ocr_text_embedding,  # 向量存储
    embedding_config=embedding_config_dict,
    processed=False,  # 🔴 重要: 标记为未处理
    processing_count=0,
    user_id=actor.id,
    organization_id=organization_id or actor.organization_id,
)

session.add(raw_memory)
session.commit()
```

**此时数据库状态**:
```sql
SELECT id, source_app, processed FROM raw_memory ORDER BY captured_at DESC LIMIT 1;
-- id: rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028
-- source_app: Chrome
-- processed: false  ⬅️ 等待 Memory Agent 处理
```

---

### Phase 4: Message Queue 分发

**分发组件**: `TemporaryMessageAccumulator` + `MessageQueue`

**构建消息** (`temporary_message_accumulator.py:688-750`):
```python
message_parts = [
    {
        "type": "text",
        "text": "The following are the screenshots taken from the computer of the user:"
    },
    {
        "type": "text",
        "text": "These are the screenshots from Chrome:"
    },
    {
        "type": "text",
        "text": "Timestamp: 2025-11-19T10:24:49"
    },
    {
        "type": "google_cloud_file_uri",  # Gemini 模型
        "google_cloud_file_uri": "gs://bucket/file.png"
    }
    # 或者 OpenAI 模型:
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,iVBORw0KG..."
        }
    }
]
```

**附加 raw_memory_ids**:
```python
payloads = {
    'agent_states': agent_states,
    'payloads': {
        'message_parts': message_parts,
        'raw_memory_ids': raw_memory_ids,  # ⬅️ 关键: 传递 raw_memory 引用
        'voice_transcription': voice_transcription
    }
}
```

**发送到队列**:
```python
# 不再直接发送给每个 agent，而是发给 Meta Memory Agent
# Meta Memory Agent 决定哪些信息需要存储到哪些 memory type
```

---

### Phase 5: Meta Memory Agent 分析

**Agent 类型**: `MetaMemoryAgent` (继承自 `Agent`)

**使用的 LLM**: Gemini 2.0 Flash (默认)

**系统提示** (简化版):
```
You are the Meta Memory Manager.

Your role:
1. Analyze the screenshots and context
2. Determine what should be stored in different memory types:
   - Semantic Memory: Concepts, knowledge, facts
   - Episodic Memory: Events, activities, timeline
   - Procedural Memory: How-to, workflows, steps
   - Resource Memory: Links, references, tools

3. Call the appropriate memory tools with raw_memory_references

Important:
- ALWAYS include raw_memory_references when calling memory tools
- raw_memory_references should be the list of rawmem-* IDs
```

**LLM 分析示例**:
```
Input: Screenshot showing YouTube page about AI

LLM 思考过程:
1. 这是用户在 Chrome 上浏览 YouTube
2. 主题是关于 AI 的视频
3. 这可以作为:
   - Episodic Memory: "用户观看了关于 AI 的视频"
   - Resource Memory: "YouTube 视频链接"
   - Semantic Memory: 如果内容有知识价值
```

**LLM 调用工具**:
```python
# LLM 生成的函数调用
episodic_memory_insert({
    "items": [
        {
            "occurred_at": "2025-11-19T10:24:49",
            "event_type": "browsing",
            "actor": "user",
            "summary": "Watched YouTube video about AI",
            "details": "User browsed to youtube.com/watch?v=VDREHIOd80k",
            "raw_memory_references": ["rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028"]
        }
    ]
})

resource_memory_insert({
    "items": [
        {
            "title": "AI Video on YouTube",
            "summary": "Educational video about AI",
            "resource_type": "video",
            "content": "https://youtube.com/watch?v=VDREHIOd80k",
            "raw_memory_references": ["rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028"]
        }
    ]
})
```

**是否使用 LLM**:
- ✅ **是的！Meta Memory Agent 使用 LLM 分析截图**
- 📸 LLM 可以"看到"截图（通过 Google Cloud File URI 或 base64）
- 📝 LLM 可以"读取" OCR 文本
- 🧠 LLM 根据内容智能决定如何分类和存储

---

### Phase 6: Specialized Memory Agents 处理

**Semantic Memory Agent 示例**:

**Agent 配置**:
```python
class SemanticMemoryAgent(Agent):
    # 继承 Agent 基类
    # 自动获得 memory tools
```

**系统提示** (简化版):
```
You are the Semantic Memory Manager.

Your role:
- Extract factual knowledge from screenshots
- Create semantic memory items for:
  - Concepts: "What is X?"
  - Facts: "X is used for Y"
  - Relationships: "X is related to Y"

When creating semantic memory:
1. Use semantic_memory_insert() tool
2. ALWAYS include raw_memory_references
3. Be concise and factual
```

**LLM 调用示例**:
```python
# LLM 看到截图显示 "Cursor AI Code Editor"
semantic_memory_insert({
    "items": [
        {
            "name": "Cursor (AI Code Editor)",
            "summary": "AI-powered code editor",
            "details": "Cursor is a fork of VSCode with built-in AI capabilities...",
            "source": "screenshots",
            "tree_path": "/Tools/AI/Code Editors",
            "raw_memory_references": [
                "rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028",
                "rawmem-10c55c46-af44-477e-a8cc-41238372c85e",
                ...  # 可能引用多个截图
            ]
        }
    ]
})
```

**工具函数实现** (`mirix/functions/function_sets/memory_tools.py:334-356`):
```python
def semantic_memory_insert(self: "Agent", items: List[SemanticMemoryItemBase]):
    """
    The tool to insert items into semantic memory.
    """
    for item in items:
        self.semantic_memory_manager.insert_semantic_item(
            agent_state=self.agent_state,
            name=item["name"],
            summary=item["summary"],
            details=item["details"],
            source=item["source"],
            tree_path=item["tree_path"],
            organization_id=self.user.organization_id,
            actor=self.user,
            raw_memory_references=item.get("raw_memory_references"),  # ⬅️ 关键
        )
```

---

### Phase 7: 数据库写入 (带 References)

**SemanticMemoryManager 实现** (简化):
```python
def insert_semantic_item(
    self,
    agent_state,
    name,
    summary,
    details,
    source,
    tree_path,
    actor,
    organization_id,
    raw_memory_references=None,  # ⬅️ 接收 references
):
    # 生成 embedding
    combined_text = f"{name}\n{summary}\n{details}"
    embedding = embed_model.get_text_embedding(combined_text)

    # 创建记录
    semantic_item = SemanticMemoryItem(
        id=f"semantic-{uuid.uuid4()}",
        name=name,
        summary=summary,
        details=details,
        source=source,
        tree_path=tree_path,
        embedding=embedding,
        raw_memory_references=raw_memory_references,  # ⬅️ 存储 references
        user_id=actor.id,
        organization_id=organization_id,
    )

    session.add(semantic_item)
    session.commit()
```

**数据库状态**:
```sql
SELECT id, name, raw_memory_references FROM semantic_memory
WHERE name = 'Cursor (AI Code Editor)';

-- Result:
-- id: semantic-abc123...
-- name: Cursor (AI Code Editor)
-- raw_memory_references: ["rawmem-6e711fee...", "rawmem-10c55c46...", ...]
```

---

### Phase 8: 标记 Raw Memory 已处理

**标记函数** (`raw_memory_manager.py:136-159`):
```python
def mark_as_processed(self, raw_memory_id: str) -> bool:
    with self.session_maker() as session:
        raw_memory = session.get(RawMemoryItem, raw_memory_id)

        if raw_memory:
            raw_memory.processed = True
            raw_memory.processing_count += 1
            raw_memory.last_modify = {
                "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                "operation": "marked_processed",
            }
            session.commit()
            return True

        return False
```

**数据库更新**:
```sql
-- Before:
SELECT id, processed, processing_count FROM raw_memory
WHERE id = 'rawmem-6e711fee...';
-- processed: false, processing_count: 0

-- After:
-- processed: true, processing_count: 1
```

---

### Phase 9: API 返回给前端

**API 端点**: `GET /memory/semantic`

**返回数据** (`fastapi_server.py`):
```json
{
  "id": "semantic-abc123",
  "name": "Cursor (AI Code Editor)",
  "summary": "AI-powered code editor",
  "details": "Cursor is a fork of VSCode...",
  "raw_memory_references": [
    {
      "id": "rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028",
      "source_app": "Chrome",
      "source_url": "https://cursor.com",
      "captured_at": "2025-11-19T10:24:49",
      "screenshot_url": "/raw_memory/rawmem-6e711fee.../screenshot"
    },
    {
      "id": "rawmem-10c55c46-af44-477e-a8cc-41238372c85e",
      "source_app": "Chrome",
      "source_url": "https://cursor.com/features",
      "captured_at": "2025-11-19T10:25:12",
      "screenshot_url": "/raw_memory/rawmem-10c55c46.../screenshot"
    }
  ],
  "created_at": "2025-11-19T10:25:30"
}
```

---

### Phase 10: 前端展示

**React 组件**: `ExistingMemory.js`

**显示效果**:
```jsx
<div className="semantic-memory-item">
  <h3>Cursor (AI Code Editor)</h3>
  <p>AI-powered code editor</p>

  {/* Memory References 徽章 */}
  <div className="memory-references">
    <span className="reference-badge" onClick={() => jumpToRawMemory('rawmem-6e711fee...')}>
      🌐 Chrome | 🔗 cursor.com | 📅 Nov 19 10:24
    </span>
    <span className="reference-badge" onClick={() => jumpToRawMemory('rawmem-10c55c46...')}>
      🌐 Chrome | 🔗 cursor.com/features | 📅 Nov 19 10:25
    </span>
  </div>

  <button onClick={() => expandDetails()}>显示详情</button>
</div>
```

**点击徽章跳转**:
```javascript
function jumpToRawMemory(rawMemoryId) {
  // 切换到 Raw Memory 标签
  setActiveTab('raw');

  // 滚动到对应记录
  const element = document.getElementById(`raw-memory-${rawMemoryId}`);
  element.scrollIntoView();

  // 高亮显示
  element.classList.add('highlighted');

  // 自动展开查看截图
  toggleExpanded(rawMemoryId);
}
```

---

## 🔑 关键要点总结

### 1. LLM 的作用

**Meta Memory Agent (LLM 驱动)**:
- ✅ 分析截图内容（视觉理解）
- ✅ 理解 OCR 文本（语义理解）
- ✅ 决定信息分类（知识、事件、流程、资源）
- ✅ 调用工具函数存储记忆
- ✅ 传递 `raw_memory_references` 建立关联

**Specialized Memory Agents (LLM 驱动)**:
- ✅ Semantic Memory Agent: 提取概念和知识
- ✅ Episodic Memory Agent: 识别事件和活动
- ✅ Procedural Memory Agent: 总结流程和步骤
- ✅ Resource Memory Agent: 整理链接和资源

### 2. 数据流关键点

| 阶段 | 数据格式 | 存储位置 | 是否有 LLM |
|------|---------|---------|-----------|
| 1. 截图捕获 | 图片文件 | 本地文件系统 | ❌ |
| 2. OCR 处理 | 文本 + URLs | 内存 | ❌ |
| 3. Raw Memory 存储 | 结构化数据 | PostgreSQL | ❌ |
| 4. Message Queue | JSON payload | 内存队列 | ❌ |
| 5. Meta Memory 分析 | 多模态输入 | LLM 内存 | ✅ Gemini 2.0 |
| 6. Memory Agent 处理 | 工具调用 | LLM 内存 | ✅ Gemini 2.0 |
| 7. 语义记忆存储 | 结构化数据 + references | PostgreSQL | ❌ |
| 8. 前端展示 | JSON API | HTTP Response | ❌ |

### 3. Raw Memory References 的关键作用

**为什么重要**:
1. **可追溯性**: 用户可以验证记忆的来源
2. **透明度**: 看到 AI 基于什么做出判断
3. **调试**: 开发者可以追踪数据流
4. **信任**: 用户信任有证据支持的记忆

**如何传递**:
```python
# 1. 截图 → Raw Memory
raw_memory_ids = []
for screenshot in screenshots:
    raw_memory = insert_raw_memory(screenshot)
    raw_memory_ids.append(raw_memory.id)

# 2. 传递给 LLM
message = build_message(screenshots, raw_memory_ids)

# 3. LLM 调用工具时包含
semantic_memory_insert({
    "items": [{
        "name": "...",
        "raw_memory_references": raw_memory_ids  # ⬅️ 关键
    }]
})

# 4. 存储到数据库
semantic_memory.raw_memory_references = raw_memory_ids

# 5. API 返回详细信息
api_response = {
    "raw_memory_references": [
        fetch_raw_memory_details(id) for id in raw_memory_ids
    ]
}

# 6. 前端展示徽章
render_reference_badges(raw_memory_references)
```

### 4. 配置开关

**控制 Memory Agent 是否运行**:
```python
# mirix/agent/app_constants.py
SKIP_META_MEMORY_MANAGER = False  # True 则跳过 Memory Agent

# 如果设为 True:
# - Raw Memory 会存储 ✅
# - 但不会生成 Semantic/Episodic/Procedural Memory ❌
# - 用于调试或节省 LLM 成本
```

**控制 Embedding 生成**:
```python
# mirix/constants.py
BUILD_EMBEDDINGS_FOR_MEMORY = True  # False 则不生成向量

# 影响:
# - True: 支持语义搜索（"找关于 AI 的记忆"）
# - False: 只能关键词搜索，但节省计算
```

---

## 🎯 实际示例：完整追踪

### 输入
```
用户在 Chrome 浏览 https://cursor.com
时间: 2025-11-19 10:24:49
```

### 步骤 1: 截图
```
文件: /Users/power/.mirix/tmp/images/screenshot-2025-11-19T10-24-49.png
来源: Electron 截图监控
```

### 步骤 2: OCR
```python
ocr_text = "Cursor - AI Code Editor | Features: AI autocomplete, AI chat, Code generation..."
urls = ["cursor.com", "cursor.com/features"]
source_url = "cursor.com"
```

### 步骤 3: Raw Memory
```sql
INSERT INTO raw_memory VALUES (
    id: 'rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028',
    screenshot_path: '/Users/power/.mirix/tmp/images/screenshot-2025-11-19T10-24-49.png',
    source_app: 'Chrome',
    source_url: 'cursor.com',
    captured_at: '2025-11-19T10:24:49',
    ocr_text: 'Cursor - AI Code Editor...',
    ocr_text_embedding: [0.123, -0.456, ...],
    processed: false
);
```

### 步骤 4-5: LLM 分析
```
Meta Memory Agent (Gemini 2.0):
Input:
  - 截图 (visual)
  - OCR text: "Cursor - AI Code Editor..."
  - Context: Chrome, cursor.com

分析:
  - 这是一个 AI 代码编辑器的产品页面
  - 包含产品特性和功能介绍
  - 应该存储为 Semantic Memory (知识)
  - 可能也作为 Resource Memory (工具链接)
```

### 步骤 6: 调用工具
```python
# LLM 生成的函数调用
semantic_memory_insert({
    "items": [{
        "name": "Cursor (AI Code Editor)",
        "summary": "AI-powered code editor with autocomplete and chat",
        "details": "Cursor is a fork of VSCode that integrates AI...",
        "source": "web_browsing",
        "tree_path": "/Tools/AI/Code Editors",
        "raw_memory_references": ["rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028"]
    }]
})
```

### 步骤 7: 数据库写入
```sql
INSERT INTO semantic_memory VALUES (
    id: 'semantic-abc123',
    name: 'Cursor (AI Code Editor)',
    summary: 'AI-powered code editor...',
    details: 'Cursor is a fork of VSCode...',
    raw_memory_references: '["rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028"]',
    embedding: [0.789, -0.234, ...],
    created_at: '2025-11-19T10:25:30'
);

UPDATE raw_memory
SET processed = true, processing_count = 1
WHERE id = 'rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028';
```

### 步骤 8: 前端展示
```
Memory Library → Semantic Memory:

┌─────────────────────────────────────────────────────────────┐
│ Cursor (AI Code Editor)                                     │
│                                                             │
│ AI-powered code editor with autocomplete and chat          │
│                                                             │
│ Memory References:                                          │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ 🌐 Chrome | 🔗 cursor.com | 📅 Nov 19 10:24        │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ [显示详情]                                                  │
└─────────────────────────────────────────────────────────────┘
```

点击徽章 → 跳转到 Raw Memory → 查看原始截图 ✅

---

## 📊 性能和成本

### LLM 调用频率
```
每批截图处理:
  1. Meta Memory Agent: 1 次调用 (分析所有截图)
  2. Specialized Agents: 0-4 次调用 (根据需要)

平均每个截图: ~1-2 次 LLM 调用
成本: ~$0.001-0.002 per screenshot (Gemini 2.0 Flash)
```

### 数据库写入
```
每个截图:
  - 1 条 raw_memory 记录
  - 0-N 条 semantic_memory 记录
  - 0-N 条 episodic_memory 记录
  - 0-N 条 其他类型记录
```

### 向量存储
```
每条记录:
  - OCR text embedding: 1536 维 (OpenAI)
  - Semantic memory embedding: 1536 维

用于语义搜索和相似性匹配
```

---

## 🎉 总结

**完整数据流**:
```
截图 → OCR → Raw Memory → Message Queue →
Meta Agent (LLM 分析) → Specialized Agents (LLM 处理) →
Memory Tools → Database (带 references) → API → 前端展示
```

**LLM 的核心作用**:
1. ✅ **理解内容**: 看懂截图和 OCR 文本
2. ✅ **智能分类**: 决定存储到哪种记忆类型
3. ✅ **提取知识**: 从原始数据提取结构化信息
4. ✅ **建立关联**: 通过 raw_memory_references 连接原始数据

**用户价值**:
1. ✅ **自动化**: 不需要手动整理笔记
2. ✅ **可追溯**: 每个记忆都有原始截图支持
3. ✅ **可搜索**: 通过向量搜索快速找到相关记忆
4. ✅ **可信赖**: 透明的数据来源增强信任

---

**相关文档**:
- `UAT_ISSUES_ANALYSIS.md` - UAT 问题分析
- `DATA_CLEANUP_AND_FIX_SUMMARY.md` - 数据清理修复
- `STRATEGIC_ROADMAP.md` - 长期规划
- `phase1_raw_memory.md` - Phase 1 技术文档
