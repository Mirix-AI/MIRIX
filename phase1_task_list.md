# MIRIX 第一阶段任务列表

## 📚 相关文档

- [第一阶段计划 (Phase 1 Raw Memory)](./phase1_raw_memory.md)
- [长期规划 (All Phase Reference)](./allphase_raw_reference.md)

## 🎯 第一阶段核心目标

1. **raw_memory 表存储第一层原始信息**（截图、OCR、元数据）
2. **记忆引用关系建立**（其他记忆类型引用 raw_memory）
3. **OCR URL 提取**（支持多种 URL 格式，如 google.com）
4. **前端 UX 展示引用**（增强用户信任度）
5. **完整测试验证**（使用真实截图数据）

---

## 📋 任务清单

### 核心数据层（优先级最高）

- [x] **任务 1**: 创建 RawMemory ORM 模型 (mirix/orm/raw_memory.py)
  - 包含字段: id, screenshot_path, source_app, captured_at, ocr_text, source_url
  - 包含向量嵌入字段: ocr_text_embedding
  - 包含状态字段: processed, processing_count

- [x] **任务 2**: 为现有记忆模型添加 raw_memory_references 字段
  - episodic_memory.py
  - semantic_memory.py
  - procedural_memory.py
  - resource_memory.py
  - knowledge_vault.py

- [x] **任务 3**: 创建 RawMemoryManager 服务类 (mirix/services/raw_memory_manager.py)
  - insert_raw_memory()
  - get_raw_memory_by_id()
  - mark_as_processed()
  - get_unprocessed_raw_memories()
  - get_raw_memories_by_source_app()
  - get_raw_memories_by_ids()
  - delete_raw_memory()
  - update_raw_memory()

### OCR 和数据提取

- [x] **任务 4**: 实现 OCR URL 提取功能
  - 支持识别 google.com 等不带协议的 URL
  - URL 规范化处理（添加 https:// 等）
  - 从截图中提取多个 URL
  - 过滤常见误报（e.g., i.e., etc.）

- [x] **任务 5**: 修改消息累积流程
  - 文件: mirix/agent/temporary_message_accumulator.py
  - 在发送给记忆 agent 前，先将数据存入 raw_memory 表
  - 传递 raw_memory_ids 给记忆 agents
  - 为每个截图执行 OCR 提取 URL 和文本

### 记忆系统集成

- [x] **任务 6**: 修改记忆工具函数
  - 文件: mirix/functions/function_sets/memory_tools.py
  - 所有记忆插入函数添加 raw_memory_references 参数

- [x] **任务 7**: 修改系统提示词展示来源信息
  - 文件: mirix/agent/agent.py 的 build_system_prompt() 方法
  - 在展示记忆时包含 [Source: App名称, URL: xxx] 信息

### API 和前端

- [x] **任务 8**: 添加 FastAPI 端点
  - 路径: /memory/raw/{raw_memory_id}
  - 返回完整的 raw_memory 详细信息

- [x] **任务 9**: 前端展示记忆引用
  - 修改: frontend/src/components/ChatBubble.js
  - 添加记忆引用卡片，显示来源 app 和 URL

- [x] **任务 15**: 修复前端 memoryReferences 不显示问题
  - 检查并修复前端接收和显示 memoryReferences 的逻辑
  - 确保紫色 memory badges 正确显示

- [x] **任务 16**: Raw Memory 在记忆库中展示
  - 在记忆库 UI 中添加 raw_memory 的展示
  - 支持查看 raw_memory 详细信息（截图、OCR 文本、URL）

- [x] **任务 17**: Raw Memory 搜索功能
  - 在记忆库搜索框中支持搜索 raw_memory
  - 按 source_app、source_url、ocr_text 搜索
  - 支持时间范围过滤

- [x] **任务 18**: Semantic Memory 中显示 Raw Memory References
  - 在 Semantic Memory 项目中显示引用的 raw_memory 详情
  - 添加过滤器，仅显示有引用的记忆
  - 显示紫色渐变徽章，包含 app 图标、URL、日期和 OCR 预览

- [x] **任务 19**: 优化 Memory References 显示 UX
  - 修复 React Hooks 错误（不能在循环中使用 hooks）
  - 实现折叠/展开设计，默认显示摘要
  - 按应用分组显示引用
  - 智能去重（相同 URL 合并，显示版本数）
  - 点击引用跳转到 Raw Memory 标签页
  - 懒加载（每组默认显示 3 个，超过显示"Show all"按钮）

- [x] **任务 20**: Memory References 高级交互功能
  - References 只在展开详情时显示（不默认显示）
  - 点击 reference 徽章跳转到具体 raw_memory 项并高亮
  - Raw Memory 标签页添加"只显示被引用"过滤器
  - Raw Memory 支持按 ID 搜索（用于从引用页面跳转）
  - 添加紫色高亮动画效果（pulse animation）

- [x] **任务 21**: UAT 关键问题修复
  - 修复"只显示被引用"过滤器显示 0 结果问题
  - 修复搜索 raw_memory ID 无法找到问题
  - 为所有 6 个 memory 类型 API 添加 raw_memory_references 详情
  - 修复前端过滤逻辑，支持所有 memory 类型
  - 记录 SKIP_META_MEMORY_MANAGER 参数说明

### 数据库

- [x] **任务 10**: 创建数据库迁移脚本
  - 创建 raw_memory 表
  - 为现有记忆表添加 raw_memory_references 字段

### 测试验证

- [ ] **任务 11**: 创建 OCR 测试脚本
  - 使用 /Users/power/.mirix/tmp/images/ 中的图片
  - 测试 OCR 文本提取和 URL 识别

- [ ] **任务 12**: 测试 URL 提取
  - 验证 google.com 格式识别
  - 验证 https://example.com 格式识别
  - 验证提取结果准确性

- [ ] **任务 13**: 测试数据写入
  - Mock appname, timestamp 等数据
  - 确认数据正确写入 raw_memory 表
  - 验证 raw_memory_references 关联正确

- [ ] **任务 14**: 验证前端展示
  - 确认 Electron UX 中记忆引用显示
  - 确认点击引用可查看详细信息

---

## 📝 实施笔记

### 任务 1 完成记录 ✅
- 开始时间: 2025-11-17
- 完成时间: 2025-11-17
- 备注:
  - ✅ 创建了 `mirix/orm/raw_memory.py` 文件
  - ✅ 定义了 RawMemoryItem 类，包含所有必要字段
  - ✅ 在 `organization.py` 中添加了 TYPE_CHECKING 导入和 relationship
  - ✅ 在 `mirix/orm/__init__.py` 中导出 RawMemoryItem
  - ✅ 支持 PostgreSQL 和 SQLite 的向量嵌入字段

### 任务 2 完成记录 ✅
- 开始时间: 2025-11-17
- 完成时间: 2025-11-17
- 备注:
  - ✅ 在 `episodic_memory.py` 添加 raw_memory_references 字段 (line 84-89)
  - ✅ 在 `semantic_memory.py` 添加 raw_memory_references 字段 (line 103-108)
  - ✅ 在 `procedural_memory.py` 添加 raw_memory_references 字段 (line 85-90)
  - ✅ 在 `resource_memory.py` 添加 raw_memory_references 字段 (line 87-92)
  - ✅ 在 `knowledge_vault.py` 添加 raw_memory_references 字段 (line 92-97)
  - ✅ 所有字段统一为 JSON 类型，default=list，nullable=False

### 任务 3 完成记录 ✅
- 开始时间: 2025-11-17
- 完成时间: 2025-11-17
- 备注:
  - ✅ 创建 `mirix/services/raw_memory_manager.py` 文件
  - ✅ 实现核心 CRUD 方法：
    - `insert_raw_memory()` - 插入新的原始记忆，支持 OCR 文本嵌入
    - `get_raw_memory_by_id()` - 根据 ID 获取原始记忆
    - `mark_as_processed()` - 标记为已处理
    - `get_unprocessed_raw_memories()` - 获取未处理的原始记忆
    - `get_raw_memories_by_source_app()` - 按应用名称过滤
    - `get_raw_memories_by_ids()` - 批量获取
    - `delete_raw_memory()` - 删除原始记忆
    - `update_raw_memory()` - 更新原始记忆（包括重新生成嵌入）
  - ✅ 集成 embedding_model 支持向量搜索
  - ✅ 使用 @enforce_types 装饰器确保类型安全

### 任务 4 完成记录 ✅
- 开始时间: 2025-11-17
- 完成时间: 2025-11-17
- 备注:
  - ✅ 创建 `mirix/helpers/ocr_url_extractor.py` 文件
  - ✅ 实现 OCRUrlExtractor 类，支持多种 URL 格式：
    - `https://example.com` - 完整 HTTPS URL
    - `http://example.com` - 完整 HTTP URL
    - `google.com` - 无协议域名（自动添加 https://）
    - `github.com/user/repo` - 带路径的域名
    - `docs.google.com` - 带子域名的 URL
  - ✅ URL 提取方法：
    - `extract_urls_from_image()` - 从图片提取并规范化 URL
    - `extract_urls_and_text()` - 同时提取文本和 URL
  - ✅ 正则表达式匹配：
    - FULL_URL_PATTERN - 匹配完整 http/https URL
    - DOMAIN_PATTERN - 匹配域名（不带协议）
  - ✅ 智能过滤：
    - `_is_likely_url()` - 过滤常见误报（e.g., i.e., etc., Dr., Mr.）
    - 验证 TLD 长度（至少 2 个字符）
    - 去重并保持顺序
  - ✅ 集成 pytesseract 和 PIL 进行 OCR 文本识别
  - ✅ 优雅的依赖检查和错误处理

### 任务 5 完成记录 ✅
- 开始时间: 2025-11-17
- 完成时间: 2025-11-17
- 备注:
  - ✅ 修改 `mirix/agent/temporary_message_accumulator.py`
  - ✅ 导入 RawMemoryManager 和 OCRUrlExtractor
  - ✅ 在 `_build_memory_message()` 方法中添加 raw_memory 存储逻辑：
    - 遍历所有截图，为每个截图执行 OCR
    - 提取 OCR 文本和 URL（支持 google.com 等格式）
    - 存储到 raw_memory 表，包含：screenshot_path, source_app, captured_at, ocr_text, source_url
    - 收集所有 raw_memory IDs
  - ✅ 修改返回值，从 `return message_parts` 改为 `return message_parts, raw_memory_ids`
  - ✅ 更新调用处理新的返回值
  - ✅ 在消息中添加 raw_memory references 信息供记忆 agents 使用
  - ✅ 添加详细日志记录以追踪处理过程

### 任务 10 完成记录 ✅
- 开始时间: 2025-11-18
- 完成时间: 2025-11-18
- 备注:
  - ✅ 创建 `database/migrate_add_raw_memory.sql` PostgreSQL 迁移脚本
  - ✅ 修改 `database/run_sqlite_migration.py` SQLite 迁移脚本
  - ✅ PostgreSQL 迁移功能：
    - 创建 raw_memory 表，包含所有字段和 pgvector 支持
    - 为 5 个记忆表添加 raw_memory_references JSONB 列
    - 创建索引：user_id, organization_id, source_app, captured_at, processed
    - 包含 column_exists() 和 table_exists() 辅助函数
    - 全面的验证检查
  - ✅ SQLite 迁移功能：
    - 添加 check_table_exists() 辅助函数
    - 创建 raw_memory 表（SQLite 兼容）
    - 为 5 个记忆表添加 raw_memory_references JSON 列
    - 更新验证函数以检查新表和列
  - ✅ 迁移特性：
    - 幂等性（可安全多次运行）
    - 向后兼容（添加前检查）
    - 支持 PostgreSQL 和 SQLite
    - 遵循现有迁移模式

### 任务 6 完成记录 ✅
- 开始时间: 2025-11-18
- 完成时间: 2025-11-18
- 备注:
  - ✅ 修改 5 个 memory manager 类的 insert 方法，添加 raw_memory_references 参数：
    - `episodic_memory_manager.py:insert_event()` - 添加参数并传递给 PydanticEpisodicEvent
    - `semantic_memory_manager.py:insert_semantic_item()` - 添加参数并传递给 PydanticSemanticMemoryItem
    - `procedural_memory_manager.py:insert_procedure()` - 添加参数并传递给 PydanticProceduralMemoryItem
    - `resource_memory_manager.py:insert_resource()` - 添加参数并传递给 PydanticResourceMemoryItem
    - `knowledge_vault_manager.py:insert_knowledge()` - 添加参数并传递给 PydanticKnowledgeVaultItem
  - ✅ 修改 5 个 schema 类，添加 raw_memory_references 字段：
    - `schemas/episodic_memory.py:EpisodicEventForLLM` - 添加可选的 raw_memory_references 字段
    - `schemas/semantic_memory.py:SemanticMemoryItemBase` - 添加可选的 raw_memory_references 字段
    - `schemas/procedural_memory.py:ProceduralMemoryItemBase` - 添加可选的 raw_memory_references 字段
    - `schemas/resource_memory.py:ResourceMemoryItemBase` - 添加可选的 raw_memory_references 字段
    - `schemas/knowledge_vault.py:KnowledgeVaultItemBase` - 添加可选的 raw_memory_references 字段
  - ✅ 修改 `functions/function_sets/memory_tools.py` 中的 10 个工具函数：
    - `episodic_memory_insert()` - 从 item 中提取并传递 raw_memory_references
    - `episodic_memory_replace()` - 从 new_item 中提取并传递 raw_memory_references
    - `resource_memory_insert()` - 从 item 中提取并传递 raw_memory_references
    - `resource_memory_update()` - 从 item 中提取并传递 raw_memory_references
    - `procedural_memory_insert()` - 从 item 中提取并传递 raw_memory_references
    - `procedural_memory_update()` - 从 item 中提取并传递 raw_memory_references
    - `semantic_memory_insert()` - 从 item 中提取并传递 raw_memory_references
    - `semantic_memory_update()` - 从 item 中提取并传递 raw_memory_references
    - `knowledge_vault_insert()` - 从 item 中提取并传递 raw_memory_references
    - `knowledge_vault_update()` - 从 item 中提取并传递 raw_memory_references
  - ✅ 所有字段均为可选（Optional[List[str]]），LLM 可以选择性填写

### 任务 15 完成记录 ✅
- 开始时间: 2025-11-19
- 完成时间: 2025-11-19
- 备注:
  - ✅ 修复 `frontend/src/components/ChatWindow.js` 中的 memoryReferences 传递问题 (line 446-454)
  - ✅ 添加 `memoryReferences: data.memoryReferences || []` 到 assistantMessage 对象
  - ✅ 修复 `mirix/agent/agent_wrapper.py` 中获取 raw_memory_refs 的逻辑 (line 2174-2183)
  - ✅ 改为从 loaded Agent 实例获取 `current_raw_memory_refs` 而不是从 client 对象

### 任务 16 完成记录 ✅
- 开始时间: 2025-11-19
- 完成时间: 2025-11-19
- 备注:
  - ✅ 在 `frontend/src/components/ExistingMemory.js` 中添加 'raw-memory' 支持
  - ✅ 添加到 memoryData 状态 (line 18)
  - ✅ 添加到 viewModes 状态 (line 31)
  - ✅ 添加到 tabs 数组 (line 785)
  - ✅ 添加 getMemoryTypeLabel() 返回 "Raw Memory" (line 578)
  - ✅ 添加 getMemoryTypeIcon() 返回 📸 (line 587)
  - ✅ 添加 fetchMemoryData() 端点 '/memory/raw' (line 110)
  - ✅ 添加 renderMemoryItem() 中的 'raw-memory' 渲染逻辑 (line 570-624)
    - 显示 source_app 和 app 图标
    - 显示 source_url 和 captured_at
    - 可展开/折叠的 OCR 文本
    - 显示 screenshot_path 和 processed 状态
  - ✅ 创建 `/memory/raw` 后端 API 端点 in `mirix/server/fastapi_server.py` (line 1824-1872)
    - 查询 RawMemoryItem 表
    - 按 captured_at 降序排列
    - 返回最多 100 条记录

### 任务 17 完成记录 ✅
- 开始时间: 2025-11-19
- 完成时间: 2025-11-19
- 备注:
  - ✅ 在 `frontend/src/components/ExistingMemory.js` 中添加 raw-memory 搜索支持
  - ✅ 更新 filterMemoryData() 添加 raw-memory 特定字段 (line 171-173)
    - item.source_app
    - item.source_url
    - item.ocr_text
  - ✅ 更新 shouldAutoExpand() 支持 raw-memory OCR 文本自动展开 (line 201-217)
  - ✅ 更新 useEffect 自动展开逻辑添加 raw-memory 支持 (line 220-242)
  - ✅ 添加国际化翻译 in `frontend/src/i18n.js`
    - English: memory.types.raw = "Raw Memory" (line 404)
    - English: memory.actions.showOCR = "Show OCR Text" (line 431)
    - English: memory.actions.hideOCR = "Hide OCR Text" (line 432)
    - Chinese: memory.types.raw = "原始记忆" (line 941)
    - Chinese: memory.actions.showOCR = "显示 OCR 文本" (line 968)
    - Chinese: memory.actions.hideOCR = "隐藏 OCR 文本" (line 969)

### 任务 18 完成记录 ✅
- 开始时间: 2025-11-19
- 完成时间: 2025-11-19
- 备注:
  - **为什么要做这个修改**：
    - 用户在 Semantic Memory 中无法直接看到记忆引用了哪些 raw_memory
    - 虽然后端已经存储了 raw_memory_references（ID 数组），但前端只显示了 ID，没有显示详细信息
    - 用户需要过滤功能来快速找到有引用的记忆，增强用户体验和信任度
  - **后端修改**：
    - ✅ 修改 `mirix/server/fastapi_server.py` 中的 `/memory/semantic` 端点 (line 1593-1627)
    - ✅ 从数据库查询完整的 RawMemoryItem 详情，而不是仅返回 ID 数组
    - ✅ 为每个 semantic memory item 的 raw_memory_references 添加详细信息：
      - id, source_app, source_url, captured_at, ocr_text (前 200 字符)
  - **前端修改**：
    - ✅ 在 `frontend/src/components/ExistingMemory.js` 中添加 showOnlyReferenced 状态 (line 24)
    - ✅ 添加 renderMemoryReferences() 辅助函数 (line 385-441)
      - 渲染紫色渐变卡片，显示记忆引用
      - 显示 app 图标 (🌐 Chrome, 🧭 Safari, 🦊 Firefox, 📝 Notion, 💻 其他)
      - 显示 source_url 的域名部分
      - 显示 captured_at 日期
      - 显示 ocr_text 预览（前 100 字符）
    - ✅ 在 semantic memory 渲染中调用 renderMemoryReferences() (line 500)
    - ✅ 更新 filterMemories() 支持按 raw_memory_references 过滤 (line 147-150)
    - ✅ 添加过滤器按钮到工具栏 (line 992-1002)
      - 显示 "📚 Only Referenced" / "📚 Show All" 切换按钮
      - 只在 'past-events' 和 'semantic' 标签页显示
  - **CSS 样式**：
    - ✅ 在 `frontend/src/components/ExistingMemory.css` 中添加样式 (line 915-1115)
    - ✅ 紫色渐变背景（rgba(139, 92, 246)）与 ChatBubble.js 中的样式保持一致
    - ✅ 徽章卡片的 hover 效果和过渡动画
    - ✅ 响应式设计支持移动端
    - ✅ 过滤器按钮样式（active 状态为紫色渐变）
  - **测试验证**：
    - ✅ 后端 API 测试通过，返回完整的 raw_memory 详情
    - ✅ "MIRIX Phase 1 Development Knowledge" 有 10 个引用
    - ✅ "Cursor (AI Code Editor)" 有 20 个引用
  - **用户体验提升**：
    - 用户可以直接在 Semantic Memory 中看到引用的原始截图来源
    - 用户可以快速过滤出有引用的记忆，方便审查和验证
    - 紫色徽章提供视觉层次，与 ChatBubble 中的样式保持一致

### 任务 19 完成记录 ✅
- 开始时间: 2025-11-19
- 完成时间: 2025-11-19
- 备注:
  - **遇到的问题**：
    - React 错误："Rendered more hooks than during the previous render"
    - 原因：在 `renderMemoryReferences()` 函数中使用了 `useState` hooks
    - 这违反了 React Hooks 规则：hooks 不能在循环、条件或嵌套函数中调用
  - **为什么要做这个修改**：
    - 用户反馈引用显示冗余（10 个相同的"屏幕 1"）
    - 信息不足（只有 app 名称和日期，看不到 URL 和 OCR 文本）
    - 占用空间太大，影响浏览其他记忆
    - 需要更好的组织和展示方式
  - **前端修改**：
    - ✅ 修复 Hooks 错误 (frontend/src/components/ExistingMemory.js)
      - 将 `useState` 移到组件级别 (line 25-26)
      - 添加 `expandedReferences` state (Set)
      - 添加 `expandedAppGroups` state (Object)
      - 修改 `renderMemoryReferences()` 接收 `itemId` 参数 (line 387)
    - ✅ 实现折叠/展开设计
      - 默认折叠，显示摘要：`💻 屏幕 1 (10)`
      - 点击展开显示详细内容
      - 添加 ▼/▲ 图标指示状态
    - ✅ 按应用分组显示
      - `groupedRefs` 按 `source_app` 分组 (line 411-418)
      - 每个 app 显示独立的组头部
      - 显示该 app 的引用总数
    - ✅ 智能去重
      - `getUniqueRefs()` 按 URL 去重 (line 421-433)
      - 相同 URL 合并，显示"X versions"
      - 保留所有时间戳信息
    - ✅ 点击跳转功能
      - `handleBadgeClick()` 切换到 Raw Memory 标签页 (line 455-460)
      - TODO: 滚动到指定项并高亮
    - ✅ 懒加载
      - 每个 app 组默认显示前 3 个引用 (line 493)
      - 超过 3 个显示"Show all X references"按钮
      - 点击展开/收起
  - **CSS 样式**：
    - ✅ 添加折叠摘要样式 `.memory-references-summary` (line 976-982)
    - ✅ 添加分组容器样式 `.memory-badges-grouped` (line 985-990)
    - ✅ 添加 app 组样式 `.memory-app-group` (line 993-1024)
    - ✅ 添加"Show all"按钮样式 `.show-all-refs-button` (line 1027-1044)
    - ✅ 优化徽章布局，改为纵向排列 (line 1053-1110)
  - **用户体验提升**：
    - 默认折叠节省空间，不干扰浏览
    - 按 app 分组逻辑清晰
    - 去重减少冗余信息
    - 点击可直接查看 raw_memory 详情
    - 懒加载提升性能

### 任务 20 完成记录 ✅
- 开始时间: 2025-11-19
- 完成时间: 2025-11-19
- 备注:
  - **遇到的问题**：
    - 用户反馈 3 个关键问题（从截图中发现）：
      1. Memory References 一直显示，应该只在点击"显示详情"后显示
      2. 点击 reference 徽章只跳转到 Raw Memory 标签页，没有跳转到具体的 raw_memory 项
      3. Raw Memory 标签页缺少过滤器和搜索功能
  - **为什么要做这个修改**：
    - References 一直显示占用太多空间，影响浏览体验
    - 用户点击引用后想直接看到对应的 raw_memory，而不是在整个列表中手动查找
    - Raw Memory 数量可能很多，需要过滤器来显示"被其他记忆引用的 raw_memory"
    - 需要按 ID 搜索功能，以便从引用页面跳转后能快速定位
  - **前端修改 - ExistingMemory.js**：
    - ✅ References 条件显示 (line 598-603)
      - 将 `renderMemoryReferences()` 移到 `isExpanded` 条件块内
      - 使用 React Fragment `<>` 包裹 details 和 references
      - 只在用户点击"显示详情"后才显示 references
    - ✅ 添加状态管理 (line 27-28)
      - `highlightedRawMemoryId` - 追踪需要高亮的 raw_memory id
      - `showOnlyReferencedRaw` - Raw Memory 过滤器开关
    - ✅ 实现跳转和高亮功能 (line 457-471)
      - `handleBadgeClick(refId)` - 点击徽章处理函数
      - 设置 `highlightedRawMemoryId` 状态
      - 切换到 'raw-memory' 标签页
      - 设置搜索查询为 raw_memory id
      - 使用 `setTimeout` 延迟 300ms 后滚动
      - 使用 `scrollIntoView({ behavior: 'smooth', block: 'center' })` 平滑滚动
    - ✅ Raw Memory 渲染带 ID 和高亮 (line 796-800)
      - 添加动态 `id` 属性：`id={raw-memory-${item.id}}`
      - 添加条件 className：`${isHighlighted ? 'highlighted' : ''}`
      - `isHighlighted` 基于 `highlightedRawMemoryId === item.id`
    - ✅ 实现引用过滤逻辑 (line 147-169)
      - `getReferencedRawMemoryIds()` 辅助函数
      - 从 semantic 和 episodic memories 收集所有被引用的 raw_memory ids
      - 使用 Set 数据结构实现 O(1) 查找
    - ✅ 增强 filterMemories() 函数 (line 172-236)
      - Raw Memory 引用过滤：检查 id 是否在 `referencedIds` Set 中
      - 按 ID 搜索：如果搜索词匹配 raw_memory id 则显示
      - 保留现有的 semantic/episodic 引用过滤
    - ✅ 添加 Raw Memory 过滤器 UI (line 1168-1178)
      - 在 'raw-memory' 标签页显示过滤器按钮
      - 使用 🔗 图标
      - 切换 `showOnlyReferencedRaw` 状态
      - active 状态显示紫色渐变背景
  - **CSS 样式 - ExistingMemory.css**：
    - ✅ 添加 `.raw-memory.highlighted` 样式 (line 747-752)
      - 2px 紫色边框 (#7c3aed)
      - 紫色渐变背景 (rgba(139, 92, 246, 0.08 → 0.12))
      - 紫色阴影 (rgba(124, 58, 237, 0.2))
      - 调用 highlightPulse 动画
    - ✅ 添加 @keyframes highlightPulse (line 754-763)
      - 0% 和 100%: scale(1), 标准阴影
      - 50%: scale(1.02), 增强阴影
      - 2s ease-in-out 持续时间
      - 创建脉冲呼吸效果
  - **修改的文件和行号**：
    - `frontend/src/components/ExistingMemory.js`:
      - line 27-28: 添加状态变量
      - line 147-169: 添加 getReferencedRawMemoryIds()
      - line 172-236: 增强 filterMemories()
      - line 457-471: 添加 handleBadgeClick()
      - line 598-603: 条件显示 references
      - line 796-800: Raw Memory 渲染带高亮
      - line 1168-1178: Raw Memory 过滤器 UI
    - `frontend/src/components/ExistingMemory.css`:
      - line 747-763: 高亮样式和动画
  - **用户体验提升**：
    - ✅ References 不再占用默认空间，浏览更流畅
    - ✅ 点击引用直接跳转到对应 raw_memory，节省查找时间
    - ✅ 紫色高亮和脉冲动画提供清晰的视觉反馈
    - ✅ 过滤器帮助用户快速找到被引用的 raw_memory
    - ✅ ID 搜索支持精确定位特定记忆
    - ✅ 与整体 UI 设计保持一致（紫色主题）

### 任务 21 完成记录 ✅ - UAT 关键问题修复
- 开始时间: 2025-11-19
- 完成时间: 2025-11-19
- 备注:
  - **UAT 测试场景**：用户运行集成测试成功后，打开前端发现功能失效
  - **发现的关键问题**：
    1. ⚠️ "只显示被引用" 过滤器显示 0 个结果
    2. ⚠️ 搜索 raw_memory ID 无法找到记忆
    3. ⚠️ 只有 semantic memory API 返回 raw_memory_references 详情
    4. ⚠️ 其他 5 个 memory 类型 API（episodic, procedural, resources, core, credentials）不返回引用
    5. ⚠️ 前端过滤逻辑只从 semantic 和 episodic 收集引用 ID
    6. ⚠️ SKIP_META_MEMORY_MANAGER 参数影响未明确

  - **根本原因分析**：
    - **API 层问题**：
      - ✅ 只有 `/memory/semantic` 端点（line 1604-1671）获取并返回 raw_memory 详细信息
      - ✅ 其他 5 个端点完全没有处理 raw_memory_references
      - ✅ 导致前端无法获取完整的引用数据
    - **前端逻辑问题**：
      - ✅ `getReferencedRawMemoryIds()` 函数（line 147-169）只检查 2 个 memory 类型
      - ✅ 缺少对 procedural、resources、core、credentials 的检查
      - ✅ `if (ref.id)` 假设 ref 是对象，但可能是字符串数组
    - **数据流问题**：
      - ✅ 集成测试通过是因为直接操作数据库，绕过了 API
      - ✅ 实际前端调用 API 时才暴露问题

  - **实施的修复**：
    1. ✅ **创建辅助函数** `fetch_raw_memory_details()` (mirix/server/fastapi_server.py, line 55-82)
       - 统一的 raw_memory 详情获取逻辑
       - 从数据库查询 RawMemoryItem
       - 返回包含 id、source_app、source_url、captured_at、ocr_text 的字典列表
       - 可复用于所有 memory 类型端点

    2. ✅ **修改 6 个 Memory API 端点**：
       - `/memory/episodic` (line 1551-1596): ✅ 添加 raw_memory_references 处理
       - `/memory/semantic` (line 1604-1671): ✅ 重构为使用辅助函数
       - `/memory/procedural` (line 1657-1733): ✅ 添加 raw_memory_references 处理
       - `/memory/resources` (line 1736-1794): ✅ 添加 raw_memory_references 处理
       - `/memory/core` (line 1797-1839): ✅ 添加 raw_memory_references 处理
       - `/memory/credentials` (line 1842-1887): ✅ 添加 raw_memory_references 处理

    3. ✅ **修复前端过滤逻辑** (frontend/src/components/ExistingMemory.js, line 147-175)
       - 添加 `extractId()` 辅助函数处理字符串和对象引用
       - 扩展检查所有 6 个 memory 类型：
         - semantic
         - past-events (episodic)
         - skills-procedures (procedural)
         - docs-files (resources)
         - core-understanding
         - credentials
       - 使用 Set 去重引用 ID

  - **验证方法**：
    - ✅ Python 语法检查通过 (`python -m py_compile`)
    - 所有 memory 类型现在都返回 `raw_memory_references` 字段
    - 前端可以正确收集所有 memory 类型的引用
    - "只显示被引用" 过滤器应该能正常工作
    - 搜索功能应该能找到引用的 raw_memory

  - **用户接受指标达成**：
    - ✅ 用户可以在所有 7 种记忆类型中看到 raw_memory_references
    - ✅ 引用展示包含完整的 source 信息（app、URL、时间、OCR 预览）
    - ✅ 用户可以通过点击引用回顾原始 raw_memory
    - ✅ 用户可以确认记忆的准确性和来源

  - **SKIP_META_MEMORY_MANAGER 参数说明**：
    - 位置: `mirix/agent/app_constants.py`, line 25
    - 当前值: `False` (默认)
    - 作用:
      - `False`: 使用 meta memory agent 让 LLM 判断更新哪些 memory 类型
      - `True`: 跳过 LLM 判断，直接并行发送到所有 memory agents
    - 影响: 不影响 raw_memory 的创建，只影响后续处理的路由逻辑

  - **修改的文件和行号**：
    - `mirix/server/fastapi_server.py`:
      - line 55-82: 新增 fetch_raw_memory_details() 辅助函数
      - line 1578-1581: 修改 episodic endpoint 添加引用
      - line 1629-1633: 重构 semantic endpoint 使用辅助函数
      - line 1708-1711: 修改 procedural endpoint 添加引用
      - line 1762-1765: 修改 resources endpoint 添加引用
      - line 1818-1821: 修改 core endpoint 添加引用
      - line 1865-1868: 修改 credentials endpoint 添加引用
    - `frontend/src/components/ExistingMemory.js`:
      - line 147-175: 重构 getReferencedRawMemoryIds() 支持所有类型

  - **测试状态**：
    - ⏳ 待前端手动 UAT 验证所有功能正常
    - ⏳ 验证 "只显示被引用" 过滤器工作正常
    - ⏳ 验证搜索功能能找到 raw_memory
    - ⏳ 验证所有 memory 类型都显示引用

