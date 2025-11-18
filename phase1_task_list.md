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

- [ ] **任务 7**: 修改系统提示词展示来源信息
  - 文件: mirix/agent/agent.py 的 build_system_prompt() 方法
  - 在展示记忆时包含 [Source: App名称, URL: xxx] 信息

### API 和前端

- [ ] **任务 8**: 添加 FastAPI 端点
  - 路径: /memory/raw/{raw_memory_id}
  - 返回完整的 raw_memory 详细信息

- [ ] **任务 9**: 前端展示记忆引用
  - 修改: frontend/src/components/ChatBubble.js
  - 添加记忆引用卡片，显示来源 app 和 URL

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

