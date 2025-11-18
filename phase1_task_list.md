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
每个任务完成后 commit 修改的文件。

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

- [ ] **任务 3**: 创建 RawMemoryManager 服务类 (mirix/services/raw_memory_manager.py)
  - insert_raw_memory()
  - get_raw_memory_by_id()
  - mark_as_processed()

### OCR 和数据提取

- [ ] **任务 4**: 实现 OCR URL 提取功能
  - 支持识别 google.com 等不带协议的 URL
  - URL 规范化处理（添加 https:// 等）
  - 从截图中提取多个 URL

- [ ] **任务 5**: 修改消息累积流程
  - 文件: mirix/agent/temporary_message_accumulator.py
  - 在发送给记忆 agent 前，先将数据存入 raw_memory 表
  - 传递 raw_memory_ids 给记忆 agents

### 记忆系统集成

- [ ] **任务 6**: 修改记忆工具函数
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

- [ ] **任务 10**: 创建数据库迁移脚本
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

