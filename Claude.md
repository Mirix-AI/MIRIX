# MIRIX 项目 - Claude 工作指南

行为模式，记录,解决,测试,commit,更新报告。
如果代码已经清晰，自己判断解决。
如果非常模糊，和用户确认
## 📖 项目概述

MIRIX 是一个基于 AI 的个人助手系统，具有多层记忆架构和自动截图监控功能。项目核心是通过记录用户屏幕活动，提取信息，并构建长期记忆系统来提供智能辅助。

## 🏗️ 项目架构

### 技术栈

**后端 (Python):**
- FastAPI - Web 框架和 API 服务
- SQLAlchemy - ORM 数据库抽象层
- PostgreSQL - 生产数据库（支持 pgvector 向量搜索）
- SQLite - 开发/测试数据库
- pytesseract - OCR 文本识别
- Google Generative AI - LLM 模型接口 (Gemini)

**前端 (React):**
- React with Hooks - UI 框架
- i18next - 国际化支持（中英文）
- CSS3 - 样式和动画

**开发工具:**
- pytest - 单元测试
- Alembic - 数据库迁移（未使用，使用自定义 SQL 脚本）

### 目录结构

```
MIRIX/
├── mirix/                    # 后端 Python 代码
│   ├── agent/               # AI Agent 核心逻辑
│   │   ├── agent.py         # Agent 主类
│   │   ├── agent_wrapper.py # Agent 包装器（处理流式响应）
│   │   └── temporary_message_accumulator.py  # 消息累积器
│   ├── orm/                 # SQLAlchemy ORM 模型
│   │   ├── raw_memory.py    # 原始记忆模型
│   │   ├── semantic_memory.py
│   │   ├── episodic_memory.py
│   │   └── ...
│   ├── services/            # 业务逻辑层
│   │   ├── raw_memory_manager.py
│   │   ├── semantic_memory_manager.py
│   │   └── ...
│   ├── helpers/             # 辅助工具类
│   │   └── ocr_url_extractor.py  # OCR 和 URL 提取
│   ├── functions/           # LLM 可调用的工具函数
│   │   └── function_sets/
│   │       └── memory_tools.py
│   ├── server/              # FastAPI 服务器
│   │   └── fastapi_server.py
│   ├── schemas/             # Pydantic 数据模型
│   └── prompts/             # AI 提示词模板
├── frontend/                # React 前端
│   └── src/
│       ├── components/
│       │   ├── ChatWindow.js      # 聊天界面
│       │   ├── ChatBubble.js      # 消息气泡（显示记忆引用）
│       │   └── ExistingMemory.js  # 记忆库界面
│       ├── i18n.js          # 国际化配置
│       └── utils/
├── database/                # 数据库迁移脚本
│   ├── migrate_add_raw_memory.sql      # PostgreSQL 迁移
│   └── run_sqlite_migration.py         # SQLite 迁移
├── scripts/                 # 实用脚本
├── tests/                   # 单元测试
├── main.py                  # 项目入口
├── phase1_task_list.md      # 第一阶段任务列表
├── phase1_raw_memory.md     # 第一阶段设计文档
└── Claude.md                # 本文件
```

## 🧠 记忆系统架构

MIRIX 使用多层记忆架构：

### 1. Raw Memory (原始记忆层)
- **作用**: 存储最原始的信息源（截图、OCR 文本、元数据）
- **表**: `raw_memory`
- **关键字段**:
  - `screenshot_path`: 截图文件路径
  - `source_app`: 来源应用名称
  - `source_url`: 提取的 URL
  - `ocr_text`: OCR 提取的文本
  - `captured_at`: 捕获时间
  - `ocr_text_embedding`: OCR 文本的向量嵌入（用于语义搜索）
  - `processed`: 是否已处理

### 2. 高层记忆（引用 Raw Memory）
- **Semantic Memory (语义记忆)**: 知识、概念、事实
- **Episodic Memory (情景记忆)**: 事件、经历
- **Procedural Memory (程序记忆)**: 技能、步骤
- **Resource Memory (资源记忆)**: 文档、文件
- **Knowledge Vault (知识库)**: 结构化知识

所有高层记忆都包含 `raw_memory_references` 字段（JSON 数组），存储引用的 raw_memory ID 列表。

### 3. 记忆引用关系

```
截图 (PNG/JPG)
    ↓ OCR 提取
Raw Memory (id, ocr_text, source_url, ...)
    ↑ 引用
Semantic/Episodic/Procedural Memory (raw_memory_references: ["rawmem-xxx", ...])
    ↓ 展示
前端 UI (紫色徽章，显示 app 图标、URL、日期)
```

## 🔄 工作流程

### 1. 截图监控流程

```
ScreenshotMonitor (定时截图)
    ↓
temporary_message_accumulator.py (_build_memory_message)
    ↓ 遍历截图
OCRUrlExtractor.extract_urls_and_text()
    ↓ 提取 OCR 文本和 URL
RawMemoryManager.insert_raw_memory()
    ↓ 存入数据库
返回 raw_memory_ids
    ↓ 传递给记忆 Agent
LLM 决定创建哪些高层记忆
    ↓ 调用工具函数
memory_tools.py (semantic_memory_insert, etc.)
    ↓ 传递 raw_memory_references
存入高层记忆表
```

### 2. 前端展示流程

```
用户打开聊天窗口
    ↓
ChatWindow.js 发起请求
    ↓
后端 /chat/stream 处理（agent_wrapper.py）
    ↓ 返回流式响应
ChatBubble.js 渲染消息
    ↓ 包含 memoryReferences
renderMemoryReferences() 显示紫色徽章
    ↓
用户看到来源信息（app 图标、URL、日期）
```

### 3. 记忆库查看流程

```
用户打开 Memory Library
    ↓
ExistingMemory.js 获取 Semantic Memory
    ↓ 调用 /memory/semantic
fastapi_server.py 查询数据库
    ↓ 获取 raw_memory_references
查询 RawMemoryItem 表获取完整详情
    ↓ 返回 JSON
前端渲染记忆项
    ↓ 调用 renderMemoryReferences()
显示紫色徽章（app、URL、日期、OCR 预览）
```

## 🎨 UI/UX 设计原则

### 记忆引用展示
- **颜色**: 紫色渐变 (rgba(139, 92, 246)) - 代表记忆和知识
- **图标**:
  - 🌐 Chrome
  - 🧭 Safari
  - 🦊 Firefox
  - 📝 Notion
  - 💻 其他应用
- **信息层次**:
  1. App 名称（粗体）
  2. URL 域名（等宽字体）
  3. 捕获日期（小号字体）
  4. OCR 预览（可选，截断到 100 字符）

### 过滤和搜索
- **过滤器**: 只显示有引用的记忆（增强用户信任）
- **搜索**: 支持全文搜索（包括 source_app, source_url, ocr_text）
- **自动展开**: 搜索命中详情时自动展开

## 📝 开发约定

### 代码风格
- **Python**: 使用 type hints，遵循 PEP 8
- **JavaScript**: 使用 ES6+，React Hooks 模式
- **CSS**: BEM 命名约定（e.g., `memory-badge`, `memory-badge-icon`）

### 数据库约定
- **PostgreSQL**: 用于生产环境，支持 pgvector
- **SQLite**: 用于开发和测试
- **迁移**: 使用自定义 SQL 脚本（`database/migrate_*.sql`）
- **幂等性**: 所有迁移脚本都应支持重复运行

### API 约定
- **端点**: RESTful 风格（`/memory/semantic`, `/memory/raw`）
- **返回格式**: JSON
- **错误处理**: 返回 HTTP 状态码和错误信息

### Git 提交约定
- **格式**: `[Task XX] 简短描述`
- **示例**: `[Task 18] Add memory references display in semantic memory`
- **分组提交**: 按任务分组，避免混合多个功能

## 🔧 常用命令

### 启动后端服务器
```bash
python main.py
# 或
python -m mirix.server.fastapi_server
```

### 启动前端开发服务器
```bash
cd frontend
npm start
```

### 运行数据库迁移
```bash
# PostgreSQL
psql -U power -d mirix -f database/migrate_add_raw_memory.sql

# SQLite
python database/run_sqlite_migration.py
```

### 运行测试
```bash
pytest tests/test_raw_memory_reference_pipeline.py -v
```

### 导入真实截图
```bash
# 导入前 10 张截图（默认）
python scripts/import_real_screenshots.py

# 导入前 50 张截图
python scripts/import_real_screenshots.py --limit 50

# 导入全部截图
python scripts/import_real_screenshots.py --limit 0

# 不跳过已存在的截图（重新导入）
python scripts/import_real_screenshots.py --no-skip
```

## 📋 当前工作状态（第一阶段 - Phase 1）

**⚠️ 重要提示：Phase 1 尚未完成！完成后才能进入 Phase 2。**

### ✅ 已完成
- [x] 任务 1-10: 核心数据层和数据库
- [x] 任务 15: 修复前端 memoryReferences 显示
- [x] 任务 16: Raw Memory 在记忆库中展示
- [x] 任务 17: Raw Memory 搜索功能
- [x] 任务 18: Semantic Memory 中显示 Raw Memory References
- [x] 任务 19: 优化 Memory References 显示 UX
  - 修复 React Hooks 错误
  - 实现折叠/展开、分组、去重
  - 点击跳转到 Raw Memory

### ⏳ 进行中
- [ ] 任务 11-14: 测试验证（使用真实截图数据）
  - 任务 11: 创建 OCR 测试脚本 ✅（`scripts/import_real_screenshots.py`）
  - 任务 12: 测试 URL 提取
  - 任务 13: 测试数据写入
  - 任务 14: 验证前端展示

### 🎯 下一步
1. **导入真实截图**: 运行 `python scripts/import_real_screenshots.py`
2. **测试 OCR**: 验证 `~/.mirix/tmp/images/` 中的截图 OCR 准确性
3. **验证 URL 提取**: 检查 `source_url` 字段是否正确提取
4. **完整流程测试**: 从截图 → raw_memory → semantic memory → 前端显示
5. **前端 UX 优化**: 优化折叠展开动画，完善点击跳转功能

### 📌 Phase 1 完成标准
- [ ] 所有任务 1-19 全部完成并测试通过
- [ ] 真实截图数据导入成功
- [ ] 记忆引用关系正确建立
- [ ] 前端正确显示引用信息
- [ ] OCR 和 URL 提取准确率达标
- [ ] 文档完整（phase1_task_list.md 全部更新）

**只有满足以上所有条件，Phase 1 才算完成，才能进入 Phase 2！**

## 💡 重要注意事项

### 给 Claude 的提示

1. **任务管理** ⚠️ 非常重要！:
   - 每次完成任务后，**必须**更新 `phase1_task_list.md` 文件
   - 添加详细的完成记录，包括：
     - **遇到的问题**（如果有）- 错误信息、原因分析
     - **为什么要做这个修改** - 从用户视角说明
     - **修改的文件和行号** - 便于后续查找
     - **测试验证** - 如何验证功能正常
     - **用户体验提升** - 这个修改带来的好处
   - 如果遇到 bug，记录错误信息和解决方案
   - **Phase 1 尚未完成**，所有记录都在 `phase1_task_list.md` 中

2. **代码修改**:
   - 修改前先阅读相关文件，理解现有逻辑
   - 保持代码风格一致（参考现有代码）
   - 添加必要的注释（尤其是复杂逻辑）

3. **测试**:
   - 修改后端 API 后，使用 `curl` 测试端点
   - 修改前端后，检查浏览器控制台是否有错误
   - 关键功能需要编写单元测试

4. **文档**:
   - 重大功能修改后更新相关 .md 文档
   - 在 git commit 消息中说明修改内容
   - 保持 `Claude.md` 文件更新

5. **调试技巧**:
   - 后端日志: 查看 `/tmp/mirix_server.log`
   - PostgreSQL 查询: 使用 `psql` 直接查询验证数据
   - 前端: 使用浏览器开发者工具查看网络请求和控制台

6. **数据库操作**:
   - 生产环境使用 PostgreSQL (port 5432)
   - 测试数据使用 mock 脚本生成（`scripts/create_raw_memory_mock_data_simple.py`）
   - 真实数据在 `/Users/power/.mirix/tmp/images/`

## 🗺️ 项目路线图

### Phase 1 (当前阶段)
- ✅ 核心记忆架构
- ✅ Raw Memory 和引用系统
- ⏳ 测试和验证

### Phase 2 (计划中)
- 记忆检索优化（向量搜索）
- 记忆清理和归档
- 更多数据源（浏览器历史、文件系统）

### Phase 3 (未来)
- 多模态支持（视频、音频）
- 记忆可视化
- 高级分析和洞察

## 🤝 协作约定

- **沟通**: 遇到问题时先搜索 `phase1_task_list.md` 和本文档
- **提交**: 每完成一个小功能就提交，不要等到全部完成
- **记录**: 详细记录修改原因和决策过程
- **测试**: 提交前确保功能正常工作

---

**最后更新**: 2025-11-19
**当前版本**: Phase 1 - v0.1.0
**维护者**: MIRIX Team
