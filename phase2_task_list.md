# Phase 2 任务清单 - Raw Memory 管理与优化

**创建时间**: 2025-11-20
**状态**: 🔴 规划中

---

## 📊 当前问题分析

### 问题 1: 未被引用的 raw_memory 截图文件占用存储空间

**现状**:
- 总 raw_memory 记录: 31 条
- 被高层记忆引用: 2 条（测试数据）
- 未被引用: 29 条（93.5%）
- 磁盘截图文件: 3,807 个 PNG 文件

**影响**:
- 大量未被使用的截图占用磁盘空间
- 每张截图约 160KB - 2.8MB
- 总存储浪费: 约 500MB - 10GB

**被引用的 2 条 raw_memory 详情**:
```
1. rawmem-10c55c46-d0fa-4fcf-a5bc-2ebc8d74eaa1
   - 来源: Chrome (Integration Test)
   - 创建时间: 2025-11-19 18:24:29
   - OCR 长度: 720 字符
   - 引用者: sem-test-2fe95b77 (Integration Test - Website Screenshot)
   - 类型: 测试数据

2. rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028
   - 来源: Chrome (Integration Test)
   - 创建时间: 2025-11-19 18:24:49
   - OCR 长度: 720 字符
   - 引用者: sem-test-1672b419 (Integration Test - Website Screenshot)
   - 类型: 测试数据
```

### 问题 2: mark_as_processed 机制未启用

**现状**:
- `RawMemoryManager.mark_as_processed()` 方法存在但从未被调用
- 所有 raw_memory 记录: `processed=false, processing_count=0`
- 无法区分哪些截图已被 Meta Memory Agent 处理

**代码位置**:
```python
# mirix/services/raw_memory_manager.py:136-160
def mark_as_processed(self, raw_memory_id: str) -> bool:
    """标记 raw_memory 为已处理"""
    raw_memory.processed = True
    raw_memory.processing_count += 1
    raw_memory.last_modify = {
        "timestamp": datetime.now(dt.timezone.utc).isoformat(),
        "operation": "marked_processed",
    }
    session.commit()
    return True
```

**根本原因**: 需要调查
- Meta Memory Agent 在哪里处理 raw_memory?
- 为什么没有调用 mark_as_processed?
- 是否有其他标记机制?

### 问题 3: 引用关系断裂

**现状**:
- 28 条 semantic_memory 引用了 raw_memory
- 其中 26 条引用的 raw_memory 已被删除（因为没有 OCR text）
- 只有 2 条引用关系完整（测试数据）

**影响**:
- 用户看不到记忆的来源截图
- Memory Library 中的 "Only Referenced" 过滤器失效
- 记忆溯源功能无法使用

---

## 🎯 Phase 2 任务列表

### 优先级 P0 - 关键功能缺失

#### ✅ 任务 1: 调查 mark_as_processed 未被调用的原因
**优先级**: P0 🔴
**估时**: 2 小时
**负责人**: Claude

**子任务**:
1. [ ] 搜索代码中所有调用 Meta Memory Agent 的地方
2. [ ] 检查 `temporary_message_accumulator.py` 中的 `absorb_content_into_memory()` 方法
3. [ ] 查看 `agent_wrapper.py` 中的记忆处理逻辑
4. [ ] 确定应该在哪里调用 `mark_as_processed()`
5. [ ] 文档化调用链: raw_memory 创建 → Meta Memory 处理 → 标记为已处理

**验收标准**:
- [ ] 找到 Meta Memory Agent 处理 raw_memory 的确切位置
- [ ] 确定未调用 mark_as_processed 的根本原因
- [ ] 提出修复方案并记录到文档

**依赖**: 无

---

#### 🔧 任务 2: 实现 raw_memory 的 processed 标记机制
**优先级**: P0 🔴
**估时**: 3 小时
**负责人**: Claude
**依赖**: 任务 1

**子任务**:
1. [ ] 在 Meta Memory Agent 处理完成后调用 `mark_as_processed()`
2. [ ] 传递 raw_memory_ids 到记忆创建函数
3. [ ] 在创建 semantic/episodic/procedural memory 后标记对应的 raw_memory
4. [ ] 添加日志记录标记操作
5. [ ] 测试标记机制是否正常工作

**实现位置**:
```python
# mirix/agent/temporary_message_accumulator.py
def absorb_content_into_memory(self, agent_states, ready_messages=None, user_id=None):
    # 处理完成后
    for raw_memory_id in raw_memory_ids:
        raw_memory_manager.mark_as_processed(raw_memory_id)
```

**验收标准**:
- [ ] 新创建的 raw_memory 在被处理后 `processed=true`
- [ ] `processing_count` 正确递增
- [ ] 日志中记录标记操作
- [ ] 数据库中可以查询到已处理和未处理的记录

**测试计划**:
```bash
# 1. 创建新的截图
# 2. 等待 Meta Memory Agent 处理
# 3. 查询数据库验证
psql -U power -d mirix -c "SELECT id, processed, processing_count FROM raw_memory ORDER BY created_at DESC LIMIT 5;"
```

---

### 优先级 P1 - 存储优化

#### 🗑️ 任务 3: 实现未被引用的截图自动清理机制
**优先级**: P1 🟡
**估时**: 4 小时
**负责人**: Claude
**依赖**: 任务 2

**需求**:
- 未被 Meta Memory Agent 选中的截图，30 分钟后删除物理文件
- 保留数据库记录（metadata、OCR text、source_url 等）
- 被选中的截图（`processed=true` 或有引用关系）永久保留

**子任务**:
1. [ ] 创建定时清理任务（cron job 或后台线程）
2. [ ] 编写清理逻辑：
   ```python
   def cleanup_unreferenced_screenshots():
       # 查找符合条件的 raw_memory
       candidates = raw_memory_manager.get_unprocessed_raw_memories(
           older_than_minutes=30
       )
       for rm in candidates:
           # 检查是否被引用
           if not is_referenced(rm.id):
               # 删除物理文件
               os.remove(rm.screenshot_path)
               # 更新数据库记录
               rm.screenshot_path = None  # 或标记为已删除
               rm.metadata_['screenshot_deleted'] = True
               rm.metadata_['deleted_at'] = datetime.now().isoformat()
   ```
3. [ ] 添加配置选项（可配置清理时间间隔）
4. [ ] 添加安全检查（不删除被引用的截图）
5. [ ] 添加日志记录删除操作
6. [ ] 实现手动触发清理的 API 端点

**配置**:
```python
# mirix/constants.py
SCREENSHOT_CLEANUP_INTERVAL_MINUTES = int(os.getenv("SCREENSHOT_CLEANUP_INTERVAL", "30"))
SCREENSHOT_CLEANUP_ENABLED = os.getenv("SCREENSHOT_CLEANUP_ENABLED", "true").lower() in ("true", "1", "yes")
```

**验收标准**:
- [ ] 30 分钟后未被处理的截图文件被删除
- [ ] 数据库记录保留（ocr_text, source_url 等）
- [ ] 被引用的截图不会被删除
- [ ] 清理操作有日志记录
- [ ] 可以通过环境变量配置清理间隔

**测试计划**:
```bash
# 1. 创建测试截图（不被 Meta Memory 处理）
# 2. 等待 30 分钟（或修改配置为 1 分钟测试）
# 3. 验证文件被删除但数据库记录存在
# 4. 验证被引用的截图未被删除
```

---

#### 📊 任务 4: 前端显示引用关系
**优先级**: P1 🟡
**估时**: 3 小时
**负责人**: Claude
**依赖**: 任务 2

**需求**:
- 在各个 agent 的详情中显示引用的 raw_memory
- "Only Referenced" 过滤器正常工作
- 点击引用跳转到对应的 raw_memory 详情

**子任务**:
1. [ ] 后端 API 添加 `only_referenced` 参数
   ```python
   @app.get("/memory/raw")
   async def get_raw_memories(
       limit: int = 100,
       offset: int = 0,
       only_referenced: bool = False
   ):
       if only_referenced:
           # 查询被引用的 raw_memory
           query = get_referenced_raw_memories_query()
       else:
           # 正常查询
           query = get_all_raw_memories_query()
   ```

2. [ ] 前端 ExistingMemory.js 连接 "Only Referenced" 开关到 API
   ```javascript
   const handleOnlyReferencedToggle = async () => {
     setShowOnlyReferencedRaw(!showOnlyReferencedRaw);
     await fetchMemoryData('raw-memory', { only_referenced: !showOnlyReferencedRaw });
   };
   ```

3. [ ] 在 Semantic/Episodic Memory 详情中显示引用的 raw_memory
4. [ ] 实现点击跳转功能（跳转到 Raw Memory 标签页并高亮）
5. [ ] 添加 "Referenced by" 信息（反向引用）

**UI 设计**:
```
Semantic Memory: "MIRIX Phase 1 Development Knowledge"
├─ Summary: ...
├─ Details: ...
└─ 📸 Source Screenshots (4):
   ├─ Chrome - https://github.com - 2025-11-19 18:30
   ├─ VS Code - 1119log.md - 2025-11-19 18:35
   ├─ Chrome - https://docs.python.org - 2025-11-19 18:40
   └─ Notion - project-meeting - 2025-11-19 18:45
```

**验收标准**:
- [ ] "Only Referenced" 开关正常工作
- [ ] 只显示被引用的 raw_memory（当前应该是 2 条）
- [ ] Semantic Memory 详情中显示来源截图
- [ ] 点击截图可以跳转到 Raw Memory 详情
- [ ] Raw Memory 详情显示 "Referenced by" 信息

---

### 优先级 P2 - 数据修复

#### 🔧 任务 5: 修复断裂的引用关系
**优先级**: P2 🟢
**估时**: 2 小时
**负责人**: Claude
**依赖**: 无

**问题**:
- 26 条 semantic_memory 引用的 raw_memory 已被删除
- 用户无法查看这些记忆的来源

**修复方案 A**: 清理无效引用
```sql
-- 从 raw_memory_references 中移除不存在的 ID
UPDATE semantic_memory
SET raw_memory_references = (
  SELECT jsonb_agg(elem)
  FROM jsonb_array_elements_text(raw_memory_references) elem
  WHERE EXISTS (SELECT 1 FROM raw_memory WHERE id = elem)
)
WHERE raw_memory_references IS NOT NULL;
```

**修复方案 B**: 标记为损坏
```sql
-- 添加 metadata 标记引用已损坏
UPDATE semantic_memory sm
SET metadata_ = jsonb_set(
  metadata_::jsonb,
  '{broken_references}',
  (
    SELECT jsonb_agg(elem)
    FROM jsonb_array_elements_text(raw_memory_references) elem
    WHERE NOT EXISTS (SELECT 1 FROM raw_memory WHERE id = elem)
  )
)
WHERE raw_memory_references IS NOT NULL;
```

**验收标准**:
- [ ] 选择一个修复方案并实施
- [ ] 验证引用关系完整性
- [ ] 前端正确显示修复后的引用

---

### 优先级 P3 - 监控与统计

#### 📈 任务 6: 添加 raw_memory 使用统计
**优先级**: P3 🟢
**估时**: 2 小时
**负责人**: Claude
**依赖**: 任务 2

**功能**:
- Dashboard 显示 raw_memory 统计信息
- 存储空间使用情况
- 处理率和引用率

**统计指标**:
```
Total Raw Memories: 31
├─ Processed: 2 (6.5%)
├─ Unprocessed: 29 (93.5%)
├─ Referenced: 2 (6.5%)
└─ Unreferenced: 29 (93.5%)

Storage:
├─ Total Screenshots: 3,807 files
├─ Total Size: ~8.5 GB
├─ Referenced Size: ~5 MB
└─ Unreferenced Size: ~8.495 GB (99.9%)
```

**API 端点**:
```python
@app.get("/memory/raw/stats")
async def get_raw_memory_stats():
    return {
        "total": 31,
        "processed": 2,
        "referenced": 2,
        "total_files": 3807,
        "total_size_bytes": 9126805504,
        "cleanup_candidates": 29
    }
```

**验收标准**:
- [ ] API 返回准确的统计数据
- [ ] 前端显示统计面板
- [ ] 自动刷新统计数据

---

## 📝 实施顺序

### 第一批（本周）
1. ✅ **任务 1**: 调查 mark_as_processed 原因（2小时）
2. ✅ **任务 2**: 实现 processed 标记机制（3小时）

### 第二批（下周）
3. ✅ **任务 4**: 前端显示引用关系（3小时）
4. ✅ **任务 3**: 实现自动清理机制（4小时）

### 第三批（后续）
5. ✅ **任务 5**: 修复断裂的引用关系（2小时）
6. ✅ **任务 6**: 添加使用统计（2小时）

**总估时**: 16 小时

---

## 🔍 调查笔记

### 调查 1: mark_as_processed 为何未被调用

**待调查文件**:
- `mirix/agent/temporary_message_accumulator.py` - `_build_memory_message()` 方法
- `mirix/agent/temporary_message_accumulator.py` - `absorb_content_into_memory()` 方法
- `mirix/agent/agent_wrapper.py` - 记忆处理逻辑
- `mirix/functions/function_sets/memory_tools.py` - 记忆创建工具函数

**关键问题**:
1. `raw_memory_ids` 是否传递到 Meta Memory Agent？
   - 答: ✅ 是的，在 `_build_memory_message()` 中返回 (line 821)

2. Meta Memory Agent 创建高层记忆后是否知道对应的 raw_memory_ids？
   - 答: ⏳ 待确认

3. 有没有其他方式标记 raw_memory 被处理？
   - 答: ⏳ 待确认

**调查进度**: 0%

---

### 调查 2: 被引用的 raw_memory 详情

**已确认**:
- 总 raw_memory: 31 条
- 被引用且存在: 2 条（测试数据）
- 被引用但已删除: 26 条（无 OCR text 被清理）
- 从未被引用: 29 条

**2 条现存被引用记录**:
```json
[
  {
    "id": "rawmem-10c55c46-d0fa-4fcf-a5bc-2ebc8d74eaa1",
    "source_app": "Chrome",
    "screenshot_path": "/Users/power/.mirix/tmp/images/screenshot-2025-09-05T06-30-37-992Z.png",
    "ocr_len": 720,
    "created_at": "2025-11-19 18:24:29",
    "referenced_by": "sem-test-2fe95b77 (Integration Test - Website Screenshot)",
    "type": "测试数据"
  },
  {
    "id": "rawmem-6e711fee-d8c0-4d16-9036-137f4c5ed028",
    "source_app": "Chrome",
    "screenshot_path": "/Users/power/.mirix/tmp/images/screenshot-2025-09-05T06-30-37-992Z.png",
    "ocr_len": 720,
    "created_at": "2025-11-19 18:24:49",
    "referenced_by": "sem-test-1672b419 (Integration Test - Website Screenshot)",
    "type": "测试数据"
  }
]
```

**结论**: 被引用的都是测试数据，真实的记忆引用关系已断裂。

---

## 🎯 成功指标

### Phase 2 完成标准

- [ ] ✅ `mark_as_processed()` 被正确调用
- [ ] ✅ 所有新 raw_memory 正确标记处理状态
- [ ] ✅ 未被引用的截图 30 分钟后自动清理
- [ ] ✅ 前端 "Only Referenced" 过滤器正常工作
- [ ] ✅ Memory 详情中显示来源截图
- [ ] ✅ 引用关系完整性修复
- [ ] ✅ 存储空间使用优化（节省 90%+）

---

**最后更新**: 2025-11-20
**文档维护**: Claude + User
