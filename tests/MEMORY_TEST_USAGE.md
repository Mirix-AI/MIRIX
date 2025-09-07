# MIRIX 内存系统指定测试使用指南

## 概述

现在您可以使用新的指定测试功能来运行单个或多个内存测试，而不需要运行所有测试。这大大提高了测试的灵活性和效率。

## 新增功能

### 1. `run_specific_memory_test(test_name, agent=None, delete_after_test=True)`

运行指定的单个内存测试函数。

**参数:**
- `test_name` (str): 要运行的测试名称
- `agent` (AgentWrapper, optional): AgentWrapper实例，如果为None则自动创建
- `delete_after_test` (bool): 是否在测试后清理测试数据，默认为True

**返回值:**
- `bool`: 测试是否成功完成

### 2. `run_multiple_memory_tests(test_names, agent=None, delete_after_test=True)`

运行多个指定的内存测试函数。

**参数:**
- `test_names` (list): 要运行的测试名称列表
- `agent` (AgentWrapper, optional): AgentWrapper实例，如果为None则自动创建
- `delete_after_test` (bool): 是否在测试后清理测试数据，默认为True

**返回值:**
- `dict`: 每个测试的结果 {'test_name': success_bool}

## 使用方法

### 方法1: 在Python代码中直接调用

```python
from tests.test_memory import run_specific_memory_test, run_multiple_memory_tests

# 运行单个测试（默认清理测试数据）
success = run_specific_memory_test('episodic_memory_direct')
if success:
    print("测试通过!")
else:
    print("测试失败!")

# 运行单个测试（保留测试数据）
success = run_specific_memory_test('episodic_memory_direct', delete_after_test=False)

# 运行多个测试（默认清理测试数据）
results = run_multiple_memory_tests([
    'episodic_memory_direct',
    'procedural_memory_direct', 
    'resource_memory_direct'
])

# 运行多个测试（保留测试数据）
results = run_multiple_memory_tests([
    'episodic_memory_direct',
    'procedural_memory_direct'
], delete_after_test=False)

# 检查结果
for test_name, success in results.items():
    print(f"{test_name}: {'通过' if success else '失败'}")
```

### 方法2: 使用独立的测试脚本

```bash
# 从项目根目录运行
python tests/run_specific_memory_test.py episodic_memory_direct

# 运行多个测试
python tests/run_specific_memory_test.py episodic_memory_direct procedural_memory_direct resource_memory_direct

# 运行所有直接内存操作测试
python tests/run_specific_memory_test.py all_direct_memory_operations

# 运行搜索相关测试
python tests/run_specific_memory_test.py search_methods fts5_comprehensive

# 保留测试数据（不清理）
python tests/run_specific_memory_test.py episodic_memory_direct --keep-data

# 查看帮助
python -m tests.run_specific_memory_test --help

# 或者进入tests目录运行
cd tests
python -m testsrun_specific_memory_test episodic_memory_direct
```

### 方法3: 修改 test_memory.py 主函数

在 `test_memory.py` 文件的 `if __name__ == "__main__":` 部分，取消注释相应的测试调用：

```python
if __name__ == "__main__":
    # 运行单个测试
    run_specific_memory_test('episodic_memory_direct')
    
    # 或者运行多个测试
    # run_multiple_memory_tests([
    #     'episodic_memory_direct',
    #     'procedural_memory_direct', 
    #     'resource_memory_direct'
    # ])
```

## 可用的测试名称

### 直接内存操作 (manager methods)
- `episodic_memory_direct`: 测试情节记忆直接操作
- `procedural_memory_direct`: 测试程序记忆直接操作
- `resource_memory_direct`: 测试资源记忆直接操作
- `knowledge_vault_direct`: 测试知识库直接操作
- `semantic_memory_direct`: 测试语义记忆直接操作
- `resource_memory_update_direct`: 测试资源记忆更新直接操作
- `tree_path_functionality_direct`: 测试树形路径功能直接操作

### 间接内存操作 (message-based)
- `episodic_memory_indirect`: 测试情节记忆间接操作
- `procedural_memory_indirect`: 测试程序记忆间接操作
- `resource_memory_indirect`: 测试资源记忆间接操作
- `knowledge_vault_indirect`: 测试知识库间接操作
- `semantic_memory_indirect`: 测试语义记忆间接操作
- `resource_memory_update_indirect`: 测试资源记忆更新间接操作

### 搜索和性能测试
- `search_methods`: 测试不同搜索方法
- `fts5_comprehensive`: 测试FTS5综合功能
- `fts5_performance_comparison`: 测试FTS5性能对比
- `fts5_advanced_features`: 测试FTS5高级功能
- `text_only_memorization`: 测试纯文本记忆功能

### 核心记忆测试
- `core_memory_update_using_chat_agent`: 测试使用聊天代理更新核心记忆

### 文件处理测试
- `greeting_with_files`: 测试文件处理功能
- `file_types`: 测试不同文件类型
- `file_with_memory`: 测试带记忆的文件处理

### 综合测试
- `all_direct_memory_operations`: 运行所有直接内存操作测试
- `all_indirect_memory_operations`: 运行所有间接内存操作测试
- `all_search_and_performance_operations`: 运行所有搜索和性能测试
- `all_memories`: 运行所有内存测试

## 使用场景示例

### 场景1: 开发时快速测试特定功能
```python
# 只测试情节记忆功能
run_specific_memory_test('episodic_memory_direct')
```

### 场景2: 测试所有直接操作
```python
# 测试所有直接内存操作，不包含消息传递
run_specific_memory_test('all_direct_memory_operations')
```

### 场景3: 性能测试
```python
# 只运行搜索和性能相关测试
run_multiple_memory_tests([
    'search_methods',
    'fts5_performance_comparison',
    'fts5_advanced_features'
])
```

### 场景4: 调试特定问题
```python
# 如果怀疑某个特定内存类型有问题
run_specific_memory_test('resource_memory_direct')
```

## 优势

1. **快速反馈**: 只运行需要的测试，节省时间
2. **精确调试**: 可以针对特定功能进行测试
3. **灵活组合**: 可以自由组合不同的测试
4. **易于集成**: 可以轻松集成到CI/CD流程中
5. **详细报告**: 提供详细的测试结果和摘要

## 测试数据清理机制

### 自动清理机制

MIRIX 的测试系统具有智能的数据清理机制：

#### **直接内存操作测试**
- 每个直接操作测试函数都会在测试完成后自动清理创建的测试数据
- 例如：`test_episodic_memory_direct` 会删除插入的事件记录
- 例如：`test_resource_memory_direct` 会删除插入的资源记录

#### **间接内存操作测试**
- 间接操作测试（通过消息传递）通常**不清理数据**
- 这些数据会保留在数据库中，供后续测试使用
- 这是为了测试消息传递和记忆累积的效果

#### **搜索和性能测试**
- 这些测试通常不创建持久数据，因此不需要清理

### 控制清理行为

您可以通过 `delete_after_test` 参数控制是否清理测试数据：

```python
# 默认行为：清理测试数据
run_specific_memory_test('episodic_memory_direct')

# 保留测试数据
run_specific_memory_test('episodic_memory_direct', delete_after_test=False)

# 命令行方式：保留测试数据
python tests/run_specific_memory_test.py episodic_memory_direct --keep-data
```

### 清理策略说明

1. **直接操作测试**: 数据在测试函数内部清理，`delete_after_test` 参数主要影响额外的清理逻辑
2. **间接操作测试**: 通常保留数据，`delete_after_test=False` 时明确保留
3. **搜索测试**: 不创建持久数据，无需清理
4. **综合测试**: 继承各个子测试的清理行为

## 嵌入模型配置

### 控制嵌入向量计算

MIRIX 支持通过环境变量控制是否计算嵌入向量：

```bash
# 禁用嵌入向量计算（推荐用于测试）
export BUILD_EMBEDDINGS_FOR_MEMORY=false

# 启用嵌入向量计算（默认）
export BUILD_EMBEDDINGS_FOR_MEMORY=true
```

### 搜索方法说明

MIRIX 支持多种搜索方法，严格按照环境变量配置执行：

#### **1. `embedding` 搜索**
- **需要**: 嵌入模型API密钥配置 + `BUILD_EMBEDDINGS_FOR_MEMORY=true`
- **优势**: 语义相似度搜索，理解上下文
- **行为**: 如果环境变量不支持，给出警告并使用 `bm25` 搜索

#### **2. `bm25` 搜索（推荐）**
- **需要**: 无需额外配置
- **优势**: 使用数据库原生全文搜索，性能优异
- **支持**: PostgreSQL 和 SQLite

#### **3. `string_match` 搜索**
- **需要**: 无需额外配置
- **优势**: 简单字符串匹配，快速可靠
- **适用**: 精确匹配场景

### 测试建议

```bash
# 推荐：禁用嵌入向量计算进行测试
export BUILD_EMBEDDINGS_FOR_MEMORY=false
python -m tests.run_specific_memory_test episodic_memory_direct

# 启用嵌入向量计算（需要配置API密钥）
export BUILD_EMBEDDINGS_FOR_MEMORY=true
export GEMINI_API_KEY=your_api_key_here
python -m tests.run_specific_memory_test episodic_memory_direct
```

### 行为说明

- **插入时**: 严格按照 `BUILD_EMBEDDINGS_FOR_MEMORY` 环境变量决定是否计算嵌入向量
- **查询时**: 如果请求嵌入搜索但环境变量不支持，会显示警告并使用 BM25 搜索
- **无自动降级**: 不再有异常捕获和自动降级，完全按照配置执行

## 注意事项

1. 某些测试（如文件处理测试）需要特定的测试文件存在
2. 测试会使用现有的数据库连接配置
3. 建议在测试环境中运行，避免影响生产数据
4. 如果测试失败，会显示详细的错误信息和堆栈跟踪
5. **测试数据清理**: 默认情况下会清理测试数据，使用 `--keep-data` 或 `delete_after_test=False` 可以保留数据
6. **间接操作测试**: 通常保留数据以测试记忆累积效果，这是正常行为
7. **嵌入模型配置**: 严格按照环境变量 `BUILD_EMBEDDINGS_FOR_MEMORY` 执行，不支持自动降级
