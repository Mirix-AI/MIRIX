#!/usr/bin/env python3
"""
MIRIX 内存系统指定测试运行器

这个脚本允许您运行特定的内存测试，而不需要运行所有测试。

使用方法:
    # 从项目根目录运行
    python tests/run_specific_memory_test.py <test_name>
    
    # 或者进入tests目录运行
    cd tests
    python run_specific_memory_test.py <test_name>
    
或者运行多个测试:
    python tests/run_specific_memory_test.py <test_name1> <test_name2> ...

示例:
    # 运行情节记忆直接操作测试
    python tests/run_specific_memory_test.py episodic_memory_direct
    
    # 运行多个直接操作测试
    python tests/run_specific_memory_test.py episodic_memory_direct procedural_memory_direct resource_memory_direct
    
    # 运行所有直接内存操作测试
    python tests/run_specific_memory_test.py all_direct_memory_operations
    
    # 运行搜索相关测试
    python tests/run_specific_memory_test.py search_methods fts5_comprehensive
    
    # 保留测试数据（不清理）
    python tests/run_specific_memory_test.py episodic_memory_direct --keep-data

可用的测试名称:
    # 直接内存操作 (manager methods)
    - episodic_memory_direct: 测试情节记忆直接操作
    - procedural_memory_direct: 测试程序记忆直接操作
    - resource_memory_direct: 测试资源记忆直接操作
    - knowledge_vault_direct: 测试知识库直接操作
    - semantic_memory_direct: 测试语义记忆直接操作
    - resource_memory_update_direct: 测试资源记忆更新直接操作
    - tree_path_functionality_direct: 测试树形路径功能直接操作
    
    # 间接内存操作 (message-based)
    - episodic_memory_indirect: 测试情节记忆间接操作
    - procedural_memory_indirect: 测试程序记忆间接操作
    - resource_memory_indirect: 测试资源记忆间接操作
    - knowledge_vault_indirect: 测试知识库间接操作
    - semantic_memory_indirect: 测试语义记忆间接操作
    - resource_memory_update_indirect: 测试资源记忆更新间接操作
    
    # 搜索和性能测试
    - search_methods: 测试不同搜索方法
    - fts5_comprehensive: 测试FTS5综合功能
    - fts5_performance_comparison: 测试FTS5性能对比
    - fts5_advanced_features: 测试FTS5高级功能
    - text_only_memorization: 测试纯文本记忆功能
    
    # 核心记忆测试
    - core_memory_update_using_chat_agent: 测试使用聊天代理更新核心记忆
    
    # 文件处理测试
    - greeting_with_files: 测试文件处理功能
    - file_types: 测试不同文件类型
    - file_with_memory: 测试带记忆的文件处理
    
    # 综合测试
    - all_direct_memory_operations: 运行所有直接内存操作测试
    - all_indirect_memory_operations: 运行所有间接内存操作测试
    - all_search_and_performance_operations: 运行所有搜索和性能测试
    - all_memories: 运行所有内存测试
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入测试函数
from tests.test_memory import run_specific_memory_test, run_multiple_memory_tests

def print_usage():
    """打印使用说明"""
    print(__doc__)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("❌ 请提供至少一个测试名称")
        print_usage()
        sys.exit(1)
    
    # 获取测试名称列表
    test_names = sys.argv[1:]
    
    # 检查是否是帮助请求
    if any(arg in ['-h', '--help', 'help'] for arg in test_names):
        print_usage()
        sys.exit(0)
    
    print(f"🎯 准备运行 {len(test_names)} 个测试")
    print(f"测试列表: {', '.join(test_names)}")
    print("="*80)
    
    # 检查是否要保留测试数据
    delete_after_test = True
    if '--keep-data' in test_names:
        delete_after_test = False
        test_names.remove('--keep-data')
        print("⚠️  将保留测试数据（不进行清理）")
    
    # 运行测试
    if len(test_names) == 1:
        # 单个测试
        test_name = test_names[0]
        success = run_specific_memory_test(test_name, delete_after_test=delete_after_test)
        
        if success:
            print(f"\n🎉 测试 '{test_name}' 成功完成!")
            sys.exit(0)
        else:
            print(f"\n💥 测试 '{test_name}' 失败!")
            sys.exit(1)
    else:
        # 多个测试
        results = run_multiple_memory_tests(test_names, delete_after_test=delete_after_test)
        
        # 检查是否有失败的测试
        failed_tests = [name for name, success in results.items() if not success]
        
        if not failed_tests:
            print(f"\n🎉 所有 {len(test_names)} 个测试都成功完成!")
            sys.exit(0)
        else:
            print(f"\n💥 {len(failed_tests)} 个测试失败: {', '.join(failed_tests)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
