#!/bin/bash
# 启用 LLM 调试日志记录的环境变量设置脚本

echo "🔍 启用 LLM 调试日志记录..."

# 设置环境变量
export LLM_DEBUG_ENABLE_FILE=true
export LLM_DEBUG_LOG_DIR="./logs/llm_debug"
export LOG_LEVEL=DEBUG

echo "✅ 环境变量已设置:"
echo "   LLM_DEBUG_ENABLE_FILE=$LLM_DEBUG_ENABLE_FILE"
echo "   LLM_DEBUG_LOG_DIR=$LLM_DEBUG_LOG_DIR"
echo "   LOG_LEVEL=$LOG_LEVEL"
echo ""
echo "📁 日志文件将保存到: $LLM_DEBUG_LOG_DIR"
echo ""
echo "🚀 现在可以运行测试了:"
echo "   python -m tests.run_specific_memory_test episodic_memory_indirect --config mirix/configs/mirix_qwen3.yaml"
