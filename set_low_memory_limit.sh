#!/bin/bash
# 设置较低的 TEMPORARY_MESSAGE_LIMIT 用于测试记忆功能

echo "🔧 设置较低的 TEMPORARY_MESSAGE_LIMIT 用于测试记忆功能..."

# 设置环境变量
export TEMPORARY_MESSAGE_LIMIT=1
export BUILD_EMBEDDINGS_FOR_MEMORY=false

echo "✅ 环境变量已设置:"
echo "   TEMPORARY_MESSAGE_LIMIT=$TEMPORARY_MESSAGE_LIMIT"
echo ""

echo "🚀 现在可以运行测试了:"
echo "   python test_memory_with_low_limit.py"
echo ""

echo "📝 或者运行原始测试:"
echo "   python -m tests.run_specific_memory_test episodic_memory_indirect --config mirix/configs/mirix_qwen3.yaml"
echo ""

echo "💡 提示: 现在只需要发送 1 条消息就会触发记忆保存，而不是默认的 20 条"
