#!/bin/bash

# MIRIX Memory Pipeline Diagnostic Script
# 用于诊断记忆生成流程的每个环节

echo "================================================================================"
echo "MIRIX 记忆生成流程诊断工具"
echo "================================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Database connection
DB_NAME="mirix"
DB_USER="power"

echo "1️⃣  Raw Memory 数量统计"
echo "--------------------------------------------------------------------------------"
psql -U $DB_USER -d $DB_NAME -c "
SELECT
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE processed = true) as processed,
    COUNT(*) FILTER (WHERE processed = false) as pending
FROM raw_memory;
" | tail -n +3

echo ""
echo "2️⃣  最新的 5 条 Raw Memory"
echo "--------------------------------------------------------------------------------"
psql -U $DB_USER -d $DB_NAME -c "
SELECT
    id,
    source_app,
    LEFT(source_url, 50) as url,
    captured_at,
    processed
FROM raw_memory
ORDER BY captured_at DESC
LIMIT 5;
" | tail -n +3

echo ""
echo "3️⃣  Semantic Memory 统计"
echo "--------------------------------------------------------------------------------"
psql -U $DB_USER -d $DB_NAME -c "
SELECT
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE raw_memory_references IS NOT NULL AND raw_memory_references::text != '[]') as with_references
FROM semantic_memory;
" | tail -n +3

echo ""
echo "4️⃣  最新的 5 条 Semantic Memory"
echo "--------------------------------------------------------------------------------"
psql -U $DB_USER -d $DB_NAME -c "
SELECT
    id,
    name,
    created_at,
    (raw_memory_references::text != '[]' AND raw_memory_references::text != 'null') as has_refs
FROM semantic_memory
ORDER BY created_at DESC
LIMIT 5;
" | tail -n +3

echo ""
echo "5️⃣  References 关联检查"
echo "--------------------------------------------------------------------------------"
psql -U $DB_USER -d $DB_NAME -c "
SELECT
    sm.name as semantic_name,
    jsonb_array_length(sm.raw_memory_references::jsonb) as ref_count
FROM semantic_memory sm
WHERE raw_memory_references IS NOT NULL
  AND raw_memory_references::text != '[]'
ORDER BY ref_count DESC
LIMIT 10;
" | tail -n +3

echo ""
echo "6️⃣  SKIP_META_MEMORY_MANAGER 配置"
echo "--------------------------------------------------------------------------------"
if [ -f "mirix/agent/app_constants.py" ]; then
    grep "SKIP_META_MEMORY_MANAGER" mirix/agent/app_constants.py
else
    echo -e "${RED}❌ 文件未找到${NC}"
fi

echo ""
echo "7️⃣  后端日志检查 (最近 50 行)"
echo "--------------------------------------------------------------------------------"
if [ -f "/tmp/mirix_server.log" ]; then
    echo -e "${GREEN}✓${NC} 后端日志存在"
    echo ""
    echo "📝 Raw Memory 相关日志:"
    tail -100 /tmp/mirix_server.log | grep -i "raw_memory\|screenshot" | tail -10
    echo ""
    echo "📝 Memory Agent 相关日志:"
    tail -100 /tmp/mirix_server.log | grep -i "memory_agent\|semantic_memory" | tail -10
else
    echo -e "${YELLOW}⚠${NC}  后端日志文件不存在: /tmp/mirix_server.log"
fi

echo ""
echo "8️⃣  数据库连接测试"
echo "--------------------------------------------------------------------------------"
psql -U $DB_USER -d $DB_NAME -c "SELECT version();" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 数据库连接正常${NC}"
else
    echo -e "${RED}❌ 数据库连接失败${NC}"
fi

echo ""
echo "9️⃣  API 端点测试"
echo "--------------------------------------------------------------------------------"
# Check if server is running
if curl -s http://localhost:47283/memory/raw > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 后端服务正在运行${NC}"

    # Test new screenshot endpoint
    FIRST_RAW_ID=$(psql -U $DB_USER -d $DB_NAME -t -c "SELECT id FROM raw_memory LIMIT 1;" | xargs)
    if [ ! -z "$FIRST_RAW_ID" ]; then
        echo "   测试 screenshot 端点: /raw_memory/$FIRST_RAW_ID/screenshot"
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:47283/raw_memory/$FIRST_RAW_ID/screenshot)
        if [ "$HTTP_CODE" = "200" ]; then
            echo -e "   ${GREEN}✅ Screenshot API 正常 (HTTP $HTTP_CODE)${NC}"
        elif [ "$HTTP_CODE" = "404" ]; then
            echo -e "   ${YELLOW}⚠  Screenshot 文件不存在 (HTTP $HTTP_CODE)${NC}"
        else
            echo -e "   ${RED}❌ Screenshot API 错误 (HTTP $HTTP_CODE)${NC}"
        fi
    fi
else
    echo -e "${RED}❌ 后端服务未运行 (期望在 http://localhost:47283)${NC}"
fi

echo ""
echo "🔟  截图文件检查"
echo "--------------------------------------------------------------------------------"
# Check screenshot directory
SCREENSHOT_DIR="$HOME/.mirix/tmp/images"
if [ -d "$SCREENSHOT_DIR" ]; then
    SCREENSHOT_COUNT=$(find "$SCREENSHOT_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} 截图目录存在: $SCREENSHOT_DIR"
    echo "   截图数量: $SCREENSHOT_COUNT"
    if [ $SCREENSHOT_COUNT -gt 0 ]; then
        echo "   最新的 3 个截图:"
        find "$SCREENSHOT_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) -print0 2>/dev/null | xargs -0 ls -lt | head -3 | awk '{print "   - " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}'
    fi
else
    echo -e "${YELLOW}⚠${NC}  截图目录不存在: $SCREENSHOT_DIR"
fi

echo ""
echo "================================================================================"
echo "诊断完成！"
echo "================================================================================"
echo ""

# Summary
echo "📊 快速摘要:"
TOTAL_RAW=$(psql -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM raw_memory;" | xargs)
TOTAL_SEM=$(psql -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM semantic_memory;" | xargs)
SEM_WITH_REFS=$(psql -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM semantic_memory WHERE raw_memory_references IS NOT NULL AND raw_memory_references::text != '[]';" | xargs)

echo "   Raw Memory: $TOTAL_RAW 条"
echo "   Semantic Memory: $TOTAL_SEM 条 (其中 $SEM_WITH_REFS 条有 references)"

if [ $TOTAL_RAW -eq 0 ]; then
    echo -e "${RED}   ⚠️  没有 Raw Memory 数据！${NC}"
    echo "   可能原因:"
    echo "   1. 截图监控未启动"
    echo "   2. OCR 提取失败"
    echo "   3. 数据库写入失败"
elif [ $SEM_WITH_REFS -eq 0 ] && [ $TOTAL_SEM -gt 0 ]; then
    echo -e "${YELLOW}   ⚠️  有 Semantic Memory 但没有 references！${NC}"
    echo "   可能原因:"
    echo "   1. 旧数据（Phase 1 之前创建）"
    echo "   2. Memory agent 未传递 raw_memory_references"
    echo "   3. 需要重新生成记忆"
else
    echo -e "${GREEN}   ✅ 数据看起来正常！${NC}"
fi

echo ""
echo "📝 下一步建议:"
echo "   1. 如果 Raw Memory 数量不增长 → 检查 Electron 截图监控"
echo "   2. 如果 Semantic Memory 没有 references → 检查 Memory Agent"
echo "   3. 如果前端看不到 references → 强制刷新浏览器 (Ctrl+Shift+R)"
echo "   4. 查看详细分析: cat UAT_ISSUES_ANALYSIS.md"
echo ""
