#!/usr/bin/env python3
"""
测试截图处理功能
直接调用后端API，测试OCR和raw_memory插入的完整流程
"""

import requests
import json
import sys
import os
from pathlib import Path

# 服务器配置
SERVER_URL = "http://localhost:47283"
API_ENDPOINT = f"{SERVER_URL}/test/process_screenshot"

def test_screenshot(screenshot_path: str, source_app: str = "TestApp"):
    """
    测试单个截图的处理流程

    Args:
        screenshot_path: 截图文件的完整路径
        source_app: 来源应用名称
    """
    print(f"\n{'='*80}")
    print(f"测试截图处理")
    print(f"{'='*80}")
    print(f"📸 截图路径: {screenshot_path}")
    print(f"📱 来源应用: {source_app}")
    print(f"🌐 API端点: {API_ENDPOINT}")
    print(f"{'='*80}\n")

    # 检查文件是否存在
    if not os.path.exists(screenshot_path):
        print(f"❌ 错误: 文件不存在: {screenshot_path}")
        return False

    file_size = os.path.getsize(screenshot_path)
    print(f"✅ 文件存在，大小: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    # 准备请求数据
    request_data = {
        "screenshot_path": screenshot_path,
        "source_app": source_app
    }

    print(f"\n📤 发送请求到后端...")
    print(f"请求数据: {json.dumps(request_data, indent=2, ensure_ascii=False)}")

    try:
        # 发送POST请求
        response = requests.post(
            API_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=120  # 120秒超时（OCR可能需要更长时间）
        )

        print(f"\n📥 收到响应: HTTP {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n{'='*80}")
            print(f"✅ 测试成功！")
            print(f"{'='*80}")
            print(f"🆔 Raw Memory ID: {result.get('raw_memory_id')}")
            print(f"📝 OCR文本长度: {result.get('ocr_text_length')} 字符")
            print(f"🔗 找到的URLs: {len(result.get('urls_found', []))} 个")
            if result.get('urls_found'):
                for i, url in enumerate(result['urls_found'], 1):
                    print(f"   {i}. {url}")
            print(f"🌐 主URL: {result.get('source_url', 'None')}")
            print(f"📅 捕获时间: {result.get('captured_at')}")

            # 显示OCR预览
            if result.get('ocr_preview'):
                print(f"\n📄 OCR文本预览 (前200字符):")
                print(f"{'─'*80}")
                print(result['ocr_preview'])
                print(f"{'─'*80}")

            print(f"\n💾 完整响应:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"{'='*80}\n")
            return True

        else:
            print(f"\n❌ 请求失败: HTTP {response.status_code}")
            print(f"错误信息: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 连接错误: 无法连接到服务器 {SERVER_URL}")
        print(f"   请确保服务器正在运行 (python main.py)")
        return False
    except requests.exceptions.Timeout:
        print(f"\n❌ 请求超时: 处理时间超过30秒")
        return False
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_in_database(raw_memory_id: str):
    """验证数据是否成功写入数据库"""
    print(f"\n{'='*80}")
    print(f"验证数据库记录")
    print(f"{'='*80}")

    import subprocess

    cmd = [
        "psql", "-U", "power", "-d", "mirix", "-c",
        f"SELECT id, source_app, source_url, LENGTH(ocr_text) as ocr_len, captured_at FROM raw_memory WHERE id = '{raw_memory_id}';"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print(result.stdout)

        if raw_memory_id in result.stdout:
            print(f"✅ 数据库验证成功！记录已存在。\n")
            return True
        else:
            print(f"❌ 数据库验证失败！未找到记录。\n")
            return False
    except Exception as e:
        print(f"⚠️  数据库验证跳过: {e}\n")
        return None


def main():
    """主函数"""
    # 默认测试文件（用户提供的真实截图）
    default_screenshot = "/Users/power/.mirix/tmp/images/screenshot-window:9554:0-2025-11-20T09-10-10-755Z.png"

    # 如果命令行提供了参数，使用命令行参数
    if len(sys.argv) > 1:
        screenshot_path = sys.argv[1]
        source_app = sys.argv[2] if len(sys.argv) > 2 else "TestApp"
    else:
        screenshot_path = default_screenshot
        source_app = "全屏"

    # 测试截图处理
    success = test_screenshot(screenshot_path, source_app)

    if success:
        # 如果成功，从响应中获取raw_memory_id并验证数据库
        # 注意：这里需要解析之前的响应，简化起见，让用户手动验证
        print(f"\n💡 提示: 你可以使用以下命令验证数据库:")
        print(f"   psql -U power -d mirix -c \"SELECT id, source_app, source_url, LENGTH(ocr_text) FROM raw_memory WHERE metadata_::jsonb @> '{{\\\"test\\\": true}}' ORDER BY created_at DESC LIMIT 5;\"")
        print(f"\n或者使用 /memory/raw API 查看:")
        print(f"   curl {SERVER_URL}/memory/raw | jq '.items[:3]'")

        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
