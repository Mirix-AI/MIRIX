"""
导入真实截图到 raw_memory 表

从 ~/.mirix/tmp/images/ 目录导入真实截图，执行 OCR 提取文本和 URL
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import uuid

# 设置项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 禁用自动嵌入构建（加快速度）
os.environ['BUILD_EMBEDDINGS_FOR_MEMORY'] = 'false'

from mirix.server.server import db_context
from mirix.orm.raw_memory import RawMemoryItem
from mirix.helpers.ocr_url_extractor import OCRUrlExtractor
from mirix.services.raw_memory_manager import RawMemoryManager


def import_real_screenshots(limit=10, skip_existing=True):
    """
    导入真实截图

    Args:
        limit: 最多导入多少张（None 表示全部）
        skip_existing: 是否跳过已导入的截图
    """
    # 获取真实截图目录
    screenshot_dir = Path.home() / '.mirix' / 'tmp' / 'images'

    if not screenshot_dir.exists():
        print(f"❌ 截图目录不存在: {screenshot_dir}")
        return

    # 获取所有截图文件
    all_screenshots = sorted(screenshot_dir.glob('screenshot-*.png'), key=os.path.getmtime, reverse=True)

    print(f"📸 找到 {len(all_screenshots)} 张截图")

    if limit:
        screenshots = all_screenshots[:limit]
        print(f"📋 将导入前 {len(screenshots)} 张\n")
    else:
        screenshots = all_screenshots
        print(f"📋 将导入全部 {len(screenshots)} 张\n")

    # 初始化 OCR 提取器
    ocr_extractor = OCRUrlExtractor()

    # 初始化 RawMemoryManager
    raw_memory_manager = RawMemoryManager()

    imported_count = 0
    skipped_count = 0
    error_count = 0

    with db_context() as session:
        # 如果需要跳过已存在的，先获取已导入的截图路径
        existing_paths = set()
        if skip_existing:
            existing_items = session.query(RawMemoryItem.screenshot_path).all()
            existing_paths = {item[0] for item in existing_items}
            print(f"ℹ️  数据库中已有 {len(existing_paths)} 张截图\n")

        for screenshot_path in screenshots:
            screenshot_path_str = str(screenshot_path)

            # 检查是否已导入
            if skip_existing and screenshot_path_str in existing_paths:
                print(f"⏭️  跳过已导入: {screenshot_path.name}")
                skipped_count += 1
                continue

            print(f"🔄 处理: {screenshot_path.name}")

            try:
                # 执行 OCR 提取
                ocr_text, urls = ocr_extractor.extract_urls_and_text(screenshot_path_str)

                # 获取文件修改时间作为捕获时间
                captured_at = datetime.fromtimestamp(os.path.getmtime(screenshot_path_str))

                # 推断 source_app（从文件名或其他元数据）
                # 这里简单使用 "Screen" 作为默认值，后续可以根据实际情况改进
                source_app = "Screen"

                # 如果有 URL，使用第一个作为 source_url
                source_url = urls[0] if urls else None

                # 创建 raw_memory 记录
                raw_memory_id = f"rawmem-{uuid.uuid4()}"

                # 使用 RawMemoryManager 插入
                raw_memory_manager.insert_raw_memory(
                    session=session,
                    raw_memory_id=raw_memory_id,
                    screenshot_path=screenshot_path_str,
                    source_app=source_app,
                    source_url=source_url,
                    ocr_text=ocr_text,
                    captured_at=captured_at,
                    user_id="user-00000000-0000-4000-8000-000000000001",
                    organization_id="org-00000000-0000-4000-8000-000000000000"
                )

                imported_count += 1
                print(f"  ✅ OCR: {len(ocr_text) if ocr_text else 0} 字符")
                print(f"  ✅ URL: {source_url or '无'}")
                print(f"  ✅ 时间: {captured_at.strftime('%Y-%m-%d %H:%M:%S')}\n")

            except Exception as e:
                error_count += 1
                print(f"  ❌ 错误: {e}\n")

        # 提交事务
        session.commit()

    # 打印统计信息
    print("\n" + "="*60)
    print("📊 导入统计")
    print("="*60)
    print(f"✅ 成功导入: {imported_count} 张")
    print(f"⏭️  跳过已存在: {skipped_count} 张")
    print(f"❌ 导入失败: {error_count} 张")
    print(f"📸 总处理: {imported_count + skipped_count + error_count} 张")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='导入真实截图到 raw_memory 表')
    parser.add_argument('--limit', type=int, default=10, help='最多导入多少张（默认 10，0 表示全部）')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已存在的截图')

    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    skip_existing = not args.no_skip

    print("🚀 开始导入真实截图...\n")
    import_real_screenshots(limit=limit, skip_existing=skip_existing)
    print("\n✅ 导入完成！")
