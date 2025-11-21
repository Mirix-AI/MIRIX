#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Volumes/Lexar/AISync90/MIRIX')

from mirix.helpers.ocr_url_extractor import OCRUrlExtractor

screenshot_path = "/Users/power/.mirix/tmp/images/screenshot-2025-11-20T16-52-29-793Z.png"
print(f"测试OCR: {screenshot_path}\n")

try:
    text, urls = OCRUrlExtractor.extract_urls_and_text(screenshot_path)
    print(f"✅ OCR成功!")
    print(f"文本长度: {len(text) if text else 0} 字符")
    print(f"URLs: {len(urls)} 个\n")

    if text:
        print(f"前500字符:")
        print("="*80)
        print(text[:500])
        print("="*80)
    else:
        print("⚠️  没有提取到文本")

    if urls:
        print(f"\nURLs:")
        for url in urls:
            print(f"  - {url}")
except Exception as e:
    print(f"❌ OCR失败: {e}")
    import traceback
    traceback.print_exc()
