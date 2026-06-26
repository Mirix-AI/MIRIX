import re
from pathlib import Path

MANAGERS = Path("mirix/services")


def test_cache_read_failed_is_not_warning():
    offenders = []
    for f in MANAGERS.glob("*_manager.py"):
        text = f.read_text()
        for m in re.finditer(r"logger\.warning\([^)]*Cache read failed", text, re.DOTALL):
            offenders.append(f"{f.name}: {m.group(0)[:60]}")
    assert not offenders, f"Cache-read should log at DEBUG, found WARNING in: {offenders}"
