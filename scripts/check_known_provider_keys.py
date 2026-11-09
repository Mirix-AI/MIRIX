"""Prevent reintroducing the provider credentials exposed in commit 421a8be."""

import hashlib
import os
import re
import subprocess
from pathlib import Path


# Store only SHA-256 fingerprints of the two previously exposed credentials.
EXPOSED_KEY_HASHES = {
    "a0d45f2ab22de9d25aca8681e239e1cad696cfc467c5531e4b240d5175205328",
    "64f60ddcedfa7c6c262e5df662bcd47d7a33b8f4e805c3f84866829ce30cd91b",
}
KEY_PATTERN = re.compile(rb"AIza[0-9A-Za-z_-]{35}|sk-proj-[0-9A-Za-z_-]{156}")


def contains_exposed_key(data: bytes) -> bool:
    return any(hashlib.sha256(match[0]).hexdigest() in EXPOSED_KEY_HASHES for match in KEY_PATTERN.finditer(data))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    failures = []
    for name in tracked.split(b"\0"):
        if not name:
            continue
        path = root / os.fsdecode(name)
        if path.is_file() and not path.is_symlink() and contains_exposed_key(path.read_bytes()):
            failures.append(os.fsdecode(KEY_PATTERN.sub(b"<redacted>", name)))
    for name in failures:
        print(f"Previously exposed provider credential in {name!r}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
