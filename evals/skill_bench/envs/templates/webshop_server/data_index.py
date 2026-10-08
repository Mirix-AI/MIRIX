"""The WebShop server image's data and index steps.

Run by the image build (``Dockerfile``) before ``server.py`` is copied in, so that a
change to the server never downloads the product files again or rebuilds the index:
their layers depend on this file and ``data.json`` only (and, for the index, on the
pinned requirements and upstream commit it is built with).

* ``fetch-data``: the pinned product files (about 5.7 GB, a Hugging Face mirror pinned by
  revision) into the data directory, each verified by size and sha256 (``data.json``).
* ``build-index``: upstream's ``convert_product_file_format.py`` and ``run_indexing.sh``
  for the full product set only, with ``--threads 1``, then the index record
  (:data:`INDEX_RECORD`, :func:`index_record`). Its sha256 is the image's
  ``index_sha256`` (``server.py write-build-record``), which ``/ready`` and the harness
  compare with the frozen manifest's.

The index record is deterministic: two builds from the same product files, upstream
commit and packages write the same bytes. Lucene writes a random segment id into the
header of every index file and a timestamp into each segment's ``.si`` file, so the index
files themselves differ on every build, even of the same documents with the same engine.
The record therefore describes what decides how the index searches: the sha256 and number
of the documents fed to pyserini, the indexer's module and arguments (with placeholders
for the two directories) and pyserini's version (which fixes Anserini and Lucene); and
what the index holds, read back with pyserini's ``IndexReader`` (its statistics). The
index files' own sha256 are written for diagnosis only (:data:`INDEX_FILES_RECORD`),
and nothing compares them.

Standard library only (the index step imports upstream's engine and pyserini, which the
image holds, inside its functions); Python 3.9 compatible. ``server.py`` imports
:func:`file_sha256`, :data:`DATA_PINS`, :data:`DATA_DIR` and :data:`INDEX_RECORD` from
here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Upstream's checkout (``git clone`` in the image).
WEBSHOP_DIR = Path(os.environ.get("WEBSHOP_DIR", "/opt/webshop"))
#: The product files. Outside upstream's checkout: they are fetched before it is cloned
#: (and before any package is installed), so no later layer's change fetches them again.
DATA_DIR = Path(os.environ.get("WEBSHOP_DATA_DIR", "/opt/webshop-data"))
#: The pins of the product files (``protocol.WEBSHOP_DATA``).
DATA_PINS = Path(__file__).with_name("data.json")
#: Where upstream's engine looks for the index (``engine.py:206``: ``search_engine/
#: indexes`` next to its package), and the index record (:func:`index_record`).
INDEX_DIR = WEBSHOP_DIR / "search_engine" / "indexes"
INDEX_RECORD = WEBSHOP_DIR / "search_engine" / "index.sha256.json"
#: The sha256 of every index file, for diagnosis only: they differ on every build.
INDEX_FILES_RECORD = INDEX_RECORD.with_name("index.files.sha256.json")
#: The documents fed to pyserini, written next to the index and removed after it.
DOCUMENTS_FILE = "documents.jsonl"
#: Upstream's ``run_indexing.sh`` for the full index, with ``--threads 1``. ``{input}``
#: (the documents' directory) and ``{index}`` (the index directory) are filled in by
#: :func:`index_argv` and kept as they are in the index record.
INDEX_MODULE = "pyserini.index.lucene"
INDEX_ARGS = (
    "--collection",
    "JsonCollection",
    "--input",
    "{input}",
    "--index",
    "{index}",
    "--generator",
    "DefaultLuceneDocumentGenerator",
    "--threads",
    "1",
    "--storePositions",
    "--storeDocvectors",
    "--storeRaw",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_data(out_dir: Path, pins_path: Path = DATA_PINS) -> Dict[str, str]:
    """Download the pinned product files and verify their size and sha256; a file
    already there with the pinned hash is not fetched again."""
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{pins['hf_repo']}/resolve/{pins['hf_revision']}/"
    verified = {}
    for name, pin in sorted(pins["files"].items()):
        target = out_dir / name
        if not (target.is_file() and file_sha256(target) == pin["sha256"]):
            partial = target.with_suffix(target.suffix + ".part")
            # Two plain ``with`` blocks: a parenthesized one is Python 3.10 syntax.
            response = urllib.request.urlopen(base + name, timeout=600)  # noqa: S310
            try:
                with open(partial, "wb") as handle:
                    for block in iter(lambda: response.read(1 << 20), b""):
                        handle.write(block)
            finally:
                response.close()
            partial.replace(target)
        size, digest = target.stat().st_size, file_sha256(target)
        if size != pin["size"] or digest != pin["sha256"]:
            raise SystemExit(f"{name}: {size} bytes sha256 {digest}, pinned {pin}")
        verified[name] = digest
    return verified


def documents(products: List[Dict]) -> List[Dict]:
    """Upstream's ``convert_product_file_format.py``: one Lucene document per product."""
    out = []
    for p in products:
        option_texts = [
            f"{name}: {', '.join(contents)}"
            for name, contents in (p.get("options") or {}).items()
        ]
        out.append(
            {
                "id": p["asin"],
                "contents": " ".join(
                    [
                        p["Title"],
                        p["Description"],
                        p["BulletPoints"][0],
                        ", and ".join(option_texts),
                    ]
                ).lower(),
                "product": p,
            }
        )
    return out


def write_documents(docs: List[Dict], path: Path) -> Tuple[str, int]:
    """Write ``docs`` to ``path`` as JSON lines (``json.dumps`` with its defaults, as
    upstream's conversion does); returns the sha256 of the bytes written and the number
    of documents."""
    digest = hashlib.sha256()
    count = 0
    with open(path, "wb") as handle:
        for doc in docs:
            line = (json.dumps(doc) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            count += 1
    return digest.hexdigest(), count


def index_argv(resources: Path, index_dir: Path) -> List[str]:
    """The indexer's command line: :data:`INDEX_ARGS` with ``{input}`` and ``{index}``
    filled in."""
    paths = {"{input}": str(resources), "{index}": str(index_dir)}
    return [sys.executable, "-m", INDEX_MODULE, *(paths.get(a, a) for a in INDEX_ARGS)]


def pyserini_version() -> str:
    from importlib import metadata

    return metadata.version("pyserini")


def index_stats(index_dir: Path) -> Dict[str, int]:
    """The index's statistics as pyserini's ``IndexReader`` reads them (documents,
    non-empty documents, unique terms, total terms)."""
    from pyserini.index.lucene import IndexReader  # in the image

    stats = IndexReader(str(index_dir)).stats() or {}
    return {str(key): int(value) for key, value in stats.items()}


def index_record(
    documents_sha256: str,
    documents_count: int,
    stats: Dict[str, int],
    pyserini: str,
) -> Dict[str, Any]:
    """The index record: the documents fed to the indexer (their sha256 and number), the
    indexer (module, arguments without paths, pyserini's version) and the index's
    statistics. Nothing in it depends on the machine, the paths or the time of the
    build."""
    return {
        "documents": {"sha256": documents_sha256, "count": documents_count},
        "indexer": {
            "module": INDEX_MODULE,
            "args": list(INDEX_ARGS),
            "pyserini": pyserini,
        },
        "index": dict(sorted(stats.items())),
    }


def record_text(record: Dict[str, Any]) -> str:
    """The bytes written for a record (whose sha256 the build record keeps)."""
    return json.dumps(record, indent=1, sort_keys=True)


def index_hashes(index_dir: Path) -> Dict[str, str]:
    """The sha256 of every index file, by its path under ``index_dir`` (diagnosis)."""
    return {
        path.relative_to(index_dir).as_posix(): file_sha256(path)
        for path in sorted(index_dir.rglob("*"))
        if path.is_file()
    }


def build_index(data_dir: Path, index_dir: Path = INDEX_DIR) -> Dict[str, Any]:
    """Upstream's ``convert_product_file_format.py`` and ``run_indexing.sh`` for the full
    product set only, with ``--threads 1``, into ``index_dir``; returns the index record
    (:func:`index_record`). The documents are written to ``resources`` next to
    ``index_dir`` and removed once the index is built."""
    sys.path.insert(0, str(WEBSHOP_DIR))
    from web_agent_site.engine import engine  # upstream, in the image

    engine.DEFAULT_ATTR_PATH = str(data_dir / "items_ins_v2.json")
    engine.HUMAN_ATTR_PATH = str(data_dir / "items_human_ins.json")
    products, *_ = engine.load_products(filepath=str(data_dir / "items_shuffle.json"))
    resources = index_dir.parent / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    jsonl = resources / DOCUMENTS_FILE
    documents_sha256, count = write_documents(documents(products), jsonl)
    subprocess.run(index_argv(resources, index_dir), check=True)
    jsonl.unlink()
    stats = index_stats(index_dir)
    indexed = stats.get("documents", 0)
    # Anserini skips a document it cannot index (and says so in its log), so the index
    # may hold fewer documents than were written, never more and never none.
    if not 0 < indexed <= count:
        raise SystemExit(f"the index holds {indexed} documents of the {count} written")
    return index_record(documents_sha256, count, stats, pyserini_version())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-data", help="download and verify the product files")
    fetch.add_argument("--out", type=Path, default=DATA_DIR)
    index = sub.add_parser("build-index", help="build the Lucene index and its record")
    index.add_argument("--data-dir", type=Path, default=DATA_DIR)
    index.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
        help="where to build the index (another directory: a rebuild to compare)",
    )
    index.add_argument(
        "--record",
        type=Path,
        default=INDEX_RECORD,
        help="the index record; the index files' sha256 are written next to it",
    )
    args = parser.parse_args(argv)
    if args.command == "fetch-data":
        print(json.dumps(fetch_data(args.out), indent=1))
        return 0
    record = build_index(args.data_dir, args.index_dir)
    args.record.write_text(record_text(record))
    files = args.record.with_name(INDEX_FILES_RECORD.name)
    files.write_text(record_text(index_hashes(args.index_dir)))
    print(f"index record {args.record}: {record_text(record)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
