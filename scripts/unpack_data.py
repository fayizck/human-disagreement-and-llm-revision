#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import lzma
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    (
        ROOT / "study1/data/state.sqlite3.xz",
        ROOT / "study1/data/state.sqlite3",
        "83fdd07ad3eb555a56b39fe96de719c793f4f349b70a947f913f0b1328119cf0",
        "58d86fad54abc6383b9804905fa07b79daf5c84c6ca4fb16fb9aa5b06449436a",
    ),
    (
        ROOT / "study2/data/state.sqlite3.xz",
        ROOT / "study2/data/live/state.sqlite3",
        "9fc7eb54d135cae73ae4e6347457499ad4b34dcb842c2e0cd9f2a690ce66a459",
        "75227918d5ef76116abe939df7158e05a0904250a25fe59466ae3c1630b1a5c4",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for archive, target, archive_hash, database_hash in DATASETS:
    if sha256(archive) != archive_hash:
        raise SystemExit(f"Archive hash mismatch: {archive}")

    if target.exists() and sha256(target) == database_hash:
        print(f"OK {target.relative_to(ROOT)}")
        continue

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".unpacking")
    with lzma.open(archive, "rb") as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, 1024 * 1024)

    if sha256(temporary) != database_hash:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Database hash mismatch after unpacking: {target}")

    temporary.replace(target)
    print(f"UNPACKED {target.relative_to(ROOT)}")
