#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASES = [
    ("study1/data/state.sqlite3", 33_000, 32_992, 32_994),
    ("study2/data/live/state.sqlite3", 1_800, 1_776, 1_776),
]
SECRET_PATTERNS = [
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads((ROOT / "checksums.json").read_text())
errors = []

for relative_path, expected_hash in manifest["files"].items():
    path = ROOT / relative_path
    if not path.is_file():
        errors.append(f"missing: {relative_path}")
    elif sha256(path) != expected_hash:
        errors.append(f"hash mismatch: {relative_path}")

subprocess.run(
    [sys.executable, str(ROOT / "scripts/unpack_data.py")],
    check=True,
)

for (
    relative_path,
    expected_slots,
    expected_observations,
    expected_attempts,
) in DATABASES:
    path = ROOT / relative_path
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)

    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        errors.append(f"SQLite integrity: {relative_path}")

    counts = (
        connection.execute("SELECT COUNT(*) FROM slots").fetchone()[0],
        connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
        connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
    )
    expected_counts = (
        expected_slots,
        expected_observations,
        expected_attempts,
    )
    if counts != expected_counts:
        errors.append(f"collection counts {relative_path}: {counts}")

    duplicate_count = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT scientific_id
            FROM observations
            GROUP BY scientific_id
            HAVING COUNT(*) > 1
        )
        """).fetchone()[0]
    if duplicate_count:
        errors.append(f"duplicate observations: {relative_path}")

    connection.close()

for path in ROOT.rglob("*"):
    if (
        not path.is_file()
        or ".git" in path.parts
        or path.suffix in {".pdf", ".xz", ".sqlite3"}
    ):
        continue
    data = path.read_bytes()
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        errors.append(f"credential-like content: {path.relative_to(ROOT)}")

if errors:
    print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
    raise SystemExit(1)

print(
    json.dumps(
        {
            "status": "PASS",
            "files_verified": len(manifest["files"]),
            "study1": {
                "slots": 33_000,
                "observations": 32_992,
                "attempts": 32_994,
            },
            "study2": {
                "slots": 1_800,
                "observations": 1_776,
                "attempts": 1_776,
            },
        },
        indent=2,
    )
)
