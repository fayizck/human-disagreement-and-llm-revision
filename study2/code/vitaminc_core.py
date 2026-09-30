#!/usr/bin/env python3
"""Shared functions for the VitaminC study. Importing this module performs no I/O."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
FREEZE = ROOT / "freeze"
LIVE_DB = ROOT / "data/live/state.sqlite3"
LABELS = ("SUPPORTS", "REFUTES")


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha_bytes(v):
    return hashlib.sha256(v).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def demand(ok, message):
    if not ok:
        raise RuntimeError(message)


def parse_label(text):
    if not isinstance(text, str):
        return {"status": "invalid", "label": None, "reason": "missing_text"}
    normalized = text.strip().casefold().upper()
    if normalized in LABELS:
        return {"status": "valid", "label": normalized, "reason": None}
    return {
        "status": "invalid",
        "label": None,
        "reason": "empty" if not text.strip() else "noncanonical_label",
    }


def load_freeze():
    out = {
        "sample": read_json(FREEZE / "sample_manifest.json"),
        "prompts": read_json(FREEZE / "prompts.json"),
        "models": read_json(FREEZE / "model_configs.json"),
        "parser": read_json(FREEZE / "parser_spec.json"),
        "analysis": read_json(FREEZE / "analysis_spec.json"),
        "transport": read_json(FREEZE / "transport_policy.json"),
    }
    if (FREEZE / "MASTER_FREEZE_MANIFEST.json").is_file():
        out["master"] = read_json(FREEZE / "MASTER_FREEZE_MANIFEST.json")
    return out


def verify_master(expected_sha=None):
    master_path = FREEZE / "MASTER_FREEZE_MANIFEST.json"
    actual = sha_file(master_path)
    if expected_sha is not None:
        demand(actual == expected_sha, "Master freeze hash mismatch")
    master = read_json(master_path)
    for rel, expected in master["artifact_sha256"].items():
        path = ROOT / rel
        demand(path.is_file(), f"Frozen artifact missing: {rel}")
        demand(sha_file(path) == expected, f"Frozen artifact changed: {rel}")
    return actual, master


def item_map(freeze):
    return {x["item_id"]: x for x in freeze["sample"]["items"]}


def model_map(freeze):
    return {x["key"]: x for x in freeze["models"]["models"]}


def initial_request(item):
    p = read_json(FREEZE / "prompts.json")
    return {
        "messages": [
            {"role": "system", "content": p["system_message"]},
            {
                "role": "user",
                "content": p["initial_user_template"].format(
                    evidence=item["evidence"], claim=item["claim"]
                ),
            },
        ]
    }


def branch_request(item, initial_text, condition, target=None):
    p = read_json(FREEZE / "prompts.json")
    follow = (
        p["neutral_followup"]
        if condition == "neutral"
        else p["directed_followup_template"].format(alternative_label=target)
    )
    base = initial_request(item)["messages"]
    return {
        "messages": base
        + [
            {"role": "assistant", "content": initial_text},
            {"role": "user", "content": follow},
        ]
    }


def payload(provider, model, request):
    messages = request["messages"]
    if provider == "openai":
        return {
            "model": model["model_id"],
            "instructions": messages[0]["content"],
            "input": messages[1:],
            "tools": [],
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "temperature": 1.0,
            "top_p": 1.0,
            "reasoning": {"effort": "none"},
            "max_output_tokens": 16,
        }
    return {
        "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
        "contents": [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages[1:]
        ],
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": "minimal"},
            "maxOutputTokens": 1024,
        },
    }


def decode(provider, raw):
    if provider == "openai":
        parts = [
            c
            for o in raw.get("output", [])
            if o.get("type") == "message"
            for c in o.get("content", [])
        ]
        text = "".join(
            x.get("text", "") for x in parts if x.get("type") == "output_text"
        )
        usage = raw.get("usage", {})
        return (
            text,
            raw.get("model"),
            raw.get("id"),
            {
                "input": int(usage.get("input_tokens") or 0),
                "output": int(usage.get("output_tokens") or 0),
                "reasoning": int(
                    (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
                    or 0
                ),
                "cached": int(
                    (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
                ),
                "total": int(usage.get("total_tokens") or 0),
            },
        )
    candidates = raw.get("candidates") or []
    demand(
        len(candidates) == 1, "Gemini response did not contain exactly one candidate"
    )
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(x.get("text", "") for x in parts if not x.get("thought", False))
    u = raw.get("usageMetadata", {})
    return (
        text,
        raw.get("modelVersion"),
        raw.get("responseId"),
        {
            "input": int(u.get("promptTokenCount") or 0),
            "output": int(u.get("candidatesTokenCount") or 0),
            "reasoning": int(u.get("thoughtsTokenCount") or 0),
            "cached": int(u.get("cachedContentTokenCount") or 0),
            "total": int(u.get("totalTokenCount") or 0),
        },
    )


def cost(provider, usage, freeze):
    r = freeze["transport"]["rates_per_million"][provider]
    uncached = max(0, usage["input"] - usage["cached"])
    return (
        Decimal(uncached) * Decimal(r["input"])
        + Decimal(usage["cached"]) * Decimal(r["cached"])
        + Decimal(usage["output"] + usage["reasoning"]) * Decimal(r["output"])
    ) / Decimal(1_000_000)


def validate_db(con, require_empty=False):
    con.row_factory = sqlite3.Row
    demand(
        con.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
        "SQLite integrity check failed",
    )
    slots = con.execute(
        "SELECT COUNT(*),COUNT(DISTINCT scientific_id),COUNT(DISTINCT plan_index) FROM slots"
    ).fetchone()
    demand(tuple(slots) == (1800, 1800, 1800), "Scientific slot catalog mismatch")
    obs = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    attempts = con.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    if require_empty:
        demand(obs == 0 and attempts == 0, "Live database is not empty")
    return {"slots": 1800, "observations": obs, "attempts": attempts}


def redact_error(exc):
    text = type(exc).__name__
    if isinstance(exc, TimeoutError):
        text = "TimeoutError"
    return re.sub(
        r"(?:sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{12,})", "[REDACTED]", text
    )
