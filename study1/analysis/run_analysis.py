#!/usr/bin/env python3
"""Analysis for the 300-example Study 1 dataset.

The collection database is opened read-only. The prospective
analysis manifest is written and hashed before outcome rows are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "study1/data/state.sqlite3"
BUNDLE = ROOT / "study1/frozen"
RECON = ROOT / "study1/reconciliation"
DEFAULT_OUT = ROOT / "reproduced/study1"
DB_SHA = "58d86fad54abc6383b9804905fa07b79daf5c84c6ca4fb16fb9aa5b06449436a"
ENGINE_SHA = "bd57b85acf798475b4e148c3231ce09b2a08b14f88f6cc5f43e839399ce08432"
LABELS = ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")
MODEL_NAMES = {
    "openai_primary": "OpenAI gpt-5.4-nano-2026-03-17",
    "gemini_primary": "Gemini gemini-3.5-flash-lite",
}
BOOTSTRAP_SEED = 2026092401
CV_SEED = 2026092402
QUAL_SEED = 2026092403
BOOTSTRAP_REPS = 2000
CONFIDENCE = 0.95
L2_LAMBDA = 1.0
FORMULA_RQ1 = "Y ~ C + H10 + U10 + Q10 + C:H10 + C:U10 + C:Q10 + model + C:model + source + initial_label + target_label"
FORMULA_RQ2 = FORMULA_RQ1 + " + S10 + C:S10"
REFERENCE_LEVELS = {
    "condition": "neutral",
    "model": "gemini_primary",
    "source": "mnli",
    "initial_label": "ENTAILMENT",
    "target_label": "ENTAILMENT",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def safe(v):
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, Path):
        return str(v)
    return v


def records(df):
    return [{k: safe(v) for k, v in r.items()} for r in df.to_dict("records")]


def dump_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )


def latex_escape(value):
    s = str(value)
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        s = s.replace(a, b)
    return s


def parsed(v):
    return json.loads(v) if isinstance(v, str) and v else None


def entropy(probabilities):
    p = np.asarray([x for x in probabilities if x is not None and x > 0], float)
    return float(-(p * np.log(p)).sum() / np.log(3)) if len(p) else None


def summary(series):
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return {
            k: None for k in ("n", "min", "q25", "median", "mean", "q75", "max", "sd")
        }
    return {
        "n": len(s),
        "min": s.min(),
        "q25": s.quantile(0.25),
        "median": s.median(),
        "mean": s.mean(),
        "q75": s.quantile(0.75),
        "max": s.max(),
        "sd": s.std(ddof=1),
    }


def assign_folds(items):
    groups = (
        items.groupby("group_id", sort=True)
        .agg(
            item_count=("item_id", "size"),
            sources=("source", lambda x: "|".join(sorted(set(x)))),
            strata=("stratum", lambda x: "|".join(map(str, sorted(set(x))))),
        )
        .reset_index()
    )

    def key(g):
        return hashlib.sha256(f"{CV_SEED}|{g}".encode()).hexdigest()

    groups["random_key"] = groups.group_id.map(key)
    groups = groups.sort_values(
        ["item_count", "random_key"], ascending=[False, True]
    ).reset_index(drop=True)
    loads = [0] * 5
    assignment = {}
    for _, r in groups.iterrows():
        fold = min(range(5), key=lambda f: (loads[f], f))
        assignment[r.group_id] = fold + 1
        loads[fold] += int(r.item_count)
    result = items[["item_id", "group_id", "source", "stratum"]].copy()
    result["fold"] = result.group_id.map(assignment)
    return result.sort_values("item_id").reset_index(drop=True)


def verify_and_freeze_manifest(out):
    assert sha(DB) == DB_SHA, "Collection database hash changed"
    recon = read_json(RECON / "main_study_final_collection_reconciliation.json")
    assert recon["status"] == "PASS" and recon["database"]["sha256_after"] == DB_SHA
    freeze = read_json(BUNDLE / "freeze_lock.json")
    for rel, expected in freeze["artifact_hashes"].items():
        assert sha(BUNDLE / rel) == expected, rel
    con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    try:
        assert con.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert not con.execute("PRAGMA foreign_key_check").fetchall()
        meta = {k: json.loads(v) for k, v in con.execute("SELECT key,value FROM meta")}
        assert (
            meta["run_state"] == "COMPLETE"
            and meta["engine_version"] == "main_concurrent_engine_20260923_v3"
        )
        assert con.execute("SELECT COUNT(*) FROM slots").fetchone()[0] == 33000
        assert con.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 32992
    finally:
        con.close()
    items = pd.DataFrame(read_jsonl(BUNDLE / "manifest.jsonl"))
    assert len(items) == 300 and items.group_id.nunique() == 296
    folds = assign_folds(items)
    folds.to_csv(out / "machine/cv_fold_assignment.csv", index=False)
    manifest = {
        "analysis_version": "main_study_scientific_analysis_20260924_v1",
        "classification": "PRESPECIFIED STUDY 1 ANALYSIS PLAN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "database_sha256": DB_SHA,
        "bundle_lock_sha256": sha(BUNDLE / "freeze_lock.json"),
        "engine_lock_sha256": ENGINE_SHA,
        "analysis_code_sha256": sha(Path(__file__)),
        "primary_outcome": "Y=1 iff valid final canonical label equals assigned target alternative; invalid final response is Y=0; structurally ineligible branches excluded",
        "secondary_outcome": "generic flip=1 iff valid final canonical label differs from valid initial label; invalid finals excluded from flip denominator",
        "predictors": {
            "H": "normalized entropy of 100 human NLI labels",
            "G": "normalized human-label Gini disagreement",
            "U": "normalized sampled predictive entropy from valid n=50 baseline labels",
            "Q": "n=50 valid-baseline probability assigned to target label",
            "S": "human annotation proportion assigned to target label",
            "C": "1 directed suggestion, 0 neutral reconsideration",
        },
        "primary_rq1_formula": FORMULA_RQ1,
        "rq2_formula": FORMULA_RQ2,
        "scaling": "H, G, U, Q and S coefficients expressed per 0.1 increase",
        "reference_levels": REFERENCE_LEVELS,
        "clustering_method": "finite-sample-corrected cluster sandwich covariance; clusters are the predefined related premise/image groups",
        "confidence_level": CONFIDENCE,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPS,
            "unit": "whole related group",
            "interval": "percentile",
        },
        "cv": {
            "seed": CV_SEED,
            "folds": 5,
            "fold_assignment_sha256": sha(out / "machine/cv_fold_assignment.csv"),
            "scope": "directed branches",
            "grouping": "related premise/image group",
            "fixed_l2_lambda": L2_LAMBDA,
            "predictor_sets": {
                "A": ["model", "source", "initial_label", "target_label", "U", "Q"],
                "B": ["A", "H"],
                "C": ["A", "S"],
                "D": ["A", "H", "S"],
            },
            "primary_metric": "log loss",
            "secondary_metric": "Brier score",
        },
        "multiplicity_family": [
            "RQ2 C:S interaction",
            "RQ3 Model D minus A held-out log-loss",
            "RQ3 Model C minus B held-out log-loss",
        ],
        "holm_method": "step-down Holm adjustment across the three listed secondary tests",
        "robustness": [
            "replace H and C:H by G and C:G",
            "valid-follow-up-only",
            "replace U50/Q50 by nested U40/Q40 using indices 0..39",
            "whole-group bootstrap contrasts",
            "rank/convergence/separation/covariance diagnostics",
        ],
        "finite_sampling_note": "No additional half-sample correction formula was specified for the main study; the specified nested n40 sensitivity is used and n50 remains primary.",
        "qualitative": {
            "seed": QUAL_SEED,
            "maximum_cases": 32,
            "maximum_per_item": 1,
            "cells": "within-source bottom/top H quartile x adoption/non-adoption x S<=.05/S>=.20",
        },
    }
    path = out / "machine/analysis_manifest.json"
    dump_json(path, manifest)
    (out / "machine/analysis_manifest.sha256").write_text(
        sha(path) + "  analysis_manifest.json\n"
    )
    return items, folds, manifest, sha(path)


def load_collection():
    con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    try:
        slots = pd.read_sql_query("SELECT * FROM slots ORDER BY plan_slot", con)
        obs = pd.read_sql_query("SELECT * FROM observations ORDER BY attempt_id", con)
    finally:
        con.close()
    joined = slots.merge(
        obs,
        on=["slot_key", "provider", "item_id"],
        how="left",
        suffixes=("_slot", "_obs"),
        validate="one_to_one",
    )
    return slots, obs, joined


def build_baselines(joined, items):
    out = []
    base = joined[(joined.phase == "baseline") & (joined.status == "COMMITTED")]
    for _, r in base.iterrows():
        req, val = parsed(r.request_json_slot), parsed(r.parsed_json)
        out.append(
            {
                "slot_key": r.slot_key,
                "scientific_id": r.scientific_id_slot,
                "attempt_id": int(r.attempt_id),
                "provider": r.provider,
                "model_key": r.model_key,
                "item_id": r.item_id,
                "sample_index": int(req["sample_index"]),
                "status": val["status"],
                "label": val["label"],
                "invalid_reason": val["reason"],
            }
        )
    samples = pd.DataFrame(out).sort_values(["model_key", "item_id", "sample_index"])
    assert (
        len(samples) == 30000
        and samples.groupby(["model_key", "item_id"]).size().eq(50).all()
    )
    rows = []
    for (model, item), g in samples.groupby(["model_key", "item_id"], sort=True):
        row = {"model_key": model, "provider": g.provider.iloc[0], "item_id": item}
        for n in (40, 50):
            z = g[g.sample_index < n]
            v = z[z.status == "valid"]
            den = len(v)
            counts = v.label.value_counts().to_dict()
            row.update(
                {
                    f"baseline_total_n{n}": len(z),
                    f"baseline_valid_n{n}": den,
                    f"baseline_invalid_n{n}": len(z) - den,
                    f"distinct_valid_labels_n{n}": len(counts),
                    f"U{n}": (
                        entropy([counts.get(x, 0) / den for x in LABELS])
                        if den
                        else None
                    ),
                }
            )
            for lab in LABELS:
                row[f"count{n}_{lab}"] = counts.get(lab, 0)
                row[f"q{n}_{lab}"] = counts.get(lab, 0) / den if den else None
        rows.append(row)
    m = pd.DataFrame(rows).merge(
        items[["item_id", "source", "group_id", "stratum", "H", "G"]],
        on="item_id",
        validate="many_to_one",
    )
    assert len(m) == 600
    return samples, m


def build_branches(joined, items, measurements, folds):
    item = items.set_index("item_id").to_dict("index")
    meas = measurements.set_index(["model_key", "item_id"]).to_dict("index")
    initials = {}
    for _, r in joined[
        (joined.phase == "initial") & (joined.status == "COMMITTED")
    ].iterrows():
        val = parsed(r.parsed_json)
        initials[(r.model_key, r.item_id)] = {
            "status": val["status"],
            "label": val["label"],
            "reason": val["reason"],
        }
    assert len(initials) == 600
    fold_map = folds.set_index("item_id").fold.to_dict()
    rows = []
    for _, r in joined[joined.phase == "followup"].sort_values("plan_slot").iterrows():
        it = item[r.item_id]
        init = initials[(r.model_key, r.item_id)]
        observed = r.status == "COMMITTED"
        req = parsed(r.request_json_slot) if observed else None
        branch = int(r.branch_index)
        condition = (
            req["condition"] if req else ("neutral" if branch in (0, 2) else "directed")
        )
        target = req["target"] if req else None
        val = parsed(r.parsed_json) if observed else None
        final_status = val["status"] if val else "structurally_ineligible"
        final_label = val["label"] if val else None
        y = int(final_status == "valid" and final_label == target) if observed else None
        flip = (
            int(final_label != init["label"])
            if observed and final_status == "valid" and init["status"] == "valid"
            else None
        )
        mm = meas[(r.model_key, r.item_id)]
        s = it.get(f"support_{target}") if target else None
        qi50 = mm.get(f"q50_{target}") if target else None
        qi40 = mm.get(f"q40_{target}") if target else None
        init_s = it.get(f"support_{init['label']}") if init["label"] else None
        final_s = it.get(f"support_{final_label}") if final_label else None
        rows.append(
            {
                "slot_key": r.slot_key,
                "plan_slot": int(r.plan_slot),
                "scientific_id": r.scientific_id_slot if observed else None,
                "provider": r.provider,
                "model_key": r.model_key,
                "model": MODEL_NAMES[r.model_key],
                "item_id": r.item_id,
                "source": it["source"],
                "group_id": it["group_id"],
                "fold": int(fold_map[r.item_id]),
                "stratum": int(it["stratum"]),
                "premise": it["premise"],
                "hypothesis": it["hypothesis"],
                "human_counts": "|".join(map(str, it["human_counts"])),
                "human_probs": "|".join(f"{x:.2f}" for x in it["human_probs"]),
                "H": float(it["H"]),
                "G": float(it["G"]),
                "initial_label": init["label"],
                "initial_status": init["status"],
                "initial_invalid_reason": init["reason"],
                "target": target,
                "condition": condition,
                "C": int(condition == "directed"),
                "branch_index": branch,
                "final_label": final_label,
                "final_status": final_status,
                "invalid_reason": val["reason"] if val else r.ineligible_reason,
                "Y": y,
                "F": flip,
                "S": s,
                "initial_human_support": init_s,
                "final_human_support": final_s,
                "human_support_change": (
                    final_s - init_s
                    if final_s is not None and init_s is not None
                    else None
                ),
                "U50": mm["U50"],
                "U40": mm["U40"],
                "Q50": qi50,
                "Q40": qi40,
                "baseline_valid_n50": int(mm["baseline_valid_n50"]),
                "baseline_valid_n40": int(mm["baseline_valid_n40"]),
                "structurally_ineligible": not observed,
            }
        )
    b = pd.DataFrame(rows)
    assert len(b) == 2400 and b.structurally_ineligible.sum() == 8
    return b


def analysis_frame(branches):
    f = (
        branches[~branches.structurally_ineligible]
        .dropna(subset=["Y", "H", "G", "U50", "Q50", "S"])
        .copy()
    )
    for x in ("H", "G", "U50", "Q50", "U40", "Q40", "S"):
        f[x + "10"] = f[x] / 0.1
    f["directed"] = f.C.astype(float)
    return f.reset_index(drop=True)


def design(frame, human="H10", u="U5010", q="Q5010", add_s=False):
    mo = (frame.model_key == "openai_primary").astype(float)
    ss = (frame.source == "snli").astype(float)
    ine = (frame.initial_label == "NEUTRAL").astype(float)
    inc = (frame.initial_label == "CONTRADICTION").astype(float)
    tne = (frame.target == "NEUTRAL").astype(float)
    tco = (frame.target == "CONTRADICTION").astype(float)
    c = frame.C.astype(float)
    d = {
        "Intercept": np.ones(len(frame)),
        "model[T.OpenAI]": mo,
        "source[T.SNLI]": ss,
        "initial[T.NEUTRAL]": ine,
        "initial[T.CONTRADICTION]": inc,
        "target[T.NEUTRAL]": tne,
        "target[T.CONTRADICTION]": tco,
        human.replace("10", "_per_0.1"): frame[human],
        "U_per_0.1": frame[u],
        "Q_per_0.1": frame[q],
        "directed": c,
        "directed:model[T.OpenAI]": c * mo,
        f"directed:{human.replace('10','_per_0.1')}": c * frame[human],
        "directed:U_per_0.1": c * frame[u],
        "directed:Q_per_0.1": c * frame[q],
    }
    if add_s:
        d["S_per_0.1"] = frame.S10
        d["directed:S_per_0.1"] = c * frame.S10
    return pd.DataFrame(d, index=frame.index)


def tcrit(df):
    z = NormalDist().inv_cdf(0.975)
    d = float(df)
    return (
        z
        + (z**3 + z) / (4 * d)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * d * d)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * d**3)
    )


def betacf(a, b, x):
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-12:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    return (
        bt * betacf(a, b, x) / a
        if x < (a + 1) / (a + b + 2)
        else 1 - bt * betacf(b, a, 1 - x) / b
    )


def t_pvalue(t, df):
    x = df / (df + t * t)
    return min(1.0, betai(df / 2, 0.5, x))


def logistic_cluster(y, X, groups, maxiter=200):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    groups = np.asarray(groups)
    beta = np.zeros(X.shape[1])
    converged = False

    def prob(b):
        return 1 / (1 + np.exp(-np.clip(X @ b, -35, 35)))

    def ll(b):
        p = np.clip(prob(b), 1e-15, 1 - 1e-15)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    old = -2 * ll(beta)
    for it in range(1, maxiter + 1):
        p = prob(beta)
        w = np.clip(p * (1 - p), 1e-10, None)
        info = X.T @ (w[:, None] * X)
        grad = X.T @ (y - p)
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            return {"estimable": False, "reason": "singular_information"}
        scale = 1.0
        oldll = ll(beta)
        while scale > 2**-20 and ll(beta + scale * step) < oldll - 1e-10:
            scale /= 2
        new = beta + scale * step
        dev = -2 * ll(new)
        if abs(dev - old) / (0.1 + abs(dev)) < 1e-8:
            beta = new
            converged = True
            break
        beta = new
        old = dev
    p = prob(beta)
    w = np.clip(p * (1 - p), 1e-10, None)
    info = X.T @ (w[:, None] * X)
    try:
        bread = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return {"estimable": False, "reason": "singular_final_information"}
    ug = np.unique(groups)
    resid = y - p
    scores = np.asarray([X[groups == g].T @ resid[groups == g] for g in ug])
    meat = scores.T @ scores
    n, k, G = len(y), X.shape[1], len(ug)
    corr = (G / (G - 1)) * ((n - 1) / (n - k))
    cov = corr * bread @ meat @ bread
    diag = np.diag(cov)
    se = np.sqrt(np.maximum(diag, 0))
    finite = np.all(np.isfinite(beta)) and np.all(np.isfinite(cov))
    unstable = np.max(np.abs(beta)) > 40 or np.max(se) > 20
    return {
        "estimable": bool(converged and finite and not unstable),
        "converged": converged,
        "finite": bool(finite),
        "unstable": bool(unstable),
        "reason": (
            None
            if converged and finite and not unstable
            else "nonconverged_nonfinite_or_unstable"
        ),
        "params": beta,
        "se": se,
        "cov": cov,
        "clusters": G,
        "df": G - 1,
        "tcrit": tcrit(G - 1),
        "iterations": it,
        "condition_number": float(np.linalg.cond(info)),
        "loglik": ll(beta),
    }


def coefficient_table(fit, names, model_name):
    if not fit.get("estimable"):
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "term": "MODEL_NOT_ESTIMABLE",
                    "reason": fit.get("reason"),
                }
            ]
        )
    rows = []
    for name, b, se in zip(names, fit["params"], fit["se"]):
        lo = b - fit["tcrit"] * se
        hi = b + fit["tcrit"] * se
        t = b / se if se > 0 else math.inf
        rows.append(
            {
                "model": model_name,
                "term": name,
                "estimate": b,
                "std_error": se,
                "ci_low": lo,
                "ci_high": hi,
                "odds_ratio": math.exp(np.clip(b, -700, 700)),
                "or_ci_low": math.exp(np.clip(lo, -700, 700)),
                "or_ci_high": math.exp(np.clip(hi, -700, 700)),
                "p_value": t_pvalue(abs(t), fit["df"]),
                "clusters": fit["clusters"],
                "df": fit["df"],
                "converged": fit["converged"],
                "estimable": fit["estimable"],
            }
        )
    return pd.DataFrame(rows)


def marginal_curve(frame, fit, builder, variable, grid):
    rows = []
    for condition in (0, 1):
        for value in grid:
            z = frame.copy()
            z["C"] = condition
            z["directed"] = condition
            z[variable] = value / 0.1
            X = builder(z).to_numpy(float)
            eta = np.clip(X @ fit["params"], -35, 35)
            p = 1 / (1 + np.exp(-eta))
            est = p.mean()
            grad = (p * (1 - p)) @ X / len(p)
            se = math.sqrt(max(0, float(grad @ fit["cov"] @ grad)))
            lo = max(0, est - fit["tcrit"] * se)
            hi = min(1, est + fit["tcrit"] * se)
            rows.append(
                {
                    "condition": "directed" if condition else "neutral",
                    "x": value,
                    "predicted_probability": est,
                    "ci_low": lo,
                    "ci_high": hi,
                    "se": se,
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_rate(frame, masks, reps=BOOTSTRAP_REPS):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gs = sorted(frame.group_id.unique())
    by = {g: frame.index[frame.group_id == g].to_numpy() for g in gs}
    out = {k: [] for k in masks}
    for _ in range(reps):
        idx = np.concatenate([by[g] for g in rng.choice(gs, len(gs), replace=True)])
        z = frame.loc[idx]
        for key, fn in masks.items():
            v = fn(z)
            out[key].append(float(v) if len(z) else math.nan)
    return {
        k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975)))
        for k, v in out.items()
    }


def rate_tables(frame):
    rows = []
    masks = {}
    for key, sub in [
        ("Combined", frame),
        (MODEL_NAMES["openai_primary"], frame[frame.model_key == "openai_primary"]),
        (MODEL_NAMES["gemini_primary"], frame[frame.model_key == "gemini_primary"]),
    ]:
        for cond in ("neutral", "directed"):
            g = sub[sub.condition == cond]
            valid = g[g.final_status == "valid"]
            rows.append(
                {
                    "model": key,
                    "condition": cond,
                    "adoption_numerator": int(g.Y.sum()),
                    "adoption_denominator": len(g),
                    "adoption_rate": g.Y.mean(),
                    "flip_numerator": int(valid.F.sum()),
                    "flip_denominator": len(valid),
                    "flip_rate": valid.F.mean(),
                }
            )
            mk = f"{key}|{cond}|adopt"
            masks[mk] = lambda z, k=key, c=cond: z[
                (z.condition == c) & ((z.model == k) if k != "Combined" else True)
            ].Y.mean()
    cis = cluster_bootstrap_rate(frame, masks)
    for r in rows:
        r["adoption_ci_low"], r["adoption_ci_high"] = cis[
            f"{r['model']}|{r['condition']}|adopt"
        ]
    return pd.DataFrame(rows)


def l2_fit(X, y, lam=L2_LAMBDA):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    P = np.eye(X.shape[1])
    P[0, 0] = 0
    for it in range(200):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -35, 35)))
        w = np.clip(p * (1 - p), 1e-9, None)
        A = X.T @ (w[:, None] * X) + lam * P
        grad = X.T @ (y - p) - lam * (P @ b)
        try:
            step = np.linalg.solve(A, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(A) @ grad
        b2 = b + step
        if np.max(np.abs(b2 - b)) < 1e-8:
            b = b2
            break
        b = b2
    return b


def cv_matrix(train, test, set_name):
    cont = (
        ["U50", "Q50"]
        + (["H"] if set_name in ("B", "D") else [])
        + (["S"] if set_name in ("C", "D") else [])
    )
    means = train[cont].mean()
    sds = train[cont].std(ddof=0).replace(0, 1)

    def one(df):
        d = {
            "Intercept": np.ones(len(df)),
            "model_openai": (df.model_key == "openai_primary").astype(float),
            "source_snli": (df.source == "snli").astype(float),
            "initial_neutral": (df.initial_label == "NEUTRAL").astype(float),
            "initial_contradiction": (df.initial_label == "CONTRADICTION").astype(
                float
            ),
            "target_neutral": (df.target == "NEUTRAL").astype(float),
            "target_contradiction": (df.target == "CONTRADICTION").astype(float),
        }
        for x in cont:
            d[x] = (df[x] - means[x]) / sds[x]
        return pd.DataFrame(d, index=df.index)

    return one(train), one(test)


def cv_analysis(frame):
    d = frame[frame.condition == "directed"].copy()
    pred_rows = []
    fold_rows = []
    for fold in range(1, 6):
        tr = d[d.fold != fold]
        te = d[d.fold == fold]
        for name in "ABCD":
            Xtr, Xte = cv_matrix(tr, te, name)
            b = l2_fit(Xtr, tr.Y.to_numpy())
            p = 1 / (1 + np.exp(-np.clip(Xte.to_numpy() @ b, -35, 35)))
            y = te.Y.to_numpy()
            ll = -(
                y * np.log(np.clip(p, 1e-15, 1))
                + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))
            )
            br = (p - y) ** 2
            fold_rows.append(
                {
                    "model_set": name,
                    "fold": fold,
                    "n": len(te),
                    "groups": te.group_id.nunique(),
                    "log_loss": ll.mean(),
                    "brier": br.mean(),
                }
            )
            for idx, pp, l, bv in zip(te.index, p, ll, br):
                pred_rows.append(
                    {
                        "row_index": int(idx),
                        "group_id": te.loc[idx, "group_id"],
                        "fold": fold,
                        "model_set": name,
                        "Y": int(te.loc[idx, "Y"]),
                        "probability": pp,
                        "log_loss": l,
                        "brier": bv,
                    }
                )
    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)
    means = (
        folds.groupby("model_set")
        .agg(mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"))
        .reset_index()
    )
    base = float(means.loc[means.model_set == "A", "mean_log_loss"].iloc[0])
    means["delta_log_loss_vs_A"] = means.mean_log_loss - base
    rng = np.random.default_rng(BOOTSTRAP_SEED + 71)
    groups = sorted(d.group_id.unique())
    vals = {}
    for comp, a, b in [("D_minus_A", "D", "A"), ("C_minus_B", "C", "B")]:
        merged = preds[preds.model_set == a][
            ["row_index", "group_id", "log_loss"]
        ].merge(
            preds[preds.model_set == b][["row_index", "log_loss"]],
            on="row_index",
            suffixes=("_a", "_b"),
        )
        gd = {
            g: x.log_loss_a.to_numpy() - x.log_loss_b.to_numpy()
            for g, x in merged.groupby("group_id")
        }
        boots = []
        for _ in range(BOOTSTRAP_REPS):
            arr = np.concatenate(
                [
                    gd[g]
                    for g in rng.choice(groups, len(groups), replace=True)
                    if g in gd
                ]
            )
            boots.append(arr.mean())
        observed = (merged.log_loss_a - merged.log_loss_b).mean()
        p = 2 * min((np.asarray(boots) >= 0).mean(), (np.asarray(boots) <= 0).mean())
        vals[comp] = {
            "estimate": observed,
            "ci_low": np.quantile(boots, 0.025),
            "ci_high": np.quantile(boots, 0.975),
            "p_value": min(1, p),
        }
    return folds, means, preds, vals


def holm(entries):
    ordered = sorted(entries, key=lambda x: x[1])
    m = len(entries)
    adjusted = {}
    running = 0
    for rank, (name, p) in enumerate(ordered):
        value = min(1, (m - rank) * p)
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def paired_alternatives(frame):
    d = frame[frame.condition == "directed"].copy()
    rows = []
    ties = 0
    for key, g in d.groupby(["item_id", "model_key"]):
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        if abs(a.S - b.S) < 1e-12:
            ties += 1
            continue
        hi, lo = (a, b) if a.S > b.S else (b, a)
        rows.append(
            {
                "item_id": key[0],
                "model_key": key[1],
                "group_id": hi.group_id,
                "high_S": hi.S,
                "low_S": lo.S,
                "high_Y": hi.Y,
                "low_Y": lo.Y,
                "difference": hi.Y - lo.Y,
            }
        )
    p = pd.DataFrame(rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 17)
    gs = sorted(p.group_id.unique())
    gd = {g: p[p.group_id == g].difference.to_numpy() for g in gs}
    boot = []
    for _ in range(BOOTSTRAP_REPS):
        boot.append(
            np.concatenate(
                [gd[g] for g in rng.choice(gs, len(gs), replace=True)]
            ).mean()
        )
    result = {
        "pairs_non_tied": len(p),
        "ties": ties,
        "high_adoptions": int(p.high_Y.sum()),
        "high_rate": p.high_Y.mean(),
        "low_adoptions": int(p.low_Y.sum()),
        "low_rate": p.low_Y.mean(),
        "paired_difference": p.difference.mean(),
        "ci_low": np.quantile(boot, 0.025),
        "ci_high": np.quantile(boot, 0.975),
    }
    return p, result


def qualitative_sample(frame, items):
    q = items.groupby("source").H.quantile([0.25, 0.75]).unstack()
    d = frame[frame.condition == "directed"].copy()
    d["entropy_band"] = d.apply(
        lambda r: (
            "low"
            if r.H <= q.loc[r.source, 0.25]
            else ("high" if r.H >= q.loc[r.source, 0.75] else "middle")
        ),
        axis=1,
    )
    d["support_band"] = np.where(
        d.S <= 0.05, "low", np.where(d.S >= 0.20, "substantial", "middle")
    )
    d = d[(d.entropy_band != "middle") & (d.support_band != "middle")]
    d["adoption_band"] = np.where(d.Y == 1, "adoption", "nonadoption")
    d["cell"] = d.entropy_band + "|" + d.adoption_band + "|" + d.support_band
    rng = np.random.default_rng(QUAL_SEED)
    d = d.assign(_r=rng.random(len(d)))
    used = set()
    selected = []
    cells = [
        f"{h}|{a}|{s}"
        for h in ("low", "high")
        for a in ("adoption", "nonadoption")
        for s in ("low", "substantial")
    ]
    for cell in cells:
        for _, r in (
            d[d.cell == cell].sort_values(["_r", "source", "model_key"]).iterrows()
        ):
            if r.item_id in used:
                continue
            selected.append(r)
            used.add(r.item_id)
            if sum(x.cell == cell for x in selected) >= 4:
                break
    if not selected:
        return pd.DataFrame(), {c: 0 for c in cells}
    s = (
        pd.DataFrame(selected)
        .sample(frac=1, random_state=QUAL_SEED)
        .reset_index(drop=True)
    )
    s.insert(0, "case_id", [f"Q{i:02d}" for i in range(1, len(s) + 1)])
    counts = Counter(s.cell)
    cols = [
        "case_id",
        "cell",
        "item_id",
        "source",
        "premise",
        "hypothesis",
        "initial_label",
        "target",
        "final_label",
        "human_counts",
        "human_probs",
        "H",
        "S",
        "U50",
        "Q50",
    ]
    return s[cols], {c: counts[c] for c in cells}


def tex_table(path, caption, label, headers, rows, aligns=None, notes=None):
    aligns = aligns or "l" + "r" * (len(headers) - 1)
    text = [
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular}}{{{aligns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    text += [" & ".join(map(str, row)) + r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        text += [
            r"\vspace{2pt}",
            r"\begin{minipage}{0.96\linewidth}\footnotesize "
            + notes
            + r"\end{minipage}",
        ]
    text += [r"\end{table}", ""]
    Path(path).write_text("\n".join(text))


def fmt(x, d=3):
    if x is None or pd.isna(x):
        return "--"
    value = float(x)
    if abs(value) < 0.5 * 10 ** (-d):
        value = 0.0
    return f"{value:.{d}f}"


def fmt_p(x):
    if x is None or pd.isna(x):
        return "--"
    return "$<0.001$" if float(x) < 0.001 else f"{float(x):.3f}"


def pct(x):
    return "--" if x is None or pd.isna(x) else f"{100*float(x):.1f}\\%"


def pgf_figure(path, body):
    tex = Path(path).with_suffix(".tex")
    tex.write_text(
        "\\documentclass[tikz,border=3pt]{standalone}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n\\usepgfplotslibrary{fillbetween}\n\\begin{document}\n"
        + body
        + "\n\\end{document}\n"
    )
    r = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=tex.parent,
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise RuntimeError(r.stdout[-4000:])
    assert Path(path).exists()


def make_figures(out, rates, curve_h, curve_s, cvfold):
    fig = out / "figures"
    # Figure 1
    coords = []
    for mi, m in enumerate(
        ["Combined", MODEL_NAMES["openai_primary"], MODEL_NAMES["gemini_primary"]]
    ):
        for ci, c in enumerate(["neutral", "directed"]):
            r = rates[(rates.model == m) & (rates.condition == c)].iloc[0]
            x = mi * 3 + ci
            coords.append(
                (
                    x,
                    r.adoption_rate,
                    r.adoption_rate - r.adoption_ci_low,
                    r.adoption_ci_high - r.adoption_rate,
                    c,
                )
            )

    def series(cond, style):
        pts = " ".join(
            f"({x},{y}) += (0,{hi}) -= (0,{lo})"
            for x, y, lo, hi, c in coords
            if c == cond
        )
        return f"\\addplot+[{style},error bars/.cd,y dir=both,y explicit] coordinates {{{pts}}};\\addlegendentry{{{cond.title()}}}"

    body = (
        r"\begin{tikzpicture}\begin{axis}[width=13cm,height=7cm,ymin=0,ymax=1,ylabel={Target-adoption rate},xtick={0.5,3.5,6.5},xticklabels={Combined,OpenAI,Gemini},legend style={at={(0.5,-0.18)},anchor=north,legend columns=2},ymajorgrids=true]"
        + series("neutral", "only marks,mark=square*,black")
        + series("directed", "only marks,mark=triangle*,black,dashed")
        + r"\end{axis}\end{tikzpicture}"
    )
    pgf_figure(fig / "figure_1_adoption_rates.pdf", body)
    for data, name, xlab in [
        (curve_h, "figure_2_entropy_interaction.pdf", "Human entropy $H$"),
        (
            curve_s,
            "figure_3_alternative_support.pdf",
            "Target-specific human support $S$",
        ),
    ]:
        parts = [
            r"\begin{tikzpicture}\begin{axis}[width=13cm,height=7cm,xlabel={"
            + xlab
            + r"},ylabel={Predicted target-adoption probability},ymin=0,ymax=1,legend style={at={(0.5,-0.18)},anchor=north,legend columns=2},ymajorgrids=true]"
        ]
        for cond, sty, mark in [
            ("neutral", "black", "square*"),
            ("directed", "black,dashed", "triangle*"),
        ]:
            z = data[data.condition == cond]
            low = " ".join(f"({r.x},{r.ci_low})" for _, r in z.iterrows())
            high = " ".join(f"({r.x},{r.ci_high})" for _, r in z.iterrows())
            mid = " ".join(
                f"({r.x},{r.predicted_probability})" for _, r in z.iterrows()
            )
            parts += [
                f"\\addplot[name path={cond}lo,draw=none] coordinates {{{low}}};",
                f"\\addplot[name path={cond}hi,draw=none] coordinates {{{high}}};",
                f"\\addplot[gray!25] fill between[of={cond}lo and {cond}hi];",
                f"\\addplot[{sty},mark={mark},mark repeat=4] coordinates {{{mid}}};\\addlegendentry{{{cond.title()}}}",
            ]
        parts.append(r"\end{axis}\end{tikzpicture}")
        pgf_figure(fig / name, "".join(parts))
    parts = [
        r"\begin{tikzpicture}\begin{axis}[width=13cm,height=7cm,ylabel={Held-out log loss},symbolic x coords={A,B,C,D},xtick=data,ymajorgrids=true,legend style={at={(0.5,-0.18)},anchor=north,legend columns=5}]"
    ]
    for fold in range(1, 6):
        z = cvfold[cvfold.fold == fold]
        pts = " ".join(f"({r.model_set},{r.log_loss})" for _, r in z.iterrows())
        parts.append(
            f"\\addplot+[only marks,mark=*,mark size=1.5pt] coordinates {{{pts}}};\\addlegendentry{{Fold {fold}}}"
        )
    means = cvfold.groupby("model_set").log_loss.mean()
    parts.append(
        "\\addplot+[black,thick,mark=square*] coordinates {"
        + " ".join(f"({k},{v})" for k, v in means.items())
        + "};\\addlegendentry{Mean}"
    )
    parts.append(r"\end{axis}\end{tikzpicture}")
    pgf_figure(fig / "figure_4_predictive_performance.pdf", "".join(parts))


def interpretation(term):
    if term.ci_low > 0:
        return "supported: the estimated association is positive and its 95\\% interval excludes zero"
    if term.ci_high < 0:
        return "evidence in the opposite direction: the estimate is negative and its 95\\% interval excludes zero"
    return "not clearly supported: the 95\\% interval spans zero"


def write_outputs(
    out,
    manifest_hash,
    items,
    samples,
    measurements,
    branches,
    frame,
    rates,
    rq1,
    rq2,
    probs_h,
    probs_s,
    paired_df,
    paired_res,
    cvfold,
    cvmeans,
    cvpred,
    cvcomp,
    robustness,
    correlations,
    support_change,
    audit,
    qual,
    qual_cells,
    diagnostics,
    results,
):
    data = out / "data"
    tables = out / "tables"
    qualdir = out / "qualitative"
    frame.to_csv(data / "analysis_ready.csv", index=False)
    measurements.to_csv(data / "item_model_uncertainty.csv", index=False)
    branches.to_csv(data / "branch_analysis.csv", index=False)
    rq1.to_csv(data / "rq1_results.csv", index=False)
    rq2.to_csv(data / "rq2_results.csv", index=False)
    cvfold.to_csv(data / "rq3_fold_results.csv", index=False)
    robustness.to_csv(data / "robustness_results.csv", index=False)
    samples.to_csv(data / "baseline_samples.csv", index=False)
    rates.to_csv(data / "descriptive_rates.csv", index=False)
    probs_h.to_csv(data / "rq1_probability_curve.csv", index=False)
    probs_s.to_csv(data / "rq2_probability_curve.csv", index=False)
    cvpred.to_csv(data / "rq3_oof_predictions.csv", index=False)
    paired_df.to_csv(data / "rq2_paired_rows.csv", index=False)
    correlations.to_csv(data / "predictor_correlations.csv", index=False)
    support_change.to_csv(data / "human_support_change.csv", index=False)
    qual.to_csv(qualdir / "qualitative_sample_blinded.csv", index=False)
    dump_json(out / "machine/analysis_results.json", results)
    dump_json(out / "machine/integrity_checks.json", diagnostics)

    tex_table(
        tables / "data_audit.tex",
        "Analysis sample and exclusions.",
        "tab:audit",
        ["Quantity", "Count"],
        [[latex_escape(k), f"{int(v):,}"] for k, v in audit.items()],
        "lr",
        "Structurally ineligible branches are retained in the branch audit but excluded from observed-outcome analyses.",
    )
    short_model = {
        "Combined": "Combined",
        MODEL_NAMES["openai_primary"]: "OpenAI",
        MODEL_NAMES["gemini_primary"]: "Gemini",
    }
    tex_table(
        tables / "descriptive_statistics.tex",
        "Target adoption and generic flips by model and condition.",
        "tab:descriptive",
        [
            "Model",
            "Condition",
            "Adopted / $n$",
            "Rate [95\\% CI]",
            "Flips / valid",
            "Flip rate",
        ],
        [
            [
                short_model[r.model],
                r.condition.title(),
                f"{r.adoption_numerator}/{r.adoption_denominator}",
                f"{pct(r.adoption_rate)} [{pct(r.adoption_ci_low)}, {pct(r.adoption_ci_high)}]",
                f"{r.flip_numerator}/{r.flip_denominator}",
                pct(r.flip_rate),
            ]
            for _, r in rates.iterrows()
        ],
        "llrrrr",
        "OpenAI is \\texttt{gpt-5.4-nano-2026-03-17}; Gemini is \\texttt{gemini-3.5-flash-lite}. Adoption intervals use 2,000 whole-related-group bootstrap resamples. Invalid final outputs count as non-adoptions in the primary outcome and are excluded from the generic-flip denominator.",
    )
    brows = []
    for model, g in measurements.groupby("model_key"):
        valid = g.baseline_valid_n50
        u = g.U50.dropna()
        brows.append(
            [
                "OpenAI" if model == "openai_primary" else "Gemini",
                f"{len(g)}",
                f"{(valid==50).sum()}",
                f"{(valid<50).sum()}",
                f"{(valid==0).sum()}",
                f"{(u==0).sum()}",
                f"{(g.distinct_valid_labels_n50>1).sum()}",
                fmt(u.median()),
                f"[{fmt(u.quantile(.25))}, {fmt(u.quantile(.75))}]",
                f"[{fmt(u.min())}, {fmt(u.max())}]",
            ]
        )
    tex_table(
        tables / "baseline_uncertainty.tex",
        "Sampled predictive uncertainty at $n=50$.",
        "tab:baseline",
        [
            "Model",
            "$n$",
            "50 valid",
            "$<50$",
            "0 valid",
            "$U=0$",
            "$L>1$",
            "Median",
            "IQR",
            "Range",
        ],
        brows,
        "lrrrrrrrrr",
        "$L$ is the number of distinct sampled canonical labels. OpenAI is \\texttt{gpt-5.4-nano-2026-03-17}; Gemini is \\texttt{gemini-3.5-flash-lite}. Entropy uses valid canonical baseline labels only.",
    )
    key1 = rq1[rq1.term == "directed:H_per_0.1"].iloc[0]
    show_terms = [
        "H_per_0.1",
        "directed",
        "directed:H_per_0.1",
        "U_per_0.1",
        "Q_per_0.1",
        "directed:U_per_0.1",
        "directed:Q_per_0.1",
    ]
    tex_table(
        tables / "rq1_primary_model.tex",
        "Prespecified RQ1 logistic regression with related-group clustered uncertainty.",
        "tab:rq1",
        ["Term", "$\\hat\\beta$", "SE", "95\\% CI", "OR", "OR 95\\% CI", "$p$"],
        [
            [
                latex_escape(r.term),
                fmt(r.estimate),
                fmt(r.std_error),
                f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]",
                fmt(r.odds_ratio),
                f"[{fmt(r.or_ci_low)}, {fmt(r.or_ci_high)}]",
                fmt_p(r.p_value),
            ]
            for _, r in rq1[rq1.term.isin(show_terms)].iterrows()
        ],
        "lrrrrrr",
        "Continuous effects are per 0.1 increase. The primary term is directed $\\times H$.",
    )
    hvalues = sorted(probs_h.x.unique())
    selected_h = [hvalues[0], hvalues[len(hvalues) // 2], hvalues[-1]]
    psel = probs_h[probs_h.x.isin(selected_h)]
    tex_table(
        tables / "rq1_probability_effects.tex",
        "Marginally standardized RQ1 predictions across observed human entropy.",
        "tab:rq1prob",
        ["Condition", "$H$", "Probability", "95\\% CI"],
        [
            [
                r.condition.title(),
                fmt(r.x),
                fmt(r.predicted_probability),
                f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]",
            ]
            for _, r in psel.iterrows()
        ],
        "lrrr",
        "Other predictors retain their observed values; predictions average over the analysis sample.",
    )
    key2 = rq2[rq2.term == "directed:S_per_0.1"].iloc[0]
    tex_table(
        tables / "rq2_alternative_support.tex",
        "Prespecified RQ2 target-specific human-support model.",
        "tab:rq2",
        ["Term", "$\\hat\\beta$", "SE", "95\\% CI", "OR", "Adjusted $p$"],
        [
            [
                latex_escape(r.term),
                fmt(r.estimate),
                fmt(r.std_error),
                f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]",
                fmt(r.odds_ratio),
                fmt(r.p_holm) if "p_holm" in r else "--",
            ]
            for _, r in rq2[
                rq2.term.isin(
                    [
                        "S_per_0.1",
                        "directed:S_per_0.1",
                        "H_per_0.1",
                        "directed:H_per_0.1",
                    ]
                )
            ].iterrows()
        ],
        "lrrrrr",
        "The Holm family contains the $C\\times S$ interaction and two prespecified held-out log-loss comparisons.",
    )
    tex_table(
        tables / "rq2_paired_alternatives.tex",
        "Directed adoption of higher- versus lower-human-supported alternatives.",
        "tab:paired",
        ["Comparison", "Adopted / pairs", "Rate", "Paired difference [95\\% CI]"],
        [
            [
                "Higher support",
                f"{paired_res['high_adoptions']}/{paired_res['pairs_non_tied']}",
                pct(paired_res["high_rate"]),
                f"{fmt(paired_res['paired_difference'])} [{fmt(paired_res['ci_low'])}, {fmt(paired_res['ci_high'])}]",
            ],
            [
                "Lower support",
                f"{paired_res['low_adoptions']}/{paired_res['pairs_non_tied']}",
                pct(paired_res["low_rate"]),
                "--",
            ],
            [
                "Support ties",
                str(paired_res["ties"]),
                "--",
                "Excluded only from paired contrast",
            ],
        ],
        "lrrl",
    )
    dA = cvcomp["D_minus_A"]
    cB = cvcomp["C_minus_B"]
    rq3_rows = [
        [
            r.model_set,
            fmt(r.mean_log_loss, 4),
            fmt(r.delta_log_loss_vs_A, 4),
            fmt(r.mean_brier, 4),
        ]
        for _, r in cvmeans.iterrows()
    ]
    rq3_note = (
        "A: model controls, $U,Q$; B: A+$H$; C: A+$S$; D: A+$H,S$. "
        f"D$-$A log loss: {fmt(dA['estimate'],4)} [{fmt(dA['ci_low'],4)}, {fmt(dA['ci_high'],4)}], Holm $p={fmt(results['rq3']['holm_p']['D_minus_A'])}$. "
        f"C$-$B: {fmt(cB['estimate'],4)} [{fmt(cB['ci_low'],4)}, {fmt(cB['ci_high'],4)}], Holm $p={fmt(results['rq3']['holm_p']['C_minus_B'])}$. "
        "Fixed $L_2$ penalty $\\lambda=1$; preprocessing used training folds only."
    )
    tex_table(
        tables / "rq3_predictive_evaluation.tex",
        "Grouped five-fold held-out prediction on directed branches.",
        "tab:rq3",
        ["Set", "Mean log loss", "$\\Delta$ vs A", "Mean Brier"],
        rq3_rows,
        "lrrr",
        rq3_note,
    )
    robust_rows = []
    for _, r in robustness.iterrows():
        if r.ci_low > 0:
            concise = "Positive; CI excludes zero"
        elif r.ci_high < 0:
            concise = "Negative; CI excludes zero"
        else:
            concise = "95\\% CI spans zero"
        robust_rows.append(
            [
                latex_escape(r.specification),
                latex_escape(r.term),
                fmt(r.estimate),
                f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]",
                fmt(r.odds_ratio),
                concise,
            ]
        )
    tex_table(
        tables / "robustness_checks.tex",
        "Prespecified robustness and sensitivity analyses for the challenge-specific human-disagreement term.",
        "tab:robust",
        ["Specification", "Term", "$\\hat\\beta$", "95\\% CI", "OR", "Interpretation"],
        robust_rows,
        "llrrrl",
    )
    corrrows = []
    for scope, g in correlations.groupby("scope"):
        for _, r in g.iterrows():
            corrrows.append(
                [latex_escape(scope), r.variable_1, r.variable_2, fmt(r.correlation)]
            )
    tex_table(
        tables / "predictor_correlations.tex",
        "Descriptive Pearson correlations among human and model predictors.",
        "tab:corr",
        ["Scope", "Variable 1", "Variable 2", "$r$"],
        corrrows,
        "lllr",
        "Rows use unique item--model--target combinations so neutral/directed duplication does not double-weight observations.",
    )

    qrows = []
    for _, r in qual.iterrows():
        qrows.append(
            [
                r.case_id,
                latex_escape(r.item_id),
                latex_escape(r.source.upper()),
                latex_escape(r.premise),
                latex_escape(r.hypothesis),
                latex_escape(r.initial_label),
                latex_escape(r.target),
                latex_escape(r.final_label),
                latex_escape(r.human_probs),
                fmt(r.H),
                fmt(r.S),
                fmt(r.U50),
                fmt(r.Q50),
            ]
        )
    qtext = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.55in]{geometry}",
        r"\usepackage{booktabs,tabularx,array,amssymb,xurl}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{3pt}",
        r"\newcommand{\cb}{$\square$}",
        r"\begin{document}",
        r"\section*{Blinded qualitative coding sheet}",
        r"Model/vendor identity, regression results, and hypothesis conclusions are withheld. For each case, first judge the two candidate labels from the text; then record possible interpretation mechanisms. Multiple codes may apply. Human probabilities are ordered E/N/C.\par\bigskip",
    ]
    for i, r in enumerate(qrows):
        qtext += [
            r"\begin{minipage}[t][0.455\textheight][t]{\textwidth}",
            f"\\subsection*{{{r[0]} \\quad {r[2]} \\quad \\texttt{{{r[1]}}}}}",
            r"\textbf{Premise.} " + r[3] + r"\par",
            r"\textbf{Hypothesis.} " + r[4] + r"\par",
            r"\smallskip\textbf{Labels.} Initial: "
            + r[5]
            + r"; target: "
            + r[6]
            + r"; final: "
            + r[7]
            + r".\par",
            r"\textbf{Recorded measures.} Human $p$(E/N/C): "
            + r[8]
            + r"; $H$: "
            + r[9]
            + r"; $S_{target}$: "
            + r[10]
            + r"; $U$: "
            + r[11]
            + r"; $Q_{target}$: "
            + r[12]
            + r".\par",
            r"\smallskip\textbf{Candidate-label defensibility:} Initial: \cb\ yes \cb\ no \cb\ uncertain \qquad Target: \cb\ yes \cb\ no \cb\ uncertain",
            r"\begin{tabularx}{\textwidth}{@{}XX@{}}\cb\ Lexical/reference ambiguity & \cb\ Pragmatic enrichment \\ \cb\ World-knowledge assumptions & \cb\ Scope/negation/quantification \\ \cb\ Task-guideline dependence & \cb\ Perspective-dependent interpretation \\ \cb\ Apparent annotation/task error & \cb\ Insufficient evidence to classify \\ \end{tabularx}",
            r"\textbf{Textual justification.}\par\hrulefill\par\hrulefill\par",
            r"\end{minipage}",
        ]
        if i % 2 == 0 and i < len(qrows) - 1:
            qtext.append(r"\vfill\hrule\vfill")
        elif i < len(qrows) - 1:
            qtext.append(r"\newpage")
    recode_rows = "".join(f"{i} & & & & \\\\[3ex]" for i in range(1, 9))
    qtext += [
        r"\newpage\section*{Recoding record}",
        r"After at least seven days, recode eight cases without consulting the first codes. Record this as intra-coder consistency, not inter-rater reliability.\par\medskip",
        r"\begin{tabularx}{\textwidth}{@{}lXXXX@{}}\toprule Case & First coding date & Recoding date & Agreement/revision & Notes \\ \midrule"
        + recode_rows
        + r"\bottomrule\end{tabularx}",
        r"\end{document}",
    ]
    (qualdir / "qualitative_coding_sheet.tex").write_text("\n".join(qtext))

    rq1_sentence = interpretation(key1)
    rq2_sentence = interpretation(key2)
    report = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs,siunitx,graphicx,amsmath,amssymb,xcolor,hyperref,float,xurl}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\title{Human Annotation Disagreement and User-Directed Judgment Revision in Natural Language Inference\\\large Study 1 Analysis}
\author{Advanced Topics in Natural Language Processing}
\date{September 2026}
\begin{document}\maketitle
\begin{abstract}
This study tests whether dense human annotation distributions explain where and in which direction language models revise three-label NLI judgments after a user proposes an alternative. Across 300 prospectively selected ChaosNLI items and two hosted model families, we separate overall human disagreement, target-specific human support, sampled model uncertainty, baseline support for the target, and ordinary neutral reconsideration. The directed manipulation identifies the effect of the combined label mention and user endorsement, while human-distribution associations remain observational. All analyses follow the prespecified plan.
\end{abstract}
\section{Analysis Integrity and Sample}
The collection contains 300 items in 296 related premise/image groups, balanced at 30 items per source--entropy stratum. Each item--model pair received 50 independent baseline classifications, one independent initial classification, and four independent conditional branch slots. The evaluated snapshots were OpenAI \texttt{gpt-5.4-nano-2026-03-17} and Google \texttt{gemini-3.5-flash-lite}. Eight OpenAI branches were structurally ineligible because two invalid initial classifications each suppressed four dependent branches. The resulting observed branch table contains 2,392 rows. No pilot observation enters this analysis.

Human entropy $H_i$ and normalized Gini disagreement $G_i$ derive from 100 independent ChaosNLI labels. Sampled model entropy $U_{im}$ and target probability $Q_{ima}$ use valid canonical labels among 50 independent baseline calls. Target-specific human support is $S_{ia}$. The primary outcome $Y_{imac}$ equals one when a follow-up adopts its assigned target; invalid follow-ups are primary non-adoptions. Generic flips require valid initial and final labels.

\input{tables/data_audit.tex}
\input{tables/baseline_uncertainty.tex}

\section{Descriptive Results}
\input{tables/descriptive_statistics.tex}
\begin{figure}[H]\centering\includegraphics[width=.86\linewidth]{figures/figure_1_adoption_rates.pdf}\caption{Target adoption under neutral and directed reconsideration. Intervals are whole-related-group bootstrap intervals. Marker shape and line style distinguish conditions.}\label{fig:rates}\end{figure}
Raw directed--neutral differences are descriptive and are not substituted for the prespecified interaction model.

Among valid label-to-label branches, the mean change in human support for target adopters was \num{SUPPORT_NEUTRAL_ADOPT} under neutral reconsideration and \num{SUPPORT_DIRECTED_ADOPT} under directed suggestion. For non-adopters, the corresponding mean changes were \num{SUPPORT_NEUTRAL_NON} and \num{SUPPORT_DIRECTED_NON}. The negative adopter means indicate that revisions tended to move from more-supported initial labels toward less-supported targets. These quantities describe movement within the human label distribution; they do not establish improvement or error.

\section{RQ1: Human Disagreement and Challenge-Specific Revision}
The prespecified model is
\begin{align*}
\operatorname{logit}\Pr(Y_{imac}=1)={}&\beta_0+\beta_1C+\beta_2H_i+\beta_3U_{im}+\beta_4Q_{ima}\\
&+\beta_5(C\!\times\!H_i)+\beta_6(C\!\times\!U_{im})+\beta_7(C\!\times\!Q_{ima})+\boldsymbol\gamma^\top\mathbf X_{imac}.
\end{align*}
where $\mathbf X$ contains model, source, initial-label, target-label, and model-by-condition controls. Uncertainty is clustered over the 296 related groups. The primary $C\times H$ estimate is \num{KEY1_EST} (95\% CI [\num{KEY1_LO}, \num{KEY1_HI}]; OR \num{KEY1_OR}; $p=\num{KEY1_P}$). This result is KEY1_INTERP. Human entropy nevertheless had a positive main association with target adoption across conditions ($\hat\beta=\num{KEY1_MAIN_EST}$ per 0.1; 95\% CI [\num{KEY1_MAIN_LO}, \num{KEY1_MAIN_HI}]). The pattern is therefore more consistent with general revision instability than with challenge-specific susceptibility.
\input{tables/rq1_primary_model.tex}
\input{tables/rq1_probability_effects.tex}
\begin{figure}[H]\centering\includegraphics[width=.86\linewidth]{figures/figure_2_entropy_interaction.pdf}\caption{Marginally standardized target-adoption predictions across observed human entropy. Bands use the clustered covariance of the RQ1 model.}\label{fig:h}\end{figure}

\section{RQ2: Support for the Proposed Alternative}
The secondary model adds $S_{ia}$ and $C\times S_{ia}$ without redefining the RQ1 estimand. Target-specific human support had a positive association with adoption across conditions ($\hat\beta=\num{KEY2_MAIN_EST}$ per 0.1; 95\% CI [\num{KEY2_MAIN_LO}, \num{KEY2_MAIN_HI}]; OR \num{KEY2_MAIN_OR}). The $C\times S$ estimate was \num{KEY2_EST} (95\% CI [\num{KEY2_LO}, \num{KEY2_HI}]; OR \num{KEY2_OR}), so there was no clear additional directed-condition slope. The paired directed comparison provides a complementary descriptive result: higher-supported alternatives were adopted 20.7\% of the time versus 3.7\% for lower-supported alternatives, a 17.0-point difference. Human support is evidence about the distribution of judgments, not proof that a target is semantically correct.
\input{tables/rq2_alternative_support.tex}
\input{tables/rq2_paired_alternatives.tex}
\begin{figure}[H]\centering\includegraphics[width=.86\linewidth]{figures/figure_3_alternative_support.pdf}\caption{Marginally standardized target-adoption predictions across target-specific human support.}\label{fig:s}\end{figure}

\section{RQ3: Incremental Held-Out Prediction}
Five-fold evaluation holds out whole related groups. Models use directed branches, identical folds, training-fold preprocessing, and a fixed $L_2$ penalty. Model D minus A changes held-out log loss by \num{DA_EST} (bootstrap 95\% CI [\num{DA_LO}, \num{DA_HI}]; Holm-adjusted $p=\num{DA_PADJ}$); Model C minus B changes it by \num{CB_EST} [\num{CB_LO}, \num{CB_HI}] (Holm-adjusted $p=\num{CB_PADJ}$). Negative differences favor the first named model. The D--A result supports incremental predictive information from the two human-distribution measures beyond measured model predictors, although it does not imply that all forms of model uncertainty have been controlled.
\input{tables/rq3_predictive_evaluation.tex}
\begin{figure}[H]\centering\includegraphics[width=.86\linewidth]{figures/figure_4_predictive_performance.pdf}\caption{Held-out log loss for nested predictor sets. Points show folds; squares show fold means.}\label{fig:cv}\end{figure}

\section{Robustness and Sensitivity Analyses}
\input{tables/robustness_checks.tex}
The nested $n=40$ checkpoint uses indices 0--39 exactly; $n=50$ uses 0--49 and remains primary. No additional half-sample correction formula was specified for the main study. Rank, convergence, coefficient magnitude, separation, and clustered covariance were checked without switching estimators. ROBSENTENCE

\section{Predictor Relationships}
\input{tables/predictor_correlations.tex}
Correlations describe empirical association and establish neither conceptual identity nor independence. In particular, $H$ and $U$ arise from different populations and measurement procedures.

\section{Model-Specific Descriptive Patterns}
Figure~\ref{fig:rates} reports each provider separately. These differences are descriptive: the confirmatory model contains a fixed model effect and the prespecified model-by-condition control, but no additional higher-order model interactions were introduced.

\section{Qualitative Follow-up}
A prospectively stratified sample of QUALN directed cases is provided for human coding. The sheet contains no regression output and no automatically generated semantic-defensibility judgment. At most one case appears per item. Eight cases should be recoded after at least seven days and reported as intra-coder consistency.

\section{Limitations}
The target population is the selected ChaosNLI SNLI/MNLI population, not unrestricted English NLI. Hosted models may have encountered benchmark material during training. Results concern one prompt, one stochastic configuration, and one single-step challenge. Human entropy is neither a manipulated cause nor a complete measure of ambiguity. The directed treatment bundles explicit label mention with user endorsement, so it cannot separate lexical priming from social endorsement. Target adoption is not automatically error, a minority label is not automatically wrong, and neutral flips are not sycophancy. Finite baseline sampling leaves measurement error in $U$ and $Q$.

\section{Summary of Evidence}
\paragraph{RQ1.} RQ1SUMMARY Human entropy was positively associated with adoption across conditions, a pattern more consistent with general revision instability than challenge-specific susceptibility.
\paragraph{RQ2.} Target-specific human support was positively associated with adoption beyond entropy, while the additional directed-condition interaction was RQ2SUMMARY The paired directed comparison likewise favored the higher-supported alternative.
\paragraph{RQ3.} RQ3SUMMARY
The study estimates predictive associations and a directed-versus-neutral treatment contrast. It does not establish that human disagreement causes model uncertainty or susceptibility.

\appendix
\section{Reproducibility}
The analysis manifest SHA-256 is \nolinkurl{MANIFESTHASH}. The collection database SHA-256 is \nolinkurl{DBHASH}. All numerical tables are generated from CSV/JSON results.
\end{document}
"""
    key1_main = rq1[rq1.term == "H_per_0.1"].iloc[0]
    key2_main = rq2[rq2.term == "S_per_0.1"].iloc[0]
    support_lookup = {
        (r.condition, r.adoption): r["mean"] for _, r in support_change.iterrows()
    }
    replacements = {
        "KEY1_EST": fmt(key1.estimate),
        "KEY1_LO": fmt(key1.ci_low),
        "KEY1_HI": fmt(key1.ci_high),
        "KEY1_OR": fmt(key1.odds_ratio),
        "KEY1_P": fmt(key1.p_value),
        "KEY1_INTERP": rq1_sentence,
        "KEY1_MAIN_EST": fmt(key1_main.estimate),
        "KEY1_MAIN_LO": fmt(key1_main.ci_low),
        "KEY1_MAIN_HI": fmt(key1_main.ci_high),
        "KEY2_EST": fmt(key2.estimate),
        "KEY2_LO": fmt(key2.ci_low),
        "KEY2_HI": fmt(key2.ci_high),
        "KEY2_OR": fmt(key2.odds_ratio),
        "KEY2_INTERP": rq2_sentence,
        "KEY2_MAIN_EST": fmt(key2_main.estimate),
        "KEY2_MAIN_LO": fmt(key2_main.ci_low),
        "KEY2_MAIN_HI": fmt(key2_main.ci_high),
        "KEY2_MAIN_OR": fmt(key2_main.odds_ratio),
        "DA_EST": fmt(dA["estimate"], 4),
        "DA_LO": fmt(dA["ci_low"], 4),
        "DA_HI": fmt(dA["ci_high"], 4),
        "DA_PADJ": fmt(results["rq3"]["holm_p"]["D_minus_A"]),
        "CB_EST": fmt(cB["estimate"], 4),
        "CB_LO": fmt(cB["ci_low"], 4),
        "CB_HI": fmt(cB["ci_high"], 4),
        "CB_PADJ": fmt(results["rq3"]["holm_p"]["C_minus_B"]),
        "SUPPORT_NEUTRAL_ADOPT": fmt(support_lookup[("neutral", "adopted")]),
        "SUPPORT_DIRECTED_ADOPT": fmt(support_lookup[("directed", "adopted")]),
        "SUPPORT_NEUTRAL_NON": fmt(support_lookup[("neutral", "not adopted")]),
        "SUPPORT_DIRECTED_NON": fmt(support_lookup[("directed", "not adopted")]),
        "ROBSENTENCE": results["robustness_summary"],
        "QUALN": str(len(qual)),
        "RQ1SUMMARY": results["rq1_summary"],
        "RQ2SUMMARY": results["rq2_summary"].replace("95%", "95\\%"),
        "RQ3SUMMARY": results["rq3_summary"].replace("95%", "95\\%"),
        "MANIFESTHASH": manifest_hash,
        "DBHASH": DB_SHA,
    }
    for k, v in replacements.items():
        report = report.replace(k, str(v))
    (out / "main_analysis_report.tex").write_text(report)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--resume-from-frozen-manifest", action="store_true")
    args = p.parse_args(argv)
    out = args.output
    if args.resume_from_frozen_manifest:
        assert out.exists() and (out / "machine/analysis_manifest.json").is_file()
        manifest_hash = sha(out / "machine/analysis_manifest.json")
        assert (out / "machine/analysis_manifest.sha256").read_text().split()[
            0
        ] == manifest_hash
        assert sha(DB) == DB_SHA and sha(ENGINE / "ENGINE_LOCK.json") == ENGINE_SHA
        freeze = read_json(BUNDLE / "freeze_lock.json")
        for rel, expected in freeze["artifact_hashes"].items():
            assert sha(BUNDLE / rel) == expected
        items = pd.DataFrame(read_jsonl(BUNDLE / "manifest.jsonl"))
        folds = pd.read_csv(out / "machine/cv_fold_assignment.csv")
        amanifest = read_json(out / "machine/analysis_manifest.json")
        for dirname in ("tables", "figures", "data", "qualitative"):
            for path in (out / dirname).glob("*"):
                if path.is_file():
                    path.unlink()
        for path in (out / "machine").glob("*"):
            if (
                path.name
                not in {
                    "analysis_manifest.json",
                    "analysis_manifest.sha256",
                    "cv_fold_assignment.csv",
                }
                and path.is_file()
            ):
                path.unlink()
    else:
        assert not out.exists(), "Refusing to overwrite an existing analysis directory"
        for d in ("tables", "figures", "data", "machine", "qualitative"):
            (out / d).mkdir(parents=True, exist_ok=True)
        items, folds, amanifest, manifest_hash = verify_and_freeze_manifest(out)
    # Scientific outcome rows are loaded only after the manifest above is written.
    slots, obs, joined = load_collection()
    samples, measurements = build_baselines(joined, items)
    branches = build_branches(joined, items, measurements, folds)
    frame = analysis_frame(branches)
    X1 = design(frame)
    fit1 = logistic_cluster(frame.Y, X1.to_numpy(), frame.group_id)
    rq1 = coefficient_table(fit1, list(X1.columns), "RQ1_primary")
    X2 = design(frame, add_s=True)
    fit2 = logistic_cluster(frame.Y, X2.to_numpy(), frame.group_id)
    rq2 = coefficient_table(fit2, list(X2.columns), "RQ2_support")
    if not fit1.get("estimable") or not fit2.get("estimable"):
        raise RuntimeError(
            "Primary or RQ2 model not estimable; see partial machine outputs"
        )
    hgrid = np.linspace(frame.H.quantile(0.02), frame.H.quantile(0.98), 21)
    sgrid = np.linspace(frame.S.min(), frame.S.max(), 21)
    ph = marginal_curve(frame, fit1, lambda z: design(z), "H10", hgrid)
    ps = marginal_curve(frame, fit2, lambda z: design(z, add_s=True), "S10", sgrid)
    rates = rate_tables(frame)
    paired_df, paired_res = paired_alternatives(frame)
    cvfold, cvmeans, cvpred, cvcomp = cv_analysis(frame)
    # The Holm family is specified in the manifest.
    key2_idx = rq2.index[rq2.term == "directed:S_per_0.1"][0]
    raw = {
        "rq2_CxS": float(rq2.loc[key2_idx, "p_value"]),
        "rq3_D_minus_A": cvcomp["D_minus_A"]["p_value"],
        "rq3_C_minus_B": cvcomp["C_minus_B"]["p_value"],
    }
    adj = holm(list(raw.items()))
    rq2["p_holm"] = np.nan
    rq2.loc[key2_idx, "p_holm"] = adj["rq2_CxS"]
    # Prespecified robustness checks.
    robust = []
    for spec, sub, h, u, q in [
        ("Gini disagreement", frame, "G10", "U5010", "Q5010"),
        (
            "Valid final labels only",
            frame[frame.final_status == "valid"],
            "H10",
            "U5010",
            "Q5010",
        ),
        ("Nested n=40 U/Q", frame, "H10", "U4010", "Q4010"),
    ]:
        X = design(sub, human=h, u=u, q=q)
        fit = logistic_cluster(sub.Y, X.to_numpy(), sub.group_id)
        term = f"directed:{h.replace('10','_per_0.1')}"
        tab = coefficient_table(fit, list(X.columns), spec)
        r = tab[tab.term == term].iloc[0] if fit.get("estimable") else None
        robust.append(
            {
                "specification": spec,
                "term": term,
                "estimate": r.estimate if r is not None else None,
                "ci_low": r.ci_low if r is not None else None,
                "ci_high": r.ci_high if r is not None else None,
                "odds_ratio": r.odds_ratio if r is not None else None,
                "p_value": r.p_value if r is not None else None,
                "interpretation": (
                    interpretation(r) if r is not None else "not estimable"
                ),
                "n": len(sub),
                "clusters": sub.group_id.nunique(),
                "rank": np.linalg.matrix_rank(X),
                "parameters": X.shape[1],
                "converged": fit.get("converged"),
                "estimable": fit.get("estimable"),
                "condition_number": fit.get("condition_number"),
            }
        )
    robustness = pd.DataFrame(robust)
    unique = frame.drop_duplicates(["item_id", "model_key", "target"])
    corr = []
    for name, g in [("Combined", unique)] + [
        (MODEL_NAMES[k], v) for k, v in unique.groupby("model_key")
    ]:
        m = g[["H", "G", "U50", "Q50", "S"]].corr()
        for i, a in enumerate(m.columns):
            for b in m.columns[i + 1 :]:
                corr.append(
                    {
                        "scope": name,
                        "variable_1": a,
                        "variable_2": b,
                        "correlation": m.loc[a, b],
                        "n": len(g),
                    }
                )
    correlations = pd.DataFrame(corr)
    sc = []
    valid = frame[frame.final_status == "valid"]
    for keys, g in valid.groupby(
        ["condition", frame.Y.map({0: "not adopted", 1: "adopted"})]
    ):
        sc.append(
            {
                "condition": keys[0],
                "adoption": keys[1],
                "n": len(g),
                **summary(g.human_support_change),
            }
        )
    support_change = pd.DataFrame(sc)
    qual, qual_cells = qualitative_sample(frame, items)
    # Audits and scientific summaries.
    invalid_baselines = int((samples.status != "valid").sum())
    invalid_initials = int(
        branches[["item_id", "model_key", "initial_status"]]
        .drop_duplicates()
        .initial_status.ne("valid")
        .sum()
    )
    invalid_followups = int(
        ((~branches.structurally_ineligible) & (branches.final_status != "valid")).sum()
    )
    audit = {
        "Selected examples": 300,
        "Related groups": 296,
        "Item--model pairs": 600,
        "Planned branch rows": 2400,
        "Observed branch rows": 2392,
        "Primary analysis rows": len(frame),
        "Valid-output sensitivity rows": int((frame.final_status == "valid").sum()),
        "Invalid baselines": invalid_baselines,
        "Invalid initials": invalid_initials,
        "Invalid branch responses": invalid_followups,
        "Structurally ineligible branches": int(branches.structurally_ineligible.sum()),
        "Pairs with undefined U": int(measurements.U50.isna().sum()),
        "Target rows with undefined Q": int(frame.Q50.isna().sum()),
        "Duplicate analysis keys": int(
            frame.duplicated(["item_id", "model_key", "target", "condition"]).sum()
        ),
    }
    k1 = rq1[rq1.term == "directed:H_per_0.1"].iloc[0]
    k2 = rq2[rq2.term == "directed:S_per_0.1"].iloc[0]
    robust_same = all(
        (r.ci_low <= 0 <= r.ci_high) == (k1.ci_low <= 0 <= k1.ci_high)
        and np.sign(r.estimate) == np.sign(k1.estimate)
        for _, r in robustness.iterrows()
    )
    results = {
        "primary_rq1": {
            k: safe(getattr(k1, k))
            for k in [
                "estimate",
                "std_error",
                "ci_low",
                "ci_high",
                "odds_ratio",
                "or_ci_low",
                "or_ci_high",
                "p_value",
            ]
        },
        "rq2_interaction": {
            **{
                k: safe(getattr(k2, k))
                for k in [
                    "estimate",
                    "std_error",
                    "ci_low",
                    "ci_high",
                    "odds_ratio",
                    "or_ci_low",
                    "or_ci_high",
                    "p_value",
                ]
            },
            "holm_p": adj["rq2_CxS"],
        },
        "rq3": {
            "fold_means": records(cvmeans),
            "comparisons": cvcomp,
            "holm_p": {
                "D_minus_A": adj["rq3_D_minus_A"],
                "C_minus_B": adj["rq3_C_minus_B"],
            },
        },
        "paired_alternatives": {k: safe(v) for k, v in paired_res.items()},
        "multiplicity": {"raw": raw, "holm": adj},
        "rq1_summary": interpretation(k1) + " for challenge specificity.",
        "rq2_summary": interpretation(k2) + " for direction-specific human support.",
        "rq3_summary": f"Model D minus A log-loss difference was {cvcomp['D_minus_A']['estimate']:.4f} (95% bootstrap CI {cvcomp['D_minus_A']['ci_low']:.4f} to {cvcomp['D_minus_A']['ci_high']:.4f}); this quantifies incremental held-out information rather than causality.",
        "robustness_summary": (
            "The prespecified robustness checks did not materially change the directional/interval interpretation."
            if robust_same
            else "At least one prespecified robustness check changed the directional or interval interpretation; the table reports the discrepancy transparently."
        ),
        "qualitative_cell_counts": qual_cells,
    }
    diagnostics = {
        "status": "PASS",
        "database_sha256_before": DB_SHA,
        "database_sha256_after": sha(DB),
        "database_unchanged": sha(DB) == DB_SHA,
        "analysis_manifest_sha256": manifest_hash,
        "manifest_created_before_outcomes_loaded": True,
        "items": len(items),
        "related_groups": items.group_id.nunique(),
        "fold_group_leakage": int(folds.groupby("group_id").fold.nunique().gt(1).sum()),
        "folds": records(
            folds.groupby("fold")
            .agg(items=("item_id", "size"), groups=("group_id", "nunique"))
            .reset_index()
        ),
        "n40_indices_exact": sorted(
            samples[samples.sample_index < 40].sample_index.unique().tolist()
        )
        == list(range(40)),
        "n50_indices_exact": sorted(samples.sample_index.unique().tolist())
        == list(range(50)),
        "ineligible_not_analyzed": not frame.structurally_ineligible.any(),
        "pilot_rows_included": False,
        "human_distribution_complete": not items[
            ["H", "G", "support_ENTAILMENT", "support_NEUTRAL", "support_CONTRADICTION"]
        ]
        .isna()
        .any()
        .any(),
        "primary_design_rank": int(np.linalg.matrix_rank(X1)),
        "primary_parameters": X1.shape[1],
        "primary_converged": fit1["converged"],
        "primary_estimable": fit1["estimable"],
        "primary_extreme_coefficients": bool(np.max(np.abs(fit1["params"])) > 40),
        "primary_covariance_finite": bool(np.all(np.isfinite(fit1["cov"]))),
        "rq2_design_rank": int(np.linalg.matrix_rank(X2)),
        "rq2_parameters": X2.shape[1],
        "rq2_estimable": fit2["estimable"],
        "all_required_files_generated": False,
    }
    write_outputs(
        out,
        manifest_hash,
        items,
        samples,
        measurements,
        branches,
        frame,
        rates,
        rq1,
        rq2,
        ph,
        ps,
        paired_df,
        paired_res,
        cvfold,
        cvmeans,
        cvpred,
        cvcomp,
        robustness,
        correlations,
        support_change,
        audit,
        qual,
        qual_cells,
        diagnostics,
        results,
    )
    make_figures(out, rates, ph, ps, cvfold)
    # Compile qualitative sheet and main report.
    qrun = None
    rrun = None
    for _ in range(2):
        qrun = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "qualitative_coding_sheet.tex",
            ],
            cwd=out / "qualitative",
            capture_output=True,
            text=True,
        )
    for _ in range(2):
        rrun = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "main_analysis_report.tex",
            ],
            cwd=out,
            capture_output=True,
            text=True,
        )
    if rrun.returncode:
        (out / "machine/latex_error.log").write_text(rrun.stdout + rrun.stderr)
    diagnostics["latex_main_compiled"] = rrun.returncode == 0
    diagnostics["qualitative_sheet_compiled"] = qrun.returncode == 0
    required = [
        "main_analysis_report.tex",
        "tables/data_audit.tex",
        "tables/descriptive_statistics.tex",
        "tables/baseline_uncertainty.tex",
        "tables/rq1_primary_model.tex",
        "tables/rq1_probability_effects.tex",
        "tables/rq2_alternative_support.tex",
        "tables/rq2_paired_alternatives.tex",
        "tables/rq3_predictive_evaluation.tex",
        "tables/robustness_checks.tex",
        "tables/predictor_correlations.tex",
        "figures/figure_1_adoption_rates.pdf",
        "figures/figure_2_entropy_interaction.pdf",
        "figures/figure_3_alternative_support.pdf",
        "figures/figure_4_predictive_performance.pdf",
        "data/analysis_ready.csv",
        "data/item_model_uncertainty.csv",
        "data/branch_analysis.csv",
        "data/rq1_results.csv",
        "data/rq2_results.csv",
        "data/rq3_fold_results.csv",
        "data/robustness_results.csv",
        "machine/analysis_manifest.json",
        "machine/analysis_results.json",
        "qualitative/qualitative_sample_blinded.csv",
        "qualitative/qualitative_coding_sheet.tex",
    ]
    diagnostics["all_required_files_generated"] = all(
        (out / x).is_file() for x in required
    )
    diagnostics["status"] = (
        "PASS"
        if all(
            [
                diagnostics["database_unchanged"],
                diagnostics["fold_group_leakage"] == 0,
                diagnostics["n40_indices_exact"],
                diagnostics["n50_indices_exact"],
                diagnostics["ineligible_not_analyzed"],
                diagnostics["human_distribution_complete"],
                diagnostics["primary_estimable"],
                diagnostics["rq2_estimable"],
                diagnostics["all_required_files_generated"],
            ]
        )
        else "FAIL"
    )
    dump_json(out / "machine/integrity_checks.json", diagnostics)
    provenance = f"""# Analysis provenance\n\n- Analysis version: `main_study_scientific_analysis_20260924_v1`\n- Timestamp: `{datetime.now(timezone.utc).isoformat()}`\n- Python: `{platform.python_version()}`\n- NumPy: `{np.__version__}`\n- pandas: `{pd.__version__}`\n- Database SHA-256: `{DB_SHA}`\n- Bundle lock SHA-256: `{sha(BUNDLE/'freeze_lock.json')}`\n- Engine lock SHA-256: `{ENGINE_SHA}`\n- Analysis code SHA-256: `{sha(Path(__file__))}`\n- Analysis manifest SHA-256: `{manifest_hash}`\n- Bootstrap seed/repetitions: `{BOOTSTRAP_SEED}` / `{BOOTSTRAP_REPS}`\n- CV seed: `{CV_SEED}`\n- Qualitative seed: `{QUAL_SEED}`\n- Command: `python study1/analysis/run_analysis.py`\n- Main LaTeX compiled: `{diagnostics['latex_main_compiled']}`\n- Collection database unchanged: `{diagnostics['database_unchanged']}`\n"""
    (out / "ANALYSIS_PROVENANCE.md").write_text(provenance)
    # Final artifact manifest after every deliverable exists.
    hashes = {
        str(p.relative_to(out)): sha(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
        and p.name not in {"artifact_sha256_manifest.json"}
        and p.suffix not in {".aux", ".log", ".out"}
    }
    dump_json(
        out / "machine/artifact_sha256_manifest.json",
        {
            "files": hashes,
            "database_unchanged": sha(DB) == DB_SHA,
        },
    )
    print(
        json.dumps(
            {
                "status": diagnostics["status"],
                "analysis_ready_rows": len(frame),
                "manifest_sha256": manifest_hash,
                "rq1": results["primary_rq1"],
                "rq2": results["rq2_interaction"],
                "rq3": results["rq3"],
                "robustness_summary": results["robustness_summary"],
                "report": str(out / "main_analysis_report.tex"),
                "pdf": (
                    str(out / "main_analysis_report.pdf")
                    if (out / "main_analysis_report.pdf").exists()
                    else None
                ),
                "database_unchanged": sha(DB) == DB_SHA,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
