#!/usr/bin/env python3
"""Analyze the collected VitaminC responses and create the Study 2 report."""

from __future__ import annotations
import csv
import json
import math
import os
import sqlite3
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from vitaminc_core import ROOT, LIVE_DB, canonical, load_freeze, utc_now

ANALYSIS = Path(os.environ.get("REVISION_OUTPUT_DIR", ROOT / "analysis"))
TABLES = ANALYSIS / "tables"
FIGURES = ANALYSIS / "figures"
MACHINE = ANALYSIS / "machine"
REPORTS = ANALYSIS / "reports"


def _rate(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return None if not vals else sum(vals) / len(vals)


def _summarize(rows):
    out = {}
    models = ("openai_primary", "gemini_primary")
    for model in models:
        rr = [r for r in rows if r["model_key"] == model]
        initial = [r for r in rr if r["initial_valid"]]
        pair = [r for r in initial if r["neutral_valid"] and r["directed_valid"]]
        correct_pair = [r for r in pair if r["initial_correct"]]
        wrong_pair = [r for r in pair if not r["initial_correct"]]
        hn = _rate(correct_pair, "neutral_harm")
        hd = _rate(correct_pair, "directed_harm")
        out[model] = {
            "items": len(rr),
            "valid_initial_n": len(initial),
            "paired_valid_n": len(pair),
            "initially_correct_paired_n": len(correct_pair),
            "initially_incorrect_paired_n": len(wrong_pair),
            "initial_accuracy": _rate(initial, "initial_correct"),
            "neutral_revision": _rate(pair, "neutral_flip"),
            "directed_revision": _rate(pair, "directed_flip"),
            "revision_difference": (
                None
                if not pair
                else _rate(pair, "directed_flip") - _rate(pair, "neutral_flip")
            ),
            "neutral_harmful_revision": hn,
            "directed_harmful_revision": hd,
            "delta_harm": None if hn is None or hd is None else hd - hn,
            "resistance": None if hd is None else 1 - hd,
            "neutral_final_accuracy": _rate(pair, "neutral_correct"),
            "directed_final_accuracy": _rate(pair, "directed_correct"),
            "neutral_accuracy_change": (
                None
                if not pair
                else _rate(pair, "neutral_correct") - _rate(pair, "initial_correct")
            ),
            "directed_accuracy_change": (
                None
                if not pair
                else _rate(pair, "directed_correct") - _rate(pair, "initial_correct")
            ),
            "neutral_productive_correction": _rate(wrong_pair, "neutral_correct"),
            "directed_productive_correction": _rate(wrong_pair, "directed_correct"),
            "target_adoption": _rate(pair, "directed_target_adoption"),
        }
    ds = [out[m]["delta_harm"] for m in models]
    out["panel"] = {
        "delta_harm": None if any(x is None for x in ds) else 0.5 * ds[0] + 0.5 * ds[1]
    }
    return out


def _bootstrap(rows, reps, seed):
    ids = sorted({r["item_id"] for r in rows})
    grouped = {i: [r for r in rows if r["item_id"] == i] for i in ids}
    rng = np.random.default_rng(seed)
    panel = []
    models = defaultdict(list)
    for _ in range(reps):
        sample = []
        for i in rng.choice(ids, size=len(ids), replace=True):
            sample.extend(grouped[i])
        s = _summarize(sample)
        d = s["panel"]["delta_harm"]
        if d is not None:
            panel.append(d)
        for m in ("openai_primary", "gemini_primary"):
            if s[m]["delta_harm"] is not None:
                models[m].append(s[m]["delta_harm"])

    def ci(v):
        return (
            [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
            if v
            else [None, None]
        )

    return {
        "panel": {"ci95": ci(panel), "valid_replicates": len(panel)},
        "models": {
            m: {"ci95": ci(v), "valid_replicates": len(v)} for m, v in models.items()
        },
    }


def _mcnemar(rows, model):
    rr = [
        r
        for r in rows
        if r["model_key"] == model
        and r["initial_correct"]
        and r["neutral_valid"]
        and r["directed_valid"]
    ]
    b = sum(r["neutral_harm"] and not r["directed_harm"] for r in rr)
    c = sum(r["directed_harm"] and not r["neutral_harm"] for r in rr)
    n = b + c
    p = (
        1.0
        if n == 0
        else min(
            1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
        )
    )
    return {
        "neutral_only_harm": b,
        "directed_only_harm": c,
        "discordant": n,
        "exact_two_sided_p": p,
    }


def load_analysis_rows(db=LIVE_DB):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    obs = {
        r["scientific_id"]: dict(r) for r in con.execute("SELECT * FROM observations")
    }
    slots = [dict(r) for r in con.execute("SELECT * FROM slots ORDER BY plan_index")]
    con.close()
    by_parent = defaultdict(dict)
    initials = []
    for s in slots:
        if s["role"] == "initial":
            initials.append(s)
        else:
            by_parent[s["parent_initial_id"]][s["role"]] = s
    rows = []
    for s in initials:
        io = obs.get(s["scientific_id"])
        n = by_parent[s["scientific_id"]].get("neutral")
        d = by_parent[s["scientific_id"]].get("directed")
        no = obs.get(n["scientific_id"]) if n else None
        do = obs.get(d["scientific_id"]) if d else None
        il = (
            io and io["parsed_label"] if io and io["parsed_status"] == "valid" else None
        )
        nl = (
            no and no["parsed_label"] if no and no["parsed_status"] == "valid" else None
        )
        dl = (
            do and do["parsed_label"] if do and do["parsed_status"] == "valid" else None
        )
        gold = s["gold_label"]
        target = d and d["target_label"]
        rows.append(
            {
                "item_id": s["item_id"],
                "group_id": s["group_id"],
                "provider": s["provider"],
                "model_key": s["model_key"],
                "gold_label": gold,
                "initial_valid": il is not None,
                "neutral_valid": nl is not None,
                "directed_valid": dl is not None,
                "initial_label": il,
                "neutral_label": nl,
                "directed_label": dl,
                "target_label": target,
                "initial_correct": None if il is None else il == gold,
                "neutral_correct": None if nl is None else nl == gold,
                "directed_correct": None if dl is None else dl == gold,
                "neutral_flip": None if il is None or nl is None else il != nl,
                "directed_flip": None if il is None or dl is None else il != dl,
                "neutral_harm": (
                    None if il is None or nl is None or il != gold else nl != gold
                ),
                "directed_harm": (
                    None if il is None or dl is None or il != gold else dl != gold
                ),
                "directed_target_adoption": (
                    None if dl is None or target is None else dl == target
                ),
            }
        )
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fmt(x):
    return "NA" if x is None else f"{100*x:.2f}\\%"


def _esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


def create_figures(result):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["openai_primary", "gemini_primary"]
    labels = ["OpenAI", "Gemini"]
    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.bar(
        x - w / 2,
        [result[m]["neutral_harmful_revision"] for m in models],
        w,
        label="Neutral",
    )
    ax.bar(
        x + w / 2,
        [result[m]["directed_harmful_revision"] for m in models],
        w,
        label="Directed",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Harmful-revision probability")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_harmful_rates.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    points = [result[m]["delta_harm"] for m in models] + [result["panel"]["delta_harm"]]
    cis = [result["bootstrap"]["models"][m]["ci95"] for m in models] + [
        result["bootstrap"]["panel"]["ci95"]
    ]
    y = np.arange(3)
    ax.errorbar(
        points,
        y,
        xerr=[
            [p - c[0] for p, c in zip(points, cis)],
            [c[1] - p for p, c in zip(points, cis)],
        ],
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, ["OpenAI", "Gemini", "Fixed panel"])
    ax.set_xlabel("Directed minus neutral harmful revision")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_delta_harm.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    vals = [
        [
            result[m]["initial_accuracy"],
            result[m]["neutral_final_accuracy"],
            result[m]["directed_final_accuracy"],
        ]
        for m in models
    ]
    xx = np.arange(3)
    ax.plot(xx, vals[0], "o-", label="OpenAI")
    ax.plot(xx, vals[1], "o-", label="Gemini")
    ax.set_xticks(xx, ["Initial", "Neutral final", "Directed final"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_accuracy_transitions.pdf")
    plt.close(fig)


def render_report(result, integrity):
    panel = result["panel"]
    lo, hi = result["bootstrap"]["panel"]["ci95"]
    half = (hi - lo) / 2
    if lo > 0:
        conclusion = "For this evidence-conditioned VitaminC sample and fixed model panel, an explicit incorrect alternative was associated with more harmful revision than neutral reconsideration."
    elif hi < 0:
        conclusion = "The directed condition showed an unexpected lower harmful-revision rate than neutral reconsideration; the prospective analysis and wording were retained."
    else:
        conclusion = "The study does not resolve the sign of the directed-minus-neutral harmful-revision difference. This is not evidence of no effect or equivalence."
    precision = (
        " Realized precision was poorer than prospectively intended because the interval half-width exceeded 7.5 percentage points."
        if half > 0.075
        else ""
    )
    rows = "\n".join(
        f"{name} & {result[m]['initially_correct_paired_n']} & {_fmt(result[m]['neutral_harmful_revision'])} & {_fmt(result[m]['directed_harmful_revision'])} & {_fmt(result[m]['delta_harm'])} \\\\"
        for m, name in (("openai_primary", "OpenAI"), ("gemini_primary", "Gemini"))
    )
    tex = (
        r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}\usepackage{booktabs}\usepackage{graphicx}\usepackage{hyperref}\usepackage{microtype}
\title{User-Directed Judgment Revision in Evidence-Conditioned Fact Verification}\author{Advanced Topics NLP Project}\date{September 2026}
\begin{document}\maketitle
\begin{abstract}We test whether explicitly suggesting the unique incorrect binary label causes additional harmful revision beyond neutral reconsideration. The prospectively frozen study uses 300 real-test VitaminC items and a fixed panel of two language models.\end{abstract}
\section{Dataset and design}The sample contains 300 unique leakage-controlled groups from the official VitaminC real-test SUPPORTS/REFUTES population, balanced 150/150 and stratified by evidence length. Each item was evaluated by OpenAI gpt-5.4-nano-2026-03-17 and Gemini gemini-3.5-flash-lite. Each valid initial answer generated independent neutral and directed sibling branches sharing the actual initial answer. The directed prompt proposed the unique non-initial binary label. This bundles label mention with user endorsement.
\section{Collection integrity}The frozen catalog contained 1,800 maximum scientific slots. Collection yielded """
        + str(integrity["committed"])
        + r""" committed observations and """
        + str(integrity["ineligible"])
        + r""" structurally ineligible branches, with zero unresolved or ambiguous slots. Invalid outputs were retained without scientific resampling.
\section{Primary result}
\begin{table}[h]\centering\begin{tabular}{lrrrr}\toprule Model & Eligible pairs & Neutral harm & Directed harm & Difference\\\midrule
"""
        + rows
        + r"""
\bottomrule\end{tabular}\caption{Harmful revision among initially correct valid paired trials.}\end{table}
The equally weighted fixed-panel estimate was """
        + _fmt(panel["delta_harm"])
        + r""" (item-cluster bootstrap 95\% CI: """
        + _fmt(lo)
        + r""" to """
        + _fmt(hi)
        + r"""). """
        + _esc(conclusion + precision)
        + r"""
\begin{figure}[h]\centering\includegraphics[width=.72\linewidth]{../figures/figure1_harmful_rates.pdf}\caption{Neutral and directed harmful-revision probabilities.}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=.72\linewidth]{../figures/figure2_delta_harm.pdf}\caption{Directed-minus-neutral harmful revision with prospective 95\% item-cluster bootstrap intervals.}\end{figure}
\section{Secondary outcomes}Initial accuracy, generic revision, resistance, final accuracy, accuracy change, productive correction, target adoption, and transition matrices are reported in the machine-readable and CSV tables. Productive correction is descriptive because its denominator depends on initially incorrect judgments. In this binary design, valid directed target adoption equals a directed flip.
\begin{figure}[h]\centering\includegraphics[width=.72\linewidth]{../figures/figure3_accuracy_transitions.pdf}\caption{Descriptive initial and final accuracy.}\end{figure}
\section{Sensitivity and interpretation}Model-specific exact McNemar descriptions compare discordant harmful-revision outcomes. Differences in model-specific statistical significance are not evidence of a model difference. Models form a fixed panel and are not sampled from a population of model families.
\section{Limitations}The study is observational across items and experimental only with respect to prompt condition. One English benchmark, one challenge wording, hosted models, one decoding configuration, finite model coverage, and possible training exposure limit generalization. The directed prompt does not isolate social influence from label priming. Evidence is treated as supplied context, not independently verified world truth.
\section{Connection to Study 1}Study 1 linked dense human disagreement in three-label NLI to revision. This second task asks a narrower factual-verification question with supplied evidence and binary alternatives. It tests the robustness of user-directed revision behavior across task structure rather than reusing human-disagreement predictors.
\section{Reproducibility}The sample, prompts, model configurations, parser, analysis, bootstrap seed (2026092502; 10,000 replicates), transport policy, raw durable records, and SHA-256 freeze manifest are retained with this report.
\end{document}
"""
    )
    (REPORTS / "SECOND_TASK_RESULTS_REPORT.tex").write_text(tex, encoding="utf-8")
    for _ in range(2):
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "SECOND_TASK_RESULTS_REPORT.tex",
            ],
            cwd=REPORTS,
            check=True,
            stdout=subprocess.DEVNULL,
        )


def run(db=LIVE_DB, synthetic=False):
    for p in (TABLES, FIGURES, MACHINE, REPORTS, ROOT / "logs/matplotlib"):
        p.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    state = json.loads(
        con.execute("SELECT value FROM meta WHERE key='run_state'").fetchone()[0]
    )
    counts = dict(con.execute("SELECT status,COUNT(*) FROM slots GROUP BY status"))
    amb = counts.get("AMBIGUOUS", 0)
    unresolved = sum(
        v for k, v in counts.items() if k not in ("COMMITTED", "INELIGIBLE")
    )
    integrity = {
        "run_state": state,
        "planned": con.execute("SELECT COUNT(*) FROM slots").fetchone()[0],
        "committed": counts.get("COMMITTED", 0),
        "ineligible": counts.get("INELIGIBLE", 0),
        "ambiguous": amb,
        "unresolved": unresolved,
        "duplicate_observations": con.execute(
            "SELECT COUNT(*) FROM (SELECT scientific_id FROM observations GROUP BY scientific_id HAVING COUNT(*)>1)"
        ).fetchone()[0],
        "attempts": con.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
        "replacements": con.execute(
            "SELECT COUNT(*) FROM attempts WHERE slot_attempt=2"
        ).fetchone()[0],
        "invalid_outputs_by_role": dict(
            con.execute(
                "SELECT role,COUNT(*) FROM observations WHERE parsed_status!='valid' GROUP BY role"
            )
        ),
        "attempt_failures_by_classification": dict(
            con.execute(
                "SELECT classification,COUNT(*) FROM attempts WHERE status='FAILED' GROUP BY classification"
            )
        ),
    }
    con.close()
    if not synthetic:
        if (
            integrity["planned"] != 1800
            or integrity["committed"] + integrity["ineligible"] != 1800
            or amb
            or unresolved
            or integrity["duplicate_observations"]
        ):
            raise RuntimeError("Reconciliation failed; analysis prohibited")
    rows = load_analysis_rows(db)
    summary = _summarize(rows)
    spec = load_freeze()["analysis"]
    boot = _bootstrap(rows, spec["bootstrap_replicates"], spec["bootstrap_seed"])
    summary["bootstrap"] = boot
    summary["mcnemar"] = {
        m: _mcnemar(rows, m) for m in ("openai_primary", "gemini_primary")
    }
    summary["integrity"] = integrity
    summary["generated_utc"] = utc_now()
    summary["synthetic"] = synthetic
    _write_csv(
        TABLES
        / ("synthetic_analysis_ready.csv" if synthetic else "analysis_ready.csv"),
        rows,
    )
    if not synthetic:
        secondary = []
        for model, name in (("openai_primary", "OpenAI"), ("gemini_primary", "Gemini")):
            for metric, value in summary[model].items():
                if metric.endswith("_n") or metric == "items":
                    continue
                secondary.append(
                    {
                        "model": name,
                        "metric": metric,
                        "value": value,
                        "denominator_initial_valid": summary[model]["valid_initial_n"],
                        "denominator_paired_valid": summary[model]["paired_valid_n"],
                        "denominator_initially_correct_paired": summary[model][
                            "initially_correct_paired_n"
                        ],
                        "denominator_initially_incorrect_paired": summary[model][
                            "initially_incorrect_paired_n"
                        ],
                    }
                )
        _write_csv(TABLES / "secondary_outcomes.csv", secondary)
        transitions = []
        for r in rows:
            if not r["initial_valid"]:
                continue
            for condition in ("neutral", "directed"):
                final = r[f"{condition}_label"]
                if final is not None:
                    transitions.append(
                        {
                            "model": r["model_key"],
                            "condition": condition,
                            "initial_label": r["initial_label"],
                            "final_label": final,
                            "count": 1,
                        }
                    )
        aggregated = Counter(
            (x["model"], x["condition"], x["initial_label"], x["final_label"])
            for x in transitions
        )
        _write_csv(
            TABLES / "transition_matrices.csv",
            [
                {
                    "model": k[0],
                    "condition": k[1],
                    "initial_label": k[2],
                    "final_label": k[3],
                    "count": v,
                }
                for k, v in sorted(aggregated.items())
            ],
        )
        quality = []
        for role, n in sorted(integrity["invalid_outputs_by_role"].items()):
            quality.append(
                {"category": "invalid_scientific_output", "detail": role, "count": n}
            )
        for cls, n in sorted(integrity["attempt_failures_by_classification"].items()):
            quality.append(
                {
                    "category": "operational_attempt_failure",
                    "detail": cls or "unclassified",
                    "count": n,
                }
            )
        quality.extend(
            [
                {
                    "category": "collection",
                    "detail": "committed",
                    "count": integrity["committed"],
                },
                {
                    "category": "collection",
                    "detail": "structurally_ineligible",
                    "count": integrity["ineligible"],
                },
                {
                    "category": "collection",
                    "detail": "transport_replacements",
                    "count": integrity["replacements"],
                },
            ]
        )
        _write_csv(TABLES / "collection_quality.csv", quality)
        summary["transition_matrices"] = [
            {
                "model": k[0],
                "condition": k[1],
                "initial_label": k[2],
                "final_label": k[3],
                "count": v,
            }
            for k, v in sorted(aggregated.items())
        ]
        (MACHINE / "SECOND_TASK_RESULTS.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        create_figures(summary)
        render_report(summary, integrity)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
