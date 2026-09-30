# Human disagreement and judgment revision in language models

**[Read the signed paper](reports/TERM_PAPER_SIGNED_FINAL.pdf)** · **[Reproduce the results](#reproduce-the-results)** · **[Browse the reported results](#what-is-included)**

## Abstract

Language models sometimes change a correct or defensible answer simply because a user asks again. A changed answer can look like agreement with the user, but it may also reflect uncertainty in the model or ambiguity in the example. This project studies those possibilities in natural language inference and evidence-based fact verification. Study 1 combines 300 ChaosNLI examples, 100 human judgments per example, and repeated classifications from two language models. Study 2 follows initially correct VitaminC answers and compares a simple request to reconsider with a follow-up that suggests the wrong label. The results show that asking again is already a meaningful intervention: models often changed their answers even when no alternative was suggested. In the NLI study, changes were more common when human annotators disagreed, and models strongly preferred alternatives that had greater human support. In the fact-verification study, a neutral request sometimes turned a correct answer into an incorrect one, while the wrong suggestion did not have a consistent additional harmful effect across the two models. These findings show why evaluations should separate ordinary answer instability, movement toward a suggested answer, and changes in answer quality.

## Main findings

| Finding | Result |
|---|---|
| Asking again often changed the answer | In Study 1, models moved to another valid label in **30.4%** of neutral follow-ups (363 of 1,196). |
| Human disagreement identified less stable examples | Models were more likely to adopt another label when annotators were more divided about the NLI example. |
| The direction of disagreement mattered | The alternative with greater human support was selected in **20.7%** of eligible pairs, compared with **3.7%** for the lower-supported alternative. |
| Reconsideration could damage a correct answer | In Study 2, neutral reconsideration produced harmful revision in **14.8%** of initially correct OpenAI cases and **5.2%** of Gemini cases. |
| A wrong suggestion did not have one consistent extra effect | Directed harmful revision was **8.8%** for OpenAI and **6.5%** for Gemini. The two model patterns differed, and the combined interval included zero. |

The experiments used `gpt-5.4-nano-2026-03-17` and `gemini-3.5-flash-lite`. The paper gives the complete statistical results, limitations, and interpretation.

## What is included

```text
study1/
  analysis/          Study 1 analysis
  data/              compressed database of collected responses
  frozen/            selected examples, prompts, model settings, and request lists
  reconciliation/    response-count checks
  results/           reported tables, figures, and machine-readable results
study2/
  code/              Study 2 analysis and shared functions
  data/              compressed response database and VitaminC license
  freeze/             selected examples, prompts, model settings, and analysis plan
  reconciliation/    response-count checks
  analysis/           reported tables, figures, and machine-readable results
reports/
  TERM_PAPER_SIGNED_FINAL.pdf
docs/
  FAILURES.md         collection problems and how they were handled
scripts/
  unpack_data.py      restores the two response databases
  verify_release.py   checks the files and databases
  reproduce.py        reruns the analyses
```

The directories named `frozen` and `freeze` contain the versions of the study materials that were fixed before the results were analyzed. The names are retained because the analysis code refers to these paths.

## Setup

Python 3.12 or newer is recommended. A TeX installation with `pdflatex` is needed to rebuild the Study 1 technical report.

```bash
git clone https://github.com/fayizck/human-disagreement-and-llm-revision.git
cd human-disagreement-and-llm-revision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/unpack_data.py
python scripts/verify_release.py
```

The analyses run from the collected data included in this repository.

## Reproduce the results

Run both studies:

```bash
python scripts/reproduce.py --study all
```

Run one study:

```bash
python scripts/reproduce.py --study study1
python scripts/reproduce.py --study study2
```

The new files are written to `reproduced/`. The original response databases and reported results are not overwritten. Study 1 uses 2,000 bootstrap samples and cross-validation, so it may take several minutes. Study 2 uses 10,000 bootstrap samples.

To check the repository without rerunning the analyses:

```bash
python scripts/verify_release.py
python -m pytest -q
```

## Data collection summary

| Study | Responses planned | Responses recorded | Follow-ups not applicable | Requests sent | Requests retried |
|---|---:|---:|---:|---:|---:|
| Study 1 | 33,000 | 32,992 | 8 | 32,994 | 2 |
| Study 2 | 1,800 | 1,776 | 24 | 1,776 | 0 |

A follow-up could not be asked when the initial response did not contain one of the required labels. These initial responses were kept rather than replaced. The two retried requests in Study 1 failed for technical reasons and were repeated once with the same content and settings. More detail is available in [`docs/FAILURES.md`](docs/FAILURES.md).

## Data and licenses

The model responses and the dataset excerpts used by the analyses are included. The source datasets remain subject to their original terms:

- ChaosNLI, SNLI, and MNLI were used in Study 1.
- VitaminC was used in Study 2. Its license is included at [`study2/data/DATA_LICENSE`](study2/data/DATA_LICENSE).

The repository's MIT license applies to the original code. It does not replace the licenses or terms that apply to the datasets and model responses.
