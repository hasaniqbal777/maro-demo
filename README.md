---
title: MARO Demo
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.56.0
app_file: demo/app.py
pinned: false
license: mit
python_version: "3.11"
short_description: Multi-agent misinformation detection (EMNLP 2025 MARO)
---

# MARO — Multi-Agent Misinformation Detection

[![Open in Spaces](https://img.shields.io/badge/🤗_Hugging_Face-Open_in_Spaces-yellow)](https://huggingface.co/spaces/hasaniqbal777/maro-demo)

**Live demo:** [huggingface.co/spaces/hasaniqbal777/maro-demo](https://huggingface.co/spaces/hasaniqbal777/maro-demo)

Course-project replication of *[A Multi-Agent Framework with Automated Decision
Rule Optimization for Cross-Domain Misinformation Detection](https://aclanthology.org/2025.emnlp-main.291.pdf)*
(Hui Li et al., EMNLP 2025) — with a live **Streamlit agent-I/O inspector** and
two extensions beyond the paper.

## What's faithful to the paper

Everything in §2.2 of the paper is implemented, in the cross-event variant
from Appendix G.4 (PHEME, 5 events):

- **Multi-Dimensional Analysis Module** — Linguistic Feature, Comment, and a
  Fact-Checking Agent Group (Fact-Questioning + Fact-Checking using Serper
  for Google search and the `wikipedia` package). System prompts are verbatim
  from Appendix B.1.
- **Question-reflection mechanism** — the Questioning Agent reviews each
  expert's initial report and generates follow-up questions; the experts then
  refine their analyses.
- **Decision Rule Optimization Module (Algorithm 1)** — seed rule r₀,
  Optimization Agent proposes candidate rules, Judge evaluates on cross-event
  validation tasks, top-K pairs feed the next iteration, early-stop after
  `N_att` failures. Prompts from Appendix B.2.
- **Cross-event protocol (Appendix G.4)** — for each target event, rules are
  optimized on validation tasks drawn from the *other 4* events. At inference
  on the target, the Judge applies each of the top-K rules and majority-votes.
- **Per-event rules files (no leak)** — `data/rules_<event>.json` is the
  output of one target-event run; the demo and evaluation scripts load the
  file matching the example's event, so rules never see their own event.

## Two extensions (course-project additions)

1. **Evidence-source trust weighting** — every Google/Serper hit is tagged
   `TRUSTED` (Wikipedia only) / `UNTRUSTED` (Iffy+ Index match or
   social-media pattern) / `NEUTRAL` (everything else), with classification
   driven entirely by externally-validated data — no hand-curated whitelist.
   When a hit appears in the [Iffy+ Mis/Disinformation Index](https://iffy.news/index/)
   the Fact-Checking Agent (and the demo UI) see all three MBFC labels as
   separate fields: **Factual Reporting**, **Bias**, **Credibility**, plus
   Iffy's composite score. This targets the paper's own Limitation #2
   (search-engine poisoning) and lets the agent cite the specific reason a
   source is being downweighted, e.g. *"MBFC rates infowars.com as Very Low
   Factual, Conspiracy-Pseudoscience bias, Low Credibility."*
2. **Confidence + agent-disagreement analysis** — the Judge's majority vote
   over the top-K rules produces a calibrated confidence score, and each
   expert agent emits an implicit verdict so inter-agent disagreement is
   surfaced in a diagnostics table.

## Architecture at a glance

```
┌─ Module 1: Multi-Dimensional Analysis ──────────────────────────┐
│                                                                 │
│   news + comments                                               │
│       ├─► LinguisticFeatureAgent ──► R_l                        │
│       ├─► CommentAnalysisAgent   ──► R_c                        │
│       └─► FactQuestioningAgent → Serper + Wikipedia ──► Fact-   │
│           CheckingAgent (with source trust tiers)    ──► R_f    │
│                                                                 │
│       QuestioningAgent re-asks each expert → experts refine     │
└─────────────────────────────────────────────────────────────────┘
                                   │
┌─ Module 2: Decision Rule Optimization (offline per target) ─────┐
│   source-domain validation tasks (paper Appendix G.4)           │
│   DecisionRuleOptimizationAgent ⇄ JudgeAgent → top-K rules      │
└─────────────────────────────────────────────────────────────────┘
                                   │
          JudgeAgent applies each top-K rule → majority vote
                                   │
          Confidence = votes_for_majority / K  (extension #2)
```

## Running locally

```bash
uv sync
# 1. Download the PHEME dataset (~46 MB compressed, one-time)
uv run python scripts/download_pheme.py

# 2. Optimize rules for one held-out event (~$3, ~5 min on gpt-5.4-mini)
uv run python scripts/run_optimization.py --target-event charliehebdo

# 3. Launch the demo — banner flips to "Partial MARO mode (1/5)"
uv run streamlit run demo/app.py
```

Set `OPENAI_API_KEY` and `SERPER_API_KEY` in `.env`.

### Optional — other target events / full evaluation

```bash
# All 5 target events (~$25, ~2 hr)
uv run python scripts/run_optimization.py --target-event all

# Held-out accuracy + F1 per event, with trust-weighting ablation
uv run python scripts/run_evaluation.py --ablation

# Presentation-friendly optimization trajectory table for slides
uv run python scripts/show_trajectory.py --target-event charliehebdo
```

## Demo highlights

Launch `uv run streamlit run demo/app.py` and pick one of the 6 pre-canned
examples (real PHEME tweets with real replies). What you see:

- **Hero verdict card** — big `REAL` / `FAKE` label, SVG confidence ring, and
  a ✓/✗ banner comparing against PHEME's ground-truth label.
- **4 agent cards** (Linguistic, Comment, Fact-Check, Judge) with structured
  key-findings tables parsed from each agent's output. Each card has inline
  expanders for reflection follow-ups, evidence used (trust-tagged), and the
  full agent report.
- **Pipeline strip** — 9 stages grouped into *Analyze → Reflect & revise →
  Judge*, lighting up pending → running (blue) → complete (green) as each
  stage finishes. Driven by a progress callback from `pipeline.analyze()`.
- **Evidence cards** — each Serper/Wikipedia hit shown with colored
  `[TRUSTED]` / `[NEUTRAL]` / `[UNTRUSTED]` pill and clickable source URL.
- **Diagnostics** — per-agent implicit verdicts, per-rule votes, raw LLM
  calls (system + user prompts, responses, tool calls, latencies).
- **Run stats** — total LLM calls, tokens, wall-clock latency.

## Performance

The inference path is parallelized via `ThreadPoolExecutor` — all independent
LLM calls (Linguistic + Comment + FactQuestioning, the three reflection
reviews, the three expert refinements, the top-K Judge votes) and all tool
calls (Serper + Wikipedia per question) run concurrently up to
`MAX_PARALLEL=8`. A typical demo run on `gpt-5.4-mini` takes ~10-12 seconds
wall time, down from ~35 seconds sequential.

Timeouts are in place at three layers (per-request OpenAI timeout, socket
timeout for Wikipedia, per-item timeout in the optimization pre-compute
phase) so a single hung call can't stall a batch.

## Running on HuggingFace Spaces

This repo is configured as a Streamlit Space (see frontmatter) with an
automated deploy via GitHub Actions on every push to `main`.

**One-time setup:**

1. Create a new Space on HuggingFace (SDK = **Streamlit**).
2. In the Space's *Settings → Variables and secrets*, add two **Space secrets**:
   - `OPENAI_API_KEY`
   - `SERPER_API_KEY`
3. In this GitHub repo's *Settings → Secrets and variables → Actions*, add
   three **repository secrets**:
   - `HF_TOKEN` — a HuggingFace access token with *write* scope
     (profile → Access Tokens → New token → Write)
   - `HF_USERNAME` — your HF username (owner of the Space)
   - `HF_SPACE_NAME` — the Space's slug (without the username prefix)
4. Push to `main`. The workflow at [.github/workflows/deploy-hf-space.yml](.github/workflows/deploy-hf-space.yml)
   force-pushes the repo to the Space's git remote, the Space rebuilds from
   `requirements.txt`, and `demo/app.py` launches.

Per-event rule files `data/rules_<event>.json` are checked in, so the Space
boots directly into MARO mode for any event whose optimization has been run
locally. Events without a rules file fall back to the seed r₀ (warning banner).

## Scale knobs

Defaults in `src/maro/config.py` are aggressively scaled for a cheap
course-project run. Paper values are 500/500; ours are:

| Hyperparameter | Default | Paper |
|---|---|---|
| `N_ITER` | 5 | 500 |
| `N_ATT` | 3 | 10 |
| `N_VAL_TASKS` | 12 | 500 |
| `TOP_K_RULES` | 3 | 3 |
| `MAX_PARALLEL` | 8 | n/a |

All four are also CLI flags on `scripts/run_optimization.py`
(`--iters`, `--attempts`, `--tasks`, `--limit-per-event`,
`--item-timeout`) so you can scale up per-run without editing code.

## Repository layout

```
src/maro/
  agents/            # 7 agents from paper §2.2 + implicit-verdict parser
  prompts/           # system prompts verbatim from Appendix B
  tools/
    serper.py        # Google search
    wikipedia.py     # Wikipedia lookup
    trust.py         # URL → trust tier (extension #1)
  dataset/
    pheme_loader.py  # load_source(target) / load_target(target)
    validation_tasks.py
  pipeline.py        # multi-dim analysis + inference (parallelized)
  optimization.py    # Algorithm 1
  trace.py           # per-agent I/O recorder (contextvar-based)
  config.py          # hyperparameters, trusted-domain whitelist, demo examples
  llm.py             # OpenAI wrapper with retry + timeout
demo/app.py          # Streamlit UI
scripts/
  download_pheme.py
  run_inference.py
  run_optimization.py
  run_evaluation.py
  show_trajectory.py # presentation-friendly trajectory renderer
tests/               # 17 pytest tests (mocked LLM end-to-end)
data/                # git-ignored: PHEME tree, rules_<event>.json, results/
```

## Third-party data credits

- **[Iffy+ Mis/Disinformation Index](https://iffy.news/index/)** (Barrett Golding) —
  2,000+ low-credibility domains with MBFC Factual/Bias/Credibility ratings.
  Released under MIT + CC BY 4.0; we redistribute a snapshot at
  `data/iffy_untrusted.json` (refresh with `scripts/refresh_iffy_untrusted.py`).
- **[Media Bias/Fact Check](https://mediabiasfactcheck.com/)** ratings are
  surfaced via Iffy+ and exposed in the Fact-Checking Agent's evidence notes.

## Dataset

[PHEME dataset for Rumour Detection and Veracity Classification](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)
(Kochkina, Liakata, Zubiaga — the canonical 5-event set the paper uses in
Appendix G.4). The download script pulls `PHEME_veracity.tar.bz2` from
Figshare (~46 MB) and extracts the 5 events we use:

- `charliehebdo` · `ferguson` · `germanwings-crash` · `ottawashooting` · `sydneysiege`

## Known limitations

- **Small scale** — `N_ITER=5`, `N_VAL_TASKS=12` gives noisy accuracy
  (±15 points from a single flipped answer on a 12-task validation set).
  Comparable to paper numbers requires the paper's 500/500 scale, which we
  don't run for cost reasons.
- **Seed-rule mode** — if no `data/rules_<event>.json` files exist, the demo
  falls back to the seed rule r₀ and shows a yellow warning banner. This is
  the starting point of Algorithm 1, not full MARO.
- **Demo's pre-canned examples are real PHEME tweets** (verbatim from the
  downloaded dataset). Some "rumours" were later confirmed and some
  "confirmed" items were later disputed — PHEME labels reflect the
  epistemic state at posting time, not hindsight.

## Citation

This project is a faithful replication of the following paper — all credit for
the MARO framework, its agents, the decision-rule optimization algorithm, and
the experimental protocol belongs to the original authors. If you use or
extend this work, please cite the original paper:

```bibtex
@inproceedings{li-etal-2025-multi-agent,
    title = "A Multi-Agent Framework with Automated Decision Rule Optimization for Cross-Domain Misinformation Detection",
    author = "Li, Hui  and
      Wang, Ante  and
      Li, Kunquan  and
      Wang, Zhihao  and
      Zhang, Liang  and
      Qiu, Delai  and
      Liu, Qingsong  and
      Su, Jinsong",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.291/",
    doi = "10.18653/v1/2025.emnlp-main.291",
    pages = "5709--5725",
    ISBN = "979-8-89176-332-6",
}
```

**Paper:** [aclanthology.org/2025.emnlp-main.291](https://aclanthology.org/2025.emnlp-main.291/)  ·  **DOI:** [10.18653/v1/2025.emnlp-main.291](https://doi.org/10.18653/v1/2025.emnlp-main.291)

## License

This replication code is released under the MIT license. The MARO framework
itself is the intellectual contribution of the original paper's authors.
