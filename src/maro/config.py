"""Centralised configuration: model list, hyperparameters, trusted-source lookup."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

AVAILABLE_MODELS: list[str] = [
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.4-pro",
]
DEFAULT_MODEL = "gpt-5.4-mini"

# Paper Algorithm 1 hyperparameters. Scaled small for a course project.
# The paper runs N_iter=500, N_val_tasks=500 on Weibo21. On PHEME at this
# scale, one target event is ~$5 on gpt-5.4-mini; wall-clock is kept in
# check by running OpenAI calls in parallel (see MAX_PARALLEL).
N_ITER = 5
N_ATT = 3
N_VAL_TASKS = 12
TOP_K_RULES = 3

# Max concurrent OpenAI requests during optimization's pre-compute analysis
# phase and per-iteration Judge evaluation. gpt-5.4-mini handles this fine.
MAX_PARALLEL = 8

# Paper §3.1 temperatures.
TEMP_OPTIMIZER = 1.0
TEMP_JUDGE = 0.0
TEMP_ANALYSIS = 0.3  # expert agents — mild creativity, mostly deterministic

# Trust-tier lookup for improvement #1 (evidence-source trust weighting).
TRUSTED_DOMAINS: frozenset[str] = frozenset({
    # Wire services
    "reuters.com", "apnews.com", "afp.com",
    # Mainstream newspapers
    "nytimes.com", "washingtonpost.com", "wsj.com", "theguardian.com",
    "bbc.com", "bbc.co.uk", "ft.com", "economist.com", "bloomberg.com",
    "npr.org", "cnn.com", "abcnews.go.com", "nbcnews.com", "cbsnews.com",
    # Science / health / gov
    "nature.com", "science.org", "nejm.org", "thelancet.com",
    "who.int", "cdc.gov", "nih.gov", "fda.gov", "europa.eu", "gov.uk",
    # Reference
    "wikipedia.org", "britannica.com",
    # Fact-checking
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org",
})

# Any URL whose host contains one of these substrings is tagged "untrusted".
UNTRUSTED_SUBSTRINGS: tuple[str, ...] = (
    "blogspot.", "wordpress.", "medium.com", "substack.com",
    "twitter.com", "x.com", "facebook.com", "t.me", "tiktok.com",
    "reddit.com", "4chan.", "truthsocial.", "rumble.com",
)

PHEME_EVENTS: tuple[str, ...] = (
    "charliehebdo",
    "sydneysiege",
    "ferguson",
    "ottawashooting",
    "germanwings-crash",
)

# Pre-canned demo examples drawn from the real PHEME dataset (verbatim source
# tweets + top replies). Each entry: (title, event, news_text, comments, label_is_rumour).
# Labels match PHEME's annotation: `rumour` = claim whose veracity was in question
# at the time, `non-rumour` = well-established reporting. Some "rumours" were later
# confirmed, some debunked — PHEME captures the epistemic state at posting time.
DEMO_EXAMPLES: tuple[tuple[str, str, str, tuple[str, ...], bool], ...] = (
    (
        "Charlie Hebdo — gunmen picked journalists by name (RUMOUR)",
        "charliehebdo",
        "Charlie Hebdo massacre in Paris carried out with military precision - gunmen appeared to seek out journalists by name, police suggest.",
        (
            "@peterallenparis @Beltrew How many of these psychopaths are out there amongst peaceful French ppl? And is it that easy to get an AK47??",
            "@peterallenparis @rascouet These weren't psychopaths. They were all good Muslims, defending the honor of their prophet. Europe be warned.",
            "@peterallenparis @Lemnoc55 Wonder if this will change media attitudes to PEGIDA?",
            "@MaguidhirP @peterallenparis I wager it won't - time for a revolution.",
        ),
        True,
    ),
    (
        "Charlie Hebdo — police close in on suspects (CONFIRMED)",
        "charliehebdo",
        "IN PICTURES: French cops close in on suspected #CharlieHebdo terrorists outside Paris http://t.co/YioY54CqRi http://t.co/IGPF43HRsy",
        (
            "@NBCNews \n#StopKillingInnocentPeople\n#FreePalestine\n#freeBurma\n#FreeSyria \n#FreeKashmir\nhttp://t.co/ggBNiIa9fq",
            "@NBCNews They are not #CharlieHebdo terrorists, they are #IslamicTerrorists and @NBC is not honestly reporting it. #RespectAllMuslims?",
            "@NBCNews PICTURES: French cops CLOSE IN on suspected #CharlieHebdo terrorists outside #Paris!!\nhttp://t.co/WiGT4JsSOK http://t.co/v2XLk3X1ln",
            "@genlady9 @NBCNews Innocent people take hostages?",
        ),
        False,
    ),
    (
        "Sydney siege — ISIS flag on display (RUMOUR)",
        "sydneysiege",
        "Sydney cafe siege: Two gunmen and up to a dozen hostages inside the cafe under siege in Martin Place, Sydney. ISIS flags remain in display.",
        (
            "@Y7News hope all hostages come out safe 💖",
            "@Y7News @Channel7 fear indoctrination #MSM",
            "@Y7News sending my prayers , I am speechless",
            "@Y7News @Farmhousehome that's awful",
        ),
        True,
    ),
    (
        "Sydney siege — thoughts with hostages (CONFIRMED)",
        "sydneysiege",
        "My thoughts are with the hostages, their families and everyone in Sydney right now.",
        (
            "@5SOS_Updates what happens",
            "@5SOS_Updates People are being help hostage in a Sydney Cafe @Emmjohnsonn",
            "@5SOS_Updates what happened? ???",
            "@5SOS_Updates wait what happened?",
        ),
        False,
    ),
    (
        "Ferguson — police chief on blocking traffic (RUMOUR)",
        "ferguson",
        "Q: \"why did he stop Michael Brown?\" \n\n#Ferguson Police Chief: \"because they were walking down the middle of the street blocking traffic\"",
        (
            "@No_Cut_Card blocking traffic on an empty street inside a neighborhood...got it",
            "right. RT @Domo_HTTR: @No_Cut_Card blocking traffic on an empty street inside a neighborhood...got it",
            "!!!!!!!!!! RT @No_Cut_Card right. RT @Domo_HTTR: @No_Cut_Card blocking traffic on an empty street inside a neighborhood...got it",
            "@Domo_HTTR @No_Cut_Card that is a pretty good reason to shoot some one",
        ),
        True,
    ),
    (
        "Ottawa — Obama briefed on the situation (CONFIRMED)",
        "ottawashooting",
        "BREAKING: President Obama has been briefed on the situation in Ottawa, White House official says. http://t.co/qsAnGNqBEw",
        (
            "@cnni Is he sending precautionary ebola?",
            "“@cnni: BREAKING: President Obama has been briefed on the situation in Ottawa http://t.co/jNdr0m67RS”",
            "“@cnni: BREAKING: President Obama has been briefed on the situation in Ottawa, White House official says.” // Pelo Twitter, igual nóis.",
            "@cnni WHOO HOO!  Obama has been briefed!!!! That makes me feel better!!!",
        ),
        False,
    ),
)

# Seed decision rule r_0.
#
# Note: the paper's Table 21 best seed (68.39% acc) only flags "fake" when
# claims CONTRADICT trusted evidence — which mis-classifies unverifiable
# rumours (e.g. "police sources say…") as REAL. For demo purposes we seed with
# a more skeptical rule that treats unverified + commenter-challenged + sensational
# language as red flags. An optimization run (scripts/run_optimization.py) will
# replace this with the top-K rules in data/rules.json.
# Seed r_0 — intentionally simple and generic, matching the paper's notion
# of a "manually-defined initial decision rule" that the Decision Rule
# Optimization Agent (Algorithm 1) then improves upon iteratively.
# Running scripts/run_optimization.py generates top-K optimized rules into
# data/rules.json; at that point this seed is no longer used at inference.
INITIAL_DECISION_RULE = (
    "Based on the multi-dimensional analysis report, determine whether the "
    "news is misinformation. If the linguistic, comment, and fact-checking "
    "analyses collectively suggest the news is untrustworthy, output \"1\" "
    "(fake); if they suggest it is trustworthy, output \"0\" (real).\n"
    "Output requirements: - Output format: judgment: <'1' represents "
    "fake-news, '0' represents real-news>"
)
