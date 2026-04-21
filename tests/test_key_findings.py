from maro.agents.base import parse_key_findings


def test_basic():
    text = """
    Some analysis prose here.

    KEY FINDINGS:
    - Tone: sensationalist / alarmist
    - Style: clickbait
    - Red flags: urgent CTAs, unverifiable superlatives

    Implicit verdict: FAKE
    """
    findings = parse_key_findings(text)
    assert findings["Tone"] == "sensationalist / alarmist"
    assert findings["Style"] == "clickbait"
    assert "urgent CTAs" in findings["Red flags"]


def test_missing_block():
    assert parse_key_findings("no findings here\n\nImplicit verdict: REAL") == {}


def test_case_insensitive_header():
    text = "key findings:\n- Supported: 3\n- Contradicted: 1\n\nImplicit verdict: REAL"
    findings = parse_key_findings(text)
    assert findings["Supported"] == "3"
    assert findings["Contradicted"] == "1"


def test_multiline_values_kept_on_one_line():
    text = "KEY FINDINGS:\n- A: x\n- B: y\n\n"
    findings = parse_key_findings(text)
    assert findings == {"A": "x", "B": "y"}
