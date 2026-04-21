from maro.agents.base import extract_implicit_verdict


def test_real():
    assert extract_implicit_verdict("lots of text\nImplicit verdict: REAL") == "REAL"


def test_fake_case_insensitive():
    assert extract_implicit_verdict("Implicit Verdict : fake") == "FAKE"


def test_absent():
    assert extract_implicit_verdict("no verdict line here") == "UNKNOWN"
