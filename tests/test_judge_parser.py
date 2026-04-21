from maro.agents.judge import _parse_judge_output


def test_clean_output():
    r = _parse_judge_output("reasoning: looks suspicious\njudgment: 1")
    assert r.label == 1
    assert "suspicious" in r.reasoning


def test_real_label():
    r = _parse_judge_output("reasoning: aligns with trusted sources\njudgment: 0")
    assert r.label == 0


def test_fallback_when_no_label_line():
    r = _parse_judge_output("some noise 1 more noise")
    assert r.label == 1
