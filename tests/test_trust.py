from maro.tools.trust import classify, classify_source


def test_wikipedia_always_trusted():
    assert classify_source("https://en.wikipedia.org/wiki/Foo") == "trusted"
    assert classify("https://en.wikipedia.org/wiki/Foo").note == "Wikipedia"


def test_social_media_untrusted():
    assert classify_source("https://x.com/user/status/123") == "untrusted"
    assert classify_source("https://someone.wordpress.com/rant") == "untrusted"
    assert classify_source("https://t.me/channel/1") == "untrusted"


def test_mainstream_news_is_neutral():
    # TRUSTED_DOMAINS hand-curated list was removed — we rely entirely on
    # Iffy+/MBFC data. Mainstream outlets not in Iffy+ are now 'neutral',
    # not 'trusted'.
    assert classify_source("https://www.reuters.com/world/foo") == "neutral"
    assert classify_source("https://www.bbc.co.uk/news/x") == "neutral"


def test_neutral():
    assert classify_source("https://example.com/page") == "neutral"
    assert classify_source("https://some-random-site.io/news") == "neutral"


def test_malformed():
    assert classify_source("") == "neutral"
    assert classify_source("not a url") == "neutral"


def test_iffy_all_three_mbfc_labels_attached():
    """An Iffy+ entry should yield an untrusted tier with all 3 MBFC fields populated."""
    r = classify("https://www.infowars.com/story")
    # This test requires data/iffy_untrusted.json to exist (checked in).
    assert r.tier == "untrusted"
    assert "Iffy+" in r.note
    # All three MBFC dimensions should be populated for a well-rated Iffy entry.
    assert r.mbfc_fact and r.mbfc_fact_label
    assert r.mbfc_bias and r.mbfc_bias_label
    assert r.mbfc_cred and r.mbfc_cred_label
