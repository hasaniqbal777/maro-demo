from maro.tools.trust import classify_source


def test_trusted_domains():
    assert classify_source("https://www.reuters.com/world/foo") == "trusted"
    assert classify_source("https://en.wikipedia.org/wiki/Foo") == "trusted"
    assert classify_source("https://www.bbc.co.uk/news/x") == "trusted"
    assert classify_source("https://subdomain.nytimes.com/a") == "trusted"


def test_untrusted():
    assert classify_source("https://x.com/user/status/123") == "untrusted"
    assert classify_source("https://someone.wordpress.com/rant") == "untrusted"
    assert classify_source("https://t.me/channel/1") == "untrusted"


def test_neutral():
    assert classify_source("https://example.com/page") == "neutral"
    assert classify_source("https://some-random-site.io/news") == "neutral"


def test_malformed():
    assert classify_source("") == "neutral"
    assert classify_source("not a url") == "neutral"
