"""Live-channel retrieval tests — requests fully mocked, NEVER hits the network.

``compass.agents.retrieval.requests.get`` and ``...retrieval.time.sleep`` are
patched for every test (an autouse guard turns any unmocked GET into a test
failure). Covers the never-raise/degrade-to-empty contract of the live
channels, Atom parsing, and the scout-side live funnel helpers with the
retrieval functions stubbed in the ``compass.agents.scout`` namespace.
"""

from __future__ import annotations

import asyncio

import pytest
import requests

import compass.agents.retrieval as retrieval
import compass.agents.scout as scout
from compass.agents.retrieval import (
    RETRY_AFTER_CAP,
    PaperStore,
    TfidfEmbedder,
    _get_with_retry,
    _parse_arxiv_atom,
    _parse_retry_after,
    _retry_after_seconds,
    arxiv_search,
    paper_finder_search,
    s2_citing_papers,
    semantic_scholar_search,
)

# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, text="", json_data=None, json_error=False,
                 headers=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self._json_error = json_error
        if headers is not None:
            self.headers = headers

    def json(self):
        if self._json_error:
            raise ValueError("response body is not JSON")
        return self._json_data


@pytest.fixture(autouse=True)
def sleep_calls(monkeypatch):
    """Record (and neutralize) retry backoff sleeps."""
    calls: list[float] = []
    monkeypatch.setattr(retrieval.time, "sleep", calls.append)
    return calls


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any GET that a test forgot to script is a hard failure, not a request."""

    def _blocked(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(retrieval.requests, "get", _blocked)


@pytest.fixture
def fake_get(monkeypatch, _no_network):
    """``fake_get(*outcomes)`` scripts requests.get; returns the call recorder.

    Each outcome is either a FakeResponse to return or an exception to raise.
    """
    state = {"outcomes": [], "calls": []}

    def _get(url, params=None, timeout=None, headers=None):
        state["calls"].append({"url": url, "params": params, "timeout": timeout,
                               "headers": headers})
        if not state["outcomes"]:
            raise AssertionError("requests.get called more times than scripted")
        outcome = state["outcomes"].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(retrieval.requests, "get", _get)

    def script(*outcomes):
        state["outcomes"] = list(outcomes)
        return state

    return script


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    """Disable the S2/arXiv throttles by default so existing tests don't record
    their sleeps; each throttle's own test re-enables it explicitly."""
    monkeypatch.setattr(retrieval, "_S2_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(retrieval, "_s2_last_call", 0.0)
    monkeypatch.setattr(retrieval, "_ARXIV_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(retrieval, "_arxiv_last_call", 0.0)
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.delenv("ASTA_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# _get_with_retry: one retry on 429/network errors, never raises
# ---------------------------------------------------------------------------


def test_429_then_200_retries_exactly_once(fake_get, sleep_calls):
    ok = FakeResponse(200, text="payload")
    state = fake_get(FakeResponse(429), ok)
    resp = _get_with_retry("http://api.test/x", {"q": 1}, backoff=2.0)
    assert resp is ok
    assert len(state["calls"]) == 2
    assert sleep_calls == [2.0], "exactly one backoff sleep before the retry"


def test_persistent_429_returns_none(fake_get, sleep_calls):
    state = fake_get(FakeResponse(429), FakeResponse(429))
    assert _get_with_retry("http://api.test/x", {}) is None
    assert len(state["calls"]) == 2  # initial + single retry, then give up
    assert len(sleep_calls) == 1


def test_500_returns_none_without_retry(fake_get, sleep_calls):
    state = fake_get(FakeResponse(500))
    assert _get_with_retry("http://api.test/x", {}) is None
    assert len(state["calls"]) == 1, "non-429 HTTP errors must not be retried"
    assert sleep_calls == []


@pytest.mark.parametrize(
    "exc_type", [requests.Timeout, requests.ConnectionError], ids=["timeout", "connerr"]
)
def test_network_exceptions_degrade_to_none_never_raise(fake_get, sleep_calls, exc_type):
    state = fake_get(exc_type("boom"), exc_type("boom"))
    assert _get_with_retry("http://api.test/x", {}) is None  # must not raise
    assert len(state["calls"]) == 2  # one retry after the first exception
    assert len(sleep_calls) == 1


# ---------------------------------------------------------------------------
# _get_with_retry: Retry-After is honoured (capped); the host throttle runs
# before every attempt, so a retry never fires inside the minimum interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers, expected",
    [
        (None, 2.0),                                        # no headers attribute at all
        ({}, 2.0),                                          # header absent
        ({"Retry-After": "7"}, 7.0),                        # numeric seconds
        ({"Retry-After": " 0.5 "}, 0.5),                    # whitespace tolerated
        ({"Retry-After": "3600"}, RETRY_AFTER_CAP),         # capped
        ({"Retry-After": "-4"}, 0.0),                       # never negative
        ({"Retry-After": "nan"}, 2.0),                      # not finite -> fallback
        ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, 2.0),  # HTTP-date -> fallback
        ({"Retry-After": ""}, 2.0),
    ],
)
def test_retry_after_seconds(headers, expected):
    resp = FakeResponse(429, headers=headers)
    assert _retry_after_seconds(resp, default=2.0) == expected


def test_429_retry_honours_numeric_retry_after(fake_get, sleep_calls):
    ok = FakeResponse(200, text="payload")
    fake_get(FakeResponse(429, headers={"Retry-After": "7"}), ok)
    assert _get_with_retry("http://api.test/x", {}, backoff=2.0) is ok
    assert sleep_calls == [7.0], "the server's wait replaces the fixed backoff"


def test_429_with_a_retry_after_beyond_the_cap_gives_up_without_sleeping(fake_get, sleep_calls):
    """P6: a Retry-After this caller will not wait out (3600 s > the 30 s cap) used to be
    clamped to 30 s and retried — a retry inside the server's window, refused again by
    construction. The call now returns None at once: no sleep, no second request."""
    state = fake_get(FakeResponse(429, headers={"Retry-After": "3600"}), FakeResponse(200))
    assert _get_with_retry("http://api.test/x", {}, backoff=2.0) is None
    assert sleep_calls == [] and len(state["calls"]) == 1


def test_429_retry_after_at_the_cap_is_slept_and_retried(fake_get, sleep_calls):
    """The cap bounds the sleep that precedes an actual retry: exactly RETRY_AFTER_CAP is
    honoured (slept, then retried); only the strict excess gives up."""
    ok = FakeResponse(200)
    fake_get(FakeResponse(429, headers={"Retry-After": str(int(RETRY_AFTER_CAP))}), ok)
    assert _get_with_retry("http://api.test/x", {}, backoff=2.0) is ok
    assert sleep_calls == [RETRY_AFTER_CAP]


@pytest.mark.parametrize(
    "headers, expected",
    [
        (None, None), ({}, None), ({"Retry-After": "7"}, 7.0), ({"Retry-After": "3600"}, 3600.0),
        ({"Retry-After": "-4"}, -4.0), ({"Retry-After": "nan"}, None), ({"Retry-After": "inf"}, None),
        ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None), ({"Retry-After": ""}, None),
    ],
)
def test_parse_retry_after_is_the_uncapped_header(headers, expected):
    assert _parse_retry_after(FakeResponse(429, headers=headers)) == expected


def test_429_retry_falls_back_to_backoff_on_http_date(fake_get, sleep_calls):
    fake_get(FakeResponse(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
             FakeResponse(200))
    _get_with_retry("http://api.test/x", {}, backoff=2.0)
    assert sleep_calls == [2.0]


def _fake_clock(monkeypatch, start=1000.0):
    """time.monotonic reads a dict; time.sleep advances it and records the request."""
    clock = {"t": start}
    sleeps: list[float] = []
    monkeypatch.setattr(retrieval.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(retrieval.time, "sleep",
                        lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)))
    return clock, sleeps


def test_throttle_runs_before_every_attempt(fake_get, monkeypatch):
    clock, sleeps = _fake_clock(monkeypatch)
    calls: list[str] = []
    fake_get(FakeResponse(429), FakeResponse(200))
    assert _get_with_retry("http://api.test/x", {}, backoff=2.0,
                           throttle=lambda: calls.append("throttle")) is not None
    assert calls == ["throttle", "throttle"]


def test_s2_429_retry_respects_the_minimum_interval(monkeypatch, fake_get):
    """429 at t=1000, fixed backoff 2.0 s < the 2.5 s S2 interval: the retry used to fire at
    t=1002 (throttle bypassed); now the throttle tops the wait up to the interval."""
    monkeypatch.setattr(retrieval, "_S2_MIN_INTERVAL", 2.5)
    monkeypatch.setattr(retrieval, "_s2_last_call", 0.0)
    clock, sleeps = _fake_clock(monkeypatch)
    state = fake_get(FakeResponse(429), FakeResponse(json_data={"data": []}))
    at: list[float] = []
    real_get = retrieval.requests.get
    monkeypatch.setattr(retrieval.requests, "get",
                        lambda *a, **k: (at.append(clock["t"]), real_get(*a, **k))[1])
    assert semantic_scholar_search("q") == []
    assert len(state["calls"]) == 2
    assert sleeps == [2.0, 0.5]                      # backoff, then the throttle's top-up
    assert at[1] - at[0] >= 2.5 - 1e-9
    assert retrieval._s2_last_call == at[1]          # stamp refreshed on the retry


def test_s2_retry_after_longer_than_interval_adds_no_throttle_sleep(monkeypatch, fake_get):
    monkeypatch.setattr(retrieval, "_S2_MIN_INTERVAL", 2.5)
    monkeypatch.setattr(retrieval, "_s2_last_call", 0.0)
    clock, sleeps = _fake_clock(monkeypatch)
    fake_get(FakeResponse(429, headers={"Retry-After": "7"}),
             FakeResponse(json_data={"data": []}))
    semantic_scholar_search("q")
    assert sleeps == [7.0]


def test_arxiv_network_error_retry_respects_the_minimum_interval(monkeypatch, fake_get):
    """A connection error waits backoff/2 = 1 s; the arXiv interval is 3 s, so the throttle
    sleeps the remaining 2 s before the retry (and the stamp moves to the retry)."""
    monkeypatch.setattr(retrieval, "_ARXIV_MIN_INTERVAL", 3.0)
    monkeypatch.setattr(retrieval, "_arxiv_last_call", 0.0)
    clock, sleeps = _fake_clock(monkeypatch)
    state = fake_get(requests.ConnectionError("boom"), FakeResponse(text=""))
    at: list[float] = []
    real_get = retrieval.requests.get
    monkeypatch.setattr(retrieval.requests, "get",
                        lambda *a, **k: (at.append(clock["t"]), real_get(*a, **k))[1])
    assert arxiv_search("q") == []
    assert len(state["calls"]) == 2
    assert sleeps == [1.0, 2.0]
    assert at[1] - at[0] >= 3.0 - 1e-9
    assert retrieval._arxiv_last_call == at[1]


def test_throttle_is_not_run_twice_before_the_first_attempt(monkeypatch, fake_get):
    """The channels hand their throttle to _get_with_retry instead of calling it themselves:
    one successful call = one throttle invocation, so the stamp is set exactly once."""
    monkeypatch.setattr(retrieval, "_S2_MIN_INTERVAL", 2.5)
    monkeypatch.setattr(retrieval, "_s2_last_call", 0.0)
    calls: list[str] = []
    real = retrieval._s2_throttle
    monkeypatch.setattr(retrieval, "_s2_throttle", lambda: (calls.append("t"), real())[1])
    fake_get(FakeResponse(json_data={"data": []}))
    semantic_scholar_search("q")
    assert calls == ["t"]


# ---------------------------------------------------------------------------
# _parse_arxiv_atom
# ---------------------------------------------------------------------------

ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1701.02434v2</id>
    <published>2017-01-09T19:00:00Z</published>
    <title>A Conceptual   Introduction to
      Hamiltonian Monte Carlo</title>
    <summary>
      Hamiltonian Monte Carlo  explained
      for applied  statisticians.
    </summary>
    <author><name>Michael Betancourt</name></author>
  </entry>
</feed>
"""


def test_parse_arxiv_atom_entry():
    papers = _parse_arxiv_atom(ATOM_XML)
    assert len(papers) == 1
    p = papers[0]
    assert p["arxiv_id"] == "1701.02434"  # /abs/ split + version suffix stripped
    assert p["year"] == 2017  # from <published>
    # internal whitespace/newlines collapsed in title and abstract
    assert p["title"] == "A Conceptual Introduction to Hamiltonian Monte Carlo"
    assert p["abstract"] == "Hamiltonian Monte Carlo explained for applied statisticians."
    assert p["authors"] == ["Michael Betancourt"]
    assert p["source"] == "arxiv"


def test_parse_arxiv_atom_malformed_xml_returns_empty():
    assert _parse_arxiv_atom("<feed><entry></feed>") == []
    assert _parse_arxiv_atom("") == []


# ---------------------------------------------------------------------------
# Channel functions degrade gracefully ([] on timeout / bad JSON)
# ---------------------------------------------------------------------------


def test_arxiv_search_degrades_to_empty_on_timeout(fake_get, sleep_calls):
    fake_get(requests.Timeout("t"), requests.Timeout("t"))
    assert arxiv_search("non-centered parameterization") == []


def test_semantic_scholar_search_degrades_on_timeout(fake_get, sleep_calls):
    fake_get(requests.Timeout("t"), requests.Timeout("t"))
    assert semantic_scholar_search("weakly informative priors") == []


def test_semantic_scholar_search_degrades_on_bad_json(fake_get, sleep_calls):
    fake_get(FakeResponse(200, json_error=True))
    assert semantic_scholar_search("weakly informative priors") == []


def test_s2_citing_papers_degrades_on_timeout(fake_get, sleep_calls):
    fake_get(requests.Timeout("t"), requests.Timeout("t"))
    assert s2_citing_papers("arXiv:0708.3797") == []


def test_s2_citing_papers_degrades_on_bad_json(fake_get, sleep_calls):
    fake_get(FakeResponse(200, json_error=True))
    assert s2_citing_papers("arXiv:0708.3797") == []


def test_s2_citing_papers_happy_path(fake_get, sleep_calls):
    payload = {
        "data": [
            {
                "citingPaper": {
                    "paperId": "p-new",
                    "title": "Newer sampler methods",
                    "abstract": "Better posterior inference.",
                    "year": 2023,
                    "citationCount": 12,
                    "externalIds": {"ArXiv": "2301.00001"},
                }
            },
            {  # below min_year -> filtered
                "citingPaper": {
                    "paperId": "p-old",
                    "title": "Too old",
                    "abstract": "B",
                    "year": 2018,
                    "citationCount": 4,
                    "externalIds": {},
                }
            },
            {  # year missing -> filtered (cannot prove recency)
                "citingPaper": {
                    "paperId": "p-noyear",
                    "title": "No year",
                    "abstract": "C",
                    "year": None,
                    "citationCount": 1,
                    "externalIds": {},
                }
            },
        ]
    }
    state = fake_get(FakeResponse(200, json_data=payload))
    papers = s2_citing_papers("arXiv:0708.3797", min_year=2020)
    assert len(papers) == 1
    p = papers[0]
    assert p["paper_id"] == "p-new"  # citingPaper extraction
    assert p["arxiv_id"] == "2301.00001"  # externalIds.ArXiv mapping
    assert p["citationCount"] == 12
    assert p["source"] == "semantic_scholar"
    assert "/paper/arXiv:0708.3797/citations" in state["calls"][0]["url"]


# ---------------------------------------------------------------------------
# Scout live-funnel helpers (retrieval fns stubbed in the scout namespace)
# ---------------------------------------------------------------------------


def test_live_hits_cosine_scores_and_skip_branches(monkeypatch):
    query_text = "non-centered parameterization hierarchical model"
    arxiv_paper = {
        "arxiv_id": "2301.01234",
        "title": "Non-centered parameterization for hierarchical models",
        "abstract": "Reparameterization improves sampling in hierarchical models.",
    }
    idless_paper = {"title": "No identifiers here", "abstract": "still scored, no id"}
    docless_paper = {"arxiv_id": "2401.99999", "title": "", "abstract": None}
    s2_paper = {
        "paper_id": "s2-1",
        "title": "Hierarchical model reparameterization tricks",
        "abstract": "non-centered tricks for samplers",
    }

    arxiv_doc = f"{arxiv_paper['title']} {arxiv_paper['abstract']}"
    s2_doc = f"{s2_paper['title']} {s2_paper['abstract']}"
    embedder = TfidfEmbedder().fit([query_text, arxiv_doc, s2_doc])

    arxiv_calls: list[str] = []
    s2_calls: list[str] = []

    def fake_arxiv_search(text, *a, **k):
        arxiv_calls.append(text)
        return [arxiv_paper, idless_paper, docless_paper]

    def fake_s2_search(text, *a, **k):
        s2_calls.append(text)
        return [s2_paper]

    monkeypatch.setattr(scout, "arxiv_search", fake_arxiv_search)
    monkeypatch.setattr(scout, "semantic_scholar_search", fake_s2_search)

    queries = [{"query": query_text}, {"query": ""}]  # empty query is skipped
    hits = scout._live_hits(queries, embedder)

    assert arxiv_calls == [query_text]
    assert s2_calls == [query_text]
    # id-less and doc-less papers are skipped; S2 hits fall back to paper_id
    assert [h["arxiv_id"] for h in hits] == ["2301.01234", "s2-1"]
    q_emb = embedder.encode(query_text)
    assert hits[0]["score"] == pytest.approx(float(q_emb @ embedder.encode(arxiv_doc)))
    assert hits[0]["score"] > 0.0
    assert hits[0]["paper"] is arxiv_paper
    assert hits[1]["score"] == pytest.approx(float(q_emb @ embedder.encode(s2_doc)))


def test_citation_walk_hits_inside_running_loop_returns_empty(caplog):
    """Called with an event loop already running -> [] plus a logged warning.

    The research test is an ``async def`` run under pytest-asyncio's auto mode; the
    package declares no asyncio plugin, so the sync helper is called from inside a
    coroutine driven by ``asyncio.run`` (it then sees a running loop).
    """
    embedder = TfidfEmbedder().fit(["hierarchical model priors"])

    async def inside_a_running_loop():
        return scout._citation_walk_hits(
            ["0708.3797"], [{"query": "hierarchical model priors"}], embedder
        )

    with caplog.at_level("WARNING", logger="compass.agents.scout"):
        out = asyncio.run(inside_a_running_loop())
    assert out == []
    assert any("citation-walk channel skipped" in r.message for r in caplog.records)


def test_citation_walk_hits_propagates_unrelated_runtime_errors(monkeypatch):
    """RuntimeErrors from the walk itself must surface, not be swallowed."""

    async def broken_walk(*a, **k):
        raise RuntimeError("unrelated bug")

    monkeypatch.setattr(scout, "walk_citations", broken_walk)
    embedder = TfidfEmbedder().fit(["hierarchical model priors"])
    with pytest.raises(RuntimeError, match="unrelated bug"):
        scout._citation_walk_hits(
            ["0708.3797"], [{"query": "hierarchical model priors"}], embedder
        )


# ---------------------------------------------------------------------------
# v3: S2 global rate-limiter (proactive throttle) + x-api-key auth
# ---------------------------------------------------------------------------

def test_s2_throttle_spaces_consecutive_calls(monkeypatch, fake_get):
    monkeypatch.setattr(retrieval, "_S2_MIN_INTERVAL", 2.5)
    monkeypatch.setattr(retrieval, "_s2_last_call", 0.0)
    clock = {"t": 1000.0}
    sleeps: list[float] = []
    monkeypatch.setattr(retrieval.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(retrieval.time, "sleep",
                        lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)))
    fake_get(FakeResponse(json_data={"data": []}), FakeResponse(json_data={"data": []}))
    semantic_scholar_search("q1")   # first call: no prior -> no throttle sleep
    semantic_scholar_search("q2")   # immediate second call -> throttled ~2.5s
    assert any(abs(s - 2.5) < 1e-9 for s in sleeps), sleeps


def test_s2_throttle_disabled_when_interval_zero(monkeypatch, fake_get):
    # _no_throttle autouse already sets _S2_MIN_INTERVAL = 0
    sleeps: list[float] = []
    monkeypatch.setattr(retrieval.time, "sleep", sleeps.append)
    fake_get(FakeResponse(json_data={"data": []}), FakeResponse(json_data={"data": []}))
    semantic_scholar_search("q1")
    semantic_scholar_search("q2")
    assert sleeps == []  # no proactive throttling


def test_s2_search_sends_api_key_header_when_set(monkeypatch, fake_get):
    monkeypatch.setenv("S2_API_KEY", "sk-test-123")
    state = fake_get(FakeResponse(json_data={"data": []}))
    semantic_scholar_search("hierarchical priors")
    assert state["calls"][0]["headers"] == {"x-api-key": "sk-test-123"}


def test_s2_search_no_header_without_key(fake_get):
    state = fake_get(FakeResponse(json_data={"data": []}))   # autouse deletes S2_API_KEY
    semantic_scholar_search("hierarchical priors")
    assert state["calls"][0]["headers"] in (None, {})


def test_s2_citations_throttle_and_key(monkeypatch, fake_get):
    monkeypatch.setenv("S2_API_KEY", "sk-test-123")
    state = fake_get(FakeResponse(json_data={"data": []}))
    s2_citing_papers("arXiv:0708.3797")
    assert state["calls"][0]["headers"] == {"x-api-key": "sk-test-123"}
    assert "/citations" in state["calls"][0]["url"]


def test_asta_key_alone_authenticates_search_and_citations(monkeypatch, fake_get):
    """README: ASTA_API_KEY is a general alternative to S2_API_KEY. The search and citation
    channels used to send it nowhere (only paper_finder_search accepted it)."""
    monkeypatch.setenv("ASTA_API_KEY", "asta-key-1")   # S2_API_KEY deleted by the autouse
    state = fake_get(FakeResponse(json_data={"data": []}), FakeResponse(json_data={"data": []}),
                     FakeResponse(json_data={"data": []}))
    semantic_scholar_search("hierarchical priors")
    s2_citing_papers("arXiv:0708.3797")
    paper_finder_search("hierarchical priors")
    assert [c["headers"] for c in state["calls"]] == [{"x-api-key": "asta-key-1"}] * 3
    assert "/paper/search" in state["calls"][0]["url"]
    assert "/citations" in state["calls"][1]["url"]
    assert "/snippet/search" in state["calls"][2]["url"]


def test_s2_key_wins_over_asta_key_everywhere(monkeypatch, fake_get):
    monkeypatch.setenv("S2_API_KEY", "s2-key")
    monkeypatch.setenv("ASTA_API_KEY", "asta-key")
    state = fake_get(FakeResponse(json_data={"data": []}), FakeResponse(json_data={"data": []}),
                     FakeResponse(json_data={"data": []}))
    semantic_scholar_search("q")
    s2_citing_papers("p")
    paper_finder_search("q")
    assert [c["headers"] for c in state["calls"]] == [{"x-api-key": "s2-key"}] * 3
    assert retrieval._s2_auth_headers() == {"x-api-key": "s2-key"}


def test_s2_auth_headers_empty_without_any_key():
    assert retrieval._s2_auth_headers() == {}


# ---------------------------------------------------------------------------
# arXiv politeness: proactive request spacing + quoted query phrase
# ---------------------------------------------------------------------------

def test_arxiv_throttle_spaces_consecutive_calls(monkeypatch, fake_get):
    monkeypatch.setattr(retrieval, "_ARXIV_MIN_INTERVAL", 3.0)
    monkeypatch.setattr(retrieval, "_arxiv_last_call", 0.0)
    clock = {"t": 1000.0}
    sleeps: list[float] = []
    monkeypatch.setattr(retrieval.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(retrieval.time, "sleep",
                        lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)))
    fake_get(FakeResponse(text=""), FakeResponse(text=""))
    arxiv_search("q1")   # first call: no prior -> no throttle sleep
    arxiv_search("q2")   # immediate second call -> throttled ~3s
    assert any(abs(s - 3.0) < 1e-9 for s in sleeps), sleeps


def test_arxiv_search_quotes_multiword_query(fake_get):
    state = fake_get(FakeResponse(text=""))
    arxiv_search('weakly "informative" priors', category="stat.CO")
    sq = state["calls"][0]["params"]["search_query"]
    # embedded quotes stripped, phrase quoted, category clause preserved
    assert sq == 'all:"weakly informative priors" AND cat:stat.CO'


# ---------------------------------------------------------------------------
# v3: PaperFinder full-text channel (credential-gated) + PaperStore cache
# ---------------------------------------------------------------------------

def test_paper_finder_gated_without_key(monkeypatch):
    # autouse _no_throttle deletes both keys -> channel is OFF and makes NO call
    monkeypatch.setattr(retrieval.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    assert paper_finder_search("non-centered parameterization") == []


def test_paper_finder_with_key_parses_snippets(monkeypatch, fake_get):
    # Real snippet/search paper stubs carry corpusId but no paperId/externalIds.
    monkeypatch.setenv("S2_API_KEY", "sk-test")
    state = fake_get(FakeResponse(json_data={"data": [
        {"snippet": {"text": "a full-text passage about reparameterization"},
         "paper": {"title": "Non-centered tricks", "corpusId": 12345,
                   "authors": ["A. Author"], "openAccessInfo": {}}},
    ]}))
    out = paper_finder_search("non-centered")
    assert len(out) == 1
    assert out[0]["snippet"] == "a full-text passage about reparameterization"
    assert out[0]["source"] == "paper_finder"
    assert out[0]["paper_id"] == "CorpusId:12345"  # corpusId fallback id
    assert "/snippet/search" in state["calls"][0]["url"]
    # the funnel's fields are requested explicitly (the default stub omits them)
    assert state["calls"][0]["params"]["fields"] == retrieval._PF_FIELDS
    assert state["calls"][0]["headers"] == {"x-api-key": "sk-test"}


def test_paper_finder_synthesizes_id_from_title_when_no_ids(monkeypatch, fake_get):
    """No paperId/externalIds/corpusId -> stable title-hash id, hit NOT dropped."""
    monkeypatch.setenv("S2_API_KEY", "sk-test")
    payload = {"data": [
        {"snippet": {"text": "passage"}, "paper": {"title": "Only A Title"}},
    ]}
    fake_get(FakeResponse(json_data=payload), FakeResponse(json_data=payload))
    first = paper_finder_search("q")
    second = paper_finder_search("q")
    assert first[0]["paper_id"].startswith("title:")
    assert first[0]["paper_id"] == second[0]["paper_id"]  # stable across calls


def test_paper_finder_keeps_paper_id_when_present(monkeypatch, fake_get):
    monkeypatch.setenv("S2_API_KEY", "sk-test")
    fake_get(FakeResponse(json_data={"data": [
        {"snippet": {"text": "p"}, "paper": {"paperId": "s2-abc", "title": "T",
                                             "corpusId": 99}},
    ]}))
    assert paper_finder_search("q")[0]["paper_id"] == "s2-abc"


def test_paper_store_put_get_and_persist(tmp_path):
    p = tmp_path / "papers.json"
    s = PaperStore(p)
    assert len(s) == 0 and "arXiv:1" not in s
    s.put("arXiv:1", {"title": "A", "abstract": "full text"})
    assert s.get("arXiv:1")["title"] == "A" and "arXiv:1" in s
    # a fresh instance reads the persisted file back
    assert PaperStore(p).get("arXiv:1")["title"] == "A"


def test_paper_store_get_or_fetch_caches(tmp_path):
    s = PaperStore(tmp_path / "s.json")
    calls = []

    def fetch(pid):
        calls.append(pid)
        return {"title": "fetched"}

    assert s.get_or_fetch("id1", fetch)["title"] == "fetched"
    assert s.get_or_fetch("id1", fetch)["title"] == "fetched"  # served from cache
    assert calls == ["id1"]  # fetched exactly once


def test_paper_store_corrupt_file_degrades(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not valid json")
    s = PaperStore(p)  # must NOT raise
    assert len(s) == 0


def test_paper_store_non_serializable_degrades(tmp_path):
    p = tmp_path / "s.json"
    s = PaperStore(p)
    s.put("bad", {"title": "X", "emb": {1, 2, 3}})  # a set is not JSON-serializable
    assert "bad" not in s                            # skipped, not stored, no raise
    s.put("good", {"title": "Y"})                    # store is NOT wedged
    assert s.get("good")["title"] == "Y"
    assert PaperStore(p).get("good")["title"] == "Y"  # persisted


def test_paper_store_flush_is_atomic(tmp_path, monkeypatch):
    """A failure between serialize and publish never corrupts the backing file."""
    p = tmp_path / "s.json"
    s = PaperStore(p)
    s.put("one", {"title": "A"})
    assert not list(tmp_path.glob("*.tmp"))  # temp file replaced, not left behind

    # Simulate an interrupt at the publish step: os.replace fails -> the
    # previously persisted file must remain intact (valid JSON, old contents).
    monkeypatch.setattr(retrieval.os, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("killed")))
    s.put("two", {"title": "B"})             # in-memory only; flush degraded
    reloaded = PaperStore(p)
    assert reloaded.get("one")["title"] == "A"  # old cache NOT lost
    assert "two" not in reloaded
