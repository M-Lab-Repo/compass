"""Retrieval stack for the Literature Scout (spec §5.2).

Pluggable, dependency-light retrieval:

* ``Embedder`` protocol + ``TfidfEmbedder`` — a pure numpy/stdlib TF-IDF
  embedder (deterministic, no torch / sentence-transformers required).
* ``SentenceTransformerEmbedder`` — optional drop-in upgrade, guarded import
  (the capsule image does not ship torch; install the ``retrieval`` extra).
* ``InMemoryVectorStore`` — cosine-similarity store that mirrors the chroma
  collection call shape (``add`` / ``query``) so a ``ChromaVectorStore``
  adapter can drop in later without touching the scout.
* ``chunk_paper`` / ``build_paper_index`` — abstract + method chunking per
  spec §5.2: an abstract match signals topical relevance, a method-section
  match signals a concretely usable technique.
* Live channels (arXiv Atom API, Semantic Scholar Graph API) — import-safe,
  NEVER called by default tests, and degrade gracefully (return ``[]``) on
  rate limits / timeouts instead of raising.
* ``is_methodology_paper`` + ``walk_citations`` — the citation-graph walker
  (spec §5.2 "SOTA-finder") with an injectable async ``fetch`` so tests can
  stub the citation graph.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer with poor-man's plural stripping.

    Trailing 's' is stripped from tokens longer than 3 chars (but not 'ss'
    endings), so "priors"/"prior" and "coefficients"/"coefficient" match.
    Applied identically to corpus and queries, so it is purely a recall aid.
    """
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


@runtime_checkable
class Embedder(Protocol):
    """Anything with fit(corpus_texts) and encode(text) -> L2-normalized vector."""

    def fit(self, corpus_texts: Sequence[str]) -> Embedder:  # pragma: no cover
        ...

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover
        ...


class TfidfEmbedder:
    """Deterministic pure-numpy TF-IDF embedder.

    - tokenizer: lowercase alnum (see ``_tokenize``)
    - sublinear tf: ``1 + log(count)``
    - smooth idf: ``log((1 + N) / (1 + df)) + 1``
    - L2-normalized output; out-of-vocabulary terms are ignored.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] | None = None
        self._idf: np.ndarray | None = None

    def fit(self, corpus_texts: Sequence[str]) -> TfidfEmbedder:
        doc_terms = [set(_tokenize(t)) for t in corpus_texts]
        df: Counter = Counter()
        for terms in doc_terms:
            df.update(terms)
        self._vocab = {term: i for i, term in enumerate(sorted(df))}
        n_docs = len(corpus_texts)
        idf = np.zeros(len(self._vocab), dtype=np.float64)
        for term, i in self._vocab.items():
            idf[i] = math.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0
        self._idf = idf
        return self

    def encode(self, text: str) -> np.ndarray:
        if self._vocab is None or self._idf is None:
            raise RuntimeError("TfidfEmbedder.encode() called before fit()")
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        counts = Counter(t for t in _tokenize(text) if t in self._vocab)
        for term, count in counts.items():
            i = self._vocab[term]
            vec[i] = (1.0 + math.log(count)) * self._idf[i]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec


class SentenceTransformerEmbedder:
    """Optional neural embedder (NOT installed in the capsule image).

    Guarded import: raises ImportError with an install hint if
    sentence-transformers (and its torch dependency) is missing.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "sentence-transformers is not installed (torch does not fit "
                "this image). Install the retrieval extras to enable it: "
                "pip install 'compass-refine[retrieval]' "
                "(or: pip install sentence-transformers)"
            ) from exc
        self._model = SentenceTransformer(model_name)  # pragma: no cover

    def fit(self, corpus_texts: Sequence[str]) -> SentenceTransformerEmbedder:
        return self  # pretrained; nothing to fit  # pragma: no cover

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - needs torch
        vec = np.asarray(self._model.encode(text), dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class InMemoryVectorStore:
    """Cosine-similarity vector store mirroring the chroma collection shape.

    ``add(ids, documents, embeddings, metadatas)`` matches
    ``chromadb.Collection.add``; ``query(embedding, n_results, where)``
    matches the single-query form of ``Collection.query`` but returns a flat
    ``list[dict]`` (``{id, document, metadata, score}``, best first) instead
    of chroma's columnar dict. A future ``ChromaVectorStore`` adapter only
    needs to translate that return shape — the scout never sees the
    difference. ``where`` is an optional metadata equality filter
    (``{"section": "abstract"}``), the subset of chroma's where-grammar the
    scout actually uses.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._metadatas: list[dict] = []

    def __len__(self) -> int:
        return len(self._ids)

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence,
        metadatas: Sequence[dict],
    ) -> None:
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError(
                "ids, documents, embeddings and metadatas must have equal length"
            )
        for i, doc, emb, meta in zip(ids, documents, embeddings, metadatas, strict=True):
            self._ids.append(str(i))
            self._documents.append(doc)
            self._embeddings.append(np.asarray(emb, dtype=np.float64))
            self._metadatas.append(dict(meta) if meta else {})

    def query(
        self,
        embedding,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Top-``n_results`` entries by cosine similarity, best first."""
        if not self._ids:
            return []
        q = np.asarray(embedding, dtype=np.float64)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0.0:
            q = q / q_norm
        scored = []
        for idx in range(len(self._ids)):
            meta = self._metadatas[idx]
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            emb = self._embeddings[idx]
            norm = float(np.linalg.norm(emb))
            score = float(np.dot(q, emb / norm)) if norm > 0.0 else 0.0
            scored.append((score, idx))
        scored.sort(key=lambda t: (-t[0], t[1]))  # deterministic tie-break
        return [
            {
                "id": self._ids[idx],
                "document": self._documents[idx],
                "metadata": dict(self._metadatas[idx]),
                "score": score,
            }
            for score, idx in scored[: max(0, int(n_results))]
        ]


# ---------------------------------------------------------------------------
# Chunking + indexing (spec §5.2: abstract chunk = topical relevance,
# method chunk = usable technique)
# ---------------------------------------------------------------------------


def chunk_paper(paper: dict) -> list[dict]:
    """Split a paper dict into an abstract chunk and a method chunk.

    The abstract chunk concatenates title + abstract + keywords (topical
    signal); the method chunk is the ``section_method`` text (technique
    signal). Chunks are ``{"section_type": "abstract"|"method", "text": str}``.
    """
    chunks: list[dict] = []
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    keywords = paper.get("keywords") or []
    abstract_text = " ".join(
        part for part in (title, abstract, " ".join(keywords)) if part
    ).strip()
    if abstract_text:
        chunks.append({"section_type": "abstract", "text": abstract_text})
    method = paper.get("section_method") or ""
    if method:
        chunks.append({"section_type": "method", "text": method})
    return chunks


def build_paper_index(
    papers: Sequence[dict],
    embedder: Embedder | None = None,
) -> tuple[InMemoryVectorStore, Embedder]:
    """Chunk + embed + index ``papers``; returns ``(store, embedder)``.

    Default embedder is a fresh ``TfidfEmbedder`` fitted on ALL chunk texts.
    A provided embedder is also (re)fitted on the chunk texts — a no-op for
    pretrained embedders whose ``fit`` returns self.
    """
    records: list[tuple[dict, dict]] = []
    for paper in papers:
        for chunk in chunk_paper(paper):
            records.append((paper, chunk))
    texts = [chunk["text"] for _, chunk in records]
    if embedder is None:
        embedder = TfidfEmbedder()
    embedder.fit(texts)
    store = InMemoryVectorStore()
    ids, docs, embs, metas = [], [], [], []
    for paper, chunk in records:
        arxiv_id = paper.get("arxiv_id") or "unknown"
        ids.append(f"{arxiv_id}_{chunk['section_type']}")
        docs.append(chunk["text"])
        embs.append(embedder.encode(chunk["text"]))
        meta = {
            "arxiv_id": arxiv_id,
            "title": paper.get("title") or "",
            "year": paper.get("year"),
            "section": chunk["section_type"],
        }
        if "category" in paper:
            meta["category"] = paper["category"]
        metas.append(meta)
    store.add(ids, docs, embs, metas)
    return store, embedder


# ---------------------------------------------------------------------------
# Full-text paper cache (v3) — a persisted JSON store keyed by paper id, so
# fetched full texts/metadata are reused across runs instead of re-fetched.
# ---------------------------------------------------------------------------


class PaperStore:
    """Persistent JSON cache of fetched papers, keyed by id.

    Avoids re-fetching the same paper across runs; stores whatever the fetcher
    returns (full text when available, abstract otherwise). Load-once at
    construction, write-through to ``path`` on every ``put``. Failures to
    read/write the backing file degrade to an in-memory-only cache rather than
    raising into the scout.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._mem: dict = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    self._mem = loaded
            except (ValueError, OSError):
                self._mem = {}

    def __len__(self) -> int:
        return len(self._mem)

    def __contains__(self, paper_id) -> bool:
        return str(paper_id) in self._mem

    def get(self, paper_id) -> dict | None:
        return self._mem.get(str(paper_id))

    def put(self, paper_id, paper: dict) -> None:
        # Validate serializability BEFORE committing, so one non-JSON paper
        # (e.g. a set/numpy value from a custom finder) cannot wedge the store
        # or abort the scout — degrade by skipping it (per the class contract).
        try:
            json.dumps(paper)
        except (TypeError, ValueError):
            return
        self._mem[str(paper_id)] = paper
        self._flush()

    def get_or_fetch(self, paper_id, fetch_fn: Callable) -> dict | None:
        """Return the cached paper, else ``fetch_fn(paper_id)`` (and cache it)."""
        hit = self.get(paper_id)
        if hit is not None:
            return hit
        paper = fetch_fn(paper_id)
        if paper:
            self.put(paper_id, paper)
        return paper

    def _flush(self) -> None:
        # Atomic write: serialize to a sibling temp file, then os.replace() it
        # over the target (atomic on POSIX), so an interrupt mid-write can
        # never leave a truncated/invalid JSON file behind.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self._mem, indent=2))
            os.replace(tmp, self.path)
        except (OSError, TypeError, ValueError):  # pragma: no cover - read-only fs etc.
            pass


# ---------------------------------------------------------------------------
# Live channels (arXiv + Semantic Scholar) — import-safe, never called by
# default tests; on 429/timeout retry once (Retry-After or a fixed backoff,
# then the host's throttle), then return [].
# ---------------------------------------------------------------------------

ARXIV_API_URL = "http://export.arxiv.org/api/query"
S2_API_URL = "https://api.semanticscholar.org/graph/v1"
_S2_SNIPPET_URL = f"{S2_API_URL}/snippet/search"  # full-text passage search
_S2_FIELDS = "title,abstract,year,citationCount,externalIds"
# The snippet endpoint returns only a minimal paper stub by default; ask for
# the fields the funnel needs (id, recency, citations) explicitly.
_PF_FIELDS = "paperId,externalIds,year,citationCount,title,abstract,authors"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# ---------------------------------------------------------------------------
# Semantic Scholar throttle + auth (v3): space live S2 calls PROACTIVELY so we
# stay inside the keyless shared-pool limits (and remain a good citizen with a
# key). The throttle is process-global and thread-safe (walk_citations fans out
# via asyncio.to_thread). Set _S2_MIN_INTERVAL = 0 to disable (tests). An
# S2_API_KEY (or, failing that, ASTA_API_KEY) in the environment is sent as the
# x-api-key header for a dedicated rate limit; without it the public endpoints
# still work, just throttled.
# ---------------------------------------------------------------------------
_S2_MIN_INTERVAL = 2.5
_s2_lock = threading.Lock()
_s2_last_call = 0.0

# arXiv asks for ~1 request per 3 seconds; same proactive-spacing scheme as S2.
_ARXIV_MIN_INTERVAL = 3.0
_arxiv_lock = threading.Lock()
_arxiv_last_call = 0.0


def _s2_throttle() -> None:
    """Block until >= _S2_MIN_INTERVAL has elapsed since the last live S2 call."""
    global _s2_last_call
    if _S2_MIN_INTERVAL <= 0:
        return
    with _s2_lock:
        wait = _S2_MIN_INTERVAL - (time.monotonic() - _s2_last_call)
        if wait > 0:
            time.sleep(wait)
        _s2_last_call = time.monotonic()


def _arxiv_throttle() -> None:
    """Block until >= _ARXIV_MIN_INTERVAL has elapsed since the last arXiv call."""
    global _arxiv_last_call
    if _ARXIV_MIN_INTERVAL <= 0:
        return
    with _arxiv_lock:
        wait = _ARXIV_MIN_INTERVAL - (time.monotonic() - _arxiv_last_call)
        if wait > 0:
            time.sleep(wait)
        _arxiv_last_call = time.monotonic()


def _s2_headers() -> dict:
    """``x-api-key`` header when S2_API_KEY is set (dedicated limit); else ``{}``."""
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def _asta_headers() -> dict:
    """``x-api-key`` header when ASTA_API_KEY is set; else ``{}``."""
    key = os.environ.get("ASTA_API_KEY")
    return {"x-api-key": key} if key else {}


def _s2_auth_headers() -> dict:
    """``x-api-key`` header for every Semantic Scholar endpoint: ``S2_API_KEY`` when set,
    else ``ASTA_API_KEY`` (the README documents the Asta key as a general alternative),
    else ``{}`` (keyless shared pool).

    Fix: ``semantic_scholar_search`` and ``s2_citing_papers`` sent only ``_s2_headers()``
    while ``paper_finder_search`` accepted either key, so a user with only ASTA_API_KEY
    ran the search and citation channels keyless. All three call sites use this helper.
    """
    return _s2_headers() or _asta_headers()


def s2_key_present() -> bool:
    """True when an S2 or Asta key is configured (gates the PaperFinder channel)."""
    return bool(os.environ.get("S2_API_KEY") or os.environ.get("ASTA_API_KEY"))


RETRY_AFTER_CAP = 30.0  # longest Retry-After (seconds) a 429 retry will wait out
_log = logging.getLogger(__name__)


def _parse_retry_after(resp) -> float | None:
    """The server's ``Retry-After`` header as finite seconds; ``None`` when the header is
    absent, an HTTP-date, or otherwise unparseable."""
    headers = getattr(resp, "headers", None) or {}
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) else None


def _retry_after_seconds(resp, default: float, cap: float = RETRY_AFTER_CAP) -> float:
    """Seconds to sleep before retrying a 429: the server's numeric ``Retry-After`` header
    when present, clamped to ``[0, cap]``; ``default`` when the header is absent, an
    HTTP-date, or otherwise unparseable. ``_get_with_retry`` does not retry at all when
    the header exceeds ``RETRY_AFTER_CAP`` (P6); the clamp stays as the bound on the
    sleep that precedes an actual retry."""
    seconds = _parse_retry_after(resp)
    if seconds is None:
        return default
    return max(0.0, min(seconds, cap))


def _get_with_retry(
    url: str, params: dict, timeout: float = 10.0, backoff: float = 2.0,
    headers: dict | None = None, throttle: Callable[[], None] | None = None,
) -> requests.Response | None:
    """GET with one retry on HTTP 429 / timeouts / connection errors.

    Returns the response on success, ``None`` on any persistent failure —
    these APIs rate-limit aggressively, so the live channels must degrade
    gracefully and never raise into the agent.

    ``throttle`` is the host's proactive spacer (``_s2_throttle`` /
    ``_arxiv_throttle``); it runs before EVERY attempt, so the retry also
    waits out the host's minimum interval and refreshes its last-call stamp.
    A 429 waits the server's numeric ``Retry-After`` when it sends one, else
    ``backoff``, then retries once — unless that ``Retry-After`` exceeds
    ``RETRY_AFTER_CAP``, in which case the call returns ``None`` at once (P6):
    sleeping the cap and retrying inside the server's window would only be
    refused again. A network error waits ``backoff / 2``. Fix: the retry used to
    sleep the fixed backoff only, bypassing the throttle (whose stamp was set
    once, before the first attempt) and ignoring Retry-After, so a rate-limited
    retry could fire too early.
    """
    for attempt in range(2):
        if throttle is not None:
            throttle()
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                 headers=headers or None)
            if resp.status_code == 429:
                if attempt == 0:
                    wait = _parse_retry_after(resp)
                    if wait is not None and wait > RETRY_AFTER_CAP:
                        # Longer than this caller will wait: a retry after the capped
                        # sleep would land inside the server's window and be refused
                        # again, so give up now (the channel degrades to empty).
                        _log.debug("%s: 429 with Retry-After %.0fs beyond the %.0fs cap; "
                                   "not retrying", url, wait, RETRY_AFTER_CAP)
                        return None
                    time.sleep(_retry_after_seconds(resp, backoff))
                    continue
                return None
            if resp.status_code != 200:
                return None
            return resp
        except requests.RequestException:
            if attempt == 0:
                time.sleep(backoff / 2.0)
                continue
            return None
    return None


def _parse_arxiv_atom(xml_text: str) -> list[dict]:
    """Parse an arXiv Atom feed with xml.etree (no feedparser dependency)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    def _text(entry, tag: str) -> str:
        el = entry.find(f"atom:{tag}", _ATOM_NS)
        return (el.text or "").strip() if el is not None and el.text else ""

    papers = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        raw_id = _text(entry, "id")
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        published = _text(entry, "published")
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            (a.find("atom:name", _ATOM_NS).text or "").strip()
            for a in entry.findall("atom:author", _ATOM_NS)
            if a.find("atom:name", _ATOM_NS) is not None
        ]
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": re.sub(r"\s+", " ", _text(entry, "title")),
                "abstract": re.sub(r"\s+", " ", _text(entry, "summary")),
                "year": year,
                "authors": authors,
                "source": "arxiv",
            }
        )
    return papers


def arxiv_search(
    query: str, category: str = "stat.ME", max_results: int = 10
) -> list[dict]:
    """Search the arXiv export API (Atom). Returns [] on any failure.

    The query phrase is quoted so multi-word queries parse deterministically
    in the search_query grammar (embedded quotes are stripped first).
    """
    phrase = " ".join(query.replace('"', " ").split())
    params = {
        "search_query": f'all:"{phrase}" AND cat:{category}',
        "start": 0,
        "max_results": max_results,
    }
    resp = _get_with_retry(ARXIV_API_URL, params, throttle=_arxiv_throttle)
    if resp is None:
        return []
    return _parse_arxiv_atom(resp.text)


def arxiv_fetch(arxiv_id: str) -> dict | None:
    """Fetch a single paper by id via the arXiv ``id_list`` API. None on failure."""
    resp = _get_with_retry(ARXIV_API_URL, {"id_list": arxiv_id, "max_results": 1},
                           throttle=_arxiv_throttle)
    if resp is None:
        return None
    papers = _parse_arxiv_atom(resp.text)
    return papers[0] if papers else None


def _s2_item_to_paper(item: dict) -> dict:
    external = item.get("externalIds") or {}
    return {
        "paper_id": item.get("paperId"),
        "arxiv_id": external.get("ArXiv"),
        "title": item.get("title") or "",
        "abstract": item.get("abstract") or "",
        "year": item.get("year"),
        "citationCount": item.get("citationCount"),
        "source": "semantic_scholar",
    }


def semantic_scholar_search(query: str, limit: int = 10,
                            fields: str | None = None) -> list[dict]:
    """Semantic Scholar Graph API paper search. Returns [] on 429/timeout.

    ``fields`` overrides the default field set (v3 dynamic per-query extraction
    threads an expanded field list through here).
    """
    resp = _get_with_retry(
        f"{S2_API_URL}/paper/search",
        {"query": query, "limit": limit, "fields": fields or _S2_FIELDS},
        headers=_s2_auth_headers(), throttle=_s2_throttle,
    )
    if resp is None:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    return [_s2_item_to_paper(item) for item in (data.get("data") or [])]


def paper_finder_search(query: str, limit: int = 10) -> list[dict]:
    """v3 'PaperFinder' channel — full-text PASSAGE search over the Semantic
    Scholar snippet index (S2ORC), the corpus behind Ai2's PaperFinder / Scholar
    QA.

    CREDENTIAL-GATED: the snippet endpoint requires auth, so this returns ``[]``
    unless ``S2_API_KEY`` (or ``ASTA_API_KEY``) is set — the channel stays OFF by
    default and degrades gracefully like the others. Each hit carries the matched
    full-text passage in ``snippet`` (a signal the abstract-only channels lack).
    To route through the hosted Asta tool instead, inject a custom ``finder``
    into the scout (``create_scout_agent(paper_finder=...)``).
    """
    if not s2_key_present():
        return []
    resp = _get_with_retry(
        _S2_SNIPPET_URL, {"query": query, "limit": limit, "fields": _PF_FIELDS},
        headers=_s2_auth_headers(), throttle=_s2_throttle,
    )
    if resp is None:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    out: list[dict] = []
    for row in (data.get("data") or []):
        snip = row.get("snippet") or {}
        item = row.get("paper") or {}
        paper = _s2_item_to_paper(item)
        if not paper["paper_id"]:
            # Degrade gracefully when the endpoint omits paperId (observed in
            # the wild): a CorpusId is a valid S2 paper-id prefix; failing
            # that, synthesize a stable id from the title so downstream
            # dedup/gating keeps the hit instead of dropping it.
            corpus_id = item.get("corpusId")
            if corpus_id is not None:
                paper["paper_id"] = f"CorpusId:{corpus_id}"
            elif paper["title"]:
                digest = hashlib.sha1(paper["title"].encode()).hexdigest()[:16]
                paper["paper_id"] = f"title:{digest}"
        paper["snippet"] = snip.get("text") or ""
        paper["source"] = "paper_finder"
        out.append(paper)
    return out


def s2_citing_papers(paper_id: str, min_year: int = 2020) -> list[dict]:
    """Forward citations of ``paper_id`` (S2 id or e.g. "arXiv:0708.3797").

    Returns [] on 429/timeout/any failure — never raises.
    """
    resp = _get_with_retry(
        f"{S2_API_URL}/paper/{paper_id}/citations",
        {"fields": _S2_FIELDS, "limit": 100},
        headers=_s2_auth_headers(), throttle=_s2_throttle,
    )
    if resp is None:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    papers = []
    for row in data.get("data") or []:
        item = row.get("citingPaper") or {}
        paper = _s2_item_to_paper(item)
        year = paper.get("year")
        if year is not None and year >= min_year:
            papers.append(paper)
    return papers


# ---------------------------------------------------------------------------
# Citation walking (spec §5.2 "the SOTA-finder")
# ---------------------------------------------------------------------------

_METHOD_KEYWORDS = (
    "method",
    "prior",
    "inference",
    "estimation",
    "sampler",
    "sampling",
    "parameterization",
    "parametrization",
    "reparameterization",
    "mcmc",
    "markov chain",
    "posterior",
    "variational",
    "algorithm",
    "shrinkage",
    "regularization",
    "hierarchical",
    "cross-validation",
)

_APPLICATION_KEYWORDS = (
    "case study",
    "we apply",
    "applied to",
    "application to",
    "survey of",
    "empirical analysis of",
    "clinical trial results",
)


def is_methodology_paper(paper: dict) -> bool:
    """Keyword filter: methods papers in, application/survey papers out."""
    text = " ".join(
        [
            paper.get("title") or "",
            paper.get("abstract") or "",
            " ".join(paper.get("keywords") or []),
        ]
    ).lower()
    method_hits = sum(1 for kw in _METHOD_KEYWORDS if kw in text)
    application_hits = sum(1 for kw in _APPLICATION_KEYWORDS if kw in text)
    return method_hits > application_hits


def _paper_walk_id(paper: dict) -> str | None:
    return (
        paper.get("paper_id")
        or paper.get("paperId")
        or paper.get("id")
        or paper.get("arxiv_id")
    )


async def walk_citations(
    seed_paper_id: str,
    depth: int = 2,
    fetch: Callable | None = None,
    min_year: int = 2020,
    max_nodes: int = 30,
    max_per_node: int = 20,
) -> list[dict]:
    """Walk forward citations from a seed paper, breadth-first, ``depth`` hops.

    ``fetch`` is an injectable async callable ``paper_id -> list[paper dict]``;
    the default wraps the Semantic Scholar channel (which degrades to [] on
    rate limits). Visited ids are deduped (each paper fetched at most once),
    findings are deduped by id, and only ``is_methodology_paper`` survivors
    are kept / expanded.

    The walk is budgeted so a broad seed cannot queue hundreds of throttled
    fetches: ``max_nodes`` caps the total number of nodes fetched across the
    whole walk, and ``max_per_node`` caps how many survivors of each node
    join the next frontier (branching factor). Both are frontier caps only —
    every methodology survivor of a fetched node is still returned.
    """
    if fetch is None:

        async def fetch(pid: str) -> list[dict]:
            return await asyncio.to_thread(s2_citing_papers, pid, min_year)

    visited: set = set()
    found_ids: set = set()
    findings: list[dict] = []
    frontier: list = [seed_paper_id]
    for _hop in range(max(0, int(depth))):
        next_frontier: list = []
        for pid in frontier:
            if pid is None or pid in visited:
                continue
            if len(visited) >= max_nodes:  # total-fetch budget exhausted
                return findings
            visited.add(pid)
            expanded = 0
            for citer in await fetch(pid):
                year = citer.get("year")
                if year is not None and year < min_year:
                    continue
                if not is_methodology_paper(citer):
                    continue
                cid = _paper_walk_id(citer)
                if cid is not None and cid in found_ids:
                    continue
                if cid is not None:
                    found_ids.add(cid)
                findings.append(citer)
                if expanded < max_per_node:
                    next_frontier.append(cid)
                    expanded += 1
        frontier = next_frontier
    return findings
