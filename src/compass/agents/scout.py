"""Literature Scout agent (spec §5.2) — the retrieval funnel.

Precision over recall: turn the analyst's structured weaknesses into 3-8-word
queries, retrieve candidates through up to three channels (vector store over
a curated corpus; live arXiv/Semantic Scholar; citation-graph walking — the
live channels only when ``live=True``), dedup by arxiv id keeping the best
cosine score, re-rank by relevance x recency x citations, run a per-paper
applicability filter (LLM by default, injectable callable for tests), and
emit at most ``max_findings`` ``state.PaperFinding`` records.

Conventions (system-wide):
- ``create_scout_agent(llm, ...) -> run``; ``run(state)`` sets the changed
  fields on the state AND returns ``{"literature_findings": ..., "status":
  "searched"}`` (LangGraph node update convention).
- ``state.literature_findings`` is a list of ``dataclasses.asdict`` dicts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import re
from collections.abc import Callable, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from compass.agents.retrieval import (
    Embedder,
    InMemoryVectorStore,
    arxiv_fetch,
    arxiv_search,
    build_paper_index,
    paper_finder_search,
    s2_citing_papers,
    semantic_scholar_search,
    walk_citations,
)
from compass.llm import StructuredOutputError, invoke_structured
from compass.schemas import PaperApplicability
from compass.state import PaperFinding, WorkflowState

# Reference year for the recency term of the re-rank score (spec §5.2 step 3).
REF_YEAR = 2026
# Applicability-gate cutoff (0-10): a paper the LLM rates below it is dropped.
# ``create_scout_agent(score_drop=...)`` overrides it per run (bench --lit-threshold).
DEFAULT_SCORE_DROP = 6

# ---------------------------------------------------------------------------
# Query generation (spec §5.2 step 1) — robust against FROZEN_ANALYSIS shape
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
_VARIANCE_ROLE_TOKENS = ("scale", "variance", "noise", "dispersion", "sigma", "tau", "sd")


def _scale_of(prior: str) -> float | None:
    """Parse the scale (last numeric argument) from a prior string.

    "Normal(0, 100)" -> 100.0; "HalfNormal(10)" -> 10.0;
    "Normal(mu_alpha, sigma_alpha)" -> None (symbolic, no numeric scale).
    """
    numbers = _NUM_RE.findall(prior or "")
    if not numbers:
        return None
    return abs(float(numbers[-1]))


def _is_variance_role(role: str) -> bool:
    role_l = (role or "").lower()
    return any(tok in role_l for tok in _VARIANCE_ROLE_TOKENS)


def _clamp_query(text: str, lo: int = 3, hi: int = 8) -> str:
    """Force a query to 3-8 whitespace words (search strings, not sentences)."""
    words = str(text or "").split()
    if not words:
        return ""
    words = words[:hi]
    while len(words) < lo:
        words.append("Bayesian")
    return " ".join(words)


def generate_queries(analysis: dict) -> list[dict]:
    """Turn the analyst's structured weaknesses into targeted search strings.

    Returns ``[{"query": str (3-8 words), "category": "prior"|"hierarchy"|
    "computational", "reason": str}, ...]``, deduped by query text.
    """
    raw: list[dict] = []

    def add(query: str, category: str, reason: str) -> None:
        clamped = _clamp_query(query)
        if clamped:
            raw.append({"query": clamped, "category": category, "reason": reason})

    # --- prior-specific -----------------------------------------------------
    for param in analysis.get("parameters") or []:
        prior = str(param.get("prior_dist") or "")
        role = str(param.get("role") or "parameter")
        name = param.get("name") or "?"
        if _is_variance_role(role):
            # Variance-role parameter on a non-Half* prior -> half-family query.
            if prior and "half" not in prior.lower():
                add(
                    "half-Cauchy half-Normal variance prior",
                    "prior",
                    f"variance-role parameter {name} uses {prior}",
                )
        else:
            scale = _scale_of(prior)
            if scale is not None and scale > 5:
                role_words = role.replace("_", " ").strip()
                add(
                    f"weakly informative prior {role_words} Bayesian",
                    "prior",
                    f"{name} has vague prior {prior}",
                )

    # --- hierarchy-specific ---------------------------------------------------
    levels = analysis.get("hierarchical_levels") or []
    for level in levels:
        n_groups = level.get("n_groups")
        if isinstance(n_groups, (int, float)) and n_groups < 5:
            add(
                "partial pooling few groups Bayesian hierarchical",
                "hierarchy",
                f"only {n_groups} groups in {level.get('grouping_var')}",
            )
    if not levels:
        add(
            "Bayesian multilevel model partial pooling",
            "hierarchy",
            "no hierarchical structure; grouped data may need a multilevel model",
        )

    # --- issue-specific (suggestions are already searchable phrases) ---------
    for issue in analysis.get("potential_issues") or []:
        if str(issue.get("severity") or "").lower() in ("high", "medium"):
            suggestion = str(issue.get("suggestion") or "").strip()
            if suggestion:
                add(suggestion, "computational", str(issue.get("issue") or "")[:240])

    # --- dedup identical queries (first occurrence wins) ----------------------
    seen: set = set()
    queries: list[dict] = []
    for q in raw:
        key = q["query"].lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(q)
    return queries


def decompose_queries(queries: Sequence[dict]) -> list[dict]:
    """v3 thin-retrieval fallback: broaden each query into focused sub-queries.

    Deterministic + offline (no LLM): for every original query, emit head-terms
    and tail-terms variants plus a ``first last Bayesian`` anchor, each clamped
    to 3-8 words and deduped against the originals. Re-running the funnel on
    these widens recall when the first pass returns too few findings.
    """
    seen = {(_query_text(q)).lower() for q in queries}
    out: list[dict] = []
    for q in queries:
        text = _query_text(q)
        cat = q.get("category", "") if isinstance(q, dict) else ""
        words = text.split()
        subs: list[str] = []
        if len(words) >= 4:
            subs.append(" ".join(words[:3]))
            subs.append(" ".join(words[-3:]))
        if len(words) >= 2:
            subs.append(f"{words[0]} {words[-1]} Bayesian")
        for s in subs:
            clamped = _clamp_query(s)
            key = clamped.lower()
            if clamped and key not in seen:
                seen.add(key)
                out.append({"query": clamped, "category": cat or "decomposed",
                            "reason": f"decomposed from: {text}"})
    return out


def _query_text(q) -> str:
    return q.get("query", "") if isinstance(q, dict) else str(q)


# ---------------------------------------------------------------------------
# Funnel internals (module-level for testability)
# ---------------------------------------------------------------------------


def _vector_hits(
    queries: Sequence,
    store: InMemoryVectorStore,
    embedder: Embedder,
    n_per_query: int = 5,
) -> list[dict]:
    """Channel 1: top-``n_per_query`` chunk hits per query from the vector store.

    Returns flat hits ``{"arxiv_id", "score", "metadata"}`` (duplicates across
    queries/chunks expected; ``_dedup_best`` collapses them).
    """
    hits: list[dict] = []
    for q in queries:
        text = q if isinstance(q, str) else q.get("query", "")
        if not text:
            continue
        emb = embedder.encode(text)
        for hit in store.query(emb, n_results=n_per_query):
            hits.append(
                {
                    "arxiv_id": hit["metadata"].get("arxiv_id"),
                    "score": float(hit["score"]),
                    "metadata": hit["metadata"],
                }
            )
    return hits


def _dedup_best(hits: Sequence[dict]) -> dict[str, dict]:
    """Funnel step 3a: dedup candidates by arxiv_id, keeping the best cosine."""
    best: dict[str, dict] = {}
    for hit in hits:
        aid = hit.get("arxiv_id")
        if not aid:
            continue
        if aid not in best or hit["score"] > best[aid]["score"]:
            best[aid] = hit
    return best


def _recency(year) -> float:
    if not year:
        return 0.0
    return 1.0 / (1.0 + max(0, REF_YEAR - int(year)))


def _combined_score(best_cosine: float, year, citations=None) -> float:
    """Funnel step 3b re-rank: 0.6*cosine + 0.25*recency + 0.15*log-citations."""
    citation_term = 0.0
    if citations:
        citation_term = min(1.0, math.log10(1.0 + float(citations)) / 4.0)
    return 0.6 * float(best_cosine) + 0.25 * _recency(year) + 0.15 * citation_term


def _technique_of(paper: dict) -> str:
    keywords = paper.get("keywords") or []
    if keywords:
        return str(keywords[0])
    return str(paper.get("title") or "")


def _method_excerpt(paper: dict, max_chars: int = 600) -> str | None:
    method = paper.get("section_method") or ""
    if not method:
        return None
    if len(method) <= max_chars:
        return method
    return method[:max_chars].rsplit(" ", 1)[0] + " ..."


def _summary_of(paper: dict, max_chars: int = 400) -> str:
    abstract = paper.get("abstract") or paper.get("title") or ""
    if len(abstract) <= max_chars:
        return abstract
    return abstract[:max_chars].rsplit(" ", 1)[0] + " ..."


def _meta_to_paper(metadata: dict) -> dict:
    return {
        "arxiv_id": metadata.get("arxiv_id"),
        "title": metadata.get("title") or "",
        "year": metadata.get("year"),
        "category": metadata.get("category"),
    }


# ---------------------------------------------------------------------------
# Applicability filter (spec §5.2 step 4)
# ---------------------------------------------------------------------------

_APPLICABILITY_SYSTEM = (
    "You are evaluating whether a Bayesian methodology paper is applicable "
    "to a SPECIFIC model. Be strict: this is a precision problem. A paper "
    "is applicable only if its technique addresses a real weakness of THIS "
    "model, fits the framework, and suits the data size. Rate applicability "
    "0-10 (<6 means drop; >=8 means clearly worth implementing) and explain "
    "concretely what in the model would change."
)

_APPLICABILITY_HUMAN = """## The model
{model_summary}

## The paper
Title: {paper_title}
Abstract: {paper_abstract}
Key technique: {paper_technique}

## Questions
1. Does this technique address a real weakness in the model?
2. Is it compatible with the model's framework ({framework})?
3. Is it appropriate for the data size (N={n_obs})?
4. Would implementing this require restructuring the entire model, or just \
modifying specific components?
5. What is the expected effort-to-improvement ratio?

Rate applicability 0-10 and explain concretely what would change."""

APPLICABILITY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _APPLICABILITY_SYSTEM), ("human", _APPLICABILITY_HUMAN)]
)


def _n_obs_from(dataset_summary: str) -> str:
    match = re.search(r"N\s*=\s*([\d,]+)", dataset_summary or "")
    return match.group(1).replace(",", "") if match else "unknown"


def _build_model_summary(state: WorkflowState, analysis: dict) -> str:
    lines = [f"Framework: {analysis.get('framework') or 'unknown'}"]
    if state.problem_description:
        lines.append(f"Problem: {state.problem_description}")
    if state.dataset_summary:
        lines.append(f"Data: {state.dataset_summary}")
    params = analysis.get("parameters") or []
    if params:
        lines.append("Parameters:")
        for p in params:
            lines.append(
                f"  - {p.get('name')} ~ {p.get('prior_dist')} (role: {p.get('role')})"
            )
    for level in analysis.get("hierarchical_levels") or []:
        lines.append(
            f"Hierarchy: {level.get('grouping_var')} with {level.get('n_groups')} "
            f"groups affecting {level.get('params_affected')}"
        )
    issues = analysis.get("potential_issues") or []
    if issues:
        lines.append("Known weaknesses:")
        for issue in issues:
            lines.append(f"  - [{issue.get('severity')}] {issue.get('issue')}")
    return "\n".join(lines)


def _not_applicable(reason: str) -> PaperApplicability:
    """A score-0 verdict with every required field filled from its type (robust to schema
    additions): the gate's conservative outcome when the model's reply cannot be parsed."""
    kwargs = {}
    for name, field in PaperApplicability.model_fields.items():
        if not field.is_required():
            continue
        ann = field.annotation
        if ann is int:
            kwargs[name] = 0
        elif ann is bool:
            kwargs[name] = False
        elif ann is float:
            kwargs[name] = 0.0
        else:
            kwargs[name] = reason if name in ("effort_to_improvement", "reasoning") else (
                "component" if name == "scope" else "unavailable")
    return PaperApplicability(**kwargs)


def _make_llm_applicability(llm, state: WorkflowState, analysis: dict) -> Callable:
    """Default per-paper applicability filter: prompt | structured output."""
    if llm is None:
        raise ValueError(
            "create_scout_agent needs an llm when no applicability callable "
            "is injected"
        )
    chain = APPLICABILITY_PROMPT | llm.with_structured_output(PaperApplicability)
    model_summary = _build_model_summary(state, analysis)
    n_obs = _n_obs_from(state.dataset_summary)
    framework = analysis.get("framework") or "unknown"

    def _apply(paper: dict, _analysis: dict) -> PaperApplicability:
        inputs = {
            "model_summary": model_summary,
            "paper_title": paper.get("title") or "",
            "paper_abstract": (paper.get("abstract") or "")[:1500],
            "paper_technique": _technique_of(paper),
            "framework": framework,
            "n_obs": n_obs,
        }
        try:
            return invoke_structured(
                chain, inputs, nudge_field="paper_abstract",
                schema_hint="`score` as an integer from 0 to 10 and every boolean field",
                what="PaperApplicability",
            )
        except StructuredOutputError as e:
            # one unparseable verdict must not end the run: the paper is treated as
            # not applicable, which is the conservative outcome for the gate
            return _not_applicable(f"unavailable: {e}")

    return _apply


# ---------------------------------------------------------------------------
# Live channels (only when live=True; never hit by default tests)
# ---------------------------------------------------------------------------


def _live_hits(queries: Sequence[dict], embedder: Embedder,
               extraction: Callable | None = None) -> list[dict]:
    """Channel 2: live arXiv + Semantic Scholar, scored against the query
    embedding so live candidates compete on the same cosine scale.

    ``extraction`` (v3) is an optional ``query_text -> S2 fields string`` hook
    that picks per-query extraction fields; default keeps the fixed field set.
    """
    hits: list[dict] = []
    for q in queries:
        text = q.get("query", "") if isinstance(q, dict) else str(q)
        if not text:
            continue
        q_emb = embedder.encode(text)
        fields = extraction(text) if extraction else None
        for paper in arxiv_search(text) + semantic_scholar_search(text, limit=5, fields=fields):
            doc = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".strip()
            if not doc:
                continue
            score = float(q_emb @ embedder.encode(doc))
            aid = paper.get("arxiv_id") or paper.get("paper_id")
            if not aid:
                continue
            hits.append(
                {"arxiv_id": aid, "score": score, "metadata": {}, "paper": paper}
            )
    return hits


def _citation_walk_hits(
    seed_ids: Sequence[str], queries: Sequence[dict], embedder: Embedder
) -> list[dict]:
    """Channel 3: forward-citation walk from the best vector-channel seeds."""

    async def _walk_all() -> list[dict]:
        results = await asyncio.gather(
            *(walk_citations(f"arXiv:{sid}", depth=2) for sid in seed_ids)
        )
        return [paper for chunk in results for paper in chunk]

    # Detect a running loop explicitly (asyncio.run would raise RuntimeError)
    # so the skip is logged and genuine RuntimeErrors from the walk propagate
    # instead of silently disabling the channel.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop — safe to asyncio.run below
    else:
        logging.getLogger(__name__).warning(
            "scout: citation-walk channel skipped — already inside a running "
            "event loop (run the scout from sync code to enable it)"
        )
        return []
    papers = asyncio.run(_walk_all())
    query_embs = [
        embedder.encode(q.get("query", "") if isinstance(q, dict) else str(q))
        for q in queries
    ]
    hits = []
    for paper in papers:
        doc = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".strip()
        aid = paper.get("arxiv_id") or paper.get("paper_id")
        if not doc or not aid:
            continue
        d_emb = embedder.encode(doc)
        score = max((float(q @ d_emb) for q in query_embs), default=0.0)
        hits.append({"arxiv_id": aid, "score": score, "metadata": {}, "paper": paper})
    return hits


def _paper_finder_hits(
    queries: Sequence[dict], embedder: Embedder, finder: Callable
) -> list[dict]:
    """Channel 4 (v3): full-text PaperFinder passage search, scored on the query
    embedding (using the matched snippet when present) so it competes on the same
    cosine scale. ``finder`` self-gates to [] without a key, so this is safe to
    call whenever ``live=True``."""
    hits: list[dict] = []
    for q in queries:
        text = q.get("query", "") if isinstance(q, dict) else str(q)
        if not text:
            continue
        q_emb = embedder.encode(text)
        for paper in finder(text):
            doc = f"{paper.get('title') or ''} " \
                  f"{paper.get('snippet') or paper.get('abstract') or ''}".strip()
            aid = paper.get("arxiv_id") or paper.get("paper_id")
            if not doc or not aid:
                continue
            hits.append({"arxiv_id": aid, "score": float(q_emb @ embedder.encode(doc)),
                         "metadata": {}, "paper": paper})
    return hits


# ---------------------------------------------------------------------------
# The agent factory
# ---------------------------------------------------------------------------


def create_scout_agent(
    llm,
    corpus: Sequence[dict] | None = None,
    embedder: Embedder | None = None,
    store: InMemoryVectorStore | None = None,
    applicability: Callable | None = None,
    live: bool = False,
    max_findings: int = 5,
    score_drop: int = DEFAULT_SCORE_DROP,
    score_fetch: int = 8,
    paper_finder: Callable | None = None,
    paper_finder_enabled: bool = False,
    paper_store=None,
    extraction: Callable | None = None,
    min_findings: int = 0,
    gate_top_k: int = 25,
):
    """Build the scout's ``run(state) -> dict`` callable.

    - ``corpus`` defaults to ``fixtures.mock_corpus.MOCK_PAPERS``; the vector
      index over it is built once, lazily, on first run.
    - ``applicability`` is ``callable(paper, analysis) -> PaperApplicability``;
      default is the per-paper LLM filter (spec §5.2 step 4).
    - ``live=True`` additionally enables arXiv/S2 search, citation walking, AND
      the v3 PaperFinder full-text channel (``paper_finder``, default
      ``retrieval.paper_finder_search``, which self-gates to [] without a key).
    - ``paper_store`` (v3): an optional ``retrieval.PaperStore`` that persists
      every live/PaperFinder paper across runs.
    - ``extraction`` (v3): optional ``query_text -> S2 fields string`` hook for
      dynamic per-query extraction (default keeps the fixed field set).
    - ``min_findings`` (v3): when > 0 and the first pass yields fewer than this,
      decompose the queries and re-run the funnel once (thin-retrieval fallback).
      Default 0 = disabled.
    - ``gate_top_k``: only the top-K candidates by combined re-rank score reach
      the per-paper applicability gate (one LLM call each), bounding LLM cost
      when live channels return large candidate sets. Default 25 (5x the
      default ``max_findings``, per the precision-over-recall contract).
    - Verdicts with ``score < score_drop`` (default ``DEFAULT_SCORE_DROP``, the
      bench ``--lit-threshold`` knob) are dropped; ``score >= score_fetch``
      attaches the paper's code example (preferred) or a method-section
      excerpt as the finding's ``code_snippet``.
    """
    cache: dict = {}

    def _corpus_papers() -> list[dict]:
        if "papers" not in cache:
            if corpus is not None:
                cache["papers"] = list(corpus)
            else:
                from compass.fixtures.mock_corpus import MOCK_PAPERS

                # Never let a production run be silently fixture-bound.
                logging.getLogger(__name__).warning(
                    "scout: no corpus supplied — falling back to the 15-paper "
                    "test fixture; pass corpus=<curated papers> and/or "
                    "live=True for production retrieval"
                )
                cache["papers"] = list(MOCK_PAPERS)
        return cache["papers"]

    def _index() -> tuple[InMemoryVectorStore, Embedder]:
        if "index" not in cache:
            if store is not None:
                if embedder is None:
                    raise ValueError(
                        "create_scout_agent(store=...) also needs the fitted "
                        "embedder that produced the store's vectors"
                    )
                cache["index"] = (store, embedder)
            else:
                cache["index"] = build_paper_index(_corpus_papers(), embedder)
        return cache["index"]

    finder_fn = paper_finder if paper_finder is not None else paper_finder_search
    # PaperFinder is OFF unless explicitly enabled or a custom finder is injected,
    # so a key set for graph-API auth alone does not silently activate it.
    use_finder = paper_finder_enabled or (paper_finder is not None)

    def run(state: WorkflowState) -> dict:
        analysis = state.analysis or {}
        if hasattr(analysis, "model_dump"):  # tolerate a stray pydantic object
            analysis = analysis.model_dump()

        vec_store, vec_embedder = _index()
        verdict_fn = applicability or _make_llm_applicability(llm, state, analysis)

        def _funnel(
            qs, exclude_ids: set, gated_ids: set
        ) -> list[tuple[PaperFinding, float]]:
            """One retrieval+rerank+gate pass over ``qs`` (skipping ``exclude_ids``).

            Every id sent to the applicability gate (accepted OR rejected) is
            added to ``gated_ids`` so a fallback pass never re-gates it.
            """
            hits = _vector_hits(qs, vec_store, vec_embedder, n_per_query=5)
            papers_by_id = {
                p.get("arxiv_id"): p for p in _corpus_papers() if p.get("arxiv_id")
            }
            if live:
                live_hits = _live_hits(qs, vec_embedder, extraction=extraction)
                seed_ids = [
                    h["arxiv_id"]
                    for h in sorted(hits, key=lambda h: -h["score"])[:2]
                    if h.get("arxiv_id")
                ]
                live_hits += _citation_walk_hits(seed_ids, qs, vec_embedder)
                if use_finder:  # channel 4 (off-by-default; see use_finder above)
                    live_hits += _paper_finder_hits(qs, vec_embedder, finder_fn)
                for h in live_hits:
                    paper = h.get("paper")
                    if paper and h["arxiv_id"] not in papers_by_id:
                        papers_by_id[h["arxiv_id"]] = paper
                    if paper_store is not None and paper and h.get("arxiv_id"):
                        paper_store.put(h["arxiv_id"], paper)  # persist across runs
                hits = hits + live_hits

            best = _dedup_best(hits)
            candidates = []
            for aid, hit in best.items():
                if aid in exclude_ids:
                    continue
                paper = papers_by_id.get(aid) or hit.get("paper") or _meta_to_paper(
                    hit.get("metadata") or {}
                )
                citations = paper.get("citationCount") or paper.get("citation_count")
                candidates.append({
                    "paper": paper, "cosine": hit["score"],
                    "combined": _combined_score(hit["score"], paper.get("year"), citations),
                })
            candidates.sort(key=lambda c: (
                -c["combined"], -c["cosine"], str(c["paper"].get("arxiv_id") or "")))
            # Gate only the top-K by combined score — the applicability gate is
            # one LLM round-trip per candidate, so this bounds LLM cost.
            candidates = candidates[: max(0, int(gate_top_k))]

            scored: list[tuple[PaperFinding, float]] = []
            for cand in candidates:
                paper = cand["paper"]
                # S2-only papers have no arXiv id; fall back to their S2
                # paper_id so cross-pass dedup/exclusion keeps their identity.
                finding_id = paper.get("arxiv_id") or paper.get("paper_id") or ""
                gated_ids.add(finding_id)
                verdict = verdict_fn(paper, analysis)
                if verdict.score < score_drop:
                    continue
                snippet = None
                if verdict.score >= score_fetch:
                    snippet = paper.get("code_example") or _method_excerpt(paper)
                scored.append((PaperFinding(
                    paper_title=str(paper.get("title") or ""),
                    arxiv_id=str(finding_id),
                    year=int(paper.get("year") or 0),
                    relevance_score=verdict.score / 10.0,
                    technique=_technique_of(paper),
                    applicable_to=verdict.what_would_change,
                    summary=_summary_of(paper),
                    code_snippet=snippet,
                ), cand["combined"]))
            return scored

        # (1) queries; (2)-(6) the funnel
        queries = generate_queries(analysis)
        gated_ids: set = set()
        scored_findings = _funnel(queries, exclude_ids=set(), gated_ids=gated_ids)

        # v3 thin-retrieval fallback: if too few survived, decompose + re-run
        # once, excluding EVERY pass-1 gated paper (accepted or rejected) so
        # no paper costs a second identical applicability LLM call.
        if min_findings and len(scored_findings) < min_findings:
            scored_findings += _funnel(
                decompose_queries(queries),
                exclude_ids={i for i in gated_ids if i},
                gated_ids=gated_ids,
            )

        scored_findings.sort(key=lambda t: (-t[0].relevance_score, -t[1]))
        # dedup by arxiv_id (decomposition can re-surface the same paper)
        seen_ids: set = set()
        deduped: list[tuple[PaperFinding, float]] = []
        for f, c in scored_findings:
            if f.arxiv_id and f.arxiv_id in seen_ids:
                continue
            seen_ids.add(f.arxiv_id)
            deduped.append((f, c))
        findings = [f for f, _ in deduped[: max(0, int(max_findings))]]

        state.literature_findings = [dataclasses.asdict(f) for f in findings]
        state.status = "searched"
        return {
            "literature_findings": state.literature_findings,
            "status": "searched",
        }

    return run


# ---------------------------------------------------------------------------
# Tools (spec §5.2) — built over the same internals with a module-level
# default corpus index; these serve live/react-style usage and orchestrator
# wiring. Default tests never invoke them (the S2/arXiv calls inside degrade
# gracefully to [] when offline or rate-limited).
# ---------------------------------------------------------------------------

_DEFAULT_INDEX_CACHE: dict = {}


def _default_index() -> tuple[InMemoryVectorStore, Embedder]:
    if "index" not in _DEFAULT_INDEX_CACHE:
        from compass.fixtures.mock_corpus import MOCK_PAPERS

        _DEFAULT_INDEX_CACHE["index"] = build_paper_index(MOCK_PAPERS)
        _DEFAULT_INDEX_CACHE["by_id"] = {p["arxiv_id"]: p for p in MOCK_PAPERS}
    return _DEFAULT_INDEX_CACHE["index"]


def _default_corpus_by_id() -> dict:
    _default_index()
    return _DEFAULT_INDEX_CACHE["by_id"]


def _paper_brief(paper: dict, combined: float) -> dict:
    return {
        "arxiv_id": paper.get("arxiv_id"),
        "title": paper.get("title") or "",
        "year": paper.get("year"),
        "technique": _technique_of(paper),
        "summary": _summary_of(paper, max_chars=300),
        "combined_score": round(float(combined), 4),
    }


# Spec §5.2 tool surface, retained for live react-style use; the funnel in
# create_scout_agent performs the equivalent retrieval directly (the
# default offline path needs no tool-calling LLM).
@tool
def search_bayesian_papers(query: str, technique_type: str, min_year: int = 2018) -> list[dict]:
    """Search the Bayesian methodology literature for a technique.

    Combines local vector search over the curated corpus with a Semantic
    Scholar search, and ranks results by relevance x recency x citations.
    """
    vec_store, embedder = _default_index()
    full_query = f"{query} {technique_type}".strip()
    best = _dedup_best(_vector_hits([full_query], vec_store, embedder, n_per_query=8))
    by_id = _default_corpus_by_id()
    results: dict[str, dict] = {}
    for aid, hit in best.items():
        paper = by_id.get(aid) or _meta_to_paper(hit.get("metadata") or {})
        year = paper.get("year")
        if year and year < min_year:
            continue
        combined = _combined_score(hit["score"], year, paper.get("citationCount"))
        results[aid] = _paper_brief(paper, combined)
    q_emb = embedder.encode(full_query)
    for paper in semantic_scholar_search(full_query, limit=5):  # [] when offline
        aid = paper.get("arxiv_id") or paper.get("paper_id")
        year = paper.get("year")
        if not aid or aid in results or (year and year < min_year):
            continue
        doc = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".strip()
        cosine = float(q_emb @ embedder.encode(doc)) if doc else 0.0
        results[aid] = _paper_brief(paper, _combined_score(cosine, year, paper.get("citationCount")))
    return sorted(results.values(), key=lambda r: -r["combined_score"])


@tool
def get_paper_details(arxiv_id: str) -> dict:
    """Fetch full paper details including the method section if available.

    Looks up the curated corpus first (which carries method sections and code
    examples), then falls back to the live arXiv API.
    """
    paper = _default_corpus_by_id().get(arxiv_id)
    if paper is not None:
        return dict(paper)
    fetched = arxiv_fetch(arxiv_id)  # None when offline / unknown id
    return fetched or {}


@tool
def get_citing_papers(arxiv_id: str, min_year: int = 2020) -> list[dict]:
    """Find papers that cite this one (forward citations = newer refinements).

    Backed by the Semantic Scholar Graph API; returns [] when offline or
    rate-limited.
    """
    return s2_citing_papers(f"arXiv:{arxiv_id}", min_year=min_year)
