"""Refinement Strategist tests (spec §5.3, §11 "Strategist tests").

Structural tests run on FixtureLLM (no API key, no network): they verify
prompt construction, state conventions, and the deterministic post-validation
pass. The research tree's live semantic test (@pytest.mark.llm) is an explicit
opt-in: it runs only when ``COMPASS_LIVE_TESTS`` is set, and conftest's
``isolated_env`` then leaves the provider keys and the ``.env`` lookup alone for
it; otherwise the package suite is offline by construction.
"""

from __future__ import annotations

import os

import pytest

from compass.agents.strategist import (
    EMPTY_BUNDLE_LOG_PREFIX,
    MAX_PROPOSALS,
    _banned_technique_names,
    create_strategist_agent,
    render_feedback,
)
from compass.fixtures.frozen_outputs import (
    FROZEN_PROPOSALS,
    PAPER_PRIORS,
)
from compass.fixtures.golden_model import GOLDEN_DATASET_SUMMARY, GOLDEN_PROBLEM
from compass.schemas import Proposal, StrategyOutput
from compass.state import WorkflowState

# The research conftest reads ANTHROPIC_API_KEY from code/.env to enable the live test. Here
# the opt-in is COMPASS_LIVE_TESTS (any non-empty value); the key itself then comes from the
# environment or from a .env file in the current working directory or one of its parents,
# the way get_llm finds it for a user. Without the variable the test is skipped.
requires_live_opt_in = pytest.mark.skipif(
    not os.environ.get("COMPASS_LIVE_TESTS"),
    reason="live-LLM test: set COMPASS_LIVE_TESTS=1 (and ANTHROPIC_API_KEY) to run it")

GOLDEN_PARAM_NAMES = {"mu_alpha", "sigma_alpha", "alpha", "beta", "sigma"}

HORSESHOE_FEEDBACK = [
    {
        "iteration": 1,
        "tried": "horseshoe_prior",
        "result": ["47 divergent transitions", "R-hat too high: 1.08 (tau)"],
        "waic_delta": None,
    }
]


def _golden_state(analysis_input, literature_input, **overrides) -> WorkflowState:
    # INPUT sites come from the conftest cascade fixtures (load_real_or_frozen)
    # so promoting fixtures/real_*.json upgrades these tests automatically.
    # FROZEN_* remains in use only for FixtureLLM canned outputs and exact
    # equality expectations.
    state = WorkflowState(
        problem_description=GOLDEN_PROBLEM,
        dataset_summary=GOLDEN_DATASET_SUMMARY,
        analysis=dict(analysis_input),
        literature_findings=[dict(f) for f in literature_input],
        status="searched",
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _mk_proposal(name, priority, change_type="prior", papers=None, parameter="beta"):
    return Proposal(
        name=name,
        description=f"Concrete refinement {name}: replace the prior on {parameter}.",
        change_type=change_type,
        changes=[
            {
                "parameter": parameter,
                "current": "Normal(0, 100)",
                "proposed": "Normal(0, 10)",
                "reason": "scale-appropriate weakly informative prior",
            }
        ],
        supporting_papers=papers if papers is not None else [PAPER_PRIORS],
        expected_improvement="better identified posterior, faster mixing",
        risk_assessment="low",
        priority=priority,
    )


def _mk_strategy(proposals) -> StrategyOutput:
    return StrategyOutput(
        reasoning="test reasoning",
        proposals=proposals,
        implementation_order=[p.name for p in proposals],
    )


# ---------------------------------------------------------------------------
# Structural happy path
# ---------------------------------------------------------------------------
def test_happy_path_proposals_stored_sorted_and_complete(
    make_fixture_llm, analysis_input, literature_input
):
    fake = make_fixture_llm(FROZEN_PROPOSALS)
    run = create_strategist_agent(fake)
    state = _golden_state(analysis_input, literature_input)

    update = run(state)

    assert state.status == "proposed"
    assert update["status"] == "proposed"
    assert "error_log" not in update  # nothing trimmed or dropped

    props = update["proposals"]
    assert props is state.proposals
    assert isinstance(props, list) and len(props) == 3
    assert all(isinstance(p, dict) for p in props)
    assert [p["priority"] for p in props] == [1, 2, 3]
    assert [p["name"] for p in props] == [p.name for p in FROZEN_PROPOSALS.proposals]
    assert "non_centered_parameterization" in [p["name"] for p in props]

    expected_fields = set(Proposal.model_fields)
    for p in props:
        assert set(p) == expected_fields  # all Proposal fields present
        assert len(p["supporting_papers"]) >= 1  # each cites >= 1 paper
        for change in p["changes"]:
            assert change["parameter"] in GOLDEN_PARAM_NAMES


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------
def test_prompt_carries_analysis_findings_and_problem(
    make_fixture_llm, analysis_input, literature_input
):
    fake = make_fixture_llm(FROZEN_PROPOSALS)
    run = create_strategist_agent(fake)

    run(_golden_state(analysis_input, literature_input))

    assert len(fake.calls) == 1
    prompt = fake.calls[0]
    assert "Normal(0, 100)" in prompt  # analysis content (beta's vague prior)
    # the literature findings actually fed in made it into the prompt
    assert literature_input[0]["paper_title"] in prompt
    assert GOLDEN_PROBLEM in prompt  # the problem text, verbatim
    assert GOLDEN_DATASET_SUMMARY in prompt  # dataset summary, verbatim


# ---------------------------------------------------------------------------
# Feedback / DO NOT PROPOSE
# ---------------------------------------------------------------------------
def test_feedback_history_renders_do_not_propose_section(
    make_fixture_llm, analysis_input, literature_input
):
    fake = make_fixture_llm(FROZEN_PROPOSALS)
    run = create_strategist_agent(fake)
    state = _golden_state(
        analysis_input,
        literature_input,
        feedback_history=[dict(e) for e in HORSESHOE_FEEDBACK],
    )

    run(state)

    prompt = fake.calls[0]
    assert "DO NOT PROPOSE" in prompt
    assert "horseshoe_prior" in prompt
    assert "47 divergent" in prompt


def test_render_feedback_unit():
    text = render_feedback(HORSESHOE_FEEDBACK)
    assert text.lstrip().startswith("- DO NOT PROPOSE:")
    assert "horseshoe_prior" in text
    assert "failed because" in text
    assert "47 divergent transitions" in text
    assert "R-hat too high: 1.08 (tau)" in text
    assert render_feedback([]) == ""


def test_render_feedback_entry_without_names_reports_the_round_not_a_ban():
    """graph.py writes "tried" explicitly as None when no proposal name exists (the
    feasibility gate's entry for a round whose bundle was emptied before synthesis).
    Such an entry bans nothing, so it renders as what happened last round — never a
    literal "DO NOT PROPOSE: None" ban and (P3) no longer a "DO NOT PROPOSE: unknown
    technique" placeholder either."""
    text = render_feedback([{"tried": None, "result": ["diverged"]}])
    assert text == "- Previous round: diverged"
    assert "DO NOT PROPOSE" not in text and "unknown technique" not in text
    # the empty-bundle entry exactly as graph.create_feasibility_gate books it
    summary = ("post-validation rejected every proposal: the LLM returned 3 proposal(s), all "
               "with an invalid change_type, no supporting papers, or a banned already-failed "
               "technique name; nothing to synthesize this round — 'horseshoe_prior' — "
               "re-proposes a technique recorded as failed in feedback_history "
               "(DO NOT PROPOSE ban)")
    entry = {"iteration": 2, "tried": None, "tried_all": [], "result": [summary],
             "waic_delta": None}
    assert render_feedback([entry]) == f"- Previous round: {summary}"
    # a mixed history: the ban bullet for the named round, the round line for the empty one
    lines = render_feedback(HORSESHOE_FEEDBACK + [entry]).splitlines()
    assert lines[0].startswith("- DO NOT PROPOSE: horseshoe_prior — failed because")
    assert lines[1] == f"- Previous round: {summary}"
    # no names and no result at all: still no placeholder ban
    assert render_feedback([{"tried": None, "result": []}]) == \
        "- Previous round: unspecified failure"


def test_render_feedback_names_every_banned_technique_once_in_order():
    """P2: the bullet used to show only ``tried`` while the deterministic ban covered
    every name in ``tried_all``. It now names them all — ``tried`` first, then the rest
    of the bundle — deduplicated case-insensitively, in order; and what is rendered is
    exactly what ``_banned_technique_names`` bans."""
    entry = {"iteration": 1, "tried": "alpha_fix",
             "tried_all": ["Alpha_Fix", "beta_fix", "gamma_fix"],
             "result": ["47 divergent transitions"]}
    text = render_feedback([entry])
    assert text == ("- DO NOT PROPOSE: alpha_fix, beta_fix, gamma_fix — failed because "
                    "47 divergent transitions")
    assert _banned_technique_names([entry]) == {"alpha_fix", "beta_fix", "gamma_fix"}
    # legacy shapes: a string-valued tried_all, and tried_all alone (tried None)
    assert render_feedback([{"tried": "x", "tried_all": "y", "result": "r"}]) == \
        "- DO NOT PROPOSE: x, y — failed because r"
    assert render_feedback([{"tried": None, "tried_all": ["only"], "result": []}]) == \
        "- DO NOT PROPOSE: only — failed because unspecified failure"


def test_prompt_names_every_banned_technique(
    make_fixture_llm, analysis_input, literature_input
):
    """P2 at the prompt level: every name the ban enforces reaches the strategist."""
    fake = make_fixture_llm(FROZEN_PROPOSALS)
    run = create_strategist_agent(fake)
    feedback = [dict(e) for e in HORSESHOE_FEEDBACK]
    feedback[0]["tried_all"] = ["horseshoe_prior", "bundled_failure", "third_wheel"]
    state = _golden_state(analysis_input, literature_input, feedback_history=feedback)

    run(state)

    prompt = fake.calls[0]
    assert "DO NOT PROPOSE: horseshoe_prior, bundled_failure, third_wheel — failed" in prompt
    for name in _banned_technique_names(feedback):
        assert name in prompt


# ---------------------------------------------------------------------------
# Post-validation pass
# ---------------------------------------------------------------------------
def test_more_than_five_valid_proposals_trimmed_to_five(
    make_fixture_llm, analysis_input, literature_input
):
    six = [_mk_proposal(f"proposal_{i}", priority=i) for i in range(1, 7)]
    fake = make_fixture_llm(_mk_strategy(six))
    run = create_strategist_agent(fake)
    state = _golden_state(analysis_input, literature_input)

    update = run(state)

    assert len(state.proposals) == MAX_PROPOSALS == 5
    assert [p["priority"] for p in state.proposals] == [1, 2, 3, 4, 5]
    assert "proposal_6" not in [p["name"] for p in state.proposals]
    assert "error_log" in update
    assert state.error_log, "trim must be logged"
    assert any("trim" in entry.lower() for entry in state.error_log)


def test_invalid_change_type_and_uncited_proposals_dropped(
    make_fixture_llm, analysis_input, literature_input
):
    proposals = [
        _mk_proposal("good_one", priority=1),
        _mk_proposal("magic_fix", priority=2, change_type="magic"),
        _mk_proposal("good_two", priority=3),
        _mk_proposal("uncited_fix", priority=4, papers=[]),
    ]
    fake = make_fixture_llm(_mk_strategy(proposals))
    run = create_strategist_agent(fake)
    state = _golden_state(analysis_input, literature_input)

    update = run(state)

    kept_names = [p["name"] for p in state.proposals]
    assert kept_names == ["good_one", "good_two"]
    assert "error_log" in update
    log_text = "\n".join(state.error_log)
    assert "magic_fix" in log_text  # invalid change_type drop logged
    assert "uncited_fix" in log_text  # empty supporting_papers drop logged


def test_uncited_proposals_are_kept_when_no_finding_reached_the_strategist(
    make_fixture_llm, analysis_input, literature_input
):
    """On real data the applicability gate can pass nothing; the citation rule is then
    unsatisfiable and must be waived (LLM-prior grounding), not crash the run."""
    proposals = [
        _mk_proposal("uncited_one", priority=1, papers=[]),
        _mk_proposal("magic_fix", priority=2, change_type="magic"),
        _mk_proposal("uncited_two", priority=3, papers=[]),
    ]
    fake = make_fixture_llm(_mk_strategy(proposals))
    run = create_strategist_agent(fake)
    state = _golden_state(analysis_input, literature_input)
    state.literature_findings = []

    update = run(state)

    assert [p["name"] for p in state.proposals] == ["uncited_one", "uncited_two"]
    assert all(p["supporting_papers"] == [] for p in state.proposals)
    assert all(p["evidence_basis"] != "literature" for p in state.proposals)
    log_text = "\n".join(update["error_log"])
    assert "citation rule is waived" in log_text and "magic_fix" in log_text
    # with findings present the rule still bites (the next test keeps raising)


def test_malformed_structured_reply_is_retried_with_a_nudge(
    make_fixture_llm, analysis_input, literature_input
):
    """A reply that does not fit StrategyOutput (empty tool call, proposals as a string) is
    retried with a schema reminder instead of crashing the run (Haiku census, 2026-10-08)."""
    good = _mk_strategy([_mk_proposal("good_one", priority=1)])
    fake = make_fixture_llm({}, good)
    run = create_strategist_agent(fake)
    state = _golden_state(analysis_input, literature_input)
    run(state)
    assert [p["name"] for p in state.proposals] == ["good_one"]
    assert len(fake.calls) == 2
    assert "did not fit the required output schema" not in fake.calls[0]
    assert "did not fit the required output schema" in fake.calls[1]
    assert "implementation_order" in fake.calls[1]


def test_persistently_malformed_structured_reply_raises_clearly(
    make_fixture_llm, analysis_input, literature_input
):
    fake = make_fixture_llm({})
    run = create_strategist_agent(fake)
    state = _golden_state(analysis_input, literature_input)
    with pytest.raises(RuntimeError, match="structured output.*invalid after 3 attempts"):
        run(state)
    assert len(fake.calls) == 3


def test_all_invalid_proposals_yield_an_empty_bundle_not_a_raise(
    make_fixture_llm, analysis_input, literature_input
):
    """P1 (was ``test_all_invalid_proposals_raises_runtime_error``): post-validation
    rejecting every proposal used to raise RuntimeError out of the graph — no report,
    candidate_history lost, a refused candidate booked as a pipeline error. The node
    now returns an EMPTY bundle plus the diagnostics; the graph's feasibility gate
    books the failed iteration (tests/test_graph_fixes.py)."""
    proposals = [
        _mk_proposal("magic_fix", priority=1, change_type="magic"),
        _mk_proposal("uncited_fix", priority=2, papers=[]),
    ]
    fake = make_fixture_llm(_mk_strategy(proposals))
    run = create_strategist_agent(fake)
    state = _golden_state(analysis_input, literature_input)

    update = run(state)

    assert update["proposals"] == [] and state.proposals == []
    assert "status" not in update and state.status == "searched"  # nothing proposed
    assert update["error_log"] is state.error_log
    log_text = "\n".join(state.error_log)
    assert "magic_fix" in log_text and "uncited_fix" in log_text  # per-drop lines kept
    summary = state.error_log[-1]
    assert summary.startswith(EMPTY_BUNDLE_LOG_PREFIX)
    assert "2 proposal(s)" in summary
    assert "'magic_fix'" in summary and "'uncited_fix'" in summary  # names + why, one line


def test_banned_failed_technique_dropped_deterministically(
    make_fixture_llm, analysis_input, literature_input
):
    # An LLM that ignores prompt rule 8 and re-proposes a failed technique
    # (any case) must be screened out deterministically; a materially renamed
    # variant is NOT fuzzy-matched and survives.
    proposals = [
        _mk_proposal("Horseshoe_Prior", priority=1),  # exact ban, case-insensitive
        _mk_proposal("regularized_horseshoe_prior", priority=2),  # variant: kept
        _mk_proposal("good_one", priority=3),
        _mk_proposal("bundled_failure", priority=4),  # banned via tried_all
    ]
    fake = make_fixture_llm(_mk_strategy(proposals))
    run = create_strategist_agent(fake)
    feedback = [dict(e) for e in HORSESHOE_FEEDBACK]
    feedback[0]["tried_all"] = ["horseshoe_prior", "bundled_failure"]
    state = _golden_state(analysis_input, literature_input, feedback_history=feedback)

    update = run(state)

    kept_names = [p["name"] for p in state.proposals]
    assert kept_names == ["regularized_horseshoe_prior", "good_one"]
    assert "error_log" in update
    log_text = "\n".join(state.error_log)
    assert "Horseshoe_Prior" in log_text
    assert "bundled_failure" in log_text
    assert "DO NOT PROPOSE" in log_text


def test_all_proposals_banned_yield_an_empty_bundle_not_a_raise(
    make_fixture_llm, analysis_input, literature_input
):
    """P1 (was ``test_all_proposals_banned_raises_runtime_error``): the proposer
    re-serving a banned name is the common trigger — a legitimately refused technique,
    not a pipeline error."""
    fake = make_fixture_llm(_mk_strategy([_mk_proposal("horseshoe_prior", priority=1)]))
    run = create_strategist_agent(fake)
    state = _golden_state(
        analysis_input,
        literature_input,
        feedback_history=[dict(e) for e in HORSESHOE_FEEDBACK],
    )

    update = run(state)

    assert update["proposals"] == [] and state.proposals == []
    summary = state.error_log[-1]
    assert summary.startswith(EMPTY_BUNDLE_LOG_PREFIX)
    assert "banned" in summary and "'horseshoe_prior'" in summary
    assert "DO NOT PROPOSE ban" in summary


def test_unsorted_priorities_are_stored_sorted(
    make_fixture_llm, analysis_input, literature_input
):
    proposals = [
        _mk_proposal("third", priority=3),
        _mk_proposal("first", priority=1),
        _mk_proposal("second", priority=2),
    ]
    fake = make_fixture_llm(_mk_strategy(proposals))
    run = create_strategist_agent(fake)
    state = _golden_state(analysis_input, literature_input)

    run(state)

    assert [p["priority"] for p in state.proposals] == [1, 2, 3]
    assert [p["name"] for p in state.proposals] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Live semantic test (opt-in: COMPASS_LIVE_TESTS=1 with ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------
@pytest.mark.llm
@requires_live_opt_in
def test_live_strategist_on_cascade_inputs_and_feedback(analysis_input, literature_input):
    from compass.llm import get_llm

    run = create_strategist_agent(get_llm())
    real_titles = [f["paper_title"] for f in literature_input]

    def cites_real_finding(cited: str) -> bool:
        c = cited.lower()
        return any(t.lower() in c or c in t.lower() for t in real_titles)

    # --- first iteration: cascade analysis + literature ---------------------
    state = _golden_state(analysis_input, literature_input)
    run(state)
    props = state.proposals

    assert 2 <= len(props) <= 5
    assert any(
        p["change_type"] == "hierarchy"
        or "non-centered" in (p["name"] + " " + p["description"]).lower()
        or "non_centered" in p["name"].lower()
        for p in props
    ), "expected a hierarchy-type (non-centered) proposal"
    for p in props:
        assert p["supporting_papers"], f"{p['name']} cites no papers"
        assert any(cites_real_finding(sp) for sp in p["supporting_papers"]), (
            f"{p['name']} cites no real finding title: {p['supporting_papers']}"
        )

    # --- second iteration: plain horseshoe already failed -------------------
    state2 = _golden_state(
        analysis_input,
        literature_input,
        feedback_history=[dict(e) for e in HORSESHOE_FEEDBACK],
    )
    run(state2)

    for p in state2.proposals:
        text = (p["name"] + " " + p["description"]).lower()
        if "horseshoe" in text:
            assert "regularized" in text, (
                "plain horseshoe re-proposed despite DO NOT PROPOSE feedback: "
                f"{p['name']}"
            )
