"""Graph-level pins for the pipeline defects fixed 2026-11-11:

* P1 — the strategist's post-validation rejecting its whole bundle no longer
  raises out of ``app.invoke`` / ``run_refinement``: the feasibility gate books
  the failed iteration and the run retries or ends in a report.
* P2 — the second-round prompt names EVERY banned technique.
* P3 — the DESIGN RESOLUTION block is memoized per model, not per compiled graph.
* P6 — the failure diagnosis distinguishes "the last code never ran" from
  "nothing reached the verifier" when earlier rounds were verified and refused.
* P8 — the heavy-tail soft-flag reads likelihood change rows only.

Round 2 (2026-11-12):

* P2 — the benchmark drivers (bench.run.run_pipeline, figure2_run.run_one) book an
  empty terminal round from the last verified candidate and mark it (research-only
  tests: see the note at the end of this file).
* P3 — zero refused candidates + an empty final bundle is diagnosed as "rejected
  before synthesis", and the iteration-history lines never show a literal `None`.
* P4 — the heavy-tail regex matches "StudentT_obs" / "student_t_likelihood".

Agents are FixtureLLM queues plus a stub verifier; nothing samples, nothing
calls an API. Helpers mirror tests/test_graph_flags.py (the package port of the
research integration suite).
"""

from __future__ import annotations

import ast
import uuid

from compass.agents.strategist import EMPTY_BUNDLE_LOG_PREFIX, _banned_technique_names
from compass.fixtures.frozen_outputs import (
    FROZEN_ANALYSIS,
    FROZEN_PROPOSALS,
    FROZEN_VALIDATION_FAIL,
    FROZEN_VALIDATION_PASS,
)
from compass.fixtures.golden_model import (
    GOLDEN_DATASET_SUMMARY,
    GOLDEN_PROBLEM,
    GOLDEN_REFINED_REFERENCE,
)
from compass.graph import (
    _HEAVY_TAIL_RE,
    build_html_report,
    build_refinement_graph,
    build_report,
    create_feasibility_gate,
    infer_data_shapes,
    route_after_feasibility,
    run_refinement,
)
from compass.llm import FixtureLLM
from compass.schemas import PaperApplicability, StrategyOutput, SynthesizedCode
from compass.state import ModelSpec, ValidationResult, WorkflowState

PROPOSAL_NAMES = [p.name for p in FROZEN_PROPOSALS.proposals]

# A global-parameter regression over the SAME data kwargs as the golden model:
# no latent plate, so its resolution audit reads "no local latent structure".
FLAT_MODEL = '''
import numpyro
import numpyro.distributions as dist

def model(X, group_idx, N_groups, y=None):
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    beta = numpyro.sample("beta", dist.Normal(0, 100))
    sigma = numpyro.sample("sigma", dist.HalfNormal(10))
    numpyro.sample("y", dist.Normal(alpha + X * beta, sigma), obs=y)
'''


# ---------------------------------------------------------------------------
# Helpers (mirroring tests/test_integration.py)
# ---------------------------------------------------------------------------
def retry_proposals(suffix: str = "_v2") -> StrategyOutput:
    """A renamed copy of FROZEN_PROPOSALS for a later strategist round: the feedback
    entry's ``tried_all`` bans the WHOLE applied bundle, so only new names survive."""
    dumped = FROZEN_PROPOSALS.model_dump()
    dumped["proposals"] = [{**p, "name": p["name"] + suffix} for p in dumped["proposals"]]
    dumped["implementation_order"] = [n + suffix for n in dumped["implementation_order"]]
    return StrategyOutput.model_validate(dumped)


def heuristic_applicability(paper: dict, analysis: dict) -> PaperApplicability:
    relevant = paper.get("category") == "relevant"
    return PaperApplicability(
        score=8 if relevant else 3,
        addresses_weakness=relevant,
        framework_compatible=True,
        data_size_appropriate=True,
        scope="component",
        effort_to_improvement="low",
        what_would_change="reparameterize the group intercepts",
    )


def ast_only_validator(code: str, data_shapes: dict) -> dict:
    try:
        ast.parse(code)
        return {"compiled": True, "stage": "ast", "stdout": "", "stderr": ""}
    except SyntaxError as exc:
        return {"compiled": False, "stage": "ast", "stdout": "", "stderr": f"SyntaxError: {exc}"}


def stub_validation(results: list[ValidationResult]):
    """validation_fn returning the queued results in order (the last one repeats)."""
    queue = list(results)

    def fn(refined_code, original_code, data, **kwargs):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return fn


def fixture_agent_kwargs(**overrides) -> dict:
    kwargs = dict(
        analyst_llm=FixtureLLM(outputs=[FROZEN_ANALYSIS]),
        scout_llm=FixtureLLM(outputs=[]),  # never consulted: heuristic filter below
        strategist_llm=FixtureLLM(outputs=[FROZEN_PROPOSALS]),
        synthesizer_llm=FixtureLLM(outputs=[SynthesizedCode(
            code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES)]),
        scout_kwargs={"applicability": heuristic_applicability},
        synthesizer_kwargs={"validator": ast_only_validator},
        validator_kwargs={"validation_fn": stub_validation([FROZEN_VALIDATION_PASS])},
    )
    kwargs.update(overrides)
    return kwargs


def make_state(model_code: str, golden_data: dict, max_iterations: int = 3) -> WorkflowState:
    return WorkflowState(
        problem_description=GOLDEN_PROBLEM,
        dataset_summary=GOLDEN_DATASET_SUMMARY,
        original_model=ModelSpec(
            framework="numpyro",
            source_code=model_code,
            data_shapes=infer_data_shapes(golden_data),
        ),
        max_iterations=max_iterations,
    )


def invoke(app, state):
    return app.invoke(state, {"configurable": {"thread_id": uuid.uuid4().hex},
                              "recursion_limit": 50})


def graph_with(golden_data, *, strategist_outputs, validations, **kwargs):
    strategist = FixtureLLM(outputs=list(strategist_outputs))
    app = build_refinement_graph(
        **fixture_agent_kwargs(
            strategist_llm=strategist,
            validator_kwargs={"validation_fn": stub_validation(validations)},
        ),
        data=golden_data, **kwargs,
    )
    return app, strategist


# ---------------------------------------------------------------------------
# P1: an empty bundle from the strategist is booked, not raised
# ---------------------------------------------------------------------------
class TestEmptyBundleFromStrategist:
    def test_rejected_bundle_books_an_iteration_and_the_run_retries(
        self, golden_model_code, golden_data
    ):
        """Round 1: FROZEN_PROPOSALS is refused, so its names are banned. Round 2: the
        proposer re-serves the SAME names -> every one is dropped -> empty bundle. Round
        3: renamed proposals -> accepted. The whole thing used to die in round 2 with a
        RuntimeError out of app.invoke."""
        app, strategist = graph_with(
            golden_data,
            strategist_outputs=[FROZEN_PROPOSALS, FROZEN_PROPOSALS, retry_proposals()],
            validations=[FROZEN_VALIDATION_FAIL, FROZEN_VALIDATION_PASS])
        out = invoke(app, make_state(golden_model_code, golden_data, max_iterations=3))

        assert out["status"] == "complete"
        assert len(strategist.calls) == 3
        # the empty round was booked as iteration 2 by the feasibility gate ...
        fb = out["feedback_history"]
        assert [e["iteration"] for e in fb] == [1, 2]
        assert fb[0]["tried_all"] == PROPOSAL_NAMES
        assert fb[1]["tried"] is None and fb[1]["tried_all"] == []  # nothing new banned
        assert len(fb[1]["result"]) == 1
        assert fb[1]["result"][0].startswith("post-validation rejected every proposal")
        assert f"'{PROPOSAL_NAMES[0]}'" in fb[1]["result"][0]  # says which, and why
        assert "DO NOT PROPOSE ban" in fb[1]["result"][0]
        # ... only the two candidates that reached the verifier are on record ...
        assert [h["passed"] for h in out["candidate_history"]] == [False, True]
        assert out["candidate_history"][1]["proposal_names"] == [n + "_v2" for n in PROPOSAL_NAMES]
        # ... the third prompt tells the proposer why round 2 produced nothing ...
        assert "rejected every proposal" in strategist.calls[2]
        assert f"DO NOT PROPOSE: {', '.join(PROPOSAL_NAMES)}" in strategist.calls[2]
        # ... and the diagnostics are in error_log (per-drop lines + the summary).
        assert any(e.startswith(EMPTY_BUNDLE_LOG_PREFIX) for e in out["error_log"])
        assert sum("DO NOT PROPOSE ban" in e for e in out["error_log"]) >= len(PROPOSAL_NAMES)

    def test_rejected_bundle_with_no_budget_left_ends_in_a_report(
        self, golden_model_code, golden_data
    ):
        """Through run_refinement (the benchmark drivers' entry point): the proposer
        re-serves the banned names forever; with max_iterations=2 the empty round
        exhausts the budget and the run ends failed WITH a report and the history."""
        kwargs = fixture_agent_kwargs(
            strategist_llm=FixtureLLM(outputs=[FROZEN_PROPOSALS]),  # repeats forever
            validator_kwargs={"validation_fn": stub_validation([FROZEN_VALIDATION_FAIL])})
        out = run_refinement(GOLDEN_PROBLEM, GOLDEN_DATASET_SUMMARY, golden_model_code,
                             framework="numpyro", data=golden_data, max_iterations=2,
                             **kwargs)

        assert out["status"] == "failed"
        assert out["iteration"] == 2
        assert len(kwargs["strategist_llm"].calls) == 2
        assert out["proposals"] == []
        assert [h["passed"] for h in out["candidate_history"]] == [False]
        assert out["candidate_history"][0]["validation"] is FROZEN_VALIDATION_FAIL
        report = out["report"]
        assert "## Why no model was accepted" in report
        assert "proposed nothing that survived" in report
        assert "1 earlier candidate was verified and refused" in report
        assert "47 divergent transitions" in report
        assert "before any refinement was proposed" not in report
        assert EMPTY_BUNDLE_LOG_PREFIX in report  # Errors section carries the summary

    def test_feasibility_gate_books_an_empty_bundle_with_the_strategist_summary(self):
        summary = (f"{EMPTY_BUNDLE_LOG_PREFIX}: the LLM returned 1 proposal(s), all with "
                   "an invalid change_type, no supporting papers, or a banned already-failed "
                   "technique name; nothing to synthesize this round — 'x' — re-proposes a "
                   "technique recorded as failed in feedback_history (DO NOT PROPOSE ban)")
        state = WorkflowState(proposals=[], iteration=1, max_iterations=3,
                              error_log=["older entry", summary])
        update = create_feasibility_gate()(state)

        assert update["iteration"] == 2 and update["proposals"] == []
        entry = update["feedback_history"][-1]
        assert entry["iteration"] == 2
        assert entry["tried"] is None and entry["tried_all"] == []
        assert entry["result"] == [summary.split("strategist: ", 1)[1]]
        assert "error_log" not in update  # the gate itself dropped nothing
        assert route_after_feasibility(state) == "retry_with_feedback"

        # a bundle emptied some other way still gets an explanatory entry
        bare = WorkflowState(proposals=[], iteration=2, max_iterations=3)
        update = create_feasibility_gate()(bare)
        assert update["feedback_history"][-1]["result"] == [
            "the strategist delivered no proposals; nothing reached synthesis"]
        assert route_after_feasibility(bare) == "fail"


# ---------------------------------------------------------------------------
# P2: the second-round prompt names every banned technique
# ---------------------------------------------------------------------------
def test_second_round_prompt_names_every_banned_technique(golden_model_code, golden_data):
    app, strategist = graph_with(
        golden_data, strategist_outputs=[FROZEN_PROPOSALS, retry_proposals()],
        validations=[FROZEN_VALIDATION_FAIL, FROZEN_VALIDATION_PASS])
    out = invoke(app, make_state(golden_model_code, golden_data))

    assert out["status"] == "complete"
    banned = _banned_technique_names(out["feedback_history"])
    assert banned == {n.lower() for n in PROPOSAL_NAMES} and len(banned) == 3
    prompt = strategist.calls[1]
    assert f"DO NOT PROPOSE: {', '.join(PROPOSAL_NAMES)} — failed because" in prompt
    for name in PROPOSAL_NAMES:
        assert name in prompt
    assert "unknown technique" not in prompt


# ---------------------------------------------------------------------------
# P3: the DESIGN RESOLUTION memo is keyed per model, not per compiled graph
# ---------------------------------------------------------------------------
def test_resolution_block_follows_the_model_across_invocations_of_one_app(
    golden_model_code, golden_data
):
    app, strategist = graph_with(
        golden_data, strategist_outputs=[FROZEN_PROPOSALS],
        validations=[FROZEN_VALIDATION_PASS])
    hier = invoke(app, make_state(golden_model_code, golden_data, max_iterations=1))
    flat = invoke(app, make_state(FLAT_MODEL, golden_data, max_iterations=1))

    assert hier["status"] == flat["status"] == "complete"
    # each report carries ITS model's audit ...
    assert "## Design resolution" in hier["report"] and "## Design resolution" in flat["report"]
    assert "latent plate 'groups'" in hier["report"]
    assert "no local latent structure detected" not in hier["report"]
    assert "no local latent structure detected" in flat["report"]
    assert "latent plate 'groups'" not in flat["report"]
    # ... and so did each run's strategist prompt
    assert "latent plate 'groups'" in strategist.calls[0]
    assert "no local latent structure detected" in strategist.calls[1]
    assert "latent plate 'groups'" not in strategist.calls[1]


# ---------------------------------------------------------------------------
# P6: the failure diagnosis with verified-and-refused rounds on record
# ---------------------------------------------------------------------------
class TestFailureDiagnosis:
    @staticmethod
    def _state(**overrides) -> WorkflowState:
        state = WorkflowState(
            problem_description="p", dataset_summary="d", status="failed",
            proposals=[{"name": "late_fix", "change_type": "prior", "description": "x",
                        "supporting_papers": [], "priority": 1}],
            refined_model_code="",
            error_log=["synthesizer: 3 attempts exhausted; last error: FAILURE: "
                       "ValueError: shapes"],
        )
        for key, value in overrides.items():
            setattr(state, key, value)
        return state

    @staticmethod
    def _refused(names, reasons, iteration=0, converged=False) -> dict:
        return {"iteration": iteration, "refined_model_code": "def model(): ...",
                "proposal_names": list(names), "proposals": [], "rationale": "",
                "passed": False,
                "validation": ValidationResult(converged=converged, failure_reasons=reasons)}

    def test_without_history_keeps_the_completeness_wording(self):
        report = build_report(self._state())
        assert "nothing reached the verifier" in report
        assert "verified and refused" not in report
        assert "- Proposed: late_fix" in report
        assert "Synthesis blocker: `ValueError: shapes`" in report

    def test_with_refused_rounds_only_the_last_code_never_ran(self):
        one = self._refused(["early_fix", "other_fix"],
                            ["47 divergent transitions", "R-hat too high: 1.08 (tau)"])
        report = build_report(self._state(candidate_history=[one]))
        assert "nothing reached the verifier" not in report
        assert "1 earlier candidate was verified and refused" in report
        assert "Proposed last (code never ran): late_fix" in report
        assert "Synthesis blocker: `ValueError: shapes`" in report
        assert ("Last refused candidate: early_fix, other_fix — arbiter's reasons: "
                "47 divergent transitions; R-hat too high: 1.08 (tau)") in report

    def test_two_refused_rounds_name_the_last_refusals_reasons(self):
        first = self._refused(["early_fix"], ["47 divergent transitions"])
        second = self._refused(["second_fix"], ["no significant improvement"],
                               iteration=1, converged=True)
        report = build_report(self._state(candidate_history=[first, second]))
        assert "2 earlier candidates were verified and refused" in report
        assert ("Last refused candidate: second_fix — arbiter's reasons: "
                "no significant improvement") in report
        assert "47 divergent transitions" not in report.split("## Why no model was accepted")[1] \
            .split("## Problem")[0]

    def test_dict_valued_validation_is_read_the_same_way(self):
        entry = {**self._refused(["fix"], []), "validation": {"failure_reasons": ["ppc failed"]}}
        report = build_report(self._state(candidate_history=[entry]))
        assert "Last refused candidate: fix — arbiter's reasons: ppc failed" in report

    def test_html_export_carries_the_same_diagnosis(self):
        from compass.graph import build_html_report

        one = self._refused(["early_fix"], ["47 divergent transitions"])
        html = build_html_report(self._state(candidate_history=[one]))
        assert "1 earlier candidate was verified and refused" in html
        assert "nothing reached the verifier" not in html


# ---------------------------------------------------------------------------
# P8: the heavy-tail soft-flag reads likelihood change rows only
# ---------------------------------------------------------------------------
class TestFeasibilityHeavyTailFlag:
    ANALYSIS = {
        "parameters": [{"name": "beta"}, {"name": "sigma"}, {"name": "sigma_alpha"}],
        "likelihood": {"family": "Normal", "link_function": "identity",
                       "observed_variable": "y"},
    }

    def _run(self, proposal: dict, dataset_summary: str = "N=280. y: mean=72, sd=15"):
        state = WorkflowState(problem_description="Predict test scores.",
                              dataset_summary=dataset_summary,
                              analysis=dict(self.ANALYSIS), proposals=[proposal],
                              iteration=0, max_iterations=3)
        update = create_feasibility_gate()(state)
        return state, update

    @staticmethod
    def _flags(state: WorkflowState) -> list[str]:
        return [e for e in state.error_log if "SOFT-FLAG" in e]

    def test_half_cauchy_scale_prior_is_not_flagged(self):
        prior = {"name": "half_cauchy_scale", "change_type": "prior",
                 "changes": [{"parameter": "sigma_alpha", "current": "HalfNormal(10)",
                              "proposed": "HalfCauchy(2.5)",
                              "reason": "weakly informative scale prior"}]}
        state, update = self._run(prior)
        assert state.proposals == [prior]
        assert "error_log" not in update and self._flags(state) == []
        # other spellings of the same prior
        for spelled in ("dist.HalfCauchy(2.5)", "half_cauchy(2.5)", "Half-Cauchy(2.5)",
                        "half cauchy with scale 2.5"):
            state, _ = self._run({**prior, "changes": [{**prior["changes"][0],
                                                          "proposed": spelled}]})
            assert self._flags(state) == [], spelled

    def test_student_t_prior_on_a_coefficient_is_not_flagged(self):
        prior = {"name": "robust_slope_prior", "change_type": "prior",
                 "changes": [{"parameter": "beta", "current": "Normal(0, 100)",
                              "proposed": "StudentT(3, 0, 2.5)", "reason": "robust prior"}]}
        state, _ = self._run(prior)
        assert state.proposals == [prior] and self._flags(state) == []

    def test_cauchy_likelihood_is_flagged_and_kept(self):
        lik = {"name": "cauchy_likelihood", "change_type": "likelihood",
               "changes": [{"parameter": "y", "current": "Normal(mu, sigma)",
                            "proposed": "Cauchy(mu, sigma)", "reason": "robustness"}]}
        state, update = self._run(lik)
        assert state.proposals == [lik]  # soft: the proposal stays
        assert len(self._flags(state)) == 1 and "'cauchy_likelihood'" in self._flags(state)[0]
        assert update["error_log"] is state.error_log

    def test_student_t_on_the_observed_variable_is_flagged_whatever_the_change_type(self):
        structure = {"name": "robust_observation_model", "change_type": "structure",
                     "changes": [{"parameter": "y", "current": "Normal(mu, sigma)",
                                  "proposed": "dist.StudentT(nu, mu, sigma)",
                                  "reason": "r"}]}
        state, _ = self._run(structure)
        assert len(self._flags(state)) == 1

    def test_flag_is_suppressed_when_the_data_show_heavy_tails(self):
        lik = {"name": "cauchy_likelihood", "change_type": "likelihood",
               "changes": [{"parameter": "y", "current": "Normal(mu, sigma)",
                            "proposed": "Cauchy(mu, sigma)", "reason": "robustness"}]}
        state, _ = self._run(lik, dataset_summary="N=280. y has several extreme outliers")
        assert self._flags(state) == []

    def test_student_t_spellings_ending_in_an_underscore_or_suffix_are_flagged(self):
        """P4: "StudentT_obs" / "student_t_likelihood" end the family name with "_", where
        the old trailing \\b never matched; a trailing "t" followed by a non-letter or
        the end now does, while "students" / "studentized" and HalfCauchy stay clear."""
        row = {"parameter": "y", "current": "Normal(mu, sigma)", "reason": "robustness"}
        for spelled in ("StudentT_obs(nu, mu, sigma)", "student_t_likelihood(mu, sigma, nu=4)",
                        "dist.StudentT(3, mu, sigma)", "Student's t with nu=4",
                        "student-t(mu, sigma)", "y ~ student t"):
            lik = {"name": "robust", "change_type": "likelihood",
                   "changes": [{**row, "proposed": spelled}]}
            state, _ = self._run(lik)
            assert len(self._flags(state)) == 1, spelled
        for spelled in ("students(mu)", "studentized residuals", "HalfCauchy(2.5)",
                        "half_cauchy_scale(2.5)", "Half-Cauchy(2.5)", "half cauchy(2.5)"):
            lik = {"name": "not_heavy", "change_type": "likelihood",
                   "changes": [{**row, "proposed": spelled}]}
            state, _ = self._run(lik)
            assert self._flags(state) == [], spelled

    def test_heavy_tail_regex_by_hand(self):
        for text in ("studentt_obs", "student_t_likelihood", "studentt(", "student-t", "student t",
                     "student's t", "cauchy(mu, sigma)", "x_cauchy"):
            assert _HEAVY_TAIL_RE.search(text), text
        for text in ("students", "studentized", "halfcauchy", "half_cauchy", "half-cauchy",
                     "half cauchy", "normal(mu, sigma)"):
            assert not _HEAVY_TAIL_RE.search(text), text


# ---------------------------------------------------------------------------
# P3 (round 2): zero verified candidates and an empty final bundle
# ---------------------------------------------------------------------------
class TestEmptyRoundsDiagnosis:
    SUMMARY = (f"{EMPTY_BUNDLE_LOG_PREFIX}: the LLM returned 2 proposal(s), all with an "
               "invalid change_type, no supporting papers, or a banned already-failed "
               "technique name; nothing to synthesize this round — 'horseshoe_prior' — "
               "empty supporting_papers (every proposal must cite >= 1 finding)")
    DROPPED = ("strategist: dropped proposal 'horseshoe_prior' — empty supporting_papers "
               "(every proposal must cite >= 1 finding)")

    @staticmethod
    def _state(**overrides) -> WorkflowState:
        state = WorkflowState(problem_description="p", dataset_summary="d", status="failed",
                              proposals=[], refined_model_code="", iteration=2, max_iterations=2)
        for key, value in overrides.items():
            setattr(state, key, value)
        return state

    @staticmethod
    def _diagnosis(report: str) -> str:
        return report.split("## Why no model was accepted")[1].split("## Problem")[0]

    def test_every_round_rejected_before_synthesis_is_said_so(self):
        gate_entry = {"iteration": 1, "tried": None, "tried_all": [],
                      "result": [self.SUMMARY.split("strategist: ", 1)[1]], "waic_delta": None}
        state = self._state(error_log=[self.DROPPED, self.SUMMARY], feedback_history=[gate_entry])
        report = build_report(state)
        diag = self._diagnosis(report)
        assert ("Every round's proposals were rejected before synthesis (post-validation or "
                "the feasibility gate); see Errors for what was dropped and why.") in diag
        assert f"- Last round: {self.SUMMARY.split('strategist: ', 1)[1]}" in diag
        assert "before any refinement was proposed" not in report
        assert self.DROPPED in report  # the Errors section carries the drops
        # the iteration-history line never shows a literal None
        assert "- iteration 1: tried (no technique) -> post-validation rejected" in report
        assert "tried `None`" not in report
        html = build_html_report(state)
        assert "rejected before synthesis" in html and "(no technique)" in html
        assert "<code>None</code>" not in html

    def test_the_last_summary_line_is_quoted(self):
        older = self.SUMMARY.replace("2 proposal(s)", "4 proposal(s)")
        report = build_report(self._state(error_log=[older, self.DROPPED, self.SUMMARY]))
        diag = self._diagnosis(report)
        assert "2 proposal(s)" in diag and "4 proposal(s)" not in diag

    def test_feedback_entry_alone_is_recognised(self):
        """A run whose error_log was trimmed still diagnoses from the gate's entry."""
        entry = {"iteration": 1, "tried": None, "tried_all": [],
                 "result": ["post-validation rejected every proposal: nothing to synthesize"]}
        report = build_report(self._state(feedback_history=[entry]))
        assert "rejected before synthesis" in report
        assert "- Last round: post-validation rejected every proposal: nothing to synthesize" in report

    def test_gate_dropping_every_proposal_is_recognised(self):
        entry = {"iteration": 1, "tried": "phantom_fix", "tried_all": ["phantom_fix"],
                 "result": ["pre-MCMC feasibility gate dropped every proposal",
                            "feasibility: dropped 'phantom_fix' — `prior` proposal retunes "
                            "parameter(s) ['ghost'] not present in the analysis"]}
        report = build_report(self._state(feedback_history=[entry]))
        assert "rejected before synthesis" in report
        assert "- Last round: pre-MCMC feasibility gate dropped every proposal" in report
        assert "tried `phantom_fix` ->" in report  # a named round keeps its name

    def test_nothing_proposed_at_all_keeps_the_old_wording(self):
        report = build_report(self._state())
        assert "before any refinement was proposed" in report
        # a refused candidate on record still routes to the P1 wording, not this one
        refused = {"iteration": 0, "refined_model_code": "def model(): ...",
                   "proposal_names": ["early_fix"], "proposals": [], "rationale": "",
                   "passed": False,
                   "validation": ValidationResult(converged=False, failure_reasons=["diverged"])}
        report = build_report(self._state(candidate_history=[refused], error_log=[self.SUMMARY]))
        assert "proposed nothing that survived" in report
        assert "Every round's proposals were rejected" not in report

    def test_through_the_graph_every_round_uncited(self, golden_model_code, golden_data):
        """FixtureLLM strategist whose every proposal is uncited while the scout found
        papers: post-validation empties the bundle in both rounds, nothing is ever
        synthesized, and the report + second prompt say so without placeholders."""
        dumped = FROZEN_PROPOSALS.model_dump()
        for p in dumped["proposals"]:
            p["supporting_papers"] = []
        uncited = StrategyOutput.model_validate(dumped)
        app, strategist = graph_with(golden_data, strategist_outputs=[uncited],
                                     validations=[FROZEN_VALIDATION_PASS])
        out = invoke(app, make_state(golden_model_code, golden_data, max_iterations=2))

        assert out["status"] == "failed" and out["candidate_history"] == []
        assert out["proposals"] == [] and len(strategist.calls) == 2
        assert [e["tried"] for e in out["feedback_history"]] == [None, None]
        report = out["report"]
        assert ("Every round's proposals were rejected before synthesis (post-validation or "
                "the feasibility gate); see Errors for what was dropped and why.") in report
        assert "- Last round: post-validation rejected every proposal" in report
        assert "before any refinement was proposed" not in report
        assert "tried (no technique) ->" in report and "`None`" not in report
        assert "- Previous round: post-validation rejected every proposal" in strategist.calls[1]
        assert "unknown technique" not in strategist.calls[1]
        assert "DO NOT PROPOSE:" not in strategist.calls[1]  # nothing was banned


# Not ported: the research tests of P2 (round 2) — ``test_bench_run_pipeline_books_an_empty_
# terminal_round_from_the_last_candidate``, ``test_bench_run_pipeline_marks_an_ordinary_run_
# with_terminal_round_none`` and ``test_figure2_run_one_books_an_empty_terminal_round_from_
# the_last_candidate`` — drive the benchmark's own ``bench.run.run_pipeline`` and
# ``figure2.figure2_run.run_one``, which the package does not ship (the figure pipeline's copy
# lives in figures/figure2 and is tested there). The graph behaviour they rest on (an empty final
# bundle after a refused round) is covered above through the real graph.
