"""Graph wiring for the 0.2.0 flags: ``verdict_only``, ``smoke_on_data``,
``guard_replicate_sites``, the ``candidate_history`` channel and the custom
``validation_fn`` contract. Agents are driven by FixtureLLM queues and a stub
verifier; nothing samples and nothing calls an API.
"""

from __future__ import annotations

import ast
import uuid

import numpy as np

import compass.graph as graph_mod
from compass.agents.strategist import VERDICT_ONLY_NOTICE, render_feedback
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
from compass.graph import build_refinement_graph, infer_data_shapes, run_refinement
from compass.llm import FixtureLLM, track_usage
from compass.schemas import PaperApplicability, StrategyOutput, SynthesizedCode
from compass.state import ModelSpec, ValidationResult, WorkflowState

PROPOSAL_NAMES = [p.name for p in FROZEN_PROPOSALS.proposals]

REDACTED_ENTRY = {"tried": None, "tried_all": [], "result": [], "waic_delta": None,
                  "confidence": None, "failed_predictions": [], "hints": [],
                  "verdict_only": True}

NO_GAIN_WITH_HINT = ValidationResult(
    converged=True, rhat_max=1.002, ess_min=900, waic_delta=0.4, loo_delta=0.3,
    failure_reasons=["no significant improvement: delta-eLPD 3.1 (se 2.0)"],
    misspecification_hints=["HINT-MARK: the variance exceeds the mean; try a dispersion parameter"],
)


# ---------------------------------------------------------------------------
# Helpers (ported from the research integration suite)
# ---------------------------------------------------------------------------


def retry_proposals(suffix: str = "_v2") -> StrategyOutput:
    """A renamed copy of FROZEN_PROPOSALS for second strategist rounds: the feedback
    entry's ``tried_all`` bans the WHOLE applied bundle, so a retry that re-serves the
    same names is (correctly) rejected wholesale."""
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
    """validation_fn returning the queued results in order (last one repeats)."""
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


def make_state(golden_model_code: str, golden_data: dict, max_iterations: int = 3) -> WorkflowState:
    return WorkflowState(
        problem_description=GOLDEN_PROBLEM,
        dataset_summary=GOLDEN_DATASET_SUMMARY,
        original_model=ModelSpec(
            framework="numpyro",
            source_code=golden_model_code,
            data_shapes=infer_data_shapes(golden_data),
        ),
        max_iterations=max_iterations,
    )


def invoke(app, state):
    return app.invoke(state, {"configurable": {"thread_id": uuid.uuid4().hex},
                              "recursion_limit": 50})


def two_round_graph(golden_data, *, validations, **kwargs):
    strategist = FixtureLLM(outputs=[FROZEN_PROPOSALS, retry_proposals()])
    app = build_refinement_graph(
        **fixture_agent_kwargs(
            strategist_llm=strategist,
            validator_kwargs={"validation_fn": stub_validation(validations)},
        ),
        data=golden_data, **kwargs,
    )
    return app, strategist


# ---------------------------------------------------------------------------
# Construction and synthesizer kwargs forwarding
# ---------------------------------------------------------------------------


def test_graph_compiles_with_every_flag_on():
    app = build_refinement_graph(llm=FixtureLLM(outputs=["unused"]), data={"y": np.zeros(4)},
                                 verdict_only=True, smoke_on_data=True,
                                 guard_replicate_sites=True)
    assert app is not None


class TestSynthesizerKwargs:
    @staticmethod
    def _capture(monkeypatch) -> list[dict]:
        seen: list[dict] = []

        def fake_create(llm, **kwargs):
            seen.append(kwargs)
            return lambda state: {}

        monkeypatch.setattr(graph_mod, "create_synthesizer_agent", fake_create)
        return seen

    def test_default_passes_no_data(self, monkeypatch, golden_data):
        seen = self._capture(monkeypatch)
        build_refinement_graph(llm=FixtureLLM(outputs=[]), data=golden_data,
                               synthesizer_kwargs={"max_attempts": 2})
        assert seen == [{"max_attempts": 2}]

    def test_smoke_on_data_passes_the_same_object_and_explicit_kwargs_win(
        self, monkeypatch, golden_data
    ):
        seen = self._capture(monkeypatch)
        build_refinement_graph(llm=FixtureLLM(outputs=[]), data=golden_data, smoke_on_data=True,
                               synthesizer_kwargs={"max_attempts": 2})
        assert set(seen[0]) == {"data", "max_attempts"} and seen[0]["data"] is golden_data
        other = {"y": np.zeros(3)}
        build_refinement_graph(llm=FixtureLLM(outputs=[]), data=golden_data, smoke_on_data=True,
                               synthesizer_kwargs={"data": other})
        assert seen[1] == {"data": other} and seen[1]["data"] is other
        # no data at all: nothing to smoke on, kwargs untouched
        build_refinement_graph(llm=FixtureLLM(outputs=[]), smoke_on_data=True)
        assert seen[2] == {}

    def test_guard_replicate_sites_is_forwarded_and_explicit_kwargs_win(self, monkeypatch):
        seen = self._capture(monkeypatch)
        build_refinement_graph(llm=FixtureLLM(outputs=[]))
        assert seen[0] == {}
        build_refinement_graph(llm=FixtureLLM(outputs=[]), guard_replicate_sites=True)
        assert seen[1] == {"guard_replicate_sites": True}
        build_refinement_graph(llm=FixtureLLM(outputs=[]), guard_replicate_sites=True,
                               synthesizer_kwargs={"guard_replicate_sites": False})
        assert seen[2] == {"guard_replicate_sites": False}

    def test_run_refinement_forwards_both_flags(self, monkeypatch, golden_model_code,
                                                golden_data):
        seen = self._capture(monkeypatch)
        kwargs = fixture_agent_kwargs(smoke_on_data=True, guard_replicate_sites=True)
        del kwargs["synthesizer_kwargs"]
        final = run_refinement(GOLDEN_PROBLEM, model_code=golden_model_code, data=golden_data,
                               max_iterations=1, **kwargs)
        assert seen[0]["data"] is golden_data and seen[0]["guard_replicate_sites"] is True
        assert final["status"] == "failed"  # the stub synthesizer yields no code


# ---------------------------------------------------------------------------
# verdict_only and candidate_history
# ---------------------------------------------------------------------------


def test_render_feedback_verdict_only_entry_is_the_single_notice():
    text = render_feedback([{"iteration": 1, **REDACTED_ENTRY}])
    assert text == VERDICT_ONLY_NOTICE
    assert "DO NOT PROPOSE" not in text and render_feedback([]) == ""


def test_candidate_history_defaults_to_empty():
    assert WorkflowState().candidate_history == []


def test_verdict_only_redacts_feedback_and_keeps_the_full_history(golden_model_code,
                                                                  golden_data):
    app, strategist = two_round_graph(
        golden_data, validations=[NO_GAIN_WITH_HINT, FROZEN_VALIDATION_PASS], verdict_only=True)
    out = invoke(app, make_state(golden_model_code, golden_data))

    assert out["status"] == "complete"
    assert out["feedback_history"] == [{"iteration": 1, **REDACTED_ENTRY}]
    assert len(strategist.calls) == 2
    prompt = strategist.calls[1]
    assert VERDICT_ONLY_NOTICE in prompt
    assert "delta-eLPD" not in prompt
    assert "HINT-MARK" not in prompt
    assert "DO NOT PROPOSE:" not in prompt
    assert PROPOSAL_NAMES[0] not in prompt
    # the validator still ran in full and the history keeps both verdicts
    hist = out["candidate_history"]
    assert [h["passed"] for h in hist] == [False, True]
    assert [h["iteration"] for h in hist] == [0, 1]
    assert hist[0]["validation"] is NO_GAIN_WITH_HINT
    assert hist[0]["validation"].misspecification_hints
    assert hist[1]["refined_model_code"] == out["refined_model_code"] == GOLDEN_REFINED_REFERENCE


def test_default_run_discloses_reasons_hints_and_names(golden_model_code, golden_data):
    app, strategist = two_round_graph(
        golden_data, validations=[NO_GAIN_WITH_HINT, FROZEN_VALIDATION_PASS])
    out = invoke(app, make_state(golden_model_code, golden_data))

    assert out["status"] == "complete"
    prompt = strategist.calls[1]
    assert "delta-eLPD" in prompt and "HINT-MARK" in prompt
    assert f"DO NOT PROPOSE: {PROPOSAL_NAMES[0]}" in prompt
    assert "verdict_only" not in out["feedback_history"][0]
    assert [h["passed"] for h in out["candidate_history"]] == [False, True]


def test_candidate_history_has_one_entry_per_validated_round(golden_model_code, golden_data):
    app, _ = two_round_graph(
        golden_data, validations=[FROZEN_VALIDATION_FAIL, FROZEN_VALIDATION_PASS])
    out = invoke(app, make_state(golden_model_code, golden_data))
    hist = out["candidate_history"]
    assert [h["passed"] for h in hist] == [False, True]
    assert hist[0]["proposal_names"] == PROPOSAL_NAMES
    assert hist[1]["proposal_names"] == [n + "_v2" for n in PROPOSAL_NAMES]
    assert all(isinstance(p, dict) for h in hist for p in h["proposals"])
    assert hist[0]["rationale"].startswith(PROPOSAL_NAMES[0] + ": ")
    assert len(out["feedback_history"]) == 1  # failures only


# ---------------------------------------------------------------------------
# custom validation_fn: the skill wrapper pattern
# ---------------------------------------------------------------------------


def test_custom_validation_fn_with_kwargs_runs_one_round(golden_model_code, golden_data):
    """``def fn(refined, original, data, **kw)`` must accept whatever the validator node
    forwards (framework, seed, proposals, ...): this is how the skill wraps
    ``run_full_validation`` with its own budget."""
    seen: dict = {}

    def fn(refined, original, data, **kw):
        seen.update(refined=refined, original=original, data=data, kw=kw)
        return FROZEN_VALIDATION_PASS

    app = build_refinement_graph(
        **fixture_agent_kwargs(validator_kwargs={"validation_fn": fn}), data=golden_data)
    out = invoke(app, make_state(golden_model_code, golden_data, max_iterations=1))

    assert out["status"] == "complete"
    assert out["validation"] is FROZEN_VALIDATION_PASS
    assert seen["refined"] == GOLDEN_REFINED_REFERENCE
    assert seen["original"] == golden_model_code
    assert seen["data"] is golden_data
    assert seen["kw"]["framework"] == "numpyro"
    assert {"fast", "cache_dir", "seed", "proposals"} <= set(seen["kw"])
    assert [p["name"] for p in seen["kw"]["proposals"]] == PROPOSAL_NAMES
    assert [h["passed"] for h in out["candidate_history"]] == [True]


def test_track_usage_counts_the_agents_inside_the_compiled_graph(golden_model_code,
                                                                 golden_data):
    """The tracker's context var must reach the chat-model calls made inside the graph
    (langgraph runs nodes on an executor)."""
    kwargs = fixture_agent_kwargs()
    for name in ("analyst_llm", "strategist_llm", "synthesizer_llm"):
        kwargs[name].usage = (1000, 100)
    kwargs["report_llm"] = FixtureLLM(outputs=["Summary."], usage=(500, 50))
    with track_usage("pipeline") as t:
        out = run_refinement(GOLDEN_PROBLEM, GOLDEN_DATASET_SUMMARY, golden_model_code,
                             framework="numpyro", data=golden_data, max_iterations=1, **kwargs)
    assert out["status"] == "complete"
    s = t.snapshot()
    # analyst + strategist + synthesizer (structured) + the free-text report call
    assert s["calls"] == 4 and s["calls_without_usage"] == 0
    assert (s["input_tokens"], s["output_tokens"]) == (3500, 350)
    assert list(s["by_model"]) == ["fixture"]
