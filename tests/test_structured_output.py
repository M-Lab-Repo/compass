"""``invoke_structured``: the structured-output retry contract shared by the agents.

A reply that does not fit the schema (pydantic ValidationError, an unparseable tool
call, or no tool call at all) is retried with a nudge appended to one prompt field;
once the retries are spent the analyst and strategist raise ``StructuredOutputError``
while the synthesizer gives up on the candidate. FixtureLLM only, no API key.
"""

from __future__ import annotations

import ast

import pytest
from langchain_core.exceptions import OutputParserException

from compass.agents.analyst import create_analyst_agent
from compass.agents.synthesizer import create_synthesizer_agent
from compass.fixtures.frozen_outputs import FROZEN_ANALYSIS, FROZEN_PROPOSALS
from compass.fixtures.golden_model import GOLDEN_ORIGINAL
from compass.llm import STRUCTURED_RETRIES, StructuredOutputError, invoke_structured
from compass.state import ModelSpec, WorkflowState

NUDGE_MARK = "did not fit the required output schema"


class Chain:
    """A stand-in for ``prompt | llm.with_structured_output(Schema)``."""

    def __init__(self, failures: int, *, exc=OutputParserException, none_first: int = 0):
        self.failures, self.exc, self.none_first, self.calls = failures, exc, none_first, []

    def invoke(self, payload):
        self.calls.append(payload["text"])
        if len(self.calls) <= self.none_first:
            return None  # a reply with no tool call
        if len(self.calls) <= self.none_first + self.failures:
            raise self.exc("Failed to parse StrategyOutput from completion")
        return {"ok": True}


def test_parser_exception_is_retried_once_with_a_nudge():
    chain = Chain(failures=1)
    assert invoke_structured(chain, {"text": "go"}, nudge_field="text") == {"ok": True}
    assert len(chain.calls) == 2
    assert chain.calls[0] == "go" and NUDGE_MARK not in chain.calls[0]
    assert chain.calls[1].startswith("go") and NUDGE_MARK in chain.calls[1]


def test_schema_hint_is_carried_in_the_nudge():
    chain = Chain(failures=1)
    invoke_structured(chain, {"text": "go"}, nudge_field="text",
                      schema_hint="`code` as a string")
    assert "`code` as a string" in chain.calls[1]


def test_reply_without_a_tool_call_is_retried():
    chain = Chain(failures=0, none_first=1)
    assert invoke_structured(chain, {"text": "go"}, nudge_field="text") == {"ok": True}
    assert len(chain.calls) == 2 and NUDGE_MARK in chain.calls[1]


def test_persistent_failure_raises_after_the_retries_are_spent():
    chain = Chain(failures=10)
    with pytest.raises(StructuredOutputError, match="OutputParserException") as exc:
        invoke_structured(chain, {"text": "go"}, nudge_field="text", what="StrategyOutput")
    assert len(chain.calls) == STRUCTURED_RETRIES + 1 == 3
    msg = str(exc.value)
    assert "structured output for StrategyOutput invalid after 3 attempts" in msg
    assert issubclass(StructuredOutputError, RuntimeError)


def test_custom_retry_budget_is_honoured():
    chain = Chain(failures=10)
    with pytest.raises(StructuredOutputError, match="after 1 attempts"):
        invoke_structured(chain, {"text": "go"}, nudge_field="text", retries=0)
    assert len(chain.calls) == 1


# --- the pydantic path, through a real agent ------------------------------------


def test_analyst_retries_an_empty_tool_call_then_succeeds(
    make_fixture_llm, golden_problem, golden_model_code, golden_dataset_summary
):
    fake = make_fixture_llm({}, FROZEN_ANALYSIS)  # {} fails ModelAnalysis validation
    run = create_analyst_agent(fake)
    state = WorkflowState(
        problem_description=golden_problem,
        dataset_summary=golden_dataset_summary,
        original_model=ModelSpec(framework="numpyro", source_code=golden_model_code),
    )
    update = run(state)
    assert update["analysis"] == FROZEN_ANALYSIS.model_dump()
    assert len(fake.calls) == 2
    assert NUDGE_MARK not in fake.calls[0]
    assert NUDGE_MARK in fake.calls[1]
    assert "`likelihood` as an object" in fake.calls[1]


def test_analyst_raises_when_every_reply_is_empty(
    make_fixture_llm, golden_problem, golden_model_code, golden_dataset_summary
):
    fake = make_fixture_llm({})  # the last queued output repeats forever
    run = create_analyst_agent(fake)
    state = WorkflowState(
        problem_description=golden_problem,
        dataset_summary=golden_dataset_summary,
        original_model=ModelSpec(framework="numpyro", source_code=golden_model_code),
    )
    with pytest.raises(StructuredOutputError, match="ModelAnalysis.*after 3 attempts"):
        run(state)
    assert len(fake.calls) == STRUCTURED_RETRIES + 1


# --- the synthesizer gives up on the candidate instead of raising ----------------


def ast_only_validator(code: str, data_shapes: dict) -> dict:
    try:
        ast.parse(code)
        return {"compiled": True, "stage": "ast", "stdout": "", "stderr": ""}
    except SyntaxError as exc:
        return {"compiled": False, "stage": "ast", "stdout": "", "stderr": f"SyntaxError: {exc}"}


def test_synthesizer_gives_up_cleanly_on_persistently_empty_replies(make_fixture_llm):
    fake = make_fixture_llm({})
    state = WorkflowState(
        original_model=ModelSpec(
            framework="numpyro", source_code=GOLDEN_ORIGINAL,
            data_shapes={"X": [280], "y": [280], "group_idx": [280], "N_groups": 8},
        ),
        proposals=FROZEN_PROPOSALS.model_dump()["proposals"],
        status="proposed",
    )
    run = create_synthesizer_agent(fake, max_attempts=3, validator=ast_only_validator)
    update = run(state)
    assert update["refined_model_code"] == "" and state.refined_model_code == ""
    assert "status" not in update  # the orchestrator routes on the empty code
    assert len(fake.calls) == STRUCTURED_RETRIES + 1  # no fix-loop calls
    entry = update["error_log"][-1]
    assert entry.startswith(
        "synthesizer: structured output for SynthesizedCode invalid after 3 attempts")
    assert "ValidationError" in entry
    assert state.error_log == update["error_log"]
