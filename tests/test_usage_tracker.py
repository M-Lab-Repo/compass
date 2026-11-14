"""``UsageTracker`` / ``track_usage``: token and wall-clock accounting. No API key:
``FixtureLLM(usage=...)`` plays the provider, plus hand-built LLMResults."""

from __future__ import annotations

import time
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel

import compass.llm as llm_mod
from compass.llm import (
    FixtureLLM,
    UsageTracker,
    current_tracker,
    empty_usage,
    merge_usage,
    track_usage,
)
from compass.schemas import JudgeVerdict


class Out(BaseModel):
    passed: bool
    reasoning: str = ""


def _result(content="x", *, usage=None, model=None, llm_output=None) -> LLMResult:
    msg = AIMessage(content=content)
    if usage is not None:
        i, o = usage
        msg.usage_metadata = {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}
    if model:
        msg.response_metadata = {"model_name": model}
    return LLMResult(generations=[[ChatGeneration(message=msg)]], llm_output=llm_output)


def _call(tracker: UsageTracker, result: LLMResult, *, sleep: float = 0.0, serialized=None):
    run_id = uuid4()
    tracker.on_chat_model_start(serialized or {}, [[]], run_id=run_id)
    if sleep:
        time.sleep(sleep)
    tracker.on_llm_end(result, run_id=run_id)


def test_fixture_default_reports_no_usage_and_is_not_counted():
    f = FixtureLLM(outputs=[Out(passed=True)])
    assert f.usage is None
    msg = f.invoke("hi")
    assert msg.usage_metadata is None and msg.response_metadata == {}
    with track_usage() as t:
        assert f.with_structured_output(Out).invoke("prompt") == Out(passed=True)
    # the structured path bypasses the callback machinery when no usage is configured
    assert t.snapshot()["calls"] == 0 and f.calls == ["hi", "prompt"]


def test_nested_scopes_count_calls_tokens_and_seconds():
    f = FixtureLLM(outputs=[Out(passed=True, reasoning="ok")], usage=(100, 20))
    plain = FixtureLLM(outputs=["text"])
    assert current_tracker() is None
    with track_usage("outer") as outer:
        assert current_tracker() is outer
        f.invoke("a")
        with track_usage("inner") as inner:
            assert current_tracker() is inner and inner.parent is outer
            assert f.with_structured_output(Out).invoke("b") == Out(passed=True, reasoning="ok")
            plain.invoke("c")  # no usage reported
        assert current_tracker() is outer
    assert current_tracker() is None
    f.invoke("after")  # outside every scope: not counted anywhere

    i = inner.snapshot()
    assert (i["calls"], i["input_tokens"], i["output_tokens"], i["total_tokens"]) == (2, 100, 20, 120)
    assert i["calls_without_usage"] == 1 and i["errors"] == 0 and i["seconds"] >= 0
    assert set(i["by_model"]) == {"fixture", "FixtureLLM"}
    assert i["by_model"]["fixture"]["calls"] == 1
    assert i["by_model"]["FixtureLLM"]["calls_without_usage"] == 1
    o = outer.snapshot()
    assert (o["calls"], o["input_tokens"], o["output_tokens"]) == (3, 200, 40)
    assert o["by_model"]["fixture"]["calls"] == 2 and o["calls_without_usage"] == 1
    assert f.calls == ["a", "b", "after"]
    assert set(o) == set(empty_usage())


def test_structured_fixture_calls_inside_a_judge_are_tracked():
    from compass.judge import create_control_judge, create_self_judge

    judge_llm = FixtureLLM(outputs=[JudgeVerdict(passed=True, reasoning="fine")], usage=(300, 50))
    self_judge, control_judge = create_self_judge(judge_llm), create_control_judge(judge_llm)
    with track_usage("judges") as t:
        v = self_judge("refined", "original", rationale="RATIONALE-MARK", problem="p")
        control_judge("refined", "original", rationale="RATIONALE-MARK", problem="p")
    assert v.passed and t.snapshot()["calls"] == 2
    assert t.snapshot()["total_tokens"] == 700 and list(t.snapshot()["by_model"]) == ["fixture"]
    assert "RATIONALE-MARK" in judge_llm.calls[0] and "RATIONALE-MARK" not in judge_llm.calls[1]


def test_batch_calls_from_worker_threads_are_counted():
    f = FixtureLLM(outputs=["x"], usage=(10, 1))
    with track_usage() as t:
        f.batch(["a", "b", "c"])
    assert t.snapshot()["calls"] == 3 and t.snapshot()["input_tokens"] == 30


def test_by_model_split_llm_output_fallback_errors_and_reset():
    t = UsageTracker("direct")
    _call(t, _result(usage=(10, 2), model="claude-x"), sleep=0.01)
    _call(t, _result(usage=(5, 1), model="claude-x"))
    _call(t, _result(llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3},
                                 "model_name": "gpt-x"}))
    _call(t, _result(), serialized={"kwargs": {"model": "gemini-x"}})  # no usage anywhere
    run_id = uuid4()
    t.on_llm_start({"id": ["langchain", "FakeChat"]}, ["p"], run_id=run_id)
    t.on_llm_error(RuntimeError("overloaded"), run_id=run_id)
    s = t.snapshot()
    assert s["calls"] == 5 and s["errors"] == 1 and s["calls_without_usage"] == 1
    assert (s["input_tokens"], s["output_tokens"], s["total_tokens"]) == (22, 6, 28)
    assert s["seconds"] >= 0.01
    assert s["by_model"]["claude-x"]["calls"] == 2 and s["by_model"]["claude-x"]["total_tokens"] == 18
    assert s["by_model"]["gpt-x"]["total_tokens"] == 10
    assert s["by_model"]["gemini-x"]["calls_without_usage"] == 1
    assert s["by_model"]["FakeChat"]["errors"] == 1
    assert sum(m["calls"] for m in s["by_model"].values()) == 5
    t.reset()
    assert t.snapshot() == empty_usage()
    s["calls"] = 99  # the snapshot is a copy
    assert t.snapshot()["calls"] == 0


def test_merge_usage_sums_blocks_and_skips_none():
    a = {"calls": 1, "input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "seconds": 0.5,
         "calls_without_usage": 0, "errors": 0,
         "by_model": {"m1": {"calls": 1, "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                             "seconds": 0.5, "calls_without_usage": 0, "errors": 0}}}
    b = {"calls": 2, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "seconds": 0.25,
         "calls_without_usage": 1, "errors": 0, "by_model": {"m2": {"calls": 2, "seconds": 0.25}}}
    m = merge_usage(a, None, b, {})
    assert (m["calls"], m["input_tokens"], m["total_tokens"], m["seconds"]) == (3, 11, 14, 0.75)
    assert m["calls_without_usage"] == 1 and set(m["by_model"]) == {"m1", "m2"}
    assert m["by_model"]["m2"]["input_tokens"] == 0
    assert merge_usage() == empty_usage() and merge_usage(None) == empty_usage()


def test_configure_hook_is_registered_once():
    from langchain_core.tracers.context import _configure_hooks

    hooks = [h for h in _configure_hooks if h[0] is llm_mod._TRACKER_VAR]
    assert len(hooks) == 1 and hooks[0][1] is True


def test_inner_scope_inside_a_langgraph_node_is_not_double_counted():
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        n: int

    f = FixtureLLM(outputs=["x"], usage=(10, 1))
    inners = []

    def node(state: S) -> S:
        with track_usage("agent") as inner:
            f.invoke("hi")
        inners.append(inner.snapshot())
        return {"n": state["n"] + 1}

    g = StateGraph(S)
    g.add_node("agent", node)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    app = g.compile()
    with track_usage("pipeline") as outer:
        out = app.invoke({"n": 0})
    assert out["n"] == 1
    assert inners[0]["calls"] == 1 and outer.snapshot()["calls"] == 1
    assert outer.snapshot()["total_tokens"] == 11


def test_errors_inside_a_nested_runnable_scope_count_once():
    from langchain_core.runnables import RunnableLambda

    f = FixtureLLM(outputs=[], usage=(1, 1))  # empty queue: every call raises

    def node(_):
        with track_usage("inner") as inner:
            try:
                f.invoke("hi")
            except RuntimeError:
                pass
        return inner.snapshot()

    with track_usage("outer") as outer:
        inner = RunnableLambda(node).invoke("go")
    assert inner["calls"] == 1 and inner["errors"] == 1
    assert outer.snapshot()["calls"] == 1 and outer.snapshot()["errors"] == 1
