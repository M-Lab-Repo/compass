"""LLM provider factory + the fixture-backed test double.

* `get_llm()` — live chat model for the configured provider (anthropic,
  openai, google; see `PROVIDERS`). The API key is read from the environment
  or from a `.env` file in the current working directory or one of its parents
  (python-dotenv's `find_dotenv(usecwd=True)`). Used for real runs and
  `@pytest.mark.llm` semantic tests.
* `FixtureLLM` — a real `BaseChatModel` subclass that returns queued fixture
  objects instead of calling an API. `with_structured_output(Schema)` pops the
  next queued object (validating it against Schema); plain chat invocation
  returns the next object serialized as message content. Rendered prompts are
  recorded in `.calls` so structural tests can assert what an agent actually
  sent. This is what lets the whole test pyramid run with no API key.
* `provider_of(llm)` / `model_id_of(llm)` — provenance labels for run records.
* `UsageTracker` / `track_usage(label)` — token + wall-clock accounting for
  every chat-model call made inside a ``with`` block (see ``track_usage``).

All live agent calls use temperature=0 for reproducibility (spec §13).
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import UUID

from dotenv import find_dotenv, load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.context import register_configure_hook
from pydantic import BaseModel, Field, PrivateAttr

__all__ = ["DEFAULT_MODEL", "PROVIDERS", "STRUCTURED_RETRIES", "FixtureLLM",
           "StructuredOutputError", "UsageTracker", "current_tracker", "empty_usage", "get_llm",
           "invoke_structured", "merge_usage", "model_id_of", "provider_of", "track_usage"]

DEFAULT_MODEL = "claude-sonnet-4-6"

# The openai/google default ids are provisional; change them here only.
PROVIDERS: dict[str, dict[str, str]] = {
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": DEFAULT_MODEL,
        "model_env": "BAYES_REFINE_MODEL",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-5",
        "model_env": "BAYES_REFINE_MODEL_OPENAI",
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        # gemini-2.5-pro is "no longer available to new users" (404, 2026-10-09)
        "default_model": "gemini-3.1-pro-preview",
        "model_env": "BAYES_REFINE_MODEL_GOOGLE",
        # Google AI Studio hands out the key as GEMINI_API_KEY; accept it as an alias
        "env_aliases": ("GEMINI_API_KEY",),
    },
}

_OPENAI_REASONING_MODEL = re.compile(r"^(o\d+|gpt-5)\b", re.IGNORECASE)
OPENAI_REASONING_MAX_TOKENS = 32768  # output cap incl. reasoning tokens for o-series / gpt-5


def _is_openai_reasoning_model(model: str) -> bool:
    """True for the o-series and gpt-5 reasoning models (no temperature; reasoning tokens
    count against the output cap). ``gpt-5-chat*`` ids (e.g. ``gpt-5-chat-latest``) are
    non-reasoning chat models that accept ``temperature``, so, as in langchain-openai's own
    rule, a match is a reasoning model only when ``chat`` is absent from the lower-cased id.
    (Fix: the bare regex also matched ``gpt-5-chat-latest`` and dropped its temperature.)"""
    return bool(_OPENAI_REASONING_MODEL.match(model)) and "chat" not in model.lower()


_CLASS_TO_PROVIDER = {
    "ChatAnthropic": "anthropic",
    "ChatOpenAI": "openai",
    "AzureChatOpenAI": "openai",
    "ChatGoogleGenerativeAI": "google",
}


def get_llm(
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    timeout: float = 120.0,
    max_retries: int = 2,
    provider: str | None = None,
):
    """Return a live chat model for ``provider`` (arg > BAYES_REFINE_PROVIDER > anthropic).

    ``timeout`` (seconds) bounds a stalled stream so a hung request fails fast
    instead of blocking the pipeline forever; ``max_retries`` retries transient
    errors/timeouts. Raises RuntimeError with a clear message if the provider
    is unknown, its API key is missing, or its client package is not installed.
    """
    # Search for a .env file from the CURRENT WORKING DIRECTORY upward (no-op when none is
    # found). A bare load_dotenv() lets python-dotenv walk the call stack and search from the
    # directory of the module that called it, i.e. the installed package's own directory: a
    # .env beside the user's script was never found by a normal install, while an editable
    # install silently picked up whatever .env sat above the source checkout.
    load_dotenv(find_dotenv(usecwd=True))
    provider = (provider or os.environ.get("BAYES_REFINE_PROVIDER") or "anthropic").strip().lower()
    if provider not in PROVIDERS:
        raise RuntimeError(
            f"Unknown LLM provider {provider!r}. Valid providers: {', '.join(PROVIDERS)}. "
            "Set BAYES_REFINE_PROVIDER or pass provider=."
        )
    spec = PROVIDERS[provider]
    env_key = spec["env_key"]
    if not os.environ.get(env_key):
        for alias in spec.get("env_aliases", ()):
            if os.environ.get(alias):
                os.environ[env_key] = os.environ[alias]
                break
    if not os.environ.get(env_key):
        if provider == "anthropic":
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Put it in a .env file in the current working "
                "directory or one of its parents (ANTHROPIC_API_KEY=sk-ant-...) or export it. "
                "Tests that need a live LLM are marked @pytest.mark.llm and skip automatically."
            )
        raise RuntimeError(
            f"{env_key} is not set (required for provider {provider!r}). Put it in a .env "
            f"file in the current working directory or one of its parents ({env_key}=...) "
            "or export it."
        )
    model = model or os.environ.get(spec["model_env"]) or spec["default_model"]

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise _missing_package(provider, "langchain-anthropic", e) from e

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise _missing_package(provider, "langchain-openai", e) from e

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        # OpenAI reasoning models (o-series, gpt-5) reject any non-default temperature, and
        # their reasoning tokens count against the output cap: at 8,192 a long prompt can
        # exhaust the budget before the tool call is emitted ("max_tokens or model output
        # limit was reached", seen on one census record), so the cap is raised for them.
        if _is_openai_reasoning_model(model):
            kwargs["max_tokens"] = max(max_tokens, OPENAI_REASONING_MAX_TOKENS)
        else:
            kwargs["temperature"] = temperature
        return _function_calling_structured(ChatOpenAI)(**kwargs)
    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise _missing_package(provider, "langchain-google-genai", e) from e

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    raise AssertionError(f"no constructor branch for provider {provider!r}")


def _function_calling_structured(cls):
    """Subclass ``cls`` (ChatOpenAI) so ``with_structured_output`` defaults to tool calling.

    langchain-openai's default is OpenAI's strict JSON-schema response format, which rejects
    the project's pydantic schemas (every nested object would need
    ``additionalProperties: false`` and no optional fields: "Invalid schema for
    response_format 'ModelAnalysis'"). Function calling parses the same schemas the way the
    Anthropic path does. Callers that pass ``method`` or ``strict`` explicitly win.
    """
    def with_structured_output(self, schema=None, **kwargs):
        kwargs.setdefault("method", "function_calling")
        kwargs.setdefault("strict", False)
        return super(sub, self).with_structured_output(schema, **kwargs)

    sub = type(f"{cls.__name__}FunctionCalling", (cls,), {
        "with_structured_output": with_structured_output,
        "__doc__": cls.__doc__,
    })
    return sub


def _missing_package(provider: str, dist: str, err: ImportError) -> RuntimeError:
    return RuntimeError(
        f"Provider {provider!r} needs the {dist} package, which is not installed: "
        f"pip install {dist}. ({err})"
    )


def provider_of(llm: Any) -> str:
    """Provider label for a chat model: fixture / anthropic / openai / google / unknown."""
    if isinstance(llm, FixtureLLM):
        return "fixture"
    for cls in type(llm).__mro__:
        if cls.__name__ in _CLASS_TO_PROVIDER:
            return _CLASS_TO_PROVIDER[cls.__name__]
    return "unknown"


def model_id_of(llm: Any) -> str:
    """Model identifier of a chat model (first of .model/.model_name/.model_id), or ""."""
    for attr in ("model", "model_name", "model_id"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)
    return ""


def _prompt_to_text(value: Any) -> str:
    """Best-effort render of whatever the chain passed in (PromptValue, messages, str)."""
    if hasattr(value, "to_string"):
        return value.to_string()
    if isinstance(value, list):  # list of BaseMessages
        return "\n".join(getattr(m, "content", str(m)) for m in value)
    return str(value)


_FIXTURE_OUT = "_fixture_out"


class FixtureLLM(BaseChatModel):
    """Drop-in replacement for ChatAnthropic in tests.

    >>> fake = FixtureLLM(outputs=[my_model_analysis])
    >>> agent = create_analyst_agent(fake)   # agent code is unchanged
    >>> ...
    >>> assert "dist.Normal(0, 100)" in fake.calls[0]  # prompt carried the code

    Queue semantics: outputs are popped in order; the final one repeats
    forever (so a single fixture serves any number of calls, while a sequence
    can script a generate->fail->fix loop).

    ``usage=(input_tokens, output_tokens)`` makes every reply carry
    ``usage_metadata`` (model name "fixture") and routes structured-output
    calls through the chat-model path so ``track_usage`` sees them; by
    default (None) nothing is attached and the structured path bypasses the
    callback machinery, exactly as before.
    """

    outputs: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)
    usage: tuple[int, int] | None = None
    _lock: Any = PrivateAttr(default_factory=threading.Lock)
    # Fixtures are stateful queues and must never be served from the global
    # LLM cache: every FixtureLLM shares one llm_string ("fixture", no
    # identifying params), so with a session cache (conftest SQLiteCache) one
    # test's canned response would be replayed into another's.
    cache: Any = False

    def _next_output(self) -> Any:
        with self._lock:  # .batch / parallel graph nodes pop from several threads
            if not self.outputs:
                raise RuntimeError("FixtureLLM has no queued outputs")
            if len(self.outputs) > 1:
                return self.outputs.pop(0)
            return self.outputs[0]

    # --- structured output path (what the agents actually use) -------------
    def with_structured_output(self, schema: type[BaseModel], **kwargs):
        def _run(value: Any, config: Any = None) -> BaseModel:
            if self.usage is not None:
                # fires the callbacks the tracker listens to; the queued object rides
                # on the reply itself, so concurrent callers never see each other's
                message = self.invoke(value, config=config)
                out = message.additional_kwargs[_FIXTURE_OUT]
            else:
                self.calls.append(_prompt_to_text(value))
                out = self._next_output()
            if isinstance(out, schema):
                return out
            if isinstance(out, BaseModel):
                return schema.model_validate(out.model_dump())
            return schema.model_validate(out)  # plain dict fixture

        return RunnableLambda(_run)

    # --- plain chat path (report agent, free-text calls) --------------------
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(_prompt_to_text(messages))
        out = self._next_output()
        if isinstance(out, BaseModel):
            text = out.model_dump_json()
        else:
            text = out if isinstance(out, str) else str(out)
        message = AIMessage(content=text)
        if self.usage is not None:
            message.additional_kwargs[_FIXTURE_OUT] = out
            i, o = (int(x) for x in self.usage)
            message.usage_metadata = {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}
            message.response_metadata = {"model_name": "fixture"}
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fixture"


# ---------------------------------------------------------------------------
# Token + wall-clock accounting
# ---------------------------------------------------------------------------

_COUNT_KEYS = ("calls", "input_tokens", "output_tokens", "total_tokens", "seconds",
               "calls_without_usage", "errors")


def _zero_counts() -> dict:
    return {**dict.fromkeys(_COUNT_KEYS, 0), "seconds": 0.0}


def empty_usage() -> dict:
    """A zeroed usage block (the shape ``UsageTracker.snapshot`` returns)."""
    return {**_zero_counts(), "by_model": {}}


def _add_counts(into: dict, counts: dict) -> None:
    for k in _COUNT_KEYS:
        into[k] = into.get(k, 0) + counts.get(k, 0)
    into["seconds"] = round(into["seconds"], 3)


def merge_usage(*blocks: dict | None) -> dict:
    """Sum usage blocks (None entries skipped), including their by_model splits."""
    out = empty_usage()
    for b in blocks:
        if not b:
            continue
        _add_counts(out, b)
        for model, counts in (b.get("by_model") or {}).items():
            _add_counts(out["by_model"].setdefault(model, _zero_counts()), counts)
    return out


def _usage_of(response: LLMResult) -> tuple[dict | None, str | None]:
    """(token counts, model name) from the first generation; llm_output as fallback."""
    tokens, model = None, None
    try:
        message = response.generations[0][0].message
    except (IndexError, AttributeError):
        message = None
    if isinstance(message, AIMessage):
        meta = message.usage_metadata or None
        if meta:
            tokens = {"input_tokens": int(meta.get("input_tokens") or 0),
                      "output_tokens": int(meta.get("output_tokens") or 0),
                      "total_tokens": int(meta.get("total_tokens") or 0)}
        rm = message.response_metadata or {}
        model = rm.get("model_name") or rm.get("model")
    out = response.llm_output or {}
    if tokens is None:
        tu = out.get("token_usage") or out.get("usage") or {}
        if tu:
            i = int(tu.get("input_tokens", tu.get("prompt_tokens", 0)) or 0)
            o = int(tu.get("output_tokens", tu.get("completion_tokens", 0)) or 0)
            tokens = {"input_tokens": i, "output_tokens": o,
                      "total_tokens": int(tu.get("total_tokens", i + o) or 0)}
    model = model or out.get("model_name") or out.get("model")
    return tokens, model


class UsageTracker(BaseCallbackHandler):
    """Accumulates calls, tokens and wall time of chat-model calls for one scope.

    Every recorded call is also forwarded to ``parent`` (the enclosing
    ``track_usage`` scope), so nested scopes each see their own total — except
    to ancestors that already receive the run directly: when the inner scope
    is opened inside a runnable (a langgraph node, a RunnableLambda, a chain)
    that started under the outer scope, the outer tracker is inherited onto
    the run and fires its own on_llm_end, so forwarding would double-count.
    ``_start`` detects this (the ancestor's start hook has already seen the
    run_id: langchain-core copies inherited handlers into a run's manager
    before appending the configure-hook ones) and skips those ancestors.
    Safe to call from several threads (langgraph fans nodes out to an executor).
    """

    raise_error = False

    def __init__(self, label: str = "usage", parent: UsageTracker | None = None):
        super().__init__()
        self.label, self.parent = label, parent
        self._lock = threading.Lock()
        # run_id -> (t0, requested model, model class name, ancestors to skip)
        self._started: dict[UUID, tuple[float, str | None, str | None, frozenset]] = {}
        self._totals = empty_usage()

    def reset(self) -> None:
        with self._lock:
            self._totals = empty_usage()

    def snapshot(self) -> dict:
        with self._lock:
            return merge_usage(self._totals)

    @staticmethod
    def _serialized_model(serialized: dict | None,
                          kwargs: dict) -> tuple[str | None, str | None]:
        """(requested model id, model class name) as seen by the start hook."""
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or params.get("ls_model_name")
        cls = None
        if serialized:
            model = model or (serialized.get("kwargs") or {}).get("model")
            ident = serialized.get("id") or []
            cls = str(ident[-1]) if ident else None
        return (str(model) if model else None), cls

    def _directly_attached(self, run_id: UUID) -> bool:
        with self._lock:
            return run_id in self._started

    def _start(self, serialized, run_id: UUID, **kwargs) -> None:
        # ancestors whose own start hook already saw this run are inherited on
        # it and will record it themselves: never forward this run to them
        skip, anc = [], self.parent
        while anc is not None:
            if anc._directly_attached(run_id):
                skip.append(anc)
            anc = anc.parent
        model, cls = self._serialized_model(serialized, kwargs)
        with self._lock:
            self._started[run_id] = (time.perf_counter(), model, cls, frozenset(map(id, skip)))

    def on_llm_start(self, serialized, prompts, *, run_id: UUID, **kwargs) -> None:
        self._start(serialized, run_id, **kwargs)

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, **kwargs) -> None:
        self._start(serialized, run_id, **kwargs)

    def _finish(self, run_id: UUID) -> tuple[float, str | None, str | None, frozenset]:
        with self._lock:
            t0, model, cls, skip = self._started.pop(run_id, (None, None, None, frozenset()))
        return (time.perf_counter() - t0) if t0 is not None else 0.0, model, cls, skip

    def record(self, *, model: str | None, seconds: float, tokens: dict | None,
               error: bool = False, _skip: frozenset = frozenset()) -> None:
        """Book one call here and in every enclosing scope (minus ``_skip``)."""
        if id(self) not in _skip:
            counts = {"calls": 1, "seconds": seconds, "errors": int(error),
                      "calls_without_usage": int(tokens is None and not error), **(tokens or {})}
            with self._lock:
                _add_counts(self._totals, counts)
                _add_counts(self._totals["by_model"].setdefault(model or "unknown",
                                                                _zero_counts()), counts)
        if self.parent is not None:
            self.parent.record(model=model, seconds=seconds, tokens=tokens, error=error,
                               _skip=_skip)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs) -> None:
        seconds, requested, cls, skip = self._finish(run_id)
        tokens, response_model = _usage_of(response)
        # by_model is keyed by the REQUESTED id so a success (whose response may
        # echo a dated id) and an error (no response) land under one key; the
        # response model only serves models the start hook could not name
        self.record(model=requested or response_model or cls, seconds=seconds, tokens=tokens,
                    _skip=skip)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        seconds, requested, cls, skip = self._finish(run_id)
        self.record(model=requested or cls, seconds=seconds, tokens=None, error=True, _skip=skip)


# Route: langchain-core's configure hook. ``CallbackManager.configure`` (run by
# every BaseChatModel.generate / RunnableSequence.invoke) appends the handler
# held in a registered ContextVar, inheritably, to the run's callback manager —
# the same mechanism as ``langchain_core.callbacks.get_usage_metadata_callback``.
# Nothing is attached to the models, so calls on any chat model built anywhere
# (get_llm, a judge's independent provider, a FixtureLLM with usage=...) are
# counted while a track_usage block is active in the calling context, and
# langchain/langgraph copy the context into their worker threads.
_TRACKER_VAR: ContextVar[UsageTracker | None] = ContextVar("compass_usage_tracker", default=None)
register_configure_hook(_TRACKER_VAR, inheritable=True)


def current_tracker() -> UsageTracker | None:
    """The innermost active ``track_usage`` scope, if any."""
    return _TRACKER_VAR.get()


@contextmanager
def track_usage(label: str = "usage") -> Iterator[UsageTracker]:
    """Count every chat-model call made inside the block; read
    ``tracker.snapshot()`` afterwards. Nested blocks also count toward their
    enclosing ones, exactly once: a block opened inside a runnable (graph node,
    chain) that started under an outer block does not forward the calls the
    outer tracker already receives as an inherited handler of that run."""
    tracker = UsageTracker(label, parent=_TRACKER_VAR.get())
    token = _TRACKER_VAR.set(tracker)
    try:
        yield tracker
    finally:
        _TRACKER_VAR.reset(token)


# ---------------------------------------------------------------------------
# Structured-output retries (shared by the analyst, strategist and synthesizer)
# ---------------------------------------------------------------------------

class StructuredOutputError(RuntimeError):
    """The model's structured reply did not fit the schema after every retry."""


STRUCTURED_RETRIES = 2
_NUDGE = ("\n\n(Your previous reply did not fit the required output schema: return every "
          "required field with the right type{hint}, as a single tool call. Lists must be JSON "
          "arrays, not strings.)")


def invoke_structured(chain, inputs: dict, *, nudge_field: str, retries: int = STRUCTURED_RETRIES,
                      schema_hint: str = "", what: str = ""):
    """Invoke a ``with_structured_output`` chain, retrying when the model's reply does not
    fit the schema (an empty or partial tool call raises ``pydantic.ValidationError``; a
    list serialized as a string does too; a reply with no tool call comes back ``None``).
    At temperature 0 a bare retry tends to repeat the failure, so each retry appends a
    short nudge to ``inputs[nudge_field]`` (``schema_hint`` names the fields to insist on).
    Raises :class:`StructuredOutputError` once the retries are spent.
    """
    from langchain_core.exceptions import OutputParserException
    from pydantic import ValidationError

    hint = f", {schema_hint}" if schema_hint else ""
    last = "no tool call in the reply"
    for attempt in range(retries + 1):
        payload = dict(inputs)
        if attempt:
            payload[nudge_field] = f"{inputs[nudge_field]}{_NUDGE.format(hint=hint)}"
        try:
            result = chain.invoke(payload)
        except (ValidationError, OutputParserException) as e:
            # pydantic: a tool call with missing or mistyped fields (Anthropic, OpenAI);
            # OutputParserException: tool-call JSON that does not parse at all (Gemini)
            last = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            continue
        if result is not None:
            return result
    label = f" for {what}" if what else ""
    raise StructuredOutputError(
        f"structured output{label} invalid after {retries + 1} attempts ({last})")
