"""Keyless tests for the multi-provider ``get_llm`` factory and the provenance helpers.

The provider client packages are replaced by fake modules in ``sys.modules``, so every
branch is exercised whether or not langchain-openai / langchain-google-genai are
installed, no real client is ever constructed, and nothing touches the network.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

import compass.llm as llm
from compass.llm import PROVIDERS, FixtureLLM, get_llm, model_id_of, provider_of


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class ChatAnthropic(Recorder):
    pass


class ChatOpenAI(Recorder):
    def with_structured_output(self, schema=None, **kwargs):
        self.structured = (schema, kwargs)
        return "chain"


class AzureChatOpenAI(Recorder):
    pass


class ChatGoogleGenerativeAI(Recorder):
    pass


PACKAGES: dict[str, type[Recorder]] = {
    "langchain_anthropic": ChatAnthropic,
    "langchain_openai": ChatOpenAI,
    "langchain_google_genai": ChatGoogleGenerativeAI,
}


def _fake_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@pytest.fixture(autouse=True)
def fake_clients(monkeypatch):
    # keys and BAYES_REFINE_* are already cleared by conftest.isolated_env
    for name, cls in PACKAGES.items():
        monkeypatch.setitem(sys.modules, name, _fake_module(name, **{cls.__name__: cls}))


def _set_all_keys(monkeypatch):
    for spec in PROVIDERS.values():
        monkeypatch.setenv(spec["env_key"], "test-key")


# --- provider table ---------------------------------------------------------


def test_providers_table_shape():
    assert set(PROVIDERS) == {"anthropic", "openai", "google"}
    for spec in PROVIDERS.values():
        assert set(spec) == {"env_key", "default_model", "model_env"} | ({"env_aliases"} & set(spec))
    assert PROVIDERS["anthropic"]["env_key"] == "ANTHROPIC_API_KEY"
    assert PROVIDERS["anthropic"]["model_env"] == "BAYES_REFINE_MODEL"
    assert PROVIDERS["anthropic"]["default_model"] == llm.DEFAULT_MODEL == "claude-sonnet-4-6"
    assert PROVIDERS["openai"]["env_key"] == "OPENAI_API_KEY"
    assert PROVIDERS["openai"]["model_env"] == "BAYES_REFINE_MODEL_OPENAI"
    assert PROVIDERS["openai"]["default_model"] == "gpt-5"
    assert PROVIDERS["google"]["env_key"] == "GOOGLE_API_KEY"
    assert PROVIDERS["google"]["model_env"] == "BAYES_REFINE_MODEL_GOOGLE"
    assert PROVIDERS["google"]["default_model"] == "gemini-3.1-pro-preview"
    assert PROVIDERS["google"]["env_aliases"] == ("GEMINI_API_KEY",)


# --- provider selection ------------------------------------------------------


def test_default_provider_is_anthropic_with_default_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    out = get_llm()
    assert isinstance(out, ChatAnthropic)
    assert out.kwargs == {
        "model": llm.DEFAULT_MODEL,
        "temperature": 0.0,
        "max_tokens": 8192,
        "timeout": 120.0,
        "max_retries": 2,
    }


def test_explicit_model_arg_beats_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("BAYES_REFINE_MODEL", "claude-opus-4-1")
    assert get_llm().kwargs["model"] == "claude-opus-4-1"
    assert get_llm(model="claude-haiku-4-5").kwargs["model"] == "claude-haiku-4-5"


def test_env_provider_honoured(monkeypatch):
    _set_all_keys(monkeypatch)
    monkeypatch.setenv("BAYES_REFINE_PROVIDER", "openai")
    assert isinstance(get_llm(), ChatOpenAI)
    monkeypatch.setenv("BAYES_REFINE_PROVIDER", "google")
    assert isinstance(get_llm(), ChatGoogleGenerativeAI)


def test_provider_arg_beats_env(monkeypatch):
    _set_all_keys(monkeypatch)
    monkeypatch.setenv("BAYES_REFINE_PROVIDER", "openai")
    assert isinstance(get_llm(provider="google"), ChatGoogleGenerativeAI)
    assert isinstance(get_llm(provider="anthropic"), ChatAnthropic)


def test_provider_name_is_case_insensitive_and_stripped(monkeypatch):
    _set_all_keys(monkeypatch)
    assert isinstance(get_llm(provider="OpenAI"), ChatOpenAI)
    assert isinstance(get_llm(provider=" openai "), ChatOpenAI)
    monkeypatch.setenv("BAYES_REFINE_PROVIDER", " Google\n")
    assert isinstance(get_llm(), ChatGoogleGenerativeAI)


def test_model_env_is_per_provider(monkeypatch):
    _set_all_keys(monkeypatch)
    monkeypatch.setenv("BAYES_REFINE_MODEL", "claude-opus-4-1")
    monkeypatch.setenv("BAYES_REFINE_MODEL_OPENAI", "gpt-4.1-mini")
    monkeypatch.setenv("BAYES_REFINE_MODEL_GOOGLE", "gemini-2.5-flash")
    assert get_llm(provider="anthropic").kwargs["model"] == "claude-opus-4-1"
    assert get_llm(provider="openai").kwargs["model"] == "gpt-4.1-mini"
    assert get_llm(provider="google").kwargs["model"] == "gemini-2.5-flash"


# --- per-provider constructor kwargs ----------------------------------------


def test_openai_default_is_a_reasoning_model_without_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out = get_llm(provider="openai")
    assert out.kwargs == {
        "model": PROVIDERS["openai"]["default_model"],
        "max_tokens": 32768,
        "timeout": 120.0,
        "max_retries": 2,
    }


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5.1", "o3", "o4-mini", "GPT-5"])
def test_openai_omits_temperature_for_o_series_and_gpt5(monkeypatch, model):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out = get_llm(provider="openai", model=model, temperature=0.7)
    assert "temperature" not in out.kwargs
    assert out.kwargs["model"] == model


@pytest.mark.parametrize("model", ["gpt-4.1", "gpt-4o", "gpt-50", "omni-1"])
def test_openai_passes_temperature_for_non_reasoning_models(monkeypatch, model):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out = get_llm(provider="openai", model=model, temperature=0.3, max_tokens=4096, timeout=30.0)
    assert out.kwargs == {
        "model": model,
        "temperature": 0.3,
        "max_tokens": 4096,
        "timeout": 30.0,
        "max_retries": 2,
    }


def test_openai_reasoning_models_get_a_larger_output_cap(monkeypatch):
    """Reasoning tokens count against max_tokens for o-series / gpt-5; the cap is raised so a
    long prompt cannot exhaust it before the tool call is emitted."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert llm.OPENAI_REASONING_MAX_TOKENS == 32768
    gpt5 = get_llm(provider="openai", model="gpt-5")
    assert gpt5.kwargs["max_tokens"] == 32768 and "temperature" not in gpt5.kwargs
    plain = get_llm(provider="openai", model="gpt-4.1")
    assert plain.kwargs["max_tokens"] == 8192 and plain.kwargs["temperature"] == 0.0
    big = get_llm(provider="openai", model="gpt-5", max_tokens=65536)
    assert big.kwargs["max_tokens"] == 65536


def test_openai_structured_output_defaults_to_function_calling(monkeypatch):
    """OpenAI's strict JSON-schema mode rejects the project's schemas; the provider switch
    returns a ChatOpenAI subclass whose with_structured_output defaults to tool calling."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out = get_llm(provider="openai")
    assert isinstance(out, ChatOpenAI) and type(out).__name__ == "ChatOpenAIFunctionCalling"
    assert out.with_structured_output(dict) == "chain"
    schema, kwargs = out.structured
    assert schema is dict and kwargs == {"method": "function_calling", "strict": False}
    out.with_structured_output(dict, method="json_schema", strict=True)  # explicit wins
    assert out.structured[1] == {"method": "json_schema", "strict": True}


def test_google_kwargs_use_max_output_tokens(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    out = get_llm(provider="google", max_tokens=1234, timeout=45.0, max_retries=5)
    assert out.kwargs == {
        "model": PROVIDERS["google"]["default_model"],
        "temperature": 0.0,
        "max_output_tokens": 1234,
        "timeout": 45.0,
        "max_retries": 5,
    }


def test_google_accepts_gemini_api_key_alias(monkeypatch):
    """Google AI Studio issues GEMINI_API_KEY; the provider switch accepts it for GOOGLE_API_KEY."""
    monkeypatch.setenv("GEMINI_API_KEY", "alias-key")
    out = get_llm(provider="google")
    assert isinstance(out, ChatGoogleGenerativeAI)
    assert out.model == "gemini-3.1-pro-preview"
    assert os.environ["GOOGLE_API_KEY"] == "alias-key"


def test_anthropic_kwargs_pass_through(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    out = get_llm(temperature=0.5, max_tokens=32000, timeout=300.0, max_retries=5)
    assert out.kwargs == {
        "model": llm.DEFAULT_MODEL,
        "temperature": 0.5,
        "max_tokens": 32000,
        "timeout": 300.0,
        "max_retries": 5,
    }


# --- errors -------------------------------------------------------------------


def test_unknown_provider_lists_valid_names(monkeypatch):
    _set_all_keys(monkeypatch)
    with pytest.raises(RuntimeError, match="Unknown LLM provider 'mistral'") as exc:
        get_llm(provider="mistral")
    for name in PROVIDERS:
        assert name in str(exc.value)
    monkeypatch.setenv("BAYES_REFINE_PROVIDER", "cohere")
    with pytest.raises(RuntimeError, match="cohere"):
        get_llm()


def test_missing_anthropic_key_message_names_the_dotenv_file():
    with pytest.raises(RuntimeError) as exc:
        get_llm()
    msg = str(exc.value)
    assert msg.startswith("ANTHROPIC_API_KEY is not set. Put it in a .env file in the current working directory or one of its parents")
    assert "code/.env" not in msg


@pytest.mark.parametrize(
    "provider, env_key",
    [("openai", "OPENAI_API_KEY"), ("google", "GOOGLE_API_KEY")],
)
def test_missing_key_names_env_var_and_dotenv(monkeypatch, provider, env_key):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # another provider's key does not help
    with pytest.raises(RuntimeError) as exc:
        get_llm(provider=provider)
    msg = str(exc.value)
    assert env_key in msg
    assert provider in msg
    assert "a .env file in the current working directory or one of its parents" in msg
    assert "code/.env" not in msg
    assert "ANTHROPIC_API_KEY" not in msg


def test_dotenv_is_searched_from_the_current_working_directory(monkeypatch):
    """A bare load_dotenv() makes python-dotenv walk the call stack and search from the
    directory of the calling module -- the installed package's own directory -- so a .env
    beside the user's script was never found. get_llm must search from the cwd upward."""
    seen = {}
    monkeypatch.setattr(llm, "find_dotenv",
                        lambda *a, **k: seen.setdefault("find", (a, k)) and "/found/.env")
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: seen.setdefault("load", (a, k)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_llm()
    assert seen["find"] == ((), {"usecwd": True})
    assert seen["load"] == (("/found/.env",), {})


def test_key_check_precedes_import(monkeypatch):
    # Neither the client package nor its constructor may be touched when the key is missing.
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm(provider="openai")


@pytest.mark.parametrize(
    "provider, package",
    [
        ("anthropic", "langchain_anthropic"),
        ("openai", "langchain_openai"),
        ("google", "langchain_google_genai"),
    ],
)
def test_missing_client_package_is_runtime_error(monkeypatch, provider, package):
    _set_all_keys(monkeypatch)
    # None in sys.modules makes `import <package>` raise ModuleNotFoundError, whether or
    # not the distribution is actually installed.
    monkeypatch.setitem(sys.modules, package, None)
    with pytest.raises(RuntimeError) as exc:
        get_llm(provider=provider)
    msg = str(exc.value)
    assert provider in msg
    assert f"pip install {package.replace('_', '-')}" in msg
    assert isinstance(exc.value.__cause__, ImportError)


# --- provenance helpers --------------------------------------------------------


def test_provider_of():
    assert provider_of(FixtureLLM(outputs=["x"])) == "fixture"
    assert provider_of(ChatAnthropic(model="m")) == "anthropic"
    assert provider_of(ChatOpenAI(model="m")) == "openai"
    assert provider_of(AzureChatOpenAI(model="m")) == "openai"
    assert provider_of(ChatGoogleGenerativeAI(model="m")) == "google"

    class MyAnthropic(ChatAnthropic):
        pass

    assert provider_of(MyAnthropic(model="m")) == "anthropic"
    assert provider_of(Recorder(model="m")) == "unknown"
    assert provider_of(object()) == "unknown"
    assert provider_of(None) == "unknown"


def test_provider_of_get_llm_results(monkeypatch):
    _set_all_keys(monkeypatch)
    assert provider_of(get_llm(provider="anthropic")) == "anthropic"
    assert provider_of(get_llm(provider="openai")) == "openai"
    assert provider_of(get_llm(provider="google")) == "google"


def test_model_id_of():
    assert model_id_of(FixtureLLM(outputs=["x"])) == ""
    assert model_id_of(Recorder(model="a", model_name="b", model_id="c")) == "a"
    assert model_id_of(Recorder(model_name="b", model_id="c")) == "b"
    assert model_id_of(Recorder(model_id="c")) == "c"
    assert model_id_of(Recorder()) == ""
    assert model_id_of(Recorder(model="", model_name="b")) == "b"
    assert model_id_of(Recorder(model=None, model_name="b")) == "b"


def test_model_id_of_get_llm_results(monkeypatch):
    _set_all_keys(monkeypatch)
    monkeypatch.setenv("BAYES_REFINE_MODEL_OPENAI", "gpt-4.1")
    assert model_id_of(get_llm(provider="anthropic")) == llm.DEFAULT_MODEL
    assert model_id_of(get_llm(provider="openai")) == "gpt-4.1"
    assert model_id_of(get_llm(provider="google")) == PROVIDERS["google"]["default_model"]


def test_all_exports_match_the_research_module():
    assert set(llm.__all__) == {
        "DEFAULT_MODEL",
        "FixtureLLM",
        "PROVIDERS",
        "STRUCTURED_RETRIES",
        "StructuredOutputError",
        "UsageTracker",
        "current_tracker",
        "empty_usage",
        "get_llm",
        "invoke_structured",
        "merge_usage",
        "model_id_of",
        "provider_of",
        "track_usage",
    }
    for name in llm.__all__:
        assert hasattr(llm, name)


# Not ported: the research suite's P7 tests (``test_driver_loads_the_checkouts_dotenv_at_import``
# and ``test_bench_run_loads_code_dotenv_when_imported_from_another_directory``) pin that every
# research driver (bench/run.py, bench/rescore.py, bench/drift.py, figure2/figure2_run.py,
# figure3/figure3_run.py, scripts/*) loads the checkout's code/.env at import. They read those
# files by their research-tree paths and import ``bench.run``, which the package does not ship;
# the package's ``get_llm`` contract (``find_dotenv(usecwd=True)``) is pinned above by
# ``test_dotenv_is_searched_from_the_current_working_directory``.
