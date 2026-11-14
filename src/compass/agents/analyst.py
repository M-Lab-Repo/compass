"""Model Analyst agent — parses the user's model into a structured ModelAnalysis.

Spec reference: SPEC_bayesian_refinement_system.md §5.1. Pure LLM, no tools.
The analyst must extract: the framework (numpyro/stan), every parameter and
its prior (including hyperpriors), the likelihood, any hierarchical structure,
parameterization choices (centered vs non-centered), and potential weaknesses
(vague priors, centered parameterization with few groups, ...). The
`potential_issues` it flags directly inform the strategist downstream.

Conventions (system-wide):
- factory: ``create_analyst_agent(llm) -> run``; ``run(state) -> dict`` of
  ONLY the changed fields (LangGraph node update), while also mutating the
  passed state for direct-call convenience.
- state storage is JSON-serializable: ``state.analysis`` holds
  ``ModelAnalysis.model_dump()`` (a plain dict), never the pydantic object.
- on success the analyst sets ``status = "analyzed"``.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from compass.llm import invoke_structured
from compass.schemas import ModelAnalysis
from compass.state import WorkflowState

ANALYST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Bayesian modeling expert. Analyze the provided
probabilistic model code and extract its complete structure.
Focus on:
1. Every parameter and its prior distribution (including hyperpriors)
2. The likelihood/observation model
3. Any hierarchical or multilevel structure
4. Parameterization choices (centered vs non-centered)
5. Potential weaknesses or outdated practices
Be precise about distribution families and their parameters.
Flag anything that modern Bayesian literature would do differently.""",
        ),
        (
            "human",
            """## Problem Context
{problem_description}

## Model Code ({framework})
{model_code}

## Dataset Summary
{dataset_summary}

Analyze this model thoroughly.""",
        ),
    ]
)


def create_analyst_agent(llm):
    """Build the analyst node. ``llm`` is a ChatAnthropic or FixtureLLM."""
    structured_llm = llm.with_structured_output(ModelAnalysis)
    chain = ANALYST_PROMPT | structured_llm

    def run(state: WorkflowState) -> dict:
        if state.original_model is None:
            raise ValueError(
                "Analyst requires state.original_model (a ModelSpec with "
                "framework and source_code); got None. Set it before invoking "
                "the analyst."
            )
        result: ModelAnalysis = invoke_structured(
            chain,
            {
                "problem_description": state.problem_description,
                "framework": state.original_model.framework,
                "model_code": state.original_model.source_code,
                "dataset_summary": state.dataset_summary,
            },
            nudge_field="dataset_summary",
            schema_hint="`likelihood` as an object, not a string",
            what="ModelAnalysis",
        )
        analysis = result.model_dump()
        state.analysis = analysis
        state.status = "analyzed"
        return {"analysis": analysis, "status": "analyzed"}

    return run
