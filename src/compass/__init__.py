"""COMPASS: statistically verified refinement of probabilistic programs.

LLM agents propose, search, and synthesize model refinements; a no-LLM MCMC
verifier holds the acceptance predicate. See README.md and AGENT_GUIDE.md.
"""

__version__ = "0.3.0"

from compass.agents.validator import run_full_validation
from compass.graph import (
    build_refinement_graph,
    build_report,
    infer_data_shapes,
    run_refinement,
)
from compass.llm import (
    PROVIDERS,
    FixtureLLM,
    StructuredOutputError,
    UsageTracker,
    get_llm,
    track_usage,
)
from compass.state import ModelSpec, PaperFinding, ValidationResult, WorkflowState

__all__ = [
    "__version__",
    "build_refinement_graph",
    "build_report",
    "FixtureLLM",
    "get_llm",
    "infer_data_shapes",
    "ModelSpec",
    "PaperFinding",
    "PROVIDERS",
    "run_full_validation",
    "run_refinement",
    "StructuredOutputError",
    "track_usage",
    "UsageTracker",
    "ValidationResult",
    "WorkflowState",
]
