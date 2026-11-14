"""Benchmark package (paper §9.B/§9.C).

A small registry of curated model-data *misspecification types*, each with a
true data-generating process, a misspecified original model, a gold reference
refinement, and a plausible-but-broken decoy. The runner drives the full
six-agent pipeline over these instances (recovery + refinement-wins, §9.B) and
the LLM-judge-vs-MCMC arbiter comparison over the decoys (false acceptance,
§9.C). The MCMC validator remains the sole arbiter throughout — the benchmark
only *measures*; it never changes acceptance.
"""
