---
name: compass
description: Refine a user's Bayesian model (NumPyro) with the COMPASS system - LLM agents propose literature-grounded refinements, an MCMC verifier accepts or refuses them. Use when a user has data plus a probabilistic model and wants it improved, diagnosed, repaired, or adjudicated; also for verifier-only adjudication of a candidate model against a baseline.
---

# COMPASS skill

You are the intermediary between a human and the COMPASS refinement system. COMPASS's verifier
— not you, not any LLM — decides acceptance. Your job: collect inputs, run the system, and
narrate the gates honestly. The full operating contract is `AGENT_GUIDE.md` at the root of the
COMPASS repo — read it the first time you use this skill in a session.

## Prerequisites

- COMPASS installed in the active Python environment:
  `pip install git+https://github.com/M-Lab-Repo/compass.git` (the repository is public from
  release 0.3.0),
  or `pip install -e <path-to-cloned-compass-repo>` from a clone; `pip install compass-refine`
  once the package is published on PyPI. Verify with `python -c "import compass"`.
- The commands below are written to run **from the COMPASS repo root**; the runner scripts live
  in the `scripts/` directory next to this SKILL.md and can be invoked from anywhere by
  absolute path. (When installing this folder as a Claude skill, name it `compass/`, e.g.
  `~/.claude/skills/compass/`.)
- The API key of the provider the agents run on, in the environment or a `.env` file in the
  current working directory or one of its parents: `ANTHROPIC_API_KEY` by default,
  `OPENAI_API_KEY` for `--provider openai`, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for
  `--provider google`. The OpenAI and Google clients are the `providers` extra:
  `pip install 'compass-refine[providers] @ git+https://github.com/M-Lab-Repo/compass.git'`
  (from a clone: `pip install -e '.[providers]'`). Verifier-only runs need no key.
- COMPASS executes model code written by the language model on this machine (the synthesizer's
  smoke test in a `python -I` subprocess whose environment is a short allowlist of runtime
  variables — `PATH`, `HOME`, the locale, `TMPDIR`, library paths, the JAX/XLA/CUDA/OpenMP
  thread knobs and the Windows start-up variables — with every `*_API_KEY`, `*_TOKEN` and
  `BAYES_REFINE_*` variable dropped and no `PYTHONPATH`; the verifier's fits in-process);
  neither is a sandbox. Prefer a container or VM, keep nothing in the environment beyond the
  provider key, and show the user `refined_model_code` before anything deploys it (README,
  "Security"). Because `-I` ignores `PYTHON*` variables and the user site-packages, NumPyro
  and JAX must be installed in the interpreter's own environment (a venv or conda
  environment, not `pip install --user`); the synthesizer probes the interpreter first and
  raises `EnvironmentError` when they are not importable under `-I` — an installation
  problem, never evidence about a candidate.

## Step 1 — collect from the user (checklist)

1. **Data**: arrays for every model argument, exact keyword names, observed variable included.
   Save as `.npz` (arrays) or `.json` (lists/scalars).
2. **Model**: source of a function named `model` (NumPyro; `numpyro`, `dist`, `jnp` are
   injected). Keyword args = data keys; observed site takes `obs=`.
3. **Problem description**: SYMPTOMS ONLY — never the fix. If the user names a fix they want,
   ask whether it is a commitment (state it as one) or a hunch (leave it out).
4. **The unit of exchangeability** (donor / subject / group / school) — this decides how
   cross-validation folds are cut when the ladder escalates.
5. **Budget & patience**: MCMC verdicts take minutes at minimum; N-fold certifications take
   hours. Agree before launching.

## Step 2 — run

Full pipeline (agents + verifier), via the bundled runner:

```bash
python skill/scripts/run_refinement.py \
    --data data.npz --model model.py \
    --problem "counts are far more dispersed than Poisson allows; occasional huge values" \
    --iterations 2 --out result.json
```

Further runner flags (`--help` lists them all):

- `--provider anthropic|openai|google` and `--model-id ID` select the chat model for every
  agent. Defaults: `BAYES_REFINE_PROVIDER`, else Anthropic; the provider's `BAYES_REFINE_MODEL*`
  variable, else its built-in default. (`--model` remains the path to the model source file.)
- `--smoke-on-data` smoke-tests candidates on the real data arrays instead of dummy data derived
  from the recorded shapes. Recommended when the data has integer index or count columns: the
  dummy data turns them into negative floats, which fails every candidate that uses them
  semantically.
- `--guard-replicate-sites` adds the NumPyro site guard (no unobserved replicate or discrete
  sample sites, which NUTS cannot sample) to the synthesizer prompts and a targeted hint to its
  fix prompt.

The output JSON also carries `candidate_history`: one entry per validated candidate with its
source, proposal names and verdict.

Verifier-only adjudication of an explicit candidate (no LLM, no key):

```bash
python skill/scripts/verify_candidate.py \
    --data data.npz --original original.py --candidate candidate.py \
    --warmup 3000 --samples 2000 --chains 2 --out verdict.json
```

Both scripts print a JSON verdict and write it to `--out`. Long runs: launch in the background
and report progress rather than blocking. A run that dies with a synthesis blocker
(`status="failed"` at iteration 0, no validation) is a *completeness* failure — the idea was
never adjudicated; retry with a higher `--synth-attempts` (default 6; the package default is 3), and never count it as evidence
against the proposal.

## Step 3 — narrate the verdict

Quote `failure_reasons` verbatim. The gate cheat-sheet (full explanations in AGENT_GUIDE.md §3):

| Gate | Test | Refusal means |
|---|---|---|
| **G** convergence | R-hat < 1.01, ESS > 400, E-BFMI > 0.3, 0 divergences; the *baseline* is checked for divergences/R-hat/ESS too | the fit (or the reference fit) cannot be trusted — numbers must not be acted on |
| **S** significance | candidate better AND \|Δ̂\| ≥ 2·SE (SE floor applies) | improvement not distinguishable from noise; reported as no win |
| **P** predictive checks | mean/sd/max/#zeros battery (the mixed-replication PPC for latent models is recorded as a diagnostic, never gates) | simulated data does not resemble the real data |
| **R** instrument reliability | Pareto k̂ ≤ 0.7 (an infinite k̂, arviz's verdict on degenerate importance weights, counts as unreliable; a NaN k̂ does not gate), else one auto-escalation: unit K-fold (K=5) when unit structure is derivable, otherwise reloo; deeper (up to K=N) only via explicit `cv_spec`; fold fits gated on fold R-hat | the *measuring instrument* is untrustworthy — not a statement about the candidate |
| comparison refused | `model comparison refused: …`: an observed site masked (`handlers.mask`, `.mask()`, `obs_mask=`) or scaled (`handlers.scale`) in the candidate but not in the baseline (a mask both programs share is allowed); likelihood parameters that depend on the observed response when the baseline's do not; observed variables, shapes or values that differ from the baseline's (values at float32 tolerance; one renamed observed site with identical shape and values is allowed); pointwise log-likelihood columns degenerate (constant or non-finite) in the candidate alone; or a program the verifier cannot trace on the data | the candidate changed *what is predicted*, so no ELPD is comparable or reported, and neither escalation nor the repair clause runs; the fix is in the candidate's code, never in the budget |
| repair clause | TOST: \|Δ̂\| ≤ 2·SE and \|Δ̂\|+2·SE ≤ 4 nats, reliable comparison, clean candidate fit (PPC incl.), baseline still fails G at doubled budget | certified *repair* (same predictions, sound computation) — never call it an improvement |

Iron rules: refusals are information, relay them fully; never re-run seeds/budgets until
something passes (every adjudication counts — keep and disclose the ledger); an
instrument-reliability refusal localizes pathology, offer escalation up the ladder instead of
retries; a run that ends `status="failed"` with no validated improvement is the honest
null — narrate it as the correct outcome for an adequate model: the original deploys
unchanged. A round in which the strategist's post-validation drops every proposal (a banned
already-failed name, an invalid change type, or no citation while findings exist) is booked as
a failed iteration — `error_log` carries a line starting `strategist: post-validation rejected
every proposal`, and the next round's DO-NOT-PROPOSE bullets still name every banned
technique — and the pipeline retries within `--iterations`; it is a round that produced nothing
to test, not an error.

When everything passes except a residual that is not the candidate's fault (e.g. "original-model
fit unreliable (comparison baseline)"), COMPASS refuses by design. Present the record and the
residual to the user; acceptance at that point is a **recorded human decision**
(AGENT_GUIDE.md §6), not something you or COMPASS grant.
