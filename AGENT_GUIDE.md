# AGENT_GUIDE — operating COMPASS on a user's behalf

This document is written for an **agent** (an LLM assistant) that sits between a human user and
the COMPASS system. The intended shape of the interaction:

```
user  ⇄  you (the agent)  ⇄  COMPASS (pipeline + MCMC verifier)
```

The user brings data, a working Bayesian model, and a conversation about what worries them.
You collect those into COMPASS's inputs, run the system, and — this is the important part —
**narrate the progression honestly**, gate by gate, including refusals. COMPASS is built so that
acceptance comes from a sampler, not from you or from any LLM. Your job is translation and
honesty, never adjudication.

---

## 1. What you must collect from the user

| Input | Type | Contract |
|---|---|---|
| `data` | `dict[str, np.ndarray \| int \| float]` | Arrays plus plain size scalars. Every array the model consumes, by the exact keyword names the model signature uses. The observed variable (e.g. `y`) is included here too. |
| `model_code` | `str` | Source of a function **named `model`** whose keyword arguments match the data keys, with the observed site passed as `obs=`. NumPyro framework: the code body may use `numpyro`, `dist` (numpyro.distributions), and `jnp` — they are injected at execution. JAX discipline applies: no Python branching on array values, sizes from scalar inputs (`n`, `n_groups`), never from array contents. |
| `problem_description` | `str` | **Symptoms only, never the fix.** Describe what is observable (skewness, excess zeros, bounded ranges, group structure, sampler complaints) and what the estimand is. If the user names a favored solution, keep it out of this field — put constraints they insist on (e.g. "the exponential link is a scientific commitment") as explicit commitments instead. |
| `dataset_summary` | `str` (optional) | Shapes/roles of each data key. If empty, COMPASS computes one from `data`. |
| `framework` | `"numpyro"` | The supported framework for full validation. |
| Budget | ints | `num_warmup`, `num_samples`, `num_chains`, `target_accept` — accepted directly by `run_full_validation`. The full pipeline's validator node does not take budget kwargs: to override the budget there, pass `validator_kwargs={"validation_fn": <closure over run_full_validation with your budget>}` (the bundled `skill/scripts/run_refinement.py` shows the pattern). `fast=True` is for smoke tests only, never for verdicts. |
| `seed` | int | Fix it and report it. The verifier's cache is content-addressed on (code, data, budget, seed). |
| `cache_dir` | path (optional) | Fit cache. Reuse it across a session so retries never resample a fit they already own. |
| API key | env | `ANTHROPIC_API_KEY` for the default provider; `OPENAI_API_KEY` or `GOOGLE_API_KEY` (alias `GEMINI_API_KEY`) when the agents run on OpenAI or Google via `get_llm(provider=...)`, which needs the `providers` extra. Read from the environment or from a `.env` file in the current working directory or one of its parents (python-dotenv's `find_dotenv(usecwd=True)`, run when `get_llm` is called; a `.env` beside the installed package is not consulted). Optional, live retrieval only: `S2_API_KEY`, or failing that `ASTA_API_KEY`, is sent as the `x-api-key` header on every Semantic Scholar channel (search, citations, paper finder) for a dedicated rate limit. The verifier alone (`run_full_validation`) needs none. |

Things to ask the user before running: What is the estimand (what should predict well)? What is
the *unit* of exchangeability (donor, school, site…)? Which modeling choices are commitments
versus accidents? How long are they willing to wait (MCMC minutes, not seconds)?

## 2. How to run

One call, full pipeline (analyst → scout → strategist → feasibility gate → synthesizer →
verifier → reporter, iterating up to `max_iterations`):

```python
from compass import run_refinement
final = run_refinement(problem_description=..., model_code=..., data=...,
                       max_iterations=2, validator_kwargs={"cache_dir": ".compass_cache", "seed": 0})
print(final["report"])         # human-readable narrative to relay
final["validation"]            # last ValidationResult
final["refined_model_code"]    # last candidate source
final["feedback_history"]      # the growing record of refusals (see §5)
```

To run every agent on another provider, pass `llm=get_llm(provider="openai")` (or `"google"`)
from `compass`; per-agent overrides such as `synthesizer_llm=` take the same objects.

Verifier-only — adjudicate a specific candidate with no LLM involved:

```python
from compass import run_full_validation
res = run_full_validation(refined_code, original_code, data,
                          seed=0, num_warmup=3000, num_samples=2000, num_chains=2)
```

(`refined_code` and `original_code` are the two model source strings; `examples/verifier_only.py`
is a complete, key-free run of this call on synthetic counts.)

By default the scout searches a bundled 15-paper fixture corpus with live retrieval off; for
production pass `scout_kwargs={"live": True}` (arXiv + Semantic Scholar) and/or a curated
`corpus`. arXiv needs no key; Semantic Scholar works keyless on the shared public rate limit,
and the paper-finder channel (`paper_finder_enabled=True`) stays off unless `S2_API_KEY` or
`ASTA_API_KEY` is set. Live calls are throttled, retried once on a 429 (after the server's
`Retry-After` when it is within the 30 s cap; a longer `Retry-After` skips the retry) and
otherwise return nothing rather than failing the run.

## 3. The gates — what each one means and how to explain it

The verifier fits **both** the candidate and the original with full NUTS, then applies a fixed
predicate. Acceptance requires: converged **and** zero failure reasons. Every failure emits a
specific, quotable string in `ValidationResult.failure_reasons` — quote these to the user
verbatim; they are the system's own words.

### Gate G — convergence of the candidate's fit
The fit must be computationally trustworthy before its numbers mean anything.
- max R-hat < **1.01** (any parameter over ⇒ `"R-hat too high: …"`)
- min ESS > **400** (`"ESS too low: …"`)
- E-BFMI > **0.3** where finite
- **zero** divergent transitions (`"N divergent transitions"`)

*Explain to the user as:* "before comparing models, the sampler itself has to be healthy;
a divergent or unmixed fit produces numbers we must not act on."

**Baseline-reliability check (part of G):** the *original* model's fit is checked for
divergences, R-hat, and ESS (E-BFMI is gated on the candidate side only). If the reference
fit is unreliable, the comparison is not certified — even when the
candidate looks better (`"original-model fit unreliable (comparison baseline): …"`). This can
produce the striking verdict that a published model is *too broken to be beaten*; the honest
resolutions are to repair the baseline's computation or to hand the human a recorded decision
(see §6).

### Observation-set checks — the two fits must score the same data
Before any comparison, the verifier checks that the candidate and the original were scored on
the same observations, and the check is comparative: both programs are traced on the data
(`check_observed_sites`) and the candidate's observed sites are compared with the baseline's.
A mask that both programs share — the data's own missingness, declared the same way on both
sides — is allowed. The candidate is refused when an observed site is **masked in the
candidate alone** (its distribution is a `MaskedDistribution` whose mask is not identically
True, which covers `numpyro.handlers.mask`, `Distribution.mask` (`dist.X(...).mask(m)`) and
the `obs_mask=` keyword of `numpyro.sample`) or **scaled in the candidate alone**
(`handlers.scale`, a `scale` other than 1), when its **likelihood parameters depend on the
observed response** while the baseline's do not (a two-faced likelihood: a scale or rate
computed from the observed values lets points opt out of the comparison without any mask),
or when the program cannot be traced at all. The comparison itself (`compare_models`) is
refused when the **observed variables, shapes or values differ** from the baseline's — the
same log-likelihood variables, except that a candidate may rename its single observed site
(`y` to `obs`) as long as the shape and the values are identical; the same per-point shapes;
observed values identical at float32 tolerance (a cast of the data to float32 is not a
difference) — or when the candidate's **pointwise log-likelihood is degenerate** where the
baseline's is not (a column constant across all draws or non-finite, the signature of a
masked or clamped point). Each refusal is a failure reason that starts with
`"model comparison refused: "` and names the finding and the side it was found on; at
release 0.3.0 they read: `observed site 'y' is masked (numpyro mask on the likelihood): <k>
of <n> observations are excluded from its log-likelihood, so the candidate would be scored on
fewer data than the baseline`; `observed site 'y' carries a log-likelihood scale (<s>;
numpyro.handlers.scale): a reweighted observation set is not comparable with the baseline's`;
`observed data differ between candidate and baseline (variable 'y': observed values differ
at <k> of <n> positions)` (or the differing variable lists or shapes); `<k> of <n> pointwise
log-likelihood values of the refined model are degenerate (constant across all draws or
non-finite): a masked or clamped observed site cannot be scored by PSIS-LOO`; `the refined
program could not be traced on the data (<error>)`. No ELPD number is reported for such a
candidate, no ladder escalation and no repair clause runs, and `converged` is untouched.
These are refusals of the *comparison*, not findings about the data: a model that
legitimately changes the likelihood of the same observations (another distribution family,
a transformed *linear predictor*) passes them, and so does a candidate that keeps the
baseline's own mask.

*Explain to the user as:* "for the comparison to mean anything, both models have to be
predicting the same numbers; a candidate that quietly changes what is being predicted cannot
win, however good its score looks." When this check fires, the fix is in the candidate's code
(restore the observed site as the baseline has it), never in the budget.

### Gate S — significance of the predictive improvement
Cross-validated ELPD comparison (PSIS-LOO by default). Accepted only if the candidate is
better **and** the improvement clears twice its standard error:
`loo_delta < 0` (candidate preferred) **and** `|Δ̂| ≥ 2·SE(Δ̂)`, with a degeneracy floor so
SE = 0 can never make the test vacuously true. Otherwise: `"no significant improvement: …"`.

*Explain as:* "the improvement has to be larger than the noise in our estimate of it; an
insignificant win is reported as no win — occasionally losing real power is the price of
soundness."

### Gate P — posterior-predictive adequacy
A battery of test statistics on replicated data — mean, sd, max, #zeros (max and #zeros
one-sided high). This battery is the only PPC that gates acceptance. For models with per-unit
latent variables there is additionally a **mixed-replication** PPC that re-draws the latents —
it catches candidates whose predictive distribution explodes even when the in-sample fit looks
fine, but it is **recorded, not gating**: its result lands in `ValidationResult.ppc_mixed` and
the report, never in `failure_reasons`. Relay it as a diagnostic.

*Explain as:* "does data simulated from the fitted model look like the data we actually have?"

### Gate R — reliability of the comparison instrument itself
The ELPD comparison is a Monte-Carlo estimate; it can be silently untrustworthy. PSIS-LOO is
trusted only when the refined model's Pareto **k̂ ≤ 0.7** (`"PSIS unreliable: …"`). When k̂
fails, COMPASS does not guess — it **escalates a ladder** of instruments:

1. PSIS-LOO first (cheap, importance-sampled);
2. on a k̂ failure, ONE automatic escalation to an exact-refit instrument, dispatched by
   structure: **unit-level K-fold CV** (folds cut at the unit of exchangeability —
   donor/school/group — K=5, or 3 under `fast`) when per-unit structure is derivable from the
   model + data; otherwise **`reloo`** (exact refits of just the influential points, at most 8);
3. deeper instruments — larger K up to exact leave-one-unit-out (`kfold_unit(K=N)`) — are not
   reached automatically: the caller requests them explicitly via `cv_spec`, a dict
   `{"spec": CVSpec(...), "scheme": "kfold", "K": <number of units>}` whose `spec`
   (`compass.crossval.CVSpec`) names the exchangeable unit — `per_unit_keys`, `local_latents`,
   `obs_site`, `obs_kwarg`, `unit_size_keys`, `shared_keys`, `unit_axis`, `per_unit_axes`,
   `plate` — and whose other keys (`scheme`: `"kfold"`, `"split"` or `"entry_mask"`; `K`, capped
   at the number of units; `split_frac`; `budget`; `s_use`; `r_prior`) go to `crossval_compare`.
   The README's configuration reference has a worked leave-one-school-out example.

Crucially, **each rung must pass its own reliability gate**: every fold fit on both sides is
checked at the fold level (R-hat; fold divergences are recorded but do not gate), and a rung
whose fits are unreliable refuses rather than reports
(`"CV fold fit unreliable (worst R-hat …)"`; the reloo rung's variant reads
`"exact-refit (reloo) fit unreliable (worst R-hat …)"`). A refusal at this gate is a statement
about the *instrument*, not about the candidate's quality — say exactly that to the user, and
treat repeated instrument failures as diagnostic information about the model class (they often
localize the pathology, e.g. weak identifiability of a latent variable).

An **infinite** k̂ — what arviz returns when the importance weights degenerate, for example on
a masked observation — is treated as unreliable, never as a pass: it escalates the ladder
exactly as k̂ > 0.7 does (unit K-fold when the structure is derivable, otherwise `reloo`), and
when neither rung re-adjudicates the comparison the gate stands with the reason
`"PSIS unreliable: refined-model Pareto-k-hat is inf (degenerate importance weights)"` (a finite
excess reads `"PSIS unreliable: refined-model Pareto-k-hat <k> > 0.7"`). Relay it as an
instrument failure, like any other refusal at this gate. (A NaN k̂ means the tail could not be
estimated, e.g. too few draws, and does not gate on its own; the repair clause below is barred
by both, since it needs positive evidence that the comparison was reliable.)

Models with intrinsic per-unit latents routinely have k̂ ≈ 1 for *every* member of the class;
in that regime only the exact-refit rungs can certify anything, and they cost N refits. Set
expectations accordingly.

### The repair-equivalence clause
A candidate that is meant to fix *computation* (a reparameterization) rather than predictions
can be certified as a **repair** if its predictions are provably equivalent: TOST-style,
`|Δ̂| ≤ 2·SE` **and** `|Δ̂| + 2·SE ≤ 4 nats`. Further preconditions all apply: the comparison
itself must be reliable (finite SE > 0, no instrument-reliability complaints, and the
candidate's Pareto k̂ finite and ≤ 0.7 unless the comparison was cross-validated — an infinite
or NaN k̂ bars a repair), the candidate's
own fit must be clean with no other failure reasons (the PPC included), and the original must
demonstrably fail G — including after one automatic retry of the baseline at **doubled budget**
(a baseline that becomes clean when given more compute is not repair-eligible). A repair is not an "improvement" and must not be narrated as
one; it is a certified maintenance action.

### Confidence
Accepted results carry a confidence level; HIGH requires a large clearance (z ≥ 4), ESS
headroom ≥ 1.5×, zero divergences, and a passing PPC. Report the level, not just the verdict.

## 4. Narrating progression to the user

Map the pipeline stages to plain language as they happen:

1. **Analyst** — "reading the model and data, listing suspected weaknesses" (relay the ranked
   weaknesses).
2. **Scout** — "searching the literature for applicable techniques" (relay retrieved papers by
   title; provenance is annotation, never evidence of correctness).
3. **Strategist** — "turning diagnosis + literature into ranked proposals, each with a
   falsifiable prediction" (relay proposals *with* their predictions; also relay the ban list —
   what the record says not to retry). A deterministic post-validation then drops any proposal
   whose name matches a banned technique, whose `change_type` is invalid, or that cites no
   finding while findings exist. If nothing survives, the round yields an **empty bundle**
   rather than an error: `error_log` gets one line starting
   `strategist: post-validation rejected every proposal` (`EMPTY_BUNDLE_LOG_PREFIX`, followed by
   what was dropped and why), the feasibility gate books a failed iteration whose feedback entry
   names no technique and carries that summary, and the graph retries the strategist within
   `max_iterations` or ends at the report. Narrate it as a round that produced nothing to test.
4. **Feasibility gate → Synthesizer** — "writing and smoke-testing the candidate program"
   (synthesis failures are completeness failures, not evidence against the idea). A related
   class of failure is structured output: every agent call asks the model for a schema-typed
   reply and retries a malformed one two more times with a schema reminder appended. After
   that, the analyst and the strategist raise `StructuredOutputError` and the run stops; the
   synthesizer gives up on the current candidate (a synthesis failure in `error_log`; the graph
   skips validation and ends at the report, and the candidate was never adjudicated); the scout
   treats the paper
   whose reply could not be parsed as not applicable (score 0). Report these as plumbing
   failures, never as evidence about the model.
5. **Verifier** — "fitting both models with MCMC and applying the gates" (this is the slow
   part; report budget and expected wall time).
6. **Verdict** — one of: **accepted** (with confidence and Δ̂ ± SE), **repair**, or
   **refused** (quote every failure reason). There is no separate "nothing to refine" status in
   the code: a run where no candidate clears the bar terminates with `status="failed"` and a
   "no validated improvement" report — for an already-adequate model that IS the honest null,
   and your narration should present it as the correct outcome (the original model deploys
   unchanged), not as an error.

Rules that keep the narration honest:
- Quote `failure_reasons` verbatim; do not soften "refused" into "almost passed".
- A refusal is information. Refusal reasons feed `feedback_history` and become bans/hints for
  the next iteration — show the user that loop working.
- Never present an unreliable-instrument refusal as "the candidate is bad".
- Never re-run with new seeds/budgets until something passes; every adjudication counts. If you
  run more than one, disclose the full ledger (see §5).
- Distinguish the three change types in plain words: *improvement* (predicts better),
  *repair* (same predictions, sound computation), *commitment change* (different model on
  purpose — the user's call, outside the verifier's authority).

## 5. The feedback ledger

`feedback_history` accumulates every attempt: what was tried, the verdict strings, optional
diagnostic hints. Seed it across sessions so the system never re-proposes a recorded failure.
The strategist renders it as one DO-NOT-PROPOSE bullet per failed iteration —
`- DO NOT PROPOSE: <name>, <name>, … — failed because <reason; reason>` — naming **every**
technique the entry bans (`tried` and all of `tried_all`, the same list the deterministic ban
enforces, so the proposer is never dropped for a name it was not shown), followed by
`- DIAGNOSTIC HINT (what the data suggest): …` lines when the validator localized the
misspecification; an entry redacted by `verdict_only` renders as a single notice with no
names, numbers or hints. Keep a per-application count of adjudications and report it with
final claims — multiplicity is part of the truth.

## 6. When the verifier refuses the final inch

Sometimes everything passes except one residual that is not the candidate's fault (canonical
example: the *baseline's* own fit is unreliable, so the comparison cannot be certified even
though the candidate is clean and the measured improvement is significant). COMPASS will
refuse — by design. That refusal is where your role ends and the human's begins:

- Present the record: what passed, the one residual, its exact magnitude.
- Offer the options: spend more budget to clear it, or make a **recorded human acceptance**
  with the residual risk disclosed.
- If the user accepts, write the decision down (who decided, on what evidence, what risk was
  disclosed) and treat the accepted model as the new baseline thereafter.

The division of labor to preserve: the verifier measures and refuses to overclaim; the human
makes the final scientific call; the record shows which mile was walked by whom.

## 7. Runtime and cost expectations

- Every validation is ≥ 2 full NUTS fits (candidate + baseline) plus LOO/PPC; minutes on CPU
  for small models, longer for hierarchical ones.
- Ladder escalations multiply: K-fold = K refits per side; leave-one-unit-out = N per side.
  Warn before launching an N-fold certification (hours), and use the fit cache.
- Each pipeline iteration adds LLM calls (analyst/scout/strategist/synthesizer; the
  reporter is deterministic unless you pass `report_llm=`); synthesis may retry a few times by
  design.
- Meter the LLM side with `track_usage`: `with track_usage() as usage: final = run_refinement(...)`,
  then `usage.snapshot()` returns calls, input and output tokens, seconds and errors, split by
  model under `by_model`. Report the snapshot alongside the run; the verifier's MCMC time is
  not in it.
- Providers differ in cost and behaviour. OpenAI reasoning models (the o-series and `gpt-5*`
  ids, except the `gpt-5-chat*` chat models, which keep the temperature and the default cap)
  accept no temperature and get an output cap of at least 32768 tokens because their reasoning
  tokens count against it, so their token totals are not comparable with the Anthropic default;
  structured output on OpenAI goes through function calling. The verifier's cost does not depend on the
  provider, and neither does the acceptance predicate.
- `fast=True` exists for plumbing checks. Its numbers are never verdicts.

## 8. Security

`run_refinement` executes Python written by a language model on the user's machine: the
synthesizer smoke-tests each candidate in a subprocess with a scrubbed environment (the
interpreter runs in isolated mode, `python -I`, with a short allowlist built by
`compass.agents.synthesizer.smoke_test_env` — `PATH`, `HOME`, the locale, `TMPDIR`, library
paths, the JAX/XLA/CUDA/OpenMP thread knobs and the Windows start-up variables; see
`smoke_test_env` for the exact names — from which every `*_API_KEY`, `*_TOKEN` and
`BAYES_REFINE_*` variable is dropped; `PYTHONPATH` is not passed through), and the verifier
then fits it in-process with NUTS (an `exec` of the candidate source in the host process).
Neither is a sandbox — the subprocess still has the host's filesystem and network, and the
in-process fit has every privilege of the Python process. One practical consequence of `-I`:
it ignores `PYTHON*` variables and the user site-packages, so NumPyro and JAX must be
installed in the interpreter's own environment (a venv or conda environment, not
`pip install --user`); the synthesizer probes the interpreter before the first smoke test and
raises `EnvironmentError` when `numpy`, `jax` or `numpyro` is not importable under `-I`,
rather than failing every candidate with an import error — relay that error as an
installation problem, never as evidence about a candidate. Before launching a run, make sure
the user knows this
and prefers a container or VM, especially with live retrieval (`scout_kwargs={"live": True}`)
or a corpus they did not curate, since retrieved text reaches the prompts that write the
code; keep nothing in the environment beyond the provider key (and the optional scout keys);
and show the user `final["refined_model_code"]` before anything deploys it. The verifier alone
(`run_full_validation`) executes only the two programs it is given.
