"""15-paper hand-curated mock corpus for literature-scout testing (spec §12).

Three strata, five papers each:

- "relevant"  — the canonical five methodology papers the scout MUST find for
  the golden problem (a flawed hierarchical linear regression).
- "near_miss" — topically Bayesian but NOT applicable to a simple hierarchical
  linear regression (spatial ICAR/BYM2, scalable GPs, DP mixtures, SMC for
  state-space models, Bayesian optimization). These exercise the
  applicability filter.
- "unrelated" — deep-learning papers (BERT, ResNet, GANs, deep RL, word2vec)
  that must NEVER surface for this problem.

Each paper dict carries: arxiv_id, title, authors, year, abstract, keywords,
section_method, category, and optionally code_example.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Relevant papers (the canonical five — ids/titles/years are contractual)
# --------------------------------------------------------------------------

_NON_CENTERED_CODE = '''\
import numpyro
import numpyro.distributions as dist


def model(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.HalfNormal(5.0))
    # Non-centered parameterization: sample standard-normal innovations,
    # then shift-and-scale deterministically: alpha = mu + tau * z.
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([J]))
    alpha = numpyro.deterministic("alpha", mu + tau * z)
    numpyro.sample("y", dist.Normal(alpha, sigma), obs=y)
'''

_HORSESHOE_CODE = '''\
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(X, y=None, scale_global=0.1, slab_scale=2.0, slab_df=4.0):
    D = X.shape[1]
    # Global-local shrinkage with a finite slab (regularized horseshoe).
    tau = numpyro.sample("tau", dist.HalfCauchy(scale_global))
    lam = numpyro.sample("lam", dist.HalfCauchy(1.0).expand([D]))
    c2 = numpyro.sample(
        "c2", dist.InverseGamma(slab_df / 2.0, slab_df * slab_scale**2 / 2.0)
    )
    lam_tilde = jnp.sqrt(c2) * lam / jnp.sqrt(c2 + tau**2 * lam**2)
    beta = numpyro.sample("beta", dist.Normal(0.0, tau * lam_tilde))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("y", dist.Normal(X @ beta, sigma), obs=y)
'''

_RELEVANT = [
    {
        "arxiv_id": "0708.3797",
        "title": "A General Framework for the Parametrization of Hierarchical Models",
        "authors": ["Omiros Papaspiliopoulos", "Gareth O. Roberts", "Martin Sköld"],
        "year": 2007,
        "abstract": (
            "Hierarchical models are central to modern Bayesian statistics, yet the "
            "efficiency of Markov chain Monte Carlo algorithms applied to them depends "
            "critically on how the latent structure is parameterized. We develop a "
            "general framework contrasting the centered parameterization, in which "
            "group-level effects alpha_j are sampled directly from N(mu, sigma^2), with "
            "the non-centered parameterization, in which one samples auxiliary variables "
            "z_j ~ N(0, 1) and reconstructs alpha_j = mu + sigma * z_j as a deterministic "
            "transform. We show that the relative performance of the two schemes is "
            "governed by how much information the data carry about the latent effects: "
            "centered parameterizations excel when groups are data-rich, while "
            "non-centered parameterizations decouple latent variables from their "
            "hyperparameters and dramatically improve mixing when groups are sparse, "
            "eliminating the funnel-shaped posterior geometry that traps samplers as the "
            "group standard deviation sigma approaches zero. We provide theory, "
            "diagnostics, and practical guidance, including partially non-centered "
            "interpolations between the two extremes."
        ),
        "keywords": [
            "non-centered parameterization",
            "hierarchical models",
            "mcmc",
            "funnel geometry",
            "reparameterization",
            "data augmentation",
        ],
        "section_method": (
            "Consider a two-level hierarchy with hyperparameters (mu, sigma) and group "
            "effects alpha_j | mu, sigma ~ N(mu, sigma^2). The centered parameterization "
            "treats alpha_j as the sampled quantity, which induces strong prior "
            "dependence between alpha_j and sigma: when the data per group are weak, the "
            "joint posterior takes the notorious funnel shape and gradient-based "
            "samplers such as HMC/NUTS produce divergent transitions near sigma = 0. The "
            "non-centered parameterization instead samples standardized innovations "
            "z_j ~ N(0, 1) and defines alpha_j = mu + sigma * z_j deterministically, so "
            "the sampled variables are a priori independent of the hyperparameters and "
            "the funnel disappears from the sampling space. In a hierarchical linear "
            "regression with varying intercepts (or slopes), the transform is applied "
            "per group-level coefficient; in NumPyro this is a one-line change wrapping "
            "a standard-normal sample in a deterministic shift-and-scale, or using "
            "TransformedDistribution / LocScaleReparam. We characterize when each scheme "
            "mixes faster via the Bayesian fraction of missing information and recommend "
            "non-centering whenever per-group sample sizes are small relative to the "
            "between-group variance, the typical regime in which practitioners observe "
            "divergences and low effective sample sizes."
        ),
        "category": "relevant",
        "code_example": _NON_CENTERED_CODE,
    },
    {
        "arxiv_id": "1708.07487",
        "title": "The Prior Can Often Only Be Understood in the Context of the Likelihood",
        "authors": ["Andrew Gelman", "Daniel Simpson", "Michael Betancourt"],
        "year": 2017,
        "abstract": (
            "A key sticking point of Bayesian analysis is the choice of prior "
            "distribution, with a common recommendation being flat or super-vague "
            "priors such as Normal(0, 100^2) intended to be noninformative. We argue "
            "that this recommendation is misguided: the information a prior provides "
            "can only be judged relative to the likelihood, because the likelihood "
            "determines the scale on which the prior acts. A prior that is weakly "
            "informative for one observational design may be hopelessly vague or "
            "unintentionally dogmatic for another. We propose thinking of priors as "
            "generative components of the model that should be checked by simulating "
            "data from the prior predictive distribution, and we advocate weakly "
            "informative defaults that softly constrain parameters to plausible orders "
            "of magnitude after the predictors and outcome have been standardized. We "
            "illustrate with hierarchical regression examples in which vague priors on "
            "coefficients and group-level scale parameters degrade both computation and "
            "inference, while priors scaled to the data, such as Normal(0, 1) on "
            "standardized coefficients and half-Normal on scales, stabilize both."
        ),
        "keywords": [
            "weakly informative priors",
            "prior choice",
            "prior predictive",
            "hierarchical regression",
            "bayesian workflow",
            "scaling",
        ],
        "section_method": (
            "We formalize prior informativeness through the prior predictive "
            "distribution p(y) = integral of p(y | theta) p(theta) dtheta: a prior is "
            "weakly informative when draws from p(y) cover a generous but physically "
            "plausible range of datasets. The procedure is concrete. First standardize "
            "inputs and outcome to unit scale. Second, place priors whose scale matches "
            "the resulting coefficients: Normal(0, 1) or Normal(0, 2.5) on regression "
            "coefficients, half-Normal or half-Student-t on standard-deviation "
            "parameters, never Normal(0, 100^2), whose prior predictive mass extends to "
            "absurd outcome magnitudes and flattens the posterior geometry that "
            "adaptive samplers rely on. Third, simulate from the prior predictive and "
            "inspect whether simulated outcomes are plausible; iterate on prior scales "
            "until they are. In hierarchical models the group-level scale tau deserves "
            "particular care: vague priors on tau put substantial mass on both "
            "near-zero and enormous between-group variation, producing funnel "
            "pathologies and inflated uncertainty. A half-Normal(0, 1) prior on tau "
            "(after standardization) typically regularizes the hierarchy without "
            "overwhelming the data. The same logic generalizes: containment priors "
            "should be tuned to the scale induced by the likelihood, not chosen by "
            "habit or a misplaced appeal to objectivity."
        ),
        "category": "relevant",
        "code_example": (
            "import numpyro\n"
            "import numpyro.distributions as dist\n\n"
            "# Weakly informative priors on standardized data: scales match the\n"
            "# likelihood instead of vague Normal(0, 100) defaults.\n"
            'beta = numpyro.sample("beta", dist.Normal(0.0, 2.5))\n'
            'tau = numpyro.sample("tau", dist.HalfNormal(1.0))\n'
        ),
    },
    {
        "arxiv_id": "1709.01449",
        "title": "Visualization in Bayesian Workflow",
        "authors": [
            "Jonah Gabry",
            "Daniel Simpson",
            "Aki Vehtari",
            "Michael Betancourt",
            "Andrew Gelman",
        ],
        "year": 2019,
        "abstract": (
            "Bayesian data analysis is about more than just computing a posterior: it "
            "is an iterative workflow of model building, inference, checking, and "
            "revision. We argue that visualization is as integral to this workflow as "
            "computation, and we organize the relevant graphics around the stages of "
            "analysis. Before seeing data, prior predictive checks display simulated "
            "datasets drawn from the prior predictive distribution, revealing whether "
            "chosen priors generate outcomes on a plausible scale. During inference, "
            "trace plots, rank histograms, and divergence scatterplots diagnose MCMC "
            "pathologies such as funnels in hierarchical models. After inference, "
            "posterior predictive checks compare replicated datasets to the observed "
            "data through test statistics and density overlays, exposing systematic "
            "model misfit. Finally, pointwise leave-one-out diagnostics, in particular "
            "Pareto-k values from PSIS-LOO, flag influential observations and "
            "untrustworthy approximations. We demonstrate the full workflow on a "
            "hierarchical regression of pollution measurements, where visualization "
            "uncovers a prior that was orders of magnitude too vague and a missing "
            "level in the hierarchy, leading to a substantively better model."
        ),
        "keywords": [
            "bayesian workflow",
            "prior predictive checks",
            "posterior predictive checks",
            "visualization",
            "mcmc diagnostics",
            "model checking",
        ],
        "section_method": (
            "The workflow proceeds in four visual stages. (1) Prior predictive "
            "checking: draw theta from the prior, then y_sim from p(y | theta), and "
            "plot the distribution of simulated outcomes against domain knowledge; "
            "vague priors such as Normal(0, 100) reveal themselves immediately by "
            "generating impossible data. (2) Computation diagnostics: for hierarchical "
            "models we overlay divergent transitions on bivariate scatterplots of a "
            "group-level scale against a group effect, where a funnel of divergences "
            "concentrated at small scale values indicates the need for a non-centered "
            "parameterization. (3) Posterior predictive checking: generate replicated "
            "datasets y_rep from the fitted model and compare summary statistics "
            "(means, standard deviations, extreme quantiles, per-group means) to the "
            "observed values; systematic discrepancies localize misfit to specific "
            "model components. (4) Model comparison and influence: pointwise expected "
            "log predictive densities from PSIS-LOO are plotted per observation, and "
            "Pareto-k diagnostics above 0.7 mark observations whose importance "
            "weights, and hence whose contribution to cross-validation, cannot be "
            "trusted. Each graphic maps to a concrete revision: rescale priors, "
            "reparameterize, enrich the likelihood, or add hierarchy. The cycle "
            "repeats until predictive checks pass."
        ),
        "category": "relevant",
    },
    {
        "arxiv_id": "1510.07151",
        "title": (
            "Practical Bayesian Model Evaluation Using Leave-One-Out "
            "Cross-Validation and WAIC"
        ),
        "authors": ["Aki Vehtari", "Andrew Gelman", "Jonah Gabry"],
        "year": 2017,
        "abstract": (
            "Leave-one-out cross-validation (LOO-CV) and the widely applicable "
            "information criterion (WAIC) are methods for estimating pointwise "
            "out-of-sample prediction accuracy from a fitted Bayesian model using only "
            "the log-likelihood evaluated at posterior simulations of the parameters. "
            "LOO and WAIC have a distinct advantage over simpler estimates of "
            "predictive error such as AIC and DIC: they use the full posterior "
            "distribution and remain valid under weak priors and hierarchical "
            "structures. Exact LOO is costly because it requires refitting the model n "
            "times, so we introduce an efficient approximation based on Pareto-smoothed "
            "importance sampling (PSIS), which stabilizes the importance weights by "
            "fitting a generalized Pareto distribution to their upper tail. The "
            "estimated shape parameter k of that distribution doubles as a diagnostic: "
            "observations with k above 0.7 indicate unreliable approximation. We show "
            "how to compare models on expected log predictive density (elpd), report "
            "the difference together with its standard error, and demonstrate the "
            "approach on hierarchical regression examples using posterior draws from "
            "standard MCMC software."
        ),
        "keywords": [
            "psis-loo",
            "waic",
            "cross-validation",
            "model comparison",
            "elpd",
            "pareto smoothed importance sampling",
        ],
        "section_method": (
            "Given posterior draws theta^(s), s = 1..S, the pointwise predictive "
            "density for observation i is approximated from the matrix of "
            "log-likelihood values log p(y_i | theta^(s)). WAIC sums the log of the "
            "posterior mean of the pointwise likelihoods and subtracts an effective "
            "number of parameters computed from the pointwise posterior variance of "
            "the log-likelihood. For LOO, raw importance ratios 1 / p(y_i | theta^(s)) "
            "have heavy right tails; PSIS fits a generalized Pareto distribution to "
            "the largest 20 percent of ratios and replaces them with expected order "
            "statistics, yielding stable weights and the diagnostic shape parameter "
            "k_i. Values k_i < 0.5 are good, 0.5-0.7 acceptable, and above 0.7 the "
            "estimate for that observation is unreliable and direct refitting is "
            "advised. Two models are compared by elpd_diff, the difference in summed "
            "pointwise elpd, with a standard error computed from the pointwise "
            "differences; a difference smaller than roughly twice its standard error "
            "should not be considered decisive. The method requires only a pointwise "
            "log-likelihood, so it applies directly to hierarchical linear regressions "
            "fit by NumPyro or Stan, and is implemented in ArviZ (az.loo, az.waic, "
            "az.compare) and the loo R package."
        ),
        "category": "relevant",
    },
    {
        "arxiv_id": "1707.01694",
        "title": (
            "Sparsity Information and Regularization in the Horseshoe and Other "
            "Shrinkage Priors"
        ),
        "authors": ["Juho Piironen", "Aki Vehtari"],
        "year": 2017,
        "abstract": (
            "The horseshoe prior has proven to be a noteworthy default for sparse "
            "Bayesian regression: a global scale tau shrinks all coefficients toward "
            "zero while heavy-tailed local scales lambda_j let truly relevant signals "
            "escape. Two practical problems remain. First, the prior on tau is often "
            "chosen carelessly; we show how to set it from a guess of the effective "
            "number of nonzero coefficients, linking tau directly to prior sparsity "
            "information. Second, the original horseshoe leaves large coefficients "
            "completely unregularized, which causes wild posteriors and divergent "
            "Hamiltonian Monte Carlo transitions when parameters are weakly "
            "identified, as in logistic regression with separable data. We introduce "
            "the regularized horseshoe, which adds a finite slab of width c so that "
            "even the largest coefficients are shrunk as if under a Gaussian "
            "slab prior, beta_j ~ N(0, tau^2 * lambda_tilde_j^2) with "
            "lambda_tilde_j^2 = c^2 lambda_j^2 / (c^2 + tau^2 lambda_j^2). The "
            "resulting prior retains horseshoe-style selective shrinkage, samples far "
            "more reliably, and improves predictive performance on real and simulated "
            "high-dimensional regression problems."
        ),
        "keywords": [
            "regularized horseshoe",
            "shrinkage priors",
            "sparsity",
            "global-local shrinkage",
            "hierarchical shrinkage",
            "slab width",
        ],
        "section_method": (
            "The regularized horseshoe places on each regression coefficient "
            "beta_j ~ N(0, tau^2 * lambda_tilde_j^2), where lambda_j ~ HalfCauchy(1) "
            "are local shrinkage scales, tau ~ HalfCauchy(tau_0) is the global scale, "
            "and the regularized local scale is lambda_tilde_j^2 = "
            "c^2 lambda_j^2 / (c^2 + tau^2 lambda_j^2). For small coefficients "
            "tau^2 lambda_j^2 << c^2 and the prior behaves like the original "
            "horseshoe, shrinking noise aggressively toward zero; for large "
            "coefficients the local scale saturates at c, so the slab acts like a "
            "Gaussian prior N(0, c^2) and prevents the unbounded posteriors that "
            "destabilize sampling. The slab width is given an inverse-Gamma prior, "
            "c^2 ~ Inv-Gamma(nu/2, nu s^2/2), corresponding to a Student-t slab with "
            "nu degrees of freedom and scale s. The global scale is calibrated by "
            "tau_0 = p_0 / (D - p_0) * sigma / sqrt(n), where p_0 is a prior guess of "
            "the number of relevant coefficients among D candidates. Implementation "
            "in NumPyro or Stan is direct, and a non-centered parameterization of "
            "beta is recommended. The construction nests within hierarchical "
            "regression: sparse priors on covariate effects coexist with standard "
            "hierarchical priors on group-level intercepts."
        ),
        "category": "relevant",
        "code_example": _HORSESHOE_CODE,
    },
]

# --------------------------------------------------------------------------
# Near-miss papers (Bayesian, but not applicable to a simple hierarchical
# linear regression)
# --------------------------------------------------------------------------

_NEAR_MISS = [
    {
        "arxiv_id": "1601.01180",
        "title": (
            "An Intuitive Bayesian Spatial Model for Disease Mapping that "
            "Accounts for Scaling"
        ),
        "authors": [
            "Andrea Riebler",
            "Sigrunn H. Sørbye",
            "Daniel Simpson",
            "Håvard Rue",
        ],
        "year": 2016,
        "abstract": (
            "In disease mapping, the Besag-York-Mollié (BYM) model combines an "
            "intrinsic conditional autoregressive (ICAR) component capturing spatially "
            "structured variation across administrative regions with an unstructured "
            "exchangeable component. The standard formulation is difficult to use in "
            "practice: the two random effects are not identifiable from a single "
            "realization, the ICAR precision depends on the graph structure so that "
            "priors do not transfer across maps, and hyperprior choice is opaque. We "
            "propose the BYM2 model, a reparameterization in which a single scale "
            "parameter controls the total marginal variance of the combined effect and "
            "a mixing parameter phi in (0, 1) allocates variance between the "
            "spatially structured and unstructured parts. Crucially, the ICAR component "
            "is scaled so that its generalized variance equals one regardless of the "
            "adjacency graph, making penalized complexity priors portable across "
            "applications. We illustrate on insulin-dependent diabetes incidence in "
            "Sardinia, demonstrating interpretable hyperparameters, stable inference "
            "in INLA and MCMC, and a principled shrinkage of the spatial effect "
            "toward the simpler unstructured model."
        ),
        "keywords": [
            "spatial statistics",
            "icar",
            "bym2",
            "disease mapping",
            "penalized complexity priors",
            "areal data",
        ],
        "section_method": (
            "Let G be the adjacency graph of n regions. The ICAR prior on the spatial "
            "effect u specifies u_i | u_{-i} ~ N(mean of neighbors, s^2 / d_i), where "
            "d_i is the number of neighbors, equivalently a Gaussian Markov random "
            "field with precision proportional to the graph Laplacian. Because the "
            "Laplacian's spectrum depends on the map, we rescale the ICAR component so "
            "that the geometric mean of its marginal variances equals one. The BYM2 "
            "linear predictor for region i is then "
            "eta_i = mu + sigma * (sqrt(phi) * u_i + sqrt(1 - phi) * v_i), with v_i "
            "iid standard normal, sigma the overall random-effect scale, and phi the "
            "proportion of variance that is spatially structured. Penalized complexity "
            "priors are placed on sigma and phi, shrinking toward no random effect and "
            "no spatial structure respectively. Inference uses INLA or MCMC with a "
            "sum-to-zero constraint on u to handle the ICAR rank deficiency. The "
            "construction is specific to areal data with a known neighborhood graph: "
            "it requires an adjacency structure and addresses spatial confounding, "
            "concerns that have no analogue in a non-spatial hierarchical regression "
            "with exchangeable groups."
        ),
        "category": "near_miss",
    },
    {
        "arxiv_id": "1903.08114",
        "title": "Exact Gaussian Processes on a Million Data Points",
        "authors": [
            "Ke Alexander Wang",
            "Geoff Pleiss",
            "Jacob R. Gardner",
            "Stephen Tyree",
            "Kilian Q. Weinberger",
            "Andrew Gordon Wilson",
        ],
        "year": 2019,
        "abstract": (
            "Gaussian processes (GPs) are flexible nonparametric Bayesian models whose "
            "exact inference cost, cubic in the number of observations for the "
            "Cholesky factorization of the kernel matrix, has historically restricted "
            "them to datasets of a few thousand points. We show that exact GP "
            "regression can be scaled to over a million training examples by replacing "
            "Cholesky decomposition with preconditioned conjugate gradients and "
            "stochastic Lanczos quadrature for the log-determinant, implemented as "
            "batched matrix-matrix multiplications distributed across multiple GPUs. "
            "Our blackbox matrix-matrix (BBMM) approach evaluates kernel matrices in "
            "tiles, never materializing the full Gram matrix, and obtains predictive "
            "means and variances with accuracy matching exact solves. On large "
            "regression benchmarks, exact GPs trained this way consistently "
            "outperform widely used scalable approximations, including stochastic "
            "variational GPs with inducing points, suggesting that approximation error "
            "rather than model capacity has limited GP performance. We release an "
            "implementation in GPyTorch and analyze the trade-offs between "
            "preconditioning quality, iteration counts, and wall-clock time."
        ),
        "keywords": [
            "gaussian processes",
            "kernel methods",
            "conjugate gradients",
            "gpu computing",
            "scalable inference",
            "nonparametric regression",
        ],
        "section_method": (
            "Exact GP inference requires solving (K + sigma^2 I) x = y and computing "
            "log det(K + sigma^2 I) for an n x n kernel matrix K. We solve the linear "
            "systems with preconditioned conjugate gradients, where each iteration "
            "needs only matrix-vector products K v; these are computed in tiles so "
            "that K is never stored, reducing memory from O(n^2) to O(n). A rank-k "
            "pivoted-Cholesky preconditioner accelerates convergence. The "
            "log-determinant and its gradient for hyperparameter learning are "
            "estimated with stochastic Lanczos quadrature using the same Krylov "
            "iterations. Partitioning kernel tiles across GPUs gives near-linear "
            "multi-device scaling. Training maximizes the exact GP marginal "
            "likelihood over kernel hyperparameters (lengthscales, signal variance, "
            "noise); prediction caches a single solve against the training targets. "
            "The machinery, kernel matrices, Krylov solvers, inducing-point baselines, "
            "is specific to nonparametric function estimation with covariance kernels. "
            "A hierarchical linear regression with a handful of coefficients and "
            "group-level effects involves no kernel matrix and gains nothing from "
            "these solvers."
        ),
        "category": "near_miss",
    },
    {
        "arxiv_id": "2102.09889",
        "title": "Memory-Efficient Variational Inference for Dirichlet Process Mixture Models",
        "authors": ["Lin Huang", "Sayan Ray", "Marta Kowalczyk"],
        "year": 2021,
        "abstract": (
            "Dirichlet process (DP) mixture models provide a principled Bayesian "
            "nonparametric approach to clustering when the number of components is "
            "unknown, but posterior inference at scale remains challenging: collapsed "
            "Gibbs samplers mix slowly on large datasets, and standard stick-breaking "
            "variational approximations require storing responsibilities for every "
            "observation-component pair. We propose a streaming variational algorithm "
            "for DP mixtures that maintains compressed sufficient statistics per "
            "component instead of per-datum responsibilities, reducing memory from "
            "O(NK) to O(K) while preserving the ability to create and prune components "
            "on the fly. Our approach combines a truncation-free stick-breaking "
            "representation with split-merge moves evaluated on minibatch evidence "
            "lower bounds, so the effective number of clusters adapts as data arrive. "
            "On image-patch, single-cell RNA, and document corpora with up to fifty "
            "million observations, the method recovers cluster structure comparable to "
            "full-batch variational Bayes at a fraction of the memory, and yields "
            "calibrated posterior uncertainty over the number of occupied components, "
            "outperforming finite mixtures selected by information criteria."
        ),
        "keywords": [
            "dirichlet process",
            "mixture models",
            "clustering",
            "variational inference",
            "stick-breaking",
            "streaming data",
        ],
        "section_method": (
            "The model places a Dirichlet process prior with concentration alpha on "
            "mixing measures, represented by stick-breaking weights "
            "pi_k = v_k * prod_{j<k} (1 - v_j) with v_k ~ Beta(1, alpha), and "
            "component parameters drawn from a conjugate base measure. Each "
            "observation x_n is assigned a latent component z_n ~ Categorical(pi). "
            "Our variational family factorizes over sticks, component parameters, and "
            "assignments, but assignments are never stored: for each minibatch we "
            "compute responsibilities, fold them into per-component conjugate "
            "sufficient statistics with a decay schedule, and discard them. Component "
            "birth proposals seed a new cluster from poorly explained minibatch "
            "points; merge proposals are accepted when they raise a cached ELBO "
            "estimate. The output is a posterior over partitions and over the number "
            "of occupied clusters. The machinery targets unsupervised density "
            "estimation and cluster discovery with a random number of components; it "
            "does not apply to a regression with a fixed design matrix, known group "
            "structure, and a fixed-dimensional coefficient vector, where no "
            "assignment uncertainty exists."
        ),
        "category": "near_miss",
    },
    {
        "arxiv_id": "1903.04797",
        "title": "Elements of Sequential Monte Carlo",
        "authors": ["Christian A. Naesseth", "Fredrik Lindsten", "Thomas B. Schön"],
        "year": 2019,
        "abstract": (
            "Sequential Monte Carlo (SMC) methods, also known as particle filters, are "
            "a family of simulation-based algorithms for online inference in "
            "state-space models and, more generally, for sampling from sequences of "
            "probability distributions. This tutorial develops SMC from first "
            "principles: sequential importance sampling, the degeneracy problem, and "
            "the resampling step that rejuvenates the particle population. We cover "
            "the bootstrap particle filter for nonlinear non-Gaussian state-space "
            "models, auxiliary and fully adapted variants that exploit the observation "
            "density when proposing particles, and particle smoothing recursions for "
            "retrospective state estimation. Building on these foundations we present "
            "particle Markov chain Monte Carlo, in which an unbiased SMC estimate of "
            "the marginal likelihood is embedded inside a Metropolis-Hastings chain "
            "for static parameters, and SMC samplers that bridge from prior to "
            "posterior through tempered intermediate distributions. Throughout we "
            "emphasize the unbiasedness of the normalizing-constant estimator, "
            "effective sample size monitoring, and practical guidance on the number "
            "of particles, with worked examples from target tracking, epidemiology, "
            "and stochastic volatility modeling."
        ),
        "keywords": [
            "sequential monte carlo",
            "particle filter",
            "state-space models",
            "resampling",
            "particle mcmc",
            "online inference",
        ],
        "section_method": (
            "A state-space model evolves a latent state x_t through a transition "
            "density f(x_t | x_{t-1}) and emits observations y_t ~ g(y_t | x_t). The "
            "bootstrap particle filter maintains N weighted particles approximating "
            "the filtering distribution p(x_t | y_{1:t}): at each time step particles "
            "are propagated through f, reweighted by the likelihood g of the new "
            "observation, and resampled (multinomial, systematic, or stratified) when "
            "the effective sample size N_eff = 1 / sum(w_i^2) drops below a threshold "
            "such as N/2. The product of average unnormalized weights gives an "
            "unbiased estimate of the marginal likelihood p(y_{1:T}), which "
            "particle-marginal Metropolis-Hastings uses to target the exact posterior "
            "over static parameters. SMC samplers generalize the recursion to "
            "artificial sequences pi_0, ..., pi_T interpolating prior and posterior "
            "via tempering, with MCMC mutation kernels between resampling steps. The "
            "entire framework presupposes sequential structure: a time index, "
            "evolving latent states, and observations arriving in order. A static "
            "hierarchical linear regression has no state dynamics to filter, and its "
            "posterior is better explored by gradient-based MCMC than by particle "
            "methods."
        ),
        "category": "near_miss",
    },
    {
        "arxiv_id": "1807.02811",
        "title": "A Tutorial on Bayesian Optimization",
        "authors": ["Peter I. Frazier"],
        "year": 2018,
        "abstract": (
            "Bayesian optimization is a strategy for global optimization of expensive "
            "black-box functions: objectives that take minutes or hours to evaluate, "
            "lack gradients, and may be observed with noise, as when tuning "
            "hyperparameters of machine learning pipelines, designing experiments, or "
            "calibrating simulators. The method maintains a probabilistic surrogate of "
            "the objective, most commonly a Gaussian process regression model over "
            "the design space, and selects each new evaluation point by maximizing an "
            "acquisition function that trades off exploration of uncertain regions "
            "against exploitation of promising ones. This tutorial derives the "
            "canonical acquisition functions, expected improvement, knowledge "
            "gradient, entropy search, and upper confidence bound, and discusses "
            "their relative merits, including behavior under observation noise and "
            "parallel evaluation budgets. We treat practical matters such as kernel "
            "and mean-function choice, input warping for non-stationary objectives, "
            "constraint handling, and initialization with space-filling designs. "
            "Extensions cover multi-fidelity optimization, where cheap low-accuracy "
            "evaluations inform expensive ones, and high-dimensional settings via "
            "embeddings. We close with software pointers and open research "
            "directions in the field."
        ),
        "keywords": [
            "bayesian optimization",
            "gaussian process surrogate",
            "acquisition functions",
            "expected improvement",
            "hyperparameter tuning",
            "black-box optimization",
        ],
        "section_method": (
            "Given an expensive objective f over a compact design space, place a GP "
            "prior on f with mean mu_0 and kernel K. After n evaluations, the "
            "posterior at any candidate x is Gaussian with closed-form mean mu_n(x) "
            "and variance sigma_n^2(x). Expected improvement selects "
            "x_{n+1} = argmax E[max(f* - f(x), 0)], which has an analytic expression "
            "in terms of the posterior mean, standard deviation, and the incumbent "
            "best value f*; the knowledge gradient instead values the improvement in "
            "the maximizer of the posterior mean, and entropy search values the "
            "reduction in entropy of the location of the optimum. The acquisition "
            "function is itself optimized with multi-start gradient ascent, which is "
            "cheap relative to evaluating f. The loop alternates evaluate, update "
            "surrogate, maximize acquisition until the budget is exhausted. The goal "
            "throughout is to find an optimizer of an unknown function in as few "
            "evaluations as possible. Nothing in this machinery concerns posterior "
            "inference for the parameters of a fixed statistical model; it is the "
            "wrong tool for diagnosing or refining a hierarchical regression, though "
            "it could tune an outer-loop quantity such as an MCMC step size."
        ),
        "category": "near_miss",
    },
]

# --------------------------------------------------------------------------
# Unrelated papers (must never surface for a hierarchical regression problem)
# --------------------------------------------------------------------------

_UNRELATED = [
    {
        "arxiv_id": "1810.04805",
        "title": (
            "BERT: Pre-training of Deep Bidirectional Transformers for "
            "Language Understanding"
        ),
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "year": 2018,
        "abstract": (
            "We introduce a new language representation model called BERT, which "
            "stands for Bidirectional Encoder Representations from Transformers. "
            "Unlike recent language representation models, BERT is designed to "
            "pre-train deep bidirectional representations from unlabeled text by "
            "jointly conditioning on both left and right context in all layers. As a "
            "result, the pre-trained BERT model can be fine-tuned with just one "
            "additional output layer to create state-of-the-art models for a wide "
            "range of tasks, such as question answering and language inference, "
            "without substantial task-specific architecture modifications. BERT is "
            "conceptually simple and empirically powerful. It obtains new "
            "state-of-the-art results on eleven natural language processing tasks, "
            "including pushing the GLUE score to 80.5 percent (7.7 points absolute "
            "improvement), MultiNLI accuracy to 86.7 percent, and SQuAD v1.1 question "
            "answering Test F1 to 93.2, a 1.5 point absolute improvement over the "
            "previous best systems, while remaining straightforward to adapt across "
            "domains and task formats with minimal additional engineering effort."
        ),
        "keywords": [
            "nlp",
            "language models",
            "transformers",
            "pretraining",
            "fine-tuning",
            "question answering",
        ],
        "section_method": (
            "BERT's architecture is a multi-layer bidirectional Transformer encoder. "
            "Input text is tokenized into WordPiece subwords; each token embedding is "
            "summed with positional and segment embeddings. Pre-training optimizes two "
            "objectives on large unlabeled corpora (BooksCorpus and English "
            "Wikipedia). First, masked language modeling: 15 percent of input tokens "
            "are selected, of which 80 percent are replaced by a [MASK] symbol, 10 "
            "percent by a random token, and 10 percent left unchanged, and the model "
            "predicts the original tokens from the bidirectional context. Second, "
            "next-sentence prediction: the model classifies whether two concatenated "
            "segments are consecutive in the source text, encouraging discourse-level "
            "representations. The base configuration uses 12 layers, hidden size 768, "
            "and 12 attention heads (110M parameters); the large configuration "
            "doubles depth and width. Fine-tuning attaches a lightweight output head, "
            "a classifier on the [CLS] token or per-token span predictors, and trains "
            "end-to-end for a few epochs with a small learning rate. The same "
            "pre-trained checkpoint transfers across sentence-level and token-level "
            "tasks with minimal architectural change, and ablations show both "
            "pre-training objectives contribute to downstream accuracy."
        ),
        "category": "unrelated",
    },
    {
        "arxiv_id": "1512.03385",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
        "year": 2015,
        "abstract": (
            "Deeper neural networks are more difficult to train. We present a "
            "residual learning framework to ease the training of networks that are "
            "substantially deeper than those used previously. We explicitly "
            "reformulate the layers as learning residual functions with reference to "
            "the layer inputs, instead of learning unreferenced functions. We provide "
            "comprehensive empirical evidence showing that these residual networks "
            "are easier to optimize, and can gain accuracy from considerably "
            "increased depth. On the ImageNet dataset we evaluate residual nets with "
            "a depth of up to 152 layers, eight times deeper than VGG nets but still "
            "having lower complexity. An ensemble of these residual nets achieves "
            "3.57 percent top-5 error on the ImageNet test set and won first place in "
            "the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 "
            "with 100 and 1000 layers. The depth of representations is of central "
            "importance for many visual recognition tasks, and solely due to our "
            "extremely deep representations we obtain a 28 percent relative "
            "improvement on the COCO object detection dataset."
        ),
        "keywords": [
            "computer vision",
            "deep learning",
            "image classification",
            "residual networks",
            "convolutional neural networks",
            "imagenet",
        ],
        "section_method": (
            "Plain convolutional networks degrade in training accuracy as depth "
            "grows, an optimization problem rather than overfitting. We address it "
            "with residual blocks: instead of asking stacked layers to fit a desired "
            "mapping H(x), the block fits the residual F(x) = H(x) - x and outputs "
            "F(x) + x via an identity shortcut connection that adds the block input "
            "to its output. Shortcuts add neither parameters nor computation; when "
            "dimensions change, a 1x1 convolution projects the identity path. The "
            "ImageNet architectures stack 3x3 convolutions with batch normalization "
            "and ReLU, downsampling by stride-2 convolutions; deeper variants "
            "(ResNet-50/101/152) use bottleneck blocks of 1x1, 3x3, 1x1 convolutions "
            "to keep computation tractable. Training uses SGD with momentum, weight "
            "decay, standard data augmentation (random crops and horizontal flips), "
            "and learning-rate steps when validation error plateaus. Experiments show "
            "34-layer residual nets outperform their plain counterparts and 18-layer "
            "nets, with the gap widening at greater depth; on CIFAR-10 a 110-layer "
            "network trains stably while a 1202-layer variant still optimizes, "
            "demonstrating that identity shortcuts resolve the degradation problem."
        ),
        "category": "unrelated",
    },
    {
        "arxiv_id": "1406.2661",
        "title": "Generative Adversarial Networks",
        "authors": [
            "Ian J. Goodfellow",
            "Jean Pouget-Abadie",
            "Mehdi Mirza",
            "Bing Xu",
            "David Warde-Farley",
            "Sherjil Ozair",
            "Aaron Courville",
            "Yoshua Bengio",
        ],
        "year": 2014,
        "abstract": (
            "We propose a new framework for estimating generative models via an "
            "adversarial process, in which we simultaneously train two models: a "
            "generative model G that captures the data distribution, and a "
            "discriminative model D that estimates the probability that a sample came "
            "from the training data rather than G. The training procedure for G is to "
            "maximize the probability of D making a mistake. This framework "
            "corresponds to a minimax two-player game. In the space of arbitrary "
            "functions G and D, a unique solution exists, with G recovering the "
            "training data distribution and D equal to one half everywhere. In the "
            "case where G and D are defined by multilayer perceptrons, the entire "
            "system can be trained with backpropagation. There is no need for any "
            "Markov chains or unrolled approximate inference networks during either "
            "training or generation of samples. Experiments demonstrate the potential "
            "of the framework through qualitative and quantitative evaluation of the "
            "generated samples on image datasets including MNIST, the Toronto Face "
            "Database, and CIFAR-10."
        ),
        "keywords": [
            "generative adversarial networks",
            "deep learning",
            "image generation",
            "adversarial training",
            "minimax games",
            "neural networks",
        ],
        "section_method": (
            "The generator G maps noise vectors z, drawn from a simple distribution "
            "such as uniform or Gaussian, through a multilayer perceptron to produce "
            "samples G(z) in data space. The discriminator D, another multilayer "
            "perceptron, outputs the probability that its input is real training data "
            "rather than a generated sample. Training alternates stochastic gradient "
            "steps: k updates of D ascending its classification objective "
            "log D(x) + log(1 - D(G(z))), then one update of G descending "
            "log(1 - D(G(z))); in practice G instead maximizes log D(G(z)) for "
            "stronger early gradients. We prove that for a fixed G the optimal "
            "discriminator is D*(x) = p_data(x) / (p_data(x) + p_g(x)), and that the "
            "resulting value function is minimized exactly when p_g = p_data, where "
            "it equals the Jensen-Shannon divergence up to constants. Sampling "
            "requires only a forward pass through G, with no Markov chain burn-in. "
            "Experiments fit Gaussian Parzen windows to generated samples to estimate "
            "test-set likelihoods and visually inspect samples, showing the framework "
            "produces sharp images where mean-field alternatives blur."
        ),
        "category": "unrelated",
    },
    {
        "arxiv_id": "1707.06347",
        "title": "Proximal Policy Optimization Algorithms",
        "authors": [
            "John Schulman",
            "Filip Wolski",
            "Prafulla Dhariwal",
            "Alec Radford",
            "Oleg Klimov",
        ],
        "year": 2017,
        "abstract": (
            "We propose a new family of policy gradient methods for reinforcement "
            "learning, which alternate between sampling data through interaction with "
            "the environment and optimizing a surrogate objective function using "
            "stochastic gradient ascent. Whereas standard policy gradient methods "
            "perform one gradient update per data sample, we propose a novel "
            "objective function that enables multiple epochs of minibatch updates. "
            "The new methods, which we call proximal policy optimization (PPO), have "
            "some of the benefits of trust region policy optimization (TRPO), but "
            "they are much simpler to implement, more general, and have better sample "
            "complexity empirically. Our experiments test PPO on a collection of "
            "benchmark tasks, including simulated robotic locomotion and Atari game "
            "playing, and we show that PPO outperforms other online policy gradient "
            "methods, and overall strikes a favorable balance between sample "
            "complexity, simplicity, and wall-time. The clipped surrogate objective "
            "in particular requires only a few lines of code change relative to "
            "vanilla policy gradient implementations while delivering markedly more "
            "stable learning across the full benchmark suite."
        ),
        "keywords": [
            "reinforcement learning",
            "policy gradient",
            "deep learning",
            "control",
            "atari",
            "robotic locomotion",
        ],
        "section_method": (
            "Let r_t(theta) denote the probability ratio between the current policy "
            "and the policy that collected the data, evaluated at state-action pair "
            "(s_t, a_t), and let A_t be an estimate of the advantage function. TRPO "
            "maximizes the surrogate E[r_t A_t] subject to a KL-divergence trust "
            "region, requiring a constrained second-order solver. PPO replaces the "
            "constraint with a clipped objective: "
            "E[min(r_t A_t, clip(r_t, 1 - eps, 1 + eps) A_t)] with eps around 0.2, "
            "which removes the incentive to move r_t outside the clipping interval "
            "and permits several epochs of minibatch SGD on each batch of "
            "trajectories. Advantages are computed with generalized advantage "
            "estimation; a learned value function (often sharing layers with the "
            "policy network) provides the baseline, trained with a squared-error "
            "loss, and an entropy bonus encourages exploration. The full objective "
            "sums the clipped policy term, the value loss, and the entropy term. On "
            "continuous-control MuJoCo tasks and 49 Atari games, PPO with Adam and a "
            "fixed set of hyperparameters matches or beats TRPO, A2C, and ACER while "
            "being implementable in a few dozen lines on top of a standard policy "
            "gradient codebase."
        ),
        "category": "unrelated",
    },
    {
        "arxiv_id": "1301.3781",
        "title": "Efficient Estimation of Word Representations in Vector Space",
        "authors": ["Tomas Mikolov", "Kai Chen", "Greg Corrado", "Jeffrey Dean"],
        "year": 2013,
        "abstract": (
            "We propose two novel model architectures for computing continuous vector "
            "representations of words from very large data sets. The quality of these "
            "representations is measured in a word similarity task, and the results "
            "are compared to the previously best performing techniques based on "
            "different types of neural networks. We observe large improvements in "
            "accuracy at much lower computational cost: it takes less than a day to "
            "learn high quality word vectors from a 1.6 billion word dataset. "
            "Furthermore, we show that these vectors provide state-of-the-art "
            "performance on our test set for measuring syntactic and semantic word "
            "similarities. The learned representations exhibit striking linear "
            "regularities: simple vector arithmetic such as vector('King') minus "
            "vector('Man') plus vector('Woman') yields a vector closest to "
            "vector('Queen'). We introduce a comprehensive analogy test set "
            "containing five types of semantic questions and nine types of syntactic "
            "questions, and demonstrate that representation quality depends on both "
            "the architecture and the amount of training data, with our simpler "
            "log-linear models dominating earlier deep architectures at equal compute."
        ),
        "keywords": [
            "word embeddings",
            "nlp",
            "representation learning",
            "neural networks",
            "word analogies",
            "log-linear models",
        ],
        "section_method": (
            "We introduce two log-linear architectures that strip away the expensive "
            "hidden layers of earlier neural language models. The continuous "
            "bag-of-words (CBOW) model predicts the current word from the average of "
            "the embedding vectors of surrounding context words within a symmetric "
            "window, sharing the projection matrix across positions. The continuous "
            "skip-gram model inverts the task: it uses the current word's embedding "
            "to predict each context word within a window whose effective size is "
            "sampled per token, so nearer words are weighted more heavily. Both "
            "models are trained with hierarchical softmax over a Huffman-coded "
            "vocabulary tree, reducing the per-example output cost from vocabulary "
            "size to its logarithm; training scales across machines with "
            "asynchronous gradient descent in the DistBelief framework. Because "
            "there is no nonlinear hidden layer, billions of tokens can be processed "
            "in hours, and embedding dimensionality and corpus size can be traded "
            "off systematically. Evaluation uses cosine similarity on an analogy "
            "benchmark, where skip-gram excels at semantic relations and CBOW at "
            "syntactic ones; the resulting vectors serve as features for downstream "
            "language tasks."
        ),
        "category": "unrelated",
    },
]

MOCK_PAPERS: list[dict] = _RELEVANT + _NEAR_MISS + _UNRELATED

RELEVANT_IDS: list[str] = [p["arxiv_id"] for p in MOCK_PAPERS if p["category"] == "relevant"]
NEAR_MISS_IDS: list[str] = [p["arxiv_id"] for p in MOCK_PAPERS if p["category"] == "near_miss"]
UNRELATED_IDS: list[str] = [p["arxiv_id"] for p in MOCK_PAPERS if p["category"] == "unrelated"]
