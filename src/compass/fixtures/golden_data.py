"""Synthetic dataset generator for the golden test case.

Spec reference: SPEC_bayesian_refinement_system.md §10 — verbatim.
8 schools, 280 students total, one continuous predictor (study hours).
"""

import numpy as np


def generate_golden_dataset(seed=42):
    rng = np.random.default_rng(seed)
    group_sizes = [20, 25, 30, 35, 40, 45, 50, 35]
    N_groups = len(group_sizes)
    true_mu_alpha, true_sigma_alpha, true_beta, true_sigma = 70.0, 8.0, 3.5, 5.0
    true_alphas = rng.normal(true_mu_alpha, true_sigma_alpha, N_groups)
    X_all, y_all, group_all = [], [], []
    for g, n in enumerate(group_sizes):
        X = rng.normal(5.2, 2.1, n)
        y = true_alphas[g] + true_beta * X + rng.normal(0, true_sigma, n)
        X_all.extend(X); y_all.extend(y); group_all.extend([g] * n)
    return {"X": np.array(X_all, dtype=np.float32),
            "y": np.array(y_all, dtype=np.float32),
            "group_idx": np.array(group_all, dtype=np.int32),
            "N_groups": N_groups}
