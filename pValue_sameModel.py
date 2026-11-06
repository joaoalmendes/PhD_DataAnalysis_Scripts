#!/usr/bin/env python3
"""
pValue_sameModel.py
===================

Compare two fits of the *same* model on the *same* data points and test
whether the observed difference in R² (or RMSE) is statistically
significant.

Input
-----
CSV files with at least the columns produced by the Fraunhofer fitting
pipeline:

    residual_uA, residual_sq, Ic_meas_uA, R2, RMSE_global, n_points, n_params
    (and optionally H_mT for alignment / diagnostics)

Typical usage is to pass two residual CSVs that correspond to two
independent optimisations (different seeds, different initial guesses,
or two slightly different parameter sets) of the identical functional
form.

Tests performed
---------------
1. Observed ΔR² and ΔRMSE (fit A − fit B).
2. Non-parametric bootstrap of ΔR²:
       - resample the n data points with replacement,
       - recompute R² from the paired residual vectors,
       - two-sided p-value = fraction of bootstrap replicates whose
         |ΔR²| is at least as large as the observed |ΔR²|.
3. Paired Wilcoxon signed-rank test on residual_sq (and optionally on
   |residual_uA|).

The bootstrap is the primary test for the R² difference; the Wilcoxon
test answers the related question “are the squared residuals of one fit
systematically smaller?”.

Example
-------
    python pValue_sameModel.py fitA.csv fitB.csv
    python pValue_sameModel.py fitA.csv fitB.csv --n-boot 20000 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def r2_from_residuals(y: np.ndarray, resid: np.ndarray) -> float:
    """Coefficient of determination from measured values and residuals."""
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def rmse_from_residuals(resid: np.ndarray) -> float:
    return float(np.sqrt(np.mean(resid ** 2)))


def bootstrap_delta_r2(
    y: np.ndarray,
    resid_a: np.ndarray,
    resid_b: np.ndarray,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """
    Bootstrap the difference ΔR² = R²_A − R²_B.

    Returns
    -------
    observed_delta : float
    p_value        : float   (two-sided)
    delta_boot     : ndarray of shape (n_boot,)
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    r2_a_obs = r2_from_residuals(y, resid_a)
    r2_b_obs = r2_from_residuals(y, resid_b)
    delta_obs = r2_a_obs - r2_b_obs

    # Pre-allocate
    delta_boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_s = y[idx]
        ra_s = resid_a[idx]
        rb_s = resid_b[idx]
        delta_boot[i] = r2_from_residuals(y_s, ra_s) - r2_from_residuals(y_s, rb_s)

    # Two-sided p-value under the null that the true difference is zero
    p_value = np.mean(np.abs(delta_boot) >= np.abs(delta_obs))
    # Continuity correction / avoid exact zero
    p_value = max(p_value, 1.0 / (n_boot + 1))

    return delta_obs, float(p_value), delta_boot


def paired_residual_tests(
    resid_a: np.ndarray,
    resid_b: np.ndarray,
) -> dict[str, float]:
    """Wilcoxon signed-rank tests on residual_sq and |residual|."""
    sq_a = resid_a ** 2
    sq_b = resid_b ** 2
    abs_a = np.abs(resid_a)
    abs_b = np.abs(resid_b)

    # Wilcoxon: positive statistic means A has larger values on average
    # (i.e. A is worse).  We report the two-sided p-value.
    w_sq, p_sq = stats.wilcoxon(sq_a, sq_b, alternative="two-sided", zero_method="wilcox")
    w_abs, p_abs = stats.wilcoxon(abs_a, abs_b, alternative="two-sided", zero_method="wilcox")

    return {
        "wilcoxon_sq_stat": float(w_sq),
        "wilcoxon_sq_p": float(p_sq),
        "wilcoxon_abs_stat": float(w_abs),
        "wilcoxon_abs_p": float(p_abs),
        "mean_sq_A": float(np.mean(sq_a)),
        "mean_sq_B": float(np.mean(sq_b)),
        "median_sq_A": float(np.median(sq_a)),
        "median_sq_B": float(np.median(sq_b)),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"residual_uA", "Ic_meas_uA"}


def load_fit_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df


def extract_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    y = df["Ic_meas_uA"].to_numpy(dtype=float)
    resid = df["residual_uA"].to_numpy(dtype=float)
    meta = {
        "R2_reported": float(df["R2"].iloc[0]) if "R2" in df.columns else np.nan,
        "RMSE_reported": float(df["RMSE_global"].iloc[0]) if "RMSE_global" in df.columns else np.nan,
        "n_points": int(df["n_points"].iloc[0]) if "n_points" in df.columns else len(df),
        "n_params": int(df["n_params"].iloc[0]) if "n_params" in df.columns else np.nan,
        "model": str(df["model"].iloc[0]) if "model" in df.columns else "?",
    }
    return y, resid, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap / paired residual test for ΔR² between two fits of the same model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_a", type=Path, help="Residual CSV for fit A")
    parser.add_argument("csv_b", type=Path, help="Residual CSV for fit B")
    parser.add_argument("--n-boot", type=int, default=10_000, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument(
        "--align-on",
        choices=["index", "H_mT"],
        default="index",
        help="How to pair the two residual vectors",
    )
    args = parser.parse_args(argv)

    df_a = load_fit_csv(args.csv_a)
    df_b = load_fit_csv(args.csv_b)

    if args.align_on == "H_mT":
        if "H_mT" not in df_a.columns or "H_mT" not in df_b.columns:
            raise ValueError("H_mT column required for --align-on H_mT")
        # Inner join on field values (rounded to avoid floating-point mismatch)
        df_a = df_a.copy()
        df_b = df_b.copy()
        df_a["_key"] = np.round(df_a["H_mT"].to_numpy(), decimals=6)
        df_b["_key"] = np.round(df_b["H_mT"].to_numpy(), decimals=6)
        merged = pd.merge(
            df_a, df_b, on="_key", suffixes=("_A", "_B"), how="inner"
        )
        if len(merged) == 0:
            raise RuntimeError("No overlapping H_mT values between the two CSVs")
        y = merged["Ic_meas_uA_A"].to_numpy(dtype=float)
        resid_a = merged["residual_uA_A"].to_numpy(dtype=float)
        resid_b = merged["residual_uA_B"].to_numpy(dtype=float)
        # Prefer reported meta from the original frames
        _, _, meta_a = extract_arrays(df_a)
        _, _, meta_b = extract_arrays(df_b)
        n = len(merged)
        print(f"Aligned on H_mT: {n} common points")
    else:
        if len(df_a) != len(df_b):
            raise ValueError(
                f"Length mismatch (index alignment): {len(df_a)} vs {len(df_b)}. "
                "Use --align-on H_mT or ensure the CSVs contain the same points in the same order."
            )
        y, resid_a, meta_a = extract_arrays(df_a)
        _, resid_b, meta_b = extract_arrays(df_b)
        n = len(y)

    # Sanity: measured values should be essentially identical
    if args.align_on == "index":
        y_b = df_b["Ic_meas_uA"].to_numpy(dtype=float)
        if not np.allclose(y, y_b, rtol=1e-5, atol=1e-8):
            print(
                "WARNING: Ic_meas_uA differs between the two files. "
                "Results assume the points are paired.",
                file=sys.stderr,
            )

    # Observed metrics (recomputed from residuals for consistency)
    r2_a = r2_from_residuals(y, resid_a)
    r2_b = r2_from_residuals(y, resid_b)
    rmse_a = rmse_from_residuals(resid_a)
    rmse_b = rmse_from_residuals(resid_b)
    delta_r2 = r2_a - r2_b
    delta_rmse = rmse_a - rmse_b

    # Bootstrap
    delta_obs, p_boot, delta_boot = bootstrap_delta_r2(
        y, resid_a, resid_b, n_boot=args.n_boot, seed=args.seed
    )
    assert np.isclose(delta_obs, delta_r2)

    # Paired residual tests
    paired = paired_residual_tests(resid_a, resid_b)

    # ---- Report ----
    print("=" * 60)
    print("Comparison of two fits of the same model")
    print("=" * 60)
    print(f"  Fit A : {args.csv_a.name}   model={meta_a['model']}")
    print(f"  Fit B : {args.csv_b.name}   model={meta_b['model']}")
    print(f"  n     : {n}")
    print()
    print("Observed performance")
    print(f"  R²_A  = {r2_a:.6f}   (reported {meta_a['R2_reported']:.6f})")
    print(f"  R²_B  = {r2_b:.6f}   (reported {meta_b['R2_reported']:.6f})")
    print(f"  ΔR²   = {delta_r2:+.6f}   (A − B)")
    print(f"  RMSE_A = {rmse_a:.6f} µA")
    print(f"  RMSE_B = {rmse_b:.6f} µA")
    print(f"  ΔRMSE  = {delta_rmse:+.6f} µA")
    print()
    print(f"Bootstrap test of ΔR²  (n_boot = {args.n_boot})")
    print(f"  two-sided p-value = {p_boot:.4g}")
    ci_lo, ci_hi = np.percentile(delta_boot, [2.5, 97.5])
    print(f"  95 % bootstrap CI for ΔR² : [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print()
    print("Paired Wilcoxon signed-rank tests")
    print(f"  residual_sq : W = {paired['wilcoxon_sq_stat']:.4g},  p = {paired['wilcoxon_sq_p']:.4g}")
    print(f"  |residual|  : W = {paired['wilcoxon_abs_stat']:.4g},  p = {paired['wilcoxon_abs_p']:.4g}")
    print(f"  mean(resid²)_A = {paired['mean_sq_A']:.6f},  mean(resid²)_B = {paired['mean_sq_B']:.6f}")
    print(f"  median(resid²)_A = {paired['median_sq_A']:.6f},  median(resid²)_B = {paired['median_sq_B']:.6f}")
    print("=" * 60)

    # Simple interpretation line
    better = "A" if delta_r2 > 0 else "B"
    if p_boot < 0.05:
        print(f"Conclusion: the difference in R² is statistically significant "
              f"(p = {p_boot:.4g}); fit {better} is superior.")
    else:
        print(f"Conclusion: the observed ΔR² is not statistically significant "
              f"(p = {p_boot:.4g}) at the 5 % level.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
