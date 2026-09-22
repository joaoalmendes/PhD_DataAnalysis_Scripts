#!/usr/bin/env python3
"""
Ic_T_fit.py
===========

Collects single-temperature Josephson-junction analysis results produced
by the IV/dV-dI pipeline (one `analysis.log` per temperature, stored as

    singles/
        2.0K/analysis.log
        2.5K/analysis.log
        ...

), builds a weighted (constant) normal-state resistance R_n, and fits the
temperature-dependent critical current Ic(T) with the Ambegaokar-Baratoff
(AB) relation

    Ic(T) = (pi * Delta(T)) / (2 * e * R_n) * tanh( Delta(T) / (2 * kB * T) )

using the standard BCS interpolation formula for the gap

    Delta(T) = Delta0 * tanh( 1.74 * sqrt(Tc/T - 1) )      (T < Tc)
    Delta(T) = 0                                            (T >= Tc)

R_n is held fixed at the weighted average determined from all the
individual `Rn_mean_mOhm` values (weight ~ 1/rn_criterion, since a larger
rn_criterion means fewer points were used in that particular fit and
should therefore contribute less). Delta0 and Tc are the free fit
parameters.

Usage
-----
    python Ic_T_fit.py
    python Ic_T_fit.py --dir /path/to/main_folder
    python Ic_T_fit.py --normalize --Tc 7.2

Output
------
    - Ic_T_fit.pdf   (figure, saved in the directory the script is run from)
    - a summary printed to stdout with:
        * weighted R_n (mOhm)
        * fitted Delta0 (meV)
        * fitted Tc (K)
        * Ic(T=0) (mA)
        * R^2 of the fit
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------
# Physical constants (SI)
# ----------------------------------------------------------------------
E_CHARGE = 1.602176634e-19   # C
K_BOLTZ = 1.380649e-23       # J/K


# ----------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------
def set_paper_style():
    """Apply a consistent, publication-quality matplotlib style.

    Call this once, near the top of your plotting script, before
    creating any figures. Mirrors the look used in `paper_figure.py`
    (serif font, STIX math font, thin frames, inward ticks).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "savefig.bbox": "tight",
    })


# ----------------------------------------------------------------------
# Parsing of analysis.log files
# ----------------------------------------------------------------------
def _grab_float(pattern, text):
    """Return the first float matched by `pattern` in `text`, or None."""
    m = re.search(pattern, text, re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_log(path):
    """Extract T_K, Ic_mA, Rn_mean_mOhm and rn_criterion from a single
    analysis.log file. Returns a dict, with any missing field set to
    None."""
    with open(path, "r") as f:
        text = f.read()

    T_K = _grab_float(r"^T_K\s*=\s*([\-0-9.eE]+)", text)
    Ic_mA = _grab_float(r"^Ic_mA\s*=\s*([\-0-9.eE]+)", text)
    Rn_mean_mOhm = _grab_float(r"^Rn_mean_mOhm\s*=\s*([\-0-9.eE]+)", text)
    # rn_criterion appears twice (input parameters + results block) with
    # the same value; take the first occurrence.
    rn_criterion = _grab_float(r"^rn_criterion\s*=\s*([\-0-9.eE]+)", text)

    return {
        "T_K": T_K,
        "Ic_mA": Ic_mA,
        "Rn_mean_mOhm": Rn_mean_mOhm,
        "rn_criterion": rn_criterion,
        "path": path,
    }


def collect_data(base_dir):
    """Walk `base_dir`/singles/*/analysis.log and collect the parsed
    values into numpy arrays, sorted by temperature."""
    pattern = os.path.join(base_dir, "singles", "*", "analysis.log")
    log_files = sorted(glob.glob(pattern))

    if not log_files:
        raise FileNotFoundError(
            f"No analysis.log files found matching: {pattern}"
        )

    records = []
    for lf in log_files:
        rec = parse_log(lf)
        missing = [k for k in ("T_K", "Ic_mA", "Rn_mean_mOhm", "rn_criterion")
                   if rec[k] is None]
        if missing:
            print(f"  [skip] {lf}: could not parse {missing}", file=sys.stderr)
            continue
        records.append(rec)

    if not records:
        raise RuntimeError("No usable analysis.log files were parsed.")

    records.sort(key=lambda r: r["T_K"])

    T = np.array([r["T_K"] for r in records], dtype=float)
    Ic = np.array([r["Ic_mA"] for r in records], dtype=float)
    Rn = np.array([r["Rn_mean_mOhm"] for r in records], dtype=float)
    rnc = np.array([r["rn_criterion"] for r in records], dtype=float)

    return T, Ic, Rn, rnc


def weighted_Rn(Rn, rn_criterion):
    """Weighted average of Rn_mean_mOhm, weight ~ 1/rn_criterion.

    A larger rn_criterion means fewer points contributed to that Rn
    estimate, so it is down-weighted.
    """
    weights = 1.0 / rn_criterion
    Rn_avg = np.average(Rn, weights=weights)
    # weighted standard error, for reporting
    Rn_var = np.average((Rn - Rn_avg) ** 2, weights=weights)
    Rn_std = np.sqrt(Rn_var)
    return Rn_avg, Rn_std


# ----------------------------------------------------------------------
# Ambegaokar-Baratoff model
# ----------------------------------------------------------------------
def bcs_gap(T, Delta0_J, Tc):
    """BCS interpolation formula for the superconducting gap Delta(T),
    in Joules. Delta(T) = 0 for T >= Tc."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    Delta = np.zeros_like(T)
    mask = T < Tc
    ratio = Tc / T[mask] - 1.0
    ratio = np.clip(ratio, 0.0, None)
    Delta[mask] = Delta0_J * np.tanh(1.74 * np.sqrt(ratio))
    return Delta


def ic_ambegaokar_baratoff(T, Delta0_meV, Tc, Rn_Ohm):
    """Ambegaokar-Baratoff critical current, in mA.

    Delta0_meV : superconducting gap at T=0, in meV
    Tc         : critical temperature, in K
    Rn_Ohm     : (fixed) normal-state resistance, in Ohm
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    Delta0_J = Delta0_meV * 1e-3 * E_CHARGE
    Delta_T = bcs_gap(T, Delta0_J, Tc)

    T_safe = np.where(T > 0, T, np.nan)
    arg = np.divide(Delta_T, 2.0 * K_BOLTZ * T_safe,
                     out=np.zeros_like(Delta_T), where=T_safe > 0)
    Ic = (np.pi * Delta_T) / (2.0 * E_CHARGE * Rn_Ohm) * np.tanh(arg)
    Ic = np.nan_to_num(Ic, nan=0.0)
    return Ic * 1e3  # A -> mA


def ic0_from_gap(Delta0_meV, Rn_Ohm):
    """Ic(T=0) in mA, from the T -> 0 limit of the AB formula
    (tanh(...) -> 1)."""
    Delta0_J = Delta0_meV * 1e-3 * E_CHARGE
    return (np.pi * Delta0_J) / (2.0 * E_CHARGE * Rn_Ohm) * 1e3


def fit_ic_vs_T(T, Ic, Rn_Ohm, Tc_fixed=None):
    """Fit Delta0 (meV) [and, unless Tc_fixed is given, Tc (K)] to Ic(T)
    data, with Rn fixed.

    If `Tc_fixed` is None, both Delta0 and Tc are free fit parameters.
    If `Tc_fixed` is given (K), Tc is held fixed at that value and only
    Delta0 is fitted.

    Returns (popt, pcov, r_squared), where popt = (Delta0_meV, Tc) in
    both cases (Tc == Tc_fixed, with zero variance, when it was fixed).
    """
    Tc_guess = 1.2 * T.max() if Tc_fixed is None else Tc_fixed
    Delta0_guess = 1.764 * K_BOLTZ * Tc_guess / (1e-3 * E_CHARGE)  # BCS weak-coupling estimate, meV

    if Tc_fixed is None:
        def model(T, Delta0_meV, Tc):
            return ic_ambegaokar_baratoff(T, Delta0_meV, Tc, Rn_Ohm)

        p0 = [Delta0_guess, Tc_guess]
        bounds = ([1e-3, T.max() * 1.001], [50.0, 500.0])
        popt, pcov = curve_fit(model, T, Ic, p0=p0, bounds=bounds, maxfev=20000)
        Ic_fit = model(T, *popt)
    else:
        def model(T, Delta0_meV):
            return ic_ambegaokar_baratoff(T, Delta0_meV, Tc_fixed, Rn_Ohm)

        popt_1d, pcov_1d = curve_fit(model, T, Ic, p0=[Delta0_guess],
                                     bounds=([1e-3], [50.0]), maxfev=20000)
        Ic_fit = model(T, *popt_1d)
        # pad into the same 2-parameter shape (Delta0, Tc) used elsewhere,
        # with Tc fixed (zero variance / covariance)
        popt = np.array([popt_1d[0], Tc_fixed])
        pcov = np.zeros((2, 2))
        pcov[0, 0] = pcov_1d[0, 0]

    ss_res = np.sum((Ic - Ic_fit) ** 2)
    ss_tot = np.sum((Ic - np.mean(Ic)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    return popt, pcov, r_squared


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def make_plot(T, Ic, popt, Rn_Ohm, r_squared, normalize=False, Tc_norm=None,
              outpath="Ic_T_fit.pdf"):
    set_paper_style()

    Delta0_meV, Tc_fit = popt
    Ic0 = ic0_from_gap(Delta0_meV, Rn_Ohm)

    T_fine = np.linspace(1e-3, min(T.max() * 1.05, Tc_fit * 0.999), 500)
    Ic_fine = ic_ambegaokar_baratoff(T_fine, Delta0_meV, Tc_fit, Rn_Ohm)

    fig, ax = plt.subplots(figsize=(8.6 / 2.54, 6.5 / 2.54), constrained_layout=True)

    if normalize:
        x_data = T / Tc_norm
        x_fine = T_fine / Tc_norm
        y_data = Ic / Ic0
        y_fine = Ic_fine / Ic0
        ax.set_xlabel(r"$T / T_c$")
        ax.set_ylabel(r"$I_c(T) / I_c(0)$")
    else:
        x_data = T
        x_fine = T_fine
        y_data = Ic
        y_fine = Ic_fine
        ax.set_xlabel(r"$T$ (K)")
        ax.set_ylabel(r"$I_c$ (mA)")

    ax.plot(x_data, y_data, "ko", label="data")
    ax.plot(x_fine, y_fine, "b-", marker="none", label="AB fit")

    label = (rf"$\Delta_0$ = {Delta0_meV:.3f} meV" "\n"
              rf"$T_c$ = {Tc_fit:.2f} K" "\n"
              rf"$R_n$ = {Rn_Ohm * 1e3:.1f} m$\Omega$" "\n"
              rf"$R^2$ = {r_squared:.4f}")
    ax.text(0.03, 0.05, label, transform=ax.transAxes, fontsize=7,
            va="bottom", ha="left")

    ax.legend(loc="upper right")
    fig.savefig(outpath)
    plt.close(fig)
    return Ic0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=("Fit the Ambegaokar-Baratoff relation to Ic(T) data "
                     "collected from singles/<T>K/analysis.log files."))
    parser.add_argument("--dir", default=".",
                        help="Main folder containing the 'singles' directory "
                             "(default: current directory)")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize the plot axes: x -> T/Tc, "
                             "y -> Ic(T)/Ic(0). Requires --Tc.")
    parser.add_argument("--Tc", type=float, default=None,
                        help="Critical temperature (K). If given, only data "
                             "points with T < Tc are used (for the Rn "
                             "average and the fit). Also used to renormalize "
                             "the x-axis when --normalize is set.")
    parser.add_argument("--out", default="Ic_T_fit.pdf",
                        help="Output plot filename (default: Ic_T_fit.pdf)")
    args = parser.parse_args()

    if args.normalize and args.Tc is None:
        parser.error("--normalize requires --Tc <value> to be given.")

    # 1. Collect data
    print(f"Scanning '{os.path.join(args.dir, 'singles')}' for analysis.log files...")
    T, Ic, Rn, rnc = collect_data(args.dir)
    print(f"Found {len(T)} usable temperature points: "
          f"{', '.join(f'{t:g}K' for t in T)}")

    # 1b. Optionally restrict to T < Tc (independent of --normalize)
    if args.Tc is not None:
        keep = T < args.Tc
        n_dropped = np.count_nonzero(~keep)
        if n_dropped:
            print(f"--Tc={args.Tc:g}K given: dropping {n_dropped} point(s) "
                  f"with T >= Tc: "
                  f"{', '.join(f'{t:g}K' for t in T[~keep])}")
        T, Ic, Rn, rnc = T[keep], Ic[keep], Rn[keep], rnc[keep]
        if len(T) < 3:
            parser.error(f"Only {len(T)} point(s) remain with T < Tc="
                         f"{args.Tc:g}K; need at least 3 for a fit.")

    # 2. Weighted Rn
    Rn_avg_mOhm, Rn_std_mOhm = weighted_Rn(Rn, rnc)
    Rn_Ohm = Rn_avg_mOhm * 1e-3
    print(f"Weighted R_n = {Rn_avg_mOhm:.3f} +/- {Rn_std_mOhm:.3f} mOhm")

    # 3. Fit (Tc is held fixed at --Tc, if given; otherwise it is fitted)
    popt, pcov, r_squared = fit_ic_vs_T(T, Ic, Rn_Ohm, Tc_fixed=args.Tc)
    Delta0_meV, Tc_fit = popt
    perr = np.sqrt(np.diag(pcov))
    Ic0 = ic0_from_gap(Delta0_meV, Rn_Ohm)
    tc_was_fixed = args.Tc is not None

    # 4. Plot
    make_plot(T, Ic, popt, Rn_Ohm, r_squared,
              normalize=args.normalize, Tc_norm=args.Tc, outpath=args.out)

    # 5. Report
    print("\n--- Ambegaokar-Baratoff fit results ---")
    print(f"R_n (fixed, weighted avg) = {Rn_avg_mOhm:.3f} mOhm")
    print(f"Delta0                    = {Delta0_meV:.4f} +/- {perr[0]:.4f} meV")
    if tc_was_fixed:
        print(f"Tc (fixed, input)         = {Tc_fit:.4f} K")
    else:
        print(f"Tc (fit)                  = {Tc_fit:.4f} +/- {perr[1]:.4f} K")
    print(f"Ic(T=0)                   = {Ic0:.4f} mA")
    print(f"R^2                       = {r_squared:.5f}")
    print(f"\nPlot saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()