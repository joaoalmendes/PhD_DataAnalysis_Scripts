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
temperature-dependent critical current Ic(T) with Ambegaokar-Baratoff (AB)
models following Talantsev, Physica C 623, 1354549 (2024).

Supported models
----------------
  ab       Classic Ambegaokar-Baratoff (s-wave weak-coupling limit).
           Gap (paper Eq. 2):
               Delta(T) = Delta0 * tanh(1.74 * sqrt(Tc/T - 1))
           Current (standard AB, tanh uses instantaneous T):
               Ic(T) = eta * (pi * Delta(T)) / (2 * e * Rn)
                            * tanh( Delta(T) / (2 * kB * T) )
           Free parameters: Delta0, [Tc], [eta].

  d-wave   Paper d-wave form (zeta = 1).  Two variants:

           * Smooth (default, Eq. 7a) — free params:
                 y0 (>=0), eta, Delta0, Tc, DeltaC_el/gammaTc
               Ic(T) = y0 + eta * (pi Delta(T))/(2 e Rn)
                            * tanh(Delta(T)/(2 kB Tc))

           * Piecewise (--piecewise, Eq. 7) — for data with a low-T
             inflection.  Requires --Tc (held fixed).  Five free params:
                 Delta0, eta, DeltaC_el/gammaTc, Tm, k
               Ic(T) = H(Tm-T)*(a + k*T) + H(T-Tm)*AB(T)
               with a chosen for continuity at Tm.

           Gap (Eq. 4, zeta = 1) in both cases:
               Delta(T) = Delta0 * tanh(
                   (pi kB Tc / Delta0)
                   * sqrt( (DeltaC_el/gamma Tc) * (Tc/T - 1) ) )

R_n is taken once from the first usable analysis.log (the pipeline
stores the same Rn_mean_mOhm in every log, evaluated near Tc; the
source temperature Rn_from_T_K is reported when present).

Data selection
--------------
  By default the script uses Ic+_f_mA (and Ic+_b_mA as the retrapping
  current Ir when plotting without a fit).  Pass --eff to use the
  effective values Ic_eff+_f_mA and Ir_eff+_b_mA instead.

CSV workflow
------------
  After collecting data from logs the script always writes a CSV
  (default: Ic_T_data.csv) with columns
      T_K, Ic_mA, Ir_mA, Rn_mean_mOhm, rn_criterion, source
  so you can edit values by hand.  Subsequent runs can skip the folder
  scan entirely with --from-csv PATH.

Usage examples
--------------
    python Ic_T_fit.py
    python Ic_T_fit.py --dir /path/to/main_folder
    python Ic_T_fit.py --eff --model d-wave --fit-eta
    python Ic_T_fit.py --model d-wave --fit-eta --fit-y0
    python Ic_T_fit.py --no-fit --eff
    python Ic_T_fit.py --from-csv Ic_T_data.csv --model ab --Tc 7.2
    python Ic_T_fit.py --icrn --model d-wave --fit-eta
    python Ic_T_fit.py --normalize --Tc 7.2 --fit-eta

Output
------
    - Ic_T_fit.pdf   (or --out path)
    - CSV of collected points
    - summary on stdout (Rn, Delta0, Tc, eta, dCel, R^2, ...)
"""

import argparse
import csv
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
    """Apply a consistent, publication-quality matplotlib style."""
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


def parse_log(path, use_eff=False):
    """Extract temperature, Ic, Ir, Rn and rn_criterion from one log.

    When use_eff is True the effective values are preferred:
        Ic  <- Ic_eff+_f_mA
        Ir  <- Ir_eff+_b_mA
    otherwise:
        Ic  <- Ic+_f_mA
        Ir  <- Ic+_b_mA

    Falls back to the older single-key format (Ic_mA) if needed.
    """
    with open(path, "r") as f:
        text = f.read()

    T_K = _grab_float(r"^  T_K\s*=\s*([\-0-9.eE]+)", text)
    if T_K is None:
        T_K = _grab_float(r"^T_K\s*=\s*([\-0-9.eE]+)", text)

    if use_eff:
        Ic_mA = _grab_float(r"^  Ic_eff\+_f_mA\s*=\s*([\-0-9.eE]+)", text)
        Ir_mA = _grab_float(r"^  Ir_eff\+_b_mA\s*=\s*([\-0-9.eE]+)", text)
    else:
        Ic_mA = _grab_float(r"^  Ic\+_f_mA\s*=\s*([\-0-9.eE]+)", text)
        Ir_mA = _grab_float(r"^  Ic\+_b_mA\s*=\s*([\-0-9.eE]+)", text)

    if Ic_mA is None:
        Ic_mA = _grab_float(r"^Ic_mA\s*=\s*([\-0-9.eE]+)", text)

    Rn_mean_mOhm = _grab_float(r"^  Rn_mean_mOhm\s*=\s*([\-0-9.eE]+)", text)
    if Rn_mean_mOhm is None:
        Rn_mean_mOhm = _grab_float(r"^Rn_mean_mOhm\s*=\s*([\-0-9.eE]+)", text)

    # temperature at which Rn was evaluated (typically ~2 K below Tc)
    Rn_from_T_K = _grab_float(r"^  Rn_from_T_K\s*=\s*([\-0-9.eE]+)", text)
    if Rn_from_T_K is None:
        Rn_from_T_K = _grab_float(r"^Rn_from_T_K\s*=\s*([\-0-9.eE]+)", text)

    rn_criterion = _grab_float(r"^rn_criterion\s*=\s*([\-0-9.eE]+)", text)

    return {
        "T_K": T_K,
        "Ic_mA": Ic_mA,
        "Ir_mA": Ir_mA,
        "Rn_mean_mOhm": Rn_mean_mOhm,
        "Rn_from_T_K": Rn_from_T_K,
        "rn_criterion": rn_criterion,
        "path": path,
    }


def collect_data(base_dir, use_eff=False):
    """Find every analysis.log under the temperature subfolders.

    Accepts either:
      base_dir/singles/<T>K/analysis.log   (base_dir is the main folder)
      base_dir/<T>K/analysis.log           (base_dir is the singles folder itself)

    Both layouts are tried; the first that yields files is used.
    """
    base_dir = os.path.abspath(os.path.expanduser(base_dir))
    candidates = [
        os.path.join(base_dir, "singles", "*", "analysis.log"),
        os.path.join(base_dir, "*", "analysis.log"),
    ]
    log_files = []
    used_pattern = None
    for pattern in candidates:
        found = sorted(glob.glob(pattern))
        if found:
            log_files = found
            used_pattern = pattern
            break

    if not log_files:
        raise FileNotFoundError(
            "No analysis.log files found. Tried:\n  "
            + "\n  ".join(candidates)
            + f"\n(base_dir resolved to: {base_dir})"
        )

    print(f"  matched pattern: {used_pattern}  ({len(log_files)} file(s))")

    records = []
    for lf in log_files:
        rec = parse_log(lf, use_eff=use_eff)
        # T and Ic are mandatory; Rn is taken from the first usable log later
        missing = [k for k in ("T_K", "Ic_mA") if rec[k] is None]
        if missing:
            print(f"  [skip] {lf}: could not parse {missing}", file=sys.stderr)
            continue
        records.append(rec)

    if not records:
        raise RuntimeError("No usable analysis.log files were parsed.")

    records.sort(key=lambda r: r["T_K"])
    return records


def records_to_arrays(records):
    """Convert list of record dicts to numpy arrays (T, Ic, Ir)."""
    T = np.array([r["T_K"] for r in records], dtype=float)
    Ic = np.array([r["Ic_mA"] for r in records], dtype=float)
    Ir = np.array([r["Ir_mA"] if r["Ir_mA"] is not None else np.nan
                   for r in records], dtype=float)
    return T, Ic, Ir


def pick_Rn(records):
    """Take Rn from the first usable log and the temperature it came from.

    The pipeline stores the same Rn_mean_mOhm in every analysis.log
    (evaluated once from the near-Tc dataset, typically ~2 K below Tc).
    We therefore use the value from the first record that has it, and
    report Rn_from_T_K when available.
    """
    for r in records:
        Rn = r.get("Rn_mean_mOhm")
        if Rn is not None and np.isfinite(Rn):
            Rn_from = r.get("Rn_from_T_K")
            return float(Rn), (float(Rn_from) if Rn_from is not None else None)
    raise RuntimeError(
        "No usable Rn_mean_mOhm found in any analysis.log / CSV row."
    )


def save_csv(path, records):
    """Write collected records to a CSV for manual editing / reuse."""
    fieldnames = ["T_K", "Ic_mA", "Ir_mA", "Rn_mean_mOhm", "Rn_from_T_K",
                  "rn_criterion", "source"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({
                "T_K": r["T_K"],
                "Ic_mA": r["Ic_mA"],
                "Ir_mA": r["Ir_mA"] if r["Ir_mA"] is not None else "",
                "Rn_mean_mOhm": (r["Rn_mean_mOhm"]
                                 if r.get("Rn_mean_mOhm") is not None else ""),
                "Rn_from_T_K": (r["Rn_from_T_K"]
                                if r.get("Rn_from_T_K") is not None else ""),
                "rn_criterion": (r["rn_criterion"]
                                 if r.get("rn_criterion") is not None else ""),
                "source": r.get("path", ""),
            })
    print(f"Data CSV written to: {os.path.abspath(path)}")


def load_csv(path):
    """Load records previously written by save_csv (or hand-edited)."""
    records = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                T = float(row["T_K"])
                Ic = float(row["Ic_mA"])
            except (KeyError, ValueError) as e:
                print(f"  [skip] CSV row: {e}", file=sys.stderr)
                continue

            def _opt_float(key):
                s = row.get(key, "").strip()
                return float(s) if s else None

            records.append({
                "T_K": T,
                "Ic_mA": Ic,
                "Ir_mA": _opt_float("Ir_mA"),
                "Rn_mean_mOhm": _opt_float("Rn_mean_mOhm"),
                "Rn_from_T_K": _opt_float("Rn_from_T_K"),
                "rn_criterion": _opt_float("rn_criterion"),
                "path": row.get("source", path),
            })
    if not records:
        raise RuntimeError(f"No usable rows in CSV: {path}")
    records.sort(key=lambda r: r["T_K"])
    return records


# ----------------------------------------------------------------------
# Gap models  (Talantsev, Physica C 623, 1354549 (2024))
# ----------------------------------------------------------------------
def bcs_gap_classic(T, Delta0_J, Tc):
    """Classic weak-coupling s-wave gap (paper Eq. 2), in Joules."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    Delta = np.zeros_like(T)
    mask = T < Tc
    ratio = Tc / T[mask] - 1.0
    ratio = np.clip(ratio, 0.0, None)
    Delta[mask] = Delta0_J * np.tanh(1.74 * np.sqrt(ratio))
    return Delta


def bcs_gap_dwave(T, Delta0_J, Tc, dCel):
    """Advanced gap for d-wave (paper Eq. 4 with zeta = 1), in Joules.

    dCel = Delta C_el / (gamma * Tc)
    At T = 0 the gap is Delta0 (limit of the formula as T -> 0).
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    Delta = np.zeros_like(T)
    if Delta0_J <= 0 or Tc <= 0:
        return Delta
    # T == 0 -> Delta0; 0 < T < Tc -> Eq. (4); T >= Tc -> 0
    zero = T == 0
    Delta[zero] = Delta0_J
    mask = (T > 0) & (T < Tc)
    if np.any(mask):
        prefactor = (np.pi * K_BOLTZ * Tc) / Delta0_J
        ratio = Tc / T[mask] - 1.0
        ratio = np.clip(ratio, 0.0, None)
        inner = prefactor * np.sqrt(dCel * ratio)
        Delta[mask] = Delta0_J * np.tanh(inner)
    return Delta


def ic_ab_classic(T, Delta0_meV, Tc, Rn_Ohm, eta=1.0):
    """Classic Ambegaokar-Baratoff Ic in mA (s-wave weak-coupling).

    Gap: paper Eq. (2).
    Current: standard AB with tanh(Delta / (2 kB T)).
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    Delta0_J = Delta0_meV * 1e-3 * E_CHARGE
    Delta_T = bcs_gap_classic(T, Delta0_J, Tc)

    T_safe = np.where(T > 0, T, np.nan)
    arg = np.divide(Delta_T, 2.0 * K_BOLTZ * T_safe,
                     out=np.zeros_like(Delta_T), where=T_safe > 0)
    Ic = eta * (np.pi * Delta_T) / (2.0 * E_CHARGE * Rn_Ohm) * np.tanh(arg)
    Ic = np.nan_to_num(Ic, nan=0.0)
    return Ic * 1e3  # A -> mA


def _ab_core_mA(Delta_T, Tc, Rn_Ohm, eta):
    """Shared AB core: eta * (pi Delta) / (2 e Rn) * tanh(Delta/(2 kB Tc)), in mA.

    Matches paper Eq. (1): tanh argument uses Tc.
    """
    if Tc <= 0:
        return np.zeros_like(Delta_T)
    arg = Delta_T / (2.0 * K_BOLTZ * Tc)
    Ic_A = eta * (np.pi * Delta_T) / (2.0 * E_CHARGE * Rn_Ohm) * np.tanh(arg)
    return np.nan_to_num(Ic_A, nan=0.0) * 1e3  # A -> mA


def ic_ab_dwave(T, Delta0_meV, Tc, Rn_Ohm, eta=1.0, dCel=0.95, y0=0.0):
    """d-wave AB without inflection (paper Eqs. 1 + 4 / 7a), Ic in mA.

    Gap: paper Eq. (4) with zeta = 1.
    Current:
        Ic = y0 + eta * (pi * Delta(T)) / (2 e Rn)
                   * tanh( Delta(T) / (2 kB Tc) )

    Free parameters of Eq. (7a): y0, eta, Delta0, Tc, dCel.
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    Delta0_J = Delta0_meV * 1e-3 * E_CHARGE
    Delta_T = bcs_gap_dwave(T, Delta0_J, Tc, dCel)
    return y0 + _ab_core_mA(Delta_T, Tc, Rn_Ohm, eta)


def ic_ab_dwave_piecewise(T, Delta0_meV, Tc, Rn_Ohm, eta=1.0, dCel=0.95,
                          Tm=None, k=0.0):
    """d-wave AB with low-T linear inflection (paper Eq. 7), Ic in mA.

        Ic(T) = H(Tm - T) * (a + k * T)
              + H(T - Tm) * AB(T)

    where AB is the standard paper Eq. (1) form (no y0), and continuity
    at T = Tm fixes
        a = AB(Tm) - k * Tm

    Free parameters (as used by Talantsev for Zhao et al. data):
        k, Tm, eta, Delta0, dCel  (+ Tc always free in this script).

    k is stored in mA/K internally; always reported / labelled in uA/K.
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    if Tm is None:
        Tm = 0.4 * Tc
    Delta0_J = Delta0_meV * 1e-3 * E_CHARGE
    Delta_T = bcs_gap_dwave(T, Delta0_J, Tc, dCel)
    AB = _ab_core_mA(Delta_T, Tc, Rn_Ohm, eta)

    # AB value at Tm for continuity
    Delta_Tm = bcs_gap_dwave(np.array([Tm]), Delta0_J, Tc, dCel)
    AB_Tm = float(_ab_core_mA(Delta_Tm, Tc, Rn_Ohm, eta)[0])
    a = AB_Tm - k * Tm

    linear = a + k * T
    out = np.where(T < Tm, linear, AB)
    # at exactly Tm both sides agree
    out = np.where(np.isclose(T, Tm), AB_Tm, out)
    return np.nan_to_num(out, nan=0.0)


def ic0_from_gap(Delta0_meV, Rn_Ohm, eta=1.0):
    """Classic AB Ic(T->0) in mA: tanh(Delta/(2 kB T)) -> 1."""
    Delta0_J = Delta0_meV * 1e-3 * E_CHARGE
    return eta * (np.pi * Delta0_J) / (2.0 * E_CHARGE * Rn_Ohm) * 1e3


def ic0_from_model(popt, Rn_Ohm):
    """Ic(T=0) = value of the fitted curve at T = 0 (mA).

    Uses the exact same model function as the plot, evaluated at T=0.
    """
    Delta0 = popt["Delta0_meV"]
    Tc = popt["Tc"]
    eta = popt.get("eta", 1.0)
    model = popt.get("model", "ab")
    piecewise = popt.get("piecewise", False)
    T0 = np.array([0.0])

    if model == "ab":
        # classic AB is singular at T=0 in tanh(Delta/(2kT)); use T->0 limit
        return float(ic_ab_classic(np.array([1e-9]), Delta0, Tc, Rn_Ohm,
                                   eta=eta)[0])

    dCel = popt.get("dCel", 0.95)
    if piecewise:
        return float(ic_ab_dwave_piecewise(
            T0, Delta0, Tc, Rn_Ohm,
            eta=eta, dCel=dCel,
            Tm=popt["Tm"], k=popt["k"])[0])

    # smooth d-wave Eq. (7a)
    y0 = popt.get("y0", 0.0)
    return float(ic_ab_dwave(T0, Delta0, Tc, Rn_Ohm,
                             eta=eta, dCel=dCel, y0=y0)[0])


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
def fit_ic_vs_T(T, Ic, Rn_Ohm, model="ab", Tc_guess=None, fit_eta=False,
                fit_y0=False, piecewise=False, Tm_initial=None):
    """Fit Ic(T) for the chosen model.

    model : 'ab' | 'd-wave'
    piecewise : if True and model=='d-wave', use paper Eq. (7) (linear
                low-T + AB above Tm) instead of Eq. (7a).
                Requires Tc_guess: Tc is held fixed so the five free
                parameters are Delta0, eta, dCel, Tm, k.
    Tm_initial : optional initial guess for Tm (K) when piecewise;
                 default is the mid-range of the data temperatures.

    For classic AB (--model ab), if Tc_guess is given then Tc is held
    fixed.  For smooth d-wave (Eq. 7a) Tc is always free; Tc_guess is
    only an initial value / lower-bound hint.

    Returns
        popt        dict of best-fit (and fixed) parameters
        pcov        covariance matrix (order matches free_names)
        free_names  list of names of free parameters
        r_squared
    """
    if piecewise and model == "d-wave":
        if Tc_guess is None:
            raise ValueError(
                "Piecewise Eq. (7) requires --Tc <value>: Tc is held fixed "
                "so there are exactly five free parameters "
                "(Delta0, eta, dCel, Tm, k)."
            )
        Tc_fixed = float(Tc_guess)
        Tc_for_guess = Tc_fixed
    elif model == "ab" and Tc_guess is not None:
        # classic AB: --Tc holds Tc fixed (data cut + fixed parameter)
        Tc_fixed = float(Tc_guess)
        Tc_for_guess = Tc_fixed
    else:
        Tc_fixed = None
        Tc_for_guess = (1.2 * float(T.max()) if Tc_guess is None
                        else float(Tc_guess))

    # weak-coupling Delta0 estimates (meV)
    if model == "d-wave":
        Delta0_guess = (4.28 / 2.0) * K_BOLTZ * Tc_for_guess / (1e-3 * E_CHARGE)
        dCel_guess = 0.95
    else:
        Delta0_guess = (3.53 / 2.0) * K_BOLTZ * Tc_for_guess / (1e-3 * E_CHARGE)
        dCel_guess = None
    eta_guess = 1.0
    y0_guess = 0.0
    # piecewise: Tm initial guess (user or mid-range of data), k near 0
    if Tm_initial is not None:
        Tm_guess = float(Tm_initial)
    else:
        Tm_guess = 0.5 * (float(T.min()) + float(T.max()))
    k_guess = 0.0

    free_names = ["Delta0_meV"]
    p0 = [Delta0_guess]
    lo = [1e-3]
    hi = [50.0]

    if Tc_fixed is None:
        # non-piecewise: Tc is free
        free_names.append("Tc")
        p0.append(Tc_for_guess)
        lo.append(float(T.max()) * 1.001)
        hi.append(500.0)

    # piecewise always fits eta (one of the five Eq. 7 parameters);
    # otherwise only if --fit-eta
    if piecewise or fit_eta:
        free_names.append("eta")
        p0.append(eta_guess)
        lo.append(1e-3)
        hi.append(10.0)

    if model == "d-wave":
        free_names.append("dCel")
        p0.append(dCel_guess)
        lo.append(0.01)
        hi.append(5.0)

        if piecewise:
            # Eq. (7): free Tm and k (Tc fixed) → five free params total
            tm_lo = float(T.min()) * 0.5
            tm_hi = min(float(T.max()) * 0.99, Tc_fixed * 0.99)
            Tm_guess = float(np.clip(Tm_guess, tm_lo, tm_hi))
            free_names.append("Tm")
            p0.append(Tm_guess)
            lo.append(tm_lo)
            hi.append(tm_hi)

            free_names.append("k")
            p0.append(k_guess)
            span = float(np.max(np.abs(Ic))) if len(Ic) else 1.0
            Tspan = max(float(T.max()) - float(T.min()), 1.0)
            lo.append(-2.0 * span / Tspan)
            hi.append(2.0 * span / Tspan)
        elif fit_y0:
            # Eq. (7a): optional positive offset only
            free_names.append("y0")
            p0.append(y0_guess)
            span = float(np.max(np.abs(Ic))) if len(Ic) else 1.0
            lo.append(0.0)          # y0 >= 0
            hi.append(span)

    def model_fn(T_arr, *params):
        kw = dict(zip(free_names, params))
        Delta0 = kw["Delta0_meV"]
        Tc = kw.get("Tc", Tc_fixed)
        eta = kw.get("eta", 1.0)
        if model == "d-wave":
            dCel = kw["dCel"]
            if piecewise:
                return ic_ab_dwave_piecewise(
                    T_arr, Delta0, Tc, Rn_Ohm,
                    eta=eta, dCel=dCel,
                    Tm=kw["Tm"], k=kw["k"])
            else:
                y0 = kw.get("y0", 0.0)
                return ic_ab_dwave(T_arr, Delta0, Tc, Rn_Ohm,
                                  eta=eta, dCel=dCel, y0=y0)
        else:
            return ic_ab_classic(T_arr, Delta0, Tc, Rn_Ohm, eta=eta)

    popt_arr, pcov = curve_fit(
        model_fn, T, Ic, p0=p0, bounds=(lo, hi), maxfev=30000)

    Ic_fit = model_fn(T, *popt_arr)
    ss_res = np.sum((Ic - Ic_fit) ** 2)
    ss_tot = np.sum((Ic - np.mean(Ic)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    popt = dict(zip(free_names, popt_arr))
    if "Tc" not in popt:
        popt["Tc"] = Tc_fixed
    if "eta" not in popt:
        popt["eta"] = 1.0
    if "dCel" not in popt:
        popt["dCel"] = None
    if "y0" not in popt:
        popt["y0"] = 0.0
    if "Tm" not in popt:
        popt["Tm"] = None
    if "k" not in popt:
        popt["k"] = None
    popt["model"] = model
    popt["piecewise"] = bool(piecewise and model == "d-wave")

    return popt, pcov, free_names, r_squared



# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def make_plot(T, Ic, Ir, popt, Rn_Ohm, r_squared, free_names,
              normalize=False, Tc_norm=None, outpath="Ic_T_fit.pdf",
              no_fit=False, plot_icrn=False, fit_eta=False, model="ab"):
    set_paper_style()

    fig, ax = plt.subplots(figsize=(8.6 / 2.54, 6.5 / 2.54),
                           constrained_layout=True)

    # only non-negative currents are plotted (physical Ic, Ir >= 0)
    pos = np.isfinite(Ic) & (Ic >= 0)
    T_pos, Ic_pos = T[pos], Ic[pos]

    if plot_icrn:
        # log convention: IcRn_mV = Ic_mA * Rn_mean_mOhm * 1e-3
        # (mA * mOhm = uV; * 1e-3 -> mV)
        Rn_mOhm = Rn_Ohm * 1e3
        y_data = Ic_pos * Rn_mOhm * 1e-3
        y_label = r"$R_n I_c$ (mV)"
        if Ir is not None and np.any(np.isfinite(Ir)):
            ir_pos = np.isfinite(Ir) & (Ir >= 0)
            y_Ir = np.where(ir_pos, Ir * Rn_mOhm * 1e-3, np.nan)
            T_Ir = T
        else:
            y_Ir = None
            T_Ir = None
    else:
        y_data = Ic_pos
        # when no-fit and Ir is shown, use generic I; otherwise Ic
        y_label = r"$I$ (mA)" if no_fit else r"$I_c$ (mA)"
        if Ir is not None and np.any(np.isfinite(Ir)):
            ir_pos = np.isfinite(Ir) & (Ir >= 0)
            y_Ir = np.where(ir_pos, Ir, np.nan)
            T_Ir = T
        else:
            y_Ir = None
            T_Ir = None

    if normalize and Tc_norm is not None:
        x_data = T_pos / Tc_norm
        ax.set_xlabel(r"$T / T_c$")
    else:
        x_data = T_pos
        ax.set_xlabel(r"$T$ (K)")

    ax.plot(x_data, y_data, "ko",
            label=r"$I_c$" if not plot_icrn else r"$R_n I_c$")
    if no_fit and y_Ir is not None:
        x_Ir = (T_Ir / Tc_norm) if (normalize and Tc_norm is not None) else T_Ir
        ax.plot(x_Ir, y_Ir, "ks", mfc="none",
                label=r"$I_r$" if not plot_icrn else r"$R_n I_r$")

    if not no_fit and popt is not None:
        Tc_fit = popt["Tc"]
        Delta0 = popt["Delta0_meV"]
        eta = popt["eta"]
        dCel = popt.get("dCel")
        y0 = popt.get("y0", 0.0)
        piecewise = popt.get("piecewise", False)
        Tm = popt.get("Tm")
        k = popt.get("k")

        T_fine = np.linspace(1e-3, min(float(T.max()) * 1.05,
                                       Tc_fit * 0.999), 500)
        if model == "d-wave":
            if piecewise:
                Ic_fine = ic_ab_dwave_piecewise(
                    T_fine, Delta0, Tc_fit, Rn_Ohm,
                    eta=eta, dCel=dCel, Tm=Tm, k=k)
            else:
                Ic_fine = ic_ab_dwave(T_fine, Delta0, Tc_fit, Rn_Ohm,
                                     eta=eta, dCel=dCel, y0=y0)
        else:
            Ic_fine = ic_ab_classic(T_fine, Delta0, Tc_fit, Rn_Ohm, eta=eta)

        # never draw negative fit values
        Ic_fine = np.maximum(Ic_fine, 0.0)

        if plot_icrn:
            Rn_mOhm = Rn_Ohm * 1e3
            y_fine = Ic_fine * Rn_mOhm * 1e-3   # mA * mOhm * 1e-3 = mV
        else:
            y_fine = Ic_fine

        if normalize and Tc_norm is not None:
            x_fine = T_fine / Tc_norm
        else:
            x_fine = T_fine

        if normalize and not plot_icrn:
            Ic0 = ic0_from_model(popt, Rn_Ohm)
            if Ic0 != 0:
                ax.clear()
                ax.plot(x_data, y_data / Ic0, "ko", label="data")
                ax.plot(x_fine, y_fine / Ic0, "b-", marker="none", label="fit")
                ax.set_xlabel(r"$T / T_c$")
                ax.set_ylabel(r"$I_c(T) / I_c(0)$")
                y_label = r"$I_c(T) / I_c(0)$"
            else:
                ax.plot(x_fine, y_fine, "b-", marker="none", label="fit")
                ax.set_ylabel(y_label)
        else:
            ax.plot(x_fine, y_fine, "b-", marker="none", label="fit")
            ax.set_ylabel(y_label)

        label_lines = [
            rf"$\Delta_0$ = {Delta0:.3f} meV",
            rf"$T_c$ = {Tc_fit:.2f} K",
        ]
        if fit_eta or ("eta" in free_names):
            label_lines.append(rf"$\eta$ = {eta:.3f}")
        if model == "d-wave" and dCel is not None:
            label_lines.append(
                rf"$\Delta C_{{\rm el}}/\gamma T_c$ = {dCel:.3f}")
            if piecewise:
                if Tm is not None:
                    label_lines.append(rf"$T_m$ = {Tm:.2f} K")
                if k is not None:
                    # always display in uA/K (fit stores mA/K)
                    label_lines.append(
                        rf"$k$ = {k * 1e3:.4g} $\mu$A/K")
        if "y0" in free_names:
            label_lines.append(rf"$y_0$ = {y0:.4f} mA")
        if r_squared is not None:
            label_lines.append(rf"$R^2$ = {r_squared:.4f}")
        ax.text(0.03, 0.05, "\n".join(label_lines),
                transform=ax.transAxes, fontsize=7, va="bottom", ha="left")
    else:
        ax.set_ylabel(y_label)

    ax.legend(loc="upper right")
    fig.savefig(outpath)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit Ambegaokar-Baratoff (classic s-wave / d-wave Eq. 7a) models "
            "to Ic(T) data from singles/<T>K/analysis.log files, or from a CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Argument summary
----------------
  Data source
    --dir DIR          Path to the main folder (containing 'singles/') OR to
                       the 'singles/' folder itself.  Globs every
                       <T>K/analysis.log underneath (default: current
                       directory).  Ignored if --from-csv is given.
    --from-csv FILE    Read T, Ic, Ir, Rn, rn_criterion from this CSV instead
                       of scanning logs.
    --save-csv FILE    Write the collected points to FILE
                       (default: Ic_T_data.csv).

  Which Ic / Ir
    --eff              Use effective values Ic_eff+_f_mA and Ir_eff+_b_mA.
                       Without this flag: Ic+_f_mA and Ic+_b_mA (or legacy
                       Ic_mA).

  Fit control
    --no-fit           Only plot the data (no curve fit).  When Ir is
                       available it is plotted as open squares.
    --model {ab,d-wave}
                       ab     = classic AB (s-wave weak-coupling, Eq. 2)
                       d-wave = paper d-wave form with zeta = 1
                       (default: ab)
    --piecewise        For --model d-wave only: use paper Eq. (7) with a
                       low-T linear segment instead of smooth Eq. (7a).
                       Requires --Tc (held fixed).  Five free parameters:
                       Delta0, eta, dCel, Tm, k.
    --Tm-initial K     Initial guess for Tm (K) under --piecewise
                       (default: mid-range of data temperatures).
    --fit-eta          Let the dimensionless prefactor eta vary; without
                       this flag eta is fixed at 1.  (Always free under
                       --piecewise.)
    --fit-y0           Also fit a constant offset y0 >= 0 (mA).  Only for
                       non-piecewise d-wave (Eq. 7a).
    --Tc FLOAT         Only keep data with T <= Tc (K).
                       ab: also holds Tc fixed in the fit.
                       d-wave --piecewise: required and held fixed.
                       d-wave (Eq. 7a): data cut + initial guess (Tc free).

  Plot options
    --icrn             Plot Rn*Ic (mV) instead of Ic (mA).
    --normalize        Axes: T/Tc  and  Ic(T)/Ic(0).  Requires --Tc.
    --out FILE         Output figure path (default: Ic_T_fit.pdf).
""",
    )
    parser.add_argument("--dir", default=".",
                        help="Path to the main folder (containing 'singles/') "
                             "or to the 'singles/' folder itself. "
                             "Globs every <T>K/analysis.log underneath.")
    parser.add_argument("--from-csv", default=None, metavar="FILE",
                        help="Load data from CSV instead of scanning logs")
    parser.add_argument("--save-csv", default="Ic_T_data.csv", metavar="FILE",
                        help="Write collected data to this CSV "
                             "(default: Ic_T_data.csv)")
    parser.add_argument("--eff", action="store_true",
                        help="Use effective Ic/Ir (Ic_eff+_f_mA, "
                             "Ir_eff+_b_mA) instead of Ic+_f_mA / Ic+_b_mA")
    parser.add_argument("--no-fit", action="store_true",
                        help="Plot data only; do not perform a fit")
    parser.add_argument("--model", choices=("ab", "d-wave"),
                        default="ab",
                        help="Gap / AB model: ab = classic s-wave AB; "
                             "d-wave = paper d-wave form (default: ab)")
    parser.add_argument("--piecewise", action="store_true",
                        help="For d-wave only: use paper Eq. (7) with low-T "
                             "linear inflection (Tm, k) instead of smooth "
                             "Eq. (7a). Requires --Tc (held fixed). Five free "
                             "params: Delta0, eta, dCel, Tm, k.")
    parser.add_argument("--Tm-initial", type=float, default=None,
                        dest="Tm_initial", metavar="K",
                        help="Initial guess for Tm (K) in --piecewise fits. "
                             "Default: mid-range of the data temperatures.")
    parser.add_argument("--fit-eta", action="store_true",
                        help="Treat eta as a free fit parameter "
                             "(default: eta = 1 fixed)")
    parser.add_argument("--fit-y0", action="store_true",
                        help="Also fit a constant offset y0 >= 0 (mA); only "
                             "for non-piecewise d-wave (Eq. 7a)")
    parser.add_argument("--Tc", type=float, default=None,
                        help="Only analyse temperatures T <= Tc (K). "
                             "For --model ab: also holds Tc fixed in the fit. "
                             "For --piecewise: required and held fixed. "
                             "For smooth d-wave (Eq. 7a): data cut + initial "
                             "guess only (Tc remains free).")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalise axes to T/Tc and Ic/Ic(0); needs --Tc")
    parser.add_argument("--icrn", action="store_true",
                        help="Plot Rn*Ic (mV) instead of Ic (mA)")
    parser.add_argument("--out", default="Ic_T_fit.pdf",
                        help="Output plot filename (default: Ic_T_fit.pdf)")
    args = parser.parse_args()

    if args.normalize and args.Tc is None:
        parser.error("--normalize requires --Tc <value>")
    if args.piecewise and args.model != "d-wave":
        parser.error("--piecewise only applies with --model d-wave")
    if args.piecewise and args.fit_y0:
        parser.error("--fit-y0 is for Eq. (7a) only; do not combine with "
                     "--piecewise (Eq. 7 uses Tm, k instead of y0)")
    if args.piecewise and args.Tc is None:
        parser.error("--piecewise requires --Tc <value> (Tc is held fixed; "
                     "the five free parameters are Delta0, eta, dCel, Tm, k)")
    if args.Tm_initial is not None and not args.piecewise:
        parser.error("--Tm-initial only applies with --piecewise")

    # ------------------------------------------------------------------
    # 1. Collect data
    # ------------------------------------------------------------------
    if args.from_csv:
        print(f"Loading data from CSV: {args.from_csv}")
        records = load_csv(args.from_csv)
    else:
        print(f"Scanning '{os.path.abspath(os.path.expanduser(args.dir))}' "
              f"for analysis.log files "
              f"({'effective' if args.eff else 'standard'} Ic/Ir)...")
        records = collect_data(args.dir, use_eff=args.eff)

    save_csv(args.save_csv, records)

    # ------------------------------------------------------------------
    # 2. Rn from the first usable log (same value in every log: evaluated
    #    once near Tc, typically ~2 K below Tc)
    # ------------------------------------------------------------------
    Rn_mOhm, Rn_from_T = pick_Rn(records)
    Rn_Ohm = Rn_mOhm * 1e-3
    if Rn_from_T is not None:
        print(f"R_n = {Rn_mOhm:.3f} mOhm  (from T = {Rn_from_T:.4g} K data)")
    else:
        print(f"R_n = {Rn_mOhm:.3f} mOhm  (Rn_from_T_K not reported in log)")

    # optional T <= Tc cut (does NOT fix Tc in the fit)
    if args.Tc is not None:
        before = len(records)
        records = [r for r in records if r["T_K"] <= args.Tc]
        n_dropped = before - len(records)
        if n_dropped:
            print(f"--Tc={args.Tc:g}K: keeping T <= Tc; dropped {n_dropped} "
                  f"point(s) above Tc")
        if len(records) < 2 and not args.no_fit:
            parser.error(f"Only {len(records)} point(s) remain with T <= Tc="
                         f"{args.Tc:g}K; need at least 2 for a fit.")

    T, Ic, Ir = records_to_arrays(records)
    print(f"Using {len(T)} temperature point(s): "
          f"{', '.join(f'{t:g}K' for t in T)}")

    # ------------------------------------------------------------------
    # 3. Fit (unless --no-fit); Tc is always free
    # ------------------------------------------------------------------
    popt = None
    pcov = None
    free_names = []
    r_squared = None
    err_map = {}

    if not args.no_fit:
        if len(T) < 3:
            parser.error(f"Only {len(T)} point(s); need at least 3 for a fit.")
        popt, pcov, free_names, r_squared = fit_ic_vs_T(
            T, Ic, Rn_Ohm,
            model=args.model,
            Tc_guess=args.Tc,          # initial guess only; Tc stays free
            fit_eta=args.fit_eta,
            fit_y0=args.fit_y0,
            piecewise=args.piecewise,
            Tm_initial=args.Tm_initial,
        )
        perr = np.sqrt(np.diag(pcov))
        err_map = dict(zip(free_names, perr))

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    make_plot(
        T, Ic, Ir, popt, Rn_Ohm, r_squared, free_names,
        normalize=args.normalize, Tc_norm=args.Tc, outpath=args.out,
        no_fit=args.no_fit, plot_icrn=args.icrn,
        fit_eta=args.fit_eta, model=args.model,
    )

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print("\n--- Ambegaokar-Baratoff fit results ---")
    model_label = args.model
    if args.model == "d-wave":
        model_label += " piecewise Eq.(7)" if args.piecewise else " Eq.(7a)"
    print(f"Model                       = {model_label}")
    if Rn_from_T is not None:
        print(f"R_n (fixed)                 = {Rn_mOhm:.3f} mOhm "
              f"(from T = {Rn_from_T:.4g} K)")
    else:
        print(f"R_n (fixed)                 = {Rn_mOhm:.3f} mOhm")
    if args.no_fit:
        print("(no fit performed)")
    else:
        Delta0 = popt["Delta0_meV"]
        Tc_fit = popt["Tc"]
        eta = popt["eta"]
        print(f"Delta0                      = {Delta0:.4f}"
              + (f" +/- {err_map['Delta0_meV']:.4f}"
                 if "Delta0_meV" in err_map else "")
              + " meV")
        if args.piecewise or (args.model == "ab" and args.Tc is not None):
            print(f"Tc (fixed, input)           = {Tc_fit:.4f} K")
        else:
            print(f"Tc (fit)                    = {Tc_fit:.4f}"
                  + (f" +/- {err_map['Tc']:.4f}" if "Tc" in err_map else "")
                  + " K")
        if args.fit_eta or args.piecewise:
            print(f"eta (fit)                   = {eta:.4f}"
                  + (f" +/- {err_map['eta']:.4f}" if "eta" in err_map else ""))
        else:
            print(f"eta (fixed)                 = {eta:.4f}")
        if args.model == "d-wave" and popt.get("dCel") is not None:
            dCel = popt["dCel"]
            print(f"DeltaC_el / (gamma Tc)      = {dCel:.4f}"
                  + (f" +/- {err_map['dCel']:.4f}" if "dCel" in err_map
                     else ""))
            kB_meV = K_BOLTZ / E_CHARGE * 1e3
            bcs_ratio = 2.0 * Delta0 / (kB_meV * Tc_fit)
            print(f"2 Delta0 / (kB Tc)          = {bcs_ratio:.3f}")
        if args.piecewise:
            Tm = popt.get("Tm")
            k = popt.get("k")
            if Tm is not None:
                print(f"Tm (fit)                    = {Tm:.4f}"
                      + (f" +/- {err_map['Tm']:.4f}" if "Tm" in err_map
                         else "")
                      + " K")
            if k is not None:
                # report in uA/K (internal fit parameter is mA/K)
                k_uA = k * 1e3
                k_err = (err_map['k'] * 1e3) if "k" in err_map else None
                print(f"k (fit)                     = {k_uA:.6g}"
                      + (f" +/- {k_err:.6g}" if k_err is not None else "")
                      + " uA/K")
        if args.fit_y0:
            print(f"y0 (fit)                    = {popt['y0']:.4f}"
                  + (f" +/- {err_map['y0']:.4f}" if "y0" in err_map else "")
                  + " mA")
        Ic0 = ic0_from_model(popt, Rn_Ohm)
        print(f"Ic(T=0)                     = {Ic0:.4f} mA")
        print(f"R^2                         = {r_squared:.5f}")
    print(f"\nPlot saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
