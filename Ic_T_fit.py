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
  Each analysis.log writes an "Ic definition for this run" block set by
  plot_csv_data.py (--ic-dvdi-range):

      Ic_definition: jump_dvdi  →  use Ic+_f_mA  (dV/dI jump)
      Ic_definition: rho_eff    →  use Ic_eff+_f_mA  (rho criterion)

  This script reads that flag per temperature and selects the matching
  Ic (and Ir) automatically.  No CLI switch is required.  Legacy logs
  without the flag default to rho_eff (Ic_eff) when available.

CSV workflow
------------
  After collecting data from logs the script always writes a CSV
  (default: Ic_T_data.csv) with columns
      T_K, Ic_mA, Ir_mA, Ic_eff_mA, Ir_eff_mA,
      Ic_active_mA, Ir_active_mA, Ic_definition,
      Rn_mean_mOhm, rn_criterion, source
  so you can edit values by hand.  Subsequent runs can skip the folder
  scan entirely with --from-csv PATH.

Usage examples
--------------
    python Ic_T_fit.py
    python Ic_T_fit.py --dir /path/to/main_folder
    python Ic_T_fit.py --model d-wave --fit-eta
    python Ic_T_fit.py --model d-wave --fit-eta --fit-y0
    python Ic_T_fit.py --no-fit
    python Ic_T_fit.py --from-csv Ic_T_data.csv --model ab --Tc 7.2
    python Ic_T_fit.py --icrn --model d-wave --fit-eta
    python Ic_T_fit.py --normalize --Tc 7.2 --fit-eta

Output
------
    - Ic_T_fit.pdf   (or --out path)
    - CSV of collected points
    - summary on stdout (Rn, per-segment Delta0, Tc, eta, dCel, Tm, k,
      linear intercept I(0), and R^2)
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq

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


def _parse_ic_definition(text):
    """Return 'jump_dvdi' or 'rho_eff' from an analysis.log body.

    Looks for the block written by plot_csv_data.py::

        --- Ic definition for this run ---
          USE_IC       : Ic  (dV/dI jump method)
          USE_IC_EFF   : no  (stored for reference only)
          Ic_definition: jump_dvdi

    or the rho_eff variant.  Falls back to scanning
    ``Ic_definition: <token>`` alone.  Logs without the flag are
    treated as ``rho_eff`` (prefer Ic_eff when available).
    """
    m = re.search(
        r"^  Ic_definition\s*:\s*(jump_dvdi|rho_eff)\s*$",
        text, re.MULTILINE,
    )
    if m:
        return m.group(1)
    m = re.search(
        r"^Ic_definition\s*[=:]\s*(jump_dvdi|rho_eff)\s*$",
        text, re.MULTILINE,
    )
    if m:
        return m.group(1)
    # USE_IC / USE_IC_EFF lines (robust to spacing)
    use_ic = re.search(
        r"^  USE_IC\s*:\s*(Ic\b|no)\b", text, re.MULTILINE | re.IGNORECASE,
    )
    use_eff_line = re.search(
        r"^  USE_IC_EFF\s*:\s*(Ic_eff\b|no)\b",
        text, re.MULTILINE | re.IGNORECASE,
    )
    if use_ic and re.match(r"Ic\b", use_ic.group(1), re.IGNORECASE):
        return "jump_dvdi"
    if use_eff_line and re.match(r"Ic_eff\b", use_eff_line.group(1), re.IGNORECASE):
        return "rho_eff"
    return "rho_eff"  # legacy default


def parse_log(path):
    """Extract temperature, Ic, Ir, Rn and rn_criterion from one log.

    Always reads both jump (Ic+_f_mA) and effective (Ic_eff+_f_mA)
    currents when present.  The active Ic/Ir used for fitting/plotting
    is chosen from the log's ``Ic_definition`` flag:

        jump_dvdi  →  Ic+_f_mA / Ic+_b_mA
        rho_eff    →  Ic_eff+_f_mA / Ir_eff+_b_mA

    Legacy logs without the flag default to rho_eff when Ic_eff is
    available, otherwise fall back to the jump value.
    """
    with open(path, "r") as f:
        text = f.read()

    T_K = _grab_float(r"^  T_K\s*=\s*([\-0-9.eE]+)", text)
    if T_K is None:
        T_K = _grab_float(r"^T_K\s*=\s*([\-0-9.eE]+)", text)

    Ic_raw = _grab_float(r"^  Ic\+_f_mA\s*=\s*([\-0-9.eE]+)", text)
    Ir_raw = _grab_float(r"^  Ic\+_b_mA\s*=\s*([\-0-9.eE]+)", text)
    Ic_eff = _grab_float(r"^  Ic_eff\+_f_mA\s*=\s*([\-0-9.eE]+)", text)
    Ir_eff = _grab_float(r"^  Ir_eff\+_b_mA\s*=\s*([\-0-9.eE]+)", text)

    legacy = _grab_float(r"^Ic_mA\s*=\s*([\-0-9.eE]+)", text)
    if Ic_raw is None and legacy is not None:
        Ic_raw = legacy

    ic_def = _parse_ic_definition(text)

    if ic_def == "jump_dvdi":
        Ic_active = Ic_raw if Ic_raw is not None else Ic_eff
        Ir_active = Ir_raw if Ir_raw is not None else Ir_eff
    else:  # rho_eff
        Ic_active = Ic_eff if Ic_eff is not None else Ic_raw
        Ir_active = Ir_eff if Ir_eff is not None else Ir_raw

    Rn_mean_mOhm = _grab_float(r"^  Rn_mean_mOhm\s*=\s*([\-0-9.eE]+)", text)
    if Rn_mean_mOhm is None:
        Rn_mean_mOhm = _grab_float(r"^Rn_mean_mOhm\s*=\s*([\-0-9.eE]+)", text)

    Rn_from_T_K = _grab_float(r"^  Rn_from_T_K\s*=\s*([\-0-9.eE]+)", text)
    if Rn_from_T_K is None:
        Rn_from_T_K = _grab_float(r"^Rn_from_T_K\s*=\s*([\-0-9.eE]+)", text)

    rn_criterion = _grab_float(r"^rn_criterion\s*=\s*([\-0-9.eE]+)", text)

    return {
        "T_K": T_K,
        "Ic_mA": Ic_active,          # active value used for fit/plot
        "Ir_mA": Ir_active,
        "Ic_raw_mA": Ic_raw,         # jump (Ic+_f_mA)
        "Ir_raw_mA": Ir_raw,         # jump retrapping (Ic+_b_mA)
        "Ic_eff_mA": Ic_eff,         # rho criterion
        "Ir_eff_mA": Ir_eff,
        "Ic_definition": ic_def,
        "Rn_mean_mOhm": Rn_mean_mOhm,
        "Rn_from_T_K": Rn_from_T_K,
        "rn_criterion": rn_criterion,
        "path": path,
    }


def collect_data(base_dir):
    """Find every analysis.log under the temperature subfolders.

    Accepts either:
      base_dir/singles/<T>K/analysis.log   (base_dir is the main folder)
      base_dir/<T>K/analysis.log           (base_dir is the singles folder itself)

    Both layouts are tried; the first that yields files is used.
    Active Ic/Ir is taken per-log from the ``Ic_definition`` flag.
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
    n_jump = n_eff = 0
    for lf in log_files:
        rec = parse_log(lf)
        # T and Ic are mandatory; Rn is taken from the first usable log later
        missing = [k for k in ("T_K", "Ic_mA") if rec[k] is None]
        if missing:
            print(f"  [skip] {lf}: could not parse {missing}", file=sys.stderr)
            continue
        if rec.get("Ic_definition") == "jump_dvdi":
            n_jump += 1
        else:
            n_eff += 1
        records.append(rec)

    if not records:
        raise RuntimeError("No usable analysis.log files were parsed.")

    print(f"  Ic_definition:  jump_dvdi = {n_jump}  |  rho_eff = {n_eff}")
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
    """Write collected records to a CSV for manual editing / reuse.

    Columns:
      Ic_mA / Ir_mA               jump values (Ic+_f_mA / Ic+_b_mA)
      Ic_eff_mA / Ir_eff_mA       rho criterion (Ic_eff+_f_mA / Ir_eff+_b_mA)
      Ic_active_mA / Ir_active_mA values used for this run (per Ic_definition)
      Ic_definition               jump_dvdi | rho_eff
    """
    fieldnames = [
        "T_K",
        "Ic_mA", "Ir_mA",
        "Ic_eff_mA", "Ir_eff_mA",
        "Ic_active_mA", "Ir_active_mA",
        "Ic_definition",
        "Rn_mean_mOhm", "Rn_from_T_K", "rn_criterion", "source",
    ]

    def _cell(v):
        return "" if v is None else v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({
                "T_K": r["T_K"],
                "Ic_mA": _cell(r.get("Ic_raw_mA", r.get("Ic_mA"))),
                "Ir_mA": _cell(r.get("Ir_raw_mA", r.get("Ir_mA"))),
                "Ic_eff_mA": _cell(r.get("Ic_eff_mA")),
                "Ir_eff_mA": _cell(r.get("Ir_eff_mA")),
                "Ic_active_mA": _cell(r.get("Ic_mA")),
                "Ir_active_mA": _cell(r.get("Ir_mA")),
                "Ic_definition": _cell(r.get("Ic_definition", "")),
                "Rn_mean_mOhm": _cell(r.get("Rn_mean_mOhm")),
                "Rn_from_T_K": _cell(r.get("Rn_from_T_K")),
                "rn_criterion": _cell(r.get("rn_criterion")),
                "source": r.get("path", ""),
            })
    print(f"Data CSV written to: {os.path.abspath(path)}")


def load_csv(path):
    """Load records previously written by save_csv (or hand-edited).

    Preference order for the active Ic:
      1. Ic_active_mA (previous run / hand edit)
      2. Rebuild from Ic_definition + Ic_mA / Ic_eff_mA columns
      3. Ic_eff_mA then Ic_mA
    """
    records = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def _opt_float(key):
                s = (row.get(key) or "").strip()
                return float(s) if s else None

            try:
                T = float(row["T_K"])
            except (KeyError, ValueError) as e:
                print(f"  [skip] CSV row: {e}", file=sys.stderr)
                continue

            ic_def = (row.get("Ic_definition") or "").strip() or None
            Ic_raw = _opt_float("Ic_mA")
            Ic_eff = _opt_float("Ic_eff_mA")
            Ir_raw = _opt_float("Ir_mA")
            Ir_eff = _opt_float("Ir_eff_mA")

            Ic = _opt_float("Ic_active_mA")
            if Ic is None and ic_def == "jump_dvdi":
                Ic = Ic_raw if Ic_raw is not None else Ic_eff
            if Ic is None and ic_def == "rho_eff":
                Ic = Ic_eff if Ic_eff is not None else Ic_raw
            if Ic is None:
                Ic = Ic_eff if Ic_eff is not None else Ic_raw
            if Ic is None:
                print(f"  [skip] CSV row: no Ic column", file=sys.stderr)
                continue

            Ir = _opt_float("Ir_active_mA")
            if Ir is None and ic_def == "jump_dvdi":
                Ir = Ir_raw if Ir_raw is not None else Ir_eff
            if Ir is None and ic_def == "rho_eff":
                Ir = Ir_eff if Ir_eff is not None else Ir_raw
            if Ir is None:
                Ir = Ir_eff if Ir_eff is not None else Ir_raw

            if ic_def not in ("jump_dvdi", "rho_eff"):
                ic_def = "rho_eff" if Ic_eff is not None else "jump_dvdi"

            records.append({
                "T_K": T,
                "Ic_mA": Ic,
                "Ir_mA": Ir,
                "Ic_raw_mA": Ic_raw,
                "Ir_raw_mA": Ir_raw,
                "Ic_eff_mA": Ic_eff,
                "Ir_eff_mA": Ir_eff,
                "Ic_definition": ic_def,
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
                          Tm=None, k=0.0, y0=0.0):
    """d-wave AB with one low-T linear segment (paper Eq. 7), Ic in mA.

        Ic(T) = H(Tm - T) * (a + k * T) + H(T - Tm) * AB(T)
        a = AB(Tm) - k * Tm   (continuity)

    k stored in mA/K internally; reported in uA/K.
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    if Tm is None:
        Tm = 0.4 * Tc
    Delta0_J = Delta0_meV * 1e-3 * E_CHARGE
    Delta_T = bcs_gap_dwave(T, Delta0_J, Tc, dCel)
    AB = _ab_core_mA(Delta_T, Tc, Rn_Ohm, eta)

    Delta_Tm = bcs_gap_dwave(np.array([Tm]), Delta0_J, Tc, dCel)
    AB_Tm = float(_ab_core_mA(Delta_Tm, Tc, Rn_Ohm, eta)[0])
    AB = AB + y0
    AB_Tm = AB_Tm + y0
    a = AB_Tm - k * Tm

    linear = a + k * T
    out = np.where(T < Tm, linear, AB)
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
    y0 = popt.get("y0", 0.0)
    if piecewise:
        return float(ic_ab_dwave_piecewise(
            T0, Delta0, Tc, Rn_Ohm,
            eta=eta, dCel=dCel,
            Tm=popt["Tm"], k=popt["k"], y0=y0)[0])

    # smooth d-wave Eq. (7a)
    return float(ic_ab_dwave(T0, Delta0, Tc, Rn_Ohm,
                             eta=eta, dCel=dCel, y0=y0)[0])


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
def fit_ic_vs_T(T, Ic, Rn_Ohm, model="ab", Tc_guess=None, fit_eta=False,
                fit_y0=False, piecewise=False, Tm_initial=None):
    """Fit Ic(T) for the chosen model.

    model : 'ab' | 'd-wave'
    piecewise : single-Heaviside Eq. (7); Tc fixed; free: Delta0, eta,
                dCel, Tm, k.
    Tm_initial : optional initial guess for Tm (K) when piecewise.

    Returns
        popt, pcov, free_names, r_squared
    """
    if piecewise and model == "d-wave":
        if Tc_guess is None:
            raise ValueError(
                "Piecewise fits require --Tc <value> (Tc held fixed)."
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
    # piecewise initial guesses
    T_lo, T_hi = float(T.min()), float(T.max())
    mid = 0.5 * (T_lo + T_hi)
    Tm_guess = float(Tm_initial) if Tm_initial is not None else mid
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
            # Eq. (7): free Tm and k  (Tc is fixed → Tc_fixed is not None)
            span = float(np.max(np.abs(Ic))) if len(Ic) else 1.0
            Tspan = max(T_hi - T_lo, 1.0)
            k_bound = 2.0 * span / Tspan
            tm_lo = T_lo * 0.5
            tm_hi = min(T_hi * 0.99, float(Tc_fixed) * 0.99)
            Tm_guess = float(np.clip(Tm_guess, tm_lo, tm_hi))
            free_names.append("Tm")
            p0.append(Tm_guess)
            lo.append(tm_lo)
            hi.append(tm_hi)
            free_names.append("k")
            p0.append(k_guess)
            lo.append(-k_bound)
            hi.append(k_bound)
            if fit_y0:
                # vertical offset of the AB branch (and continuity at Tm)
                free_names.append("y0")
                p0.append(y0_guess)
                lo.append(0.0)          # y0 >= 0
                hi.append(span)
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
            y0 = kw.get("y0", 0.0)
            if piecewise:
                return ic_ab_dwave_piecewise(
                    T_arr, Delta0, Tc, Rn_Ohm,
                    eta=eta, dCel=dCel,
                    Tm=kw["Tm"], k=kw["k"], y0=y0)
            else:
                return ic_ab_dwave(T_arr, Delta0, Tc, Rn_Ohm,
                                  eta=eta, dCel=dCel, y0=y0)
        else:
            return ic_ab_classic(T_arr, Delta0, Tc, Rn_Ohm, eta=eta)

    popt_arr, pcov = curve_fit(
        model_fn, T, Ic, p0=p0, bounds=(lo, hi), maxfev=8000)

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
    for key in ("Tm", "k"):
        if key not in popt:
            popt[key] = None
    popt["model"] = model
    popt["piecewise"] = bool(piecewise and model == "d-wave")

    return popt, pcov, free_names, r_squared


def fit_piecewise_multiples(T, Ic, Rn_Ohm, n_segments, Tc, splits=None,
                            Tm_initial=None, fit_y0=False):
    """Fit N independent piecewise Eq. (7) models in consecutive T windows.

    If ``splits`` is supplied, its values are fixed window boundaries. If it is
    omitted, the boundaries are selected automatically by minimizing the total
    squared residual across all segments (dynamic programming over contiguous
    temperature windows). Each segment has independent Delta0, eta, dCel, Tm,
    and k; Tc is fixed globally.

    With ``fit_y0=True`` the highest-temperature segment (fit 1) also frees
    the vertical offset y0 of its AB branch.  Lower-T segments still inherit
    y0 from the continuity anchor at the shared boundary.
    """
    from functools import lru_cache

    n_segments = int(n_segments)
    if n_segments < 2:
        raise ValueError("--piecewise-multiples requires an integer >= 2")
    T = np.asarray(T, dtype=float)
    Ic = np.asarray(Ic, dtype=float)
    if T.ndim != 1 or Ic.ndim != 1 or len(T) != len(Ic) or len(T) < 5 * n_segments:
        raise ValueError("T and Ic must be equal-length 1D arrays with at least 5 points per segment")
    order = np.argsort(T)
    T, Ic = T[order], Ic[order]
    T_min, T_max = float(T.min()), float(T.max())
    n_splits = n_segments - 1
    fixed_splits = splits is not None and len(splits) > 0

    if fixed_splits:
        boundaries = sorted(float(v) for v in splits)
        if len(boundaries) != n_splits:
            raise ValueError(f"--piecewise-multiples {n_segments} needs {n_splits} split value(s); got {len(boundaries)}.")
        if any(not T_min < v < T_max for v in boundaries):
            raise ValueError("All split temperatures must lie strictly inside the data temperature range.")
        if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
            raise ValueError("Split temperatures must be distinct and strictly increasing.")
        edges = [T_min] + boundaries + [T_max]
        windows = []
        for i in range(n_segments):
            # Half-open windows prevent a measured point being counted twice.
            mask = ((T >= edges[i]) & (T < edges[i + 1])) if i < n_segments - 1 else ((T >= edges[i]) & (T <= edges[i + 1]))
            indices = np.flatnonzero(mask)
            if len(indices) < 5:
                raise ValueError(
                    f"Segment {i + 1} [{edges[i]:.4g}, {edges[i+1]:.4g}] K has only {len(indices)} point(s); "
                    "each independent five-parameter fit needs at least 5. Change the split temperatures or use fewer segments.")
            windows.append((int(indices[0]), int(indices[-1] + 1)))
    else:
        # Split candidates lie between adjacent measured temperatures. Dynamic
        # programming selects the partition with the smallest total SSE, so the
        # complete curve—not an equal-spacing guess—determines the boundaries.
        windows = None

    span = max(float(np.max(np.abs(Ic))), 1e-6)
    k_bound = max(2.0 * span / max(T_max - T_min, 1e-6), 1e-6)
    Delta0_0 = (4.28 / 2.0) * K_BOLTZ * Tc / (1e-3 * E_CHARGE)

    @lru_cache(maxsize=None)
    def fit_window(i0, i1, anchor=None, free_y0=False):
        Ts, Ics = T[i0:i1], Ic[i0:i1]
        if len(Ts) < 5:
            return None
        tm_lo = max(1e-6, float(Ts.min()) * 0.5)
        tm_hi = min(float(Ts.max()) * 0.999, float(Tc) * 0.99)
        if tm_hi <= tm_lo:
            tm_lo = max(1e-6, float(Ts.min()) * 0.25)
            tm_hi = min(float(Ts.max()) * 0.999, float(Tc) * 0.99)
        if tm_hi <= tm_lo:
            return None
        # --Tm-initial values are ordered by fitting order: highest-T segment
        # first, then progressively lower-temperature segments. During automatic
        # boundary search, infer the segment rank from this candidate window.
        tm_guess = None
        if Tm_initial is not None:
            tm_values = list(Tm_initial) if isinstance(Tm_initial, (list, tuple, np.ndarray)) else [Tm_initial]
            if len(tm_values) == n_segments:
                mid_index = 0.5 * (i0 + i1)
                low_to_high_rank = min(n_segments - 1, int(mid_index / max(len(T), 1) * n_segments))
                fit_rank = n_segments - 1 - low_to_high_rank
                tm_guess = float(tm_values[fit_rank])
            elif len(tm_values) == 1:
                tm_guess = float(tm_values[0])
        tm0 = tm_guess if tm_guess is not None else 0.5 * (float(Ts.min()) + float(Ts.max()))
        tm0 = float(np.clip(tm0, tm_lo + 1e-9, tm_hi - 1e-9))
        # free y0 only for the highest-T segment (anchor is None) when requested
        use_free_y0 = bool(free_y0) and anchor is None
        y0_hi = max(float(np.max(np.abs(Ics))), 1e-6)
        if use_free_y0:
            free_names_win = ["Delta0_meV", "eta", "dCel", "Tm", "k", "y0"]
            p0 = [Delta0_0, 1.0, 0.95, tm0, 0.0, 0.0]
            lo = [1e-3, 1e-3, 0.01, tm_lo, -k_bound, 0.0]
            hi = [50.0, 10.0, 5.0, tm_hi, k_bound, y0_hi]
        else:
            free_names_win = ["Delta0_meV", "eta", "dCel", "Tm", "k"]
            p0 = [Delta0_0, 1.0, 0.95, tm0, 0.0]
            lo = [1e-3, 1e-3, 0.01, tm_lo, -k_bound]
            hi = [50.0, 10.0, 5.0, tm_hi, k_bound]

        def segment_fn(x, *params):
            Delta0, eta, dCel, Tm, k = params[:5]
            if use_free_y0:
                y0 = params[5]
            else:
                y0 = 0.0
                if anchor is not None:
                    ta, ia = anchor
                    da = bcs_gap_dwave(np.array([ta]), Delta0 * 1e-3 * E_CHARGE, Tc, dCel)
                    ab_a = float(_ab_core_mA(da, Tc, Rn_Ohm, eta)[0])
                    y0 = ia - ab_a
            return ic_ab_dwave_piecewise(x, Delta0, Tc, Rn_Ohm,
                                         eta=eta, dCel=dCel, Tm=Tm, k=k, y0=y0)
        try:
            popt_arr, pcov = curve_fit(segment_fn, Ts, Ics, p0=p0,
                                       bounds=(lo, hi), maxfev=8000)
            pred = segment_fn(Ts, *popt_arr)
        except (RuntimeError, ValueError, FloatingPointError):
            return None
        sse = float(np.sum((Ics - pred) ** 2))
        ss_tot = float(np.sum((Ics - np.mean(Ics)) ** 2))
        r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0
        n_par = len(popt_arr)
        perr = (np.sqrt(np.maximum(np.diag(pcov), 0.0))
                if pcov is not None else np.full(n_par, np.nan))
        Delta0, eta, dCel, Tm, k = map(float, popt_arr[:5])
        if use_free_y0:
            y0 = float(popt_arr[5])
        else:
            y0 = 0.0
            if anchor is not None:
                ta, ia = anchor
                da = bcs_gap_dwave(np.array([ta]), Delta0 * 1e-3 * E_CHARGE, Tc, dCel)
                y0 = ia - float(_ab_core_mA(da, Tc, Rn_Ohm, eta)[0])
        Delta_Tm = bcs_gap_dwave(np.array([Tm]), Delta0 * 1e-3 * E_CHARGE, Tc, dCel)
        AB_Tm = float(_ab_core_mA(Delta_Tm, Tc, Rn_Ohm, eta)[0])
        a0 = AB_Tm + y0 - k * Tm
        return {"popt_arr": popt_arr, "pcov": pcov, "pred": pred, "sse": sse,
                "r_squared": r2, "param_err": perr, "a0_mA": float(a0),
                "y0": float(y0), "free_names": free_names_win,
                "T": Ts.copy(), "Ic": Ics.copy(), "i0": i0, "i1": i1}

    if not fixed_splits:
        n = len(T)
        # Start from equal-count windows, then coordinate-optimize each split.
        # Every candidate is scored using the summed SSE of the adjacent fits,
        # so the boundaries respond to the full curve rather than one segment.
        # Ts split boundaries are distinct from each segment's internal Tm.
        # Start the automatic boundary search from equal-count windows; the
        # --Tm-initial values are used only as initial guesses for each segment's
        # internal Tm parameter inside fit_window().
        split_idx = [round((i + 1) * n / n_segments) for i in range(n_splits)]
        split_idx = [max(5 * (i + 1), min(n - 5 * (n_segments - i - 1), v))
                     for i, v in enumerate(split_idx)]
        split_idx = sorted(split_idx)
        for _iteration in range(4):
            changed = False
            for b_idx in range(n_splits):
                left_edge = 0 if b_idx == 0 else split_idx[b_idx - 1]
                right_edge = n if b_idx == n_splits - 1 else split_idx[b_idx + 1]
                best_j = split_idx[b_idx]
                best_cost = np.inf
                for j in range(left_edge + 5, right_edge - 4):
                    left_fit = fit_window(left_edge, j)
                    right_fit = fit_window(j, right_edge)
                    if left_fit is None or right_fit is None:
                        continue
                    cost = left_fit["sse"] + right_fit["sse"]
                    if cost < best_cost:
                        best_cost, best_j = cost, j
                if best_j != split_idx[b_idx]:
                    split_idx[b_idx] = best_j
                    changed = True
            if not changed:
                break
        windows = []
        endpoints = [0] + split_idx + [n]
        windows = [(endpoints[i], endpoints[i + 1]) for i in range(n_segments)]
        boundaries = [0.5 * (T[j - 1] + T[j]) for j in split_idx]
    else:
        # The supplied boundaries can fall between measured temperatures;
        # windows were assigned by those fixed temperatures above.
        boundaries = list(sorted(float(v) for v in splits))

    results = []
    all_pred = np.full_like(Ic, np.nan, dtype=float)
    # Number fits from highest-T window downwards. Each lower-T segment's AB
    # branch is vertically shifted so it meets the previous (higher-T) fit at
    # the shared boundary; the offset is recalculated during every fit trial.
    # Fit 1 (highest T) may free y0 when fit_y0 is True.
    windows = sorted(windows, key=lambda w: w[1], reverse=True)
    anchor = None
    for i, (i0, i1) in enumerate(windows):
        # free y0 only on the first (highest-T) segment
        fit = fit_window(i0, i1, anchor=anchor, free_y0=(fit_y0 and i == 0))
        if fit is None:
            raise ValueError(f"Could not fit segment {i + 1}; try changing the split temperatures or using fewer segments.")
        all_pred[i0:i1] = fit["pred"]
        Delta0, eta, dCel, Tm, k = map(float, fit["popt_arr"][:5])
        y0 = float(fit["y0"])
        # For fits after the highest-temperature segment, report the zero-
        # current temperature of that segment's shifted AB branch. This is
        # the root of y0 + AB(T) = 0 on the AB branch T in [Tm, Tc].
        # It may not exist in that interval; in that case report None.
        Tc_expected = None
        if i > 0:
            def _ab_zero_function(temp):
                delta_temp = bcs_gap_dwave(
                    np.array([temp]), Delta0 * 1e-3 * E_CHARGE, Tc, dCel)
                return float(y0 + _ab_core_mA(delta_temp, Tc, Rn_Ohm, eta)[0])

            t_left = float(np.clip(Tm, 0.0, Tc))
            f_left = _ab_zero_function(t_left)
            f_right = _ab_zero_function(float(Tc))
            if abs(f_right) < 1e-12:
                Tc_expected = float(Tc)
            elif f_left == 0.0:
                Tc_expected = t_left
            elif f_left * f_right < 0.0:
                Tc_expected = float(brentq(_ab_zero_function, t_left, float(Tc)))

        popt = {"Delta0_meV": Delta0, "eta": eta, "dCel": dCel,
                "Tm": Tm, "k": k, "y0": y0, "a0_mA": fit["a0_mA"], "Tc": float(Tc),
                "Tc_expected": Tc_expected,
                "model": "d-wave", "piecewise": True, "segment": i + 1,
                "splits": boundaries, "y0_fitted": bool(fit_y0 and i == 0)}
        results.append({"popt": popt, "pcov": fit["pcov"],
                        "free_names": list(fit["free_names"]),
                        "r_squared": fit["r_squared"], "T_lo": float(T[i0]),
                        "T_hi": float(T[i1 - 1]), "n_points": int(i1 - i0),
                        "T": fit["T"], "Ic": fit["Ic"], "param_err": fit["param_err"],
                        "split_err": []})
        # Anchor the next lower-T fit at its highest-T measured point, using
        # the previous (higher-T) model evaluated at exactly that temperature.
        if i + 1 < len(windows):
            next_i0, next_i1 = windows[i + 1]
            ta = float(T[next_i1 - 1])
            y_prev = float(ic_ab_dwave_piecewise(
                np.array([ta]), Delta0, Tc, Rn_Ohm, eta=eta, dCel=dCel,
                Tm=Tm, k=k, y0=y0)[0])
            anchor = (ta, y_prev)

    ss_res = float(np.nansum((Ic - all_pred) ** 2))
    ss_tot = float(np.sum((Ic - np.mean(Ic)) ** 2))
    r_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mode = "fixed" if fixed_splits else "optimized"
    print(f"  {mode} segment boundaries: " + ", ".join(f"{v:.4f} K" for v in boundaries)
          + f"   (combined R^2 = {r_global:.5f})")
    for result in results:
        result["popt"]["r_global"] = r_global
    return results


# ----------------------------------------------------------------------
# Plotting

# ----------------------------------------------------------------------
def make_plot(T, Ic, Ir, popt, Rn_Ohm, r_squared, free_names,
              normalize=False, Tc_norm=None, outpath="Ic_T_fit.pdf",
              no_fit=False, plot_icrn=False, fit_eta=False, model="ab",
              segment_results=None):
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

    if not no_fit and (popt is not None or segment_results):
        colors = ["b", "r", "g", "m", "c", "y"]

        def _draw_one(p, fr, r2, T_lo=None, T_hi=None, color="b", tag=""):
            Tc_f = p["Tc"]
            D0 = p["Delta0_meV"]
            et = p["eta"]
            dC = p.get("dCel")
            y0v = p.get("y0", 0.0)
            pw = p.get("piecewise", False)
            t_hi = min(float(T.max()) * 1.05, Tc_f * 0.999)
            t_lo = 1e-3 if T_lo is None else max(1e-3, T_lo)
            if T_hi is not None:
                t_hi = min(t_hi, T_hi)
            T_fine = np.linspace(t_lo, t_hi, 300)
            if model == "d-wave":
                if pw:
                    Ic_fine = ic_ab_dwave_piecewise(
                        T_fine, D0, Tc_f, Rn_Ohm,
                        eta=et, dCel=dC, Tm=p.get("Tm"), k=p.get("k"), y0=p.get("y0", 0.0))
                else:
                    Ic_fine = ic_ab_dwave(T_fine, D0, Tc_f, Rn_Ohm,
                                         eta=et, dCel=dC, y0=y0v)
            else:
                Ic_fine = ic_ab_classic(T_fine, D0, Tc_f, Rn_Ohm, eta=et)
            Ic_fine = np.maximum(Ic_fine, 0.0)
            if plot_icrn:
                y_fine = Ic_fine * (Rn_Ohm * 1e3) * 1e-3
            else:
                y_fine = Ic_fine
            x_fine = (T_fine / Tc_norm) if (normalize and Tc_norm) else T_fine
            lbl = f"fit{tag}" if tag else "fit"
            ax.plot(x_fine, y_fine, "-", color=color, marker="none", label=lbl)
            # For each independent segment, show its low-T linear branch as a
            # dashed extrapolation all the way to T=0, even outside its fit window.
            if segment_results is not None and pw and p.get("Tm") is not None:
                Tm_line = float(p["Tm"])
                k_line = float(p.get("k", 0.0))
                a_line = float(p.get("a0_mA", np.nan))
                if not np.isfinite(a_line):
                    dTm = bcs_gap_dwave(np.array([Tm_line]), D0 * 1e-3 * E_CHARGE, Tc_f, dC)
                    abTm = float(_ab_core_mA(dTm, Tc_f, Rn_Ohm, et)[0])
                    a_line = abTm - k_line * Tm_line
                T_linear = np.linspace(0.0, Tm_line, 120)
                y_linear = a_line + k_line * T_linear
                if plot_icrn:
                    y_linear = y_linear * (Rn_Ohm * 1e3) * 1e-3
                x_linear = (T_linear / Tc_norm) if (normalize and Tc_norm) else T_linear
                ax.plot(x_linear, y_linear, "--", color=color, marker="none",
                        linewidth=1.0, label="_nolegend_")
            return D0, Tc_f, et, dC, pw, r2, fr

        ax.set_ylabel(y_label)
        label_lines = []

        if segment_results:
            for i, seg in enumerate(segment_results):
                col = colors[i % len(colors)]
                D0, Tc_f, et, dC, pw, r2, fr = _draw_one(
                    seg["popt"], seg["free_names"], seg["r_squared"],
                    T_lo=seg["T_lo"], T_hi=seg["T_hi"],
                    color=col, tag=f" {i + 1}")
                # Multiple-fit parameters are reported in the terminal only;
                # keep the plot clean and reserve the legend for curve types.
        else:
            D0, Tc_f, et, dC, pw, r2, fr = _draw_one(
                popt, free_names, r_squared, color="b")
            label_lines = [
                rf"$\Delta_0$ = {D0:.3f} meV",
                rf"$T_c$ = {Tc_f:.2f} K",
            ]
            if fit_eta or ("eta" in free_names):
                label_lines.append(rf"$\eta$ = {et:.3f}")
            if model == "d-wave" and dC is not None:
                label_lines.append(
                    rf"$\Delta C_{{\rm el}}/\gamma T_c$ = {dC:.3f}")
                if pw:
                    Tm = popt.get("Tm")
                    k = popt.get("k")
                    if Tm is not None:
                        label_lines.append(rf"$T_m$ = {Tm:.2f} K")
                    if k is not None:
                        label_lines.append(
                            rf"$k$ = {k * 1e3:.4g} $\mu$A/K")
            if "y0" in free_names:
                label_lines.append(rf"$y_0$ = {popt.get('y0', 0):.4f} mA")
            if r2 is not None:
                label_lines.append(rf"$R^2$ = {r2:.4f}")

        if not segment_results:
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
def _main():
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
    (automatic)        Taken from each analysis.log's Ic_definition flag
                       written by plot_csv_data.py (jump_dvdi or rho_eff).
                       No CLI switch; edit the CSV Ic_active_mA column if
                       a manual override is needed.

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
    --Tm-initial K [K ...] Initial guess(es) for Tm (K); N values for --piecewise-multiples N
                       (default: mid-range of data temperatures).
    --fit-eta          Let the dimensionless prefactor eta vary; without
                       this flag eta is fixed at 1.  (Always free under
                       --piecewise.)
    --fit-y0           Also fit a constant offset y0 >= 0 (mA):
                       Eq. (7a)  — free y0 on the whole curve.
                       --piecewise — free y0 on the AB branch.
                       --piecewise-multiples — free y0 only on fit 1
                       (highest-T segment); lower segments keep the
                       continuity-derived offset.
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
    parser.add_argument("--no-fit", action="store_true",
                        help="Plot data only; do not perform a fit. "
                             "Default output: IcIr_T.pdf")
    parser.add_argument("--model", choices=("ab", "d-wave"),
                        default="ab",
                        help="Gap / AB model: ab = classic s-wave AB; "
                             "d-wave = paper d-wave form (default: ab)")
    parser.add_argument("--piecewise", action="store_true",
                        help="For d-wave only: paper Eq. (7) with one low-T "
                             "linear segment (Tm, k). Requires --Tc. Free: "
                             "Delta0, eta, dCel, Tm, k.")
    parser.add_argument("--piecewise-multiples", type=int, default=None,
                        metavar="N", dest="piecewise_multiples",
                        help="Fit N independent piecewise Eq. (7) models in consecutive "
                             "temperature windows (N>=2). Requires --Tc. "
                             "If --Ts-splits is supplied, those boundaries are fixed; "
                             "otherwise the boundaries are optimized from the full data.")
    parser.add_argument("--Tm-initial", type=float, nargs="+", default=None,
                        dest="Tm_initial", metavar="K",
                        help="Initial guess(es) for internal Tm (K). With "
                             "--piecewise-multiples N, provide N values in fit "
                             "order: highest-T segment first, then lower-T "
                             "segments. With --piecewise, provide one value.")
    parser.add_argument("--Ts-splits", "--Tm-splits", type=float, nargs="+", default=None,
                        dest="Tm_splits", metavar="K",
                        help="Optional fixed Ts temperatures separating the N consecutive "
                             "fitting windows (N-1 values). If omitted, split "
                             "temperatures are optimized automatically.")
    parser.add_argument("--fit-eta", action="store_true",
                        help="Treat eta as a free fit parameter "
                             "(default: eta = 1 fixed)")
    parser.add_argument("--fit-y0", action="store_true",
                        help="Fit constant offset y0 >= 0 (mA).  Eq. (7a): "
                             "whole curve.  --piecewise: AB branch.  "
                             "--piecewise-multiples: only fit 1 (highest-T "
                             "segment); lower segments keep continuity y0.")
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
    parser.add_argument("--out", default=None,
                        help="Output plot filename (default: Ic_T_fit.pdf, or IcIr_T.pdf with --no-fit)")
    args = parser.parse_args()
    print("--- Ic_T_fit run input parameters ---")
    for _key, _value in vars(args).items():
        print(f"{_key} = {_value!r}")
    print("--- Fit output ---")

    if args.normalize and args.Tc is None:
        parser.error("--normalize requires --Tc <value>")
    if args.piecewise and args.piecewise_multiples is not None:
        parser.error("Use only one of --piecewise / --piecewise-multiples")
    any_pw = args.piecewise or (args.piecewise_multiples is not None)
    if any_pw and args.model != "d-wave":
        parser.error("--piecewise / --piecewise-multiples only apply with "
                     "--model d-wave")
    if any_pw and args.Tc is None:
        parser.error("Piecewise modes require --Tc <value> (Tc held fixed)")
    if args.piecewise_multiples is not None and args.piecewise_multiples < 2:
        parser.error("--piecewise-multiples requires an integer >= 2")
    if args.Tm_splits is not None and args.piecewise_multiples is None:
        parser.error("--Ts-splits only applies with --piecewise-multiples")
    if args.Tm_initial is not None:
        expected_tm_guesses = args.piecewise_multiples if args.piecewise_multiples is not None else 1
        if len(args.Tm_initial) != expected_tm_guesses:
            parser.error(f"--Tm-initial expects {expected_tm_guesses} value(s) for the selected fitting mode; got {len(args.Tm_initial)}")
    if args.out is None:
        args.out = "IcIr_T.pdf" if args.no_fit else "Ic_T_fit.pdf"
    # ------------------------------------------------------------------
    # 1. Collect data  (Ic/Ir chosen per log from Ic_definition)
    # ------------------------------------------------------------------
    if args.from_csv:
        print(f"Loading data from CSV: {args.from_csv}")
        records = load_csv(args.from_csv)
        n_jump = sum(1 for r in records if r.get("Ic_definition") == "jump_dvdi")
        n_eff = len(records) - n_jump
        print(f"  Ic_definition:  jump_dvdi = {n_jump}  |  rho_eff = {n_eff}")
    else:
        print(f"Scanning '{os.path.abspath(os.path.expanduser(args.dir))}' "
              f"for analysis.log files (Ic from per-log Ic_definition)...")
        records = collect_data(args.dir)

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
    # 3. Fit (unless --no-fit)
    # ------------------------------------------------------------------
    popt = None
    pcov = None
    free_names = []
    r_squared = None
    err_map = {}
    segment_results = None

    if not args.no_fit:
        if len(T) < 3:
            parser.error(f"Only {len(T)} point(s); need at least 3 for a fit.")

        if args.piecewise_multiples is not None:
            splits = list(args.Tm_splits) if args.Tm_splits else None
            print(f"Piecewise-multiples N={args.piecewise_multiples}"
                  + (f", fixed split temperatures={sorted(splits)}" if splits
                     else " (optimizing split temperatures automatically)"))
            segment_results = fit_piecewise_multiples(
                T, Ic, Rn_Ohm,
                n_segments=args.piecewise_multiples,
                Tc=args.Tc,
                splits=splits if splits else None,
                Tm_initial=args.Tm_initial if args.Tm_initial is not None else None,
                fit_y0=args.fit_y0,
            )
            popt = segment_results[0]["popt"]
            free_names = segment_results[0]["free_names"]
            r_squared = segment_results[0]["r_squared"]
            err_map = {}
        else:
            popt, pcov, free_names, r_squared = fit_ic_vs_T(
                T, Ic, Rn_Ohm,
                model=args.model,
                Tc_guess=args.Tc,
                fit_eta=args.fit_eta,
                fit_y0=args.fit_y0,
                piecewise=args.piecewise,
                Tm_initial=args.Tm_initial[0] if args.Tm_initial is not None else None,
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
        segment_results=segment_results,
    )

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print("\n--- Ambegaokar-Baratoff fit results ---")
    model_label = args.model
    if args.model == "d-wave":
        if args.piecewise_multiples is not None:
            model_label += f" piecewise-multiples N={args.piecewise_multiples}"
        elif args.piecewise:
            model_label += " piecewise Eq.(7)"
        else:
            model_label += " Eq.(7a)"
    print(f"Model                       = {model_label}")
    if Rn_from_T is not None:
        print(f"R_n (fixed)                 = {Rn_mOhm:.3f} mOhm "
              f"(from T = {Rn_from_T:.4g} K)")
    else:
        print(f"R_n (fixed)                 = {Rn_mOhm:.3f} mOhm")
    if args.no_fit:
        print("(no fit performed)")
    elif segment_results is not None:
        print(f"Tc (fixed, input)           = {args.Tc:.4f} K")
        splits = segment_results[0]["popt"].get("splits", [])
        split_err = segment_results[0].get("split_err", [])
        if splits:
            print("Fixed segment boundaries    = "
                  + ", ".join(f"{v:.4f}" for v in splits) + " K")
        r_g = segment_results[0]["popt"].get("r_global")
        if r_g is not None:
            print(f"R^2 (global)                = {r_g:.5f}")
        for seg in segment_results:
            i = seg["popt"]["segment"]
            p = seg["popt"]
            fn = seg["free_names"]
            perr = np.sqrt(np.diag(seg["pcov"]))
            em = dict(zip(fn, perr))
            print(f"\n  --- fit {i}  "
                  f"[{seg['T_lo']:.3g}, {seg['T_hi']:.3g}] K  "
                  f"({seg['n_points']} pts) ---")
            print(f"  Delta0                    = {p['Delta0_meV']:.4f}"
                  + (f" +/- {em['Delta0_meV']:.4f}"
                     if "Delta0_meV" in em else "")
                  + " meV")
            print(f"  eta                       = {p['eta']:.4f}"
                  + (f" +/- {em['eta']:.4f}" if "eta" in em else ""))
            if p.get("dCel") is not None:
                print(f"  DeltaC_el/(gamma Tc)      = {p['dCel']:.4f}"
                      + (f" +/- {em['dCel']:.4f}" if "dCel" in em else ""))
                kB_meV = K_BOLTZ / E_CHARGE * 1e3
                ratio = 2.0 * p["Delta0_meV"] / (kB_meV * p["Tc"])
                print(f"  2 Delta0 / (kB Tc)        = {ratio:.3f}")
            if p.get("Tm") is not None:
                print(f"  Tm                        = {p['Tm']:.4f}"
                      + (f" +/- {em['Tm']:.4f}" if "Tm" in em else "")
                      + " K")
            if i > 1:
                if p.get("Tc_expected") is None:
                    print("  Tc_expected               = no zero crossing on AB branch")
                else:
                    print(f"  Tc_expected               = {p['Tc_expected']:.4f} K")
            if p.get("k") is not None:
                k_uA = p["k"] * 1e3
                k_err = em["k"] * 1e3 if "k" in em else None
                print(f"  k                         = {k_uA:.6g}"
                      + (f" +/- {k_err:.6g}" if k_err is not None else "")
                      + " uA/K")
                print(f"  linear intercept I(0)     = {p['a0_mA']:.6g} mA")
            if p.get("y0") is not None and (p.get("y0_fitted") or abs(p["y0"]) > 1e-12):
                y0_tag = "fit" if p.get("y0_fitted") else "continuity"
                print(f"  y0 ({y0_tag})                 = {p['y0']:.4f}"
                      + (f" +/- {em['y0']:.4f}" if "y0" in em else "")
                      + " mA")
            Ic0 = ic0_from_model(p, Rn_Ohm)
            print(f"  Ic(T=0)                   = {Ic0:.4f} mA")
            print(f"  R^2                       = {seg['r_squared']:.5f}")
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
                k_uA = k * 1e3
                k_err = (err_map['k'] * 1e3) if "k" in err_map else None
                print(f"k (fit)                     = {k_uA:.6g}"
                      + (f" +/- {k_err:.6g}" if k_err is not None else "")
                      + " uA/K")
        if args.fit_y0 or (popt.get("y0") is not None and abs(popt.get("y0", 0.0)) > 1e-12):
            print(f"y0 (fit)                    = {popt['y0']:.4f}"
                  + (f" +/- {err_map['y0']:.4f}" if "y0" in err_map else "")
                  + " mA")
        Ic0 = ic0_from_model(popt, Rn_Ohm)
        print(f"Ic(T=0)                     = {Ic0:.4f} mA")
        print(f"R^2                         = {r_squared:.5f}")

    print(f"\nPlot saved to: {os.path.abspath(args.out)}")


class _TeeOutput:
    """Write console output simultaneously to the terminal and the run log."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    # Overwrite the log on every invocation and mirror terminal output into it.
    original_stdout = sys.stdout
    try:
        with open("Ic_T_fit.log", "w", encoding="utf-8") as log_file:
            sys.stdout = _TeeOutput(original_stdout, log_file)
            _main()
    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
