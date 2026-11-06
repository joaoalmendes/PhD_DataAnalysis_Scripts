#!/usr/bin/env python3
"""
Fraunhofer_analysis.py
======================

Collects single-field Josephson-junction analysis results produced by the
IV/dV-dI pipeline (one `analysis.log` per field value, stored as

    singles/
        H-2.5Oe_Hbwd/analysis.log
        H+5.0Oe_Hfwd/analysis.log
        ...

), builds Ic(H) (and optional bi-directional average), and fits conventional
or second-harmonic Fraunhofer patterns.

Supported models
----------------
  1H   Conventional Fraunhofer (first harmonic only)::

           Ic(H) = I_bg + Ic0 * |sin(π (H − H_off) / ΔH₀)
                                / (π (H − H_off) / ΔH₀)|

       Free parameters: Ic0, ΔH₀ (mT), H_off (mT), I_bg.
       H_off is the field location of the central-lobe maximum; its
       initial guess is set by ``--H=0-offset``.

  2H   Fraunhofer with co-tunneling of Cooper pairs (first + second
       harmonic)::

           Ic(H) = I_bg + Ic1 * |sinc(π (H−H_off)/ΔH₀)|
                        + Ic2 * |sinc(2π (H−H_off)/ΔH₀)|

       Free parameters: Ic1, Ic2, ΔH₀ (mT), H_off (mT), I_bg.
       Reports the ratio Ic1/Ic2 in addition to the individual amplitudes
       and d_eff (when --width is given).

Units
-----
  - Critical current is always plotted and fitted in **µA**.
  - Magnetic field is always plotted and fitted in **mT**.
  - CLI options that take a field value (--H=0-offset, --Delta0-initial,
    --H-plot-range) are therefore expressed in **mT**.  Raw logs still
    store H in Oe; conversion is applied once at load time (1 Oe = 0.1 mT).

Data selection
--------------
  Each analysis.log writes an "Ic definition for this run" block set by
  plot_csv_data.py (--ic-dvdi-range):

      Ic_definition: jump_dvdi  →  use Ic+_f_mA  (dV/dI jump)
      Ic_definition: rho_eff    →  use Ic_eff+_f_mA  (rho criterion)

  This script reads that flag per field point and selects the matching
  Ic automatically.  Legacy logs without the flag default to rho_eff
  when available.

Branch selection
----------------
  - Modelling uses the forward sweep by default (or the averaged set when
    --BF --average is used).  Override with ``--model-use-data F|B``.
  - Plotting follows the same primary branch; the opposite sweep is
    overlaid only when both --BF and --plot-BF are True.

CSV workflow
------------
  After collecting data from logs the script always writes a CSV
  (default: Fraunhofer_data.csv) with columns
      H_Oe, H_mT, H_branch, Hsample_mean_Oe, Ic_mA, Ic_uA, Ic_eff_mA,
      Ic_active_uA, Ic_definition, Rn_mean_mOhm, is_bf, source
  so you can edit values by hand.  Subsequent runs can skip the folder
  scan entirely with --from-csv PATH.

Usage examples
--------------
    python Fraunhofer_analysis.py
    python Fraunhofer_analysis.py --dir /path/to/main_folder
    python Fraunhofer_analysis.py --from-csv Fraunhofer_data.csv --model 1H
    python Fraunhofer_analysis.py --model 1H --width 1.2 --normalize
    python Fraunhofer_analysis.py --model 2H --H-plot-range -5 5 --width 0.8
    python Fraunhofer_analysis.py --BF --average --H=0-offset 0.15 -0.08
    python Fraunhofer_analysis.py --model 1H --H=0-offset 0.12 --Delta0-initial 2.5
    python Fraunhofer_analysis.py --BF --plot-BF --model 1H --plot-x-Phi

Output
------
    - Fraunhofer_fit.pdf   (or --out path)
    - CSV of collected points
    - analysis.log         (rewritten each run with fit parameters & metrics)
    - summary on stdout
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr

# ----------------------------------------------------------------------
# Physical constants (SI)
# ----------------------------------------------------------------------
PHI0 = 2.067833848e-15   # Wb = T·m²  (magnetic flux quantum)
OE_TO_MT = 0.1           # 1 Oe = 0.1 mT
MA_TO_UA = 1e3           # 1 mA = 1000 µA


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
        "lines.linewidth": 0.9,
        "lines.markersize": 2.5,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# ----------------------------------------------------------------------
# Parsing of analysis.log files
# ----------------------------------------------------------------------
def _grab_float(pattern, text):
    """Return the first float matched by `pattern` in `text`, or None."""
    m = re.search(pattern, text, re.MULTILINE)
    return float(m.group(1)) if m else None


def _grab_str(pattern, text):
    """Return the first string matched by `pattern` in `text`, or None."""
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _parse_ic_definition(text):
    """Return 'jump_dvdi' or 'rho_eff' from an analysis.log body."""
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
    return "rho_eff"


def parse_log(path):
    """Extract field, branch, Ic and related quantities from one log.

    Active Ic is chosen from the log's ``Ic_definition`` flag:
        jump_dvdi  →  Ic+_f_mA
        rho_eff    →  Ic_eff+_f_mA

    Returns H in both Oe (raw) and mT, and Ic in both mA and µA.
    """
    with open(path, "r") as f:
        text = f.read()

    H_label = _grab_str(r"^H_label\s*:\s*(.+)$", text)

    H_Oe = _grab_float(r"^H_nominal\s*:\s*([\-0-9.eE]+)", text)
    if H_Oe is None:
        H_Oe = _grab_float(r"^H_used\s*:\s*([\-0-9.eE]+)", text)

    H_branch = _grab_str(r"^H_branch\s*:\s*(\w+)", text)
    if H_branch is not None:
        H_branch = H_branch.lower()

    # Fallback: parse from folder / label name  H[+|-]<val>Oe_H(fwd|bwd)
    if H_Oe is None or H_branch is None:
        name = H_label or os.path.basename(os.path.dirname(path))
        m = re.search(
            r"H([+\-]?[0-9]*\.?[0-9]+)\s*Oe[_\s]*H?(fwd|bwd)",
            name, re.IGNORECASE,
        )
        if m:
            if H_Oe is None:
                H_Oe = float(m.group(1))
            if H_branch is None:
                H_branch = m.group(2).lower()

    Hsample_mean = _grab_float(r"^Hsample_mean\s*:\s*([\-0-9.eE]+)", text)
    Hsample_std = _grab_float(r"^Hsample_std\s*:\s*([\-0-9.eE]+)", text)
    Hsample_sem = _grab_float(r"^Hsample_sem\s*:\s*([\-0-9.eE]+)", text)
    # Hsample_min/max : 4.7260 / 9.6410 Oe
    m_mm = re.search(
        r"^Hsample_min/max\s*:\s*([\-0-9.eE]+)\s*/\s*([\-0-9.eE]+)",
        text, re.MULTILINE,
    )
    Hsample_min = float(m_mm.group(1)) if m_mm else None
    Hsample_max = float(m_mm.group(2)) if m_mm else None
    n_Hsample = _grab_float(r"^n_Hsample_pts\s*:\s*([\-0-9.eE]+)", text)
    if n_Hsample is not None:
        n_Hsample = int(n_Hsample)

    is_bf_str = _grab_str(r"^is_bf\s*:\s*(\w+)", text)
    is_bf = (is_bf_str or "").lower() in ("true", "1", "yes")

    Ic_raw = _grab_float(r"^  Ic\+_f_mA\s*=\s*([\-0-9.eE]+)", text)
    Ic_eff = _grab_float(r"^  Ic_eff\+_f_mA\s*=\s*([\-0-9.eE]+)", text)
    Ir_raw = _grab_float(r"^  Ic\+_b_mA\s*=\s*([\-0-9.eE]+)", text)
    Ir_eff = _grab_float(r"^  Ir_eff\+_b_mA\s*=\s*([\-0-9.eE]+)", text)

    ic_def = _parse_ic_definition(text)

    if ic_def == "jump_dvdi":
        Ic_active_mA = Ic_raw if Ic_raw is not None else Ic_eff
    else:
        Ic_active_mA = Ic_eff if Ic_eff is not None else Ic_raw

    Rn_mean_mOhm = _grab_float(r"^  Rn_mean_mOhm\s*=\s*([\-0-9.eE]+)", text)

    H_mT = H_Oe * OE_TO_MT if H_Oe is not None else None
    Ic_active_uA = (Ic_active_mA * MA_TO_UA) if Ic_active_mA is not None else None

    # Working field uncertainty in mT: prefer SEM (uncertainty on the mean)
    H_err_mT = None
    if Hsample_sem is not None and np.isfinite(Hsample_sem):
        H_err_mT = float(Hsample_sem) * OE_TO_MT
    elif Hsample_std is not None and np.isfinite(Hsample_std):
        H_err_mT = float(Hsample_std) * OE_TO_MT

    return {
        "H_Oe": H_Oe,
        "H_mT": H_mT,
        "H_branch": H_branch,
        "Hsample_mean_Oe": Hsample_mean,
        "Hsample_std_Oe": Hsample_std,
        "Hsample_sem_Oe": Hsample_sem,
        "Hsample_min_Oe": Hsample_min,
        "Hsample_max_Oe": Hsample_max,
        "n_Hsample_pts": n_Hsample,
        "H_err_mT": H_err_mT,
        "Ic_mA": Ic_raw,
        "Ic_eff_mA": Ic_eff,
        "Ic_active_mA": Ic_active_mA,
        "Ic_active_uA": Ic_active_uA,
        "Ir_mA": Ir_raw,
        "Ir_eff_mA": Ir_eff,
        "Ic_definition": ic_def,
        "Rn_mean_mOhm": Rn_mean_mOhm,
        "is_bf": is_bf,
        "H_label": H_label,
        "path": path,
    }


def collect_data(base_dir):
    """Find every analysis.log under field subfolders.

    Accepts either:
      base_dir/singles/<Hlabel>/analysis.log
      base_dir/<Hlabel>/analysis.log
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
    n_jump = n_eff = n_dvdi = 0
    for lf in log_files:
        rec = parse_log(lf)
        missing = [k for k in ("H_mT", "Ic_active_uA") if rec[k] is None]
        if missing:
            print(f"  [skip] {lf}: could not parse {missing}", file=sys.stderr)
            continue
        if rec.get("Ic_definition") == "jump_dvdi":
            n_jump += 1
        else:
            n_eff += 1

        # Optional dV/dI trace from the same measurement folder
        folder = os.path.dirname(lf)
        I_uA, dVdI = load_dvdi_dat(folder)
        rec["I_array_uA"] = I_uA
        rec["dVdI_array_Ohm"] = dVdI
        if I_uA is not None:
            n_dvdi += 1

        records.append(rec)

    if not records:
        raise RuntimeError("No usable analysis.log files were parsed.")

    print(f"  Ic_definition:  jump_dvdi = {n_jump}  |  rho_eff = {n_eff}")
    print(f"  dVdI.dat loaded for {n_dvdi} / {len(records)} point(s)")
    records.sort(key=lambda r: (r["H_mT"], r.get("H_branch") or ""))
    return records


# ----------------------------------------------------------------------
# CSV I/O
# ----------------------------------------------------------------------
CSV_FIELDS = [
    "H_Oe", "H_mT", "H_branch",
    "Hsample_mean_Oe", "Hsample_std_Oe", "Hsample_sem_Oe",
    "Hsample_min_Oe", "Hsample_max_Oe", "n_Hsample_pts",
    "H_err_mT",
    "Ic_mA", "Ic_eff_mA", "Ic_active_mA", "Ic_active_uA",
    "Ir_mA", "Ir_eff_mA",
    "Ic_definition", "Rn_mean_mOhm", "is_bf", "source",
    "I_array_uA", "dVdI_array_Ohm",
]


def _array_to_str(arr):
    """Serialize a 1-D array to a semicolon-separated string for CSV storage.

    Semicolons avoid ambiguity with CSV commas and with decimal points.
    """
    if arr is None:
        return ""
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return ""
    return ";".join(f"{x:.8g}" for x in a)


def _str_to_array(s):
    """Deserialize an array cell from CSV into a 1-D float ndarray.

    Accepts semicolon-, space-, tab-, or comma-separated values, optionally
    wrapped in brackets or quotes (as produced by some spreadsheet tools).
    """
    if s is None:
        return None
    if isinstance(s, (list, tuple, np.ndarray)):
        a = np.asarray(s, dtype=float).ravel()
        return a if a.size else None
    s = str(s).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return None
    # Strip optional brackets / quotes (possibly nested once)
    for _ in range(2):
        if len(s) >= 2 and (s[0], s[-1]) in {
            ("[", "]"), ("(", ")"), ("{", "}"), ('"', '"'), ("'", "'"),
        }:
            s = s[1:-1].strip()
    if not s:
        return None
    # Prefer ';' (our save format); fall back to other separators
    if ";" in s:
        parts = s.split(";")
    elif "," in s and " " not in s.strip():
        # comma-separated with no spaces (ambiguous with decimal commas —
        # only use when there is no whitespace)
        parts = s.split(",")
    else:
        for sep in (",", "\t", "|"):
            s = s.replace(sep, " ")
        parts = s.split()
    vals = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            # skip non-numeric tokens rather than failing the whole cell
            continue
    if len(vals) < 2:
        return None
    a = np.asarray(vals, dtype=float)
    if not np.any(np.isfinite(a)):
        return None
    return a


def load_dvdi_dat(folder):
    """Load ``dVdI.dat`` from a single-measurement folder.

    Expected format (as written by ``plot_csv_data.py``)::

        # Current_A    dVdI_Ohm
        -1.0e-04       12.3
        ...

    Filename match is case-insensitive.  Returns (I_uA, dVdI_Ohm) or
    (None, None) if the file is missing/invalid.
    """
    path = None
    try:
        for name in os.listdir(folder):
            if name.lower() == "dvdi.dat":
                path = os.path.join(folder, name)
                break
    except OSError:
        path = None
    if path is None:
        path = os.path.join(folder, "dVdI.dat")
    if not os.path.isfile(path):
        return None, None
    try:
        # Default comment char '#' skips the header line written by the pipeline
        data = np.loadtxt(path, ndmin=2)
    except Exception:
        try:
            data = np.genfromtxt(
                path, dtype=float, ndmin=2, invalid_raise=False,
                comments="#",
            )
        except Exception as exc:
            print(f"  [warn] could not read {path}: {exc}", file=sys.stderr)
            return None, None
    if data is None or data.size == 0:
        return None, None
    if data.ndim == 1:
        if data.size < 2:
            return None, None
        data = data.reshape(-1, 2)
    if data.shape[1] < 2:
        return None, None
    # Keep rows with finite current; allow nan dV/dI only if enough finite remain
    mask = np.isfinite(data[:, 0])
    data = data[mask]
    if data.shape[0] < 2:
        return None, None
    I_A = np.asarray(data[:, 0], dtype=float)
    dVdI = np.asarray(data[:, 1], dtype=float)
    if np.count_nonzero(np.isfinite(dVdI)) < 2:
        return None, None
    return I_A * 1e6, dVdI  # A → µA


def _folder_label_key(name):
    """Normalise a singles/ folder name for matching (lower, no spaces)."""
    return re.sub(r"\s+", "", name.strip().lower())


def attach_dvdi_from_dir(records, base_dir):
    """Fill missing I_array_uA / dVdI_array_Ohm by scanning *base_dir*.

    Looks for ``dVdI.dat`` under::

        base_dir/singles/*/dVdI.dat
        base_dir/*/dVdI.dat

    and matches each file to a record by
    (1) ``source`` path directory, (2) folder name ↔ H_label / H value.
    """
    if not base_dir:
        return records, 0
    base_dir = os.path.abspath(os.path.expanduser(base_dir))
    patterns = [
        os.path.join(base_dir, "singles", "*", "dVdI.dat"),
        os.path.join(base_dir, "singles", "*", "dvdi.dat"),
        os.path.join(base_dir, "*", "dVdI.dat"),
        os.path.join(base_dir, "*", "dvdi.dat"),
    ]
    dat_files = []
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            ap = os.path.abspath(p)
            if ap not in seen and os.path.isfile(ap):
                seen.add(ap)
                dat_files.append(ap)

    if not dat_files:
        # case-insensitive walk one level
        for sub in ("singles", ""):
            root = os.path.join(base_dir, sub) if sub else base_dir
            if not os.path.isdir(root):
                continue
            try:
                for entry in os.listdir(root):
                    folder = os.path.join(root, entry)
                    if not os.path.isdir(folder):
                        continue
                    for name in os.listdir(folder):
                        if name.lower() == "dvdi.dat":
                            ap = os.path.abspath(os.path.join(folder, name))
                            if ap not in seen:
                                seen.add(ap)
                                dat_files.append(ap)
            except OSError:
                pass

    if not dat_files:
        print(
            f"  [dvdi] no dVdI.dat found under {base_dir}",
            file=sys.stderr,
        )
        return records, 0

    # Index dat files by folder basename
    by_label = {}
    by_folder = {}
    for dp in dat_files:
        folder = os.path.dirname(dp)
        label = _folder_label_key(os.path.basename(folder))
        by_label[label] = folder
        by_folder[os.path.abspath(folder)] = folder

    n_filled = 0
    for r in records:
        if (
            r.get("dVdI_array_Ohm") is not None
            and isinstance(r.get("dVdI_array_Ohm"), np.ndarray)
            and r["dVdI_array_Ohm"].size >= 2
            and r.get("I_array_uA") is not None
            and isinstance(r.get("I_array_uA"), np.ndarray)
            and r["I_array_uA"].size >= 2
        ):
            continue

        folder = None
        src = r.get("path") or ""
        if src:
            src_abs = os.path.abspath(os.path.expanduser(src))
            if os.path.isfile(src_abs):
                cand = os.path.dirname(src_abs)
                if os.path.abspath(cand) in by_folder:
                    folder = by_folder[os.path.abspath(cand)]
                elif os.path.isdir(cand):
                    folder = cand
            elif os.path.isdir(src_abs):
                folder = src_abs

        if folder is None:
            # Match by H_label
            lab = r.get("H_label") or ""
            key = _folder_label_key(lab) if lab else ""
            if key and key in by_label:
                folder = by_label[key]
            else:
                # Match by field value encoded in folder name, e.g. H+5.0Oe_Hbwd
                H = r.get("H_Oe")
                br = (r.get("H_branch") or "").lower()
                if H is not None:
                    for lab_key, fold in by_label.items():
                        m = re.search(
                            r"h([+\-]?[0-9]*\.?[0-9]+)\s*oe",
                            lab_key,
                        )
                        if not m:
                            continue
                        try:
                            H_folder = float(m.group(1))
                        except ValueError:
                            continue
                        if abs(H_folder - float(H)) > 1e-6:
                            continue
                        if br and br not in lab_key:
                            # allow match if branch not in name
                            if "fwd" in lab_key or "bwd" in lab_key:
                                continue
                        folder = fold
                        break

        if folder is None:
            continue
        I_uA, dVdI = load_dvdi_dat(folder)
        if dVdI is None:
            continue
        r["I_array_uA"] = I_uA
        r["dVdI_array_Ohm"] = dVdI
        if not r.get("path"):
            r["path"] = os.path.join(folder, "analysis.log")
        n_filled += 1

    print(
        f"  [dvdi] found {len(dat_files)} dVdI.dat file(s) under {base_dir}; "
        f"attached to {n_filled} / {len(records)} record(s)"
    )
    return records, n_filled


def save_csv(path, records):
    """Write collected records to a CSV for manual editing / reuse."""
    def _cell(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(v)
        return v

    with open(path, "w", newline="") as f:
        # QUOTE_MINIMAL still quotes fields that contain the delimiter;
        # array cells use ';' so they stay in a single column even if a
        # spreadsheet re-saves the file with different quoting.
        w = csv.DictWriter(
            f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for r in records:
            w.writerow({
                "H_Oe": _cell(r.get("H_Oe")),
                "H_mT": _cell(r.get("H_mT")),
                "H_branch": _cell(r.get("H_branch")),
                "Hsample_mean_Oe": _cell(r.get("Hsample_mean_Oe")),
                "Hsample_std_Oe": _cell(r.get("Hsample_std_Oe")),
                "Hsample_sem_Oe": _cell(r.get("Hsample_sem_Oe")),
                "Hsample_min_Oe": _cell(r.get("Hsample_min_Oe")),
                "Hsample_max_Oe": _cell(r.get("Hsample_max_Oe")),
                "n_Hsample_pts": _cell(r.get("n_Hsample_pts")),
                "H_err_mT": _cell(r.get("H_err_mT")),
                "Ic_mA": _cell(r.get("Ic_mA")),
                "Ic_eff_mA": _cell(r.get("Ic_eff_mA")),
                "Ic_active_mA": _cell(r.get("Ic_active_mA")),
                "Ic_active_uA": _cell(r.get("Ic_active_uA")),
                "Ir_mA": _cell(r.get("Ir_mA")),
                "Ir_eff_mA": _cell(r.get("Ir_eff_mA")),
                "Ic_definition": _cell(r.get("Ic_definition", "")),
                "Rn_mean_mOhm": _cell(r.get("Rn_mean_mOhm")),
                "is_bf": _cell(r.get("is_bf", False)),
                "source": r.get("path", ""),
                "I_array_uA": _array_to_str(r.get("I_array_uA")),
                "dVdI_array_Ohm": _array_to_str(r.get("dVdI_array_Ohm")),
            })
    print(f"Data CSV written to: {os.path.abspath(path)}")


def load_csv(path):
    """Load records previously written by save_csv (or hand-edited)."""
    # Array cells can be long; raise the csv field-size limit just in case
    try:
        csv.field_size_limit(max(2**24, csv.field_size_limit()))
    except (OverflowError, AttributeError):
        pass

    records = []
    n_array_ok = n_array_fail = 0
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        # utf-8-sig strips a BOM if present (Excel on Windows)
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        # Normalise header names (strip whitespace / BOM leftovers)
        raw_fields = list(reader.fieldnames)
        norm_map = {h: h.strip() for h in raw_fields}
        # Detect array columns under mild name variants
        def _find_col(*candidates):
            lower = {h.strip().lower(): h for h in raw_fields}
            for c in candidates:
                if c.lower() in lower:
                    return lower[c.lower()]
            return None

        col_I = _find_col("I_array_uA", "I_array", "I_uA_array")
        col_Z = _find_col("dVdI_array_Ohm", "dVdI_array", "dvdi_array_ohm")
        print(
            f"  CSV dialect delimiter={dialect.delimiter!r}, "
            f"I_array col={col_I!r}, dVdI col={col_Z!r}, "
            f"n_fields={len(raw_fields)}"
        )

        for row in reader:
            # Re-key row with stripped header names
            row = {norm_map.get(k, k): v for k, v in row.items()}

            def _opt_float(key):
                s = (row.get(key) or "").strip()
                return float(s) if s else None

            H_Oe = _opt_float("H_Oe")
            H_mT = _opt_float("H_mT")
            if H_mT is None and H_Oe is not None:
                H_mT = H_Oe * OE_TO_MT
            if H_Oe is None and H_mT is not None:
                H_Oe = H_mT / OE_TO_MT
            if H_mT is None:
                print("  [skip] CSV row: missing H", file=sys.stderr)
                continue

            ic_def = (row.get("Ic_definition") or "").strip() or None
            Ic_raw = _opt_float("Ic_mA")
            Ic_eff = _opt_float("Ic_eff_mA")
            Ic_mA = _opt_float("Ic_active_mA")
            Ic_uA = _opt_float("Ic_active_uA")

            if Ic_mA is None and ic_def == "jump_dvdi":
                Ic_mA = Ic_raw if Ic_raw is not None else Ic_eff
            if Ic_mA is None:
                Ic_mA = Ic_eff if Ic_eff is not None else Ic_raw
            if Ic_uA is None and Ic_mA is not None:
                Ic_uA = Ic_mA * MA_TO_UA
            if Ic_uA is None:
                print("  [skip] CSV row: no Ic column", file=sys.stderr)
                continue

            branch = (row.get("H_branch") or "").strip().lower() or None
            is_bf_s = (row.get("is_bf") or "").strip().lower()
            is_bf = is_bf_s in ("true", "1", "yes")

            H_err = _opt_float("H_err_mT")
            if H_err is None:
                sem = _opt_float("Hsample_sem_Oe")
                std = _opt_float("Hsample_std_Oe")
                if sem is not None:
                    H_err = sem * OE_TO_MT
                elif std is not None:
                    H_err = std * OE_TO_MT

            records.append({
                "H_Oe": H_Oe,
                "H_mT": H_mT,
                "H_branch": branch,
                "Hsample_mean_Oe": _opt_float("Hsample_mean_Oe"),
                "Hsample_std_Oe": _opt_float("Hsample_std_Oe"),
                "Hsample_sem_Oe": _opt_float("Hsample_sem_Oe"),
                "Hsample_min_Oe": _opt_float("Hsample_min_Oe"),
                "Hsample_max_Oe": _opt_float("Hsample_max_Oe"),
                "n_Hsample_pts": (
                    int(v) if (v := _opt_float("n_Hsample_pts")) is not None
                    else None
                ),
                "H_err_mT": H_err,
                "Ic_mA": Ic_raw,
                "Ic_eff_mA": Ic_eff,
                "Ic_active_mA": Ic_mA,
                "Ic_active_uA": Ic_uA,
                "Ir_mA": _opt_float("Ir_mA"),
                "Ir_eff_mA": _opt_float("Ir_eff_mA"),
                "Ic_definition": ic_def or (
                    "rho_eff" if Ic_eff is not None else "jump_dvdi"
                ),
                "Rn_mean_mOhm": _opt_float("Rn_mean_mOhm"),
                "is_bf": is_bf,
                "path": row.get("source", path),
                "I_array_uA": _str_to_array(
                    row.get(col_I) if col_I else row.get("I_array_uA")
                ),
                "dVdI_array_Ohm": _str_to_array(
                    row.get(col_Z) if col_Z else row.get("dVdI_array_Ohm")
                ),
            })
            if records[-1]["dVdI_array_Ohm"] is not None:
                n_array_ok += 1
            else:
                # Cell present but failed to parse?
                raw_z = row.get(col_Z) if col_Z else row.get("dVdI_array_Ohm")
                if raw_z and str(raw_z).strip():
                    n_array_fail += 1
    if not records:
        raise RuntimeError(f"No usable rows in CSV: {path}")

    print(
        f"  CSV arrays: {n_array_ok}/{len(records)} rows with dVdI_array_Ohm"
        + (f"  ({n_array_fail} non-empty cells failed to parse)" if n_array_fail else "")
    )

    # Recover dV/dI from source folders when CSV cells are empty
    n_recovered = 0
    for r in records:
        if r.get("dVdI_array_Ohm") is not None and r.get("I_array_uA") is not None:
            continue
        src = r.get("path") or ""
        folder = None
        if src and os.path.isfile(src):
            folder = os.path.dirname(src)
        elif src and os.path.isdir(src):
            folder = src
        if not folder:
            continue
        I_uA, dVdI = load_dvdi_dat(folder)
        if dVdI is None:
            continue
        if r.get("I_array_uA") is None and I_uA is not None:
            r["I_array_uA"] = I_uA
        if r.get("dVdI_array_Ohm") is None:
            r["dVdI_array_Ohm"] = dVdI
        n_recovered += 1
    if n_recovered:
        print(
            f"  recovered dV/dI for {n_recovered} row(s) from source folders"
        )

    records.sort(key=lambda r: (r["H_mT"], r.get("H_branch") or ""))
    return records


# ----------------------------------------------------------------------
# Data preparation helpers
# ----------------------------------------------------------------------
def apply_Hsample_mean(records):
    """Replace nominal H with Hsample_mean_Oe for every record that has it.

    Updates both ``H_Oe`` and ``H_mT``.  Points lacking a finite
    Hsample_mean_Oe are left unchanged and reported.
    """
    out = []
    n_used = n_skip = 0
    for r in records:
        r = dict(r)
        h_mean = r.get("Hsample_mean_Oe")
        if h_mean is not None and np.isfinite(h_mean):
            r["H_Oe"] = float(h_mean)
            r["H_mT"] = float(h_mean) * OE_TO_MT
            n_used += 1
        else:
            n_skip += 1
        out.append(r)
    print(
        f"  --H-mean: using Hsample_mean for {n_used} point(s)"
        + (f"; {n_skip} kept nominal (no Hsample_mean)" if n_skip else "")
    )
    out.sort(key=lambda r: (r["H_mT"], r.get("H_branch") or ""))
    return out


def apply_branch_alignment(records, offsets_mT, bf=False):
    """Relative branch alignment for bi-directional averaging only.

    When *two* offset values are supplied together with ``bf=True``, each
    branch is shifted by its own value so that fwd and bwd can be averaged.
    A single offset is **not** applied here — it is used as the initial
    guess for the fitted H(=0) parameter instead (see ``fit_model``).
    """
    if offsets_mT is None:
        return records

    offs = (
        list(offsets_mT)
        if hasattr(offsets_mT, "__len__")
        else [float(offsets_mT)]
    )

    # Only pre-shift when two distinct branch offsets are given under --BF
    if not (bf and len(offs) == 2):
        return records

    print(
        f"  branch alignment: fwd −={offs[0]:.4g} mT, "
        f"bwd −={offs[1]:.4g} mT"
    )
    out = []
    for r in records:
        r = dict(r)
        branch = (r.get("H_branch") or "").lower()
        if branch == "fwd":
            delta = offs[0]
        elif branch == "bwd":
            delta = offs[1]
        else:
            delta = offs[0]
        r["H_mT"] = r["H_mT"] - delta
        if r.get("H_Oe") is not None:
            r["H_Oe"] = r["H_Oe"] - delta / OE_TO_MT
        out.append(r)
    return out


def _average_dvdi_pair(r_a, r_b):
    """Average two dV/dI traces onto a shared current grid.

    Returns (I_uA, dVdI_Ohm) or (None, None) if neither side has data.
    """
    def _get(r):
        I = r.get("I_array_uA") if r is not None else None
        Z = r.get("dVdI_array_Ohm") if r is not None else None
        if I is None or Z is None:
            return None, None
        I = np.asarray(I, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        n = min(I.size, Z.size)
        if n < 2:
            return None, None
        I, Z = I[:n], Z[:n]
        finite = np.isfinite(I) & np.isfinite(Z)
        if np.count_nonzero(finite) < 2:
            return None, None
        order = np.argsort(I[finite])
        return I[finite][order], Z[finite][order]

    I1, Z1 = _get(r_a)
    I2, Z2 = _get(r_b)
    if I1 is None and I2 is None:
        return None, None
    if I1 is None:
        return I2, Z2
    if I2 is None:
        return I1, Z1

    I_lo = max(float(I1.min()), float(I2.min()))
    I_hi = min(float(I1.max()), float(I2.max()))
    if not np.isfinite(I_lo) or not np.isfinite(I_hi) or I_hi <= I_lo:
        return I1, Z1
    n = max(I1.size, I2.size)
    I_grid = np.linspace(I_lo, I_hi, n)
    Z_avg = 0.5 * (
        np.interp(I_grid, I1, Z1) + np.interp(I_grid, I2, Z2)
    )
    return I_grid, Z_avg


def average_bf(records, atol=1e-5):
    """Average forward and backward sweeps over the overlapping H range (mT).

    Points that cannot be paired after relative offset correction are
    discarded.  Matching uses linear interpolation onto the denser grid.

    dV/dI traces (``I_array_uA``, ``dVdI_array_Ohm``) are also averaged
    pairwise so ``--plot-dvdi`` still works after ``--average``.
    """
    fwd = [r for r in records if (r.get("H_branch") or "") == "fwd"]
    bwd = [r for r in records if (r.get("H_branch") or "") == "bwd"]
    other = [
        r for r in records
        if (r.get("H_branch") or "") not in ("fwd", "bwd")
    ]

    if not fwd or not bwd:
        print(
            "  [warn] --average requested but one branch is missing; "
            "returning original data.",
            file=sys.stderr,
        )
        return records

    fwd = sorted(fwd, key=lambda r: r["H_mT"])
    bwd = sorted(bwd, key=lambda r: r["H_mT"])
    H_fwd = np.array([r["H_mT"] for r in fwd])
    H_bwd = np.array([r["H_mT"] for r in bwd])
    Ic_fwd = np.array([r["Ic_active_uA"] for r in fwd])
    Ic_bwd = np.array([r["Ic_active_uA"] for r in bwd])
    # Field uncertainties (mT); missing → nan so interp still works
    Err_fwd = np.array([
        r["H_err_mT"] if r.get("H_err_mT") is not None else np.nan
        for r in fwd
    ])
    Err_bwd = np.array([
        r["H_err_mT"] if r.get("H_err_mT") is not None else np.nan
        for r in bwd
    ])

    H_lo = max(H_fwd.min(), H_bwd.min())
    H_hi = min(H_fwd.max(), H_bwd.max())
    if H_lo >= H_hi:
        print(
            "  [warn] no overlapping H range after offset; cannot average.",
            file=sys.stderr,
        )
        return records

    mask_f = (H_fwd >= H_lo - atol) & (H_fwd <= H_hi + atol)
    mask_b = (H_bwd >= H_lo - atol) & (H_bwd <= H_hi + atol)
    Hf, If, Ef = H_fwd[mask_f], Ic_fwd[mask_f], Err_fwd[mask_f]
    Hb, Ib, Eb = H_bwd[mask_b], Ic_bwd[mask_b], Err_bwd[mask_b]
    fwd_m = [r for r, m in zip(fwd, mask_f) if m]
    bwd_m = [r for r, m in zip(bwd, mask_b) if m]

    if len(Hf) >= len(Hb):
        H_ref, Ic_ref, E_ref = Hf, If, Ef
        Ic_other = np.interp(H_ref, Hb, Ib)
        E_other = np.interp(H_ref, Hb, Eb)
    else:
        H_ref, Ic_ref, E_ref = Hb, Ib, Eb
        Ic_other = np.interp(H_ref, Hf, If)
        E_other = np.interp(H_ref, Hf, Ef)

    Ic_avg = 0.5 * (Ic_ref + Ic_other)
    # Combined field uncertainty: RSS of the two branch SEMs / √2
    E_avg = np.sqrt(0.5 * (E_ref ** 2 + E_other ** 2))

    averaged = []
    n_dvdi = 0
    for h, ic, e in zip(H_ref, Ic_avg, E_avg):
        e_val = float(e) if np.isfinite(e) else None
        # Nearest fwd / bwd records at this H for dV/dI averaging
        i_f = int(np.argmin(np.abs(Hf - h))) if len(Hf) else None
        i_b = int(np.argmin(np.abs(Hb - h))) if len(Hb) else None
        r_f = fwd_m[i_f] if i_f is not None else None
        r_b = bwd_m[i_b] if i_b is not None else None
        I_arr, Z_arr = _average_dvdi_pair(r_f, r_b)
        if Z_arr is not None:
            n_dvdi += 1
        averaged.append({
            "H_Oe": float(h) / OE_TO_MT,
            "H_mT": float(h),
            "H_branch": "avg",
            "Hsample_mean_Oe": None,
            "Hsample_std_Oe": None,
            "Hsample_sem_Oe": (
                e_val / OE_TO_MT if e_val is not None else None
            ),
            "Hsample_min_Oe": None,
            "Hsample_max_Oe": None,
            "n_Hsample_pts": None,
            "H_err_mT": e_val,
            "Ic_mA": None,
            "Ic_eff_mA": None,
            "Ic_active_mA": float(ic) / MA_TO_UA,
            "Ic_active_uA": float(ic),
            "Ir_mA": None,
            "Ir_eff_mA": None,
            "Ic_definition": "averaged",
            "Rn_mean_mOhm": None,
            "is_bf": True,
            "path": "averaged",
            "I_array_uA": I_arr,
            "dVdI_array_Ohm": Z_arr,
        })
    print(
        f"  averaged {len(averaged)} points over "
        f"H ∈ [{H_lo:.4g}, {H_hi:.4g}] mT"
        + (f"  (dV/dI kept for {n_dvdi})" if n_dvdi else
           "  (no dV/dI arrays to average)")
    )
    return averaged + other


def normalize_Ic(records, H0=0.0):
    """Divide every Ic_active_uA by the value at the point closest to H0 (mT)."""
    if not records:
        return records, None
    H = np.array([r["H_mT"] for r in records])
    Ic = np.array([r["Ic_active_uA"] for r in records])
    idx = int(np.argmin(np.abs(H - H0)))
    Ic0 = Ic[idx]
    if Ic0 == 0 or not np.isfinite(Ic0):
        print(
            "  [warn] Ic at H≈0 is zero/NaN; skipping normalisation.",
            file=sys.stderr,
        )
        return records, None
    out = []
    for r in records:
        r = dict(r)
        r["Ic_active_uA"] = r["Ic_active_uA"] / Ic0
        if r.get("Ic_active_mA") is not None:
            r["Ic_active_mA"] = r["Ic_active_mA"] / (Ic0 / MA_TO_UA)
        out.append(r)
    print(f"  normalised by Ic(H={H[idx]:.4g} mT) = {Ic0:.6g} µA")
    return out, float(Ic0)


def select_H_range(records, H_range_mT):
    """Keep only points with H_mT inside [lo, hi]."""
    if H_range_mT is None:
        return records
    lo, hi = float(H_range_mT[0]), float(H_range_mT[1])
    if lo > hi:
        lo, hi = hi, lo
    kept = [r for r in records if lo <= r["H_mT"] <= hi]
    print(
        f"  H-plot-range [{lo:g}, {hi:g}] mT → "
        f"{len(kept)} / {len(records)} points"
    )
    return kept


def select_branch_for_fit(records, which="F"):
    """Return the records used for modelling.

    Priority:
      1. averaged set  (branch == 'avg')  if present
      2. the branch selected by *which* ('F' → fwd, 'B' → bwd)
      3. all records if the requested branch is missing
    """
    which = (which or "F").upper()
    if which not in ("F", "B"):
        raise ValueError(f"--model-use-data must be 'F' or 'B', got {which!r}")

    avg = [r for r in records if (r.get("H_branch") or "") == "avg"]
    if avg:
        print(f"  fit branch: averaged ({len(avg)} points)")
        return avg

    target = "fwd" if which == "F" else "bwd"
    label = "forward" if which == "F" else "backward"
    selected = [r for r in records if (r.get("H_branch") or "") == target]
    if selected:
        print(f"  fit branch: {label} only ({len(selected)} points)")
        return selected

    print(
        f"  [warn] requested branch '{label}' not found; "
        f"using all records (n={len(records)})",
        file=sys.stderr,
    )
    return records


def select_branches_for_plot(records, plot_bf=False, which="F"):
    """Return the records drawn on the figure.

    - Primary series: averaged set if present, else the branch given by
      *which* ('F' → fwd, 'B' → bwd; default forward).
    - Opposite sweep is overlaid only when plot_bf is True.
    """
    which = (which or "F").upper()
    avg = [r for r in records if (r.get("H_branch") or "") == "avg"]
    if avg:
        # Averaged data already replaces both branches
        return avg

    fwd = [r for r in records if (r.get("H_branch") or "") == "fwd"]
    bwd = [r for r in records if (r.get("H_branch") or "") == "bwd"]
    other = [
        r for r in records
        if (r.get("H_branch") or "") not in ("fwd", "bwd", "avg")
    ]

    primary = fwd if which == "F" else bwd
    opposite = bwd if which == "F" else fwd

    out = list(primary) if primary else []
    if plot_bf and opposite:
        out.extend(opposite)
    if not out:
        # No branch tags → plot everything
        out = list(records)
    else:
        out.extend(other)

    out.sort(key=lambda r: (r["H_mT"], r.get("H_branch") or ""))
    return out


# ----------------------------------------------------------------------
# Model functions
# ----------------------------------------------------------------------
def _sinc(x):
    """sin(x)/x with the proper limit at zero.

    Note: numpy.sinc is sin(πx)/(πx); here we use the ordinary definition
    so that sinc(π H/H0) = sin(π H/H0) / (π H/H0).
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    near = np.abs(x) < 1e-12
    out[near] = 1.0
    xn = x[~near]
    out[~near] = np.sin(xn) / xn
    return out


def fraunhofer_1H(H, Ic0, DeltaH0, H_off, I_bg):
    """Conventional Fraunhofer pattern with field offset and background.

    Ic(H) = I_bg + Ic0 * |sin(π (H−H_off)/ΔH₀) / (π (H−H_off)/ΔH₀)|

    The central lobe peaks at H = H_off.
    All field quantities are in mT; currents in µA (or normalised units).
    """
    arg = np.pi * (np.asarray(H, dtype=float) - H_off) / DeltaH0
    return I_bg + Ic0 * np.abs(_sinc(arg))


def fraunhofer_2H(H, Ic1, Ic2, DeltaH0, H_off, I_bg):
    """First + second harmonic Fraunhofer pattern with field offset and background.

    Ic(H) = I_bg + Ic1 * |sinc(π (H−H_off)/ΔH₀)|
                 + Ic2 * |sinc(2π (H−H_off)/ΔH₀)|

    The central lobe peaks at H = H_off.
    """
    arg = np.pi * (np.asarray(H, dtype=float) - H_off) / DeltaH0
    return I_bg + Ic1 * np.abs(_sinc(arg)) + Ic2 * np.abs(_sinc(2.0 * arg))


def _estimate_H0(H, Ic):
    """Estimate the Fraunhofer period from the central lobe.

    Strategy
    --------
    1. Locate the point closest to H = 0 (central-peak reference).
    2. Walk outward until Ic falls below ~5 % of the peak.  For a pure
       |sinc| the first zero sits at |H| = H0, so this half-width is a
       direct estimate of H0.
    3. Also look for the first local minimum away from zero (side-lobe
       trough of a pure Fraunhofer) — another estimator of H0.
    4. Return the median of the available estimators; fall back to half
       the H span if nothing is found.
    """
    H = np.asarray(H, dtype=float)
    Ic = np.asarray(Ic, dtype=float)
    order = np.argsort(H)
    H, Ic = H[order], Ic[order]

    i0 = int(np.argmin(np.abs(H)))
    Ic0 = max(float(Ic[i0]), float(np.nanmax(Ic)) * 0.5)
    thresh = 0.05 * Ic0

    estimates = []

    # Half-width at low threshold (≈ first zero for pure |sinc|)
    for direction in (1, -1):
        idx = i0
        H_cross = None
        while 0 <= idx + direction < len(H):
            idx += direction
            if Ic[idx] < thresh:
                H_cross = abs(float(H[idx]))
                break
        if H_cross is not None and H_cross > 0:
            estimates.append(H_cross)

    # First local minimum on each side
    for direction in (1, -1):
        idx = i0
        prev = float(Ic[i0])
        found_min = None
        while 0 <= idx + direction < len(H):
            idx += direction
            cur = float(Ic[idx])
            if cur < 0.9 * Ic0 and 0 <= idx + direction < len(H):
                nxt = float(Ic[idx + direction])
                if cur <= prev and cur <= nxt:
                    found_min = abs(float(H[idx]))
                    break
            prev = cur
        if found_min is not None and found_min > 0:
            estimates.append(found_min)

    if estimates:
        H0_est = float(np.median(estimates))
    else:
        H0_est = max(float(np.ptp(H)) / 2.0, 0.1)

    return max(H0_est, 1e-3)


def fit_model(
    H, Ic, model, p0=None, Delta0_initial=None, H_off_initial=None,
    H_err=None,
):
    """Fit the selected Fraunhofer model and return popt, perr, metrics.

    Fitting process (detailed)
    --------------------------
    1. **Field offset.**  The model includes a free parameter H_off — the
       field location of the central-lobe maximum.  Its initial guess is
       taken from ``--H=0-offset`` (or 0 if not supplied).  After a
       two-value branch alignment under ``--BF``, H_off_initial is reset
       to 0 because the data have already been shifted.

    2. **Initial guesses**
       - Ic0 / Ic1  ← value of Ic at the point closest to H_off_initial.
       - Ic2        ← ~0.3 × Ic1  (typical second-harmonic weight).
       - ΔH₀        ← ``Delta0_initial`` if supplied (``--Delta0-initial``),
         otherwise estimated from the half-width / first minimum of the
         central lobe relative to H_off_initial (see ``_estimate_H0``).
       - H_off      ← ``H_off_initial`` (from ``--H=0-offset``).
       - I_bg       ← min(Ic) clipped at ≥ 0, or a small fraction of Ic0.

    3. **Bounds**
       - All amplitudes ≥ 0;  ΔH₀ > 0;  I_bg ≥ 0.
       - H_off unrestricted (can be positive or negative).

    4. **Optimiser**
       - Default: ``scipy.optimize.curve_fit`` (Trust-Region Reflective).
       - With ``H_err`` (``--H-plot-error``): Orthogonal Distance
         Regression (``scipy.odr``) so that field uncertainties enter
         the residual.  Several ΔH₀ starts are tried; lowest SSR wins.

    5. **Metrics**  R², RMSE, SSR, AIC, BIC.
    """
    H = np.asarray(H, dtype=float)
    Ic = np.asarray(Ic, dtype=float)
    if H_err is not None:
        H_err = np.asarray(H_err, dtype=float)
        mask = np.isfinite(H) & np.isfinite(Ic) & np.isfinite(H_err) & (H_err > 0)
    else:
        mask = np.isfinite(H) & np.isfinite(Ic)
        H_err = None
    H, Ic = H[mask], Ic[mask]
    if H_err is not None:
        H_err = H_err[mask]
    if len(H) < 5:
        raise RuntimeError(
            f"Only {len(H)} finite points; need ≥ 5 for a fit "
            f"(Ic, ΔH₀, H_off, I_bg)."
        )

    # --- initial guesses ------------------------------------------------
    H_off_guess = float(H_off_initial) if H_off_initial is not None else 0.0
    i0 = int(np.argmin(np.abs(H - H_off_guess)))
    Ic_at_peak = float(Ic[i0])
    Ic_max = float(np.nanmax(Ic))
    Ic_min = float(np.nanmin(Ic))
    # Prefer the value near the predicted peak; fall back to global max
    Ic0_guess = Ic_at_peak if Ic_at_peak > 0.1 * Ic_max else Ic_max

    # Period estimate relative to the predicted peak position
    if Delta0_initial is not None and Delta0_initial > 0:
        DeltaH0_guess = float(Delta0_initial)
        Delta_source = "user (--Delta0-initial)"
    else:
        DeltaH0_guess = _estimate_H0(H - H_off_guess, Ic)
        Delta_source = "auto (_estimate_H0)"

    I_bg_guess = max(0.0, min(Ic_min, 0.05 * Ic0_guess))

    print("  fit initial guesses:")
    print(f"    Ic near H_off      = {Ic_at_peak:.4g} µA  (index {i0})")
    print(f"    Ic0 / Ic1 guess    = {Ic0_guess:.4g} µA")
    print(f"    ΔH₀ guess          = {DeltaH0_guess:.4g} mT  [{Delta_source}]")
    print(f"    H(=0) offset guess = {H_off_guess:.4g} mT")
    print(f"    I_bg guess         = {I_bg_guess:.4g} µA")

    # H span for H_off bounds (keep the peak inside the measured window)
    H_span = float(np.ptp(H)) if len(H) > 1 else 1.0
    H_off_lo = float(np.min(H)) - 0.5 * H_span
    H_off_hi = float(np.max(H)) + 0.5 * H_span

    if model == "1H":
        func = fraunhofer_1H
        names = ["Ic0_uA", "DeltaH0_mT", "H_off_mT", "I_bg_uA"]
        bounds = (
            [0.0, 1e-6, H_off_lo, 0.0],
            [np.inf, np.inf, H_off_hi, np.inf],
        )
        starts = (
            [p0] if p0 is not None
            else [
                [Ic0_guess, DeltaH0_guess, H_off_guess, I_bg_guess],
                [Ic0_guess, DeltaH0_guess * 0.7, H_off_guess, I_bg_guess],
                [Ic0_guess, DeltaH0_guess * 1.4, H_off_guess, I_bg_guess],
            ]
        )
    elif model == "2H":
        func = fraunhofer_2H
        names = ["Ic1_uA", "Ic2_uA", "DeltaH0_mT", "H_off_mT", "I_bg_uA"]
        bounds = (
            [0.0, 0.0, 1e-6, H_off_lo, 0.0],
            [np.inf, np.inf, np.inf, H_off_hi, np.inf],
        )
        starts = (
            [p0] if p0 is not None
            else [
                [0.7 * Ic0_guess, 0.3 * Ic0_guess,
                 DeltaH0_guess, H_off_guess, I_bg_guess],
                [0.8 * Ic0_guess, 0.2 * Ic0_guess,
                 DeltaH0_guess * 0.7, H_off_guess, I_bg_guess],
                [0.6 * Ic0_guess, 0.4 * Ic0_guess,
                 DeltaH0_guess * 1.3, H_off_guess, I_bg_guess],
                [Ic0_guess, 0.1 * Ic0_guess,
                 DeltaH0_guess, H_off_guess, I_bg_guess],
            ]
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    # Multi-start: keep the solution with the lowest residual SSR
    use_odr = H_err is not None and np.any(H_err > 0)
    if use_odr:
        print(
            f"  using ODR with field errors "
            f"(median σ_H = {np.nanmedian(H_err):.4g} mT)"
        )

    best = None
    for start in starts:
        try:
            if use_odr:
                # scipy.odr: beta = parameters, x = independent variable
                def _f(beta, x, _func=func):
                    return _func(x, *beta)

                odr_model = odr.Model(_f)
                data = odr.RealData(H, Ic, sx=H_err)
                odr_obj = odr.ODR(data, odr_model, beta0=list(start))
                odr_obj.set_job(fit_type=0)  # explicit ODR
                out = odr_obj.run()
                if out.info >= 4:  # did not converge
                    continue
                popt_try = np.asarray(out.beta, dtype=float)
                # Covariance from ODR (may be None)
                if out.cov_beta is not None:
                    pcov_try = np.asarray(out.cov_beta, dtype=float)
                else:
                    pcov_try = np.full((len(popt_try), len(popt_try)), np.nan)
                # Enforce physical bounds after the fact
                lo, hi = bounds
                if np.any(popt_try < lo) or np.any(popt_try > hi):
                    continue
            else:
                popt_try, pcov_try = curve_fit(
                    func, H, Ic, p0=start, bounds=bounds, maxfev=30000,
                )
            resid = float(np.sum((Ic - func(H, *popt_try)) ** 2))
            if best is None or resid < best[0]:
                best = (resid, popt_try, pcov_try)
        except Exception:
            continue

    if best is None:
        raise RuntimeError(
            "fit failed for every starting point"
            + (" (ODR)" if use_odr else " (curve_fit)")
        )

    _, popt, pcov = best
    perr = np.sqrt(np.diag(pcov))

    # Goodness-of-fit metrics
    Ic_fit = func(H, *popt)
    ss_res = float(np.sum((Ic - Ic_fit) ** 2))
    ss_tot = float(np.sum((Ic - np.mean(Ic)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(Ic)))
    n, k = len(Ic), len(popt)
    aic = n * np.log(ss_res / n + 1e-30) + 2 * k
    bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)

    metrics = {
        "R2": r2,
        "RMSE": rmse,
        "SSR": ss_res,
        "AIC": aic,
        "BIC": bic,
        "n_points": n,
        "n_params": k,
    }
    return popt, perr, names, metrics, func


def write_fit_results_csv(path, fit_records, popt, names, metrics, model, func):
    """Write per-point measured / predicted values for statistical comparison.

    Columns
    -------
    H_mT, H_err_mT, H_branch,
    Ic_meas_uA, Ic_pred_uA, residual_uA, residual_sq,
    model, RMSE_global, R2, n_points, n_params
    """
    H = np.array([r["H_mT"] for r in fit_records], dtype=float)
    Ic = np.array([r["Ic_active_uA"] for r in fit_records], dtype=float)
    H_err = [
        r["H_err_mT"] if r.get("H_err_mT") is not None else ""
        for r in fit_records
    ]
    branches = [r.get("H_branch") or "" for r in fit_records]

    mask = np.isfinite(H) & np.isfinite(Ic)
    H_ok, Ic_ok = H[mask], Ic[mask]
    Ic_pred = func(H_ok, *popt)
    resid = Ic_ok - Ic_pred
    resid_sq = resid ** 2

    # Map full-length arrays for rows that were dropped by the mask
    pred_full = np.full(len(H), np.nan)
    resid_full = np.full(len(H), np.nan)
    resid_sq_full = np.full(len(H), np.nan)
    pred_full[mask] = Ic_pred
    resid_full[mask] = resid
    resid_sq_full[mask] = resid_sq

    fieldnames = [
        "H_mT", "H_err_mT", "H_branch",
        "Ic_meas_uA", "Ic_pred_uA", "residual_uA", "residual_sq",
        "model", "RMSE_global", "R2", "n_points", "n_params",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(len(H)):
            w.writerow({
                "H_mT": H[i],
                "H_err_mT": H_err[i],
                "H_branch": branches[i],
                "Ic_meas_uA": Ic[i] if np.isfinite(Ic[i]) else "",
                "Ic_pred_uA": (
                    pred_full[i] if np.isfinite(pred_full[i]) else ""
                ),
                "residual_uA": (
                    resid_full[i] if np.isfinite(resid_full[i]) else ""
                ),
                "residual_sq": (
                    resid_sq_full[i] if np.isfinite(resid_sq_full[i]) else ""
                ),
                "model": model,
                "RMSE_global": metrics["RMSE"],
                "R2": metrics["R2"],
                "n_points": metrics["n_points"],
                "n_params": metrics["n_params"],
            })
    print(f"Fit results CSV written to: {os.path.abspath(path)}")


def period_to_deff(H0_mT, width_um):
    """Convert fitted period (mT) and junction width (μm) to d_eff (nm).

    ΔB [T] = H0 [mT] × 10⁻³
    d_eff  = Φ0 / (w × ΔB)
    """
    if width_um is None or width_um <= 0:
        return None
    H0_T = H0_mT * 1e-3
    w_m = width_um * 1e-6
    d_eff_m = PHI0 / (w_m * H0_T)
    return d_eff_m * 1e9                     # m → nm


# ----------------------------------------------------------------------
# Logging of analysis results
# ----------------------------------------------------------------------
def write_analysis_log(path, args, records, fit_result, norm_Ic0):
    """Rewrite analysis.log with the parameters and fit summary of this run."""
    lines = []
    lines.append("Fraunhofer analysis")
    lines.append("=" * 60)
    lines.append(
        f"timestamp          : {datetime.now().isoformat(timespec='seconds')}"
    )
    lines.append(f"n_points           : {len(records)}")
    if records:
        H = [r["H_mT"] for r in records]
        lines.append(f"H_range_mT         : {min(H):.6g} … {max(H):.6g}")
    lines.append("")
    lines.append("--- CLI options ---")
    for k, v in sorted(vars(args).items()):
        lines.append(f"  {k:20s} = {v}")
    lines.append("")

    if norm_Ic0 is not None:
        lines.append(f"normalisation_Ic0_uA = {norm_Ic0:.8g}")
        lines.append("")

    if fit_result is not None:
        (popt, perr, names, metrics, model, width_um,
         H_off_init, Delta0_init) = fit_result
        lines.append(f"--- model fit ({model}) ---")
        # Friendly names for the log
        label_map = {
            "Ic0_uA": "Ic0 (µA)",
            "Ic1_uA": "Ic1 (µA)",
            "Ic2_uA": "Ic2 (µA)",
            "DeltaH0_mT": "ΔH₀ (mT)",
            "H_off_mT": "H(=0) (offset) (mT)",
            "I_bg_uA": "I_bg (µA)",
        }
        for n, v, e in zip(names, popt, perr):
            label = label_map.get(n, n)
            extra = ""
            if n == "H_off_mT" and H_off_init is not None:
                extra = f"  [initial: {H_off_init:.6g}]"
            if n == "DeltaH0_mT" and Delta0_init is not None:
                extra = f"  [initial: {Delta0_init:.6g}]"
            lines.append(f"  {label:28s} = {v:.8g}  ± {e:.4g}{extra}")
        DeltaH0 = popt[names.index("DeltaH0_mT")]
        if width_um is not None:
            d_eff = period_to_deff(DeltaH0, width_um)
            lines.append(f"  {'width (µm)':28s} = {width_um:.6g}")
            if d_eff is not None:
                lines.append(f"  {'d_eff (nm)':28s} = {d_eff:.6g}")
        if model == "2H":
            Ic1 = popt[names.index("Ic1_uA")]
            Ic2 = popt[names.index("Ic2_uA")]
            ratio = Ic1 / Ic2 if Ic2 != 0 else np.nan
            lines.append(f"  {'Ic1/Ic2':28s} = {ratio:.6g}")
        lines.append("")
        lines.append("--- goodness of fit ---")
        for k in ("R2", "RMSE", "SSR", "AIC", "BIC", "n_points", "n_params"):
            lines.append(f"  {k:20s} = {metrics[k]}")
    else:
        lines.append("--- no model fit performed ---")

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Analysis log written to: {os.path.abspath(path)}")


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def _build_dvdi_background(records):
    """Assemble a (H, I, Z) grid from per-point dV/dI arrays.

    Returns
    -------
    H_vals : 1-D array of field positions (mT), sorted
    I_grid : 1-D common current grid (µA)
    Z      : 2-D array shape (n_I, n_H) of dV/dI in Ohm
    or (None, None, None) if no usable data.
    """
    strips = []
    n_total = len(records)
    n_has_z = n_has_i = n_ok = 0
    for r in records:
        I_arr = r.get("I_array_uA")
        Z_arr = r.get("dVdI_array_Ohm")
        H = r.get("H_mT")
        # Tolerate arrays that arrived as strings (e.g. stale in-memory state)
        if isinstance(I_arr, str):
            I_arr = _str_to_array(I_arr)
        if isinstance(Z_arr, str):
            Z_arr = _str_to_array(Z_arr)
        if Z_arr is not None:
            n_has_z += 1
        if I_arr is not None:
            n_has_i += 1
        if Z_arr is None or H is None:
            continue
        Z_arr = np.asarray(Z_arr, dtype=float).ravel()
        if Z_arr.size < 2 or not np.any(np.isfinite(Z_arr)):
            continue
        if I_arr is None:
            # Synthesise a linear current axis if only dV/dI was stored
            I_arr = np.arange(Z_arr.size, dtype=float)
        else:
            I_arr = np.asarray(I_arr, dtype=float).ravel()
        n = min(I_arr.size, Z_arr.size)
        if n < 2:
            continue
        I_arr, Z_arr = I_arr[:n], Z_arr[:n]
        finite = np.isfinite(I_arr) & np.isfinite(Z_arr)
        if np.count_nonzero(finite) < 2:
            continue
        I_arr, Z_arr = I_arr[finite], Z_arr[finite]
        order = np.argsort(I_arr)
        strips.append((float(H), I_arr[order], Z_arr[order]))
        n_ok += 1

    if len(strips) < 1:
        print(
            f"  [diag] dV/dI background: {n_total} records, "
            f"{n_has_z} with dVdI_array_Ohm, {n_has_i} with I_array_uA, "
            f"{n_ok} usable strips",
            file=sys.stderr,
        )
        return None, None, None

    if len(strips) == 1:
        print(
            "  [warn] only one field point has dV/dI data; "
            "background map will be a single column.",
            file=sys.stderr,
        )

    strips.sort(key=lambda t: t[0])
    H_vals = np.array([s[0] for s in strips])

    I_min = min(float(s[1].min()) for s in strips)
    I_max = max(float(s[1].max()) for s in strips)
    if not np.isfinite(I_min) or not np.isfinite(I_max) or I_max <= I_min:
        return None, None, None
    n_I = int(np.median([s[1].size for s in strips]))
    n_I = max(n_I, 50)
    I_grid = np.linspace(I_min, I_max, n_I)

    Z = np.full((n_I, len(strips)), np.nan)
    for j, (_, I_arr, Z_arr) in enumerate(strips):
        Z[:, j] = np.interp(I_grid, I_arr, Z_arr, left=np.nan, right=np.nan)

    print(
        f"  dV/dI background: {len(strips)} strips, "
        f"I ∈ [{I_min:.4g}, {I_max:.4g}] µA, "
        f"grid {n_I} × {len(strips)}"
    )
    return H_vals, I_grid, Z


def plot_fraunhofer(
    records,
    fit_result=None,
    normalize=False,
    I_plot_range=None,
    H_plot_range=None,
    plot_x_Phi=False,
    plot_H_error=False,
    plot_dvdi=False,
    smooth_dvdi=None,
    Rn_Ohm=None,
    out_path="Fraunhofer_fit.pdf",
):
    """Produce a publication-quality Ic(H) figure.

    Data are drawn as small markers connected by a thin solid line.
    Field axis is always in mT (or Φ/Φ0); current axis is µA by default
    (or normalised units).  With *Rn_Ohm*, the y-axis is Ic·Rn in mV and
    dV/dI is shown as (dV/dI)/Rn (dimensionless).

    When *plot_dvdi* is True a false-colour dV/dI map is drawn behind the
    Ic points; the legend is then placed below the figure.  Otherwise the
    legend stays inside the axes.

    *smooth_dvdi*: ``None`` = no filter; ``int`` = median-filter kernel
    size along the current axis (odd, ≥ 3).
    """
    set_paper_style()
    # Wider with colourbar; taller only when legend sits below
    fig_w = 4.2 if plot_dvdi else 3.5
    fig_h = 3.2 if plot_dvdi else 2.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    H = np.array([r["H_mT"] for r in records], dtype=float)
    Ic = np.array([r["Ic_active_uA"] for r in records], dtype=float)
    branches = [r.get("H_branch") for r in records]

    # Optional Ic → Ic·Rn (mV):  I[µA]·Rn[Ω]·1e-3 = mV
    use_Rn = Rn_Ohm is not None and float(Rn_Ohm) > 0
    Rn = float(Rn_Ohm) if use_Rn else None
    y_scale = (Rn * 1e-3) if use_Rn else 1.0
    Ic = Ic * y_scale
    H_err = np.array([
        r["H_err_mT"] if r.get("H_err_mT") is not None else np.nan
        for r in records
    ], dtype=float)

    # Decide x-axis units
    if plot_x_Phi and fit_result is not None:
        popt, perr, names, metrics, model, width_um, _, _ = fit_result
        DeltaH0 = popt[names.index("DeltaH0_mT")]
        H_off = popt[names.index("H_off_mT")]
        x = (H - H_off) / DeltaH0
        xerr = H_err / DeltaH0 if plot_H_error else None
        x_label = r"$\Phi / \Phi_0$"
        x_model = np.linspace(x.min(), x.max(), 800)
        H_model = x_model * DeltaH0 + H_off
        H_to_x = lambda h: (h - H_off) / DeltaH0
    else:
        x = H
        xerr = H_err if plot_H_error else None
        x_label = r"$H$ (mT)"
        if fit_result is not None:
            if H_plot_range is not None:
                H_lo, H_hi = float(H_plot_range[0]), float(H_plot_range[1])
            else:
                H_lo, H_hi = float(H.min()), float(H.max())
            H_model = np.linspace(H_lo, H_hi, 800)
            x_model = H_model
        H_to_x = lambda h: h

    # --- dV/dI intensity background ------------------------------------
    show_dvdi = bool(plot_dvdi)
    if show_dvdi and normalize:
        print(
            "  [warn] --plot-dvdi ignored with --normalize "
            "(background needs absolute current axis).",
            file=sys.stderr,
        )
        show_dvdi = False

    if show_dvdi:
        H_bg, I_bg, Z_bg = _build_dvdi_background(records)
        if H_bg is None:
            print(
                "  [warn] --plot-dvdi requested but no usable "
                "dVdI_array_Ohm data found.",
                file=sys.stderr,
            )
            show_dvdi = False
        else:
            x_bg = H_to_x(H_bg)
            # pcolormesh needs edge coordinates; pad half-steps at ends
            if len(x_bg) >= 2:
                dx = np.diff(x_bg)
                x_edges = np.concatenate([
                    [x_bg[0] - dx[0] / 2],
                    x_bg[:-1] + dx / 2,
                    [x_bg[-1] + dx[-1] / 2],
                ])
            else:
                x_edges = np.array([x_bg[0] - 0.5, x_bg[0] + 0.5])
            if len(I_bg) >= 2:
                di = np.diff(I_bg)
                I_edges = np.concatenate([
                    [I_bg[0] - di[0] / 2],
                    I_bg[:-1] + di / 2,
                    [I_bg[-1] + di[-1] / 2],
                ])
            else:
                I_edges = np.array([I_bg[0] - 0.5, I_bg[0] + 0.5])

            # Scale current axis and resistance when --Rn is set
            if use_Rn:
                I_bg = I_bg * y_scale          # µA → mV (Ic·Rn)
                Z_bg = Z_bg / Rn               # Ω → (dV/dI)/Rn
                I_edges = I_edges * y_scale

            if smooth_dvdi is not None:
                try:
                    from scipy.ndimage import median_filter
                except ImportError:
                    from scipy.signal import medfilt as _medfilt

                    def median_filter(a, size):
                        if isinstance(size, tuple):
                            return _medfilt(a, kernel_size=size)
                        return _medfilt(a, kernel_size=size)

                k = int(smooth_dvdi)
                if k < 1:
                    k = 1
                if k % 2 == 0:
                    k += 1  # median filter needs odd kernel
                if Z_bg.shape[0] < k:
                    k = max(1, Z_bg.shape[0] // 2 * 2 - 1) or 1
                if k >= 3:
                    Z_bg = median_filter(Z_bg, size=(k, 1))
                    print(f"  --smooth-dvdi: median filter size={k} along I")
                else:
                    print(
                        f"  [warn] --smooth-dvdi k={k} too small; skipped",
                        file=sys.stderr,
                    )

            finite = Z_bg[np.isfinite(Z_bg)]
            if finite.size:
                vmin, vmax = np.percentile(finite, [2, 98])
            else:
                vmin, vmax = 0.0, 1.0
            im = ax.pcolormesh(
                x_edges, I_edges, Z_bg,
                cmap="jet", shading="flat",
                vmin=vmin, vmax=vmax, zorder=0,
                rasterized=True,
            )
            cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            if use_Rn:
                cbar.set_label(r"$(dV/dI)/R_n$", fontsize=8)
            else:
                cbar.set_label(r"dV/dI ($\Omega$)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    # Bright overlay colours when a dark background map is present
    if show_dvdi:
        series_cmap = {"fwd": "#00FFFF", "bwd": "#FF66FF", "avg": "#FFFF00"}
        fit_color = "#FFFFFF"
        text_color = "white"
        mew = 0.6
    else:
        series_cmap = {"fwd": "C0", "bwd": "C1", "avg": "C2"}
        fit_color = "0.25"
        text_color = "black"
        mew = 0.4

    unique_branches = sorted(set(b for b in branches if b))
    markers = {"fwd": "o", "bwd": "s", "avg": "D"}

    def _draw_series(xx, yy, xxe, color, marker, label):
        order = np.argsort(xx)
        xx, yy = xx[order], yy[order]
        if xxe is not None:
            xxe = xxe[order]
        ax.plot(
            xx, yy,
            marker=marker, ms=2.5, mew=mew,
            markeredgecolor="k" if show_dvdi else color,
            linestyle="-", lw=0.9 if show_dvdi else 0.8,
            color=color, label=label, zorder=3,
        )
        if xxe is not None and np.any(np.isfinite(xxe)):
            ax.errorbar(
                xx, yy, xerr=xxe,
                fmt="none", ecolor=color, elinewidth=0.6,
                capsize=1.2, alpha=0.7, zorder=2,
            )

    if not unique_branches:
        _draw_series(x, Ic, xerr, series_cmap.get("fwd", "C0"), "o", "data")
    else:
        for br in unique_branches:
            mask = np.array([b == br for b in branches])
            xe = xerr[mask] if xerr is not None else None
            _draw_series(
                x[mask], Ic[mask], xe,
                series_cmap.get(br, "C0"),
                markers.get(br, "o"),
                br,
            )

    # Model overlay (dashed) — scale to display units when --Rn is set
    Ic_model = None
    r2_text = None
    if fit_result is not None:
        popt, perr, names, metrics, model, width_um, _, _ = fit_result
        func = fraunhofer_1H if model == "1H" else fraunhofer_2H
        Ic_model = func(H_model, *popt) * y_scale
        DeltaH0 = popt[names.index("DeltaH0_mT")]
        H_off = popt[names.index("H_off_mT")]
        label = f"fit ({model})"
        label += rf"  $\Delta H_0$={DeltaH0:.3g} mT"
        label += rf", $H_{{\rm off}}$={H_off:.3g} mT"
        ax.plot(
            x_model, Ic_model, "--", color=fit_color, lw=1.2,
            label=label, zorder=4,
        )
        r2_text = rf"$R^2 = {metrics['R2']:.4f}$"

    ax.set_xlabel(x_label)
    if normalize:
        ax.set_ylabel(r"$I_c / I_c(0)$")
    elif use_Rn:
        ax.set_ylabel(r"$I_c R_n$ (mV)")
    else:
        ax.set_ylabel(r"$I_c$ (µA)")

    # Y-limits always follow Ic data (+ fit), never the full dV/dI I-range
    if I_plot_range is not None:
        ax.set_ylim(float(I_plot_range[0]), float(I_plot_range[1]))
    else:
        y_vals = Ic[np.isfinite(Ic)]
        if Ic_model is not None:
            y_vals = np.concatenate([y_vals, Ic_model[np.isfinite(Ic_model)]])
        if y_vals.size:
            y_lo, y_hi = float(np.min(y_vals)), float(np.max(y_vals))
            if y_hi > y_lo:
                pad = 0.05 * (y_hi - y_lo)
                ax.set_ylim(y_lo - pad, y_hi + pad)
            else:
                ax.set_ylim(y_lo - 1.0, y_hi + 1.0)

    if H_plot_range is not None and not plot_x_Phi:
        ax.set_xlim(float(H_plot_range[0]), float(H_plot_range[1]))

    handles, labels = ax.get_legend_handles_labels()
    if show_dvdi:
        # Legend + R² below the figure (black) — keeps the map uncluttered
        if handles:
            fig.tight_layout(rect=(0, 0.14, 1, 1))
            leg = fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.08),
                ncol=min(len(handles), 3),
                frameon=False,
                fontsize=7,
            )
            for t in leg.get_texts():
                t.set_color("black")
            if r2_text is not None:
                fig.text(
                    0.98, 0.02, r2_text,
                    ha="right", va="bottom", fontsize=7, color="black",
                )
        else:
            fig.tight_layout()
    else:
        # Original behaviour: legend inside the axes
        if handles:
            leg = ax.legend(loc="best")
        if r2_text is not None:
            ax.text(
                0.97, 0.95, r2_text,
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="black",
            )
        fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to: {os.path.abspath(out_path)}")


# ----------------------------------------------------------------------
# Argument parser & main
# ----------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Analyse Fraunhofer Ic(H) patterns from singles/<Hlabel>/analysis.log "
            "files (or from a previously saved CSV) and optionally fit "
            "conventional (1H) or second-harmonic (2H) models.  "
            "Currents are plotted in µA, fields in mT."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Argument summary
----------------
  Data source
    --dir DIR          Path to the main folder (containing 'singles/') OR to
                       the 'singles/' folder itself.  Globs every
                       <Hlabel>/analysis.log underneath (default: current
                       directory).  Ignored if --from-csv is given.
    --from-csv FILE    Read data from this CSV instead of scanning folders.
    --save-csv FILE    Where to write the collected CSV (default:
                       Fraunhofer_data.csv).

  Field handling  (all field values in mT)
    --H=0-offset MT [MT]
                       Initial guess for the field location of the central-
                       lobe maximum (fitted as H_off).  When --BF is set,
                       two values may be given (fwd, bwd) to align the
                       branches before averaging; after alignment H_off
                       starts at 0.
    --H-mean           Use Hsample_mean_Oe as the field for each point
                       instead of the nominal H (still converted to mT).
    --H-plot-error     Use Hsample_sem as σ_H in the fit (ODR) and draw
                       horizontal error bars on the figure.
    --BF               Bi-directional field sweeps present.
    --average          Average fwd and bwd over the overlapping H window
                       (requires --BF).  Branch offsets are applied first.
    --plot-BF          Overlay the opposite sweep on the figure (requires
                       --BF).
    --model-use-data {F,B}
                       Which sweep to fit and plot as primary: F = forward
                       (default), B = backward.  Must be given a value.

  Plot options
    --normalize        Normalise Ic by the value nearest H = 0 (after any
                       offset).
    --H-plot-range LO HI
                       Restrict the plotted / fitted H window (mT).
    --I-plot-range LO HI
                       Y-axis limits (µA, or dimensionless if --normalize).
    --plot-x-Phi       When a model is fitted, plot versus Φ/Φ0 instead of H.
    --plot-dvdi        Overlay a dV/dI intensity map as the plot background
                       (colourbar on the right).  Loads dVdI.dat from each
                       singles/<H>/ folder when scanning, or reads
                       I_array_uA / dVdI_array_Ohm from the CSV.
    --out PATH         Output figure path (default: Fraunhofer_fit.pdf).

  Models
    --model {1H,2H}    Fit the corresponding Fraunhofer pattern (always on
                       the forward / averaged set).  Both models include a
                       constant background I_bg and a free H(=0) offset.
    --Delta0-initial MT
                       Initial guess for the Fraunhofer period ΔH₀ (mT).
                       Overrides the automatic half-width estimate.
    --width UM         Junction width in μm; used to compute d_eff from the
                       fitted period.
    --no-fit           Explicitly skip fitting even if --model is given.
    --write-fit-results
                       Write Fraunhofer_fit_data.csv (measured vs predicted
                       Ic, per-point residuals, global RMSE / R²).

Examples
--------
    python Fraunhofer_analysis.py --dir ./my_run --model 1H --width 1.2
    python Fraunhofer_analysis.py --from-csv Fraunhofer_data.csv --model 2H \\
        --H-plot-range -3 3 --normalize --plot-x-Phi
    python Fraunhofer_analysis.py --model 1H --H=0-offset 0.12 \\
        --Delta0-initial 2.5 --width 0.8
    python Fraunhofer_analysis.py --BF --average --H=0-offset 0.12 -0.05 \\
        --model 1H --width 0.8
    python Fraunhofer_analysis.py --BF --plot-BF --model 1H
""",
    )

    # Data source
    parser.add_argument(
        "--dir", default=".",
        help="Main folder or singles/ folder (default: .)",
    )
    parser.add_argument(
        "--from-csv", default=None, metavar="FILE",
        help="Load data from CSV instead of scanning folders",
    )
    parser.add_argument(
        "--save-csv", default="Fraunhofer_data.csv", metavar="FILE",
        help="CSV path written after collection "
             "(default: Fraunhofer_data.csv)",
    )

    # Field handling (mT)
    parser.add_argument(
        "--H=0-offset", type=float, nargs="+", default=None, metavar="MT",
        dest="H_eq0_offset",
        help="Initial guess for central-lobe field position H_off (mT); "
             "with --BF two values (fwd, bwd) also align branches",
    )
    parser.add_argument(
        "--H-mean", action="store_true",
        help="Use Hsample_mean_Oe (converted to mT) as the field value "
             "for each point instead of the nominal H",
    )
    parser.add_argument(
        "--H-plot-error", action="store_true",
        help="Use field uncertainty (Hsample_sem → H_err_mT) in the fit "
             "(ODR) and draw horizontal error bars on the plot",
    )
    parser.add_argument(
        "--BF", action="store_true",
        help="Bi-directional field sweeps present",
    )
    parser.add_argument(
        "--average", action="store_true",
        help="Average fwd/bwd over overlapping H (requires --BF)",
    )
    parser.add_argument(
        "--plot-BF", action="store_true",
        help="Also plot the opposite sweep (requires --BF)",
    )
    parser.add_argument(
        "--model-use-data", choices=("F", "B"), default=None, metavar="{F,B}",
        help="Primary sweep for fit and plot: F=forward (default), "
             "B=backward.  A value is required when the flag is used.",
    )

    # Plot options
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalise Ic by the value nearest H=0",
    )
    parser.add_argument(
        "--H-plot-range", type=float, nargs=2, default=None,
        metavar=("LO", "HI"),
        help="H window (mT) for plot and fit",
    )
    parser.add_argument(
        "--I-plot-range", type=float, nargs=2, default=None,
        metavar=("LO", "HI"),
        help="Y-axis limits (µA or normalised units)",
    )
    parser.add_argument(
        "--plot-x-Phi", action="store_true",
        help="Plot versus Φ/Φ0 (requires a successful model fit)",
    )
    parser.add_argument(
        "--plot-dvdi", action="store_true",
        help="Draw dV/dI intensity map as plot background "
             "(needs dVdI.dat per folder, or dVdI_array_Ohm in CSV)",
    )
    parser.add_argument(
        "--smooth-dvdi", nargs="?", const=5, type=int, default=None,
        metavar="K",
        help="Median-filter dV/dI along the current axis (only with "
             "--plot-dvdi).  Optional odd kernel size K (default: 5)",
    )
    parser.add_argument(
        "--Rn", type=float, default=None, metavar="OHM",
        help="Normal-state resistance Rn (Ohm).  If set, y-axis is "
             "Ic·Rn in mV and dV/dI is plotted as (dV/dI)/Rn",
    )
    parser.add_argument(
        "--out", default="Fraunhofer_fit.pdf",
        help="Output figure path (default: Fraunhofer_fit.pdf)",
    )

    # Models
    parser.add_argument(
        "--model", choices=("1H", "2H"), default=None,
        help="Fraunhofer model to fit (forward / averaged data only)",
    )
    parser.add_argument(
        "--width", type=float, default=None, metavar="UM",
        help="Junction width in μm (for d_eff calculation)",
    )
    parser.add_argument(
        "--Delta0-initial", type=float, default=None, metavar="MT",
        help="Initial guess for the Fraunhofer period ΔH₀ in mT "
             "(overrides the automatic half-width estimate)",
    )
    parser.add_argument(
        "--no-fit", action="store_true",
        help="Skip model fitting even if --model is given",
    )
    parser.add_argument(
        "--write-fit-results", action="store_true",
        help="Write Fraunhofer_fit_data.csv with measured/predicted Ic, "
             "per-point residuals, and global fit metrics",
    )
    parser.add_argument(
        "--log", default="analysis.log",
        help="Path for the rewritten analysis.log (default: analysis.log)",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.average and not args.BF:
        parser.error("--average requires --BF")
    if args.plot_BF and not args.BF:
        parser.error("--plot-BF requires --BF")
    if args.plot_x_Phi and (args.model is None or args.no_fit):
        parser.error("--plot-x-Phi requires a model fit (--model 1H|2H)")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if args.from_csv:
        print(f"Loading data from CSV: {args.from_csv}")
        records = load_csv(args.from_csv)
        n_jump = sum(
            1 for r in records if r.get("Ic_definition") == "jump_dvdi"
        )
        n_eff = len(records) - n_jump
        print(f"  Ic_definition:  jump_dvdi = {n_jump}  |  rho_eff = {n_eff}")
    else:
        print(
            f"Scanning '{os.path.abspath(os.path.expanduser(args.dir))}' "
            f"for analysis.log files …"
        )
        records = collect_data(args.dir)

    # Attach / recover dV/dI traces so --plot-dvdi works even when the CSV
    # was written without array columns, or analysis.log and dVdI.dat live
    # in slightly different layouts under --dir.
    records, n_att = attach_dvdi_from_dir(records, args.dir)
    if n_att == 0 and args.from_csv:
        csv_dir = os.path.dirname(os.path.abspath(args.from_csv)) or "."
        if os.path.abspath(csv_dir) != os.path.abspath(
            os.path.expanduser(args.dir)
        ):
            records, n_att = attach_dvdi_from_dir(records, csv_dir)

    # Persist collection (includes any recovered dV/dI arrays)
    save_csv(args.save_csv, records)

    # ------------------------------------------------------------------
    # 2. Optional Hsample_mean → branch alignment → average → H-range
    # ------------------------------------------------------------------
    if args.H_mean:
        records = apply_Hsample_mean(records)

    # Two-value --H=0-offset under --BF only pre-shifts branches for
    # averaging; a single value is the H_off fit initial guess (no shift).
    records = apply_branch_alignment(
        records, args.H_eq0_offset, bf=args.BF,
    )

    if args.BF and args.average:
        records = average_bf(records)

    records = select_H_range(records, args.H_plot_range)

    # Resolve H_off initial guess for the fitter
    H_off_initial = None
    if args.H_eq0_offset is not None:
        offs = list(args.H_eq0_offset)
        if args.BF and len(offs) == 2:
            # Data already aligned → residual offset starts at 0
            H_off_initial = 0.0
        else:
            H_off_initial = float(offs[0])

    norm_Ic0 = None
    if args.normalize:
        # Normalise by Ic nearest the predicted peak position
        peak_ref = H_off_initial if H_off_initial is not None else 0.0
        records, norm_Ic0 = normalize_Ic(records, H0=peak_ref)

    if len(records) < 1:
        raise RuntimeError("No data points remain after selection.")

    H_all = np.array([r["H_mT"] for r in records])
    print(
        f"Using {len(H_all)} point(s) after selection: "
        f"H ∈ [{H_all.min():.4g}, {H_all.max():.4g}] mT"
    )

    # ------------------------------------------------------------------
    # 3. Fit  (primary branch from --model-use-data, default forward)
    # ------------------------------------------------------------------
    use_data = args.model_use_data or "F"
    fit_result = None
    if args.model and not args.no_fit:
        fit_records = select_branch_for_fit(records, which=use_data)
        H_fit = np.array([r["H_mT"] for r in fit_records])
        Ic_fit = np.array([r["Ic_active_uA"] for r in fit_records])
        H_err_fit = None
        if args.H_plot_error:
            H_err_fit = np.array([
                r["H_err_mT"] if r.get("H_err_mT") is not None else np.nan
                for r in fit_records
            ], dtype=float)
            n_err = int(np.sum(np.isfinite(H_err_fit) & (H_err_fit > 0)))
            if n_err < 5:
                print(
                    f"  [warn] --H-plot-error set but only {n_err} point(s) "
                    f"have usable H_err; falling back to ordinary fit.",
                    file=sys.stderr,
                )
                H_err_fit = None

        print(f"Fitting model '{args.model}' on {len(H_fit)} point(s) …")
        try:
            popt, perr, names, metrics, func = fit_model(
                H_fit, Ic_fit, args.model,
                Delta0_initial=args.Delta0_initial,
                H_off_initial=H_off_initial,
                H_err=H_err_fit,
            )
            # Store initials alongside the fit for the analysis log
            Delta0_init_used = (
                args.Delta0_initial
                if args.Delta0_initial is not None
                else None
            )
            fit_result = (
                popt, perr, names, metrics, args.model, args.width,
                H_off_initial, Delta0_init_used,
            )

            print("  Fitted parameters:")
            label_map = {
                "Ic0_uA": "Ic0",
                "Ic1_uA": "Ic1",
                "Ic2_uA": "Ic2",
                "DeltaH0_mT": "ΔH₀",
                "H_off_mT": "H(=0) (offset)",
                "I_bg_uA": "I_bg",
            }
            for n, v, e in zip(names, popt, perr):
                label = label_map.get(n, n)
                print(f"    {label:20s} = {v:.6g}  ± {e:.3g}")
            DeltaH0 = popt[names.index("DeltaH0_mT")]
            H_off = popt[names.index("H_off_mT")]
            if H_off_initial is not None:
                print(
                    f"    H(=0) initial      = {H_off_initial:.6g} mT  "
                    f"→ fitted {H_off:.6g} mT"
                )
            if args.width is not None:
                d_eff = period_to_deff(DeltaH0, args.width)
                print(
                    f"    d_eff                = {d_eff:.4g} nm  "
                    f"(w = {args.width:g} μm)"
                )
            if args.model == "2H":
                Ic1 = popt[names.index("Ic1_uA")]
                Ic2 = popt[names.index("Ic2_uA")]
                ratio = Ic1 / Ic2 if Ic2 != 0 else np.nan
                print(f"    Ic1/Ic2              = {ratio:.4g}")
            print(
                f"  R² = {metrics['R2']:.5f}   "
                f"RMSE = {metrics['RMSE']:.4g}   "
                f"AIC = {metrics['AIC']:.2f}   "
                f"BIC = {metrics['BIC']:.2f}"
            )
            if args.write_fit_results:
                write_fit_results_csv(
                    "Fraunhofer_fit_data.csv",
                    fit_records, popt, names, metrics, args.model, func,
                )
        except Exception as exc:
            print(f"  [fit failed] {exc}", file=sys.stderr)
            fit_result = None

    # ------------------------------------------------------------------
    # 4. Select what to plot, write log, draw figure
    # ------------------------------------------------------------------
    plot_records = select_branches_for_plot(
        records,
        plot_bf=(args.BF and args.plot_BF),
        which=use_data,
    )
    print(
        f"Plotting {len(plot_records)} point(s) "
        f"(primary={use_data}, plot-BF={bool(args.BF and args.plot_BF)})"
    )

    if args.plot_dvdi:
        n_z = sum(
            1 for r in plot_records
            if r.get("dVdI_array_Ohm") is not None
            and (
                (isinstance(r["dVdI_array_Ohm"], np.ndarray)
                 and r["dVdI_array_Ohm"].size >= 2)
                or (
                    isinstance(r["dVdI_array_Ohm"], str)
                    and len(r["dVdI_array_Ohm"].strip()) > 0
                )
            )
        )
        n_i = sum(
            1 for r in plot_records
            if r.get("I_array_uA") is not None
            and (
                (isinstance(r["I_array_uA"], np.ndarray)
                 and r["I_array_uA"].size >= 2)
                or (
                    isinstance(r["I_array_uA"], str)
                    and len(r["I_array_uA"].strip()) > 0
                )
            )
        )
        print(
            f"  --plot-dvdi: {n_z}/{len(plot_records)} points have "
            f"dVdI_array_Ohm, {n_i}/{len(plot_records)} have I_array_uA"
        )
        if n_z == 0:
            print(
                "  [hint] Re-run without --from-csv so dVdI.dat is loaded "
                "from each singles/<H>/ folder, or check that the CSV "
                "columns I_array_uA / dVdI_array_Ohm are non-empty.",
                file=sys.stderr,
            )

    write_analysis_log(args.log, args, records, fit_result, norm_Ic0)
    smooth_k = None
    if args.smooth_dvdi is not None:
        if not args.plot_dvdi:
            print(
                "  [warn] --smooth-dvdi ignored without --plot-dvdi",
                file=sys.stderr,
            )
        else:
            smooth_k = int(args.smooth_dvdi)

    if args.Rn is not None and args.Rn <= 0:
        parser.error("--Rn must be a positive resistance in Ohm")

    plot_fraunhofer(
        plot_records,
        fit_result=fit_result,
        normalize=args.normalize,
        I_plot_range=args.I_plot_range,
        H_plot_range=args.H_plot_range,
        plot_x_Phi=args.plot_x_Phi and fit_result is not None,
        plot_H_error=args.H_plot_error,
        plot_dvdi=args.plot_dvdi,
        smooth_dvdi=smooth_k,
        Rn_Ohm=args.Rn,
        out_path=args.out,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
