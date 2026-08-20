"""
plot_csv_data.py

Analysis + plotting helpers for transport-measurement CSV files
(PPMS/MultiVu-style logs).

Design philosophy
------------------
For every measurement type (R(T), I(V), ...) there are two functions:

    1. An "analyze_*" function: pure data wrangling / math. It reads
       the CSV, picks out the relevant columns, and returns a plain
       dict of numpy arrays. It never touches matplotlib.

    2. A "plot_*" function: takes the dict produced by the analyze_*
       function and draws it on a (given or new) matplotlib Axes.
       This is the *only* place that knows about colors, markers,
       labels, fonts, etc.

All purely cosmetic / journal-style choices (fonts, line widths,
tick direction, default colors...) live in `set_paper_style()` and
the small `_default_*_style()` helpers below, so the look of every
figure can be changed from one place without touching any analysis
code.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import glob

import warnings

# ==================================================================
# Style / display helpers
# (the only section that should contain rcParams, default colors,
#  fonts, etc. — analysis functions below must never reference this)
# ==================================================================

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

def _default_RT_branch_colors():
    """Default branch colors for R(T) plots."""
    return {"cooldown": "tab:blue", "warmup": "tab:red"}

def _fit_color(show_branches, dataset_color):
    """Return an appropriate color for a linear fit overlay line.
 
    When cooldown and warmup branches are drawn with *different*
    colors, the fit line must use a third color so the viewer cannot
    mistake it for either branch.  Black ('k') is chosen as a neutral
    default because it is never a branch color in the default palette.
 
    When only a single color is used for the data (branches not split),
    the fit inherits that same color so the line and its data remain
    visually linked.
 
    Parameters
    ----------
    show_branches : bool
        True when cooldown/warmup are drawn in different colors.
    dataset_color : str
        The color used to draw the scatter data for this dataset.
    """
    return "k" if show_branches else dataset_color

def _scale_fit_for_plot(fit, R_ref):
    """Return a copy of a fit dict whose polynomial is scaled by 1/R_ref.

    Used when overlaying a fit (computed on raw R in Ohm) onto a
    normalised R/R_ref plot. When R_ref == 1 the original dict is
    returned unchanged.
    """
    if R_ref == 1.0:
        return fit
    scaled = dict(fit)
    scaled["poly"] = np.poly1d([fit["slope"] / R_ref, fit["intercept"] / R_ref])
    return scaled

# ==================================================================
# CSV loading — one public entry point, one private loader per
# instrument format.  Adding a new format later means writing a new
# _load_csv_<name> function and adding its key to load_csv's dispatch
# dict; nothing else in the module needs to change.
# ==================================================================

def _load_csv_ppms(csv_path):
    """Load a PPMS / MultiVu-style measurement CSV.

    Rows are sorted by 'Time Stamp (sec)' if that column is present,
    so downstream functions can assume chronological order.
    Column names are returned as-is (PPMS naming conventions).
    """
    df = pd.read_csv(csv_path)
    if "Time Stamp (sec)" in df.columns:
        df = df.sort_values("Time Stamp (sec)").reset_index(drop=True)
    return df

def _load_csv_rack(csv_path, mode="rt", **kwargs):
    """Unified loader for rack data. Modes are cleanly separated."""
    df = pd.read_csv(csv_path)

    if mode == "rt":
        # R(T) mode - needs current scaling
        signal_col = kwargs.get("signal_col")
        current = kwargs.get("current")
        if current is None:
            raise ValueError("R(T) mode requires 'current' (source current in Amperes).")
        V_col = signal_col or f"R{kwargs.get('channel', 2)}"
        if V_col not in df.columns:
            raise ValueError(f"Column '{V_col}' not found for R(T).")
        return pd.DataFrame({
            "Temperature (K)": df["Tsample"].astype(float).values,
            "Bridge 1 Resistivity (Ohm)": df[V_col].astype(float).values / current,
            "Bridge 1 Std. Dev. (Ohm)": 0.0,
        })

    elif mode == "iv":
        # I(V) mode - no scaling needed, Is and Vdmm are direct
        channel_dV = kwargs.get("channel_dV", 2)
        channel_dI = kwargs.get("channel_dI", 1)
        dV_col = f"R{channel_dV}"
        dI_col = f"X{channel_dI}"
        
        return pd.DataFrame({
            "Current (A)": df["Is"].astype(float).values,
            "Voltage (V)": df["Vdmm"].astype(float).values,
            "dV": df[dV_col].astype(float).values if dV_col in df.columns else np.nan,
            "dI": df[dI_col].astype(float).values if dI_col in df.columns else np.nan,
            "Tsample": df["Tsample"].astype(float).values,
        })

    else:
        raise ValueError(f"Unknown mode: {mode}")

_LOADERS = {
    "ppms": _load_csv_ppms,
    "rack": _load_csv_rack,
}

def load_csv(csv_path, source="ppms", current=None, channel=2, signal_col=None):
    """Load a transport-measurement CSV file from any supported instrument."""
    if source not in _LOADERS:
        raise ValueError(f"Unknown source '{source}'.")
    if source == "rack":
        if current is None:
            raise ValueError("source='rack' requires the 'current' parameter.")
        if signal_col is None:
            signal_col = f"R{channel}"
        return _load_csv_rack(csv_path, signal_col=signal_col, current=current)
    return _load_csv_ppms(csv_path)

def parse_hall_mr_comments(comments_path):
    """Parse a Hall/MR measurement comments .txt file.
 
    The comments file is produced alongside the .mat/.csv data and
    records which LIA channel measures the Hall vs. longitudinal voltage,
    the input resistance, AC frequency, sensitivity and time constant.
 
    Parameters
    ----------
    comments_path : str
        Full path to the *_comments.txt file.
 
    Returns
    -------
    dict with keys (where parseable):
        'hall_lia'    : int, LIA number measuring Hall (transverse) voltage
        'mr_lia'      : int, LIA number measuring longitudinal (MR) voltage
        'R_input_Ohm' : float, input resistance (Ω) → I = V_source / R_input
        'freq_Hz'     : float, LIA AC excitation frequency (Hz)
        'sensitivity_V': float, LIA voltage sensitivity (V)
        'time_const_s' : float, LIA time constant (s)
        'raw'         : list[str], every line verbatim (for manual inspection)
 
    Notes
    -----
    The parser uses flexible regex matching; it does not require a fixed
    format.  Lines it cannot parse are silently stored under 'raw'.
    """
    info = {'raw': []}
    if not os.path.isfile(comments_path):
        warnings.warn(
            f"Comments file not found: {comments_path}. "
            "Proceeding without metadata.", UserWarning, stacklevel=2
        )
        return info
 
    with open(comments_path, 'r') as fh:
        for line in fh:
            info['raw'].append(line.rstrip())
 
    for line in info['raw']:
        lo = line.lower()
        # Hall LIA assignment e.g. "LIA1 → Hall" or "Hall: LIA 2"
        m = re.search(r'lia\s*(\d)\s*[\-\→:]+\s*hall', lo)
        if m:
            info['hall_lia'] = int(m.group(1))
        m = re.search(r'hall\s*[\-\→:]+\s*lia\s*(\d)', lo)
        if m:
            info['hall_lia'] = int(m.group(1))
        # MR / longitudinal LIA assignment
        m = re.search(r'lia\s*(\d)\s*[\-\→:]+\s*(mr|long|magneto|xx|rxx)', lo)
        if m:
            info['mr_lia'] = int(m.group(1))
        m = re.search(r'(mr|long|magneto|xx|rxx)\s*[\-\→:]+\s*lia\s*(\d)', lo)
        if m:
            info['mr_lia'] = int(m.group(2))
        # Input resistance
        m = re.search(r'input\s*resist[a-z]*\s*[:\=]\s*([\d\.eE\+\-]+)', lo)
        if m:
            try:
                info['R_input_Ohm'] = float(m.group(1))
            except ValueError:
                pass
        # Frequency
        m = re.search(r'freq[a-z]*\s*[:\=]\s*([\d\.eE\+\-]+)', lo)
        if m:
            try:
                info['freq_Hz'] = float(m.group(1))
            except ValueError:
                pass
 
    return info

def _remove_channel_spikes(df, sigma_thresh=5.0, min_channels=3):
    """Detect and remove isolated spikes common to multiple lock-in channels.

    A "circuit spike" appears as a single-point outlier that is anomalously
    far from its two immediate neighbours compared with the typical step size,
    AND this anomaly is consistent across at least `min_channels` of the R,
    X, and theta columns. Detected spike values are replaced by linear
    interpolation from the neighbouring points.

    Detection metric for channel array x at index i:
        spike_score[i] = |x[i] − (x[i−1] + x[i+1])/2| / median(|Δx|)

    Cross-channel consistency ensures instrument noise in one channel is not
    mistaken for a real spike; genuine circuit artefacts affect every channel
    simultaneously at the same field value.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw rack CSV (from pd.read_csv), before column extraction.
    sigma_thresh : float
        spike_score threshold (default 5). Lower → more aggressive removal.
    min_channels : int
        Minimum number of channels that must flag the same index for it to
        be classified as a spike (default 2).

    Returns
    -------
    pandas.DataFrame
        Cleaned copy of df with spike indices linearly interpolated.
    """
    df = df.copy()
    n = len(df)
    if n < 3:
        return df

    probe_cols = [c for c in
                  ['X1', 'R1', 'theta1', 'X2', 'R2', 'theta2',
                   'X3', 'R3', 'theta3']
                  if c in df.columns]
    if not probe_cols:
        return df

    spike_votes = np.zeros(n, dtype=int)

    for col in probe_cols:
        x = df[col].to_numpy(dtype=float)
        if np.all(np.isnan(x)) or np.all(x == x[0]):
            continue                        # constant / all-NaN: skip

        # Expected value at each interior point = midpoint of its neighbours
        mid = np.empty(n)
        mid[:] = np.nan
        mid[1:-1] = (x[:-2] + x[2:]) / 2.0

        deviation = np.abs(x - mid)
        deviation[0] = 0.0                 # endpoints cannot be assessed
        deviation[-1] = 0.0

        # Robust scale: median of |first differences| (ignores NaN)
        diffs = np.abs(np.diff(x[np.isfinite(x)]))
        if len(diffs) == 0:
            continue
        scale = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
        if scale < 1e-30:
            continue

        spike_votes += (deviation / scale > sigma_thresh).astype(int)

    spike_idx = np.where(spike_votes >= min_channels)[0]
    # Exclude endpoints (cannot interpolate them)
    spike_idx = spike_idx[(spike_idx > 0) & (spike_idx < n - 1)]

    if len(spike_idx) > 0:
        H_col = 'Hsample' if 'Hsample' in df.columns else None
        H_vals = (df[H_col].to_numpy()[spike_idx] if H_col else spike_idx)
        print(f"  [spikes] {len(spike_idx)} spike(s) detected and removed "
              f"at H ≈ {H_vals.tolist()} "
              f"({'Oe' if H_col else 'index'})")
        # Interpolate ALL columns at spike positions (not just probe channels)
        for col in df.columns:
            x = df[col].to_numpy(dtype=float)
            for idx in spike_idx:
                x[idx] = (x[idx - 1] + x[idx + 1]) / 2.0
            df[col] = x

    return df
 
def load_hall_mr_csv(csv_path, hall_n, mr_n, hall_col='X', mr_col='X'):
    """Load one Hall/MR field-sweep CSV file from the custom rack.
 
    Each file contains one complete sweep direction
    (−Hmax→+Hmax  or  +Hmax→−Hmax).  Forward and backward sweeps are
    stored in separate files; use two calls and pass both results to
    ``analyze_hall_mr``.
 
    Parameters
    ----------
    csv_path : str
    hall_n : int
        LIA channel number (1–3) measuring the Hall (transverse) voltage.
    mr_n : int
        LIA channel number measuring the longitudinal (MR) voltage.
    hall_col : {'X', 'R'}
        'X' → in-phase component (recommended); 'R' → magnitude.
    mr_col : {'X', 'R'}
 
    Returns
    -------
    dict
        'H_Oe'         : ndarray, field in Oe (chronological)
        'Vy_V'         : ndarray, Hall voltage (V)
        'Vx_V'         : ndarray, longitudinal voltage (V)
        'theta_H_deg'  : ndarray, Hall LIA phase (degrees)
        'theta_MR_deg' : ndarray, MR LIA phase (degrees)
        'T_K'          : ndarray, sample temperature (K)
        'sweep_dir'    : 'forward' or 'backward'
        'H_max_Oe'     : float, maximum |H| recorded
        'n_pts'        : int
        'hall_col_name', 'mr_col_name', 'csv_path'
    """
    df = pd.read_csv(csv_path)
    df = _remove_channel_spikes(df)
 
    hall_col_name = f"{hall_col}{hall_n}"
    mr_col_name   = f"{mr_col}{mr_n}"
    ph_hall_name  = f"theta{hall_n}"
    ph_mr_name    = f"theta{mr_n}"
 
    for col in (hall_col_name, mr_col_name, 'Tsample', 'Hsample'):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {csv_path}. "
                f"Available columns: {df.columns.tolist()}"
            )
 
    H  = df['Hsample'].to_numpy(dtype=float)
    Vy = df[hall_col_name].to_numpy(dtype=float)
    Vx = df[mr_col_name].to_numpy(dtype=float)
    T  = df['Tsample'].to_numpy(dtype=float)
    th_H  = (df[ph_hall_name].to_numpy(dtype=float)
             if ph_hall_name in df.columns else np.full(len(H), np.nan))
    th_MR = (df[ph_mr_name].to_numpy(dtype=float)
             if ph_mr_name in df.columns else np.full(len(H), np.nan))
 
    sweep_dir = 'forward' if np.nanmean(np.diff(H)) > 0 else 'backward'
 
    return {
        'H_Oe':          H,
        'Vy_V':          Vy,
        'Vx_V':          Vx,
        'theta_H_deg':   th_H,
        'theta_MR_deg':  th_MR,
        'T_K':           T,
        'sweep_dir':     sweep_dir,
        'H_max_Oe':      float(np.nanmax(np.abs(H))),
        'n_pts':         len(H),
        'hall_col_name': hall_col_name,
        'mr_col_name':   mr_col_name,
        'csv_path':      csv_path,
    }

# ==========================================================================
# 1. PHYSICAL CONSTANTS — Hall / MR
#    Add this block after the existing _LOADERS dict definition.
# ==========================================================================
 
_HALL_e        = 1.602176634e-19   # C
_HALL_mu_B     = 9.2740100783e-24  # J/T (Bohr magneton)
_HALL_Oe_to_T  = 1e-4              # 1 Oe = 1e-4 T (free-space convention)
_HALL_A3_to_m3 = 1e-30             # Å³ → m³
_HALL_Ohm_m_to_uOhm_cm = 1e8      # Ω·m → µΩ·cm (standard cuprate unit)
 
# Default V_cell/Z for Bi-2201 (Bi₂Sr₂CuO₆₊δ), tetragonal subcell Z=2
# a ≈ 3.79 Å, c ≈ 24.6 Å → V ≈ 353 Å³, V/Z ≈ 177 Å³/Cu
# Ref: Nat. Commun. 2025, arXiv:2411.06603; pull lattice params from your
# crystal's own characterisation for better precision (±25–30% otherwise).
_BI2201_VCELL_PER_Z_A3 = 177.5    # Å³ per Cu

# ==================================================================
# R(T): resistance vs. temperature
# ==================================================================

def _RT_columns(bridge):
    """Column names used by the instrument for a given bridge number.

    Bridge 1 was acquired with area = 1, so its 'Resistivity' column
    is already a resistance in Ohm. Other bridges (if ever used) are
    logged in Ohm-m and would need an actual area/length to convert
    to a resistance.
    """
    if bridge == 1:
        return ("Bridge 1 Resistivity (Ohm)", "Bridge 1 Std. Dev. (Ohm)")
    return (f"Bridge {bridge} Resistivity (Ohm-m)",
            f"Bridge {bridge} Std. Dev. (Ohm-m)")

def analyze_RT(data_source, bridge=1, area_correction=1.0,
                split_branches=True, dropna=True, skip_points=0,
                source="ppms", current=None, channel=2, signal_col=None):
    """Extract R(T) data from a transport CSV file.

    Parameters
    ----------
    data_source : str or pandas.DataFrame
        Path to the .csv file, or an already-loaded DataFrame
        (e.g. from `load_csv`) if you want to reuse it for several
        analyses without re-reading the file.
    bridge : int
        Which bridge channel to use (1-4). Only bridge 1 is currently
        wired up to area = 1 in the acquisition software; see
        `_RT_columns`.
    area_correction : float
        Multiplicative factor applied to the raw column value, in
        case you ever need to rescale a resistivity into a
        resistance (R = rho * area_correction). Defaults to 1.0,
        i.e. no correction, which is correct for bridge 1.
    split_branches : bool
        If True, the sweep is split into a 'cooldown' and a 'warmup'
        branch using the index of the global temperature minimum.
        This assumes a single cool-down + warm-up cycle (as in the
        uploaded file); set to False for simpler sweeps or to skip
        the splitting.
    dropna : bool
        Drop rows where resistance or temperature is missing
        (e.g. bridges that were not actually connected/measured).
    skip_points : int
        Number of points to discard from the *start* of the
        chronological run before any other processing. Use this to
        drop the initial excitation-current calibration points
        (taken while searching for a good SNR, all at the starting
        temperature) so they don't show up as a cluster of noisy
        points in the plot. Inspect 'Bridge N Excitation (uA)' in the
        raw file to figure out how many points that calibration took.

    Returns
    -------
    dict with keys:
        'T'      : ndarray, temperature (K)
        'R'      : ndarray, resistance (Ohm)
        'dR'     : ndarray, standard deviation of R (Ohm)
        'branch' : ndarray of str, 'cooldown'/'warmup' for each point
                   (only present if split_branches=True)
        'bridge' : int, the bridge number used
    """
    df = (data_source if isinstance(data_source, pd.DataFrame)
        else load_csv(data_source, source=source,
                    current=current, channel=channel, signal_col=signal_col))

    res_col, err_col = _RT_columns(bridge)

    cols = ["Temperature (K)", res_col, err_col]
    sub = df[cols].copy()
    if skip_points:
        sub = sub.iloc[skip_points:].reset_index(drop=True)
    if dropna:
        sub = sub.dropna(subset=[res_col]).reset_index(drop=True)

    # Reference point for normalization: prefer a high-T point (normal state)
    # Fall back to first point if no high-T data is available
    T = sub["Temperature (K)"].to_numpy(dtype=float)
    R = sub[res_col].to_numpy(dtype=float) * area_correction
    dR = sub[err_col].to_numpy(dtype=float) * area_correction

    if len(T) > 0:
        # Try to find a reference near the highest temperature (normal state)
        T_max = float(np.max(T))
        T_ref_candidate = int(round(T_max))
        mask_ref = np.abs(T - T_ref_candidate) <= 2.0  # wider window for sparse data
        if np.any(mask_ref):
            R_ref = float(np.mean(R[mask_ref]))
            T_ref = T_ref_candidate
        else:
            # Fallback to first point
            T_ref = int(round(float(T[0])))
            mask_ref = np.abs(T - T_ref) <= 1.0
            R_ref = float(np.mean(R[mask_ref])) if np.any(mask_ref) else float(R[0])
    else:
        T_ref = 0
        R_ref = 1.0
    R_norm  = R  / R_ref
    dR_norm = dR / R_ref

    result = {
        "T": T, "R": R, "dR": dR,
        "T_ref": T_ref, "R_ref": R_ref,
        "R_norm": R_norm, "dR_norm": dR_norm,
        "bridge": bridge,
        "source": source,
        "signal_col": signal_col,
    }

    if split_branches and len(T) > 0:
        i_min = int(np.argmin(T))
        branch = np.full(T.shape, "warmup", dtype=object)
        branch[:i_min + 1] = "cooldown"
        result["branch"] = branch

    return result

def _draw_RT(ax, T, R, dR, show_errorbars, color, marker, markersize,
             label, **kwargs):
    """Low-level draw step shared by both branches/no-branches cases."""
    ms = markersize if markersize is not None else plt.rcParams["lines.markersize"]
    if show_errorbars:
        ax.errorbar(T, R, yerr=dR, fmt=marker, color=color, ms=ms,
                    label=label, **kwargs)
    else:
        ax.plot(T, R, marker, color=color, ms=ms, label=label, **kwargs)

def plot_RT(data, ax=None, show_errorbars=False, show_branches=False,
            normalized=False, color="k", branch_colors=None, marker="o",
            markersize=None, label=None, xlabel=r"Temperature (K)",
            ylabel=None, legend=None, **kwargs):
    """Plot R(T) data produced by `analyze_RT`.

    Parameters
    ----------
    data : dict
        Output of `analyze_RT`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if omitted.
    show_errorbars : bool
        Draw error bars from `data['dR']` if True, otherwise a plain
        marker/line plot.
    show_branches : bool
        If True (and `data` contains branch info), the cooldown and
        warmup branches are drawn separately with different colors,
        which is useful to check for thermal hysteresis. If False,
        all points are drawn together in one color, in their
        original chronological order (so the cooldown->warmup loop
        is still traced correctly if you connect the markers).
    normalized : bool
        If True, plot R/R(T_ref) instead of R, where T_ref is the
        initial measurement temperature rounded to the nearest integer
        (stored in data['T_ref'] by analyze_RT). The y-axis label is
        updated automatically to show the reference temperature.
    color : str
        Color used when show_branches=False.
    branch_colors : dict, optional
        {'cooldown': color, 'warmup': color}, used when
        show_branches=True. Defaults to `_default_RT_branch_colors()`.
    marker : str
        Marker/format string passed to errorbar/plot.
    markersize : float, optional
        Overrides rcParams['lines.markersize'] if given.
    label : str, optional
        Legend label (or label prefix when show_branches=True).
    xlabel, ylabel : str
        Axis labels (override if you want different units/wording).
    legend : bool, optional
        Whether to draw a legend. Defaults to True if a label was
        given, False otherwise.
    **kwargs :
        Passed through to ax.errorbar / ax.plot for extra
        customization (e.g. linestyle, alpha, zorder...).

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots()

    # Select which arrays to draw based on normalization flag
    R_key  = "R_norm"  if normalized else "R"
    dR_key = "dR_norm" if normalized else "dR"

    if ylabel is None:
        if normalized:
            T_ref = data.get("T_ref", "?")
            ylabel = rf"$R/R({T_ref}\,\mathrm{{K}})$"
        else:
            ylabel = r"Resistance ($\Omega$)"

    if show_branches and "branch" in data:
        colors = branch_colors or _default_RT_branch_colors()
        for branch_name in ("cooldown", "warmup"):
            mask = data["branch"] == branch_name
            if not np.any(mask):
                continue
            lbl = f"{label} ({branch_name})" if label else branch_name.capitalize()
            _draw_RT(ax, data["T"][mask], data[R_key][mask], data[dR_key][mask],
                      show_errorbars, colors[branch_name], marker, markersize,
                      lbl, **kwargs)
    else:
        _draw_RT(ax, data["T"], data[R_key], data[dR_key], show_errorbars,
                  color, marker, markersize, label, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    draw_legend = legend if legend is not None else bool(label) or show_branches
    if draw_legend:
        ax.legend()

    if created_fig:
        fig.tight_layout()

    return ax

# ==================================================================
# I(V) and dV/dI: Loading and Plotting
# ==================================================================

def analyze_IV_dVdI(data_source, channel_dV=2, channel_dI=1, source="rack", current=None):
    """Extract I(V) and dV/dI data from rack CSV.

    Returns a dict with:
        'I': current array
        'V': voltage array
        'dV': lock-in dV signal
        'dI': lock-in dI signal
        'dVdI': computed differential resistance (dV/dI)
        'T': temperature (usually constant)
    """
    df = _load_csv_rack(data_source, mode="iv", channel_dV=channel_dV, channel_dI=channel_dI)

    I = df["Current (A)"].to_numpy(dtype=float)
    V = df["Voltage (V)"].to_numpy(dtype=float)
    dV = df["dV"].to_numpy(dtype=float)
    dI = df["dI"].to_numpy(dtype=float)
    T = df["Tsample"].to_numpy(dtype=float)

    # Compute dV/dI (handle small dI)
    with np.errstate(divide='ignore', invalid='ignore'):
        dVdI = np.where(np.abs(dI) > 1e-9, dV / dI, np.nan)

    # Simple branch detection (forward vs backward sweep)
    branch = np.full(len(I), "forward", dtype=object)
    diffs = np.diff(I)
    if np.any(diffs < 0):
        # has backward sweep
        i_turn = np.where(diffs < 0)[0][0] + 1
        branch[i_turn:] = "backward"

    is_bf = np.any(diffs < 0) if 'diffs' in locals() else False

    return {
        "I": I,
        "V": V,
        "dV": dV,
        "dI": dI,
        "dVdI": dVdI,
        "T": T,
        "branch": branch,
        "source": source,
        "is_bf": is_bf,
    }

def load_multi_iv_files(pattern_or_list, channel_dV=2, channel_dI=1):
    import glob
    if isinstance(pattern_or_list, str):
        expanded = os.path.expanduser(pattern_or_list)
        files = glob.glob(expanded)
    else:
        files = pattern_or_list if isinstance(pattern_or_list, list) else [pattern_or_list]
    
    files = [f for f in files if f.endswith('.csv')]
    
    datasets = []
    for f in sorted(files):
        match = re.search(r'_(\d+\.?\d*)K_', f)
        T = float(match.group(1)) if match else None
        data = analyze_IV_dVdI(f, channel_dV=channel_dV, channel_dI=channel_dI)
        data['T_mean'] = T or np.mean(data['T'])
        data['filename'] = f
        datasets.append(data)

    if datasets:
        bf_statuses = [d.get('is_bf', False) for d in datasets]
        if not all(x == bf_statuses[0] for x in bf_statuses):
            raise ValueError("All files must be either Forward-only or BF measurements. Mixed types detected.")
        print(f"Loaded {len(datasets)} files. BF mode: {bf_statuses[0]}")
    
    return datasets

def plot_IV_dVdI(data, ax=None, plot_type="iv", show_branches=True, color="k", 
                 marker="o", markersize=None, label=None, figsize=None, **kwargs):
    """Plot I(V) related quantities with paper style and figsize support."""
    set_paper_style()

    created_fig = ax is None
    if created_fig:
        if figsize is None:
            figsize = (8.6, 6.0)
        fig, ax = plt.subplots(figsize=(figsize[0]/2.54, figsize[1]/2.54))

    # Unit scaling
    I_ma = data["I"] * 1e3
    V_mv = data["V"] * 1e3
    dV_uv = data["dV"] * 1e6
    dI_ma = data["dI"] * 1e3

    if plot_type == "iv":
        x, y = I_ma, V_mv
        xlabel = "Current (mA)"
        ylabel = "Voltage (mV)"
    elif plot_type == "dvdI":
        x, y = I_ma, data["dVdI"] * 1000   # Ohm → mΩ
        xlabel = "Current (mA)"
        ylabel = r"dV/dI (m$\Omega$)"
    elif plot_type == "dv":
        x, y = I_ma, dV_uv
        xlabel = "Current (mA)"
        ylabel = "dV ($\mu$V)"
    elif plot_type == "di":
        x, y = I_ma, dI_ma
        xlabel = "Current (mA)"
        ylabel = "dI (mA)"
    elif plot_type == "t":
        x, y = I_ma, data["T"]
        xlabel = "Current (mA)"
        ylabel = "Temperature (K)"
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    if show_branches and "branch" in data:
        colors = {"forward": color, "backward": "tab:red"}
        for b_name in ["forward", "backward"]:
            mask = data["branch"] == b_name
            if not np.any(mask):
                continue
            lbl = b_name.capitalize()
            ax.plot(x[mask], y[mask], marker, color=colors.get(b_name, color), 
                    ms=markersize or 3, label=lbl, **kwargs)
    else:
        ax.plot(x, y, marker, color=color, ms=markersize or 3, label=label, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction='in', top=True, right=True, labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    if label or show_branches:
        ax.legend(frameon=False, fontsize=8)
    if created_fig:
        fig.tight_layout()
    return ax

def plot_iv_diagnostics(data, base_name="", figsize=(8.6, 6.0)):
    set_paper_style()
    types = ['iv', 'dvdI', 'dv', 'di', 't']
    for t in types:
        fig, ax = plt.subplots(figsize=(figsize[0]/2.54, figsize[1]/2.54))
        plot_IV_dVdI(data, ax=ax, plot_type=t, show_branches=True, figsize=figsize)
        fig.savefig(f"{base_name}_{t}.pdf")
        plt.close(fig)
    print(f"Saved diagnostic plots for {base_name}")

def plot_multi_temp_iv(datasets, output_prefix="multi_temp", figsize=(8.6, 6.0)):
    """Create I(V) overlay and 2D dV/dI map for multiple temperatures."""
    set_paper_style()
    
    is_bf = datasets[0].get('is_bf', False) if datasets else False

    # 1. I(V) overlay plot(s)
    if not is_bf:
        # Forward-only case - same as before
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
        
        for i, data in enumerate(datasets):
            T = data['T_mean']
            label = f"{T:.1f} K"
            color = colors[i]
            I_ma = data['I'] * 1e3
            V_mv = data['V'] * 1e3
            ax.plot(I_ma, V_mv, color=color, label=label)
        
        ax.set_xlabel("Current (mA)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("I(V) at different temperatures")
        ax.legend(title="Temperature")
        fig.tight_layout()
        fig.savefig(f"{output_prefix}_IV_overlay.pdf")
        plt.close(fig)
    else:
        # BF case: separate Forward and Backward plots
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
        
        # Forward sweep
        fig, ax = plt.subplots(figsize=figsize)
        for i, data in enumerate(datasets):
            T = data['T_mean']
            label = f"{T:.1f} K"
            color = colors[i]
            I_ma = data['I'] * 1e3
            V_mv = data['V'] * 1e3
            ax.plot(I_ma, V_mv, color=color, label=label)
        
        ax.set_xlabel("Current (mA)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("I(V) at different temperatures (Forward sweep)")
        ax.legend(title="Temperature")
        fig.tight_layout()
        fig.savefig(f"{output_prefix}_IV_forward.pdf")
        plt.close(fig)
        
        # Backward sweep
        fig, ax = plt.subplots(figsize=figsize)
        for i, data in enumerate(datasets):
            T = data['T_mean']
            label = f"{T:.1f} K"
            color = colors[i]
            I_ma = data['I'] * 1e3
            V_mv = data['V'] * 1e3
            ax.plot(I_ma, V_mv, color=color, label=label)
        
        ax.set_xlabel("Current (mA)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("I(V) at different temperatures (Backward sweep)")
        ax.legend(title="Temperature")
        fig.tight_layout()
        fig.savefig(f"{output_prefix}_IV_backward.pdf")
        plt.close(fig)
    
    # 2. 2D dV/dI intensity map
    fig, ax = plt.subplots(figsize=figsize)
    I_ma = datasets[0]["I"] * 1e3   # mA
    T_values = np.array([d['T_mean'] for d in datasets])
    dVdI_grid = np.zeros((len(T_values), len(I_ma)))
    for i, d in enumerate(datasets):
        dVdI_grid[i] = d["dVdI"] * 1e3  # mΩ
    
    im = ax.pcolormesh(I_ma, T_values, dVdI_grid, shading='auto', cmap='plasma')
    fig.colorbar(im, ax=ax, label=r'dV/dI (m$\Omega$)')
    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("dV/dI Intensity Map")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_dVdI_map.pdf")
    plt.close(fig)
    
    print(f"Saved multi-T plots: {output_prefix}_*.pdf")

def plot_2d_dvdi_map_from_single_file(filepath, channel_dV=2, channel_dI=1,
                                       n_T_bins=100, n_I_bins=200,
                                       clim_pct=(2, 98)):
    """Load a multi-T rack IV file and produce a 2D dV/dI colour map.

    Uses 2D binning (not a pivot) so it works correctly even when T drifts
    continuously during the current sweep, giving nearly-unique (T, I) pairs
    rather than a clean regular grid.
    """
    from scipy.stats import binned_statistic_2d

    print("Loading multi-T file...")
    df = _load_csv_rack(filepath, mode="iv",
                        channel_dV=channel_dV, channel_dI=channel_dI)
    print(f"  {len(df):,} rows, "
          f"{df['Tsample'].nunique()} unique temperatures, "
          f"{df['Current (A)'].nunique()} unique currents")

    # ---- 1. Vectorised dV/dI ------------------------------------------------
    valid = (
        (np.abs(df["dI"]) > 1e-12)
        & df["dV"].notna()
        & df["dI"].notna()
    )
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with |dI| ≈ 0 or NaN")
    df = df.loc[valid].copy()
    df["dVdI"] = (df["dV"] / df["dI"]) * 1000   # mΩ

    T_vals    = df["Tsample"].values
    I_vals_mA = df["Current (A)"].values * 1e3   # mA for display
    dVdI_vals = df["dVdI"].values

    # ---- 2. 2D binning (replaces pivot_table) --------------------------------
    # pivot_table only fills cells that have an exact (T, I) match; when T
    # drifts continuously, every row has a unique pair, giving a grid that is
    # >99.9% NaN.  binned_statistic_2d aggregates nearby points into shared
    # bins, giving a dense, displayable grid.
    print(f"Binning into {n_T_bins} T × {n_I_bins} I grid...")
    Z, T_edges, I_edges, _ = binned_statistic_2d(
        T_vals, I_vals_mA, dVdI_vals,
        statistic="mean",
        bins=[n_T_bins, n_I_bins],
    )
    T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
    I_centers = 0.5 * (I_edges[:-1] + I_edges[1:])

    nan_frac = np.isnan(Z).mean()
    print(f"  NaN fraction in grid: {nan_frac:.1%}  "
          f"(if still high, reduce n_T_bins / n_I_bins)")

    vmin, vmax = np.nanpercentile(dVdI_vals, list(clim_pct))
    print(f"  dV/dI colour range "
          f"({clim_pct[0]}–{clim_pct[1]}th pct): {vmin:.3f} – {vmax:.3f} mΩ")

    # ---- 3. Plot ------------------------------------------------------------
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8.6 / 2.54, 6 / 2.54),
                            constrained_layout=True)

    # Build a masked array so NaN cells are transparent, not white
    Z_masked = np.ma.masked_invalid(Z)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")          # NaN → white (matches paper background)

    im = ax.pcolormesh(
        I_centers, T_centers, Z_masked,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        rasterized=True,                 # bitmap inside PDF → small file
        shading="nearest",
    )
    fig.colorbar(im, ax=ax, label=r"dV/dI (m$\Omega$)")
    ax.set_xlabel(r"Current (mA)")
    ax.set_ylabel("Temperature (K)")

    base = os.path.splitext(os.path.basename(filepath))[0]
    out = f"{base}_dVdI_map.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

# ==================================================================
# I(V) and dV/dI: Data Analysis and Numerical Calculations
# ==================================================================

from scipy.signal import find_peaks

import numpy as np
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d

# ==================================================================
# dI/dV analysis for superconducting gap extraction
#
# Physical picture
# ----------------
# In a current-biased V(I) measurement:
#   - I is the independent (swept) variable → uniform spacing
#   - V is the output → not uniform
#   - dV/dI  = differential resistance (high in resistive state, ~0 in SC state)
#   - dI/dV  = differential conductance (Riedel peaks appear here)
#
# The Riedel singularity (Josephson pair tunneling) appears as a peak
# in dI/dV at |V| = 2Δ/e.  For a weak-link / SNS junction, Multiple
# Andreev Reflection (MAR) produces additional features at |V| = 2Δ/(n·e)
# for n = 2, 3, ... The main gap is extracted from the n=1 peak (highest |V|).
#
# The superconducting branch (|V| ≈ 0 for |I| < Ic) must be excluded
# before computing dI/dV because 1/(dV/dI) → ∞ there.
# ==================================================================

def analyze_didv(data, smooth_window=11, smooth_poly=3,
                  sc_voltage_threshold_mV=0.1,
                  peak_prominence_rel=0.15,
                  n_mar_peaks=3):
    """Compute dI/dV from current-biased V(I) data and find gap features.

    Uses a Savitzky-Golay filter to simultaneously smooth and differentiate
    V(I) — the standard approach for noisy spectroscopy data. The Riedel
    peak (and MAR subharmonics) are located with scipy.signal.find_peaks on
    the resistive branch only.

    Parameters
    ----------
    data : dict
        Must contain:
          'I'      : ndarray, current in Amperes (the swept quantity)
          'V'      : ndarray, measured voltage in Volts
        Optionally:
          'branch' : ndarray of 'forward'/'backward' per point
          'is_bf'  : bool, True if both branches are present
    smooth_window : int, optional
        Savitzky-Golay window length (must be odd). Defaults to ~5% of
        the number of points (minimum 11). Increase for noisier data.
    smooth_poly : int
        Polynomial order for the SG filter (default 3). Must be less
        than smooth_window.
    sc_voltage_threshold_mV : float
        Points with |V| < this threshold (mV) are considered to be on
        the superconducting branch and are excluded from dI/dV
        (avoids the 1/(dV/dI≈0) divergence). Default: 0.1 mV.
    peak_prominence_rel : float
        Minimum peak prominence as a fraction of the maximum dI/dV on
        the resistive branch. Increase to suppress noise peaks (0–1).
    n_mar_peaks : int
        How many MAR sub-harmonic peaks to search for (n=1 is the
        main gap at 2Δ, n=2 at Δ, etc.). Default: 3.

    Returns
    -------
    dict with keys:
        'I_sorted', 'V_mV', 'dVdI_Ohm', 'dIdV_mA_per_mV'
            Arrays on the sorted-I grid (full data, including SC branch).
        'V_res_mV', 'dIdV_res'
            Arrays restricted to the resistive branch (|V| > threshold)
            where peak-finding is done.
        'peaks_V_mV', 'peaks_dIdV'
            Voltage and dI/dV values at each detected peak (sorted by
            decreasing |V|, i.e. n=1 first).
        'gap_V_mV'
            Estimated gap voltage = voltage of the highest-|V| peak.
            None if no peaks found.
        'Delta_meV'
            Estimated gap energy = gap_V_mV / 2 (for a symmetric S-I-S
            junction where V_gap = 2Δ/e). For MAR in an SNS junction
            this is still a reasonable first estimate. None if no peaks.
        'smooth_window'
            Actual window length used (for reproducibility reporting).
        'branches_processed' : list of branch names analyzed
    """
    from scipy.signal import savgol_filter, find_peaks

    I = np.asarray(data["I"], dtype=float)
    V = np.asarray(data["V"], dtype=float)
    branch_arr = np.asarray(data.get("branch", np.full(len(I), "all")))
    is_bf = data.get("is_bf", False)

    # ---- sort by current (SG filter requires uniform independent variable) --
    order = np.argsort(I)
    I_s = I[order]
    V_s = V[order]
    b_s = branch_arr[order]

    # Convert to working units
    V_mV = V_s * 1e3       # mV
    I_mA = I_s * 1e3       # mA

    # ---- Savitzky-Golay window selection ------------------------------------
    n = len(I_s)
    if smooth_window is None:
        smooth_window = max(11, n // 20)
    if smooth_window % 2 == 0:
        smooth_window += 1          # must be odd
    smooth_window = min(smooth_window, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if smooth_poly >= smooth_window:
        smooth_poly = smooth_window - 1

    # ---- dV/dI via SG derivative (smooth + differentiate simultaneously) ----
    # delta = spacing of the independent variable (I in mA)
    dI_step = float(np.median(np.diff(I_mA)))
    if abs(dI_step) < 1e-12:
        raise ValueError(
            "Current step size is effectively zero — check that 'I' is "
            "the swept (source) variable, not the measured one."
        )

    # savgol_filter(y, window, poly, deriv=1, delta=dx) returns dy/dx
    dVdI_mV_per_mA = savgol_filter(
        V_mV, smooth_window, smooth_poly,
        deriv=1, delta=abs(dI_step)
    )   # units: mV/mA = Ω

    # ---- restrict to resistive branch for dI/dV ----------------------------
    # In the SC branch dV/dI ≈ 0 → 1/(dV/dI) diverges and carries no
    # gap information; exclude those points.
    res_mask = np.abs(V_mV) > sc_voltage_threshold_mV
    V_res = V_mV[res_mask]
    dVdI_res = dVdI_mV_per_mA[res_mask]

    # Safe inversion: only where dV/dI is reliably non-zero
    safe = np.abs(dVdI_res) > 1e-6        # mV/mA = Ω; avoids divide-by-zero
    dIdV_res = np.full(dVdI_res.shape, np.nan)
    dIdV_res[safe] = 1.0 / dVdI_res[safe]  # 1/Ω = mA/mV (conductance)

    # ---- peak finding in dI/dV vs |V| (positive half only for symmetry) ----
    pos_mask = V_res > 0
    V_pos = V_res[pos_mask]
    dIdV_pos = dIdV_res[pos_mask]

    peaks_V, peaks_dIdV = np.array([]), np.array([])
    gap_V_mV = None
    Delta_meV = None

    if np.any(np.isfinite(dIdV_pos)):
        baseline = np.nanmedian(dIdV_pos)
        max_val  = np.nanmax(dIdV_pos)
        prominence_abs = peak_prominence_rel * (max_val - baseline)

        peak_idx, props = find_peaks(
            np.nan_to_num(dIdV_pos, nan=baseline),
            prominence=max(prominence_abs, 1e-9),
            distance=max(3, n // 100),      # minimum separation between peaks
        )

        if len(peak_idx) > 0:
            # Sort by decreasing V: n=1 (2Δ) is the rightmost peak
            order_p = np.argsort(V_pos[peak_idx])[::-1]
            peak_idx_sorted = peak_idx[order_p][:n_mar_peaks]
            peaks_V    = V_pos[peak_idx_sorted]
            peaks_dIdV = dIdV_pos[peak_idx_sorted]
            gap_V_mV   = float(peaks_V[0])        # highest-V peak = n=1
            Delta_meV  = gap_V_mV / 2.0           # Δ = e·V/2 for SIS

    branches = ["all"] if not is_bf else list(np.unique(b_s))

    return {
        "I_sorted":        I_s,
        "V_mV":            V_mV,
        "dVdI_Ohm":        dVdI_mV_per_mA,   # mV/mA = Ω
        "dIdV_mA_per_mV":  np.where(
                               np.abs(dVdI_mV_per_mA) > 1e-6,
                               1.0 / dVdI_mV_per_mA,
                               np.nan
                           ),
        "V_res_mV":        V_res,
        "dIdV_res":        dIdV_res,
        "peaks_V_mV":      peaks_V,
        "peaks_dIdV":      peaks_dIdV,
        "gap_V_mV":        gap_V_mV,
        "Delta_meV":       Delta_meV,
        "smooth_window":   smooth_window,
        "sc_threshold_mV": sc_voltage_threshold_mV,
        "branches_processed": branches,
    }

def plot_didv(result, ax=None, show_peaks=True, xlim=None,
               ylabel=r"d$I$/d$V$ (mA/mV)",
               xlabel="Voltage (mV)",
               color="tab:blue", peak_color="tab:red",
               legend=True):
    """Plot dI/dV vs V from the output of `analyze_didv`.

    Parameters
    ----------
    result : dict
        Output of `analyze_didv`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; new figure created if omitted.
    show_peaks : bool
        Mark the detected Riedel/MAR peaks with vertical lines.
    xlim : (float, float), optional
        x-axis limits in mV. Defaults to the full voltage range.
    color : str
        Colour for the dI/dV curve.
    peak_color : str
        Colour for peak markers.
    legend : bool
        Whether to draw a legend.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(constrained_layout=True)

    V_plot    = result["V_res_mV"]
    dIdV_plot = result["dIdV_res"]

    # Clip extreme outliers for display only (does not affect analysis)
    finite = np.isfinite(dIdV_plot)
    if np.any(finite):
        lo, hi = np.nanpercentile(dIdV_plot[finite], [1, 99])
        margin = (hi - lo) * 0.2
        dIdV_display = np.clip(dIdV_plot, lo - margin, hi + margin)
    else:
        dIdV_display = dIdV_plot

    ax.plot(V_plot, dIdV_display, color=color, lw=0.8, label="d$I$/d$V$")
    # Negative branch by symmetry
    ax.plot(-V_plot, dIdV_display, color=color, lw=0.8, alpha=0.5)

    if show_peaks and len(result["peaks_V_mV"]) > 0:
        for n_idx, (V_pk, dIdV_pk) in enumerate(
            zip(result["peaks_V_mV"], result["peaks_dIdV"]), start=1
        ):
            lbl = (rf"$2\Delta/e$ ≈ {V_pk:.3f} mV  →  $\Delta$ ≈ {V_pk/2:.3f} meV"
                   if n_idx == 1
                   else rf"MAR $n={n_idx}$: {V_pk:.3f} mV")
            ax.axvline( V_pk, color=peak_color, ls="--", lw=0.9, label=lbl)
            ax.axvline(-V_pk, color=peak_color, ls="--", lw=0.9)

        ax.axvspan(
            -result["sc_threshold_mV"], result["sc_threshold_mV"],
            color="gray", alpha=0.12, label="SC branch (excluded)"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if legend:
        ax.legend(fontsize=7, frameon=False)

    if created_fig:
        plt.tight_layout()

    return ax

def compute_iv_parameters(data, area_um2=1.0, rn_criterion=0.5, ic_span=10, outlier_thresh=5.0, advanced=False, diff_threshold=0.001, figsize=None):
    """Compute Ic, Jc, R_N, Jc*R_N from I(V) data."""
    I = data['I']
    V = data['V']
    dVdI = data.get('dVdI', None)
    is_bf = data.get('is_bf', False)
    branch = data.get('branch', np.full(len(I), "forward", dtype=object))
    
    results = {
        'T_K': round(np.mean(data.get('T', np.nan)),2),
        'is_bf': is_bf,
    }
    
    # Signal for transition detection
    signal = dVdI if dVdI is not None and not np.all(np.isnan(dVdI)) else np.abs(V)
    
    def find_ic_in_direction(I_seg, signal_seg, direction_sign=1, span=10, outlier_threshold=5.0, diff_threshold=0.001):
        """Outlier removal + only sum significant differences."""
        #signal_seg = gaussian_filter1d(signal_seg, sigma=2) #If the data is too noisy, e.g. near Tc
        if len(I_seg) < 2 * span:
            return abs(I_seg[-1])
        if direction_sign > 0:
            mask = I_seg > 0
        else:
            mask = I_seg < 0
        if not np.any(mask):
            return np.nan
        I_dir = I_seg[mask]
        sig_dir = signal_seg[mask]
        
        max_change = 0
        best_idx = 0
        for i in range(len(sig_dir) - span):
            window = sig_dir[i:i+span+1]
            # Outlier removal
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            if mad == 0:
                mad = 1e-10
            outliers = np.abs(window - median) > outlier_threshold * mad
            clean_window = window[~outliers]
            if len(clean_window) < 3:
                continue
            # Sum only significant differences
            diffs = np.abs(np.diff(clean_window))
            significant_diffs = diffs[diffs > diff_threshold]
            change = np.sum(significant_diffs) if len(significant_diffs) > 0 else 0
            if change > max_change:
                max_change = change
                best_idx = i + span // 2
        return abs(I_dir[best_idx])
    
    # Overall Ic+ and Ic-
    Ic_plus = find_ic_in_direction(I, signal, direction_sign=1, span=ic_span, outlier_threshold=outlier_thresh, diff_threshold=diff_threshold)
    Ic_minus = find_ic_in_direction(I, signal, direction_sign=-1, span=ic_span, outlier_threshold=outlier_thresh, diff_threshold=diff_threshold)
    
    results['Ic+_mA'] = Ic_plus * 1000
    results['Ic-_mA'] = Ic_minus * 1000
    results['Ic_mA'] = (Ic_plus + Ic_minus) / 2 * 1000 if not np.isnan(Ic_plus) and not np.isnan(Ic_minus) else np.nan
    
    if is_bf:
        # Per-branch analysis
        fwd = branch == "forward"
        bwd = branch == "backward"
        
        # Forward branch
        Ic_plus_fwd = find_ic_in_direction(I[fwd], signal[fwd], direction_sign=1, span=ic_span, outlier_threshold=outlier_thresh, diff_threshold=diff_threshold) if np.any(fwd) else np.nan
        Ic_minus_fwd = find_ic_in_direction(I[fwd], signal[fwd], direction_sign=-1, span=ic_span, outlier_threshold=outlier_thresh, diff_threshold=diff_threshold) if np.any(fwd) else np.nan
        
        # Backward branch
        Ic_plus_bwd = find_ic_in_direction(I[bwd], signal[bwd], direction_sign=1, span=ic_span, outlier_threshold=outlier_thresh, diff_threshold=diff_threshold) if np.any(bwd) else np.nan
        Ic_minus_bwd = find_ic_in_direction(I[bwd], signal[bwd], direction_sign=-1, span=ic_span, outlier_threshold=outlier_thresh, diff_threshold=diff_threshold) if np.any(bwd) else np.nan
        
        results['Ic+_f_mA'] = Ic_plus_fwd * 1000
        results['Ic-_f_mA'] = Ic_minus_fwd * 1000
        results['Ic+_b_mA'] = Ic_plus_bwd * 1000
        results['Ic-_b_mA'] = Ic_minus_bwd * 1000
    
    # Jc
    results['Jc_kA/cm2'] = results.get('Ic_mA', np.nan) / (area_um2 * 1e-2) if not np.isnan(results.get('Ic_mA', np.nan)) else np.nan
    
    # R_N linear fits
    def fit_Rn(I_seg, V_seg, criterion):
        if len(I_seg) < 5:
            return np.nan
        V_max = np.max(np.abs(V_seg))
        high_bias = np.abs(V_seg) > criterion * V_max
        if np.sum(high_bias) < 3:
            return np.nan
        slope, _, _, _, _ = linregress(I_seg[high_bias], V_seg[high_bias])
        return slope
    
    # Positive side
    pos = I > 0
    results['Rn+_mOhm'] = fit_Rn(I[pos], V[pos], rn_criterion) * 1000 if np.any(pos) else np.nan
    
    # Negative side
    neg = I < 0
    results['Rn-_mOhm'] = fit_Rn(I[neg], V[neg], rn_criterion) * 1000 if np.any(neg) else np.nan
    
    results['Rn_mean_mOhm'] = np.nanmean([results['Rn+_mOhm'], results['Rn-_mOhm']])
    
    # Ic * R_N
    results['IcRn_mV'] = results['Ic_mA'] * results['Rn_mean_mOhm'] * 1e-3 if not np.isnan(results['Ic_mA']) and not np.isnan(results['Rn_mean_mOhm']) else np.nan
    # Jc * R_N
    results['JcRn_V/m2'] = results['Jc_kA/cm2'] * results['Rn_mean_mOhm'] * 1e-4 if not np.isnan(results['Jc_kA/cm2']) and not np.isnan(results['Rn_mean_mOhm']) else np.nan
    
    # === Advanced Analysis ===
    if advanced:
        # Diode efficiency
        if is_bf:
            asym_fwd = 100 * ((abs(results['Ic+_f_mA']) - abs(results['Ic-_b_mA'])) / (abs(results['Ic+_f_mA']) + abs(results['Ic-_b_mA'])))
            results['Diode_Efficiency_%'] = asym_fwd
        # Stewart-McCumber from retrapping current (BF only)
        if is_bf and 'Ic+_f_mA' in results and 'Ic+_b_mA' in results:
            Ic = results['Ic+_f_mA']
            Ir = results['Ic+_b_mA']
            if Ir > 1e-9:  # avoid division by zero
                beta_c = (4*Ic / (np.pi * Ir)) ** 2
                results['Beta_C'] = beta_c
                
                # Capacitance
                if not np.isnan(results.get('Rn_mean_mOhm', np.nan)):
                    Rn = results['Rn_mean_mOhm'] / 1000  # Ohm
                    Phi0 = 2.067833848e-15
                    C = (beta_c * Phi0) / (2 * np.pi * Ic*1e-3 * Rn**2)
                    results['C_fF'] = C * 1e15  # fF
                    results['C_uF/cm2'] = (C * 1e6) / (area_um2 * 1e-8)  # uF/cm² (area in um2 -> cm2)
        set_paper_style()

        result = analyze_didv(
            data,
            smooth_window=21,           # increase for noisier data
            sc_voltage_threshold_mV=0.1,
            peak_prominence_rel=0.15,
        )

        print(f"Gap voltage: {result['gap_V_mV']:.3f} mV")
        print(f"Gap energy Δ: {result['Delta_meV']:.3f} meV")
        print(f"All peaks (n=1,2,...): {result['peaks_V_mV']} mV")

        fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 6/2.54), constrained_layout=True)

        # Full range
        plot_didv(result, ax=axes[0])

        # Zoomed around gap
        if result["gap_V_mV"] is not None:
            plot_didv(result, ax=axes[1],
                    xlim=(-result["gap_V_mV"]*1.5, result["gap_V_mV"]*1.5))

        fig.savefig("didv_gap.pdf")
    return results

# ==================================================================
# Generic "zoom-in" helpers
#
# These are plot-type agnostic display helpers: they work with any
# analyze_*/plot_* pair, not just R(T). They never touch the analysis
# functions above -- they just call an existing plot_* function again
# with the x-axis (and matching y-axis) restricted to a sub-range.
# ==================================================================

def _autoscale_y_to_xrange(ax, x, y, xmin, xmax, pad_frac=0.1):
    """Rescale an Axes' y-limits to fit only the data whose x falls
    inside [xmin, xmax]. Purely cosmetic helper used by zoom panels."""
    mask = (x >= xmin) & (x <= xmax)
    if not np.any(mask):
        return
    ymin, ymax = float(np.min(y[mask])), float(np.max(y[mask]))
    span = ymax - ymin if ymax > ymin else (abs(ymax) if ymax else 1.0)
    pad = pad_frac * span
    ax.set_ylim(ymin - pad, ymax + pad)

def plot_zoomed(plot_func, data, x_key, y_key, xlim, ax=None,
                 autoscale_y=True, y_pad_frac=0.1, **plot_kwargs):
    """Draw `data` with `plot_func` restricted to a given x-range.

    This is the building block for "zoomed-in" figures (e.g. around
    Tc or T_CDW for an R(T) curve): it just calls an existing plot_*
    function (e.g. `plot_RT`) and then narrows the x-limits, rescaling
    the y-axis to fit only the data that remains visible.

    Parameters
    ----------
    plot_func : callable
        Any plot_* function with signature plot_func(data, ax=..., **kwargs),
        e.g. `plot_RT`.
    data : dict
        Output of the matching analyze_* function (e.g. `analyze_RT`).
    x_key, y_key : str
        Keys in `data` holding the x/y arrays. Only used to compute
        sensible y-limits for the zoom window (e.g. 'T'/'R' for R(T));
        never used by the analysis itself.
    xlim : (float, float)
        (xmin, xmax) window to zoom into, e.g. (4, 10) to zoom around
        a 7 K transition.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (e.g. one panel of a bigger figure). A new
        standalone figure/axes is created if omitted.
    autoscale_y : bool
        If True, the y-axis is rescaled to fit only the data inside
        `xlim` (instead of inheriting the full-range y-limits).
    y_pad_frac : float
        Fractional padding added above/below the autoscaled y-range.
    **plot_kwargs :
        Forwarded to `plot_func` (color, label, show_errorbars, ...).

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots()

    plot_func(data, ax=ax, **plot_kwargs)
    ax.set_xlim(*xlim)

    if autoscale_y:
        x = np.asarray(data[x_key])
        y = np.asarray(data[y_key])
        _autoscale_y_to_xrange(ax, x, y, *xlim, pad_frac=y_pad_frac)

    if created_fig:
        fig.tight_layout()

    return ax

# ==================================================================
# Linear fit helpers — generic analysis + display
#
# `fit_linear` is pure analysis (no matplotlib). It works for any
# analyze_* data dict, not just R(T): the same function will serve an
# Ohmic I(V) region check later.  `fit_linear_RT` is a thin R(T)-
# specific wrapper that picks the right keys. `plot_linear_fit` is
# the matching display helper: it overlays the fitted line on an
# existing Axes, extrapolated across the *full current x-limits* of
# that Axes (not just the fit window). That extrapolation is the key
# feature: fit R(T) in a clean normal-state window, then overlay onto
# a zoom around T_CDW — the CDW onset appears as a deviation from the
# straight line.
# ==================================================================

def fit_linear(data, x_key, y_key, x_range, err_key=None, weighted=True,
                branch=None):
    """Fit y = slope*x + intercept over a window of x.

    Parameters
    ----------
    data : dict
        An analyze_* output dict (e.g. from `analyze_RT`).
    x_key, y_key : str
        Keys in `data` holding the x/y arrays to fit, e.g. 'T'/'R'.
    x_range : (float, float)
        (xmin, xmax) window over which to perform the fit.
    err_key : str, optional
        Key in `data` holding the y-error array (e.g. 'dR'), used
        for inverse-sigma weighting if `weighted=True`.
    weighted : bool
        If True and `err_key` is given, weight each point by 1/dR,
        skipping any points with dR <= 0 or NaN. Falls back to an
        unweighted fit if no usable errors are found.
    branch : str, optional
        If `data` has a 'branch' entry (e.g. 'cooldown'/'warmup' from
        `analyze_RT`) and you want to fit only one of them, name it
        here. None uses all points in `x_range` regardless of branch.

    Returns
    -------
    dict with keys:
        'slope'          : float, fit slope (Ohm/K for R(T))
        'slope_err'      : float, 1-sigma uncertainty on slope
        'intercept'      : float, fit intercept (y at x=0, i.e. R(T=0))
        'intercept_err'  : float, 1-sigma uncertainty on intercept
        'n_points'       : int, number of points used in the fit
        'x_range'        : (float, float), the window actually used
        'x_fit', 'y_fit' : ndarrays of the x, y data points used
        'poly'           : numpy.poly1d, callable as poly(x) -> predicted y
    """
    x = np.asarray(data[x_key], dtype=float)
    y = np.asarray(data[y_key], dtype=float)

    mask = (x >= x_range[0]) & (x <= x_range[1]) & ~np.isnan(x) & ~np.isnan(y)
    if branch is not None and "branch" in data:
        mask &= (np.asarray(data["branch"]) == branch)

    x_fit, y_fit = x[mask], y[mask]
    if len(x_fit) < 2:
        raise ValueError(
            f"Not enough points in range {x_range} to fit a line "
            f"(found {len(x_fit)}). Check --fit-range and --fit-branch."
        )

    w = None
    if weighted and err_key is not None and err_key in data:
        dy = np.asarray(data[err_key], dtype=float)[mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(dy > 0, 1.0 / dy, 0.0)
        if not np.any(w > 0):
            w = None  # no usable errors -> fall back to unweighted

    coeffs, cov = np.polyfit(x_fit, y_fit, deg=1, w=w, cov=True)
    slope, intercept = coeffs
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    return {
        "slope":          float(slope),
        "slope_err":      float(slope_err),
        "intercept":      float(intercept),
        "intercept_err":  float(intercept_err),
        "n_points":       int(np.sum(mask)),
        "x_range":        tuple(x_range),
        "x_fit":          x_fit,
        "y_fit":          y_fit,
        "poly":           np.poly1d(coeffs),
    }

def fit_linear_RT(data, T_range, weighted=True, branch=None):
    """Linear fit of R(T) over a temperature window.

    Convenience wrapper around `fit_linear` that picks the right
    keys ('T', 'R', 'dR') for an `analyze_RT` output dict.

    Typical uses:
      - Fit the normal-state region just above Tc to verify Ohmic
        behaviour and read off the residual resistance R(T→0).
      - Fit a clean region away from T_CDW, then extrapolate the
        result onto a zoom panel around the transition to make the
        CDW onset visible as a deviation from the fitted line.

    Parameters
    ----------
    data : dict
        Output of `analyze_RT`.
    T_range : (float, float)
        (Tmin, Tmax) temperature window for the fit (in K).
    weighted : bool
        Weight each point by 1/dR (inverse-sigma). Recommended: the
        calibration-current points at the start of the run have much
        larger dR than the actual measurement points, so weighting
        prevents them from biasing the fit even if skip_points is not
        set large enough.
    branch : {'cooldown', 'warmup', None}
        Restrict the fit to one branch. Useful if cooldown and warmup
        show a hysteretic offset near a transition: fitting only one
        branch gives a cleaner baseline.

    Returns
    -------
    dict — see `fit_linear` for the full key list.
    """
    return fit_linear(data, x_key="T", y_key="R", x_range=T_range,
                       err_key="dR", weighted=weighted, branch=branch)

# ==================================================================
# Superconducting transition temperature
# ==================================================================

def _find_Tc_crossing(T_sorted, R_sorted, R_threshold):
    """Find T where R crosses R_threshold upward (SC → normal state).

    Operates on arrays already sorted by ascending T. Returns Tc
    rounded to 0.1 K, or None if no crossing is found.

    When noise causes multiple crossings (e.g. R flickering around
    zero in the SC state), the *highest-T* crossing is returned --
    that is the physical SC→normal transition, not a noise spike.
    """
    below = R_sorted < R_threshold
    # Indices where R goes from below to above threshold as T increases
    crossings = np.where(below[:-1] & ~below[1:])[0]
    if len(crossings) == 0:
        return None
    # Highest-T crossing = last entry in ascending-T-sorted array
    i = crossings[-1]
    T1, T2 = float(T_sorted[i]), float(T_sorted[i + 1])
    R1, R2 = float(R_sorted[i]), float(R_sorted[i + 1])
    if R2 == R1:
        Tc = (T1 + T2) / 2.0
    else:
        Tc = T1 + (T2 - T1) * (R_threshold - R1) / (R2 - R1)
    return round(Tc, 1)

def find_Tc_RT(data, criterion=0.5, T_normal_range=None):
    """Determine the superconducting transition temperature Tc from R(T).

    Tc is defined as the temperature at which R drops to `criterion`
    times the normal-state resistance R_normal. The crossing is found
    by linear interpolation between consecutive data points, giving
    0.1 K precision.

    If branch information is present ('cooldown' / 'warmup'), Tc is
    computed independently for each branch and the mean is reported.

    Common `criterion` values used in the literature:
      0.9  →  onset  (resistance just starts to drop)
      0.5  →  midpoint (default, most widely reported)
      0.1  →  near-zero (resistance almost fully suppressed)

    Parameters
    ----------
    data : dict
        Output of `analyze_RT`.
    criterion : float
        Fraction of R_normal that defines Tc (default 0.5).
    T_normal_range : (float, float), optional
        Temperature window (K) over which to compute the mean
        normal-state R, e.g. (8, 20) for NbSe2 just above Tc.
        **Strongly recommended**: if omitted, R_normal falls back to
        the mean R in the top quartile of the measured T range, which
        is only appropriate if the measurement starts in the normal
        state at a temperature well above Tc.

    Returns
    -------
    dict with keys:
        'Tc'             : float or None, mean Tc across branches (K)
        'Tc_branches'    : dict mapping branch name → Tc (K) or None
        'criterion'      : float, the criterion fraction used
        'R_normal'       : float, normal-state R used for the threshold
        'R_threshold'    : float, criterion × R_normal
        'T_normal_range' : tuple or None, the range used for R_normal
        'n_normal_points': int, number of points used to compute R_normal
    """
    T = np.asarray(data["T"], dtype=float)
    R = np.asarray(data["R"], dtype=float)

    # ---- determine R_normal ----
    if T_normal_range is not None:
        mask_n = (T >= T_normal_range[0]) & (T <= T_normal_range[1])
        if not np.any(mask_n):
            raise ValueError(
                f"No data points found in T_normal_range {T_normal_range}. "
                f"Data spans {T.min():.1f}–{T.max():.1f} K."
            )
    else:
        # Fallback: top quartile of the measured T range
        T_cutoff = np.percentile(T, 75)
        mask_n = T >= T_cutoff
        import warnings
        warnings.warn(
            "T_normal_range not specified; using mean R in the top quartile "
            f"of the measured T range (T ≥ {T_cutoff:.1f} K) as R_normal. "
            "Pass T_normal_range=(Tmin, Tmax) for a reliable result.",
            UserWarning, stacklevel=2,
        )

    R_normal = float(np.mean(R[mask_n]))
    n_normal = int(np.sum(mask_n))
    R_threshold = criterion * R_normal

    # ---- find Tc per branch ----
    Tc_branches = {}

    if "branch" in data:
        branch_arr = np.asarray(data["branch"])
        for branch_name in ("cooldown", "warmup"):
            mask_b = branch_arr == branch_name
            if not np.any(mask_b):
                continue
            T_b = T[mask_b]
            R_b = R[mask_b]
            order = np.argsort(T_b)
            Tc_branches[branch_name] = _find_Tc_crossing(
                T_b[order], R_b[order], R_threshold
            )
    else:
        order = np.argsort(T)
        Tc_branches["all"] = _find_Tc_crossing(T[order], R[order], R_threshold)

    valid = [v for v in Tc_branches.values() if v is not None]
    Tc_mean = round(float(np.mean(valid)), 1) if valid else None

    return {
        "Tc":              Tc_mean,
        "Tc_branches":     Tc_branches,
        "criterion":       criterion,
        "R_normal":        R_normal,
        "R_threshold":     R_threshold,
        "T_normal_range":  T_normal_range,
        "n_normal_points": n_normal,
    }

def plot_linear_fit(fit, ax, x_range=None, color="k", ls="--", lw=1.0,
                     label=None, show_fit_window=True, **kwargs):
    """Overlay a linear fit produced by `fit_linear` / `fit_linear_RT`
    onto an existing Axes.
 
    The line is drawn (and optionally extrapolated) over `x_range`.
    By default `x_range` matches the *current x-limits of the Axes*,
    so if you call this on a zoom panel after `plot_zoomed`, the fit
    is automatically stretched across the full zoom window -- even if
    it was only computed over a narrower temperature range. That
    extrapolation is what makes a CDW bump (or any other deviation
    from Ohmic behaviour) stand out clearly against the fitted
    normal-state baseline.
 
    Parameters
    ----------
    fit : dict
        Output of `fit_linear` or `fit_linear_RT`.
    ax : matplotlib.axes.Axes
        Axes to draw on (must already exist with sensible x-limits if
        you rely on the default x_range behaviour).
    x_range : (float, float), optional
        Range over which to draw the line. Defaults to the current
        x-limits of `ax`.
    color : str
        Colour for the fit line and the shaded fit-window band.
        When only a single data colour is used in the plot, pass the
        same colour so the fit stays visually linked to the data.
        When cooldown/warmup branches are shown with *different*
        colours, pass a neutral colour (e.g. 'k') so the fit is not
        confused with either branch. The helper `_fit_color(show_branches,
        dataset_color)` encodes this rule and is used by the CLI runner.
    ls : str
        Line style (default: '--' dashed, distinguishes the fit from
        the data markers).
    lw : float
        Line width.
    label : str, optional
        Legend label. Auto-generates a "slope / intercept" string if
        not given.
    show_fit_window : bool
        If True, lightly shade the x-window that was actually used to
        compute the fit, making it easy to see how far the line is
        being extrapolated beyond the fitted region.
    **kwargs :
        Forwarded to ax.plot.
 
    Returns
    -------
    ax
    """
    if x_range is None:
        x_range = ax.get_xlim()
 
    x = np.linspace(x_range[0], x_range[1], 300)
    y = fit["poly"](x)
 
    if label is None:
        s, ds = fit["slope"], fit["slope_err"]
        b, db = fit["intercept"], fit["intercept_err"]
        label = (rf"linear fit: $\alpha={s:.3g}\pm{ds:.1g}$ $\Omega$/K, "
                  rf"$R_0={b:.3g}\pm{db:.1g}$ $\Omega$")
 
    ax.plot(x, y, ls=ls, lw=lw, color=color, marker="none", label=label, **kwargs)
 
    if show_fit_window:
        # Clip the shaded band to the axes' current x-limits before
        # calling axvspan. Without clipping, a fit range that extends
        # beyond the zoom window (e.g. fit over 12–300 K on a 2–20 K
        # panel) causes matplotlib to silently expand the x-axis to
        # fit the full span, de-zooming the panel and making the fit
        # line invisible. The clip ensures axvspan never draws outside
        # what is already visible; if the fit window doesn't intersect
        # the zoom at all, nothing is drawn.
        view_lo, view_hi = ax.get_xlim()
        span_lo = max(fit["x_range"][0], view_lo)
        span_hi = min(fit["x_range"][1], view_hi)
        if span_lo < span_hi:
            ax.axvspan(span_lo, span_hi, color=color, alpha=0.08, lw=0,
                        label="_nolegend_")
 
    return ax

def compute_RRR_RT(data, T_low, window=0.5):
    """Compute the Residual Resistance Ratio (RRR) from R(T) data.

    RRR = R(T_high) / R(T_low)

    T_high is the initial measurement temperature (rounded to the
    nearest integer K), already stored in data['T_ref'] by
    `analyze_RT`. T_low is a user-supplied temperature just above Tc
    (e.g. 8 K for NbSe2 with Tc ≈ 7 K), rounded to the nearest
    integer for display. R at each temperature is computed as the mean
    over a symmetric window of ±`window` K to average out noise.

    Parameters
    ----------
    data : dict
        Output of `analyze_RT`. Must contain 'T', 'R', 'T_ref',
        'R_ref' (all produced by analyze_RT automatically).
    T_low : float
        Temperature just above Tc for the denominator, e.g. 8.0.
        Rounded to the nearest integer for display purposes.
    window : float
        Half-width in K of the averaging window around T_low
        (default: 0.5 K). Increase if few points fall near T_low.

    Returns
    -------
    dict with keys:
        'RRR'        : float, R(T_high) / R(T_low)
        'R_high'     : float, R at T_high (= data['R_ref'])
        'R_low'      : float, mean R in [T_low - window, T_low + window]
        'T_high'     : int, rounded initial temperature (K)
        'T_low'      : int, rounded T_low (K)
        'n_pts_low'  : int, number of points used to compute R_low
    """
    T = np.asarray(data["T"], dtype=float)
    R = np.asarray(data["R"], dtype=float)

    T_low_rounded = int(round(float(T_low)))
    mask_low = np.abs(T - T_low_rounded) <= window
    if not np.any(mask_low):
        raise ValueError(
            f"No data points found within {window} K of T_low = {T_low_rounded} K "
            f"(data spans {T.min():.1f}–{T.max():.1f} K). "
            f"Try a larger --rrr-window."
        )

    R_low      = float(np.mean(R[mask_low]))
    n_pts_low  = int(np.sum(mask_low))
    R_high     = data["R_ref"]
    T_high     = data["T_ref"]

    return {
        "RRR":       R_high / R_low,
        "R_high":    R_high,
        "R_low":     R_low,
        "T_high":    T_high,
        "T_low":     T_low_rounded,
        "n_pts_low": n_pts_low,
    }

# ==========================================================================
# 3. ANALYSIS HELPERS for Hall and MG measurement
# ==========================================================================

def _trim_H_plateau(H, *arrays, min_plateau_pts=3, tol_frac=0.005):
    """Remove the constant-field plateau that the PPMS holds after reaching
    the target field, keeping only the first point at the plateau value.
 
    Strategy (from measurement notes): "when the final field value is
    reached for more than two consecutive points, the scan should finish
    at the first value reached."
 
    Parameters
    ----------
    H : array-like
        Field values in chronological order.
    *arrays : array-like
        Co-arrays to trim consistently with H.
    min_plateau_pts : int
        Minimum consecutive near-zero-ΔH points that define a plateau.
    tol_frac : float
        |ΔH| threshold as a fraction of the median active-sweep step.
 
    Returns
    -------
    tuple : (H_trimmed, *arrays_trimmed)
    """
    H_arr = np.asarray(H, dtype=float)
    n = len(H_arr)
    if n < min_plateau_pts + 1:
        return (H_arr,) + tuple(np.asarray(a) for a in arrays)
 
    dH = np.abs(np.diff(H_arr))
    active = dH[dH > dH.max() * 0.01]
    med_dH = float(np.median(active)) if len(active) else float(dH.max())
    if med_dH < 1e-12:
        med_dH = 1.0
    threshold = tol_frac * med_dH
 
    # Walk backward to find where the end-plateau begins
    plateau_start = n          # default: no plateau
    consec = 0
    for i in range(n - 1, 0, -1):
        if dH[i - 1] < threshold:
            consec += 1
        else:
            if consec >= min_plateau_pts:
                plateau_start = i + 1   # first index of plateau
            break
    else:
        if consec >= min_plateau_pts:
            plateau_start = n - consec
 
    if plateau_start < n:
        print(f"  [trim] removed {n - plateau_start} plateau point(s) "
              f"at H ≈ {H_arr[plateau_start]:.0f} Oe")
 
    sl = slice(0, plateau_start)
    return (H_arr[sl],) + tuple(np.asarray(a)[sl] for a in arrays)
 
def _interp_to_grid(H_raw, vals_raw, H_grid):
    """Sort H_raw and interpolate vals_raw onto H_grid."""
    order = np.argsort(H_raw)
    return np.interp(H_grid, H_raw[order], np.asarray(vals_raw)[order])
 
def _antisymmetrize(H_grid, rho):
    """Odd-in-H part: ρ^odd(H) = [ρ(+H) − ρ(−H)] / 2.
 
    H_grid must span both positive and negative values.  Returns
    (H_pos, rho_odd_pos, H_full, rho_odd_full) where H_full is the
    full ±H grid and rho_odd_full is antisymmetric by construction.
    """
    pos = H_grid > 0
    neg = H_grid < 0
    H_pos   = H_grid[pos]
    rho_pos = rho[pos]
    # |H| values and ρ values on the negative branch, ascending in |H|
    H_neg_abs = np.abs(H_grid[neg])[::-1]
    rho_neg   = rho[neg][::-1]
    rho_at_negH = np.interp(H_pos, H_neg_abs, rho_neg)
    rho_odd_pos = (rho_pos - rho_at_negH) / 2.0
    H_full       = np.concatenate([-H_pos[::-1],  H_pos])
    rho_odd_full = np.concatenate([-rho_odd_pos[::-1], rho_odd_pos])
    return H_pos, rho_odd_pos, H_full, rho_odd_full
 
def _symmetrize(H_grid, rho):
    """Even-in-H part: ρ^even(H) = [ρ(+H) + ρ(−H)] / 2."""
    pos = H_grid > 0
    neg = H_grid < 0
    H_pos   = H_grid[pos]
    rho_pos = rho[pos]
    H_neg_abs = np.abs(H_grid[neg])[::-1]
    rho_neg   = rho[neg][::-1]
    rho_at_negH = np.interp(H_pos, H_neg_abs, rho_neg)
    rho_even_pos = (rho_pos + rho_at_negH) / 2.0
    H_full        = np.concatenate([-H_pos[::-1],  H_pos])
    rho_even_full = np.concatenate([rho_even_pos[::-1], rho_even_pos])
    return H_pos, rho_even_pos, H_full, rho_even_full

# ==========================================================================
# 4. Main analysis function Hall and MG measurement
# ==========================================================================
def _detect_H_irr_from_phase(H_sorted, theta_sorted,
                               window=7, std_threshold=5.0,
                               min_normal_frac=0.9):
    """Detect the irreversibility field H_irr from lock-in phase stability.

    In the SC state the measured voltage → 0 and the lock-in phase becomes
    random (large local variance). In the normal state the phase is stable.
    H_irr is the lowest field at which the phase becomes stable and stays
    stable for the remainder of the sweep.

    Parameters
    ----------
    H_sorted : ndarray
        Positive field values, sorted ascending (Oe).
    theta_sorted : ndarray
        Phase values (degrees) at the corresponding H points.
    window : int
        Half-width (in data points) of the rolling-std window.
    std_threshold : float
        Std (degrees) below which the phase is considered 'stable'
        (normal state). Default 20° is conservative — a random signal
        has std ≈ 104°.
    min_normal_frac : float
        Fraction of remaining points (from candidate H_irr onward) that
        must also be stable to confirm the transition. Prevents early
        false positives from a momentarily quiet SC phase.

    Returns
    -------
    float or None
        H_irr in Oe, or None if no stable (normal-state) region is found
        (sample remains SC throughout the measured field range).
    """
    n = len(H_sorted)
    if n < 2 * window + 2:
        return None

    # Only use finite theta values
    valid = np.isfinite(theta_sorted)
    if not np.any(valid):
        return None

    # Rolling std of phase over sliding window
    local_std = np.full(n, np.inf)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        vals = theta_sorted[lo:hi]
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) >= 3:
            local_std[i] = float(np.std(finite_vals))

    is_stable = local_std < std_threshold

    # Find the lowest H where is_stable, AND at least min_normal_frac of
    # all subsequent points are also stable → confirms entry into normal state.
    for i in range(n):
        if is_stable[i]:
            remaining = is_stable[i:]
            if np.mean(remaining) >= min_normal_frac:
                return float(H_sorted[i])

    return None

def analyze_hall_mr(fwd_source, bwd_source=None,
                     hall_n=1, mr_n=2, hall_col='X', mr_col='X',
                     current=1e-6,
                     t=1e-9, w=None, l=None,
                     V_cell_per_Z_A3=_BI2201_VCELL_PER_Z_A3,
                     fit_H_range_Oe=None,
                     rho_xx0_field_Oe=None,
                     T_label=None,
                     n_grid=500):
    """Full Hall and magnetoresistance analysis for one temperature scan.
 
    Implements: raw voltage → resistivity → antisymmetrize/symmetrize →
    linear R_H fit → n_H, p, µ_H, cot(θ_H) → ΔR/R₀ → modified Kohler.
 
    Physical note (Protocol v2, §2.2)
    ----------------------------------
    The antisymmetrization removes even-in-H contamination (contact
    misalignment).  It does NOT remove the Nernst signal (odd in H,
    like the Hall signal); verify T stability across the sweep and check
    for non-zero ρ_yx(H→0) as a Nernst diagnostic.
 
    Parameters
    ----------
    fwd_source : str or dict
        Forward sweep (−Hmax→+Hmax): CSV path or dict from
        ``load_hall_mr_csv``.
    bwd_source : str or dict, optional
        Backward sweep (+Hmax→−Hmax).  When provided, fwd and bwd are
        averaged on the common H grid before symmetrization to suppress
        time-dependent thermal drift.
    hall_n : int
        LIA channel number for the Hall (transverse) voltage (1–3).
    mr_n : int
        LIA channel number for the longitudinal (MR) voltage.
    hall_col, mr_col : {'X', 'R'}
        'X' = in-phase output (recommended); 'R' = magnitude.
    current : float
        Source current in Amperes.
    t : float
        Sample thickness in metres.  Required for ρ_yx = (V_y/I)·t.
    w : float, optional
        Sample width in metres.  Required for ρ_xx = (V_x/I)·(t·w/l).
        If None, ρ_xx = (V_x/I)·t  (sheet resistance × t).
    l : float, optional
        Voltage-probe separation in metres.
    V_cell_per_Z_A3 : float
        Unit-cell volume per Cu atom in Å³.  Default: Bi-2201 literature
        value (177.5 Å³).  Pull from your crystal's own characterisation
        for <25% precision on p.
    fit_H_range_Oe : (float, float), optional
        Positive-H window (Oe) for the linear R_H fit: R_H = ρ_yx^odd/B.
        Should be above H_irr(T) (determined from sweep hysteresis or
        from the phase channel θ becoming constant).  If None, the full
        field range is used (valid only if the sample is fully normal).
    rho_xx0_field_Oe : float
        Field (Oe) at which to evaluate ρ_xx for the µ_H denominator.
        Default: 0 (zero-field limit, appropriate in the normal state).
        If the sample is SC at H=0, set this to a field above H_irr(T).
    T_label : str or float, optional
        Temperature label; defaults to mean(Tsample).
    n_grid : int
        Points in the common H interpolation grid (default 500).
 
    Returns
    -------
    dict — keys described inline.
    """
    # ── 1. Load ────────────────────────────────────────────────────────────
    fwd = (fwd_source if isinstance(fwd_source, dict)
           else load_hall_mr_csv(fwd_source, hall_n, mr_n, hall_col, mr_col))
    bwd = None
    if bwd_source is not None:
        bwd = (bwd_source if isinstance(bwd_source, dict)
               else load_hall_mr_csv(bwd_source, hall_n, mr_n, hall_col, mr_col))
 
    # ── 2. Trim field plateaus ─────────────────────────────────────────────
    H_f, Vy_f, Vx_f, T_f, thH_f, thMR_f = _trim_H_plateau(
        fwd['H_Oe'], fwd['Vy_V'], fwd['Vx_V'], fwd['T_K'],
        fwd['theta_H_deg'], fwd['theta_MR_deg']
    )
    if bwd is not None:
        H_b, Vy_b, Vx_b, T_b, thH_b, thMR_b = _trim_H_plateau(
            bwd['H_Oe'], bwd['Vy_V'], bwd['Vx_V'], bwd['T_K'],
            bwd['theta_H_deg'], bwd['theta_MR_deg']
        )
 
    T_nominal = float(np.nanmean(T_f))
    if T_label is None:
        T_label = f"{T_nominal:.1f} K"
 
    # ── 3. Common H grid ───────────────────────────────────────────────────
    H_lo = float(np.nanmin(H_f))
    H_hi = float(np.nanmax(H_f))
    if bwd is not None:
        H_lo = max(H_lo, float(np.nanmin(H_b)))
        H_hi = min(H_hi, float(np.nanmax(H_b)))
    H_grid = np.linspace(H_lo, H_hi, n_grid)
 
    # ── 4. Resistivities ───────────────────────────────────────────────────
    geom_xx = t * (w / l) if (w is not None and l is not None) else t
    ryx_f = (Vy_f / current) * t
    rxx_f = (Vx_f / current) * geom_xx
 
    ryx_g = _interp_to_grid(H_f, ryx_f, H_grid)
    rxx_g = _interp_to_grid(H_f, rxx_f, H_grid)
 
    if bwd is not None:
        ryx_b = (Vy_b / current) * t
        rxx_b = (Vx_b / current) * geom_xx
        ryx_g = (ryx_g + _interp_to_grid(H_b, ryx_b, H_grid)) / 2.0
        rxx_g = (rxx_g + _interp_to_grid(H_b, rxx_b, H_grid)) / 2.0
 
    # ── 5. Antisymmetrize Hall, symmetrize MR ─────────────────────────────
    if np.any(H_grid < 0) and np.any(H_grid > 0):
        H_pos, ryx_odd_pos, H_asym, ryx_odd = _antisymmetrize(H_grid, ryx_g)
        H_pos, rxx_even_pos, H_sym,  rxx_even = _symmetrize(H_grid, rxx_g)
    else:
        # Sweep does not cross zero — can only return raw data
        warnings.warn(
            "H sweep does not cross zero; antisymmetrization not possible. "
            "Raw ρ_yx and ρ_xx are returned without symmetrization.",
            UserWarning, stacklevel=2
        )
        H_pos       = np.abs(H_grid)
        ryx_odd_pos = ryx_g
        H_asym      = H_grid
        ryx_odd     = ryx_g
        H_pos       = H_grid
        rxx_even_pos = rxx_g
        H_sym        = H_grid
        rxx_even     = rxx_g
 
    B_pos = H_pos * _HALL_Oe_to_T     # Tesla

    # ── 5b. Auto-detect H_irr from MR phase (if fit range not user-supplied) ──
    # Use the positive H branch of the forward sweep's MR phase channel.
    _user_supplied_fit_range  = fit_H_range_Oe is not None
    _user_supplied_rho_ref    = rho_xx0_field_Oe is not None and rho_xx0_field_Oe > 0

    always_sc = False   # assume normal state accessible until proven otherwise

    pos_mask_f = H_f > 0
    if np.any(pos_mask_f) and not np.all(np.isnan(thMR_f)):
        _H_det   = H_f[pos_mask_f]
        _th_det  = thMR_f[pos_mask_f]
        _ord_det = np.argsort(_H_det)
        _H_det_s = _H_det[_ord_det]
        _th_det_s = _th_det[_ord_det]

        H_irr_detected = _detect_H_irr_from_phase(_H_det_s, _th_det_s)
    else:
        H_irr_detected = None
        warnings.warn(
            "MR phase channel (theta_MR) is all-NaN; cannot auto-detect H_irr. "
            "Pass --fit-H-range manually.",
            UserWarning, stacklevel=2
        )

    if not _user_supplied_fit_range:
        if H_irr_detected is not None:
            H_max_pos = float(H_pos.max()) if len(H_pos) > 0 else 0.0
            fit_H_range_Oe = (H_irr_detected, H_max_pos)
            print(f"  [auto] H_irr ≈ {H_irr_detected:.0f} Oe from MR phase; "
                  f"fit range set to [{H_irr_detected:.0f}, {H_max_pos:.0f}] Oe")
        else:
            always_sc = True
            print(
                f"\n  ⚠  No normal state detected in θ_MR across the entire "
                f"measured field range — sample appears SC at all measured fields. "
                f"R_H, n_H, p, µ_H cannot be extracted at T = {T_label}.\n"
                f"  Raw ρ_yx and ρ_xx are still computed and returned for plotting."
            )

    if not _user_supplied_rho_ref:
        if H_irr_detected is not None:
            rho_xx0_field_Oe = H_irr_detected
            print(f"  [auto] ρ_xx reference field set to H_irr = "
                  f"{H_irr_detected:.0f} Oe")
        elif not always_sc:
            rho_xx0_field_Oe = 0.0   # phase suggests normal at H=0
 
  # ── 6–8. Derived quantities, fit, and summary ──────────────────────────
    # Initialise ALL derived scalars to NaN so the always-SC branch never
    # hits an undefined variable or a None format string.
    R_H = R_H_err = RH_offset = np.nan
    n_H_m3 = n_H_cm3 = p = mu_H_SI = mu_H_cm2 = np.nan
    cot_theta_scalar = cot_theta_at_ref = H_ref_Oe = np.nan
    rxx_ref = np.nan
    ryx_at_ref_Ohm_m = np.nan
    fit_mask = np.zeros(len(H_pos), dtype=bool)

    # cot(θ_H) array — computed regardless of SC state (used for plotting)
    with np.errstate(divide='ignore', invalid='ignore'):
        cot_theta = np.where(
            np.abs(ryx_odd_pos) > 1e-20,
            rxx_even_pos / np.abs(ryx_odd_pos),
            np.nan
        )

    if not always_sc:
        # ── 6. R_H linear fit ─────────────────────────────────────────────
        if fit_H_range_Oe is not None:
            fit_mask = ((H_pos >= fit_H_range_Oe[0]) &
                        (H_pos <= fit_H_range_Oe[1]))
        else:
            fit_mask = np.ones(len(H_pos), dtype=bool)

        n_fit = int(np.sum(fit_mask))
        if n_fit < 2:
            print(
                f"  ⚠  fit_H_range_Oe={fit_H_range_Oe} gives only {n_fit} "
                f"point(s) — skipping R_H fit."
            )
        else:
            coeffs, cov = np.polyfit(
                B_pos[fit_mask], ryx_odd_pos[fit_mask], deg=1, cov=True
            )
            R_H       = float(coeffs[0])
            R_H_err   = float(np.sqrt(cov[0, 0]))
            RH_offset = float(coeffs[1])

            # Carrier density, doping
            if abs(R_H) > 1e-20:
                n_H_m3  = 1.0 / (_HALL_e * R_H)
                n_H_cm3 = n_H_m3 * 1e-6
                V_cell_m3 = V_cell_per_Z_A3 * _HALL_A3_to_m3
                p = abs(n_H_m3) * V_cell_m3 - 1.0

            # ρ_xx at reference field
            _ref_H = rho_xx0_field_Oe if rho_xx0_field_Oe is not None else 0.0
            rxx_ref = float(np.interp(_ref_H, H_pos, rxx_even_pos))
            if abs(rxx_ref) > 1e-20:
                mu_H_SI  = abs(R_H) / abs(rxx_ref)
                mu_H_cm2 = mu_H_SI * 1e4
            else:
                _ref_str = (f"{_ref_H:.0f} Oe"
                            if rho_xx0_field_Oe is not None else "auto")
                warnings.warn(
                    f"ρ_xx ≈ 0 at the reference field ({_ref_str}) — "
                    "µ_H cannot be computed.",
                    UserWarning, stacklevel=2
                )

                        # ── cot(θ_H) from fitted ρ_yx (immune to low-field noise) ──────
            # Using the raw antisymmetrised ryx_odd to compute cot_theta is
            # unreliable when the Hall signal is a tiny fraction of the total
            # measured voltage (e.g. <1% at 200K in Bi-2201). Interpolation
            # offsets between the fwd/bwd H grids introduce noise at low H that
            # dominates ryx_odd there, driving cot_theta → ∞ at sign crossings
            # and wildly inflating any mean.
            # Fix: use the fitted ryx = R_H × B + offset (smooth, high-field
            # constrained) evaluated at H_ref = the maximum field in the fit
            # window, where the Hall SNR is largest.
            H_ref_Oe = float(H_pos[fit_mask].max())
            B_ref    = H_ref_Oe * _HALL_Oe_to_T
            ryx_fit_at_ref = R_H * B_ref + RH_offset
            rxx_at_ref     = float(np.interp(H_ref_Oe, H_pos, rxx_even_pos))

            if abs(ryx_fit_at_ref) > 1e-20:
                cot_theta_scalar  = rxx_at_ref / abs(ryx_fit_at_ref)
                cot_theta_at_ref  = cot_theta_scalar   # same quantity, consistent
            else:
                cot_theta_scalar = cot_theta_at_ref = np.nan
            ryx_at_ref_Ohm_m = ryx_fit_at_ref   # store for multi-T plot

    # ── 7. MR quantities ──────────────────────────────────────────────────
    rxx0 = float(np.interp(0.0, H_sym, rxx_even))
    if abs(rxx0) > 1e-20 and not always_sc:
        MR = (rxx_even_pos - rxx0) / abs(rxx0)
    else:
        MR = np.full_like(rxx_even_pos, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        tan2_theta = np.where(
            np.abs(cot_theta) > 1e-10, 1.0 / cot_theta**2, np.nan
        )
 
    # ── 7. Derived Hall quantities ─────────────────────────────────────────
    n_H_m3  = (1.0 / (_HALL_e * R_H)) if abs(R_H) > 1e-20 else np.nan
    n_H_cm3 = n_H_m3 * 1e-6 if np.isfinite(n_H_m3) else np.nan
 
    # Doping: p = n_H · V_cell/Z − 1  (hole-doped convention; R_H > 0 → holes)
    V_cell_m3 = V_cell_per_Z_A3 * _HALL_A3_to_m3
    p = abs(n_H_m3) * V_cell_m3 - 1.0 if np.isfinite(n_H_m3) else np.nan
 
    # ρ_xx at reference field for µ_H denominator
    rxx_ref = float(np.interp(rho_xx0_field_Oe, H_pos, rxx_even_pos))
    if abs(rxx_ref) > 1e-20:
        mu_H_SI   = abs(R_H) / abs(rxx_ref)    # m²/(V·s)
        mu_H_cm2  = mu_H_SI * 1e4               # cm²/(V·s)
    else:
        mu_H_SI = mu_H_cm2 = np.nan
        warnings.warn(
            f"  ρ_xx (ref H)   = "
            f"{_fmt(rxx_ref * to_uOcm) if np.isfinite(rxx_ref) else 'N/A'} µΩ·cm  "
            f"(ref H = {f'{rho_xx0_field_Oe:.0f} Oe' if rho_xx0_field_Oe is not None else 'N/A (always SC)'})\n",
            UserWarning, stacklevel=2
        )
 
    # ── 8. Console summary ────────────────────────────────────────────────
    to_uOcm = _HALL_Ohm_m_to_uOhm_cm

    def _fmt(v, spec='.4g'):
        """Format a scalar; return 'N/A' for NaN/inf/None."""
        if v is None:
            return 'N/A'
        try:
            return f'{v:{spec}}' if np.isfinite(float(v)) else 'N/A'
        except (TypeError, ValueError):
            return 'N/A'

    _ref_H_display = (f"{rho_xx0_field_Oe:.0f} Oe"
                      if rho_xx0_field_Oe is not None else "N/A (always SC)")
    _hirr_display  = (f"{H_irr_detected:.0f} Oe"
                      if H_irr_detected is not None else "not detected (always SC)")

    print(
        f"\n{'─'*62}\n"
        f"  Hall/MR — T = {T_label}   "
        f"({'fwd+bwd' if bwd is not None else 'fwd only'})\n"
        f"{'─'*62}\n"
        f"  H_irr (phase)  : {_hirr_display}\n"
        f"  H fit range    : {fit_H_range_Oe}\n"
        f"  R_H            = {_fmt(R_H)} m³/C\n"
        f"  Fit offset     = {_fmt(RH_offset * to_uOcm if np.isfinite(RH_offset) else np.nan)} µΩ·cm\n"
        f"  n_H            = {_fmt(n_H_cm3)} cm⁻³\n"
        f"  p (doping)     = {_fmt(p)}\n"
        f"  µ_H            = {_fmt(mu_H_cm2)} cm²/(V·s)\n"
        f"  ρ_xx (ref H)   = {_fmt(rxx_ref * to_uOcm if np.isfinite(rxx_ref) else np.nan)} µΩ·cm"
        f"  (ref H = {_ref_H_display})\n"
        f"  ⟨cot θ_H⟩     = {_fmt(cot_theta_scalar)}\n"
        f"{'─'*62}"
    )
 
    return {
        # ── Common-grid arrays ────────────────────────────────────────────
        'H_grid_Oe':       H_grid,
        'ryx_grid_Ohm_m':  ryx_g,
        'rxx_grid_Ohm_m':  rxx_g,
        # ── Processed: antisymmetrized Hall ─────────────────────────────
        'H_pos_Oe':        H_pos,
        'H_asym_Oe':       H_asym,
        'ryx_odd_pos':     ryx_odd_pos,    # Ω·m, positive-H half
        'ryx_odd_full':    ryx_odd,        # Ω·m, ±H
        'ryx_at_ref_Ohm_m':   ryx_at_ref_Ohm_m,
        'B_pos_T':         B_pos,
        # ── Processed: symmetrized MR ────────────────────────────────────
        'H_sym_Oe':        H_sym,
        'rxx_even_pos':    rxx_even_pos,   # Ω·m
        'rxx_even_full':   rxx_even,       # Ω·m
        # ── Hall quantities ──────────────────────────────────────────────
        'R_H_m3_C':        R_H,
        'R_H_err_m3_C':    R_H_err,
        'RH_offset_Ohm_m': RH_offset,
        'n_H_m3':          n_H_m3,
        'n_H_cm3':         n_H_cm3,
        'p_doping':        p,
        'mu_H_m2_Vs':      mu_H_SI,
        'mu_H_cm2_Vs':     mu_H_cm2,
        'cot_theta_H':     cot_theta,      # array vs H_pos
        'cot_theta_scalar': cot_theta_scalar,
        'cot_theta_at_Href': cot_theta_at_ref,
        'H_ref_Oe':        H_ref_Oe,
        'H_irr_Oe':       H_irr_detected,   # auto-detected or None
        'always_sc':      always_sc,         # True if no normal state found
        'fit_mask':       fit_mask,
        # ── MR quantities ────────────────────────────────────────────────
        'MR':              MR,             # ΔR/R₀, dimensionless
        'tan2_theta_H':    tan2_theta,     # for modified Kohler
        'rxx0_Ohm_m':      rxx0,
        'rxx_ref_Ohm_m':   rxx_ref,
        # ── Raw trimmed data (for diagnostic raw-voltage plots) ──────────
        'fwd_H_Oe':        H_f,
        'fwd_Vy_V':        Vy_f,
        'fwd_Vx_V':        Vx_f,
        'fwd_T_K':         T_f,
        'fwd_theta_H':     thH_f,
        'fwd_theta_MR':    thMR_f,
        'has_bwd':         bwd is not None,
        'bwd_H_Oe':        H_b  if bwd is not None else None,
        'bwd_Vy_V':        Vy_b if bwd is not None else None,
        'bwd_Vx_V':        Vx_b if bwd is not None else None,
        'bwd_T_K':         T_b  if bwd is not None else None,
        # ── Metadata ────────────────────────────────────────────────────
        'T_nominal_K':     T_nominal,
        'T_label':         str(T_label),
        'current_A':       current,
        't_m':             t,
        'w_m':             w,
        'l_m':             l,
        'V_cell_per_Z_A3': V_cell_per_Z_A3,
        'fit_H_range_Oe':  fit_H_range_Oe,
        'rho_xx0_field_Oe': rho_xx0_field_Oe,
        'geom_factor':     geom_xx,
    }

# Fitting for the costheta

def fit_cot_theta_vs_T2(results_list):
    """Fit cot(θ_H) = α·T² + β.  Uses cot_theta_scalar (mean over the fit
    window) — the same quantity printed in the terminal analysis — so the
    fit and the plot are always consistent with the reported values.
    NaN entries (always-SC temperatures) are silently excluded.
    """
    T_arr  = np.array([r['T_nominal_K']    for r in results_list])
    ct_arr = np.array([r['cot_theta_scalar'] for r in results_list])

    finite = np.isfinite(ct_arr) & np.isfinite(T_arr)
    n_valid = int(np.sum(finite))

    if n_valid < 2:
        warnings.warn(
            f"Only {n_valid} temperature(s) with finite cot(θ_H) — "
            "cannot fit α·T² + β.", UserWarning
        )
        return {
            'T_K': T_arr, 'cot_theta': ct_arr, 'T2': T_arr**2,
            'alpha': np.nan, 'alpha_err': np.nan,
            'beta':  np.nan, 'beta_err':  np.nan,
            'poly':  None,   'R2':        np.nan,
            'n_valid': n_valid,
        }

    T2 = T_arr[finite]**2
    ct = ct_arr[finite]
    coeffs, cov = np.polyfit(T2, ct, deg=1, cov=True)
    alpha, beta   = float(coeffs[0]), float(coeffs[1])
    alpha_err     = float(np.sqrt(cov[0, 0]))
    beta_err      = float(np.sqrt(cov[1, 1]))
    ct_fit        = np.polyval(coeffs, T2)
    ss_res        = np.sum((ct - ct_fit)**2)
    ss_tot        = np.sum((ct - ct.mean())**2)
    R2            = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Report excluded temperatures
    excluded = T_arr[~finite]
    excl_str = (f"  Excluded (NaN / always-SC): T = {excluded.tolist()} K\n"
                if len(excluded) > 0 else "")
    print(
        f"\n  cot(θ_H) = α·T² + β  [{n_valid} temperatures used]\n"
        f"{excl_str}"
        f"  α = {alpha:.4g} ± {alpha_err:.2g}  K⁻²\n"
        f"  β = {beta:.4g}  ± {beta_err:.2g}\n"
        f"  R² = {R2:.4f}"
    )
    return {
        'T_K':       T_arr,    'cot_theta':  ct_arr,
        'T2':        T_arr**2, 'alpha':      alpha,
        'alpha_err': alpha_err,'beta':       beta,
        'beta_err':  beta_err, 'poly':       np.poly1d(coeffs),
        'R2':        R2,       'n_valid':    n_valid,
    }

# ==========================================================================
# 6. PLOTTING FUNCTIONS
# ==========================================================================
 
def _to_uOhm_cm(rho_Ohm_m):
    """Ω·m → µΩ·cm (standard resistivity unit for cuprates in literature)."""
    return rho_Ohm_m * _HALL_Ohm_m_to_uOhm_cm
 
 
def plot_hall_raw(result, show_T=False, axes=None, figsize=None, colors=None):
    """Plot raw Hall (V_y), MR (V_x), phase (θ_H and θ_MR), and optionally
    Tsample, all stacked vertically and sharing the x-axis.

    Panel order (top to bottom):
      0 — V_y (Hall voltage)
      1 — V_x (MR / longitudinal voltage)
      2 — Phase: θ_Hall (solid) and θ_MR (dashed) in the same panel
      3 — T_sample  (only when show_T=True)
    """
    nc = 4 if show_T else 3
    c  = colors or {'fwd': 'tab:blue', 'bwd': 'tab:red'}

    if axes is None:
        w    = (figsize[0] if figsize else 8.6) / 2.54
        h_pp = (figsize[1] if figsize else 4.5) / 2.54   # height per panel
        fig, axes = plt.subplots(nc, 1, figsize=(w, h_pp * nc),
                                  sharex=True, constrained_layout=True)
    else:
        fig = axes[0].get_figure()

    def _plot_pair(ax, H_fwd, y_fwd, H_bwd, y_bwd, ylabel,
                    fwd_label='Forward', bwd_label='Backward',
                    fwd_ls='-', bwd_ls='-', fwd_alpha=1.0, bwd_alpha=0.7):
        ax.plot(H_fwd, y_fwd, color=c['fwd'], lw=0.8,
                ls=fwd_ls, label=fwd_label, alpha=fwd_alpha)
        if H_bwd is not None and y_bwd is not None:
            ax.plot(H_bwd, y_bwd, color=c['bwd'], lw=0.8,
                    ls=bwd_ls, label=bwd_label, alpha=bwd_alpha)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, loc='best')

    H_fwd_T = result['fwd_H_Oe'] * 1e-4
    H_bwd_T = (result['bwd_H_Oe'] * 1e-4
               if result['has_bwd'] and result.get('bwd_H_Oe') is not None
               else None)

    # Panel 0 — V_y
    _plot_pair(axes[0], H_fwd_T, result['fwd_Vy_V'],
               H_bwd_T, result.get('bwd_Vy_V'),
               r'$V_y$ (Hall, V)')

    # Panel 1 — V_x
    _plot_pair(axes[1], H_fwd_T, result['fwd_Vx_V'],
               H_bwd_T, result.get('bwd_Vx_V'),
               r'$V_x$ (MR, V)')

    # Panel 2 — Phase (both Hall and MR channels on the same panel)
    ax_ph = axes[2]
    th_H_fwd  = result.get('fwd_theta_H')
    th_MR_fwd = result.get('fwd_theta_MR')
    th_H_bwd  = result.get('bwd_theta_H')
    th_MR_bwd = result.get('bwd_theta_MR')

    if th_H_fwd is not None and np.any(np.isfinite(th_H_fwd)):
        ax_ph.plot(H_fwd_T, th_H_fwd, color=c['fwd'], lw=0.8, ls='-',
                   label=r'$\theta_H$ fwd')
        if H_bwd_T is not None and th_H_bwd is not None:
            ax_ph.plot(H_bwd_T, th_H_bwd, color=c['bwd'], lw=0.8, ls='-',
                       label=r'$\theta_H$ bwd', alpha=0.7)
    if th_MR_fwd is not None and np.any(np.isfinite(th_MR_fwd)):
        ax_ph.plot(H_fwd_T, th_MR_fwd, color=c['fwd'], lw=0.8, ls='--',
                   label=r'$\theta_{MR}$ fwd', alpha=0.8)
        if H_bwd_T is not None and th_MR_bwd is not None:
            ax_ph.plot(H_bwd_T, th_MR_bwd, color=c['bwd'], lw=0.8, ls='--',
                       label=r'$\theta_{MR}$ bwd', alpha=0.6)
    ax_ph.set_ylabel(r'Phase (deg)')
    ax_ph.legend(fontsize=6, loc='best', ncol=2)
    # Shade where phase is random → SC state (std > 20°) on fwd branch
    if th_MR_fwd is not None and np.any(np.isfinite(th_MR_fwd)):
        window = 7
        n = len(th_MR_fwd)
        local_std = np.array([
            np.std(th_MR_fwd[max(0, i-window):min(n, i+window+1)])
            for i in range(n)
        ])
        sc_mask = local_std > 20.0   # same threshold as _detect_H_irr_from_phase
        if np.any(sc_mask):
            ax_ph.fill_between(H_fwd_T, ax_ph.get_ylim()[0], ax_ph.get_ylim()[1],
                                where=sc_mask, color='gray', alpha=0.15,
                                label='SC (unstable phase)')

    # Panel 3 — T_sample (optional)
    if show_T:
        _plot_pair(axes[3], H_fwd_T, result['fwd_T_K'],
                   H_bwd_T, result.get('bwd_T_K'),
                   r'$T_{\rm sample}$ (K)')

    # x-axis label on bottom panel only
    axes[nc - 1].set_xlabel(r'$\mu_0 H$ (T)')

    return fig, axes
 
 
def plot_hall_antisym(result, ax=None, color='tab:blue',
                       fit_color='k', figsize=None):
    """Plot antisymmetrized ρ_yx^odd(H) with the linear R_H fit overlay.
 
    Parameters
    ----------
    result : dict
        Output of analyze_hall_mr.
    ax : Axes, optional
    color : str
        Colour for the data.
    fit_color : str
        Colour for the fit line (should differ from data for clarity).
 
    Returns
    -------
    ax
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7)  / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
 
    # Data in µΩ·cm vs Tesla
    H_T  = result['H_asym_Oe'] * 1e-4
    ryx  = _to_uOhm_cm(result['ryx_odd_full'])
    ax.plot(H_T, ryx, color=color, lw=0.9, label=r'$\rho_{yx}^{\rm odd}(H)$')
 
    # R_H fit (positive half only, extended across full range for display)
    R_H    = result['R_H_m3_C']
    offset = result['RH_offset_Ohm_m']
    H_fit  = result['H_asym_Oe'] * 1e-4  # same axis as plot
    ryx_fit_uOcm = _to_uOhm_cm(R_H * H_fit + offset)
 
    # Shade the fit window
    if result['fit_H_range_Oe'] is not None:
        lo, hi = [v * 1e-4 for v in result['fit_H_range_Oe']]
        ax.axvspan(lo, hi, color=fit_color, alpha=0.08, lw=0)
 
    ax.plot(H_fit, ryx_fit_uOcm, color=fit_color, ls='--', lw=1.0,
            label=(rf"$R_H = {R_H*1e9:.3g}$ mm³/C"))
 
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel(r'$\mu_0 H$ (T)')
    ax.set_ylabel(r'$\rho_{yx}^{\rm odd}$ ($\mu\Omega\cdot$cm)')
    ax.legend()
 
    if created:
        ax.set_title(f"T = {result['T_label']}")
    return ax
 
 
def plot_rho_xx(result, ax=None, color='tab:blue', figsize=None):
    """Plot symmetrized ρ_xx^even(H) vs field.
 
    Returns
    -------
    ax
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7)  / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
 
    H_T  = result['H_sym_Oe'] * 1e-4
    rxx  = _to_uOhm_cm(result['rxx_even_full'])
    ax.plot(H_T, rxx, color=color, lw=0.9,
            label=r'$\rho_{xx}^{\rm even}(H)$')
 
    rxx0_display = _to_uOhm_cm(result['rxx0_Ohm_m'])
    ax.axhline(rxx0_display, color='gray', ls=':', lw=0.7,
               label=rf'$\rho_{{xx}}(0)={rxx0_display:.3g}\ \mu\Omega\cdot$cm')
 
    ax.set_xlabel(r'$\mu_0 H$ (T)')
    ax.set_ylabel(r'$\rho_{xx}^{\rm even}$ ($\mu\Omega\cdot$cm)')
    ax.legend()
 
    if created:
        ax.set_title(f"T = {result['T_label']}")
    return ax
 
 
def plot_MR_hall(result, ax=None, color='tab:blue', figsize=None):
    """Plot normalised magnetoresistance ΔR/R₀ vs field.
 
    Named plot_MR_hall to avoid collision with any existing plot_MR.
 
    Returns
    -------
    ax
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7)  / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
 
    H_T = result['H_pos_Oe'] * 1e-4
    MR  = result['MR']
    ax.plot(H_T, MR * 100, color=color, lw=0.9,
            label=rf"$T={result['T_label']}$")
 
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel(r'$\mu_0 H$ (T)')
    ax.set_ylabel(r'$\Delta\rho/\rho_0$ (%)')
    ax.legend()
 
    if created:
        ax.set_title(f"T = {result['T_label']}")
    return ax
 
 
# ── Multi-temperature plot helpers ─────────────────────────────────────────
 
def plot_RH_vs_T(results_list, ax=None, figsize=None, color='tab:blue'):
    """R_H vs temperature from a list of single-T results.
 
    Returns ax.
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7)  / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
 
    T   = np.array([r['T_nominal_K'] for r in results_list])
    RH  = np.array([r['R_H_m3_C']   for r in results_list]) * 1e9  # → mm³/C
    RHe = np.array([r['R_H_err_m3_C'] for r in results_list]) * 1e9
 
    ax.errorbar(T, RH, yerr=RHe, fmt='o', color=color, ms=4, capsize=3)
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$R_H$ (mm³/C)')
    ax.set_title(r'Hall coefficient $R_H(T)$')
    return ax
 
 
def plot_nH_vs_T(results_list, ax=None, figsize=None, color='tab:orange'):
    """Carrier density n_H vs T.
 
    Returns ax.
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7)  / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
 
    T  = np.array([r['T_nominal_K'] for r in results_list])
    nH = np.array([r['n_H_cm3']     for r in results_list])
 
    ax.plot(T, nH, 'o', color=color, ms=4)
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$n_H$ (cm$^{-3}$)')
    ax.set_title(r'Hall carrier density $n_H(T)$')
    return ax
 
 
def plot_muH_vs_T(results_list, ax=None, figsize=None, color='tab:green'):
    """Hall mobility µ_H vs T.
 
    Returns ax.
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7)  / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
 
    T  = np.array([r['T_nominal_K']  for r in results_list])
    mu = np.array([r['mu_H_cm2_Vs']  for r in results_list])
 
    ax.plot(T, mu, 'o', color=color, ms=4)
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$\mu_H$ (cm²/V·s)')
    ax.set_title(r'Hall mobility $\mu_H(T)$')
    return ax
 
 
def plot_cot_theta_vs_T2(results_list, cot_fit=None, ax=None,
                           figsize=None, color='tab:purple'):
    """cot(θ_H) vs T² — uses cot_theta_scalar (mean over fit window),
    matching the printed terminal values exactly.
    NaN entries (always-SC, no normal state) are excluded from the plot.
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7.0) / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)

    T   = np.array([r['T_nominal_K']     for r in results_list])
    ct  = np.array([r['cot_theta_scalar'] for r in results_list])

    # Explicit NaN filter — do NOT rely on matplotlib to hide NaN markers
    finite = np.isfinite(ct) & np.isfinite(T)
    if not np.any(finite):
        ax.set_xlabel(r'$T^2$ (K²)')
        ax.set_ylabel(r'$\cot\theta_H$')
        ax.set_title(r'$\cot\theta_H$ vs $T^2$  (no valid data)')
        return ax

    ax.plot(T[finite]**2, ct[finite], 'o', color=color, ms=4,
            label=r'$\cot\theta_H$ (mean over fit window)')

    if cot_fit is not None and cot_fit['poly'] is not None:
        T2_plot = np.linspace(0, (T[finite]**2).max() * 1.05, 300)
        ax.plot(T2_plot, cot_fit['poly'](T2_plot), color='k', ls='--', lw=1.0,
                label=(rf"$\alpha T^2+\beta$,  "
                        rf"$\alpha={cot_fit['alpha']:.3g}$ K$^{{-2}}$,  "
                        rf"$R^2={cot_fit['R2']:.3f}$"))

    # Mark excluded (NaN/SC) temperatures if any
    n_excl = int(np.sum(~finite))
    if n_excl > 0:
        excl_T = T[~finite]
        ax.annotate(
            f"{n_excl} T excluded (always SC): "
            f"{[f'{v:.0f} K' for v in excl_T]}",
            xy=(0.02, 0.97), xycoords='axes fraction',
            va='top', fontsize=6, color='gray'
        )

    ax.set_xlabel(r'$T^2$ (K²)')
    ax.set_ylabel(r'$\cot\theta_H$  (dimensionless)')
    ax.set_title(r'Strange-metal diagnostic: $\cot\theta_H\propto T^2$')
    ax.legend()
    return ax
 
 
def plot_kohler(results_list, axes=None, figsize=None):
    """Plain and modified (quadrature) Kohler plots from multi-T results.
 
    Two panels:
      Left : ΔR/R₀ vs (H/ρ_xx)²  — plain Kohler (expected to fail in cuprates)
      Right: ΔR/R₀ vs tan²(θ_H)  — modified Kohler (tends to hold)
 
    Returns
    -------
    fig, axes
    """
    if axes is None:
        w = (figsize[0] if figsize else 17.8) / 2.54
        h = (figsize[1] if figsize else 7)   / 2.54
        fig, axes = plt.subplots(1, 2, figsize=(w, h), constrained_layout=True)
    else:
        fig = axes[0].get_figure()
 
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
 
    for i, r in enumerate(results_list):
        c = colors[i % len(colors)]
        lbl = r['T_label']
        H_T  = r['H_pos_Oe'] * 1e-4
        MR   = r['MR']
        rxx0 = abs(_to_uOhm_cm(r['rxx0_Ohm_m'])) or np.nan
 
        # Plain Kohler: x = (H / ρ_xx0)²  in units (T / µΩ·cm)²
        x_kohler = (H_T / rxx0)**2
        axes[0].plot(x_kohler, MR * 100, lw=0.8, color=c, label=lbl)
 
        # Modified Kohler: x = tan²(θ_H)
        tan2 = r['tan2_theta_H']
        finite = np.isfinite(tan2) & np.isfinite(MR)
        if np.any(finite):
            axes[1].plot(tan2[finite], MR[finite] * 100,
                         lw=0.8, color=c, label=lbl)
 
    axes[0].set_xlabel(r'$(H/\rho_0)^2$  (T/µΩ·cm)²')
    axes[0].set_ylabel(r'$\Delta\rho/\rho_0$ (%)')
    axes[0].set_title('Plain Kohler')
    axes[0].legend(fontsize=6)
 
    axes[1].set_xlabel(r'$\tan^2\theta_H$')
    axes[1].set_ylabel(r'$\Delta\rho/\rho_0$ (%)')
    axes[1].set_title('Modified (quadrature) Kohler')
    axes[1].legend(fontsize=6)
 
    return fig, axes
 
 
def plot_HT_scaling(results_list, ax=None, figsize=None):
    """H/T scaling plot: ΔR/T vs H/T for each temperature.
 
    Data collapse onto a single curve tests the quadrature
    two-scattering-rate form τ⁻¹ ∝ √[(aT)² + (b·µ_B·H)²], which
    produces H-linear MR at high field (Protocol §4.2).
 
    Returns
    -------
    ax
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7)  / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
 
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
 
    for i, r in enumerate(results_list):
        T   = r['T_nominal_K']
        c   = colors[i % len(colors)]
        H_T = r['H_pos_Oe'] * 1e-4    # Tesla
        MR  = r['MR']
        finite = np.isfinite(MR)
        if not np.any(finite):
            continue
        ax.plot(H_T[finite] / T, MR[finite] * 100 / T,
                color=c, lw=0.8, label=r['T_label'])
 
    ax.set_xlabel(r'$H/T$  (T/K)')
    ax.set_ylabel(r'$(\Delta\rho/\rho_0)/T$  (%/K)')
    ax.set_title(r'$H/T$ scaling — quadrature MR test')
    ax.legend(fontsize=6)
 
    if created:
        plt.tight_layout()
    return ax

def plot_rxx_vs_T(results_list, ax=None, figsize=None, color='tab:blue'):
    """ρ_xx(H_ref) vs T with a linear fit (expected for strange-metal cuprates).

    ρ_xx is taken at the reference field used for each temperature's analysis
    (H_irr for SC-at-low-H temperatures, H→0 for fully-normal temperatures),
    so it represents the normal-state longitudinal resistivity.

    Parameters
    ----------
    results_list : list of dict
        Multi-T outputs of analyze_hall_mr.

    Returns
    -------
    ax
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7.0) / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)

    T    = np.array([r['T_nominal_K']    for r in results_list])
    rxx  = np.array([r['rxx_ref_Ohm_m']  for r in results_list])

    finite = np.isfinite(rxx) & np.isfinite(T)
    rxx_uOcm = _to_uOhm_cm(rxx)

    ax.plot(T[finite], rxx_uOcm[finite], 'o', color=color, ms=4, zorder=3,
            label=r'$\rho_{xx}(H_{\rm ref})$')

    # Linear fit: ρ_xx = A·T + B  (strange-metal / bad-metal T-linear scattering)
    if np.sum(finite) >= 2:
        coeffs, cov = np.polyfit(T[finite], rxx_uOcm[finite], deg=1, cov=True)
        A, B     = float(coeffs[0]), float(coeffs[1])
        A_err    = float(np.sqrt(cov[0, 0]))
        T_fit    = np.linspace(T[finite].min(), T[finite].max(), 300)
        ax.plot(T_fit, np.polyval(coeffs, T_fit), color='k', ls='--', lw=1.0,
                label=rf'$A\cdot T+B$,  $A={A:.3g}\pm{A_err:.1g}$ µΩ·cm/K')
        print(
            f"\n  ρ_xx(T) linear fit:\n"
            f"  A (dρ/dT)  = {A:.4g} ± {A_err:.2g}  µΩ·cm/K\n"
            f"  B (T=0)    = {B:.4g}  µΩ·cm\n"
            f"  [excludes {np.sum(~finite)} NaN temperatures]"
        )

    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$\rho_{xx}(H_{\rm ref})$ (µΩ·cm)')
    ax.set_title(r'Longitudinal resistivity $\rho_{xx}$ vs $T$')
    ax.legend()
    return ax


def plot_ryx_vs_T(results_list, ax=None, figsize=None, color='tab:green'):
    """ρ_yx (fitted Hall signal) vs T — no fit, for visual inspection.

    ρ_yx is taken from the R_H linear fit evaluated at H_ref (the maximum
    field in the fit window), so it is immune to the low-field antisymmetri-
    sation noise that plagues the raw ryx_odd at small Hall-to-misalignment
    signal ratios.

    Returns
    -------
    ax
    """
    created = ax is None
    if created:
        w = (figsize[0] if figsize else 8.6) / 2.54
        h = (figsize[1] if figsize else 7.0) / 2.54
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)

    T    = np.array([r['T_nominal_K']       for r in results_list])
    ryx  = np.array([r['ryx_at_ref_Ohm_m']  for r in results_list])

    finite = np.isfinite(ryx) & np.isfinite(T)
    ryx_uOcm = _to_uOhm_cm(ryx)

    ax.plot(T[finite], ryx_uOcm[finite], 'o', color=color, ms=4,
            label=r'$\rho_{yx}(H_{\rm ref})$ from R_H fit')

    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$\rho_{yx}(H_{\rm ref})$ (µΩ·cm)')
    ax.set_title(r'Hall resistivity $\rho_{yx}$ vs $T$ (fitted, at $H_{\rm ref}$)')
    ax.legend()
    if created:
        plt.tight_layout()
    return ax

# ==================================================================
# Command-line interface
#
# Each plot type gets one (parser-builder, runner) pair registered in
# PLOT_TYPES below. Adding a new measurement type later (I-V, etc.)
# means writing its own analyze_*/plot_* functions above, plus one
# more entry here -- main() itself never needs to change.
# ==================================================================

def _add_RT_parser(subparsers):
    """Define the `RT` subcommand: arguments + help text only."""
    p = subparsers.add_parser("RT", help="Resistance vs. Temperature")
    p.add_argument("csv_files", nargs="+",
                    help="One or more measurement CSV file(s) to plot together")
    p.add_argument("--bridge", type=int, default=1,
                    help="Bridge channel to use (default: 1)")
    p.add_argument("--errorbars", action="store_true",
                    help="Plot with error bars (default: off)")
    p.add_argument("--branches", action="store_true",
                    help="Color cooldown/warmup branches separately")
    p.add_argument("--no-split", action="store_true",
                    help="Do not split data into cooldown/warmup branches")
    p.add_argument("--skip-points", type=int, default=0,
                    help="Discard this many points from the start of the run "
                         "(e.g. excitation-current SNR calibration points)")
    p.add_argument("--labels", nargs="+", default=None,
                    help="Legend label(s), one per CSV file "
                         "(defaults to each file's name)")
    p.add_argument("-o", "--output", default="RT_plot.pdf",
                    help="Output figure path (default: RT_plot.pdf)")
    p.add_argument("--figsize", nargs=2, type=float, default=(8.6, 6.0),
                    metavar=("WIDTH_CM", "HEIGHT_CM"),
                    help="Figure size in cm (default: 8.6 6.0)")
    p.add_argument("--zoom", nargs=2, type=float, action="append",
                    metavar=("XMIN", "XMAX"), default=None,
                    help="Also produce a zoomed-in version of the plot "
                         "restricted to [XMIN, XMAX] (in K). Repeatable "
                         "for multiple windows, e.g.: "
                         "--zoom 2 12 --zoom 25 40 "
                         "(around Tc~7K and T_CDW~33K for NbSe2). "
                         "Each zoom is saved with a '_zoom_XMIN-XMAX' suffix.")
    p.add_argument("--fit-range", nargs=2, type=float,
                    metavar=("TMIN", "TMAX"), default=None,
                    help="Temperature window (K) for a weighted linear fit "
                         "to the normal-state R(T), e.g. --fit-range 10 25. "
                         "Fit results (slope dR/dT, intercept R(T=0)) are "
                         "always printed to stdout. Use --show-fit-on to "
                         "also draw the line on specific panels.")
    p.add_argument("--fit-branch", choices=["cooldown", "warmup"], default=None,
                    help="Restrict the linear fit to one branch "
                         "(default: use all points in the fit window)")
    p.add_argument("--show-fit-on", nargs="+", default=None, metavar="PANEL",
                    help="Panel(s) on which to draw the linear fit line. "
                         "Use 'main' for the full-range plot and/or "
                         "'zoom1', 'zoom2', ... referring to the --zoom "
                         "windows in the order they are given. The line is "
                         "automatically extrapolated to the full x-range of "
                         "each panel, so a deviation from linearity (e.g. "
                         "the CDW bump at T_CDW) stands out clearly. "
                         "Example for NbSe2: "
                         "--fit-range 10 25 --zoom 25 40 --show-fit-on zoom1")
    p.add_argument("--find-tc", action="store_true",
                    help="Determine the superconducting Tc by finding where R "
                         "drops to a fraction (--tc-criterion) of the "
                         "normal-state R. If branches are present, Tc is "
                         "computed per branch and averaged. Result is printed "
                         "to stdout (precision: 0.1 K).")
    p.add_argument("--tc-criterion", type=float, default=0.5,
                    metavar="FRACTION",
                    help="Fraction of R_normal that defines Tc "
                         "(default: 0.5 = midpoint). Use 0.9 for onset, "
                         "0.1 for near-zero resistance.")
    p.add_argument("--tc-normal-range", nargs=2, type=float,
                    metavar=("TMIN", "TMAX"), default=None,
                    help="Temperature window (K) for computing the "
                         "normal-state resistance R_normal, e.g. "
                         "--tc-normal-range 8 20 for NbSe2. "
                         "Strongly recommended; if omitted, R_normal is "
                         "estimated from the top quartile of measured T.")
    p.add_argument("--normalized", action="store_true",
                    help="Plot R/R(T_ref) instead of R, where T_ref is the "
                         "first measurement temperature after --skip-points, "
                         "rounded to the nearest integer K (e.g. 300 K). "
                         "Applies to the main plot and all zoom panels.")
    p.add_argument("--rrr", action="store_true",
                    help="Compute and print the Residual Resistance Ratio "
                         "RRR = R(T_high) / R(T_low), where T_high is the "
                         "initial temperature (rounded) and T_low is set by "
                         "--rrr-temp. Requires --rrr-temp.")
    p.add_argument("--rrr-temp", type=float, default=None, metavar="T_LOW",
                    help="Temperature just above Tc for the RRR denominator, "
                         "in K (e.g. --rrr-temp 8 for NbSe2 with Tc≈7 K). "
                         "Rounded to the nearest integer for display. "
                         "Required when --rrr is used.")
    p.add_argument("--rrr-window", type=float, default=10, metavar="DT",
                    help="Half-width in K of the averaging window around "
                         "--rrr-temp when computing R_low (default: 0.5 K). "
                         "Increase if few data points fall near that "
                         "temperature.")
    p.add_argument("--source", choices=["ppms", "rack"], default="rack",
                    help="Instrument that produced the CSV file(s): "
                         "'ppms' for PPMS/MultiVu (default), "
                         "'rack' for the custom rack with lock-in amplifier.")
    p.add_argument("--current", type=float, default=None, metavar="AMPS",
                    help="Source current in Amperes. Required when "
                         "--source rack (e.g. --current 1e-6 for 1 µA). "
                         "Used to convert the recorded voltage to resistance: "
                         "R = V / I.")
    p.add_argument("--channel", type=int, default=2, choices=[1, 2, 3],
                    metavar="{1,2,3}",
                    help="Lock-in amplifier channel to read from the rack CSV "
                         "(1→R1, 2→R2, 3→R3). Default: 2. "
                         "Only relevant when --source rack. Legacy single-channel.")
    p.add_argument("--rack-signals", nargs="*", default=None,
                    help="For source='rack', specify multiple signal columns "
                         "and optional legends, e.g. 'X1:Josephson' 'R2:bottom_flake' "
                         "X2:top_flake. Each item is 'col' or 'col:legend'. "
                         "If provided, treats as multiple datasets from one CSV, "
                         "overrides --channel and --labels for rack.")
    return p

def _add_IV_dVdI_parser(subparsers):
    p = subparsers.add_parser("IV", help="I(V) and dV/dI analysis")
    p.add_argument('csv_files', nargs='*', default=[], 
                       help='Input CSV file(s) for single file mode')
    p.add_argument("--multi-temp", type=str, nargs='?', const='',
                    help="Run multi-temperature analysis. Provide glob pattern (required)," \
                    " e.g. '~/PhD/scripts/260626_NbSe2_IV_*K*_004_processed.csv'")
    p.add_argument("--channel-dV", type=int, default=2, choices=[1,2,3],
                    help="Channel for dV (default: 2)")
    p.add_argument("--channel-dI", type=int, default=1, choices=[1,2,3],
                    help="Channel for dI (default: 1)")
    p.add_argument("--errorbars", action="store_true")
    p.add_argument("-o", "--output", default="IV_plot.pdf")
    p.add_argument("--figsize", nargs=2, type=float, default=(8.6, 6.0),
                    metavar=("WIDTH_CM", "HEIGHT_CM"),
                    help="Figure size in cm (default: 8.6 6.0)")
    p.add_argument("--source", choices=["rack"], default="rack")
    p.add_argument('--area', type=float, default=1.0, 
                       help="Cross-section area in um² for Jc calculation (default=1.0)")

    p.add_argument('--rn-criterion', type=float, default=0.5,
                       help="Fraction of max |V| to use for normal-state R_N linear fit (default=0.5)")
    p.add_argument('--analyze', action='store_true', 
                       help="Perform numerical analysis (Ic, Jc, RN, etc.) in addition to plotting")
    p.add_argument('--advanced', action='store_true', 
                       help="Perform advanced analysis (diode efficiency, Stewart-McCumber, gap, etc.)")
    p.add_argument('--plot-didv', action='store_true', help="Plot dI/dV and find Riedel peaks")
    p.add_argument('--multi-t-file', type=str, nargs='?', const='',
                       help="Single file with multiple temperatures. Provide filename for 2D dV/dI map")
    p.add_argument("--n-T-bins", type=int, default=23, metavar="N",
                help="Number of temperature bins for the 2D dV/dI grid "
                     "(default: 23). Decrease if NaN fraction is still high.")
    p.add_argument("--n-I-bins", type=int, default=200, metavar="N",
                help="Number of current bins for the 2D dV/dI grid "
                     "(default: 200). Decrease if NaN fraction is still high.")
    p.add_argument('--ic-span', type=int, default=10,
                   help="Span for jump detection in Ic (default 10)")
    p.add_argument('--outlier-thresh', type=float, default=5.0,
                   help="Outlier threshold for Ic detection (default 5.0)")
    p.add_argument('--diff-threshold', type=float, default=0.001,
                   help="Minimum difference to consider in sum for Ic detection (default 0.001)")
    return p

def _add_Hall_MR_parser(subparsers):
    """Define the `Hall_MR` subcommand."""
    p = subparsers.add_parser(
        "Hall_MR",
        help="Hall effect and magnetoresistance analysis"
    )
    # ── Input files ───────────────────────────────────────────────────────
    p.add_argument(
        "fwd_files", nargs="+",
        help="Forward-sweep CSV file(s) (−Hmax→+Hmax), one per temperature. "
             "Multiple files → multi-T analysis."
    )
    p.add_argument(
        "--bwd-files", nargs="*", default=None,
        help="Backward-sweep CSV file(s) (+Hmax→−Hmax), matched by position "
             "to fwd_files.  When provided, fwd and bwd are averaged on the "
             "common H grid before symmetrization to cancel thermal drift."
    )
    p.add_argument(
        "--comments", nargs="*", default=None,
        help="Comments .txt file(s), one per temperature (same order as "
             "fwd_files).  Used for metadata display only."
    )
    # ── Measurement configuration ─────────────────────────────────────────
    p.add_argument(
        "--hall-n", type=int, default=1,
        help="LIA channel number for the Hall (transverse) voltage (1–3). "
             "Check the comments file for the correct assignment."
    )
    p.add_argument(
        "--mr-n", type=int, default=2,
        help="LIA channel number for the longitudinal (MR) voltage."
    )
    p.add_argument(
        "--hall-col", choices=["X", "R"], default="R",
        help="Column type for Hall voltage: 'X' (in-phase, default) "
             "or 'R' (magnitude)."
    )
    p.add_argument(
        "--mr-col", choices=["X", "R"], default="R",
        help="Column type for MR voltage."
    )
    p.add_argument(
        "--current", type=float, required=True, metavar="AMPS",
        help="Source current in Amperes (e.g. --current 1e-6 for 1 µA)."
    )
    # ── Sample geometry ───────────────────────────────────────────────────
    p.add_argument(
        "--thickness", type=float, required=True, metavar="METRES",
        help="Sample thickness t in metres (e.g. --thickness 1e-9 for 1 nm). "
             "Used in ρ_yx = (V_y/I)·t."
    )
    p.add_argument(
        "--width", type=float, default=None, metavar="METRES",
        help="Sample width w in metres. With --length, enables "
             "ρ_xx = (V_x/I)·(t·w/l). If omitted, ρ_xx = (V_x/I)·t."
    )
    p.add_argument(
        "--length", type=float, default=None, metavar="METRES",
        help="Voltage-probe separation l in metres."
    )
    # ── Hall analysis options ─────────────────────────────────────────────
    p.add_argument(
        "--fit-H-range", nargs=2, type=float, default=None,
        metavar=("HMIN_OE", "HMAX_OE"),
        help="Positive-field window in Oe for the linear R_H fit, e.g. "
             "--fit-H-range 20000 50000.  Should be above H_irr(T) "
             "(identify from sweep hysteresis or from the phase θ becoming "
             "constant across H). If omitted, the full positive-H range is "
             "used — valid only if the sample is fully normal everywhere."
    )
    p.add_argument(
        "--rho-xx-field", type=float, default=0.0, metavar="FIELD_OE",
        help="Field (Oe) at which to evaluate ρ_xx for the µ_H denominator. "
             "Default: 0 (zero field, appropriate in the normal state). "
             "If the sample is SC at H=0, set this above H_irr(T)."
    )
    p.add_argument(
        "--V-cell-Z", type=float, default=_BI2201_VCELL_PER_Z_A3,
        metavar="A3_PER_CU",
        help=f"Unit-cell volume per Cu atom in Å³ for the doping estimate "
             f"p = n_H·V_cell/Z − 1. Default: {_BI2201_VCELL_PER_Z_A3} Å³ "
             f"(Bi-2201 literature value, tetragonal subcell Z=2). "
             "Pull the actual value from your crystal's characterisation "
             "for <25%% precision on p."
    )
    # ── Temperature labelling ─────────────────────────────────────────────
    p.add_argument(
        "--T-labels", nargs="*", default=None,
        help="Temperature label(s) for each scan, one per fwd_file "
             "(e.g. --T-labels 200K 150K 100K). Defaults to the mean "
             "Tsample from each file."
    )
    # ── Output / display ──────────────────────────────────────────────────
    p.add_argument(
        "-o", "--output", default="Hall_MR.pdf",
        help="Output PDF path (default: Hall_MR.pdf).  All figures for "
             "a single scan are saved as 'basename_*.pdf'; multi-T figures "
             "get an additional 'multiT_*' prefix."
    )
    p.add_argument(
        "--figsize", nargs=2, type=float, default=(8.6, 7.0),
        metavar=("WIDTH_CM", "HEIGHT_CM"),
        help="Single-panel figure size in cm (default: 8.6 7.0)."
    )
    p.add_argument(
        "--show-T", action="store_true",
        help="Add a Tsample(H) panel to the raw-voltage figure as a "
             "temperature-stability diagnostic."
    )
    p.add_argument(
        "--n-grid", type=int, default=500,
        help="Points in the common H interpolation grid (default: 500)."
    )
    return p

def _run_RT(args):
    """Execute the `RT` subcommand: call analyze_RT / plot_RT and save."""
    fig, ax = plt.subplots(figsize=(args.figsize[0] / 2.54, args.figsize[1] / 2.54),
                            constrained_layout=True)
 
    labels = args.labels or [os.path.splitext(os.path.basename(f))[0]
                              for f in args.csv_files]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    show_fit_on = set(args.show_fit_on or [])
 
    # ---- load + fit data for every input file ----
    datasets = []
    is_multi_rack = (args.source == "rack" and getattr(args, 'rack_signals', None) is not None and len(getattr(args, 'rack_signals', [])) > 0)

    for i, csv_file in enumerate(args.csv_files):
        if is_multi_rack:
            signal_list = []
            for item in args.rack_signals:
                if ':' in item:
                    col, leg = [x.strip() for x in item.split(':', 1)]
                    signal_list.append((col, leg))
                else:
                    col = item.strip()
                    signal_list.append((col, col))
            
            csv_base = os.path.splitext(os.path.basename(csv_file))[0]
            for s_idx, (sig_col, leg) in enumerate(signal_list):
                color = color_cycle[s_idx % len(color_cycle)]
                data = analyze_RT(csv_file,
                                   bridge=1,
                                   split_branches=not args.no_split,
                                   skip_points=args.skip_points,
                                   source=args.source,
                                   current=args.current,
                                   signal_col=sig_col)
                label = leg
                fit = None
                if args.fit_range is not None:
                    fit = fit_linear_RT(data, tuple(args.fit_range), branch=args.fit_branch)
                    print(f"[{csv_base} | {label}] Linear fit over {args.fit_range[0]:g}–{args.fit_range[1]:g} K ({fit['n_points']} pts, branch={args.fit_branch or 'both'})")
                if args.find_tc:
                    tc_result = find_Tc_RT(
                        data,
                        criterion=args.tc_criterion,
                        T_normal_range=(tuple(args.tc_normal_range)
                                        if args.tc_normal_range else None),
                    )
                    pct = int(round(tc_result["criterion"] * 100))
                    print(
                        f"[{label}] Tc ({pct}% criterion, "
                        f"R_normal = {tc_result['R_normal']:.4g} Ω from "
                        f"{tc_result['n_normal_points']} pts):"
                    )
                    for branch_name, Tc_val in tc_result["Tc_branches"].items():
                        val_str = f"{Tc_val:.1f} K" if Tc_val is not None else "not found"
                        print(f"  {branch_name.capitalize():<12}: {val_str}")
                    mean_str = (f"{tc_result['Tc']:.1f} K"
                                if tc_result["Tc"] is not None else "not found")
                    print(f"  {'Mean':<12}: {mean_str}")
                if args.rrr:
                    if args.rrr_temp is None:
                        raise ValueError("--rrr requires --rrr-temp T_LOW")
                    rrr_result = compute_RRR_RT(data, T_low=args.rrr_temp,
                                                window=args.rrr_window)
                    print(
                        f"[{label}] RRR = R({rrr_result['T_high']} K) / "
                        f"R({rrr_result['T_low']} K) "
                        f"= {rrr_result['R_high']:.4g} / {rrr_result['R_low']:.4g} "
                        f"({rrr_result['n_pts_low']} pts in window) "
                        f"= {rrr_result['RRR']:.2f}"
                    )
                plot_RT(data, ax=ax, show_errorbars=args.errorbars,
                        show_branches=args.branches, normalized=args.normalized,
                        label=label, color=color)
                datasets.append({"data": data, "label": label, "color": color, "fit": fit})
        else:
            # Original single-dataset per file logic
            label = labels[i] if i < len(labels) else os.path.splitext(os.path.basename(csv_file))[0]
            color = color_cycle[i % len(color_cycle)]
            data = analyze_RT(csv_file,
                               bridge=1 if args.source == "rack" else args.bridge,
                               split_branches=not args.no_split,
                               skip_points=args.skip_points,
                               source=args.source,
                               current=args.current,
                               channel=args.channel)

            fit = None
            if args.fit_range is not None:
                fit = fit_linear_RT(data, tuple(args.fit_range),
                                     branch=args.fit_branch)
                print(
                    f"[{label}] Linear fit over "
                    f"{args.fit_range[0]:g}–{args.fit_range[1]:g} K "
                    f"({fit['n_points']} pts, "
                    f"branch={args.fit_branch or 'both'}):\n"
                    f"  dR/dT  = {fit['slope']:.4g} ± {fit['slope_err']:.2g} Ω/K\n"
                    f"  R(T=0) = {fit['intercept']:.4g} ± {fit['intercept_err']:.2g} Ω"
                )

            if args.find_tc:
                tc_result = find_Tc_RT(
                    data,
                    criterion=args.tc_criterion,
                    T_normal_range=(tuple(args.tc_normal_range)
                                    if args.tc_normal_range else None),
                )
                pct = int(round(tc_result["criterion"] * 100))
                print(
                    f"[{label}] Tc ({pct}% criterion, "
                    f"R_normal = {tc_result['R_normal']:.4g} Ω from "
                    f"{tc_result['n_normal_points']} pts):"
                )
                for branch_name, Tc_val in tc_result["Tc_branches"].items():
                    val_str = f"{Tc_val:.1f} K" if Tc_val is not None else "not found"
                    print(f"  {branch_name.capitalize():<12}: {val_str}")
                mean_str = (f"{tc_result['Tc']:.1f} K"
                            if tc_result["Tc"] is not None else "not found")
                print(f"  {'Mean':<12}: {mean_str}")

            if args.rrr:
                if args.rrr_temp is None:
                    raise ValueError("--rrr requires --rrr-temp T_LOW")
                rrr_result = compute_RRR_RT(data, T_low=args.rrr_temp,
                                             window=args.rrr_window)
                print(
                    f"[{label}] RRR = R({rrr_result['T_high']} K) / "
                    f"R({rrr_result['T_low']} K) "
                    f"= {rrr_result['R_high']:.4g} / {rrr_result['R_low']:.4g} "
                    f"({rrr_result['n_pts_low']} pts in window) "
                    f"= {rrr_result['RRR']:.2f}"
                )

            plot_RT(data, ax=ax, show_errorbars=args.errorbars,
                    show_branches=args.branches, normalized=args.normalized,
                    label=label, color=color)
            datasets.append({"data": data, "label": label, "color": color, "fit": fit})
 
    # ---- optionally draw fit on the main panel ----
    if "main" in show_fit_on:
        for ds in datasets:
            if ds["fit"] is not None:
                display_fit = _scale_fit_for_plot(
                        ds["fit"],
                        ds["data"]["R_ref"] if args.normalized else 1.0,
                    )
                plot_linear_fit(display_fit, ax=ax,
                                 color=_fit_color(args.branches, ds["color"]),
                                 label=f"{ds['label']} linear fit")
        ax.legend()
    fig.savefig(args.output)
    print(f"Saved {args.output}")
 
    # ---- zoom panels ----
    y_key = "R_norm" if args.normalized else "R"
    base, ext = os.path.splitext(args.output)
    for i, (xmin, xmax) in enumerate(args.zoom or [], start=1):
        zoom_fig, zoom_ax = plt.subplots(
            figsize=(args.figsize[0] / 2.54, args.figsize[1] / 2.54),
            constrained_layout=True)
 
        for ds in datasets:
            plot_zoomed(plot_RT, ds["data"], x_key="T", y_key=y_key,
                        xlim=(xmin, xmax), ax=zoom_ax,
                        show_errorbars=args.errorbars,
                        show_branches=args.branches,
                        normalized=args.normalized,
                        label=ds["label"], color=ds["color"])
        # fit overlay: x_range=None lets plot_linear_fit default to
        # the zoom panel's x-limits, so the line is extrapolated across
        # the full window even if fit was computed on a different range.
        if f"zoom{i}" in show_fit_on:
            for ds in datasets:
                if ds["fit"] is not None:
                    display_fit = _scale_fit_for_plot(
                            ds["fit"],
                            ds["data"]["R_ref"] if args.normalized else 1.0,
                        )
                    plot_linear_fit(display_fit, ax=zoom_ax,
                                    color=_fit_color(args.branches, ds["color"]),
                                    label=f"{ds['label']} linear fit")
            zoom_ax.legend()
        zoom_path = f"{base}_zoom_{xmin:g}-{xmax:g}{ext}"
        zoom_fig.savefig(zoom_path)
        print(f"Saved {zoom_path}")

def _run_IV_dVdI(args):
    set_paper_style()
    if args.multi_temp is not None:
        if not args.multi_temp or args.multi_temp.strip() == "":
            raise ValueError("Error: --multi-temp requires a file pattern. Example:\n"
                           "  --multi-temp '~/PhD/scripts/260626_NbSe2_IV_*K*_004_processed.csv'")
        pattern = args.multi_temp
        if not glob.glob(os.path.expanduser(pattern)):
            print(f"Warning: No files matched pattern: {pattern}")
        else:
            print(f"Running multi-temperature mode with pattern: {pattern}")
            datasets = load_multi_iv_files(pattern)
            plot_multi_temp_iv(datasets, output_prefix="multi_temp")
            return
    elif args.csv_files:
        for csv_file in args.csv_files:
            data = analyze_IV_dVdI(csv_file, channel_dV=args.channel_dV, channel_dI=args.channel_dI)
            base = os.path.splitext(os.path.basename(csv_file))[0]
            plot_iv_diagnostics(data, base_name=f"{base}", figsize=args.figsize)

            if getattr(args, 'analyze', False):
                params = compute_iv_parameters(data, area_um2=args.area, 
                                               rn_criterion=args.rn_criterion,
                                               ic_span=args.ic_span,
                                               outlier_thresh=args.outlier_thresh,
                                               diff_threshold=args.diff_threshold,
                                               advanced=getattr(args, 'advanced', False),
                                               figsize=args.figsize)
                print(f"\n=== IV Analysis Results for {csv_file} ===")
                for k, v in params.items():
                    if isinstance(v, float):
                        print(f"  {k:12s}: {v:.6g}")
                    else:
                        print(f"  {k:12s}: {v}")
            
            print(f"Processed {csv_file}")
            return
    elif getattr(args, 'multi_t_file', None):
        filepath = args.multi_t_file
        print(f"Running 2D dV/dI map from single multi-T file: {filepath}")
        plot_2d_dvdi_map_from_single_file(
                                            filepath,
                                            channel_dV=args.channel_dV,
                                            channel_dI=args.channel_dI,
                                            n_T_bins=args.n_T_bins,
                                            n_I_bins=args.n_I_bins,
                                        )
        return

def _run_Hall_MR(args):
    """Execute the Hall_MR subcommand."""
    n_T = len(args.fwd_files)
 
    bwd_files  = args.bwd_files  or [None] * n_T
    comments   = args.comments   or [None] * n_T
    T_labels   = args.T_labels   or [None] * n_T
 
    if len(bwd_files) not in (0, n_T):
        raise ValueError(
            f"--bwd-files must match the number of fwd_files ({n_T}), "
            f"got {len(bwd_files)}."
        )
    bwd_files = list(bwd_files) + [None] * (n_T - len(bwd_files))
    T_labels  = list(T_labels)  + [None] * (n_T - len(T_labels))
 
    fs = (args.figsize[0] / 2.54, args.figsize[1] / 2.54)
    base, ext = os.path.splitext(args.output)
    if not ext:
        ext = ".pdf"
 
    print(f"\nHall/MR analysis — {n_T} temperature scan(s)")
 
    results = []
 
    for i, (fwd, bwd, cmt, lbl) in enumerate(
        zip(args.fwd_files, bwd_files, comments, T_labels)
    ):
        # ── Banner — printed BEFORE anything else for this temperature ──────
        T_display = lbl if lbl else "auto"
        print(f"\n{'═'*62}")
        print(f"  Scan {i+1}/{n_T}   T = {T_display}")
        print(f"  fwd : {os.path.basename(fwd)}")
        if bwd:
            print(f"  bwd : {os.path.basename(bwd)}")
        print(f"{'─'*62}")

        # ── Comments — immediately after banner, before any analysis ────────
        for cmt_path, label in [(cmt, 'comments')]:
            if cmt_path is not None and os.path.isfile(cmt_path):
                meta = parse_hall_mr_comments(cmt_path)
                print(f"  [{label}]")
                for line in meta['raw']:
                    if line.strip():
                        print(f"    {line}")
                print(f"  {'─'*56}")

        # ── Analysis (prints spikes / trim / H_irr / results internally) ────
        #    All internal messages now come after the banner and comments.
        result = analyze_hall_mr(
            fwd_source       = fwd,
            bwd_source       = bwd,
            hall_n           = args.hall_n,
            mr_n             = args.mr_n,
            hall_col         = args.hall_col,
            mr_col           = args.mr_col,
            current          = args.current,
            t                = args.thickness,
            w                = args.width,
            l                = args.length,
            V_cell_per_Z_A3  = args.V_cell_Z,
            fit_H_range_Oe   = (tuple(args.fit_H_range)
                                 if args.fit_H_range else None),
            rho_xx0_field_Oe = (args.rho_xx_field
                                 if args.rho_xx_field else None),
            T_label          = lbl,
            n_grid           = args.n_grid,
        )
        results.append(result)

        # ── Per-temperature figures ─────────────────────────────────────────
        T_str = f"{int(round(result['T_nominal_K']))}K"
        set_paper_style()

        fig_raw, _ = plot_hall_raw(result, show_T=args.show_T,
                                    figsize=args.figsize)
        path = f"{base}_raw_{T_str}{ext}"
        fig_raw.savefig(path, dpi=300)
        plt.close(fig_raw)
        print(f"  Saved {path}")

        fig_ah, ax_ah = plt.subplots(figsize=fs, constrained_layout=True)
        plot_hall_antisym(result, ax=ax_ah)
        path = f"{base}_ryx_{T_str}{ext}"
        fig_ah.savefig(path, dpi=300)
        plt.close(fig_ah)
        print(f"  Saved {path}")

        fig_rx, ax_rx = plt.subplots(figsize=fs, constrained_layout=True)
        plot_rho_xx(result, ax=ax_rx)
        path = f"{base}_rxx_{T_str}{ext}"
        fig_rx.savefig(path, dpi=300)
        plt.close(fig_rx)
        print(f"  Saved {path}")

        if np.any(np.isfinite(result['MR'])):
            fig_mr, ax_mr = plt.subplots(figsize=fs, constrained_layout=True)
            plot_MR_hall(result, ax=ax_mr)
            path = f"{base}_MR_{T_str}{ext}"
            fig_mr.savefig(path, dpi=300)
            plt.close(fig_mr)
            print(f"  Saved {path}")

        # ── Closing divider ─────────────────────────────────────────────────
        print(f"{'═'*62}")
 
    # ── Multi-temperature figures (only if >1 temperature) ────────────────
    if n_T > 1:
        set_paper_style()
 
        for plot_fn, fname_suffix in [
            (plot_RH_vs_T,   'multiT_RH'),
            (plot_nH_vs_T,   'multiT_nH'),
            (plot_muH_vs_T,  'multiT_muH'),
        ]:
            fig, ax = plt.subplots(figsize=fs, constrained_layout=True)
            plot_fn(results, ax=ax)
            path = f"{base}_{fname_suffix}{ext}"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            print(f"  Saved {path}")
 
        # cot(θ_H) vs T²
        cot_fit = fit_cot_theta_vs_T2(results)
        fig, ax = plt.subplots(figsize=fs, constrained_layout=True)
        plot_cot_theta_vs_T2(results, cot_fit=cot_fit, ax=ax)
        path = f"{base}_multiT_cotTheta{ext}"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"  Saved {path}")
 
        # Kohler
        w2 = (args.figsize[0] * 2 / 2.54, args.figsize[1] / 2.54)
        fig_k, axes_k = plt.subplots(1, 2, figsize=w2, constrained_layout=True)
        plot_kohler(results, axes=axes_k)
        path = f"{base}_multiT_Kohler{ext}"
        fig_k.savefig(path, dpi=300)
        plt.close(fig_k)
        print(f"  Saved {path}")
 
        # H/T scaling
        fig_ht, ax_ht = plt.subplots(figsize=fs, constrained_layout=True)
        plot_HT_scaling(results, ax=ax_ht)
        path = f"{base}_multiT_HT_scaling{ext}"
        fig_ht.savefig(path, dpi=300)
        plt.close(fig_ht)
        print(f"  Saved {path}")

        # ρ_xx(H_ref) vs T — linear fit
        fig_rxxT, ax_rxxT = plt.subplots(figsize=fs, constrained_layout=True)
        plot_rxx_vs_T(results, ax=ax_rxxT)
        path = f"{base}_multiT_rxx_vs_T{ext}"
        fig_rxxT.savefig(path, dpi=300)
        plt.close(fig_rxxT)
        print(f"  Saved {path}")

        # ρ_yx(H_ref) vs T — no fit
        fig_ryxT, ax_ryxT = plt.subplots(figsize=fs, constrained_layout=True)
        plot_ryx_vs_T(results, ax=ax_ryxT)
        path = f"{base}_multiT_ryx_vs_T{ext}"
        fig_ryxT.savefig(path, dpi=300)
        plt.close(fig_ryxT)
        print(f"  Saved {path}")


PLOT_TYPES = {
    "RT": (_add_RT_parser, _run_RT),
    "IV": (_add_IV_dVdI_parser, _run_IV_dVdI),
    "Hall_MR": (_add_Hall_MR_parser, _run_Hall_MR),
}

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot transport-measurement CSV files."
    )
    subparsers = parser.add_subparsers(dest="plot_type", required=True,
                                        help="Type of plot to produce")
    for add_parser, _ in PLOT_TYPES.values():
        add_parser(subparsers)

    args = parser.parse_args()

    set_paper_style()

    _, run = PLOT_TYPES[args.plot_type]
    run(args)


if __name__ == "__main__":
    main()