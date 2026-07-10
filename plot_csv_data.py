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
    p.add_argument("--rrr-window", type=float, default=0.5, metavar="DT",
                    help="Half-width in K of the averaging window around "
                         "--rrr-temp when computing R_low (default: 0.5 K). "
                         "Increase if few data points fall near that "
                         "temperature.")
    p.add_argument("--source", choices=["ppms", "rack"], default="ppms",
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

    for csv_file in args.csv_files:
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
            label = labels[csv_file]
            color = color_cycle[csv_file % len(color_cycle)]
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

PLOT_TYPES = {
    "RT": (_add_RT_parser, _run_RT),
    "IV": (_add_IV_dVdI_parser, _run_IV_dVdI),
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