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

def _load_csv_rack(csv_path, signal_col, current):
    df = pd.read_csv(csv_path)

    T_col = "Tsample"
    V_col = signal_col
    for col in (T_col, V_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")

    return pd.DataFrame({
        "Temperature (K)": df[T_col].astype(float).values,
        "Bridge 1 Resistivity (Ohm)": df[V_col].astype(float).values / current,
        "Bridge 1 Std. Dev. (Ohm)": 0.0,
    })

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
# I(V) and dV/dI: current vs voltage
# ==================================================================

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
                         "and optional legends, e.g. X1:Josephson R2:bottom_flake "
                         "X2:top_flake. Each item is 'col' or 'col:legend'. "
                         "If provided, treats as multiple datasets from one CSV, "
                         "overrides --channel and --labels for rack.")
    return p

def _add_IV_dVdI_parser(subparsers):
    p = subparsers.add_parser("IV_dVdI", help="I(V) and dV/dI")
    p.add_argument("csv_files", nargs="+",
                    help="One or more measurement CSV file(s) to plot together")
    p.add_argument("--errorbars", action="store_true",
                    help="Plot with error bars (default: off)")
    p.add_argument("--branches", action="store_true",
                    help="Color forward/backward branches separately")
    p.add_argument("--skip-points", type=int, default=0,
                    help="Discard this many points from the start of the run "
                         "(e.g. turning on the sourcemeter and the current jumping from 0 to a given value)")
    p.add_argument("--labels", nargs="+", default=None,
                    help="Legend label(s), one per CSV file "
                         "(defaults to each file's name)")
    p.add_argument("-o", "--output", default="RT_plot.pdf",
                    help="Output figure path (default: RT_plot.pdf)")
    p.add_argument("--figsize", nargs=2, type=float, default=(8.6, 6.0),
                    metavar=("WIDTH_CM", "HEIGHT_CM"),
                    help="Figure size in cm (default: 8.6 6.0)")
    p.add_argument("--find-Rn", action="store_true",
                    help="Determine the normal state resistance." \
                    "defined as the slope of the I(V) curve in the" \
                    "normal region at positive bias.")
    p.add_argument("--find-Jc", action="store_true",
                    help="Determine the critical current density in kA/cm2")
    p.add_argument("--find-IcRn", action="store_true",
                    help="Calculate the Josephson coupling strength")
    p.add_argument("--normalized", action="store_true",
                    help="Calculate dV/dI normalized with Rn")
    p.add_argument("--source", choices=["ppms", "rack"], default="ppms",
                    help="Instrument that produced the CSV file(s): "
                         "'ppms' for PPMS/MultiVu (default), "
                         "'rack' for the custom rack with lock-in amplifier.")
    p.add_argument("--current", type=float, default=None, metavar="AMPS",
                    help="Source current in Amperes. Required when "
                         "--source rack (e.g. --current 1e-6 for 1 µA). "
                         "Used to convert the recorded voltage to resistance: "
                         "R = V / I.")
    p.add_argument("--channel-dV", type=int, default=2, choices=[1, 2, 3],
                    metavar="{1,2,3}",
                    help="Lock-in amplifier channel to read from the rack CSV "
                         "(1→R1/X1, 2→R2/X2, 3→R3/X3). Default: 2. "
                         "Only relevant when --source rack.")
    p.add_argument("--channel-dI", type=int, default=1, choices=[1, 2, 3],
                    metavar="{1,2,3}",
                    help="Lock-in amplifier channel to read from the rack CSV "
                         "(1→R1/X1, 2→R2/X2, 3→R3/X3). Default: 1. "
                         "Only relevant when --source rack.")
    p.add_argument("--compute-dIdV", action="store_true",
                    help="From the values of the dV and dI data, numerically" \
                    "compute the differenciated dI/dV values and plot them " \
                    "in function of the bias voltage calculated from the sourced current" \
                    "Is with the value of the circuit resistance given as an " \
                    "input by the user.")
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
    print()

PLOT_TYPES = {
    "RT": (_add_RT_parser, _run_RT),
    "IV_dVdI": (_add_IV_dVdI_parser, _run_IV_dVdI),
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