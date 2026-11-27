#!/usr/bin/env python3
"""
iv_conductance_plot.py
----------------------
Phase-corrected differential conductance from rack IV CSVs.

Single-file / multi-file overlay
    G(V) or dV/dI(V) with interactive zoom.

Multi-T waterfall (one CSV, many temperatures)
    G vs V and G vs I, each temperature offset vertically, coloured
    cold (dark blue) → warm (red).

Phase-corrected formula (MATLAB transcription)
    R1   = X1 / cos(theta1 * pi/180)
    dVdI = R_sense * R2 / R1 * sign(cos((theta2 - theta1)*pi/180))
    G    = 1 / dVdI

CSV columns expected
    Is, Vdmm, X1, R2, theta1, theta2
Optional: Tsample (or Tset), Hsample

Back-and-forth (bf)
    is_bf=True → one temperature occupies 2*n_points - 1 rows
    (forward N + backward N sharing the turnaround). Primary arrays
    keep the first n_points points (MATLAB ``(:,1:n_points)``).

Plot modes
    Default (no --final): interactive matplotlib window only (testing).
    --final: publication PDF only; requires --xlim, --ylim and -o.

CLI examples
------------
    # interactive test of one or more single-T files
    python iv_conductance_plot.py a.csv b.csv \\
        --R 4700 10000 --n-points 101 --bf 0 0 --x-scale 1e3

    # final PDF for the same overlay
    python iv_conductance_plot.py a.csv b.csv \\
        --R 4700 10000 --n-points 101 --bf 0 0 \\
        --x-scale 1e3 --final --xlim -2 2 --ylim 0 0.8 -o overlay.pdf

    # multi-T waterfall (file is the argument of --multi-t)
    python iv_conductance_plot.py --multi-t map.csv \\
        --R 4700 --n-points 251 --bf 1 \\
        --offset 0.05 --x-scale 1e3 1e3

    # subset 2–10 K: colours still span the full file range (stay blue)
    python iv_conductance_plot.py --multi-t map.csv \\
        --R 4700 --n-points 251 --bf 1 \\
        --T-initial 2 --T-final 10 --offset 0.05 --x-scale 1e3 1e3

    # single temperature extracted from a multi-T file (colour from full range)
    python iv_conductance_plot.py --multi-t map.csv --plot-single-T 5 \\
        --R 4700 --n-points 251 --bf 1 --x-scale 1e3 1e3

    # multi-T final PDFs — xlim/ylim are the *single-T* window (SI or scaled);
    # y is expanded by (n_T-1)*offset for the stack
    python iv_conductance_plot.py --multi-t map.csv --final \\
        --R 4700 --n-points 251 --bf 1 \\
        --offset 0.05 --x-scale 1e3 1e3 \\
        --xlim -5 5 -0.5 0.5 --ylim 0 0.8 -o waterfall.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


def _cmap(name: str = "coolwarm"):
    """Matplotlib ≥3.7 compatible colormap lookup."""
    try:
        return plt.colormaps[name]
    except (AttributeError, KeyError):
        return plt.get_cmap(name)


# ---------------------------------------------------------------------------
# Publication-style defaults
# ---------------------------------------------------------------------------

def set_paper_style() -> None:
    """Serif fonts, inward ticks – suitable for journal figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class IVData:
    """Container returned by ``load_iv`` / ``load_multi_t_iv``."""
    I: np.ndarray          # current (A), shape (n_points,)
    V: np.ndarray          # voltage (V)
    dVdI: np.ndarray       # differential resistance (Ohm)
    G: np.ndarray          # differential conductance (S) = 1/dVdI
    T: Optional[np.ndarray] = None
    T_setpoint: Optional[float] = None   # nominal T for multi-T segments
    branch: Optional[np.ndarray] = None
    is_bf: bool = False
    n_points: int = 0
    R_sense: float = 0.0
    label: str = ""
    path: str = ""
    I_full: Optional[np.ndarray] = field(default=None, repr=False)
    V_full: Optional[np.ndarray] = field(default=None, repr=False)
    dVdI_full: Optional[np.ndarray] = field(default=None, repr=False)
    G_full: Optional[np.ndarray] = field(default=None, repr=False)
    branch_full: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Loading & phase-corrected dV/dI
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> pd.DataFrame:
    """Load a rack CSV; tolerate common column-name variants."""
    df = pd.read_csv(path)
    rename = {
        "Tset": "Tsample",
        "T_set": "Tsample",
        "Temperature (K)": "Tsample",
        "Current (A)": "Is",
        "Voltage (V)": "Vdmm",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def _phase_corrected_dVdI(
    X1: np.ndarray,
    R2: np.ndarray,
    theta1_deg: np.ndarray,
    theta2_deg: np.ndarray,
    R_sense: float,
) -> np.ndarray:
    """
    MATLAB transcription:

        R1   = X1 ./ cos(theta1*pi/180);
        dVdI = R * R2 ./ R1 .* sign(cos((theta2-theta1)*pi/180));
    """
    th1 = np.deg2rad(theta1_deg)
    th2 = np.deg2rad(theta2_deg)

    with np.errstate(divide="ignore", invalid="ignore"):
        R1 = X1 / np.cos(th1)
        dVdI = R_sense * R2 / R1 * np.sign(np.cos(th2 - th1))

    bad = (~np.isfinite(dVdI)) | (np.abs(R1) < 1e-30)
    return np.where(bad, np.nan, dVdI)


def _label_branches(n_total: int, n_points: int, is_bf: bool) -> np.ndarray:
    """Forward = first n_points; if bf, remainder is backward."""
    branch = np.full(n_total, "forward", dtype=object)
    if is_bf and n_total > n_points:
        branch[n_points:] = "backward"
    return branch


def _extract_channels(
    df: pd.DataFrame,
    channel_dI: str,
    channel_dV: str,
    theta_dI: str,
    theta_dV: str,
    path: str,
) -> tuple[np.ndarray, ...]:
    required = ["Is", "Vdmm", channel_dI, channel_dV, theta_dI, theta_dV]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path}: missing columns {missing}. Available: {list(df.columns)}"
        )
    I = df["Is"].to_numpy(dtype=float)
    V = df["Vdmm"].to_numpy(dtype=float)
    X1 = df[channel_dI].to_numpy(dtype=float)
    R2 = df[channel_dV].to_numpy(dtype=float)
    th1 = df[theta_dI].to_numpy(dtype=float)
    th2 = df[theta_dV].to_numpy(dtype=float)
    T = df["Tsample"].to_numpy(dtype=float) if "Tsample" in df.columns else None
    return I, V, X1, R2, th1, th2, T


def load_iv(
    csv_path: str,
    R_sense: float,
    n_points: int,
    is_bf: bool = False,
    label: Optional[str] = None,
    channel_dI: str = "X1",
    channel_dV: str = "R2",
    theta_dI: str = "theta1",
    theta_dV: str = "theta2",
) -> IVData:
    """Load one single-T (or first-segment) IV CSV."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    df = _read_csv(csv_path)
    I_full, V_full, X1, R2, th1, th2, T_full = _extract_channels(
        df, channel_dI, channel_dV, theta_dI, theta_dV, csv_path,
    )

    dVdI_full = _phase_corrected_dVdI(X1, R2, th1, th2, R_sense)
    with np.errstate(divide="ignore", invalid="ignore"):
        G_full = np.where(np.abs(dVdI_full) > 1e-30, 1.0 / dVdI_full, np.nan)

    n_total = len(I_full)
    if is_bf:
        expected = 2 * n_points - 1
        if n_total < n_points:
            raise ValueError(
                f"{csv_path}: is_bf=True but only {n_total} rows "
                f"(need at least n_points={n_points})."
            )
        if n_total != expected:
            warnings.warn(
                f"{csv_path}: is_bf=True, expected {expected} rows "
                f"(2*n_points-1), found {n_total}. "
                f"Using first {n_points} points for primary arrays.",
                UserWarning,
                stacklevel=2,
            )
    else:
        if n_total < n_points:
            raise ValueError(
                f"{csv_path}: only {n_total} rows, n_points={n_points}."
            )

    branch_full = _label_branches(n_total, n_points, is_bf)
    sl = slice(0, n_points)

    if label is None:
        label = os.path.splitext(os.path.basename(csv_path))[0]

    return IVData(
        I=I_full[sl], V=V_full[sl], dVdI=dVdI_full[sl], G=G_full[sl],
        T=T_full[sl] if T_full is not None else None,
        branch=branch_full[sl], is_bf=is_bf, n_points=n_points,
        R_sense=R_sense, label=label, path=csv_path,
        I_full=I_full, V_full=V_full, dVdI_full=dVdI_full,
        G_full=G_full, branch_full=branch_full,
    )


# ---------------------------------------------------------------------------
# Multi-T segmentation
# ---------------------------------------------------------------------------

def _parse_temp_grid_from_filename(filepath: str) -> dict:
    """Extract T_initial, T_final, T_step from names like ``2to35K_0.5Kstep``."""
    name = os.path.basename(str(filepath))
    out: dict = {}
    m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*k", name)
    if m:
        out["T_initial"] = float(m.group(1))
        out["T_final"] = float(m.group(2))
    m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*k\s*step", name)
    if m:
        out["T_step"] = float(m.group(1))
    return out


def _build_T_grid(
    T_initial: Optional[float],
    T_final: Optional[float],
    T_step: Optional[float],
    filepath: str,
) -> np.ndarray:
    """Full temperature grid used to segment the file (not the plot window)."""
    meta = _parse_temp_grid_from_filename(filepath)
    if T_initial is None:
        T_initial = meta.get("T_initial")
    if T_final is None:
        T_final = meta.get("T_final")
    if T_step is None:
        T_step = meta.get("T_step")

    if T_initial is None or T_final is None or T_step is None or T_step <= 0:
        raise ValueError(
            "Temperature grid unknown. Pass --T-grid-initial, --T-grid-final, "
            "--T-step or encode them in the filename (e.g. 2to35K_0.5Kstep)."
        )

    n = int(round((T_final - T_initial) / T_step)) + 1
    return T_initial + np.arange(n) * T_step


def load_multi_t_iv(
    csv_path: str,
    R_sense: float,
    n_points: int,
    is_bf: bool = True,
    T_grid_initial: Optional[float] = None,
    T_grid_final: Optional[float] = None,
    T_step: Optional[float] = None,
    channel_dI: str = "X1",
    channel_dV: str = "R2",
    theta_dI: str = "theta1",
    theta_dV: str = "theta2",
) -> tuple[list[IVData], float, float]:
    """
    Split a multi-T rack IV file into one ``IVData`` per temperature setpoint.

    Always segments the **full** grid encoded in the filename (or explicit
    grid arguments). Plot-range filtering (``--T-initial`` / ``--T-final``)
    is applied later so colours stay anchored to the full data set.

    Returns
    -------
    datasets : list[IVData]
    T_full_min, T_full_max : float
        Extremes of the loaded setpoints (for colormap normalisation).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    df = _read_csv(csv_path)
    I_all, V_all, X1, R2, th1, th2, T_all = _extract_channels(
        df, channel_dI, channel_dV, theta_dI, theta_dV, csv_path,
    )
    dVdI_all = _phase_corrected_dVdI(X1, R2, th1, th2, R_sense)
    with np.errstate(divide="ignore", invalid="ignore"):
        G_all = np.where(np.abs(dVdI_all) > 1e-30, 1.0 / dVdI_all, np.nan)

    T_grid = _build_T_grid(T_grid_initial, T_grid_final, T_step, csv_path)
    stride = (2 * n_points - 1) if is_bf else n_points
    n_total = len(I_all)
    n_expected = len(T_grid) * stride

    if n_total < stride:
        raise ValueError(
            f"{csv_path}: only {n_total} rows, need at least stride={stride}."
        )
    if n_total != n_expected:
        warnings.warn(
            f"{csv_path}: {n_total} rows vs grid expectation "
            f"{len(T_grid)} × {stride} = {n_expected}. "
            f"Using min(available, grid) complete temperatures.",
            UserWarning,
            stacklevel=2,
        )

    n_seg = min(len(T_grid), n_total // stride)
    datasets: list[IVData] = []

    for k in range(n_seg):
        i0 = k * stride
        i1 = i0 + stride
        block_I = I_all[i0:i1]
        block_V = V_all[i0:i1]
        block_dVdI = dVdI_all[i0:i1]
        block_G = G_all[i0:i1]
        block_T = T_all[i0:i1] if T_all is not None else None
        branch_full = _label_branches(len(block_I), n_points, is_bf)

        sl = slice(0, n_points)
        Tset = float(T_grid[k])
        label = f"{Tset:g} K"

        datasets.append(IVData(
            I=block_I[sl], V=block_V[sl], dVdI=block_dVdI[sl], G=block_G[sl],
            T=block_T[sl] if block_T is not None else None,
            T_setpoint=Tset,
            branch=branch_full[sl], is_bf=is_bf, n_points=n_points,
            R_sense=R_sense, label=label, path=csv_path,
            I_full=block_I, V_full=block_V, dVdI_full=block_dVdI,
            G_full=block_G, branch_full=branch_full,
        ))

    Ts = np.array([float(d.T_setpoint) for d in datasets])
    T_full_min = float(Ts.min()) if len(Ts) else float("nan")
    T_full_max = float(Ts.max()) if len(Ts) else float("nan")

    print(
        f"  [multi-T] {n_seg} temperature(s) from {csv_path}  "
        f"(stride={stride}, n_points={n_points}, bf={is_bf}, "
        f"T full range [{T_full_min:g}, {T_full_max:g}] K)"
    )
    return datasets, T_full_min, T_full_max


# ---------------------------------------------------------------------------
# Plotting – single / overlay
# ---------------------------------------------------------------------------

# SI base symbols for axis quantities
_BASE_UNIT = {
    "V": "V",
    "I": "A",
    "G": "S",
    "dVdI": r"$\Omega$",
}


def _si_prefix(scale: float) -> tuple[str, str]:
    """
    Map a multiplicative scale factor to an SI prefix and a short name.

    scale multiplies the *raw SI value* before plotting, so:
        1    → no prefix (V, A, S, Ω)
        1e3  → milli  (mV, mA, mS, mΩ)
        1e6  → micro  (µV, µA, µS, µΩ)
        1e-3 → kilo   (kV, kA, kS, kΩ)
    """
    # tolerate float noise around exact decade values
    decades = {
        1e-12: ("T", "tera"),
        1e-9: ("G", "giga"),
        1e-6: ("M", "mega"),
        1e-3: ("k", "kilo"),
        1.0: ("", ""),
        1e3: ("m", "milli"),
        1e6: (r"$\mu$", "micro"),
        1e9: ("n", "nano"),
        1e12: ("p", "pico"),
    }
    for factor, (sym, name) in decades.items():
        if abs(scale - factor) / factor < 1e-9:
            return sym, name
    # arbitrary scale: no named prefix
    return "", ""


def _axis_label(quantity: str, scale: float) -> str:
    """Pretty axis label with SI prefix inferred from *scale*."""
    base = _BASE_UNIT[quantity]
    prefix, _ = _si_prefix(scale)
    if quantity == "dVdI":
        # base already contains mathtext Omega
        if prefix:
            return rf"$dV/dI$ ({prefix}$\Omega$)"
        return r"$dV/dI$ ($\Omega$)"
    symbol = {"V": "V", "I": "I", "G": "G"}[quantity]
    unit = f"{prefix}{base}" if prefix else base
    return rf"${symbol}$ ({unit})"


def _fmt_limit_with_unit(value: float, quantity: str, scale: float) -> str:
    """Format a limit value with the unit implied by *scale* (plot units)."""
    prefix, _ = _si_prefix(scale)
    base = _BASE_UNIT[quantity]
    if quantity == "dVdI":
        unit = f"{prefix}Ω" if prefix and prefix != r"$\mu$" else (
            "µΩ" if prefix == r"$\mu$" else "Ω"
        )
    else:
        plain_prefix = "µ" if prefix == r"$\mu$" else prefix
        unit = f"{plain_prefix}{base}"
    return f"{value:g} {unit}"


def plot_G_vs_V(
    data: Union[IVData, Sequence[IVData]],
    ax: Optional[Axes] = None,
    *,
    quantity: str = "G",
    x_quantity: str = "V",       # "V" or "I"
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    use_full: bool = False,
    color: Optional[Union[str, tuple]] = None,
    marker: str = ".",
    markersize: float = 6,
    linestyle: str = "none",
    alpha: float = 0.9,
    show: bool = True,
    title: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """Plot G (or dV/dI) versus V or I for one or more single-T datasets."""
    if isinstance(data, IVData):
        datasets = [data]
    else:
        datasets = list(data)

    if ax is None:
        set_paper_style()
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
    else:
        fig = ax.figure

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["k", "C0", "C1", "C2", "C3"]
    )

    for i, d in enumerate(datasets):
        c = color if color is not None else cycle[i % len(cycle)]
        if use_full and d.is_bf and d.V_full is not None:
            x_raw = d.V_full if x_quantity == "V" else d.I_full
            y_raw = d.G_full if quantity == "G" else d.dVdI_full
            br = d.branch_full
            for bname, bcolor in (("forward", c), ("backward", "C3")):
                m = br == bname
                if not np.any(m):
                    continue
                ax.plot(
                    x_raw[m] * x_scale, y_raw[m] * y_scale,
                    marker=marker, markersize=markersize,
                    linestyle=linestyle, color=bcolor, alpha=alpha,
                    label=f"{d.label} ({bname})",
                )
        else:
            x_raw = d.V if x_quantity == "V" else d.I
            y_raw = d.G if quantity == "G" else d.dVdI
            ax.plot(
                x_raw * x_scale, y_raw * y_scale,
                marker=marker, markersize=markersize,
                linestyle=linestyle, color=c,
                alpha=alpha, label=d.label,
            )

    if xlabel is None:
        xlabel = _axis_label(x_quantity, x_scale)
    if ylabel is None:
        ylabel = _axis_label(quantity, y_scale)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(datasets) > 1 or (use_full and any(d.is_bf for d in datasets)):
        ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def overlay_files(
    datasets: Sequence[IVData],
    **kwargs,
) -> tuple[Figure, Axes]:
    """Convenience wrapper: plot many ``IVData`` objects on one axes."""
    return plot_G_vs_V(datasets, **kwargs)


# ---------------------------------------------------------------------------
# Plotting – multi-T waterfall
# ---------------------------------------------------------------------------

def _T_key(d: IVData) -> float:
    if d.T_setpoint is not None:
        return float(d.T_setpoint)
    if d.T is not None and np.any(np.isfinite(d.T)):
        return float(np.nanmean(d.T))
    return 0.0


def _color_for_T(T: float, T_min: float, T_max: float):
    """Colour of temperature *T* on the full-range coolwarm scale."""
    cmap = _cmap("coolwarm")
    if not np.isfinite(T_min) or not np.isfinite(T_max) or T_max <= T_min:
        return cmap(0.0)
    return cmap((float(T) - T_min) / (T_max - T_min))


def plot_waterfall(
    datasets: Sequence[IVData],
    ax: Optional[Axes] = None,
    *,
    quantity: str = "G",
    x_quantity: str = "V",       # "V" or "I"
    offset: float = 0.05,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    use_full: bool = False,
    T_norm_min: Optional[float] = None,
    T_norm_max: Optional[float] = None,
    marker: str = ".",
    markersize: float = 3,
    linestyle: str = "none",
    alpha: float = 0.85,
    show_colorbar: bool = True,
    show: bool = True,
    title: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """
    Offset-stack of G (or dV/dI) vs V or I for a temperature series.

    Each curve is shifted by ``k * offset`` in **SI units** of the y-quantity
    (S or Ω); data and offset are then multiplied by ``y_scale`` so the
    visual spacing is preserved when the axis unit changes.

    Colours use the **full** temperature range ``[T_norm_min, T_norm_max]``
    (defaults to the min/max of *datasets*), so a cold subset stays blue.
    """
    if not datasets:
        raise ValueError("plot_waterfall: empty dataset list.")

    datasets = sorted(datasets, key=_T_key)
    Ts = np.array([_T_key(d) for d in datasets])
    t0 = float(T_norm_min) if T_norm_min is not None else float(Ts.min())
    t1 = float(T_norm_max) if T_norm_max is not None else float(Ts.max())

    if ax is None:
        set_paper_style()
        fig, ax = plt.subplots(figsize=(6.0, 7.0))
    else:
        fig = ax.figure

    for k, d in enumerate(datasets):
        if use_full and d.is_bf and d.V_full is not None:
            x_raw = d.V_full if x_quantity == "V" else d.I_full
            y_raw = d.G_full if quantity == "G" else d.dVdI_full
        else:
            x_raw = d.V if x_quantity == "V" else d.I
            y_raw = d.G if quantity == "G" else d.dVdI

        # offset is SI; scale data and offset together
        y_plot = (y_raw + k * offset) * y_scale
        ax.plot(
            x_raw * x_scale, y_plot,
            marker=marker, markersize=markersize,
            linestyle=linestyle,
            color=_color_for_T(_T_key(d), t0, t1),
            alpha=alpha,
            label=d.label,
        )

    if xlabel is None:
        xlabel = _axis_label(x_quantity, x_scale)
    if ylabel is None:
        base = _axis_label(quantity, y_scale)
        if offset:
            off_scaled = offset * y_scale
            ylabel = base + f"  + offset ({off_scaled:g} / curve)"
        else:
            ylabel = base

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)

    if show_colorbar:
        norm = Normalize(vmin=t0, vmax=t1 if t1 > t0 else t0 + 1.0)
        sm = plt.cm.ScalarMappable(cmap=_cmap("coolwarm"), norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
        cbar.set_label(r"$T$ (K)")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_bool_token(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(
        f"invalid boolean '{s}' (use 1/0, true/false, yes/no)"
    )


def _expand_per_file(values: Sequence, n_files: int, name: str) -> list:
    if len(values) == 1:
        return list(values) * n_files
    if len(values) == n_files:
        return list(values)
    raise SystemExit(
        f"{name}: pass either 1 value (applied to all files) "
        f"or exactly {n_files} values (one per file). Got {len(values)}."
    )


def _scales_VI(values: Optional[Sequence[float]], name: str, default: float = 1.0) -> tuple[float, float]:
    """
    Parse --x-scale / --y-scale style args for the V-plot and I-plot.

    0 values → (default, default)
    1 value  → broadcast to both plots
    2 values → (scale_V, scale_I)
    """
    if values is None or len(values) == 0:
        return default, default
    if len(values) == 1:
        return float(values[0]), float(values[0])
    if len(values) == 2:
        return float(values[0]), float(values[1])
    raise SystemExit(
        f"{name}: pass 1 value (both plots) or 2 values "
        f"(V-plot, I-plot). Got {len(values)}."
    )


def _limits_VI(
    values: Optional[Sequence[float]],
    name: str,
) -> tuple[Optional[list[float]], Optional[list[float]]]:
    """
    Parse --xlim / --ylim for the V-plot and I-plot.

    None / empty → (None, None)
    2 values     → same [min, max] for both plots
    4 values     → [Vmin, Vmax], [Imin, Imax]

    For waterfall plots these are interpreted as the axis window of **one
    single temperature** (no offset).  Vertical limits are expanded later
    with ``_expand_ylim_for_waterfall``.
    """
    if values is None or len(values) == 0:
        return None, None
    if len(values) == 2:
        pair = [float(values[0]), float(values[1])]
        return pair, list(pair)
    if len(values) == 4:
        return (
            [float(values[0]), float(values[1])],
            [float(values[2]), float(values[3])],
        )
    raise SystemExit(
        f"{name}: pass 2 values (shared by both plots) or 4 values "
        f"(Vmin Vmax Imin Imax). Got {len(values)}."
    )


def _expand_ylim_for_waterfall(
    ylim_single: Optional[Sequence[float]],
    n_curves: int,
    offset: float,
    y_scale: float = 1.0,
) -> Optional[list[float]]:
    """
    Map a single-T y-window (in *scaled* plot units) onto the waterfall stack.

    Curve k is drawn at ``(y_SI + k * offset_SI) * y_scale``, so the top of
    the stack sits ``(n_curves - 1) * offset * y_scale`` above the single-T
    ymax (offset is SI; ylim is already scaled).
    """
    if ylim_single is None:
        return None
    ymin, ymax = float(ylim_single[0]), float(ylim_single[1])
    if n_curves < 1:
        return [ymin, ymax]
    return [ymin, ymax + (n_curves - 1) * offset * y_scale]


def _apply_limits(ax: Axes, xlim, ylim) -> None:
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _finish_figure(
    fig: Figure,
    ax: Axes,
    *,
    final: bool,
    output: Optional[str],
    xlim,
    ylim,
    no_show: bool,
) -> None:
    """
    --final True  → require xlim/ylim/output; save PDF; never show.
    --final False → interactive only; do not save PDF (testing mode).
    """
    _apply_limits(ax, xlim, ylim)

    if final:
        if output is None:
            raise SystemExit("--final requires -o/--output.")
        if xlim is None or ylim is None:
            raise SystemExit(
                "--final requires --xlim and --ylim "
                "(single-T window; y is expanded for waterfalls automatically)."
            )
        fig.savefig(output)
        print(f"Saved {output}")
        plt.close(fig)
        return

    # testing / interactive mode: never write the publication PDF
    if output is not None:
        print(
            f"Note: -o {output} ignored without --final "
            "(interactive testing mode)."
        )
    if not no_show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _report_limits(
    which: str,
    xlim,
    ylim,
    *,
    x_quantity: str,
    y_quantity: str,
    x_scale: float,
    y_scale: float,
    expanded: bool = False,
) -> None:
    """Print axis windows with explicit units (plot units after scaling)."""
    parts = []
    if xlim is not None:
        parts.append(
            f"xlim=[{_fmt_limit_with_unit(xlim[0], x_quantity, x_scale)}, "
            f"{_fmt_limit_with_unit(xlim[1], x_quantity, x_scale)}]"
        )
    if ylim is not None:
        tag = "ylim (waterfall)" if expanded else "ylim (single-T)"
        parts.append(
            f"{tag}=[{_fmt_limit_with_unit(ylim[0], y_quantity, y_scale)}, "
            f"{_fmt_limit_with_unit(ylim[1], y_quantity, y_scale)}]"
        )
    if parts:
        print(f"  [limits {which}] " + "  ".join(parts))
    # always remind of the SI base of the raw arrays
    print(
        f"  [units] raw data in SI: V→V, I→A, G→S, dV/dI→Ω; "
        f"plot scales x={x_scale:g}, y={y_scale:g}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Phase-corrected differential conductance from rack IV CSVs. "
            "Default: interactive testing. Use --final for publication PDF "
            "(requires --xlim, --ylim, -o)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "csv_files", nargs="*", default=[],
        help="Single-T measurement CSV path(s). Not used with --multi-t.",
    )
    p.add_argument(
        "--multi-t", metavar="CSV", default=None,
        help=(
            "Path to a multi-temperature IV CSV. Enables the waterfall "
            "(or --plot-single-T) pipeline on that file."
        ),
    )
    p.add_argument(
        "--plot-single-T", type=float, default=None, metavar="T_K",
        help=(
            "With --multi-t: plot only this temperature (K), closest setpoint. "
            "Colour is taken from the full data-set range. No offset."
        ),
    )

    p.add_argument(
        "--R", "--R-sense", type=float, nargs="+", required=True, dest="R_sense",
        metavar="OHM",
        help="Sense resistor (Ohm). One value, or one per file (single-T mode).",
    )
    p.add_argument(
        "--n-points", type=int, nargs="+", required=True, metavar="N",
        help="Points per sweep direction. One value, or one per file.",
    )
    p.add_argument(
        "--bf", type=_parse_bool_token, nargs="+", default=["0"], metavar="BOOL",
        help="Back-and-forth flag(s). One value, or one per file.",
    )

    # plot window in temperature (filters curves; does not change colormap)
    p.add_argument(
        "--T-initial", type=float, default=None,
        help="Lowest T (K) to *plot* from a multi-T file (colour still uses full range).",
    )
    p.add_argument(
        "--T-final", type=float, default=None,
        help="Highest T (K) to *plot* from a multi-T file (colour still uses full range).",
    )
    # optional overrides for the segmentation grid when the filename lacks it
    p.add_argument(
        "--T-grid-initial", type=float, default=None,
        help="Full-grid start (K) for segmentation if not in the filename.",
    )
    p.add_argument(
        "--T-grid-final", type=float, default=None,
        help="Full-grid end (K) for segmentation if not in the filename.",
    )
    p.add_argument(
        "--T-step", type=float, default=None,
        help="T step (K) for the full segmentation grid if not in the filename.",
    )
    p.add_argument(
        "--offset", type=float, default=0.05,
        help=(
            "Vertical offset per temperature curve in SI units of the "
            "y-quantity (S for G, Ω for dV/dI). Scaled together with --y-scale."
        ),
    )

    p.add_argument("--quantity", choices=["G", "dVdI"], default="G",
                   help="Vertical axis quantity.")
    p.add_argument(
        "--x-scale", type=float, nargs="+", default=[1.0],
        metavar="S",
        help=(
            "Multiply raw SI x-data before plotting. "
            "1 → V/A, 1e3 → mV/mA, 1e6 → µV/µA. "
            "One value for both plots, or two (V-plot, I-plot)."
        ),
    )
    p.add_argument(
        "--y-scale", type=float, nargs="+", default=[1.0],
        metavar="S",
        help=(
            "Multiply raw SI y-data before plotting. "
            "1 → S or Ω, 1e3 → mS or mΩ, 1e6 → µS or µΩ. "
            "One value for both plots, or two (V-plot, I-plot)."
        ),
    )
    p.add_argument(
        "--xlim", type=float, nargs="+", default=None, metavar="X",
        help=(
            "X-axis window for a *single temperature* in *scaled* units: "
            "2 values (shared) or 4 values (Vmin Vmax Imin Imax)."
        ),
    )
    p.add_argument(
        "--ylim", type=float, nargs="+", default=None, metavar="Y",
        help=(
            "Y-axis window for a *single temperature* (no offset) in *scaled* units: "
            "2 values (shared) or 4 values (V-plot, I-plot). "
            "Waterfalls expand ymax by (n_T-1)*--offset."
        ),
    )
    p.add_argument("--labels", nargs="+", default=None,
                   help="Legend labels (one per file; single-T mode).")
    p.add_argument("--use-full", action="store_true",
                   help="Plot both forward and backward branches when bf.")

    p.add_argument(
        "--final", action="store_true",
        help=(
            "Publication mode: skip interactive window, write PDF only. "
            "Requires --xlim, --ylim and -o/--output."
        ),
    )
    p.add_argument(
        "-o", "--output", default=None,
        help=(
            "Output path for the PDF/PNG. Used only with --final. "
            "For multi-T, this is the G-vs-V file; G-vs-I gets suffix _vs_I."
        ),
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="Do not open an interactive window (testing mode only).",
    )
    return p


def _filter_by_T_window(
    datasets: list[IVData],
    T_lo: Optional[float],
    T_hi: Optional[float],
) -> list[IVData]:
    if T_lo is None and T_hi is None:
        return datasets
    out = []
    for d in datasets:
        T = _T_key(d)
        if T_lo is not None and T < T_lo - 1e-9:
            continue
        if T_hi is not None and T > T_hi + 1e-9:
            continue
        out.append(d)
    return out


def _pick_single_T(datasets: list[IVData], T_target: float) -> IVData:
    if not datasets:
        raise SystemExit("No temperature segments available.")
    Ts = np.array([_T_key(d) for d in datasets])
    idx = int(np.argmin(np.abs(Ts - T_target)))
    chosen = datasets[idx]
    print(
        f"  [single-T] requested {T_target:g} K → closest setpoint "
        f"{_T_key(chosen):g} K (index {idx}/{len(datasets)-1})"
    )
    return chosen


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    x_scale_V, x_scale_I = _scales_VI(args.x_scale, "--x-scale")
    y_scale_V, y_scale_I = _scales_VI(args.y_scale, "--y-scale")
    xlim_V, xlim_I = _limits_VI(args.xlim, "--xlim")
    ylim_V, ylim_I = _limits_VI(args.ylim, "--ylim")
    y_qty = args.quantity

    # ----- multi-T pipeline -----
    if args.multi_t is not None:
        if args.csv_files:
            raise SystemExit(
                "Do not pass positional CSV files together with --multi-t CSV; "
                "the multi-T file is the argument of --multi-t."
            )
        R = _expand_per_file(args.R_sense, 1, "--R")[0]
        n_pts = _expand_per_file(args.n_points, 1, "--n-points")[0]
        is_bf = _expand_per_file(args.bf, 1, "--bf")[0]

        all_datasets, T_full_min, T_full_max = load_multi_t_iv(
            args.multi_t,
            R_sense=R,
            n_points=n_pts,
            is_bf=is_bf,
            T_grid_initial=args.T_grid_initial,
            T_grid_final=args.T_grid_final,
            T_step=args.T_step,
        )
        if not all_datasets:
            raise SystemExit("No temperature segments extracted.")

        # --- single temperature from multi-T file ---
        if args.plot_single_T is not None:
            d = _pick_single_T(all_datasets, args.plot_single_T)
            color = _color_for_T(_T_key(d), T_full_min, T_full_max)

            fig_v, ax_v = plot_G_vs_V(
                d, quantity=args.quantity, x_quantity="V",
                x_scale=x_scale_V, y_scale=y_scale_V,
                use_full=args.use_full, color=color, show=False,
                title=rf"{args.quantity}$(V)$ at {_T_key(d):g} K",
            )
            _report_limits(
                "V", xlim_V, ylim_V,
                x_quantity="V", y_quantity=y_qty,
                x_scale=x_scale_V, y_scale=y_scale_V,
            )
            _finish_figure(
                fig_v, ax_v, final=args.final, output=args.output,
                xlim=xlim_V, ylim=ylim_V, no_show=args.no_show,
            )

            fig_i, ax_i = plot_G_vs_V(
                d, quantity=args.quantity, x_quantity="I",
                x_scale=x_scale_I, y_scale=y_scale_I,
                use_full=args.use_full, color=color, show=False,
                title=rf"{args.quantity}$(I)$ at {_T_key(d):g} K",
            )
            out_i = None
            if args.output:
                root, ext = os.path.splitext(args.output)
                out_i = f"{root}_vs_I{ext or '.pdf'}"
            _report_limits(
                "I", xlim_I, ylim_I,
                x_quantity="I", y_quantity=y_qty,
                x_scale=x_scale_I, y_scale=y_scale_I,
            )
            _finish_figure(
                fig_i, ax_i, final=args.final, output=out_i,
                xlim=xlim_I, ylim=ylim_I, no_show=args.no_show,
            )
            return

        # --- waterfall (possibly T-windowed) ---
        datasets = _filter_by_T_window(
            all_datasets, args.T_initial, args.T_final,
        )
        if not datasets:
            raise SystemExit(
                "No temperatures left after --T-initial/--T-final filter."
            )
        if args.T_initial is not None or args.T_final is not None:
            print(
                f"  [plot window] showing {len(datasets)}/{len(all_datasets)} "
                f"curves in "
                f"[{args.T_initial if args.T_initial is not None else T_full_min:g}, "
                f"{args.T_final if args.T_final is not None else T_full_max:g}] K; "
                f"colours normalised to full range "
                f"[{T_full_min:g}, {T_full_max:g}] K"
            )

        n_T = len(datasets)
        ylim_V_plot = _expand_ylim_for_waterfall(
            ylim_V, n_T, args.offset, y_scale=y_scale_V,
        )
        ylim_I_plot = _expand_ylim_for_waterfall(
            ylim_I, n_T, args.offset, y_scale=y_scale_I,
        )

        if ylim_V is not None:
            print(
                f"  [limits] single-T ylim V "
                f"[{_fmt_limit_with_unit(ylim_V[0], y_qty, y_scale_V)}, "
                f"{_fmt_limit_with_unit(ylim_V[1], y_qty, y_scale_V)}] "
                f"→ waterfall "
                f"[{_fmt_limit_with_unit(ylim_V_plot[0], y_qty, y_scale_V)}, "
                f"{_fmt_limit_with_unit(ylim_V_plot[1], y_qty, y_scale_V)}] "
                f"({n_T} curves, offset={args.offset:g} SI "
                f"→ {args.offset * y_scale_V:g} plot units)"
            )
        if ylim_I is not None and ylim_I != ylim_V:
            print(
                f"  [limits] single-T ylim I "
                f"[{_fmt_limit_with_unit(ylim_I[0], y_qty, y_scale_I)}, "
                f"{_fmt_limit_with_unit(ylim_I[1], y_qty, y_scale_I)}] "
                f"→ waterfall "
                f"[{_fmt_limit_with_unit(ylim_I_plot[0], y_qty, y_scale_I)}, "
                f"{_fmt_limit_with_unit(ylim_I_plot[1], y_qty, y_scale_I)}] "
                f"({n_T} curves, offset={args.offset:g} SI "
                f"→ {args.offset * y_scale_I:g} plot units)"
            )

        title_v = r"$G(V)$" if args.quantity == "G" else r"$dV/dI(V)$"
        title_i = r"$G(I)$" if args.quantity == "G" else r"$dV/dI(I)$"

        fig_v, ax_v = plot_waterfall(
            datasets,
            quantity=args.quantity, x_quantity="V",
            offset=args.offset,
            x_scale=x_scale_V, y_scale=y_scale_V,
            use_full=args.use_full,
            T_norm_min=T_full_min, T_norm_max=T_full_max,
            show=False,
            title=title_v,
        )
        _report_limits(
            "V", xlim_V, ylim_V_plot,
            x_quantity="V", y_quantity=y_qty,
            x_scale=x_scale_V, y_scale=y_scale_V, expanded=True,
        )
        _finish_figure(
            fig_v, ax_v, final=args.final, output=args.output,
            xlim=xlim_V, ylim=ylim_V_plot, no_show=args.no_show,
        )

        fig_i, ax_i = plot_waterfall(
            datasets,
            quantity=args.quantity, x_quantity="I",
            offset=args.offset,
            x_scale=x_scale_I, y_scale=y_scale_I,
            use_full=args.use_full,
            T_norm_min=T_full_min, T_norm_max=T_full_max,
            show=False,
            title=title_i,
        )
        out_i = None
        if args.output:
            root, ext = os.path.splitext(args.output)
            out_i = f"{root}_vs_I{ext or '.pdf'}"
        _report_limits(
            "I", xlim_I, ylim_I_plot,
            x_quantity="I", y_quantity=y_qty,
            x_scale=x_scale_I, y_scale=y_scale_I, expanded=True,
        )
        _finish_figure(
            fig_i, ax_i, final=args.final, output=out_i,
            xlim=xlim_I, ylim=ylim_I_plot, no_show=args.no_show,
        )
        return

    # ----- single-T / multi-file overlay (vs V) -----
    if not args.csv_files:
        raise SystemExit(
            "Pass one or more single-T CSV files, or use --multi-t FILE."
        )
    if args.plot_single_T is not None:
        raise SystemExit("--plot-single-T requires --multi-t FILE.")

    n = len(args.csv_files)
    labels = args.labels
    if labels is not None and len(labels) != n:
        raise SystemExit("--labels must have one entry per csv file.")

    R_list = _expand_per_file(args.R_sense, n, "--R")
    npts_list = _expand_per_file(args.n_points, n, "--n-points")
    bf_list = _expand_per_file(args.bf, n, "--bf")

    datasets = []
    for i, path in enumerate(args.csv_files):
        lab = labels[i] if labels else None
        d = load_iv(
            path,
            R_sense=R_list[i],
            n_points=npts_list[i],
            is_bf=bf_list[i],
            label=lab,
        )
        datasets.append(d)
        print(
            f"Loaded {path}: n={d.n_points}, is_bf={d.is_bf}, "
            f"R_sense={d.R_sense:g} Ohm, "
            f"V range [{d.V.min():.4g}, {d.V.max():.4g}] V"
        )

    fig, ax = plot_G_vs_V(
        datasets,
        quantity=args.quantity,
        x_quantity="V",
        x_scale=x_scale_V,
        y_scale=y_scale_V,
        use_full=args.use_full,
        show=False,
    )
    _report_limits(
        "V", xlim_V, ylim_V,
        x_quantity="V", y_quantity=y_qty,
        x_scale=x_scale_V, y_scale=y_scale_V,
    )
    _finish_figure(
        fig, ax,
        final=args.final, output=args.output,
        xlim=xlim_V, ylim=ylim_V, no_show=args.no_show,
    )


if __name__ == "__main__":
    main()
