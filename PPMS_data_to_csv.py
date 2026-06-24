#!/usr/bin/env python3
"""
ppms_processor.py
=================
Processes Quantum Design PPMS MultiVu R(T) data files (.dat format).

Why two output files?
---------------------
  - OriginPro understands the native PPMS .dat format (header + CSV data block)
    natively; modifying the file risks breaking that recognition.  A byte-exact
    copy is therefore the safest Origin deliverable.
  - Python / pandas works best with a plain CSV that contains no header block
    and has proper NaN values — quite different requirements.

  The two formats are incompatible enough that a single file cannot serve both
  purposes optimally, so this script produces one of each.

Outputs
-------
  <stem>_origin.dat   — Byte-exact copy of the input.  Open directly in
                        OriginPro (File → Open or drag-and-drop).  Origin
                        recognises the Quantum Design format automatically and
                        imports all numeric columns.

  <stem>_python.csv   — Plain CSV (no PPMS header block).  Load with:
                            import pandas as pd
                            df = pd.read_csv("<stem>_python.csv")
                        All numeric columns are already cast to float64/int64.
                        Empty cells become NaN.

Usage
-----
  python ppms_processor.py <input.dat> [output_directory]

  output_directory defaults to the same directory as the input file.

Programmatic use
----------------
  from ppms_processor import process_ppms_file
  df, origin_path, python_path = process_ppms_file("mydata.dat")
"""

import io
import re
import sys
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Core parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_ppms_dat(filepath: Path):
    """
    Parse a Quantum Design PPMS MultiVu .dat file.

    The file is structured as:

        [Header]
        KEY, value, ...
        ...
        [Data]
        ColName1, ColName2, ...     ← first non-blank line after [Data]
        row1_val1, row1_val2, ...
        ...

    Parameters
    ----------
    filepath : Path

    Returns
    -------
    raw_bytes : bytes
        Original file bytes (for the bit-exact Origin copy).
    header_meta : dict
        Key→value pairs extracted from INFO lines in the header block.
    column_names : list[str]
        Column names in the order they appear in the file.
    df : pd.DataFrame
        Parsed data with numeric dtypes where applicable.
    """
    filepath = Path(filepath)
    raw_bytes = filepath.read_bytes()

    # Decode; errors='replace' handles any non-UTF-8 bytes gracefully
    text = raw_bytes.decode("utf-8", errors="replace")

    # ── Split header / data ──────────────────────────────────────────────────
    split_re = re.compile(r"\[Data\]", re.IGNORECASE)
    parts = split_re.split(text, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            "Could not locate a [Data] section in the file.  "
            "Is this a valid PPMS .dat file?"
        )
    header_text, data_text = parts

    # ── Extract useful metadata from header INFO lines ───────────────────────
    header_meta = {}
    for line in header_text.splitlines():
        line = line.strip().rstrip("\r")
        if line.upper().startswith("INFO,"):
            parts_info = line.split(",", 2)
            if len(parts_info) == 3:
                key = parts_info[2].strip()
                val = parts_info[1].strip()
                if key and val:
                    header_meta[key] = val
        elif line.upper().startswith("FILEOPENTIME,"):
            parts_ft = line.split(",", 3)
            if len(parts_ft) >= 4:
                header_meta["File date/time"] = (
                    f"{parts_ft[2].strip()} {parts_ft[3].strip()}"
                )

    # ── Locate column-header line (first non-empty line after [Data]) ────────
    data_lines = data_text.splitlines()
    col_idx = next(
        (i for i, ln in enumerate(data_lines) if ln.strip()),
        None,
    )
    if col_idx is None:
        raise ValueError("No data rows found after the [Data] tag.")

    col_header_raw = data_lines[col_idx].strip().rstrip("\r")
    column_names = [c.strip() for c in col_header_raw.split(",")]

    # ── Build a clean string for pandas to parse ─────────────────────────────
    body_lines = [ln.rstrip("\r") for ln in data_lines[col_idx + 1:] if ln.strip()]
    csv_string = col_header_raw + "\n" + "\n".join(body_lines)

    df = pd.read_csv(
        io.StringIO(csv_string),
        dtype=str,           # read all as str first → convert below
        na_values=[""],
        keep_default_na=True,
        skipinitialspace=True,
        low_memory=False,
    )

    # ── Numeric conversion ────────────────────────────────────────────────────
    # 'Comment' stays as string; everything else gets converted to numeric
    for col in df.columns:
        if col.strip().lower() == "comment":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return raw_bytes, header_meta, column_names, df


# ─────────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(out_path: Path, df: pd.DataFrame) -> None:
    """
    Write a clean CSV suitable for pd.read_csv() with zero extra arguments.

    - No PPMS header block (only column names + data)
    - Standard comma delimiter
    - Empty cells for NaN / missing values
    - Numeric dtypes preserved from the parsed DataFrame
    """
    df.to_csv(out_path, index=False, na_rep="")
    n_valid_cols = df.notna().any(axis=0).sum()
    print(
        f"  [Python]   {out_path.name}  "
        f"({len(df):,} rows × {len(df.columns)} cols, "
        f"{n_valid_cols} cols with data)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(header_meta: dict, df: pd.DataFrame) -> None:
    LINE = "─" * 60

    print(f"\n{LINE}")
    print("  Sample / measurement metadata")
    print(LINE)
    for k, v in header_meta.items():
        print(f"  {k:<30}: {v}")

    print(f"\n{LINE}")
    print("  Data overview")
    print(LINE)
    print(f"  Total data rows      : {len(df):,}")

    # Temperature range
    t_col = "Temperature (K)"
    if t_col in df.columns:
        t = df[t_col].dropna()
        print(f"  Temperature range    : {t.min():.3f} K  →  {t.max():.3f} K")

    # Magnetic field (usually constant in R(T) sweeps)
    h_col = "Magnetic Field (Oe)"
    if h_col in df.columns:
        h_vals = df[h_col].dropna().unique()
        if len(h_vals) == 1:
            print(f"  Magnetic field       : {h_vals[0]:.3f} Oe (constant)")
        else:
            print(f"  Magnetic field       : {df[h_col].min():.3f} – {df[h_col].max():.3f} Oe")

    # Bridge data
    print()
    for b in range(1, 5):
        # Resistivity column (units differ per bridge)
        r_col = next(
            (c for c in df.columns if f"Bridge {b} Resistivity" in c),
            None,
        )
        res_col = next(
            (c for c in df.columns if f"Bridge {b} Resistance" in c),
            None,
        )
        if r_col is None:
            continue
        n = df[r_col].notna().sum()
        if n == 0:
            print(f"  Bridge {b}              : no data")
            continue
        unit_match = re.search(r"\((.+?)\)", r_col)
        unit = unit_match.group(1) if unit_match else "?"
        vals = df[r_col].dropna()
        print(
            f"  Bridge {b} resistivity : {n:,} pts  |  "
            f"{vals.min():.5g} – {vals.max():.5g} {unit}"
        )
        exc_col = next(
            (c for c in df.columns if f"Bridge {b} Excitation" in c), None
        )
        if exc_col and df[exc_col].notna().any():
            i_vals = df[exc_col].dropna().unique()
            i_str = ", ".join(f"{v:.4g}" for v in sorted(i_vals)[:5])
            print(f"            excitation : {i_str} µA")

    # Columns with no data at all
    empty = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty:
        print(f"\n  Empty columns ({len(empty)}/{len(df.columns)}):", ", ".join(empty))
    print(f"{LINE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_ppms_file(input_path, output_dir=None):
    """
    Process a PPMS .dat file and write both output formats.

    Parameters
    ----------
    input_path : str or Path
    output_dir : str or Path, optional
        Where to write the two output files.
        Defaults to the same directory as *input_path*.

    Returns
    -------
    df : pd.DataFrame
        Parsed data (useful when calling this function from another script).
    origin_path : Path
    python_path : Path

    Example
    -------
    >>> from ppms_processor import process_ppms_file
    >>> df, _, _ = process_ppms_file("2026-06-04-NbSe2_RT.dat")
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(df["Temperature (K)"], df["Bridge 1 Resistivity (Ohm)"])
    """
    input_path = Path(input_path).resolve()
    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    print(f"Input  : {input_path}")

    # Parse
    raw_bytes, header_meta, column_names, df = parse_ppms_dat(input_path)

    # Summary
    print_summary(header_meta, df)

    # Write outputs
    print("Outputs:")
    origin_path = output_dir / f"{stem}_origin.dat"
    python_path = output_dir / f"{stem}.csv"
    write_csv(python_path, df)
    print()

    return df, origin_path, python_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_path.exists():
        print(f"Error: file not found — {input_path}", file=sys.stderr)
        sys.exit(1)

    process_ppms_file(input_path, output_dir)


if __name__ == "__main__":
    main()