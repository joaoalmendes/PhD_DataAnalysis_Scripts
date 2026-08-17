import scipy.io
import numpy as np
import pandas as pd
import os
from pathlib import Path


def _clean_chan_name(name):
    """Normalize MATLAB channel names like '\\theta_2', 'T_{SAMPLE}', 'T_{set}'."""
    s = str(name)
    s = s.replace('\\', '').replace('_{', '_').replace('}', '').strip()
    # Also strip accidental list/array stringification leftovers
    s = s.strip("[]'\" ")
    return s


def _unwrap_matlab(obj):
    """Dig out the actual value from nested MATLAB cell/array structures."""
    while isinstance(obj, np.ndarray) and obj.size == 1:
        obj = obj.flat[0]
    return obj


def _unwrap_matlab_string(obj):
    """Dig out the actual string from nested MATLAB cell/array structures."""
    obj = _unwrap_matlab(obj)
    if isinstance(obj, np.ndarray) and obj.size > 0:
        # e.g. array(['T_{set}'])
        obj = obj.flat[0]
    return str(obj)


def extract_data_v1_python(mat_path):
    """Python version of extract_data_v1.m"""
    mat = scipy.io.loadmat(mat_path)
    scan = mat['scan'][0, 0]
    data = mat.get('data')

    D = {
        'X1': None, 'theta1': None, 'R1': None,
        'X2': None, 'theta2': None, 'R2': None,
        'X3': None, 'theta3': None, 'R3': None,
        'Vlia1': None, 'Vlia2': None, 'Vlia3': None,
        'Vs': None, 'Is': None, 'Vdmm': None, 'Vnvm': None,
        'Tsample': None, 'Tset': None,
        'Hsample': None, 'Hset': None,
        'freq': None, 'ampl': None, 'offs': None, 'phas': None,
        'symm': None, 'power': None,
        'comments': None
    }

    loops = scan['loops'][0]

    for loop in loops:
        # Measured channels
        if 'getchan' in loop.dtype.names:
            getchan = loop['getchan'][0]
            for ch_idx, ch_name_arr in enumerate(getchan):
                ch_name = _unwrap_matlab_string(ch_name_arr)
                ch_clean = _clean_chan_name(ch_name)

                if data is not None and ch_idx < data.shape[1]:
                    val = data[0, ch_idx]
                    if ch_clean == 'X_1' and D['X1'] is None:           D['X1'] = val
                    elif ch_clean == 'theta_1' and D['theta1'] is None: D['theta1'] = val
                    elif ch_clean == 'R_1' and D['R1'] is None:         D['R1'] = val
                    elif ch_clean == 'X_2' and D['X2'] is None:         D['X2'] = val
                    elif ch_clean == 'theta_2' and D['theta2'] is None: D['theta2'] = val
                    elif ch_clean == 'R_2' and D['R2'] is None:         D['R2'] = val
                    elif ch_clean == 'X_3' and D['X3'] is None:         D['X3'] = val
                    elif ch_clean == 'theta_3' and D['theta3'] is None: D['theta3'] = val
                    elif ch_clean == 'R_3' and D['R3'] is None:         D['R3'] = val
                    elif ch_clean == 'I_s' and D['Is'] is None:         D['Is'] = val
                    elif ch_clean == 'V_dmm' and D['Vdmm'] is None:     D['Vdmm'] = val
                    elif ch_clean == 'T_SAMPLE' and D['Tsample'] is None: D['Tsample'] = val
                    elif ch_clean == 'H_SAMPLE' and D['Hsample'] is None: D['Hsample'] = val

        # Setpoint channels
        if 'setchan' in loop.dtype.names and 'setchanranges' in loop.dtype.names and 'npoints' in loop.dtype.names:
            setchan = loop['setchan']
            if setchan.size > 0:
                set_name = _unwrap_matlab_string(setchan)
                set_clean = _clean_chan_name(set_name)

                rng = _unwrap_matlab(loop['setchanranges'])
                # After unwrapping we may still have a 1×2 or 2-element array
                if isinstance(rng, np.ndarray):
                    rng_vals = rng.flatten()
                else:
                    rng_vals = np.asarray(rng).flatten()

                npts = int(_unwrap_matlab(loop['npoints']))

                if len(rng_vals) >= 2:
                    start = float(rng_vals[0])
                    stop = float(rng_vals[1])
                    if set_clean == 'V_s' and D['Vs'] is None:
                        D['Vs'] = np.linspace(start, stop, npts)
                    elif set_clean == 'T_set' and D['Tset'] is None:
                        D['Tset'] = np.linspace(start, stop, npts)
                    elif set_clean == 'H_set' and D['Hset'] is None:
                        D['Hset'] = np.linspace(start, stop, npts)

    if 'comments' in scan.dtype.names:
        D['comments'] = scan['comments']

    return D


def save_to_csv(mat_path):
    """Extract data → save CSV (in current folder) + comments.txt. Removes all-NaN rows."""
    D = extract_data_v1_python(mat_path)
    if not D:
        print("Extraction failed")
        return False

    base_name = Path(mat_path).stem
    cwd = Path(os.getcwd())

    csv_path = cwd / f"{base_name}.csv"
    comments_path = cwd / f"{base_name}_comments.txt"

    # Save comments
    comments = D.pop('comments', None)
    if comments is not None:
        with open(comments_path, 'w', encoding='utf-8') as f:
            if isinstance(comments, np.ndarray):
                for line in comments.flatten():
                    f.write(str(line) + '\n')
            else:
                f.write(str(comments))
        print(f"✓ Comments saved: {comments_path}")

    # ---- Build DataFrame correctly ----
    # 1) First pass: collect real arrays and determine max length
    data_dict = {}
    max_len = 0
    for key, value in D.items():
        if isinstance(value, np.ndarray) and value.size > 0:
            flat = value.flatten()
            data_dict[key] = flat
            max_len = max(max_len, len(flat))
        else:
            data_dict[key] = None   # placeholder

    # 2) Second pass: pad missing columns to the same length
    if max_len == 0:
        max_len = 1
    for key in data_dict:
        if data_dict[key] is None:
            data_dict[key] = np.full(max_len, np.nan)
        elif len(data_dict[key]) != max_len:
            # Safety: pad or truncate if lengths ever differ
            arr = np.asarray(data_dict[key], dtype=float)
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
            else:
                arr = arr[:max_len]
            data_dict[key] = arr

    df = pd.DataFrame(data_dict)

    # Remove completely empty rows (all NaN)
    df = df.dropna(how='all').reset_index(drop=True)

    df.to_csv(csv_path, index=False)

    print(f"✓ CSV saved: {csv_path}")
    print(f"   Final shape: {df.shape} (all-NaN rows removed)")
    print(f"   Columns: {list(df.columns)}")
    return True


# ====================== USAGE ======================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Accept either a bare relative path or an absolute path
        arg = sys.argv[1]
        if os.path.isabs(arg) or os.path.exists(arg):
            file_path = arg
        else:
            file_path = f"/home/joaoalmendes/PhD/{arg}"
        if os.path.exists(file_path):
            print(f"Processing: {file_path}")
            save_to_csv(file_path)
        else:
            print(f"File not found: {file_path}")
    else:
        attachments_dir = "/home/joaoalmendes/PhD/"
        import glob
        mat_files = glob.glob(os.path.join(attachments_dir, "*.mat"))
        for f in mat_files:
            if "extract_data_v1" not in f.lower():
                print(f"\n--- Processing {os.path.basename(f)} ---")
                save_to_csv(f)