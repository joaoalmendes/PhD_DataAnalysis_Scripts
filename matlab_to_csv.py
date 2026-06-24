import scipy.io
import numpy as np
import pandas as pd
import os
from pathlib import Path

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
                ch_name = str(ch_name_arr[0]) if isinstance(ch_name_arr, np.ndarray) else str(ch_name_arr)
                ch_clean = ch_name.replace('\\', '').replace('_{', '_').replace('}', '').strip()

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
                set_name = str(setchan.flat[0])
                set_clean = set_name.replace('\\', '').replace('_{', '_').replace('}', '').strip()

                rng = loop['setchanranges'][0]
                npts = int(loop['npoints'].flat[0])

                if rng.size >= 2:
                    start = float(rng.flat[0])
                    stop = float(rng.flat[1])
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
    
    csv_path = cwd / f"{base_name}_processed.csv"
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

    # Build DataFrame (all columns, even empty)
    data_dict = {}
    max_len = 0

    for key, value in D.items():
        if isinstance(value, np.ndarray) and value.size > 0:
            flat = value.flatten()
            data_dict[key] = flat
            max_len = max(max_len, len(flat))
        else:
            data_dict[key] = [np.nan] * max(1, max_len) if max_len > 0 else [np.nan]

    df = pd.DataFrame(data_dict)

    # === Remove completely empty rows (all NaN) ===
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
        file_path = f"/home/joaoalmendes/PhD/260519B/{sys.argv[1]}"
        if os.path.exists(file_path):
            print(f"Processing: {file_path}")
            save_to_csv(file_path)
        else:
            print(f"File not found: {file_path}")
    else:
        attachments_dir = "/home/joaoalmendes/PhD/260519B/"
        import glob
        mat_files = glob.glob(os.path.join(attachments_dir, "*.mat"))
        for f in mat_files:
            if "extract_data_v1" not in f.lower():
                print(f"\n--- Processing {os.path.basename(f)} ---")
                save_to_csv(f)