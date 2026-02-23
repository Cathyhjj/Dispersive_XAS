import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from DXAS import xas
from DXAS import io
import glob
import scipy.ndimage as nd
import skimage as sk
import os
import h5py

import os
import hdf5plugin   # auto-registers bitshuffle, lz4, blosc, etc.
import h5py

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

# Use a safe default font and silence warnings
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

date = xas.date_today()


def load_nexus_entry(filepath):
    """
    Load a NeXus-style HDF5 (e.g., from areaDetector) with layout:

      /entry (NXentry)
        /data (NXdata)
          - data: image stack, shape (N, H, W)
        /instrument
          /NDAttributes
            - NDArrayEpicsTSSec, NDArrayEpicsTSnSec, NDArrayTimeStamp, NDArrayUniqueId
          /performance
            - timestamp (optional, may be empty)

    Returns:
      {
        'meta': {
            'nx_entry_class', 'nx_data_class',
            'shape': (N,H,W), 'dtype': str,
            'n_frames', 'height', 'width'
        },
        'data': np.ndarray,                  # image stack
        'nd_attributes': {name: np.ndarray}, # NDAttributes datasets (1D)
        'performance': {'timestamp': np.ndarray or None},
        'timestamps': {
            'epics_sec': np.ndarray or None,
            'epics_nsec': np.ndarray or None,
            'epics_ts': np.ndarray or None,   # sec + nsec*1e-9 if both exist
            'nd_time': np.ndarray or None     # NDArrayTimeStamp as-is (float)
        },
        'frame_ids': np.ndarray or None       # NDArrayUniqueId, if present
      }
    """
    out = {
        'meta': {},
        'data': None,
        'nd_attributes': {},
        'performance': {},
        'timestamps': {'epics_sec': None, 'epics_nsec': None, 'epics_ts': None, 'nd_time': None},
        'frame_ids': None,
    }

    with h5py.File(filepath, 'r') as f:
        entry = f['/entry']
        data_grp = entry.get('data')
        if data_grp is None or 'data' not in data_grp:
            raise KeyError("Expected dataset '/entry/data/data' not found.")

        ds = data_grp['data']
        arr = ds[()]  # (N,H,W)
        out['data'] = arr

        # Meta
        nx_entry_class = entry.attrs.get('NX_class', b'')
        nx_data_class = data_grp.attrs.get('NX_class', b'')
        if isinstance(nx_entry_class, bytes): nx_entry_class = nx_entry_class.decode()
        if isinstance(nx_data_class, bytes): nx_data_class = nx_data_class.decode()

        n, h, w = arr.shape
        out['meta'].update({
            'nx_entry_class': nx_entry_class,
            'nx_data_class': nx_data_class,
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'n_frames': int(n),
            'height': int(h),
            'width': int(w),
        })

        # NDAttributes (if present)
        ndattr_grp = entry.get('instrument/NDAttributes')
        if ndattr_grp is not None:
            for name, obj in ndattr_grp.items():
                if isinstance(obj, h5py.Dataset):
                    out['nd_attributes'][name] = obj[()]

            # Timestamps & Unique IDs
            epics_sec = out['nd_attributes'].get('NDArrayEpicsTSSec')
            epics_nsec = out['nd_attributes'].get('NDArrayEpicsTSnSec')
            nd_time = out['nd_attributes'].get('NDArrayTimeStamp')
            frame_ids = out['nd_attributes'].get('NDArrayUniqueId')

            out['timestamps']['epics_sec'] = epics_sec
            out['timestamps']['epics_nsec'] = epics_nsec
            out['timestamps']['nd_time'] = nd_time
            out['frame_ids'] = frame_ids

            if epics_sec is not None and epics_nsec is not None:
                # Combine seconds + nanoseconds into a float (same length as attributes)
                out['timestamps']['epics_ts'] = epics_sec.astype(np.float64) + epics_nsec.astype(np.float64)*1e-9

        # Optional performance group
        perf_grp = entry.get('instrument/performance')
        if perf_grp is not None and 'timestamp' in perf_grp:
            ts = perf_grp['timestamp'][()]
            out['performance']['timestamp'] = ts if ts.size > 0 else None
        else:
            out['performance']['timestamp'] = None

    return out

# ---------- normalization ----------
factor = 200

def norm_spec(spec, x1, x2):
    spec = np.nan_to_num(
        spec,
        nan=np.nanmean(spec),
        posinf=np.nanmax(spec),
        neginf=np.nanmin(spec)
    )
    ref = spec[x1:x2]
    smin, smax = np.min(ref), np.max(ref)
    return ((spec - smin) / (smax - smin + 1e-12)) * factor


# def norm_spec(spec, x1, x2):
#     return spec


# ---------- main plotting ----------
def plot_spectra_in_chunks(data_path, flat_path,
                           start_frame=0, end_frame=None,
                           aver_n=1, flat_range=(180, 230),
                           norm_x1 = 180, norm_x2 = 200,
                           x1=100, x2=160, chunk_size=1000,
                           cmap_name='magma'):
    """
    Saves per chunk:
      1) Lines (no averaging across frames) default colors
      2) Lines averaged in groups of aver_n gradient colors
      3) Imshow of per-frame spectra (no averaging)
    Output folder: preliminary_results_<data_path without extension>/
    Filenames include frame range and number of frames plotted/saved.
    """
    # You already defined this in your session:
    # from your previous step: load_nexus_entry(data_path) -> dict with ['data']
    data_all = load_nexus_entry(data_path)
    flat_all = load_nexus_entry(flat_path)
    data = data_all['data']   # (N,H,W)
    flat = flat_all['data']   # (N,H,W)

    if end_frame is None:
        end_frame = data.shape[0]

    # --- Output directory ---
    base_name = os.path.splitext(data_path)[0]
    save_dir = os.path.join(
        os.path.dirname(base_name),
        "preliminary_results_" + os.path.basename(base_name)
    )
    os.makedirs(save_dir, exist_ok=True)

    cmap = cm.get_cmap(cmap_name)
    fr0, fr1 = flat_range
    flat_avg = np.average(flat[:, fr0:fr1, :], axis=0)  # (H_sel, W)

    for chunk_start in range(start_frame, end_frame, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_frame)
        n_frames = chunk_end - chunk_start
        if n_frames <= 0:
            continue

        # -------- Per-frame spectra (NO averaging) --------
        per_frame_specs = []
        for f in range(chunk_start, chunk_end):
            data_avg = np.average(data[f:f+1, fr0:fr1, :], axis=0)
            mux = np.log(flat_avg / data_avg)
            mux[~np.isfinite(mux)] = 0.0
            spec = np.average(mux, axis=0)
            normed = norm_spec(spec, norm_x1, norm_x2)
            per_frame_specs.append(normed)
        per_frame_specs = np.asarray(per_frame_specs)  # (n_frames, W)

        # -------- 1) Line plot without averaging --------
        plt.figure(figsize=(5, 8))
        for i, spec in enumerate(per_frame_specs):
            plt.plot(spec + i)  # offset by 1 per frame
        plt.xlim(x1, x2 + 50)
        plt.xlabel(data_path)
        plt.ylabel('Frame index')
        plt.title(f"Lines (no averaging) frames {chunk_start}-{chunk_end} [N={n_frames}]")

        # y-ticks show actual frame numbers
        y_positions = np.arange(0, n_frames, max(1, n_frames // 10))
        y_labels = [str(chunk_start + y) for y in y_positions]
        plt.yticks(y_positions, y_labels)

        # Dynamic y-range: first spec min + offset(0), last spec max + offset(n_frames-1)
        y_min = float(np.min(per_frame_specs[0][x1:x2])) + 0
        y_max = float(np.max(per_frame_specs[-1][x1:x2])) + (n_frames - 1)
        plt.ylim(y_min, y_max)

        path_noavg = os.path.join(
            save_dir, f"lines_noavg_{chunk_start:05d}-{chunk_end:05d}_N{n_frames}.png"
        )
        plt.savefig(path_noavg, dpi=200, bbox_inches='tight')
        plt.close()

        # -------- 2) Line plot with averaging --------
        num_groups = n_frames // aver_n if aver_n > 0 else 0
        plt.figure(figsize=(5, 8))
        specs_avg = []
        if num_groups > 0:
            colors = cmap(np.linspace(0, 1, num_groups))
            for gi, color in zip(range(num_groups), colors):
                s = chunk_start + gi * aver_n
                e = s + aver_n
                data_avg = np.average(data[s:e, fr0:fr1, :], axis=0)
                mux = np.log(flat_avg / data_avg)
                mux[~np.isfinite(mux)] = 0.0
                spec = np.average(mux, axis=0)
                normed = norm_spec(spec, norm_x1, norm_x2)
                specs_avg.append(normed)
                plt.plot(normed + gi, color=color, linewidth=1)

        plt.xlim(x1, x2 + 50)
        plt.xlabel(data_path)
        plt.ylabel('Averaged frame index')
        plt.title(f"Lines (averaged over {aver_n}) frames {chunk_start}-{chunk_end} [N={num_groups}]")

        # y-ticks for averaged stacks (label the starting frame of each group)
        if num_groups > 0:
            y_positions = np.arange(0, num_groups, max(1, num_groups // 10))
            y_labels = [str(chunk_start + y * aver_n) for y in y_positions]
            plt.yticks(y_positions, y_labels)

            # Dynamic y-range: first avg spec min + offset(0), last avg spec max + offset(num_groups-1)
            y_min_avg = float(np.min(specs_avg[0][x1:x2])) + 0
            y_max_avg = float(np.max(specs_avg[-1][x1:x2])) + (num_groups - 1)
            plt.ylim(y_min_avg, y_max_avg)

        path_avg = os.path.join(
            save_dir, f"lines_avg{aver_n}_{chunk_start:05d}-{chunk_end:05d}_N{num_groups}.png"
        )
        plt.savefig(path_avg, dpi=200, bbox_inches='tight')
        plt.close()

        # -------- 3) Imshow (no averaging) --------
        plt.figure(figsize=(8, 10))
        plt.imshow(
            per_frame_specs,
            aspect='auto',
            cmap=cmap_name,
            vmin=0,
            vmax=1.5 * factor,
            extent=[0, per_frame_specs.shape[1], chunk_start, chunk_end],
            origin='lower'  # ensure frame index increases upward
        )
        plt.xlim(x1, x2 + 50)
        plt.xlabel(data_path)
        plt.ylabel('Frame index')
        plt.title(f"Imshow (no averaging) frames {chunk_start}-{chunk_end} [N={n_frames}]")

        # Major (every 200) + minor (every 100) y-ticks at real frame indices
        major_step = 100
        minor_step = 10
        yticks_major = np.arange(chunk_start, chunk_end + 1, major_step)
        yticks_minor = np.arange(chunk_start, chunk_end + 1, minor_step)
        ax = plt.gca()
        ax.set_yticks(yticks_major)
        ax.set_yticks(yticks_minor, minor=True)

        path_im = os.path.join(
            save_dir, f"imshow_{chunk_start:05d}-{chunk_end:05d}_N{n_frames}.png"
        )
        plt.colorbar(label='Normalized intensity')
        plt.savefig(path_im, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Saved:\n  {path_noavg}\n  {path_avg}\n  {path_im}")