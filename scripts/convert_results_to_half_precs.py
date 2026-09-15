import os
import numpy as np
import h5py
import shutil
from pathlib import Path
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

MIN_SIZE_BYTES = 4 * 1024 ** 3  # 3 GB


def _worker_init():
    # Prevent NumPy/OpenBLAS/MKL from spawning extra threads per worker.
    # This workload is I/O + memory bound; oversubscribing threads hurts.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def convert_file(path):
    # if path.stat().st_size < MIN_SIZE_BYTES:
    #     # print(f"skipped (< 3 GB): {path}")
    #     return

    tmp = path.with_suffix('.tmp.h5')
    try:
        
        with h5py.File(path, 'r') as src, h5py.File(tmp, 'w') as dst:
            for key, val in src.attrs.items():
                dst.attrs[key] = val
            for name in src:
                if name == 'correlation_matrix':
                    data = src['correlation_matrix'][:]
                    if data.dtype == np.float16:
                        return
                    data = data.astype(np.float16)
                    dst.create_dataset('correlation_matrix', data=data, dtype=np.float16,
                                       compression='gzip', compression_opts=4)
                else:
                    src.copy(name, dst)
        shutil.move(tmp, path)
        # print(f"converted: {path}")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        # print(f"FAILED: {path}: {e}")


def main(path, n_workers=None):
    files = sorted(Path(path).rglob("*.h5"))
    print(f"Found {len(files)} .h5 files under {path}")
    with Pool(n_workers, initializer=_worker_init) as pool:
        with tqdm(total=len(files), unit="file") as pbar:
            for _ in pool.imap_unordered(convert_file, files):
                pbar.update()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the correlation matricies from all results files in path to half precision floating point recursiveley."
    )
    parser.add_argument(
        "path",
        help="Root directory containing waven .h5 result files (searched recursively).",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel worker processes (default: all CPU cores).",
    )
    args = parser.parse_args()
    main(args.path, n_workers=args.n_workers)