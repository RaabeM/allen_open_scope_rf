#!/bin/bash

#SBATCH --job-name=waven-zebra-meso

# GPU partition: WaveletGenerator.getWTfromNPY calls .cuda() unconditionally,
# so the wavelet decomposition cannot run on a CPU-only node at all.
# PearsonCorrelationPinkNoise also tiles its matmul onto the GPU when one is
# there, so the sweep wants it too.
#SBATCH -p scc-gpu
#SBATCH -G A100

################################################################
# ONE ARRAY TASK PER IMAGING PLANE.
#
# The array index is passed straight through as --plane-idx, an index
# into stream.imaging_planes():
#
#   0-3 -> VISl_4 VISl_5 VISl_6 VISl_7
#   4-7 -> VISp_0 VISp_1 VISp_2 VISp_3
#
# Each task runs every Zebra trial and both wavelet phases (0 and 1)
# serially inside compute_waven_pipeline_mesoscope.py, then reduces
# each (trial, phase) grid to one optimized file.
#
# #SBATCH directives are read before the script runs, so the array
# SIZE cannot be computed from the data. It is fixed at the 8 planes
# the 2026-01 mesoscope sessions carry; an index past the end of a
# session's plane list is caught below. Check a new session first with:
#
#   python -c "import sys; sys.path.append('..'); import utils; \
#              print(utils.open_local('<nwb>').imaging_planes())"
#
# What is NOT safe to parallelise: the wavelet decomposition. It is
# keyed on the movie and the filter library only - NOT on the plane,
# the signal or even the modality - so all eight planes (and the SLAP2
# job, which uses the same movie) want the same dwt_videodata_{0,1}.npy.
# get_path_wavelet_decomposition tests then writes, so tasks arriving
# at a cold cache would all decide it is missing and all write the same
# file on shared storage. The guard below lets only task 0 build it.
################################################################
#SBATCH -a 0-7

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

################################################################
# Memory. At the current settings (nx=67, ny=53, 10 thetas, 6 sigmas,
# 13 frequencies, 9000 frames) one phase of the decomposition is
#   9000 x 67 x 53 x 10 x 6 x 13 x 4 B = 99.7 GB
# and it is np.loaded whole. The mesoscope branch then fancy-indexes
# the frames matching the ~9.5 Hz imaging samples (~1/3 of the movie,
# ~32 GB), and holds the previous grid cell's copy while assigning the
# next one, so two of those coexist. The correlation adds a float32
# (n_features, T) copy of about the same size. That is ~200 GB per
# node; the array tasks land on separate nodes, so it does not sum.
################################################################
#SBATCH --mem 300G

# The correlation sweep is numpy-threaded, not process-parallel; leave BLAS
# free to use the allocated cores rather than pinning to 1.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load miniforge3
source activate /user/raabe14/u19361/.conda/envs/waven_dandi

echo "CONDA env: ${CONDA_DEFAULT_ENV:-$(basename "${CONDA_PREFIX:-none}")}"

CODE_DIR=${CODE_DIR:-/mnt/ceph-hdd/projects/cidbn_wibral_neuro_nonhuman/SPP2205_mraabe/code/allen_open_scope/scripts}
cd "${CODE_DIR}" || { echo "ERROR: no ${CODE_DIR}"; exit 1; }

RAWDATA=${RAWDATA:-/mnt/ceph-hdd/projects/cidbn_wibral_neuro_nonhuman/SPP2205_mraabe/rawdata}

# The per-(delay, duration) files carry the full correlation matrix, so they
# go to the HDD workspace. The optimized files go to the project results tree,
# but that path is hardcoded in the driver (.../zebra/optimized/<signal>/<session>).
RESULTS=${RESULTS:-/mnt/ceph-hdd/workspaces/ws/cidbn_wibral_neuro_nonhuman/u19361-allen-hdd/results}

NWB=${NWB:-${RAWDATA}/allen_open_scope/meso/sub-832700/sub-832700_ses-multiplane-ophys-832700-2026-01-24-12-06-12_ophys.nwb}
# 'dff' is the DF/F trace; 'events' is the OASIS-deconvolved series stored in
# the NWB. Separate runs, separate result trees (<results-dir>/<signal>/).
SIGNAL=${SIGNAL:-events}

if [ ! -f "${NWB}" ]; then
    echo "ERROR: no NWB file at ${NWB}"
    exit 1
fi

# The driver writes <results-dir>/<signal>/<plane>/... with no session in the
# path, and skips files that already exist - so without the session here, a
# second session would silently reuse the first one's results.
SESSION=$(basename "${NWB}" .nwb)
RESULTS_DIR=${RESULTS}/zebra/meso/${SESSION}

# Resolve this task's plane from the session itself, so the log shows what
# the file actually holds. Importing utils prints to stdout, so pick the value
# out by its marker rather than taking the last line.
PLANE=$(python - "${NWB}" "${SLURM_ARRAY_TASK_ID}" <<'PY' | sed -n 's/^PLANE\t//p'
import sys
sys.path.append('..')
import utils

nwb, idx = sys.argv[1], int(sys.argv[2])
planes = utils.open_local(nwb).imaging_planes()
if idx < len(planes):
    print(f"PLANE\t{planes[idx]}")
PY
)

if [ -z "${PLANE}" ]; then
    echo "ERROR: array index ${SLURM_ARRAY_TASK_ID} has no plane in"
    echo "         ${NWB}"
    echo "       Resize the array to match that session's plane count."
    exit 1
fi

################################################################
# Where the shared decomposition lives. The library id is the SHA-1 of
# the filter parameters (create_gabor_library), so it can be derived
# from waven_settings without building anything - which matters,
# because building is exactly what must not happen twice at once.
################################################################
DECOMP_DIR=$(python - <<'PY' | sed -n 's/^DECOMP_DIR\t//p'
import sys, json, hashlib
from pathlib import Path
sys.path.append('..')
from waven_settings import *

props = {
    "xs": list(map(int, xs)),
    "ys": list(map(int, ys)),
    "thetas": list(map(float, thetas)),
    "sigmas": list(map(float, sigmas)),
    "offsets": list(map(float, offsets)),
    "frequencies": list(map(float, frequencies)),
}
serialized = json.dumps(props, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
lib_id = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]

movie = Path(movpath)
print(f"DECOMP_DIR\t{movie.parent / 'wavelet_decompositions' / movie.stem / f'lib-{lib_id}'}")
PY
)

if [ -z "${DECOMP_DIR}" ]; then
    echo "ERROR: could not resolve the decomposition directory from waven_settings."
    exit 1
fi

CACHE_WARM=1
for PHASE in 0 1; do
    [ -f "${DECOMP_DIR}/dwt_videodata_${PHASE}.npy" ] || CACHE_WARM=0
done

if [ "${CACHE_WARM}" -eq 0 ] && [ "${SLURM_ARRAY_TASK_ID}" -ne 0 ]; then
    echo "ERROR: the wavelet decomposition is cold:"
    echo "         ${DECOMP_DIR}"
    echo "       Task ${SLURM_ARRAY_TASK_ID} (${PLANE}) would race task 0 building it."
    echo "       Submit plane 0 first and chain the rest on it for the first run:"
    echo "         FIRST=\$(sbatch --parsable -a 0 -o ... $(basename "$0"))"
    echo "         sbatch --dependency=afterok:\${FIRST} -a 1-7 -o ... $(basename "$0")"
    echo "       Every later submission finds the cache warm and can run -a 0-7."
    exit 1
fi

echo "pwd              $(pwd)"
echo "SLURM_JOB_ID     $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK $SLURM_ARRAY_TASK_ID"
echo "SLURM_NODELIST   $SLURM_JOB_NODELIST"
echo "plane            ${PLANE}"
echo "signal           ${SIGNAL}"
echo "nwb              ${NWB}"
echo "results          ${RESULTS_DIR}/${SIGNAL}"
echo "decomposition    ${DECOMP_DIR}  (warm=${CACHE_WARM})"
nvidia-smi
free -h
df -h /mnt/ceph-hdd | tail -1

python compute_waven_pipeline_mesoscope.py \
    --nwb-path "${NWB}" \
    --results-dir "${RESULTS_DIR}" \
    --plane-idx "${SLURM_ARRAY_TASK_ID}" \
    --signal "${SIGNAL}"
RC=$?
[ ${RC} -ne 0 ] && echo "ERROR: ${PLANE} / ${SIGNAL} failed (exit ${RC})."
exit ${RC}

# Submit with a timestamped log file:
#   sbatch -o `date +%y%m%d-%H%M`_waven-zebra-meso_%A_%3a.out cluster_submission_waven_zebra_mesoscope.sh
#
# First run on a cold decomposition - plane 0 builds it, the rest wait:
#   FIRST=$(sbatch --parsable -a 0 -o `date +%y%m%d-%H%M`_waven-meso_%A_%3a.out cluster_submission_waven_zebra_mesoscope.sh)
#   sbatch --dependency=afterok:${FIRST} -a 1-7 -o `date +%y%m%d-%H%M`_waven-meso_%A_%3a.out cluster_submission_waven_zebra_mesoscope.sh
#
# The deconvolved events are a separate run into <results-dir>/events/:
#   SIGNAL=events sbatch -o ... cluster_submission_waven_zebra_mesoscope.sh
#
# One plane only, or another session:
#   sbatch -a 4 -o ... cluster_submission_waven_zebra_mesoscope.sh     # VISp_0
#   NWB=/path/to/other_session.nwb sbatch -o ... cluster_submission_waven_zebra_mesoscope.sh
