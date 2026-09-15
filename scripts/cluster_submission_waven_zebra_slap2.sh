#!/bin/bash

#SBATCH --job-name=waven-zebra-slap2

# GPU partition: WaveletGenerator.getWTfromNPY calls .cuda() unconditionally,
# so the wavelet decomposition cannot run on a CPU-only node at all.
# PearsonCorrelationPinkNoise also tiles its matmul onto the GPU when one is
# there (Analysis_Utils._pearson_cross_corr_chunked), so the sweep wants it too.
#SBATCH -p scc-gpu
#SBATCH -G A100

################################################################
# ONE ARRAY TASK PER COLOUR CHANNEL.
#
#   task 0 -> green (iGluSnFR4f)      task 1 -> red (RCaMP3)
#
# Each task runs both DMDs, sequentially, and each of those runs
# both wavelet phases (0 and 1) inside compute_waven_pipeline_slap2.py.
# So the full grid is
#
#   2 channels (parallel) x 2 DMDs (serial) x 2 phases (serial)
#     x the channel's delay x duration grid
#
# Channel is the right axis to parallelise on. The two channels are
# separate NWB datasets with their own timestamps, their own ROI
# responses and their own delay/duration grids (GRIDS in the driver),
# so the two tasks share nothing but read-only caches. The DMDs would
# be a valid axis too, but splitting on both would put four tasks on
# one decomposition for no gain - the run time is dominated by the
# correlation sweep, which is per (DMD, channel) either way.
#
# What is NOT safe to parallelise: the wavelet decomposition. It is
# keyed on the movie and the filter library only - NOT on the channel
# or the DMD - so all four (DMD, channel) runs want the same
# dwt_videodata_{0,1}.npy. get_path_wavelet_decomposition tests then
# writes, so two tasks arriving at a cold cache would both decide it
# is missing and both write the same file. Different nodes do not help
# here: the cache is on shared storage. The guard below refuses to
# start the red task against a cold cache and prints the fix.
################################################################
#SBATCH -a 0-1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

################################################################
# Memory. At the current settings (nx=67, ny=53, 10 thetas, 6 sigmas,
# 13 frequencies, 9000 frames) one phase of the decomposition is
#   9000 x 67 x 53 x 10 x 6 x 13 x 4 B = 99.7 GB
# and load_phase_dependent_wavelet_decompositions np.loads it whole.
# The correlation then holds a float32 (n_features, T) copy alongside
# it. The two array tasks land on separate nodes, so this is a
# per-node figure, not a sum over the grid.
################################################################
#SBATCH --mem 350G

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

# Two destinations. The per-(delay, duration) files carry the full correlation
# matrix and are by far the bigger of the two, so they go to the HDD workspace;
# only the reduced one-row-per-unit results land in the project results tree.
RESULTS=${RESULTS:-/mnt/ceph-hdd/workspaces/ws/cidbn_wibral_neuro_nonhuman/u19361-allen-hdd/results}
OPTIMIZED=${OPTIMIZED:-/mnt/ceph-hdd/projects/cidbn_wibral_neuro_nonhuman/SPP2205_mraabe/results/allen_open_scope/rf/waven/zebra/optimized}

NWB=${NWB:-${RAWDATA}/allen_open_scope/slap2/sub-829704/sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys.nwb}
DMDS=${DMDS:-"DMD1 DMD2"}
PHASES=${PHASES:-"0 1"}

CHANNELS=(green red)
CHANNEL=${CHANNELS[${SLURM_ARRAY_TASK_ID}]}

if [ -z "${CHANNEL}" ]; then
    echo "ERROR: array index ${SLURM_ARRAY_TASK_ID} has no channel."
    echo "       This job takes indices 0-$(( ${#CHANNELS[@]} - 1 )): ${CHANNELS[*]}"
    exit 1
fi

if [ ! -f "${NWB}" ]; then
    echo "ERROR: no NWB file at ${NWB}"
    exit 1
fi

################################################################
# Where the shared decomposition lives. The library id is the SHA-1 of
# the filter parameters (create_gabor_library), so it can be derived
# from waven_settings without building anything - which matters,
# because building is exactly what must not happen twice at once.
# Importing waven_settings and the waven package both print to stdout,
# so the value is picked out by its marker rather than by taking the
# last line.
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

# Only the first task may build the cache; a second task arriving at a cold
# cache would race it. Once the files are there both tasks are pure readers and
# run in parallel as intended.
if [ "${CACHE_WARM}" -eq 0 ] && [ "${SLURM_ARRAY_TASK_ID}" -ne 0 ]; then
    echo "ERROR: the wavelet decomposition is cold:"
    echo "         ${DECOMP_DIR}"
    echo "       Task ${SLURM_ARRAY_TASK_ID} (${CHANNEL}) would race task 0 building it."
    echo "       Submit the channels chained for the first run instead:"
    echo "         GREEN=\$(sbatch --parsable -a 0 -o ... $(basename "$0"))"
    echo "         sbatch --dependency=afterok:\${GREEN} -a 1 -o ... $(basename "$0")"
    echo "       Every later submission finds the cache warm and can run -a 0-1."
    exit 1
fi

echo "pwd              $(pwd)"
echo "SLURM_JOB_ID     $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK $SLURM_ARRAY_TASK_ID"
echo "SLURM_NODELIST   $SLURM_JOB_NODELIST"
echo "channel          ${CHANNEL}  (of ${CHANNELS[*]})"
echo "dmds             ${DMDS}"
echo "phases           ${PHASES}"
echo "nwb              ${NWB}"
echo "results          ${RESULTS}"
echo "optimized        ${OPTIMIZED}"
echo "decomposition    ${DECOMP_DIR}  (warm=${CACHE_WARM})"
nvidia-smi
free -h
df -h /mnt/ceph-hdd | tail -1

# One DMD failing must not hide the other: run both, then exit non-zero so the
# failure still shows up in the SLURM accounting.
STATUS=0
for DMD in ${DMDS}; do
    COMMAND="python compute_waven_pipeline_slap2.py \
        --nwb-path ${NWB} \
        --results-dir ${RESULTS} \
        --optimized-dir ${OPTIMIZED} \
        --dmd ${DMD} \
        --channel ${CHANNEL} \
        --phases ${PHASES}"
    echo "${COMMAND}"
    ${COMMAND}
    RC=$?
    # `if ! ${COMMAND}` would report $? as 0 here - the status of the negation,
    # not of the run - so capture it before branching.
    if [ ${RC} -ne 0 ]; then
        echo "ERROR: ${DMD} / ${CHANNEL} failed (exit ${RC})."
        STATUS=1
    fi
done
exit ${STATUS}

# Submit with a timestamped log file:
#   sbatch -o `date +%y%m%d-%H%M`_waven-zebra-slap2_%A_%3a.out cluster_submission_waven_zebra_slap2.sh
#
# First run on a cold decomposition - chain the channels so only one builds it:
#   GREEN=$(sbatch --parsable -a 0 -o `date +%y%m%d-%H%M`_waven-slap2-green_%A.out cluster_submission_waven_zebra_slap2.sh)
#   sbatch --dependency=afterok:${GREEN} -a 1 -o `date +%y%m%d-%H%M`_waven-slap2-red_%A.out cluster_submission_waven_zebra_slap2.sh
#
# One channel only:
#   sbatch -a 0 -o ... cluster_submission_waven_zebra_slap2.sh     # green
#   sbatch -a 1 -o ... cluster_submission_waven_zebra_slap2.sh     # red
#
# One DMD only, or another session:
#   DMDS=DMD1 sbatch -o ... cluster_submission_waven_zebra_slap2.sh
#   NWB=/path/to/other_session.nwb sbatch -o ... cluster_submission_waven_zebra_slap2.sh
