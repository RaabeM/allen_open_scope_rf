#!/bin/bash

#SBATCH --job-name=siegle-meso
#SBATCH -p cidbn

################################################################
# ONE ARRAY TASK PER IMAGING PLANE.
#
# The array index is an index into stream.imaging_planes(), which is
# sorted, so the mapping is stable across tasks and resubmissions:
#
#   0-3 -> VISl_4 VISl_5 VISl_6 VISl_7
#   4-7 -> VISp_0 VISp_1 VISp_2 VISp_3
#
# Each task walks its own delay x duration grid and then reduces it to
# one optimized file per plane.
#
# All planes may run at once with nothing to coordinate: the Siegle
# test correlates the traces against the RF mapping presentation table
# directly, so there is no Gabor library and no wavelet decomposition
# for two tasks to race on building - the difference from the waven
# job. CPU only, for the same reason: nothing here touches torch.
#
# #SBATCH directives are read before the script runs, so the array
# SIZE cannot be computed from the data. It is fixed at the 8 planes
# the 2026-01 mesoscope sessions carry; an index past the end of a
# session's plane list is caught below rather than silently running
# the wrong plane. Check a new session first with:
#
#   python -c "import sys; sys.path.append('..'); import utils; \
#              print(utils.open_local('<nwb>').imaging_planes())"
################################################################
#SBATCH -a 0-7

#SBATCH --nodes=1
#SBATCH --ntasks=1

################################################################
# compute_siegle_ophys fans out over ROIs with multiprocessing.Pool,
# so give it processes rather than BLAS threads.
#
# CAVEAT: it is called with n_procs=None, and Pool then sizes itself
# from os.cpu_count() - the whole node, NOT this allocation, which
# os.cpu_count() does not see. On a shared node that oversubscribes the
# cgroup. Until the driver takes an --n-procs, either ask for the whole
# node (--exclusive) or keep the allocation large enough that the
# oversubscription is mild. A mesoscope plane is a few hundred ROIs
# sampled at ~9.5 Hz, so the shuffles dominate and the job is short.
################################################################
#SBATCH --cpus-per-task=32
#SBATCH --mem 32G

# Process-parallel, not thread-parallel: pin BLAS to 1 so the Pool workers
# do not each spawn their own thread pool on top.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

module load miniforge3
source activate /user/raabe14/u19361/.conda/envs/waven_dandi

echo "CONDA env: ${CONDA_DEFAULT_ENV:-$(basename "${CONDA_PREFIX:-none}")}"

CODE_DIR=${CODE_DIR:-/mnt/ceph-hdd/projects/cidbn_wibral_neuro_nonhuman/SPP2205_mraabe/code/allen_open_scope/scripts}
cd "${CODE_DIR}" || { echo "ERROR: no ${CODE_DIR}"; exit 1; }

RAWDATA=${RAWDATA:-/mnt/ceph-hdd/projects/cidbn_wibral_neuro_nonhuman/SPP2205_mraabe/rawdata}
# The Siegle outputs are small - z_score_response is (n_rois, n_orientations,
# nx, ny) - so both the grid files and the reduction go to the project results
# tree; there is no reason to send these to the HDD workspace.
RESULTS=${RESULTS:-/mnt/ceph-hdd/projects/cidbn_wibral_neuro_nonhuman/SPP2205_mraabe/results/allen_open_scope/rf/gabors}

NWB=${NWB:-${RAWDATA}/allen_open_scope/meso/sub-832700/sub-832700_ses-multiplane-ophys-832700-2026-01-24-12-06-12_ophys.nwb}
# 'dff' is the DF/F trace; 'events' is the OASIS-deconvolved series, which is
# non-negative and so the closer analogue of a spike rate. Separate runs into
# separate trees: ${RESULTS}/<signal>/<session>/.
SIGNAL=${SIGNAL:-events}
N_SHUFFLE=${N_SHUFFLE:-1000}
BASELINE=${BASELINE:-0.0}

if [ ! -f "${NWB}" ]; then
    echo "ERROR: no NWB file at ${NWB}"
    exit 1
fi

# Resolve this task's plane from the session itself, so the mapping never
# drifts from what the file actually holds. Importing utils prints to stdout,
# so pick the value out by its marker rather than taking the last line.
PLANE=$(python - "${NWB}" "${SLURM_ARRAY_TASK_ID}" <<'PY' | sed -n 's/^PLANE\t//p'
import sys
sys.path.append('..')
import utils

nwb, idx = sys.argv[1], int(sys.argv[2])
planes = utils.open_local(nwb).imaging_planes()
if idx < len(planes):
    print(f"PLANE\t{planes[idx]}")
else:
    print(f"NPLANES\t{len(planes)}", file=sys.stderr)
PY
)

if [ -z "${PLANE}" ]; then
    echo "ERROR: array index ${SLURM_ARRAY_TASK_ID} has no plane in"
    echo "         ${NWB}"
    echo "       Resize the array to match that session's plane count."
    exit 1
fi

echo "pwd              $(pwd)"
echo "SLURM_JOB_ID     $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK $SLURM_ARRAY_TASK_ID"
echo "SLURM_NODELIST   $SLURM_JOB_NODELIST"
echo "plane            ${PLANE}"
echo "signal           ${SIGNAL}"
echo "nwb              ${NWB}"
echo "results          ${RESULTS}/${SIGNAL}/$(basename "${NWB}" .nwb)"
echo "n_shuffle        ${N_SHUFFLE}"
free -h
df -h /mnt/ceph-hdd | tail -1

python compute_rf_siegle_gabors_mesoscope.py \
    --nwb-path "${NWB}" \
    --results-path "${RESULTS}" \
    --plane-idx "${SLURM_ARRAY_TASK_ID}" \
    --signal "${SIGNAL}" \
    --baseline "${BASELINE}" \
    --n-shuffle "${N_SHUFFLE}"
RC=$?
[ ${RC} -ne 0 ] && echo "ERROR: ${PLANE} / ${SIGNAL} failed (exit ${RC})."
exit ${RC}

# Submit with a timestamped log file:
#   sbatch -o `date +%y%m%d-%H%M`_siegle-meso_%A_%3a.out cluster_submission_siegle_mesoscope.sh
#
# One plane only:
#   sbatch -a 4 -o ... cluster_submission_siegle_mesoscope.sh     # VISp_0
#
# The deconvolved events are a separate run into ${RESULTS}/events/<session>/:
#   SIGNAL=events sbatch -o ... cluster_submission_siegle_mesoscope.sh
#
# A quick shape check before committing the full shuffle count:
#   N_SHUFFLE=10 sbatch -o ... cluster_submission_siegle_mesoscope.sh
