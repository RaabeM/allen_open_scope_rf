#!/bin/bash

#SBATCH --job-name=siegle-slap2
#SBATCH -p cidbn


################################################################
# ONE ARRAY TASK PER (DMD, CHANNEL).
#
#   0 -> DMD1/green   1 -> DMD1/red   2 -> DMD2/green   3 -> DMD2/red
#
# Each task walks its own delay x duration grid (DELAYS x DURATIONS in
# compute_rf_siegle_gabors_slap2.py) and then reduces it to one
# optimized file.
#
# Unlike the waven job, all four may run at once with nothing to
# coordinate. The Siegle test correlates the traces against the
# rf_mapping presentation table directly - there is no Gabor library
# and no wavelet decomposition, so there is no shared cache for two
# tasks to race on building. Each task reads the NWB and writes files
# whose names carry its own DMD and channel.
#
# CPU only, for the same reason: nothing here touches torch.
################################################################
#SBATCH -a 0-3

#SBATCH --nodes=1
#SBATCH --ntasks=1

################################################################
# compute_siegle_ophys fans out over ROIs with multiprocessing.Pool,
# so give it processes rather than BLAS threads.
#
# CAVEAT: it is called with n_procs=None, and Pool then sizes itself
# from os.cpu_count() - the whole node, NOT this allocation, which
# os.cpu_count() does not see. On a shared node that oversubscribes the
# cgroup. Until the driver takes an --n-procs, the honest options are to
# ask for the whole node (--exclusive) or to keep the allocation large
# enough that oversubscription is mild. The work itself is small: the
# traces are ~50-75 ROIs x ~60k samples and the shuffles are the only
# real cost, so this job is short either way.
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

NWB=${NWB:-${RAWDATA}/allen_open_scope/slap2/sub-829704/sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys.nwb}
N_SHUFFLE=${N_SHUFFLE:-1000}
BASELINE=${BASELINE:-0.0}

# index -> (dmd, channel); keep the order stable so a resubmitted index
# always addresses the same combination.
COMBOS=("DMD1 green" "DMD1 red" "DMD2 green" "DMD2 red")
COMBO=${COMBOS[${SLURM_ARRAY_TASK_ID}]}

if [ -z "${COMBO}" ]; then
    echo "ERROR: array index ${SLURM_ARRAY_TASK_ID} has no (DMD, channel)."
    echo "       This job takes indices 0-$(( ${#COMBOS[@]} - 1 )):"
    for i in "${!COMBOS[@]}"; do echo "         ${i} -> ${COMBOS[$i]}"; done
    exit 1
fi
read -r DMD CHANNEL <<< "${COMBO}"

if [ ! -f "${NWB}" ]; then
    echo "ERROR: no NWB file at ${NWB}"
    exit 1
fi

echo "pwd              $(pwd)"
echo "SLURM_JOB_ID     $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK $SLURM_ARRAY_TASK_ID"
echo "SLURM_NODELIST   $SLURM_JOB_NODELIST"
echo "dmd / channel    ${DMD} / ${CHANNEL}"
echo "nwb              ${NWB}"
echo "results          ${RESULTS}"
echo "n_shuffle        ${N_SHUFFLE}"
free -h
df -h /mnt/ceph-hdd | tail -1

python compute_rf_siegle_gabors_slap2.py \
    --nwb-path "${NWB}" \
    --results-path "${RESULTS}" \
    --dmd "${DMD}" \
    --channel "${CHANNEL}" \
    --baseline "${BASELINE}" \
    --n-shuffle "${N_SHUFFLE}"
RC=$?
[ ${RC} -ne 0 ] && echo "ERROR: ${DMD} / ${CHANNEL} failed (exit ${RC})."
exit ${RC}

# Submit with a timestamped log file:
#   sbatch -o `date +%y%m%d-%H%M`_siegle-slap2_%A_%3a.out cluster_submission_siegle_slap2.sh
#
# One combination only (see the index table above):
#   sbatch -a 0 -o ... cluster_submission_siegle_slap2.sh      # DMD1/green
#
# A quick shape check before committing the full shuffle count:
#   N_SHUFFLE=10 sbatch -o ... cluster_submission_siegle_slap2.sh
#
# Another session:
#   NWB=/path/to/other_session.nwb sbatch -o ... cluster_submission_siegle_slap2.sh
