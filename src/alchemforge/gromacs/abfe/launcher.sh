#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:?Usage: $0 /path/to/abfe_config.sh}"
source "${CONFIG_FILE}"
PROJECT_DIR="${WORK_DIR}/${JOB_NAME}"
WORKER="${PROJECT_DIR}/ABFE_GROMACS.sh"
SETUP="${PROJECT_DIR}/setup_abfe.sh"

if [[ ! -x "${WORKER}" || ! -x "${SETUP}" ]]; then
    echo "ERROR: ABFE setup or worker script is missing in ${PROJECT_DIR}" >&2
    exit 1
fi

if [[ "${EXECUTION_MODE}" == "local" ]]; then
    bash "${SETUP}" "${CONFIG_FILE}"
    running=0
    for ((task_id = 0; task_id < N_ABFE_TASKS; task_id++)); do
        bash "${WORKER}" "${CONFIG_FILE}" "${task_id}" &
        running=$((running + 1))
        if (( running >= MAX_ABFE_JOBS )); then
            wait
            running=0
        fi
    done
    wait
    exit 0
fi

SETUP_JOB=$(sbatch --parsable \
    --job-name="${JOB_NAME}_setup" \
    --partition="${ABFE_PARTITION}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${ABFE_CPUS_PER_TASK}" \
    --mem="${ABFE_MEMORY}" --time="${ABFE_WALLTIME}" \
    --export="ALL,CONFIG_FILE=${CONFIG_FILE}" \
    "${SETUP}" "${CONFIG_FILE}")
SETUP_JOB="${SETUP_JOB%%;*}"

SBATCH_ARGS=(
    --job-name="${JOB_NAME}_abfe"
    --partition="${ABFE_PARTITION}"
    --nodes="${ABFE_NODES}"
    --ntasks="${ABFE_NTASKS}"
    --cpus-per-task="${ABFE_CPUS_PER_TASK}"
    --mem="${ABFE_MEMORY}"
    --time="${ABFE_WALLTIME}"
    --array="0-$((N_ABFE_TASKS - 1))%${MAX_ABFE_JOBS}"
    --dependency="afterok:${SETUP_JOB}"
    --export="ALL,CONFIG_FILE=${CONFIG_FILE}"
)
if (( ABFE_GPUS > 0 )); then
    SBATCH_ARGS+=(--gpus="${ABFE_GPUS}")
fi
exec sbatch "${SBATCH_ARGS[@]}" "${WORKER}"
