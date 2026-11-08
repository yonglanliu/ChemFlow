#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-${1:?Usage: $0 /path/to/abfe_config.sh [task-id]}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${2:-}}"
source "${CONFIG_FILE}"

if [[ -z "${TASK_ID}" ]]; then
    echo "ERROR: An array task ID is required." >&2
    exit 1
fi
if (( TASK_ID < 0 || TASK_ID >= N_ABFE_TASKS )); then
    echo "ERROR: Task ID ${TASK_ID} is outside 0..$((N_ABFE_TASKS - 1))." >&2
    exit 1
fi

PROJECT_DIR="${WORK_DIR}/${JOB_NAME}"
TASKS_PER_LEG=$((NREP * NLAMBDA))
LEG_INDEX=$((TASK_ID / TASKS_PER_LEG))
LEG_OFFSET=$((TASK_ID % TASKS_PER_LEG))
REP=$((LEG_OFFSET / NLAMBDA + 1))
WINDOW=$((LEG_OFFSET % NLAMBDA))

if (( LEG_INDEX == 0 )); then
    LEG="complex"
    SOURCE_LEG="complex"
    TOP_NAME="complex.top"
else
    LEG="solvent"
    SOURCE_LEG="ligand"
    TOP_NAME="ligand.top"
fi

RUN_DIR="${PROJECT_DIR}/${LEG}/rep_${REP}/lambda_$(printf '%03d' "${WINDOW}")"
SOURCE_DIR="${PROJECT_DIR}/setup/${SOURCE_LEG}"
MDP_DIR="${PROJECT_DIR}/mdp/${LEG}/lambda_$(printf '%03d' "${WINDOW}")"
mkdir -p "${RUN_DIR}"

if [[ ! -f "${RUN_DIR}/.inputs_ready" ]]; then
    cp -a "${SOURCE_DIR}/." "${RUN_DIR}/"
    touch "${RUN_DIR}/.inputs_ready"
fi
cd "${RUN_DIR}"

run_stage() {
    local stage="$1"
    local coordinates="$2"
    local checkpoint="${3:-}"
    if [[ -s "${stage}.gro" ]]; then
        echo "Reusing completed ${LEG} replicate ${REP} window ${WINDOW} ${stage}"
        return
    fi
    local seed=$((RESTRAINT_SEED + TASK_ID + 1))
    sed "s/__GEN_SEED__/${seed}/g" "${MDP_DIR}/${stage}.mdp" > "${stage}.mdp"
    local grompp_args=(
        -f "${stage}.mdp" -c "${coordinates}" -r "${coordinates}"
        -p "${TOP_NAME}" -o "${stage}.tpr" -maxwarn 1
    )
    if [[ -n "${checkpoint}" && -s "${checkpoint}" ]]; then
        grompp_args+=(-t "${checkpoint}")
    fi
    "${GMX}" grompp "${grompp_args[@]}"
    # shellcheck disable=SC2086
    "${GMX}" mdrun -deffnm "${stage}" ${MDRUN_OPTIONS}
}

run_stage em genion.gro
run_stage nvt em.gro
run_stage npt nvt.gro nvt.cpt
run_stage prod npt.gro npt.cpt

echo "Completed ${LEG} replicate ${REP} lambda window ${WINDOW}"
