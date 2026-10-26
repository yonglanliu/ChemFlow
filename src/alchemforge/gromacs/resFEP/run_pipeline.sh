#!/usr/bin/env bash

#SBATCH --job-name=rbfe_fep_pipeline
#SBATCH --partition=norm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00


set -euo pipefail

# The caller owns environment activation. Sites that require environment
# modules can pass a shell fragment through ALCHEMFORGE_ENV_SETUP.
if [[ -n "${ALCHEMFORGE_ENV_SETUP:-}" ]]; then
    if [[ ! -f "${ALCHEMFORGE_ENV_SETUP}" ]]; then
        echo "ERROR: Environment setup script not found: ${ALCHEMFORGE_ENV_SETUP}" >&2
        exit 1
    fi
    source "${ALCHEMFORGE_ENV_SETUP}"
fi

echo "Python: $(command -v python || echo unavailable)"
echo "Conda environment: ${CONDA_PREFIX:-not active}"



# ============================================================
# Configuration
#
# Under sbatch, BASH_SOURCE points to Slurm's spool copy.
# Use the original submission directory instead.
# A config path may also be supplied explicitly as argument 1.
# ============================================================

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONFIG_FILE="${CONFIG_FILE:-${1:-${SUBMIT_DIR}/ResAlchemFEP_config.inp}}"
CONFIG_FILE="$(readlink -f "${CONFIG_FILE}")"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Project config not found:" >&2
    echo "    ${CONFIG_FILE}" >&2
    exit 1
fi

echo "Config file: ${CONFIG_FILE}"
source "${CONFIG_FILE}"

# Resource requests for each FEP array task. Values supplied by the desktop or
# TOML configuration override these portable defaults.
FEP_PARTITION="${FEP_PARTITION:-gpu}"
FEP_NODES="${FEP_NODES:-1}"
FEP_NTASKS="${FEP_NTASKS:-1}"
FEP_CPUS_PER_TASK="${FEP_CPUS_PER_TASK:-8}"
FEP_GPUS="${FEP_GPUS:-1}"
FEP_MEMORY="${FEP_MEMORY:-16G}"
FEP_WALLTIME="${FEP_WALLTIME:-08:00:00}"

# ============================================================
# Normalize / validate simulation mode
# ============================================================

SIMULATION_MODE="${SIMULATION_MODE:-RESUME}"
SIMULATION_MODE="${SIMULATION_MODE^^}"

case "${SIMULATION_MODE}" in
    OVERWRITE|SKIP|RESUME)
        ;;
    *)
        echo "ERROR: Invalid SIMULATION_MODE: ${SIMULATION_MODE}"
        echo "Allowed values: OVERWRITE, SKIP, RESUME"
        exit 1
        ;;
esac


# ============================================================
# Required configuration
# ============================================================

: "${WORK_DIR:?WORK_DIR must be defined in CONFIG_FILE}"
: "${JOB_NAME:?JOB_NAME must be defined in CONFIG_FILE}"
: "${START_REP:?START_REP must be defined in CONFIG_FILE}"
: "${NREP:?NREP must be defined in CONFIG_FILE}"
: "${NLAMBDA:?NLAMBDA must be defined in CONFIG_FILE}"
: "${MAX_FEP_JOBS:?MAX_FEP_JOBS must be defined in CONFIG_FILE}"
: "${FEP_DT_PS:?FEP_DT_PS must be defined in CONFIG_FILE}"
: "${FEP_NVT_PS:?FEP_NVT_PS must be defined in CONFIG_FILE}"
: "${FEP_NPT_PS:?FEP_NPT_PS must be defined in CONFIG_FILE}"
: "${FEP_PROD_NS:?FEP_PROD_NS must be defined in CONFIG_FILE}"
: "${FEP_NVT_STEPS:?FEP_NVT_STEPS must be defined in CONFIG_FILE}"
: "${FEP_NPT_STEPS:?FEP_NPT_STEPS must be defined in CONFIG_FILE}"
: "${FEP_PROD_STEPS:?FEP_PROD_STEPS must be defined in CONFIG_FILE}"
: "${FEP_LAMBDA_FUNCTION:?FEP_LAMBDA_FUNCTION must be defined in CONFIG_FILE}"
: "${SIMULATION_MODE:?SIMULATION_MODE must be defined in CONFIG_FILE}"

# ============================================================
# Derived replicate range
#
# Example:
#
# START_REP=4
# NREP=2
#
# -> run rep_4 and rep_5
# ============================================================

END_REP=$(( START_REP + NREP - 1 ))


# ============================================================
# Project directories
# ============================================================

PROJECT_DIR="${WORK_DIR}/${JOB_NAME}"

LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${LOG_DIR}"


# ============================================================
# Header
# ============================================================

echo
echo "============================================================"
echo "Equilibrium FEP master pipeline"
echo "============================================================"
echo "Date:          $(date)"
echo "Job name:      ${JOB_NAME}"
echo "Project dir:   ${PROJECT_DIR}"
echo "Master job:    ${SLURM_JOB_ID:-manual}"
echo "Simulation:    ${SIMULATION_MODE}"
echo "============================================================"


# ============================================================
# Pipeline script
#
# Preparation has already been completed.
# Analysis is NOT submitted by this pipeline.
# ============================================================

FEP_SCRIPT="${PROJECT_DIR}/ResAlchemFEP.sh"


# ============================================================
# Helper: check script
# ============================================================

check_script() {

    local script="$1"

    if [[ ! -f "${script}" ]]; then

        echo
        echo "ERROR: Script not found:"
        echo "    ${script}"

        exit 1
    fi


    if [[ ! -r "${script}" ]]; then

        echo
        echo "ERROR: Script is not readable:"
        echo "    ${script}"

        exit 1
    fi
}


# ============================================================
# Helper: check file
# ============================================================

check_file() {

    local file="$1"
    local description="$2"

    if [[ ! -s "${file}" ]]; then

        echo
        echo "ERROR: ${description} missing or empty:"
        echo "    ${file}"

        exit 1
    fi
}


# ============================================================
# Helper: safe SLURM submission
# ============================================================

submit_job() {

    local raw_job_id
    local job_id


    if ! raw_job_id=$(sbatch --parsable "$@"); then

        echo
        echo "ERROR: SLURM job submission failed." >&2

        printf "Command: sbatch" >&2

        for arg in "$@"
        do
            printf " %q" "${arg}" >&2
        done

        printf "\n" >&2

        return 1
    fi


    job_id="${raw_job_id%%;*}"


    if [[ -z "${job_id}" ]]; then

        echo "ERROR: sbatch returned an empty job ID." >&2

        return 1
    fi


    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then

        echo "ERROR: Invalid SLURM job ID:"
        echo "    ${job_id}"

        return 1
    fi


    echo "${job_id}"
}


# ============================================================
# Validate workflow script
# ============================================================

echo
echo "Checking FEP workflow files..."


check_script "${FEP_SCRIPT}"


check_file \
    "${PROJECT_DIR}/mdp/fep_base.mdp" \
    "base FEP MDP"


echo "Workflow script QC: PASS"


# ============================================================
# Configuration sanity checks
# ============================================================

if (( START_REP < 1 )); then

    echo "ERROR: START_REP must be >= 1"

    exit 1
fi


if (( NREP < 1 )); then

    echo "ERROR: NREP must be >= 1"

    exit 1
fi


if (( NLAMBDA < 2 )); then

    echo "ERROR: NLAMBDA must be >= 2"

    exit 1
fi


if (( MAX_FEP_JOBS < 1 )); then

    echo "ERROR: MAX_FEP_JOBS must be >= 1"

    exit 1
fi


if (( FEP_NVT_STEPS < 1 )); then

    echo "ERROR: FEP_NVT_STEPS must be >= 1"

    exit 1
fi


if (( FEP_NPT_STEPS < 1 )); then

    echo "ERROR: FEP_NPT_STEPS must be >= 1"

    exit 1
fi


if (( FEP_PROD_STEPS < 1 )); then

    echo "ERROR: FEP_PROD_STEPS must be >= 1"

    exit 1
fi

for RESOURCE_VAR in FEP_NODES FEP_NTASKS FEP_CPUS_PER_TASK FEP_GPUS
do
    if [[ ! "${!RESOURCE_VAR}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${RESOURCE_VAR} must be a non-negative integer"
        exit 1
    fi
done

if (( FEP_NODES < 1 || FEP_NTASKS < 1 || FEP_CPUS_PER_TASK < 1 )); then
    echo "ERROR: FEP_NODES, FEP_NTASKS, and FEP_CPUS_PER_TASK must be >= 1"
    exit 1
fi

if [[ -z "${FEP_PARTITION}" || -z "${FEP_MEMORY}" || -z "${FEP_WALLTIME}" ]]; then
    echo "ERROR: FEP_PARTITION, FEP_MEMORY, and FEP_WALLTIME cannot be empty"
    exit 1
fi


# ============================================================
# Validate prepared systems
#
# Only check requested replicate range:
#
#     START_REP ... END_REP
# ============================================================

echo
echo "============================================================"
echo "Checking prepared replicate systems"
echo "============================================================"


for LEG in apo bound
do

    for REP in $(seq "${START_REP}" "${END_REP}")
    do

        SETUP="${PROJECT_DIR}/${LEG}/rep_${REP}/setup"

        echo
        echo "Checking:"
        echo "    ${LEG} rep_${REP}"


        check_file \
            "${SETUP}/system.gro" \
            "${LEG} rep_${REP} system.gro"


        check_file \
            "${SETUP}/topol.top" \
            "${LEG} rep_${REP} topol.top"

        if grep -q "/AlchemForge/.alchemforge_env/" "${SETUP}/topol.top"; then
            echo
            echo "ERROR: Stale old-environment path detected:"
            echo "    ${SETUP}/topol.top"
            grep -n "/AlchemForge/.alchemforge_env/" "${SETUP}/topol.top" || true
            exit 1
        fi

        echo "    PASS"

    done

done


echo
echo "Prepared-system QC: PASS"


# ============================================================
# Calculate FEP array size
#
# NREP means number of NEW replicates to submit.
#
# Example:
#
# START_REP = 4
# NREP      = 2
# NLAMBDA   = 17
#
# Replicates:
#     rep_4
#     rep_5
#
# Total tasks:
#
#     2 legs × 2 replicates × 17 lambda
#     = 68 tasks
# ============================================================

N_LEGS=2

TASKS_PER_REP="${NLAMBDA}"

TASKS_PER_LEG=$(( NREP * NLAMBDA ))

N_FEP_TASKS=$(( N_LEGS * NREP * NLAMBDA ))

FEP_ARRAY_MAX=$(( N_FEP_TASKS - 1 ))


# ============================================================
# SLURM array
# ============================================================

FEP_ARRAY="0-${FEP_ARRAY_MAX}%${MAX_FEP_JOBS}"


# ============================================================
# Configuration summary
# ============================================================

echo
echo "============================================================"
echo "FEP configuration"
echo "============================================================"


echo
echo "Project:"
echo "    Job name:                ${JOB_NAME}"
echo "    Project directory:       ${PROJECT_DIR}"


echo
echo "Mutation:"
echo "    Chain:                   ${CHAIN:-NA}"
echo "    Residue:                 ${RESID:-NA}"
echo "    Mutation:                ${MUT:-NA}"


echo
echo "Replicates:"
echo "    Starting replicate:      ${START_REP}"
echo "    Ending replicate:        ${END_REP}"
echo "    Number to run:           ${NREP}"


echo
echo "Sampling:"
echo "    Lambda mode:             ${LAMBDA_MODE:-single}"
echo "    Lambda states:           ${NLAMBDA}"
echo "    Lambda function:         ${FEP_LAMBDA_FUNCTION}"
echo "    Lambda power:            ${FEP_LAMBDA_POWER:-NA}"


echo
echo "Integration:"
echo "    FEP timestep:            ${FEP_DT_PS} ps"


echo
echo "Equilibration:"
echo "    NVT:                     ${FEP_NVT_PS} ps"
echo "    NVT steps:               ${FEP_NVT_STEPS}"
echo "    NPT:                     ${FEP_NPT_PS} ps"
echo "    NPT steps:               ${FEP_NPT_STEPS}"


echo
echo "Production:"
echo "    Production / window:     ${FEP_PROD_NS} ns"
echo "    Production steps:        ${FEP_PROD_STEPS}"

echo
echo "Mode:"
echo "    Simulation mode:         ${SIMULATION_MODE}"


echo
echo "SLURM:"
echo "    Legs:                    ${N_LEGS}"
echo "    Tasks / replicate:       ${TASKS_PER_REP}"
echo "    Tasks / leg:             ${TASKS_PER_LEG}"
echo "    Total FEP tasks:         ${N_FEP_TASKS}"
echo "    Maximum simultaneous:    ${MAX_FEP_JOBS}"
echo "    Array:                   ${FEP_ARRAY}"
echo "    Partition:               ${FEP_PARTITION}"
echo "    Nodes / task:            ${FEP_NODES}"
echo "    SLURM tasks / array job: ${FEP_NTASKS}"
echo "    CPUs / task:             ${FEP_CPUS_PER_TASK}"
echo "    GPUs / task:             ${FEP_GPUS}"
echo "    Memory / task:           ${FEP_MEMORY}"
echo "    Walltime / task:         ${FEP_WALLTIME}"


echo
echo "============================================================"


# ============================================================
# Simulation mode handling
# ============================================================

case "${SIMULATION_MODE}" in

    OVERWRITE)
        echo
        echo "============================================================"
        echo "SIMULATION_MODE = OVERWRITE"
        echo "============================================================"
        echo "Existing FEP simulation directories for requested replicates"
        echo "will be removed before submission."

        for LEG in apo bound
        do
            for REP in $(seq "${START_REP}" "${END_REP}")
            do
                FEP_DIR="${PROJECT_DIR}/${LEG}/rep_${REP}/fep"

                if [[ -d "${FEP_DIR}" ]]; then
                    echo "Removing:"
                    echo "    ${FEP_DIR}"
                    rm -rf "${FEP_DIR}"
                fi
            done
        done
        ;;

    SKIP)
        echo
        echo "============================================================"
        echo "SIMULATION_MODE = SKIP"
        echo "============================================================"
        echo "FEP submission is skipped."
        echo "Prepared systems remain unchanged."
        echo "============================================================"

        exit 0
        ;;

    RESUME)
        echo
        echo "============================================================"
        echo "SIMULATION_MODE = RESUME"
        echo "============================================================"
        echo "Existing FEP directories are preserved."
        echo "Worker jobs may resume from available checkpoints."
        echo "============================================================"
        ;;

esac


# ============================================================
# Submit parallel FEP array
#
# No preparation dependency.
# No automatic analysis job.
# ============================================================

echo
echo "============================================================"
echo "Submit parallel FEP array"
echo "============================================================"


# Ensure the log directory exists before sbatch tries to open its files.
mkdir -p "${LOG_DIR}"

FEP_SBATCH_RESOURCES=(
    --partition="${FEP_PARTITION}"
    --nodes="${FEP_NODES}"
    --ntasks="${FEP_NTASKS}"
    --cpus-per-task="${FEP_CPUS_PER_TASK}"
    --mem="${FEP_MEMORY}"
    --time="${FEP_WALLTIME}"
)

if (( FEP_GPUS > 0 )); then
    FEP_SBATCH_RESOURCES+=(--gpus="${FEP_GPUS}")
fi

FEP_JOB=$(
    submit_job \
        "${FEP_SBATCH_RESOURCES[@]}" \
        --job-name="${JOB_NAME}_fep" \
        --array="${FEP_ARRAY}" \
        --output="${LOG_DIR}/fep_%A_%a.out" \
        --error="${LOG_DIR}/fep_%A_%a.err" \
        --export="ALL,CONFIG_FILE=${CONFIG_FILE},SIMULATION_MODE=${SIMULATION_MODE}" \
        "${FEP_SCRIPT}"
)


echo
echo "FEP array submitted:"
echo "    ${FEP_JOB}"

echo
echo "Array:"
echo "    ${FEP_ARRAY}"

echo
echo "Replicate range:"
echo "    rep_${START_REP} through rep_${END_REP}"

echo
echo "Dependency:"
echo "    none"


# ============================================================
# Save pipeline job information
# ============================================================

PIPELINE_JOB_FILE="${PROJECT_DIR}/pipeline_jobs.txt"


cat > "${PIPELINE_JOB_FILE}" <<EOF
submission_date=$(date)

job_name=${JOB_NAME}

fep_job=${FEP_JOB}

start_rep=${START_REP}
end_rep=${END_REP}
nrep=${NREP}

nlambda=${NLAMBDA}
n_fep_tasks=${N_FEP_TASKS}

max_fep_jobs=${MAX_FEP_JOBS}

fep_partition=${FEP_PARTITION}
fep_nodes=${FEP_NODES}
fep_ntasks=${FEP_NTASKS}
fep_cpus_per_task=${FEP_CPUS_PER_TASK}
fep_gpus=${FEP_GPUS}
fep_memory=${FEP_MEMORY}
fep_walltime=${FEP_WALLTIME}

simulation_mode=${SIMULATION_MODE}

fep_array=${FEP_ARRAY}
EOF


# ============================================================
# Final summary
# ============================================================

echo
echo "============================================================"
echo "Equilibrium FEP pipeline submitted successfully"
echo "============================================================"


echo
echo "Project:"
echo "    ${JOB_NAME}"


echo
echo "Replicates:"
echo "    rep_${START_REP} through rep_${END_REP}"


echo
echo "FEP array:"
echo "    Job ID: ${FEP_JOB}"
echo "    Array:  ${FEP_ARRAY}"


echo
echo "Analysis:"
echo "    not submitted automatically"


echo
echo "Monitor:"
echo
echo "    sjobs"


echo
echo "Detailed FEP status:"
echo
echo "    sacct -j ${FEP_JOB} \\"
echo "        --format=JobID,JobName,State,ExitCode,Elapsed"


echo
echo "Failed FEP tasks:"
echo
echo "    sacct -j ${FEP_JOB} \\"
echo "        --format=JobID,State,ExitCode,Elapsed \\"
echo "        | grep FAILED"


echo
echo "Pipeline job IDs:"
echo "    ${PIPELINE_JOB_FILE}"


echo
echo "============================================================"
