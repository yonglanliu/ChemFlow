#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:?Usage: $0 /path/to/abfe_config.sh}"
source "${CONFIG_FILE}"

PROJECT_DIR="${WORK_DIR}/${JOB_NAME}"
SETUP_DIR="${PROJECT_DIR}/setup"
INPUT_DIR="${PROJECT_DIR}/input"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${SETUP_DIR}" "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/setup.log") 2>&1

if [[ -n "${PREPARED_ABFE_SETUP:-}" ]]; then
    echo "Installing prepared ${SYSTEM_TYPE:-soluble} ABFE setup"
    if [[ ! -d "${PREPARED_ABFE_SETUP}" ]]; then
        echo "ERROR: Prepared ABFE setup does not exist: ${PREPARED_ABFE_SETUP}" >&2
        exit 1
    fi
    mkdir -p "${SETUP_DIR}"
    cp -a "${PREPARED_ABFE_SETUP}/." "${SETUP_DIR}/"
    echo "Prepared ABFE setup installed: ${SETUP_DIR}"
    exit 0
fi

for command in "${GMX}" obabel acpype python; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "ERROR: Required ABFE command is not available: ${command}" >&2
        exit 1
    fi
done
if ! python -c "import pmx" >/dev/null 2>&1; then
    echo "ERROR: PMX is not installed in the active Python environment." >&2
    exit 1
fi

PROTEIN_INPUT="${INPUT_DIR}/$(basename "${PROTEIN}")"
LIGAND_INPUT="${INPUT_DIR}/$(basename "${LIGAND_SDF}")"

if [[ "${ABFE_SETUP_MODE:-RESUME}" == "RESUME" && \
      -s "${SETUP_DIR}/complex/genion.gro" && \
      -s "${SETUP_DIR}/complex/complex.top" && \
      -s "${SETUP_DIR}/ligand/genion.gro" && \
      -s "${SETUP_DIR}/ligand/ligand.top" ]]; then
    echo "Reusing completed ABFE setup in ${SETUP_DIR}"
    exit 0
fi

rm -rf "${SETUP_DIR}/protein" "${SETUP_DIR}/ligand_parameters" \
       "${SETUP_DIR}/complex" "${SETUP_DIR}/ligand"
mkdir -p "${SETUP_DIR}/protein" "${SETUP_DIR}/ligand_parameters"

echo "Preparing protein topology with GROMACS"
cd "${SETUP_DIR}/protein"
"${GMX}" pdb2gmx \
    -f "${PROTEIN_INPUT}" -o protein.gro -p protein.top -i posre.itp \
    -ff "${FF}" -water "${WATER}" -ignh

echo "Preparing ligand topology with Open Babel and ACPYPE"
cd "${SETUP_DIR}/ligand_parameters"
obabel "${LIGAND_INPUT}" -O ligand.mol2
acpype \
    -i ligand.mol2 -b "${LIGAND_RESNAME}" -c "${CHARGE_METHOD}" \
    -a gaff2 -n "${LIGAND_CHARGE}" -o gmx

ACPYPE_DIR="${SETUP_DIR}/ligand_parameters/${LIGAND_RESNAME}.acpype"
LIGAND_ITP="${ACPYPE_DIR}/${LIGAND_RESNAME}_GMX.itp"
LIGAND_GRO="${ACPYPE_DIR}/${LIGAND_RESNAME}_GMX.gro"
if [[ ! -s "${LIGAND_ITP}" || ! -s "${LIGAND_GRO}" ]]; then
    echo "ERROR: ACPYPE did not create the expected GROMACS ligand files." >&2
    exit 1
fi

echo "Generating Boresch restraints and solvated ABFE legs with PMX"
cd "${SETUP_DIR}"
python "${PROJECT_DIR}/.alchemforge/pmx_setup.py" \
    --protein-top protein/protein.top --protein-gro protein/protein.gro \
    --ligand-itp "${LIGAND_ITP}" --ligand-gro "${LIGAND_GRO}" \
    --temperature "${TEMPERATURE}" \
    --distance-k "${RESTRAINT_K_DISTANCE}" \
    --angle-k "${RESTRAINT_K_ANGLE}" \
    --dihedral-k "${RESTRAINT_K_DIHEDRAL}" \
    --seed "${RESTRAINT_SEED}"

for required in complex/genion.gro complex/complex.top ligand/genion.gro ligand/ligand.top restraints.info; do
    if [[ ! -s "${SETUP_DIR}/${required}" ]]; then
        echo "ERROR: PMX ABFE setup did not create ${required}" >&2
        exit 1
    fi
done

echo "ABFE setup complete: ${SETUP_DIR}"
