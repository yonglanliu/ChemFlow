#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:?Usage: $0 /path/to/charge_molecules.conf}"
source "${CONFIG_FILE}"

for variable in INPUT_SDF OUTPUT_SDF N_CORES SETTINGS_YAML; do
    if [[ -z "${!variable:-}" ]]; then
        echo "ERROR: ${variable} is missing from ${CONFIG_FILE}" >&2
        exit 1
    fi
done

OPENFE_COMMAND="${OPENFE_COMMAND:-openfe}"
if ! command -v "${OPENFE_COMMAND}" >/dev/null 2>&1; then
    echo "ERROR: OpenFE command is unavailable: ${OPENFE_COMMAND}" >&2
    echo "Activate the AlchemForge/OpenFE environment before launching the desktop." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_SDF}")"
arguments=(
    charge-molecules
    -M "${INPUT_SDF}"
    -o "${OUTPUT_SDF}"
    -n "${N_CORES}"
    -s "${SETTINGS_YAML}"
)
if [[ "${OVERWRITE_CHARGES:-false}" == "true" ]]; then
    arguments+=(--overwrite-charges)
fi

echo "Charging molecules with OpenFE"
echo "Input: ${INPUT_SDF}"
echo "Output: ${OUTPUT_SDF}"
echo "Processes: ${N_CORES}"
"${OPENFE_COMMAND}" "${arguments[@]}"

if [[ ! -s "${OUTPUT_SDF}" ]]; then
    echo "ERROR: OpenFE did not create a non-empty output SDF: ${OUTPUT_SDF}" >&2
    exit 1
fi
echo "Charged SDF created: ${OUTPUT_SDF}"

QC_DIR="$(dirname "${OUTPUT_SDF}")"
QC_STEM="$(basename "${OUTPUT_SDF}" .sdf)"
QC_HTML="${QC_DIR}/${QC_STEM}_charge_qc.html"
QC_CSV="${QC_DIR}/${QC_STEM}_charge_qc.csv"
python -m alchemforge.openfe.charge_qc \
    "${INPUT_SDF}" "${OUTPUT_SDF}" --html "${QC_HTML}" --csv "${QC_CSV}"
echo "Charge QC report: ${QC_HTML}"
echo "Charge QC table:  ${QC_CSV}"
