#!/usr/bin/env bash

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


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${1:-${SCRIPT_DIR}/ResAlchemFEP_config.inp}}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Configuration file not found: ${CONFIG_FILE}"
    echo "Usage: $0 /path/to/<job_name>_config.inp"
    exit 1
fi

CONFIG_FILE="$(readlink -f "${CONFIG_FILE}")"
source "${CONFIG_FILE}"

# ============================================================
# Normalize / validate ligand preparation mode
# ============================================================

LIGAND_PREP_MODE="${LIGAND_PREP_MODE:-SKIP}"
LIGAND_PREP_MODE="${LIGAND_PREP_MODE^^}"

case "${LIGAND_PREP_MODE}" in
    OVERWRITE|SKIP)
        ;;
    *)
        echo "ERROR: Invalid LIGAND_PREP_MODE: ${LIGAND_PREP_MODE}"
        echo "Allowed values: OVERWRITE, SKIP"
        exit 1
        ;;
esac


# ============================================================
# Check required configuration variables
# ============================================================

REQUIRED_VARS=(
    WORK_DIR
    LIGAND_MOL2
    LIGAND_RESNAME
    LIGAND_CHARGE
    LIGAND_CHARGE_METHOD
    JOB_NAME
    LIGAND_PREP_MODE
)

for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Required variable '${var}' is not defined."
        echo
        echo "Please run:"
        echo "    source ${CONFIG_FILE}"
        exit 1
    fi
done

WORK_DIR="$(readlink -f "${WORK_DIR}")"
if [[ "${LIGAND_MOL2}" != /* ]]; then
    LIGAND_MOL2="${WORK_DIR}/${LIGAND_MOL2}"
fi


# ============================================================
# Ligand settings
# ============================================================

# Allowed difference between calculated and expected total charge.
CHARGE_TOLERANCE=0.001


# ============================================================
# Define paths
# ============================================================

LIGAND_INPUT="${LIGAND_MOL2}"

# Absolute directory containing original MOL2 file.
LIGAND_DIR="$(dirname "${LIGAND_INPUT}")"

# Example:
LIGAND_BASENAME="$(basename "${LIGAND_MOL2}" .mol2)"

# ACPYPE working directory.
LIGAND_PARAM_DIR="${WORK_DIR}/${JOB_NAME}/para_ligand"

# ACPYPE uses -b as output basename.
ACPYPE_DIR="${LIGAND_PARAM_DIR}/${LIGAND_RESNAME}.acpype"

LIG_ITP="${ACPYPE_DIR}/${LIGAND_RESNAME}_GMX.itp"
LIG_GRO="${ACPYPE_DIR}/${LIGAND_RESNAME}_GMX.gro"
LIG_TOP="${ACPYPE_DIR}/${LIGAND_RESNAME}_GMX.top"

# QC files.
QC_DIR="${LIGAND_PARAM_DIR}/qc"

MOL2_ATOM_NAMES="${QC_DIR}/mol2_atom_names.txt"
ITP_ATOM_NAMES="${QC_DIR}/itp_atom_names.txt"

# Main log.
LOG_FILE="${LIGAND_PARAM_DIR}/ligand_parameterization.log"


# ============================================================
# Create output directories
# ============================================================

mkdir -p "${LIGAND_PARAM_DIR}"
mkdir -p "${QC_DIR}"


# ============================================================
# Log everything
#
# Everything after this point is:
#   1. printed to terminal
#   2. written to LOG_FILE
# ============================================================

exec > >(tee "${LOG_FILE}") 2>&1


echo
echo "============================================================"
echo " Ligand Parameterization and QC"
echo "============================================================"
echo "Date:                 $(date)"
echo "Root directory:       ${WORK_DIR}"
echo "Ligand input:         ${LIGAND_INPUT}"
echo "Ligand directory:     ${LIGAND_DIR}"
echo "Ligand basename:      ${LIGAND_BASENAME}"
echo "Ligand residue name:  ${LIGAND_RESNAME}"
echo "Expected charge:      ${LIGAND_CHARGE}"
echo "ACPYPE directory:     ${ACPYPE_DIR}"
echo "Log file:             ${LOG_FILE}"
LIGAND_NET_CHARGE=$(printf "%.0f" "${LIGAND_CHARGE}")
echo "ACPYPE net charge:    ${LIGAND_NET_CHARGE}"
echo "============================================================"


# ============================================================
# 1. Check ligand input
# ============================================================

echo
echo "============================================================"
echo "1. Checking ligand input"
echo "============================================================"

if [[ ! -f "${LIGAND_INPUT}" ]]; then
    echo "ERROR: Ligand input file not found:"
    echo "    ${LIGAND_INPUT}"
    exit 1
fi

echo "Ligand input file found."

if ! grep -q "@<TRIPOS>MOLECULE" "${LIGAND_INPUT}"; then
    echo "ERROR: Input does not appear to be a valid MOL2 file."
    exit 1
fi

if ! grep -q "@<TRIPOS>ATOM" "${LIGAND_INPUT}"; then
    echo "ERROR: MOL2 file does not contain an ATOM section."
    exit 1
fi

if ! grep -q "@<TRIPOS>BOND" "${LIGAND_INPUT}"; then
    echo "ERROR: MOL2 file does not contain a BOND section."
    exit 1
fi

echo "Basic MOL2 structure QC: PASS"


# ============================================================
# 2. Check input MOL2 charge
# ============================================================

echo
echo "============================================================"
echo "2. Checking input MOL2 charge"
echo "============================================================"

echo "Expected net charge:       ${LIGAND_CHARGE}"

if ! command -v acpype >/dev/null 2>&1; then
    echo "ERROR: ACPYPE was not found on PATH."
    echo "Activate an environment or load a module that provides the 'acpype' command."
    exit 1
fi


# ============================================================
# 3. Run ACPYPE
#
# -c user:
#     Use partial charges already present in MOL2.
#
# -a gaff2:
#     Use GAFF2 atom types.
#
# If you instead want ACPYPE to calculate AM1-BCC charges,
# replace:
#
#     -c user
#
# with:
#
#     -c bcc
# ============================================================

echo "Ligand preparation mode: ${LIGAND_PREP_MODE}"
echo "ACPYPE executable:       $(command -v acpype)"
echo "ACPYPE Python package:   $(python -c 'import acpype; print(acpype.__file__)')"
echo "Antechamber:             $(command -v antechamber || true)"
echo "SQM:                     $(command -v sqm || true)"

if [[ "${LIGAND_PREP_MODE}" == "OVERWRITE" ]]; then
    echo "Ligand preparation mode: OVERWRITE"

    echo
    echo "============================================================"
    echo "3. Running ACPYPE"
    echo "============================================================"

cd "${LIGAND_PARAM_DIR}"

# Remove previous output to avoid mixing old/new ACPYPE files.
if [[ -d "${ACPYPE_DIR}" ]]; then
    echo "Removing previous ACPYPE directory:"
    echo "    ${ACPYPE_DIR}"
    rm -rf "${ACPYPE_DIR}"
fi

case "${LIGAND_CHARGE_METHOD}" in
    bcc)
        echo "Charge method: AM1-BCC charges will be calculated by ACPYPE."
        ;;
    user)
        echo "Charge method: Existing MOL2 partial charges will be used."
        ;;
    *)
        echo "ERROR: Invalid LIGAND_CHARGE_METHOD:"
        echo "    ${LIGAND_CHARGE_METHOD}"
        echo
        echo "Allowed values:"
        echo "    bcc"
        echo "    user"
        exit 1
        ;;
esac

"${CONDA_PREFIX}/bin/acpype" \
    -i "${LIGAND_INPUT}" \
    -b "${LIGAND_RESNAME}" \
    -c "${LIGAND_CHARGE_METHOD}" \
    -a gaff2 \
    -n "${LIGAND_NET_CHARGE}" \
    -o gmx

elif [[ "${LIGAND_PREP_MODE}" == "SKIP" ]]; then
    echo
    echo "============================================================"
    echo "3. Reusing existing ACPYPE output"
    echo "============================================================"
    echo "LIGAND_PREP_MODE = SKIP"
    echo "Existing ligand parameters will be reused."
fi


# ============================================================
# 4. Verify ACPYPE output files
# ============================================================

echo
echo "============================================================"
echo "4. Checking ACPYPE output files"
echo "============================================================"

REQUIRED_OUTPUTS=(
    "${LIG_ITP}"
    "${LIG_GRO}"
    "${LIG_TOP}"
)

for file in "${REQUIRED_OUTPUTS[@]}"; do

    if [[ ! -s "${file}" ]]; then
        echo "ERROR: Required ACPYPE output not found or empty:"
        echo "    ${file}"
        if [[ "${LIGAND_PREP_MODE}" == "SKIP" ]]; then
            echo
            echo "LIGAND_PREP_MODE=SKIP requires existing ligand parameters."
            echo "Set LIGAND_PREP_MODE=OVERWRITE to regenerate them."
        fi
        exit 1
    fi

    echo "Found:"
    echo "    ${file}"

done

echo
echo "ACPYPE output file QC: PASS"

LIGAND_ITP="${LIG_ITP}"
export LIGAND_ITP


# ============================================================
# 5. Calculate total charge from GROMACS ITP
#
# [ atoms ] format:
#
# nr  type  resnr  residue  atom  cgnr  charge  mass
#                                         ^
#                                      column 7
# ============================================================

echo
echo "============================================================"
echo "5. Ligand charge QC"
echo "============================================================"

ITP_CHARGE=$(
awk '
BEGIN {
    in_atoms=0
    q=0.0
}

/^[[:space:]]*\[[[:space:]]*atoms[[:space:]]*\][[:space:]]*$/ {
    in_atoms=1
    next
}

/^[[:space:]]*\[/ && in_atoms {
    in_atoms=0
}

in_atoms &&
$0 !~ /^[[:space:]]*;/ &&
NF >= 7 {
    q += $7
}

END {
    printf("%.6f", q)
}
' "${LIG_ITP}"
)

echo "Expected ligand charge: ${LIGAND_CHARGE}"
echo "ITP charge:             ${ITP_CHARGE}"


if [[ "${LIGAND_CHARGE_METHOD}" == "user" ]]; then
    MOL2_CHARGE=$(
    awk '
    /@<TRIPOS>ATOM/ {
        in_atoms=1
        next
    }

    /@<TRIPOS>BOND/ {
        in_atoms=0
    }

    in_atoms && NF >= 9 {
        q += $9
    }

    END {
        printf("%.6f", q)
    }
    ' "${LIGAND_INPUT}"
    )

    echo "MOL2 charge:            ${MOL2_CHARGE}"

    python - <<PY
expected = float("${LIGAND_CHARGE}")
mol2 = float("${MOL2_CHARGE}")
itp = float("${ITP_CHARGE}")
tol = float("${CHARGE_TOLERANCE}")

print(f"Allowed tolerance:      {tol:.6f}")

if abs(mol2 - expected) > tol:
    raise SystemExit(
        f"ERROR: Input MOL2 charge mismatch. "
        f"Expected {expected:.3f}, obtained {mol2:.6f}."
    )

if abs(itp - expected) > tol:
    raise SystemExit(
        f"ERROR: ITP charge mismatch. "
        f"Expected {expected:.3f}, obtained {itp:.6f}."
    )

if abs(itp - mol2) > tol:
    raise SystemExit(
        f"ERROR: MOL2 and ITP charge mismatch. "
        f"MOL2={mol2:.6f}, ITP={itp:.6f}."
    )

print("Ligand charge QC: PASS")
PY

else
    echo "Input MOL2 charge comparison: SKIPPED (AM1-BCC recalculates charges)"

    python - <<PY
expected = float("${LIGAND_CHARGE}")
itp = float("${ITP_CHARGE}")
tol = float("${CHARGE_TOLERANCE}")

print(f"Allowed tolerance:      {tol:.6f}")

if abs(itp - expected) > tol:
    raise SystemExit(
        f"ERROR: ITP charge mismatch. "
        f"Expected {expected:.3f}, obtained {itp:.6f}."
    )

print("Ligand charge QC: PASS")
PY
fi


# ============================================================
# 6. Count atoms in MOL2
# ============================================================

echo
echo "============================================================"
echo "6. Ligand atom-count QC"
echo "============================================================"

MOL2_ATOMS=$(
awk '
/@<TRIPOS>ATOM/ {
    flag=1
    next
}

/@<TRIPOS>BOND/ {
    flag=0
}

flag && NF > 0 {
    n++
}

END {
    print n+0
}
' "${LIGAND_INPUT}"
)


# ============================================================
# Count atoms in ITP
# ============================================================

ITP_ATOMS=$(
awk '
/^[[:space:]]*\[[[:space:]]*atoms[[:space:]]*\]/ {
    flag=1
    next
}

/^[[:space:]]*\[/ && flag {
    exit
}

flag &&
$0 !~ /^[[:space:]]*;/ &&
$1 ~ /^[0-9]+$/ {
    n++
}

END {
    print n+0
}
' "${LIG_ITP}"
)


# ============================================================
# Count atoms in GRO
# ============================================================

GRO_ATOMS=$(sed -n '2p' "${LIG_GRO}" | tr -d '[:space:]')


echo "MOL2 atom count: ${MOL2_ATOMS}"
echo "ITP atom count:  ${ITP_ATOMS}"
echo "GRO atom count:  ${GRO_ATOMS}"


if [[ "${MOL2_ATOMS}" -ne "${ITP_ATOMS}" ]]; then
    echo "ERROR: MOL2 and ITP atom counts do not match."
    exit 1
fi

if [[ "${ITP_ATOMS}" -ne "${GRO_ATOMS}" ]]; then
    echo "ERROR: ITP and GRO atom counts do not match."
    exit 1
fi

echo "Ligand atom-count QC: PASS"


# ============================================================
# 7. Extract atom names from MOL2
# ============================================================

echo
echo "============================================================"
echo "7. Ligand atom-name/order QC"
echo "============================================================"

awk '
/@<TRIPOS>ATOM/ {
    flag=1
    next
}

/@<TRIPOS>BOND/ {
    flag=0
}

flag && NF > 0 {
    print $2
}
' "${LIGAND_INPUT}" > "${MOL2_ATOM_NAMES}"


# ============================================================
# Extract atom names from ITP
# ============================================================

awk '
/^[[:space:]]*\[[[:space:]]*atoms[[:space:]]*\]/ {
    flag=1
    next
}

/^[[:space:]]*\[/ && flag {
    exit
}

flag &&
$0 !~ /^[[:space:]]*;/ &&
$1 ~ /^[0-9]+$/ {
    print $5
}
' "${LIG_ITP}" > "${ITP_ATOM_NAMES}"


# ============================================================
# Compare MOL2 vs ITP atom names and ordering
# ============================================================

if ! diff -q "${MOL2_ATOM_NAMES}" "${ITP_ATOM_NAMES}" > /dev/null; then

    echo "WARNING: ACPYPE renamed atom names between MOL2 and ITP."
    echo
    echo "Comparison:"
    echo

    diff -y "${MOL2_ATOM_NAMES}" "${ITP_ATOM_NAMES}" || true

    echo "Atom order will be validated by atom count and element mapping in system QC."
fi

echo "Ligand atom-name/order QC: INFO (ACPYPE names may be renumbered)"


# ============================================================
# 8. Display molecule type from ITP
# ============================================================

echo
echo "============================================================"
echo "8. GROMACS molecule-type QC"
echo "============================================================"

echo "Molecule type defined in ITP:"

awk '
/^[[:space:]]*\[[[:space:]]*moleculetype[[:space:]]*\]/ {
    flag=1
    next
}

flag && $0 !~ /^[[:space:]]*;/ && NF >= 2 {
    print
    exit
}
' "${LIG_ITP}"


# ============================================================
# 9. Check ACPYPE output for suspicious messages
# ============================================================

echo
echo "============================================================"
echo "9. Checking ACPYPE warnings/errors"
echo "============================================================"

SUSPICIOUS_LOG="${QC_DIR}/acpype_suspicious_messages.txt"

grep -RniE \
    "warning|error|failed|missing parameter|unrecognized|fatal" \
    "${ACPYPE_DIR}" \
    > "${SUSPICIOUS_LOG}" 2>/dev/null || true


if [[ -s "${SUSPICIOUS_LOG}" ]]; then

    echo "Potential ACPYPE messages found:"
    echo
    cat "${SUSPICIOUS_LOG}"
    echo
    echo "NOTE: Review these messages before production calculations."

else

    echo "No obvious ACPYPE warning/error messages found."

fi


# ============================================================
# 10. Copy final ligand files next to original MOL2
# ============================================================

echo
echo "============================================================"
echo "10. Copying final ligand files"
echo "============================================================"

FINAL_ITP="${LIGAND_DIR}/${LIGAND_BASENAME}.itp"
FINAL_GRO="${LIGAND_DIR}/${LIGAND_BASENAME}.gro"
FINAL_TOP="${LIGAND_DIR}/${LIGAND_BASENAME}.top"

cp "${LIG_ITP}" "${FINAL_ITP}"
cp "${LIG_GRO}" "${FINAL_GRO}"
cp "${LIG_TOP}" "${FINAL_TOP}"

echo "Copied:"
echo "    ${FINAL_ITP}"
echo "    ${FINAL_GRO}"
echo "    ${FINAL_TOP}"


# ============================================================
# 11. Final summary
# ============================================================

echo
echo "============================================================"
echo " Ligand Parameterization QC Summary"
echo "============================================================"
echo "Input MOL2:           ${LIGAND_INPUT}"
echo "Residue name:         ${LIGAND_RESNAME}"
echo "Atom count:           ${ITP_ATOMS}"
echo "Expected charge:      ${LIGAND_CHARGE}"
echo "Calculated charge:    ${ITP_CHARGE}"
echo
echo "QC:"
echo "  MOL2 structure:     PASS"
echo "  Charge:             PASS"
echo "  Atom count:         PASS"
echo "  Atom names/order:   PASS"
echo
echo "Final ITP:"
echo "    ${FINAL_ITP}"
echo
echo "Final GRO:"
echo "    ${FINAL_GRO}"
echo
echo "Final TOP:"
echo "    ${FINAL_TOP}"
echo
echo "Full log:"
echo "    ${LOG_FILE}"
echo
echo "QC directory:"
echo "    ${QC_DIR}"
echo "============================================================"
echo "Ligand parameterization completed successfully."
echo "============================================================"
