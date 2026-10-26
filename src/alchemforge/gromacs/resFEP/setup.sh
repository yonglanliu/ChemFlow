#!/usr/bin/env bash

set -e

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
# Configuration file
#
# Usage:
#
#     bash scripts/rbfe/residue_alchemical_fep/01_setup.sh /path/to/config/Res_Alchemical_FEP_config.inp
#
# If no path is supplied, use the project-level config file in ../config.
# ============================================================

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
# Normalize / validate system setup mode
# ============================================================

SYSTEM_SETUP_MODE="${SYSTEM_SETUP_MODE:-SKIP}"
SYSTEM_SETUP_MODE="${SYSTEM_SETUP_MODE^^}"

case "${SYSTEM_SETUP_MODE}" in
    OVERWRITE|SKIP)
        ;;
    *)
        echo "ERROR: Invalid SYSTEM_SETUP_MODE: ${SYSTEM_SETUP_MODE}"
        echo "Allowed values: OVERWRITE, SKIP"
        exit 1
        ;;
esac

echo "Config file:    ${CONFIG_FILE}"
echo "Working Directory: ${WORK_DIR}"

# ============================================================
# Check required variables
# ============================================================

REQUIRED_VARS=(
    WORK_DIR
    GMX
    FF
    WATER
    COMPLEX
    CHAIN
    RESID
    MUT
    BOX_DISTANCE
    ION_CONC
    LIGAND_MOL2
    LIGAND_RESNAME
    LIGAND_CHARGE
    LIGAND_CHARGE_METHOD
    JOB_NAME
    START_REP
    NREP
    SYSTEM_SETUP_MODE
)

for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Required variable '${var}' is not defined."
        echo "Please check ${CONFIG_FILE}"
        exit 1
    fi
done


# ============================================================
# Replicate range
# ============================================================

if (( START_REP < 1 )); then
    echo "ERROR: START_REP must be >= 1"
    exit 1
fi

if (( NREP < 1 )); then
    echo "ERROR: NREP must be >= 1"
    exit 1
fi

END_REP=$(( START_REP + NREP - 1 ))

LIGAND_ITP="${WORK_DIR}/${JOB_NAME}/para_ligand/${LIGAND_RESNAME}.acpype/${LIGAND_RESNAME}_GMX.itp"
export LIGAND_ITP


# ============================================================
# Project directory
# ============================================================

PROJECT_DIR="${WORK_DIR}/${JOB_NAME}"
mkdir -p "${PROJECT_DIR}"

IONS_MDP="${PROJECT_DIR}/mdp/ions.mdp"
if [[ ! -s "${IONS_MDP}" ]]; then
    mkdir -p "$(dirname "${IONS_MDP}")"
    cp "${SCRIPT_DIR}/mdp/ions.mdp" "${IONS_MDP}"
fi

# ============================================================
# Logging
# ============================================================

LOG_DIR="${WORK_DIR}/${JOB_NAME}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/01_prepare_system.log"

exec > >(tee "${LOG_FILE}") 2>&1


echo "============================================================"
echo "System preparation started"
echo "============================================================"
echo "Date:              $(date)"
echo "Working directory:    ${WORK_DIR}"
echo "Complex:           ${COMPLEX}"
echo "Mutation:          ${CHAIN} ${RESID} ${MUT}"
echo "Force field:       ${FF}"
echo "Water model:       ${WATER}"
echo "Ligand residue:    ${LIGAND_RESNAME}"
echo "Ligand topology:   ${LIGAND_ITP}"
echo "Log file:          ${LOG_FILE}"
echo "Config file:       ${CONFIG_FILE}"
echo "Start replicate:   ${START_REP}"
echo "End replicate:     ${END_REP}"
echo "Replicates to add: ${NREP}"
echo "System setup mode: ${SYSTEM_SETUP_MODE}"
echo "============================================================"


# ============================================================
# Prepare ligand if needed
# ============================================================

prepare_ligand() {
    local ligand_input="${LIGAND_MOL2}"
    local ligand_dir
    local ligand_basename
    local ligand_param_dir
    local acpype_dir
    local lig_itp
    local lig_gro
    local lig_top

    if [[ -z "${ligand_input}" ]]; then
        echo "ERROR: LIGAND_MOL2 is not defined."
        exit 1
    fi

    if [[ "${ligand_input}" != /* ]]; then
        ligand_input="${WORK_DIR}/${ligand_input}"
    fi

    if [[ ! -f "${ligand_input}" ]]; then
        echo "ERROR: Ligand MOL2 file not found:"
        echo "    ${ligand_input}"
        exit 1
    fi

    ligand_dir="$(dirname "${ligand_input}")"
    ligand_basename="$(basename "${ligand_input}" .mol2)"
    ligand_param_dir="${WORK_DIR}/${JOB_NAME}/para_ligand"
    acpype_dir="${ligand_param_dir}/${LIGAND_RESNAME}.acpype"
    lig_itp="${acpype_dir}/${LIGAND_RESNAME}_GMX.itp"
    lig_gro="${acpype_dir}/${LIGAND_RESNAME}_GMX.gro"
    lig_top="${acpype_dir}/${LIGAND_RESNAME}_GMX.top"

    # Keep a project-local copy of the para-ligand helper script.
    local para_src="${SCRIPT_DIR}/para_ligand.sh"
    if [[ ! -f "${para_src}" ]]; then
        para_src="${WORK_DIR}/scripts/common/para_ligand.sh"
    fi
    local para_copy="${ligand_param_dir}/${LIGAND_RESNAME}_para_ligand.sh"
    if [[ -f "${para_src}" ]]; then
        cp -f "${para_src}" "${para_copy}"
    fi

    if [[ -s "${LIGAND_ITP}" ]]; then
        echo "Using existing ligand topology:"
        echo "    ${LIGAND_ITP}"
        return 0
    fi

    echo
    echo "============================================================"
    echo "Preparing ligand topology with ACPYPE"
    echo "============================================================"
    echo "Ligand input:        ${ligand_input}"
    echo "Ligand residue:      ${LIGAND_RESNAME}"
    echo "Expected charge:     ${LIGAND_CHARGE}"
    echo "Charge method:       ${LIGAND_CHARGE_METHOD}"

    mkdir -p "${ligand_param_dir}"
    cd "${ligand_param_dir}"

    if [[ -d "${acpype_dir}" ]]; then
        rm -rf "${acpype_dir}"
    fi

    case "${LIGAND_CHARGE_METHOD}" in
        bcc|user)
            ;;
        *)
            echo "ERROR: Invalid LIGAND_CHARGE_METHOD: ${LIGAND_CHARGE_METHOD}"
            exit 1
            ;;
    esac

    acpype \
        -i "${ligand_input}" \
        -b "${LIGAND_RESNAME}" \
        -c "${LIGAND_CHARGE_METHOD}" \
        -a gaff2 \
        -n "${LIGAND_CHARGE}" \
        -o gmx

    if [[ ! -s "${lig_itp}" || ! -s "${lig_gro}" || ! -s "${lig_top}" ]]; then
        echo "ERROR: ACPYPE ligand generation failed. Missing required outputs:"
        echo "    ${lig_itp}"
        echo "    ${lig_gro}"
        echo "    ${lig_top}"
        exit 1
    fi

    LIGAND_ITP="${lig_itp}"
    export LIGAND_ITP

    cp "${lig_itp}" "${ligand_dir}/${ligand_basename}.itp"
    cp "${lig_gro}" "${ligand_dir}/${ligand_basename}.gro"
    cp "${lig_top}" "${ligand_dir}/${ligand_basename}.top"

    echo "Generated ligand topology:"
    echo "    ${LIGAND_ITP}"
}

prepare_ligand

# ============================================================
# Check input files
# ============================================================

if [[ ! -f "${COMPLEX}" ]]; then
    echo "ERROR: Complex file not found:"
    echo "    ${COMPLEX}"
    exit 1
fi

if [[ ! -f "${LIGAND_ITP}" ]]; then
    echo "ERROR: Ligand ITP file not found:"
    echo "    ${LIGAND_ITP}"
    exit 1
fi

if [[ ! -f "${IONS_MDP}" ]]; then
    echo "ERROR: ions.mdp not found:"
    echo "    ${IONS_MDP}"
    exit 1
fi


# ============================================================
# Define directories
# ============================================================

COMMON_DIR="${WORK_DIR}/${JOB_NAME}/common"
APO_SETUP="${WORK_DIR}/${JOB_NAME}/apo/setup"
BOUND_SETUP="${WORK_DIR}/${JOB_NAME}/bound/setup"

mkdir -p "${COMMON_DIR}"
mkdir -p "${APO_SETUP}"
mkdir -p "${BOUND_SETUP}"
mkdir -p "${LOG_DIR}"


# ============================================================
# Base-system setup mode
# ============================================================

BASE_SETUP_READY=0

if [[ "${SYSTEM_SETUP_MODE}" == "OVERWRITE" ]]; then

    echo
    echo "============================================================"
    echo "SYSTEM_SETUP_MODE = OVERWRITE"
    echo "============================================================"
    echo "Removing existing APO/BOUND setup directories."

    rm -rf "${APO_SETUP}" "${BOUND_SETUP}"
    mkdir -p "${APO_SETUP}" "${BOUND_SETUP}"

elif [[ "${SYSTEM_SETUP_MODE}" == "SKIP" ]]; then

    if [[ -s "${APO_SETUP}/system.gro" && \
          -s "${APO_SETUP}/topol.top" && \
          -s "${BOUND_SETUP}/system.gro" && \
          -s "${BOUND_SETUP}/topol.top" ]]; then

        BASE_SETUP_READY=1

        echo
        echo "============================================================"
        echo "SYSTEM_SETUP_MODE = SKIP"
        echo "============================================================"
        echo "Existing APO/BOUND base setups found."
        echo "Reusing:"
        echo "    ${APO_SETUP}"
        echo "    ${BOUND_SETUP}"
        echo "============================================================"

    else

        echo
        echo "============================================================"
        echo "SYSTEM_SETUP_MODE = SKIP"
        echo "============================================================"
        echo "Existing base setup is incomplete or missing."
        echo "Building the missing system setup."
        echo "============================================================"

    fi
fi


if (( BASE_SETUP_READY == 0 )); then

# ============================================================
# 1. Prepare residue classification
# ============================================================

echo
echo "============================================================"
echo "1. Preparing residue type definitions"
echo "============================================================"

cd "${COMMON_DIR}"

RESIDUE_TYPES_FILE="${GMXLIB:-}/residuetypes.dat"
if [[ ! -f "${RESIDUE_TYPES_FILE}" ]]; then
    GMX_PATH="$(command -v "${GMX}" || true)"
    if [[ -n "${GMX_PATH}" ]]; then
        GMX_PREFIX="$(cd "$(dirname "${GMX_PATH}")/.." && pwd)"
        RESIDUE_TYPES_FILE="${GMX_PREFIX}/share/gromacs/top/residuetypes.dat"
    fi
fi

if [[ ! -f "${RESIDUE_TYPES_FILE}" ]]; then
    echo "ERROR: Could not locate GROMACS residuetypes.dat."
    echo "Set GMXLIB or ensure the configured GMX executable has a standard installation layout."
    exit 1
fi

cp "${RESIDUE_TYPES_FILE}" ./residuetypes.dat

if ! grep -qEi '^MG[[:space:]]+Ion' residuetypes.dat; then
    echo "MG    Ion" >> residuetypes.dat
fi

echo "Ion residue classifications:"
grep -Ei '^(ZN|MG)[[:space:]]' residuetypes.dat || true


# ============================================================
# 2. Extract protein/cofactors without ligand
# ============================================================

echo
echo "============================================================"
echo "2. Extracting protein/cofactor system"
echo "============================================================"

awk -v lig="${LIGAND_RESNAME}" '
($1=="ATOM") ||
($1=="HETATM" && substr($0,18,3)!=lig)
' "${COMPLEX}" > protein_no_ligand.pdb

if [[ ! -s protein_no_ligand.pdb ]]; then
    echo "ERROR: protein_no_ligand.pdb is empty."
    exit 1
fi

echo "Generated:"
echo "    ${COMMON_DIR}/protein_no_ligand.pdb"


# ============================================================
# 3. First pdb2gmx pass
# ============================================================

echo
echo "============================================================"
echo "3. First pdb2gmx pass"
echo "============================================================"

${GMX} pdb2gmx \
    -f protein_no_ligand.pdb \
    -o protein_prepared.pdb \
    -p temp.top \
    -ff "${FF}" \
    -water "${WATER}" \
    -ignh


# ============================================================
# 4. Define mutation
# ============================================================

echo
echo "============================================================"
echo "4. Defining mutation"
echo "============================================================"

cat > mutation.txt << EOF
${CHAIN} ${RESID} ${MUT}
EOF

cat mutation.txt


# ============================================================
# 5. Create hybrid residue
# ============================================================

echo
echo "============================================================"
echo "5. Running pmx mutate"
echo "============================================================"

pmx mutate \
    -f protein_prepared.pdb \
    -o hybrid.pdb \
    -ff "${FF}" \
    --script mutation.txt \
    --keep_resid


# ============================================================
# 6. Generate hybrid GROMACS topology
# ============================================================

echo
echo "============================================================"
echo "6. Generating hybrid GROMACS topology"
echo "============================================================"

echo
echo "Running second pdb2gmx..."

if ! ${GMX} pdb2gmx \
    -f hybrid.pdb \
    -o hybrid_gmx.pdb \
    -p topol.top \
    -ff "${FF}" \
    -water "${WATER}"
then

    STATUS=$?

    echo
    echo "============================================================"
    echo "ERROR: second pdb2gmx failed"
    echo "============================================================"
    echo "Exit status: ${STATUS}"
    echo "Working dir: ${PWD}"
    echo "Input:"
    echo "    ${PWD}/hybrid.pdb"

    echo
    echo "Files present:"
    ls -lh

    exit 1
fi

# ============================================================
# 7. Fill B-state parameters
# ============================================================

echo
echo "============================================================"
echo "7. Running pmx gentop"
echo "============================================================"

pmx gentop \
    -p topol.top \
    -o hybrid.top \
    -ff "${FF}"

echo "Hybrid topology generated:"
echo "    ${COMMON_DIR}/hybrid_gmx.pdb"
echo "    ${COMMON_DIR}/hybrid.top"


# ============================================================
# 8. Prepare APO system
# ============================================================

echo
echo "============================================================"
echo "8. Preparing APO system"
echo "============================================================"

cp "${COMMON_DIR}/hybrid_gmx.pdb" \
   "${APO_SETUP}/protein.pdb"

cp "${COMMON_DIR}/hybrid.top" \
   "${APO_SETUP}/topol.top"

find "${COMMON_DIR}" \
    -maxdepth 1 \
    -type f \
    -name "*.itp" \
    ! -iname "*temp*" \
    -exec cp {} "${APO_SETUP}/" \;

cd "${APO_SETUP}"


# ------------------------------------------------------------
# APO box
# ------------------------------------------------------------

echo
echo "Creating APO simulation box..."

${GMX} editconf \
    -f protein.pdb \
    -o boxed.gro \
    -bt dodecahedron \
    -d "${BOX_DISTANCE}"


# ------------------------------------------------------------
# APO solvation
# ------------------------------------------------------------

echo
echo "Solvating APO system..."

${GMX} solvate \
    -cp boxed.gro \
    -cs spc216.gro \
    -p topol.top \
    -o solvated.gro


# ------------------------------------------------------------
# APO ion TPR
# ------------------------------------------------------------

echo
echo "Generating APO ions.tpr..."

${GMX} grompp \
    -f "${IONS_MDP}" \
    -c solvated.gro \
    -p topol.top \
    -o ions.tpr \
    -maxwarn 1


# ------------------------------------------------------------
# APO neutralization
# ------------------------------------------------------------

echo
echo "Adding ions to APO system..."

echo "SOL" | ${GMX} genion \
    -s ions.tpr \
    -o system.gro \
    -p topol.top \
    -neutral \
    -conc "${ION_CONC}"

echo "APO system generated:"
echo "    ${APO_SETUP}/system.gro"


# ============================================================
# 9. Prepare BOUND system
# ============================================================

echo
echo "============================================================"
echo "9. Preparing BOUND system"
echo "============================================================"

cd "${BOUND_SETUP}"

cp "${COMMON_DIR}/hybrid_gmx.pdb" protein.pdb
cp "${COMMON_DIR}/hybrid.top" topol.top

# LIGAND_ITP is expected to be the full path to the ACPYPE-generated ITP.
LIGAND_ITP_PATH="${LIGAND_ITP}"

if [[ ! -f "${LIGAND_ITP_PATH}" ]]; then
    echo "ERROR: Ligand ITP file not found:"
    echo "    ${LIGAND_ITP_PATH}"
    exit 1
fi

echo "Ligand ITP found:"
echo "    ${LIGAND_ITP_PATH}"

cp "${LIGAND_ITP_PATH}" ligand_raw.itp


# ------------------------------------------------------------
# Split ACPYPE ligand ITP
#
# ligand_atomtypes.itp : contains only [ atomtypes ]
# ligand.itp           : contains [ moleculetype ], [ atoms ],
#                        [ bonds ], [ pairs ], [ angles ],
#                        [ dihedrals ], etc.
#
# GROMACS requires [ atomtypes ] before any [ moleculetype ].
# ------------------------------------------------------------

python - <<'PY'
from pathlib import Path

src = Path("ligand_raw.itp")
lines = src.read_text().splitlines()

atomtypes = []
molecule = []

section = None
found_atomtypes = False
found_moleculetype = False

for line in lines:
    stripped = line.strip()

    if stripped.startswith("[") and stripped.endswith("]"):
        section = stripped.strip("[] ").lower()

        if section == "atomtypes":
            found_atomtypes = True

        if section == "moleculetype":
            found_moleculetype = True

    if section == "atomtypes":
        atomtypes.append(line)
    else:
        molecule.append(line)

if not found_atomtypes:
    raise SystemExit(
        "ERROR: No [ atomtypes ] section found in ligand_raw.itp."
    )

if not found_moleculetype:
    raise SystemExit(
        "ERROR: No [ moleculetype ] section found in ligand_raw.itp."
    )

Path("ligand_atomtypes.itp").write_text(
    "\n".join(atomtypes).strip() + "\n"
)

Path("ligand.itp").write_text(
    "\n".join(molecule).strip() + "\n"
)

print("Ligand topology split successfully:")
print("    ligand_atomtypes.itp")
print("    ligand.itp")
PY


# Copy hybrid protein ITP files required by hybrid.top
find "${COMMON_DIR}" \
    -maxdepth 1 \
    -type f \
    -name "*.itp" \
    ! -iname "*temp*" \
    -exec cp {} "${BOUND_SETUP}/" \;


# ============================================================
# 10. Extract ligand coordinates from original complex
# ============================================================

echo
echo "============================================================"
echo "10. Extracting ligand coordinates"
echo "============================================================"

awk -v lig="${LIGAND_RESNAME}" '
($1=="HETATM" || $1=="ATOM") &&
substr($0,18,3)==lig
' "${COMPLEX}" > ligand.pdb

if [[ ! -s ligand.pdb ]]; then
    echo "ERROR: Ligand '${LIGAND_RESNAME}' was not found in:"
    echo "    ${COMPLEX}"
    exit 1
fi


# ============================================================
# 11. Ligand coordinate/topology mapping QC
# ============================================================

echo
echo "============================================================"
echo "11. Ligand coordinate/topology mapping QC"
echo "============================================================"

LIGAND_QC_DIR="${BOUND_SETUP}/ligand_qc"
mkdir -p "${LIGAND_QC_DIR}"

python - <<'PY'
from pathlib import Path
import sys
import math

pdb_file = Path("ligand.pdb")
itp_file = Path("ligand.itp")
qc_dir = Path("ligand_qc")

mapping_file = qc_dir / "ligand_atom_mapping.tsv"


# ============================================================
# Parse PDB ligand atoms
# ============================================================

pdb_atoms = []

for line in pdb_file.read_text().splitlines():

    if not line.startswith(("ATOM", "HETATM")):
        continue

    atom_name = line[12:16].strip()
    resname = line[17:20].strip()

    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        print(f"ERROR: Could not parse coordinates from:")
        print(line)
        sys.exit(1)

    pdb_atoms.append(
        {
            "name": atom_name,
            "resname": resname,
            "x": x,
            "y": y,
            "z": z,
        }
    )

if not pdb_atoms:
    print("ERROR: No ligand atoms found in ligand.pdb")
    sys.exit(1)


# ============================================================
# Parse ITP [ atoms ]
# ============================================================

itp_atoms = []
in_atoms = False

for raw in itp_file.read_text().splitlines():

    line = raw.strip()

    if not line:
        continue

    if line.startswith(";"):
        continue

    if line.startswith("["):
        section = line.strip("[] ").lower()
        in_atoms = section == "atoms"
        continue

    if not in_atoms:
        continue

    fields = line.split()

    if len(fields) < 7:
        continue

    try:
        atom_nr = int(fields[0])
        charge = float(fields[6])
    except ValueError:
        continue

    itp_atoms.append(
        {
            "nr": atom_nr,
            "type": fields[1],
            "resnr": fields[2],
            "resname": fields[3],
            "name": fields[4],
            "charge": charge,
        }
    )

if not itp_atoms:
    print("ERROR: Could not read [ atoms ] from ligand.itp")
    sys.exit(1)


# ============================================================
# Atom-count QC
# ============================================================

print(f"PDB ligand atom count: {len(pdb_atoms)}")
print(f"ITP ligand atom count: {len(itp_atoms)}")

if len(pdb_atoms) != len(itp_atoms):

    print()
    print("ERROR: Ligand atom-count mismatch.")
    print(f"    PDB: {len(pdb_atoms)}")
    print(f"    ITP: {len(itp_atoms)}")

    sys.exit(1)

print("Atom-count QC: PASS")


# ============================================================
# Helper for element sanity check
# ============================================================

def element_from_atom_name(name):

    name = name.strip()

    while name and name[0].isdigit():
        name = name[1:]

    upper = name.upper()

    if upper.startswith("CL"):
        return "CL"

    if upper.startswith("BR"):
        return "BR"

    if not upper:
        return "?"

    return upper[0]


def element_from_itp_type(atom_type):
    """Map a GAFF/ACPYPE atom type to its element symbol."""
    atom_type = atom_type.strip().upper()
    if atom_type.startswith("CL"):
        return "CL"
    if atom_type.startswith("BR"):
        return "BR"
    return atom_type[:1] or "?"


# ============================================================
# Mapping QC
# ============================================================

name_mismatches = []
residue_mismatches = []
element_mismatches = []

with mapping_file.open("w") as f:

    f.write(
        "Index\t"
        "PDB_Atom\t"
        "ITP_Atom\t"
        "PDB_Residue\t"
        "ITP_Residue\t"
        "ITP_Type\t"
        "ITP_Charge\t"
        "X\tY\tZ\t"
        "Status\n"
    )

    print()
    print(
        "Index   PDB_atom   ITP_atom   "
        "PDB_res   ITP_res   Status"
    )

    print("-" * 70)

    for i, (pdb_atom, itp_atom) in enumerate(
        zip(pdb_atoms, itp_atoms),
        start=1
    ):

        status = []

        if pdb_atom["name"] != itp_atom["name"]:

            name_mismatches.append(
                (i, pdb_atom["name"], itp_atom["name"])
            )

            status.append("ATOM_NAME_MISMATCH")

        if pdb_atom["resname"] != itp_atom["resname"]:

            residue_mismatches.append(
                (
                    i,
                    pdb_atom["resname"],
                    itp_atom["resname"]
                )
            )

            status.append("RESNAME_MISMATCH")

        pdb_element = element_from_atom_name(
            pdb_atom["name"]
        )

        itp_element = element_from_itp_type(itp_atom["type"])

        if pdb_element != itp_element:

            element_mismatches.append(
                (
                    i,
                    pdb_atom["name"],
                    itp_atom["name"]
                )
            )

            status.append("ELEMENT_MISMATCH")

        if not status:
            status = ["OK"]

        status_string = ",".join(status)

        print(
            f"{i:5d}   "
            f"{pdb_atom['name']:<9s} "
            f"{itp_atom['name']:<9s} "
            f"{pdb_atom['resname']:<8s} "
            f"{itp_atom['resname']:<8s} "
            f"{status_string}"
        )

        f.write(
            f"{i}\t"
            f"{pdb_atom['name']}\t"
            f"{itp_atom['name']}\t"
            f"{pdb_atom['resname']}\t"
            f"{itp_atom['resname']}\t"
            f"{itp_atom['type']}\t"
            f"{itp_atom['charge']:.6f}\t"
            f"{pdb_atom['x']:.3f}\t"
            f"{pdb_atom['y']:.3f}\t"
            f"{pdb_atom['z']:.3f}\t"
            f"{status_string}\n"
        )


# ============================================================
# Hard-fail on mapping problems
# ============================================================

if name_mismatches:

    print()
    print("WARNING: ACPYPE renamed ligand atoms:")

    for i, pdb_name, itp_name in name_mismatches:
        print(
            f"    atom {i}: "
            f"PDB={pdb_name} "
            f"ITP={itp_name}"
        )

if residue_mismatches:

    print()
    print("ERROR: Residue-name mismatch detected:")

    for i, pdb_res, itp_res in residue_mismatches:
        print(
            f"    atom {i}: "
            f"PDB={pdb_res} "
            f"ITP={itp_res}"
        )

if element_mismatches:

    print()
    print("ERROR: Element mismatch detected:")

    for i, pdb_name, itp_name in element_mismatches:
        print(
            f"    atom {i}: "
            f"PDB={pdb_name} "
            f"ITP={itp_name}"
        )

if residue_mismatches or element_mismatches:
    print()
    print("Ligand mapping QC: FAILED")
    print(f"Mapping report: {mapping_file}")
    sys.exit(1)

print("Ligand mapping QC: PASS")

# ACPYPE may rename atoms. Rewrite the ligand coordinate records with the
# ITP names so GROMACS sees identical atom names in solvated.gro and topol.top.
pdb_lines = Path("ligand.pdb").read_text().splitlines()
atom_index = 0
normalized_lines = []
for line in pdb_lines:
    if line.startswith(("ATOM", "HETATM")):
        if atom_index >= len(itp_atoms):
            raise SystemExit("ERROR: Ligand atom normalization exceeded ITP atom count.")
        replacement = f"{itp_atoms[atom_index]['name'][:4]:>4}"
        line = line[:12] + replacement + line[16:]
        atom_index += 1
    normalized_lines.append(line)

if atom_index != len(itp_atoms):
    raise SystemExit("ERROR: Ligand atom normalization count mismatch.")

Path("ligand.pdb").write_text("\n".join(normalized_lines) + "\n")
print("Normalized ligand PDB atom names to match ligand.itp.")


# ============================================================
# Charge QC
# ============================================================

total_charge = sum(
    atom["charge"]
    for atom in itp_atoms
)

print()
print(f"ITP total ligand charge: {total_charge:.6f}")


# ============================================================
# Final QC result
# ============================================================

print()
print("Atom-name QC:       INFO (ACPYPE names may be renumbered)")
print("Atom-order QC:      PASS")
print("Residue-name QC:    PASS")
print("Element mapping QC: PASS")
print("Ligand mapping QC:  PASS")

print()
print(
    "Detailed ligand mapping saved to:"
)
print(
    f"    {mapping_file}"
)
PY


# ============================================================
# 12. Combine protein + ligand coordinates
# ============================================================

echo
echo "============================================================"
echo "12. Combining protein + ligand coordinates"
echo "============================================================"

python - <<'PY'
from pathlib import Path

protein = Path("protein.pdb").read_text().splitlines()
ligand = Path("ligand.pdb").read_text().splitlines()

with open("complex_hybrid.pdb", "w") as f:

    for line in protein:
        if not line.startswith(("END", "ENDMDL")):
            f.write(line + "\n")

    for line in ligand:
        if line.startswith(("ATOM", "HETATM")):
            f.write(line + "\n")

    f.write("END\n")
PY

echo "Combined coordinate file generated:"
echo "    ${BOUND_SETUP}/complex_hybrid.pdb"


# ============================================================
# 13. Insert ligand topology includes in correct GROMACS order
# ============================================================

echo
echo "============================================================"
echo "13. Adding ligand topology includes"
echo "============================================================"

python - <<'PY'
from pathlib import Path

topfile = Path("topol.top")
lines = topfile.read_text().splitlines()

atomtypes_include = '#include "ligand_atomtypes.itp"'
ligand_include = '#include "ligand.itp"'


# ------------------------------------------------------------
# A. Insert ligand [ atomtypes ] immediately after forcefield.itp
#
# This MUST occur before the first [ moleculetype ] definition.
# ------------------------------------------------------------

if not any(atomtypes_include in line for line in lines):

    new_lines = []
    inserted = False

    for line in lines:
        new_lines.append(line)

        if (
            not inserted
            and line.strip().startswith("#include")
            and "forcefield.itp" in line
        ):
            new_lines.append("")
            new_lines.append("; GAFF2 ligand atom types")
            new_lines.append(atomtypes_include)
            inserted = True

    if not inserted:
        raise SystemExit(
            "ERROR: Could not locate forcefield.itp include in topol.top."
        )

    lines = new_lines


# ------------------------------------------------------------
# B. Insert ligand molecular topology before [ system ]
# ------------------------------------------------------------

if not any(ligand_include in line for line in lines):

    new_lines = []
    inserted = False

    for line in lines:

        if (
            not inserted
            and line.strip().lower() == "[ system ]"
        ):
            new_lines.append("")
            new_lines.append("; Ligand molecular topology")
            new_lines.append(ligand_include)
            new_lines.append("")
            inserted = True

        new_lines.append(line)

    if not inserted:
        raise SystemExit(
            "ERROR: Could not locate [ system ] in topol.top."
        )

    lines = new_lines


topfile.write_text("\n".join(lines) + "\n")

print("Ligand topology includes inserted.")
PY


# ------------------------------------------------------------
# Verify include ordering
# ------------------------------------------------------------

echo
echo "Relevant topology ordering:"
grep -n -E '#include|\[ system \]|\[ molecules \]' topol.top || true

python - <<'PY'
from pathlib import Path

lines = Path("topol.top").read_text().splitlines()

def idx(predicate):
    for i, line in enumerate(lines):
        if predicate(line):
            return i
    return None

ff_i = idx(
    lambda x:
        x.strip().startswith("#include")
        and "forcefield.itp" in x
)

atomtypes_i = idx(
    lambda x:
        '#include "ligand_atomtypes.itp"' in x
)

ligand_i = idx(
    lambda x:
        '#include "ligand.itp"' in x
)

system_i = idx(
    lambda x:
        x.strip().lower() == "[ system ]"
)

if None in (ff_i, atomtypes_i, ligand_i, system_i):
    raise SystemExit(
        "ERROR: Could not verify topology include order."
    )

if not (ff_i < atomtypes_i < ligand_i < system_i):
    raise SystemExit(
        "ERROR: Invalid topology include order."
    )

print("Topology include-order QC: PASS")
PY


# ============================================================
# 14. Detect ligand moleculetype
# ============================================================

echo
echo "============================================================"
echo "14. Checking ligand moleculetype"
echo "============================================================"

LIGAND_MOLTYPE=$(
awk '
/^[[:space:]]*\[[[:space:]]*moleculetype[[:space:]]*\]/ {
    flag=1
    next
}

flag &&
$0 !~ /^[[:space:]]*;/ &&
NF >= 2 {
    print $1
    exit
}
' ligand.itp
)

if [[ -z "${LIGAND_MOLTYPE}" ]]; then
    echo "ERROR: Could not determine ligand moleculetype."
    exit 1
fi

echo "Ligand moleculetype:"
echo "    ${LIGAND_MOLTYPE}"


# ============================================================
# 15. Add ligand to [ molecules ] if absent
# ============================================================

echo
echo "============================================================"
echo "15. Checking [ molecules ] section"
echo "============================================================"

python - "${LIGAND_MOLTYPE}" <<'PY'
from pathlib import Path
import sys

ligand_moltype = sys.argv[1]

topfile = Path("topol.top")
lines = topfile.read_text().splitlines()

in_molecules = False
already_present = False

for line in lines:

    stripped = line.strip()

    if stripped.lower() == "[ molecules ]":
        in_molecules = True
        continue

    if in_molecules and stripped.startswith("["):
        break

    if (
        in_molecules
        and stripped
        and not stripped.startswith(";")
    ):

        fields = stripped.split()

        if fields and fields[0] == ligand_moltype:
            already_present = True
            break


if already_present:

    print(
        f"Ligand '{ligand_moltype}' already present "
        "in [ molecules ]."
    )

else:

    with topfile.open("a") as f:
        f.write(
            f"\n{ligand_moltype:<20s} 1\n"
        )

    print(
        f"Added ligand '{ligand_moltype}' "
        "to [ molecules ]."
    )
PY


echo
echo "Current [ molecules ] section:"

awk '
/^[[:space:]]*\[[[:space:]]*molecules[[:space:]]*\]/ {
    flag=1
}

flag {
    print
}
' topol.top


# ============================================================
# 16. BOUND box
# ============================================================

echo
echo "============================================================"
echo "16. Creating BOUND simulation box"
echo "============================================================"

${GMX} editconf \
    -f complex_hybrid.pdb \
    -o boxed.gro \
    -bt dodecahedron \
    -d "${BOX_DISTANCE}"


# ============================================================
# 17. BOUND solvation
# ============================================================

echo
echo "============================================================"
echo "17. Solvating BOUND system"
echo "============================================================"

${GMX} solvate \
    -cp boxed.gro \
    -cs spc216.gro \
    -p topol.top \
    -o solvated.gro


# ============================================================
# 18. BOUND ion TPR
# ============================================================

echo
echo "============================================================"
echo "18. Generating BOUND ions.tpr"
echo "============================================================"

${GMX} grompp \
    -f "${IONS_MDP}" \
    -c solvated.gro \
    -p topol.top \
    -o ions.tpr \
    -maxwarn 1


# ============================================================
# 19. BOUND neutralization
# ============================================================

echo
echo "============================================================"
echo "19. Adding ions to BOUND system"
echo "============================================================"

echo "SOL" | ${GMX} genion \
    -s ions.tpr \
    -o system.gro \
    -p topol.top \
    -neutral \
    -conc "${ION_CONC}"


fi  # BASE_SETUP_READY == 0


# ============================================================
# Base-system preparation summary
# ============================================================

echo
echo "============================================================"
echo "System preparation completed successfully"
echo "============================================================"
echo "Date:"
echo "    $(date)"
echo
echo "APO system:"
echo "    ${APO_SETUP}/system.gro"
echo
echo "BOUND system:"
echo "    ${BOUND_SETUP}/system.gro"
echo
echo "Ligand QC report:"
echo "    ${BOUND_SETUP}/ligand_qc/ligand_atom_mapping.tsv"
echo
echo "Full log:"
echo "    ${WORK_DIR}/${JOB_NAME}/logs/${LOG_FILE}"
echo "============================================================"


# ============================================================
# APO/BOUND preparation above
# ============================================================

# ... existing preparation code ...


# ============================================================
# Create replicate setup directories
# ============================================================

echo
echo "============================================================"
echo "Creating replicate setup directories"
echo "============================================================"

for LEG in apo bound
do

    BASE_SETUP="${WORK_DIR}/${JOB_NAME}/${LEG}/setup"

    # --------------------------------------------------------
    # QC base setup
    # --------------------------------------------------------

    if [[ ! -s "${BASE_SETUP}/system.gro" ]]; then
        echo "ERROR: Base system missing:"
        echo "    ${BASE_SETUP}/system.gro"
        exit 1
    fi

    if [[ ! -s "${BASE_SETUP}/topol.top" ]]; then
        echo "ERROR: Base topology missing:"
        echo "    ${BASE_SETUP}/topol.top"
        exit 1
    fi


    # --------------------------------------------------------
    # Create replicate setups
    # --------------------------------------------------------

    for REP in $(seq "${START_REP}" "${END_REP}")
    do

        REP_SETUP="${WORK_DIR}/${JOB_NAME}/${LEG}/rep_${REP}/setup"

        echo
        echo "Preparing:"
        echo "    ${LEG} / rep_${REP}"

        # ----------------------------------------------------
        # Reuse a complete replicate setup if it already exists.
        # This makes the preparation script safe to rerun.
        # ----------------------------------------------------

        if [[ "${SYSTEM_SETUP_MODE}" == "SKIP" && \
              -s "${REP_SETUP}/system.gro" && \
              -s "${REP_SETUP}/topol.top" ]]; then

            echo "REUSE:"
            echo "    ${REP_SETUP}"
            echo "    Existing system.gro/topol.top are complete."
            continue
        fi

        if [[ "${SYSTEM_SETUP_MODE}" == "OVERWRITE" && -d "${REP_SETUP}" ]]; then
            echo "OVERWRITE:"
            echo "    ${REP_SETUP}"
            rm -rf "${REP_SETUP}"
        fi

        mkdir -p "${REP_SETUP}"

        # ----------------------------------------------------
        # Copy the complete base setup.
        #
        # Only rep_X/setup is replaced. rep_X/fep is preserved.
        # ----------------------------------------------------

        cp -a "${BASE_SETUP}/." "${REP_SETUP}/"


        # ----------------------------------------------------
        # QC
        # ----------------------------------------------------

        if [[ ! -s "${REP_SETUP}/system.gro" ]]; then
            echo "ERROR: system.gro missing:"
            echo "    ${REP_SETUP}/system.gro"
            exit 1
        fi

        if [[ ! -s "${REP_SETUP}/topol.top" ]]; then
            echo "ERROR: topol.top missing:"
            echo "    ${REP_SETUP}/topol.top"
            exit 1
        fi

        if grep -q "/AlchemForge/.alchemforge_env/" "${REP_SETUP}/topol.top"; then
            echo "ERROR: Stale old-environment path detected in topology:"
            echo "    ${REP_SETUP}/topol.top"
            grep -n "/AlchemForge/.alchemforge_env/" "${REP_SETUP}/topol.top" || true
            exit 1
        fi

        echo "PASS:"
        echo "    ${LEG}/rep_${REP}/setup"

    done

done


# ============================================================
# Final QC
# ============================================================

echo
echo "============================================================"
echo "Replicate setup QC"
echo "============================================================"

for LEG in apo bound
do
    for REP in $(seq "${START_REP}" "${END_REP}")
    do

        SETUP="${WORK_DIR}/${JOB_NAME}/${LEG}/rep_${REP}/setup"

        printf "%-8s rep_%d : " "${LEG}" "${REP}"

        if [[ -s "${SETUP}/system.gro" && \
              -s "${SETUP}/topol.top" ]]; then

            echo "PASS"

        else

            echo "FAIL"
            exit 1

        fi

    done
done


echo
echo "============================================================"
echo "System preparation completed successfully"
echo "============================================================"

echo
echo "Configuration used:"
echo "    ${CONFIG_FILE}"
echo "Replicate range prepared/reused:"
echo "    rep_${START_REP} through rep_${END_REP}"
