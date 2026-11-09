#!/usr/bin/env bash
# dev_install.sh — link ChemFlow (+ AlchemForge) and PMX into an EXISTING conda
# env in editable (development) mode. It does NOT create or solve the env; build
# the compute stack first with either
#     mamba env create -f env/environment-alchemforge.yaml      # workstation
#     conda env create -n alchemforge -f env/environment-alchemforge-hpc.yaml  # HPC GPU
# or `chemflow fe-env install` (the desktop Environments page), then run this to
# (re)link the editable sources after a `git pull`.
#
# Usage:
#   scripts/dev_install.sh [ENV] [REPO]
#     ENV   conda env name          (default: $CHEMFLOW_FE_ENV or "alchemforge")
#     REPO  repo root to link       (default: this repo, i.e. the script's parent)
#
# Mirrors `chemflow fe-env install`'s link step:
#   <tool> run -n ENV pip install -e REPO --no-deps --no-build-isolation
#   <tool> run -n ENV pip install REPO/pmx --no-deps --no-build-isolation
# PMX is skipped if it already imports in the env.

set -euo pipefail

ENV_NAME="${1:-${CHEMFLOW_FE_ENV:-alchemforge}}"

# Repo root = parent of this script's directory, unless overridden.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${2:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Prefer mamba, fall back to conda.
if command -v mamba >/dev/null 2>&1; then
  TOOL="mamba"
elif command -v conda >/dev/null 2>&1; then
  TOOL="conda"
else
  echo "error: neither mamba nor conda found on PATH. On HPC, load conda first" >&2
  echo "       (e.g. 'module load miniconda' or 'source <conda-prefix>/etc/profile.d/conda.sh')." >&2
  exit 1
fi

# The env must already exist.
if ! "$TOOL" run -n "$ENV_NAME" python -c "" >/dev/null 2>&1; then
  echo "error: conda env '$ENV_NAME' not found (or has no python)." >&2
  echo "       Create it first, e.g.:" >&2
  echo "         mamba env create -f ${REPO}/environment-alchemforge.yaml" >&2
  echo "       or: chemflow fe-env install --env ${ENV_NAME}" >&2
  exit 1
fi

echo "Linking ChemFlow (editable) into '$ENV_NAME' from $REPO"
echo "\$ $TOOL run -n $ENV_NAME pip install -e $REPO --no-deps --no-build-isolation"
"$TOOL" run -n "$ENV_NAME" pip install -e "$REPO" --no-deps --no-build-isolation

# PMX: only needed for the GROMACS hybrid-topology step. Skip if already importable.
if "$TOOL" run -n "$ENV_NAME" python -c "import pmx" >/dev/null 2>&1; then
  echo "PMX already importable in '$ENV_NAME' — skipping."
elif [ -f "${REPO}/pmx/setup.py" ] || [ -f "${REPO}/pmx/pyproject.toml" ]; then
  echo "Linking bundled PMX from ${REPO}/pmx"
  echo "\$ $TOOL run -n $ENV_NAME pip install ${REPO}/pmx --no-deps --no-build-isolation"
  "$TOOL" run -n "$ENV_NAME" pip install "${REPO}/pmx" --no-deps --no-build-isolation
else
  echo "note: no bundled pmx/ checkout at ${REPO}/pmx and PMX not importable." >&2
  echo "      Install it with your conda tool if you need GROMACS hybrid topologies:" >&2
  echo "        $TOOL install -n $ENV_NAME -c conda-forge pmx" >&2
fi

echo
echo "Done. '$ENV_NAME' now links the editable ChemFlow/AlchemForge sources."
echo "The desktop auto-detects this env; or set CHEMFLOW_FE_PYTHON to its python."
