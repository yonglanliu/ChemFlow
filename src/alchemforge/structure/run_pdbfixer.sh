#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:?Usage: $0 /path/to/pdbfixer_manifest.json}"
python -m alchemforge.structure.pdbfix "${MANIFEST}"
