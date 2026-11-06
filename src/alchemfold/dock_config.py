"""Shared configuration for AlchemFold docking (multi-engine co-folding).

One config drives any supported co-folding engine (Boltz-2, Chai-1). The engine
is selected with the ``engine`` field; engine-specific input files and commands
are produced by :mod:`alchemfold.engines`. Optionally a binding-site **pocket**
constraint steers the ligand into a known site.

Import-light (PyYAML only): the CLI / desktop / tests work without any model
installed; only ``run`` needs the chosen engine's package.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alchemfold.boltz_config import _deep_merge, _load_yaml, _resolve, _to_namespace

SUPPORTED_ENGINES = {"boltz", "chai", "diffdock"}
SUPPORTED_OUTPUT_FORMATS = {"pdb", "mmcif"}
SUPPORTED_ACCELERATORS = {"gpu", "cpu", "tpu"}

DEFAULTS: dict[str, Any] = {
    "engine": "boltz",            # boltz | chai
    "workdir": ".",
    "out_dir": "docking_output",
    "seed": 1,
    "use_msa_server": True,
    "accelerator": "gpu",         # gpu | cpu | tpu
    "devices": 1,
    "diffusion_samples": 1,       # boltz --diffusion_samples / chai --num-diffn-samples
    "recycling_steps": 3,         # boltz --recycling_steps / chai --num-trunk-recycles
    "sampling_steps": 200,        # boltz only
    "output_format": "pdb",       # boltz pdb|mmcif (chai writes mmCIF)
    "override": False,
    # --- Boltz-2 specific ---
    "boltz_command": "boltz",
    "predict_affinity": False,
    "affinity_binder": None,
    # --- Chai-1 specific ---
    "chai_command": "chai-lab",
    # --- DiffDock specific (diffusion docking into a receptor STRUCTURE) ---
    "diffdock_command": "python -m inference",  # run from the DiffDock repo
    "diffdock_inference_steps": 20,
    "diffdock_model_dir": None,
    "diffdock_config": None,
    # Receptor structure for DiffDock (boltz/chai are sequence-based and ignore
    # this; DiffDock docks into this PDB, or folds the sequence with ESMFold if
    # no PDB is given).
    "receptor_pdb": None,
    # --- inputs ---
    "proteins": [],               # [{id, sequence, [msa]}]
    "ligands": [],                # [{id, smiles|ccd}] — components of ONE complex
    # Screen a ligand library from a file instead of inline ligands: one docking
    # run per entry, output under <workdir>/ligands/<id>/. CSV (needs a 'smiles'
    # column, optional 'name'/'id'), a .smi file ("SMILES [name]" per line), or a
    # .cxsmiles file (CXSMILES + id, e.g. Enamine REAL). Plain or .bz2/.gz.
    "ligand_file": None,
    "ligand_file_format": "auto",  # auto | csv | smiles | cxsmiles
    # --- optional binding-site pocket constraint ---
    # pocket:
    #   binder: L
    #   max_distance: 6.0
    #   contacts: [{chain: A, residue: 123}, {chain: A, residue: 145}]
    "pocket": None,
}


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a docking YAML, merged over the defaults; resolve paths."""
    config_path = Path(path).expanduser().resolve()
    conf = _to_namespace(_deep_merge(DEFAULTS, _load_yaml(config_path)))

    base_dir = config_path.parent
    workdir = getattr(conf, "workdir", ".")
    conf.workdir = str(base_dir) if (not workdir or str(workdir) == ".") else _resolve(base_dir, workdir)

    for key in ("receptor_pdb", "diffdock_model_dir", "diffdock_config", "ligand_file"):
        value = getattr(conf, key, None)
        if value:
            setattr(conf, key, _resolve(base_dir, value))

    for protein in getattr(conf, "proteins", []) or []:
        msa = getattr(protein, "msa", None)
        if msa and str(msa) not in {"empty", "server"}:
            protein.msa = _resolve(base_dir, msa)
    return conf


def _chain_ids(conf) -> set[str]:
    ids = {str(p.id).strip() for p in (getattr(conf, "proteins", []) or [])}
    ids |= {str(l.id).strip() for l in (getattr(conf, "ligands", []) or [])}
    return ids


def _sanitize_id(name: str, fallback: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name).strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _read_library_text(path: Path) -> str:
    """Read a (optionally bz2/gz-compressed) ligand file as text."""
    name = path.name.lower()
    if name.endswith(".bz2"):
        import bz2

        return bz2.open(path, "rt", encoding="utf-8").read()
    if name.endswith(".gz"):
        import gzip

        return gzip.open(path, "rt", encoding="utf-8").read()
    return path.read_text(encoding="utf-8")


def _inner_suffix(path: Path) -> str:
    """Suffix ignoring a trailing .bz2/.gz, lowercased (e.g. '.cxsmiles')."""
    name = path.name.lower()
    for comp in (".bz2", ".gz"):
        if name.endswith(comp):
            name = name[: -len(comp)]
            break
    return Path(name).suffix


def _split_smiles_line(line: str, cxsmiles: bool) -> tuple[str, str]:
    """Return (smiles, id) for one SMILES/CXSMILES line.

    Tab-separated files use the first field as SMILES and the second as the id.
    Otherwise the last whitespace token is the id, EXCEPT a trailing CXSMILES
    extension block ``|...|``, which stays part of the SMILES.
    """
    if "\t" in line:
        fields = line.split("\t")
        return fields[0].strip(), (fields[1].strip() if len(fields) > 1 else "")
    tokens = line.split()
    if len(tokens) == 1:
        return tokens[0], ""
    if cxsmiles:
        if tokens[-1].startswith("|"):  # trailing CX extension block, no id
            return line.strip(), ""
        return " ".join(tokens[:-1]), tokens[-1]
    return tokens[0], tokens[1]


def load_ligands_from_file(path: str | Path, fmt: str = "auto") -> list[dict]:
    """Parse a ligand library file into ``[{id, smiles}, ...]`` (deduped ids).

    Supports plain or ``.bz2`` / ``.gz`` compressed files:
      * CSV: a ``smiles``/``smi`` column (plus optional ``name``/``id``/``title``);
        with no header, column 0 = SMILES, column 1 = name.
      * SMILES (``.smi`` / ``.txt``): one ``SMILES [name]`` per line.
      * CXSMILES (``.cxsmiles``): ``SMILES[ |cx-extensions|] [id]``, tab- or
        space-separated (e.g. Enamine REAL). ``#`` comment lines are ignored.
    """
    path = Path(path)
    text = _read_library_text(path)
    fmt = str(fmt or "auto").lower()
    inner = _inner_suffix(path)
    if fmt == "auto":
        if inner == ".csv":
            fmt = "csv"
        elif inner == ".cxsmiles":
            fmt = "cxsmiles"
        else:
            fmt = "smiles"

    parsed: list[dict] = []
    if fmt == "csv":
        import csv as _csv
        import io

        rows = [r for r in _csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
        if not rows:
            return []
        header = [c.strip().lower() for c in rows[0]]
        has_header = any(h in {"smiles", "smi"} for h in header)
        smi_i, name_i = 0, (1 if len(rows[0]) > 1 else None)
        data = rows
        if has_header:
            for i, h in enumerate(header):
                if h in {"smiles", "smi"}:
                    smi_i = i
                if h in {"name", "id", "title", "ligand", "molecule"}:
                    name_i = i
            data = rows[1:]
        for n, row in enumerate(data, start=1):
            if smi_i >= len(row) or not row[smi_i].strip():
                continue
            raw_name = row[name_i].strip() if (name_i is not None and name_i < len(row)) else ""
            parsed.append({"id": _sanitize_id(raw_name, f"lig{n}"), "smiles": row[smi_i].strip()})
    else:  # smiles / .smi / cxsmiles
        cxsmiles = fmt == "cxsmiles"
        n = 0
        for raw in text.splitlines():
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            n += 1
            smiles, raw_name = _split_smiles_line(line.strip(), cxsmiles)
            if not smiles:
                continue
            parsed.append({"id": _sanitize_id(raw_name, f"lig{n}"), "smiles": smiles})

    # Deduplicate ids.
    seen: set[str] = set()
    out: list[dict] = []
    for entry in parsed:
        base = entry["id"]
        cid, k = base, 2
        while cid in seen:
            cid = f"{base}_{k}"
            k += 1
        seen.add(cid)
        out.append({"id": cid, "smiles": entry["smiles"]})
    return out


def validate_config(conf: SimpleNamespace) -> None:
    """Validate a loaded docking configuration (no model package needed)."""
    engine = str(getattr(conf, "engine", "boltz"))
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(
            f"engine must be one of {sorted(SUPPORTED_ENGINES)} (got {engine!r})"
        )

    proteins = getattr(conf, "proteins", []) or []
    ligands = getattr(conf, "ligands", []) or []
    if not proteins:
        raise ValueError("docking requires at least one protein (receptor) chain")

    # DiffDock can dock into a receptor PDB without a sequence (or fold the
    # sequence with ESMFold); sequence-based engines always need a sequence.
    sequence_required = not (
        str(getattr(conf, "engine", "boltz")) == "diffdock"
        and getattr(conf, "receptor_pdb", None)
    )

    ids: set[str] = set()
    for protein in proteins:
        pid = str(getattr(protein, "id", "")).strip()
        if not pid:
            raise ValueError("each protein needs an 'id'")
        if sequence_required and not str(getattr(protein, "sequence", "")).strip():
            raise ValueError(f"protein '{pid}' needs a 'sequence'")
        if pid in ids:
            raise ValueError(f"duplicate chain id: {pid}")
        ids.add(pid)
    for ligand in ligands:
        lid = str(getattr(ligand, "id", "")).strip()
        if not lid:
            raise ValueError("each ligand needs an 'id'")
        if lid in ids:
            raise ValueError(f"duplicate chain id: {lid}")
        ids.add(lid)
        if not getattr(ligand, "smiles", None) and not getattr(ligand, "ccd", None):
            raise ValueError(f"ligand '{lid}' needs a 'smiles' or a 'ccd' code")

    # A ligand library file (screened one run per entry) counts as "has ligand".
    ligand_file = getattr(conf, "ligand_file", None)
    has_library = False
    if ligand_file:
        lib_path = Path(ligand_file)
        if not lib_path.is_file():
            raise ValueError(f"ligand_file not found: {ligand_file}")
        library = load_ligands_from_file(ligand_file, getattr(conf, "ligand_file_format", "auto"))
        if not library:
            raise ValueError(f"ligand_file produced no ligands: {ligand_file}")
        if any(not entry.get("smiles") for entry in library):
            raise ValueError("every ligand_file entry needs a SMILES")
        has_library = True

    if not ligands and not has_library and len(proteins) < 2:
        raise ValueError(
            "docking/co-folding needs a ligand (inline or ligand_file) or >= 2 "
            "chains; a single apo chain is plain structure prediction"
        )

    if str(getattr(conf, "output_format", "pdb")) not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}")
    if str(getattr(conf, "accelerator", "gpu")) not in SUPPORTED_ACCELERATORS:
        raise ValueError(f"accelerator must be one of {sorted(SUPPORTED_ACCELERATORS)}")
    for key in ("diffusion_samples", "recycling_steps", "sampling_steps", "devices"):
        if int(getattr(conf, key, 0)) < 1:
            raise ValueError(f"{key} must be >= 1")

    _validate_pocket(conf)

    # Engine-specific checks (affinity support, etc.).
    from alchemfold.engines import get_engine

    get_engine(engine).validate(conf)


def _validate_pocket(conf) -> None:
    pocket = getattr(conf, "pocket", None)
    if not pocket:
        return
    binder = str(getattr(pocket, "binder", "")).strip()
    # When screening a ligand_file, the binder is assigned per run, so skip the
    # membership check (the runner sets pocket.binder to each library ligand).
    if not getattr(conf, "ligand_file", None):
        ids = _chain_ids(conf)
        if not binder or binder not in ids:
            raise ValueError(
                f"pocket.binder must be one of the chain ids {sorted(ids)} (got {binder!r})"
            )
    contacts = getattr(pocket, "contacts", None) or []
    if not contacts:
        raise ValueError("pocket needs at least one contact {chain, residue}")
    protein_ids = {str(p.id).strip() for p in (getattr(conf, "proteins", []) or [])}
    for contact in contacts:
        chain = str(getattr(contact, "chain", "")).strip()
        if chain not in protein_ids:
            raise ValueError(f"pocket contact chain '{chain}' is not a protein chain id")
        residue = getattr(contact, "residue", None)
        if residue is None or int(residue) < 1:
            raise ValueError("each pocket contact needs a positive integer 'residue'")


def stage(config_path: str | Path, task_directory: str | Path) -> tuple[Path, Path]:
    """Write a runner script that runs docking for ``config_path``."""
    config_path = Path(config_path).expanduser().resolve()
    task = Path(task_directory).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    runner = task / "run_docking.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'trap \'echo "[FAILED] line $LINENO: $BASH_COMMAND (exit $?)" >&2\' ERR\n'
        'CONFIG="${1:?Usage: $0 docking.yaml}"\n'
        f"exec {shlex.quote(sys.executable)} -m alchemfold.dock_run "
        '--config "$CONFIG"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, config_path


def default_yaml_text(
    *,
    engine: str = "boltz",
    protein_id: str = "A",
    ligand_id: str = "L",
    sequence: str = "PUT_YOUR_PROTEIN_SEQUENCE_HERE",
    smiles: str = "CCO",
    with_pocket: bool = False,
) -> str:
    """Return a commented example docking config for the chosen engine."""
    affinity_block = (
        "predict_affinity: false         # Boltz-2 only; binding-affinity prediction\n"
        '# affinity_binder: "L"          # ligand id to score when predict_affinity: true\n'
    )
    pocket_block = (
        "pocket:                         # optional binding-site constraint\n"
        f'  binder: "{ligand_id}"                  # which ligand binds the pocket\n'
        "  max_distance: 6.0             # Angstrom (Boltz pocket constraint)\n"
        "  contacts:                     # residues lining the pocket\n"
        f"    - {{chain: {protein_id}, residue: 123}}\n"
        f"    - {{chain: {protein_id}, residue: 145}}\n"
        if with_pocket
        else (
            "# pocket:                       # optional binding-site constraint\n"
            f'#   binder: "{ligand_id}"\n'
            "#   max_distance: 6.0\n"
            "#   contacts:\n"
            f"#     - {{chain: {protein_id}, residue: 123}}\n"
        )
    )
    return (
        "# ===================================================================\n"
        "# AlchemFold docking (co-folding). Jointly predict a receptor+ligand\n"
        "# complex with the chosen model; optionally steer the ligand into a\n"
        "# known pocket. Output feeds OpenMM modeling -> FEP.\n"
        "#   engine: boltz -> Boltz-2 (needs `boltz`); supports affinity\n"
        "#   engine: chai  -> Chai-1  (needs `chai-lab`)\n"
        "#   engine: diffdock -> DiffDock (docks a ligand into a receptor PDB;\n"
        "#                       no pocket/affinity; needs the DiffDock repo)\n"
        "# ===================================================================\n"
        "\n"
        f'engine: "{engine}"              # boltz | chai\n'
        'out_dir: "docking_output"\n'
        'output_format: "pdb"            # pdb | mmcif (Chai writes mmCIF)\n'
        "use_msa_server: true            # compute MSAs via the model's server (needs internet)\n"
        "seed: 1\n"
        'accelerator: "gpu"              # gpu | cpu | tpu\n'
        "devices: 1\n"
        "diffusion_samples: 1            # predicted poses\n"
        "recycling_steps: 3\n"
        "sampling_steps: 200             # Boltz only\n"
        "override: false\n"
        f"{affinity_block}"
        'boltz_command: "boltz"          # used when engine: boltz\n'
        'chai_command: "chai-lab"        # used when engine: chai\n'
        "# --- DiffDock (engine: diffdock) docks into a receptor structure ---\n"
        '# receptor_pdb: "receptor.pdb"  # receptor structure (or omit to fold the sequence via ESMFold)\n'
        '# diffdock_command: "python -m inference"   # run from the DiffDock repo\n'
        "# diffdock_inference_steps: 20\n"
        '# diffdock_model_dir: "/path/to/DiffDock/workdir"\n'
        "\n"
        "proteins:\n"
        f'  - id: "{protein_id}"\n'
        f'    sequence: "{sequence}"\n'
        "\n"
        "ligands:\n"
        f'  - id: "{ligand_id}"\n'
        f'    smiles: "{smiles}"          # or use `ccd: "ATP"`\n'
        "\n"
        "# Screen a ligand LIBRARY instead of the single ligand above — one run per\n"
        "# entry, output under <workdir>/ligands/<id>/. CSV ('smiles' column +\n"
        "# optional 'name'), a .smi file, or a .cxsmiles file (e.g. Enamine REAL);\n"
        "# plain or .bz2/.gz compressed:\n"
        '# ligand_file: "library.cxsmiles.bz2"\n'
        '# ligand_file_format: "auto"    # auto | csv | smiles | cxsmiles\n'
        "\n"
        f"{pocket_block}"
    )
