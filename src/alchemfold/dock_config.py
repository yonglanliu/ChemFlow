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

from alchemfold.utils._config import _deep_merge, _load_yaml, _resolve, _to_namespace

SUPPORTED_ENGINES = {"boltz", "chai", "diffdock"}
SUPPORTED_OUTPUT_FORMATS = {"pdb", "mmcif"}
SUPPORTED_ACCELERATORS = {"gpu", "cpu", "tpu"}

DEFAULTS: dict[str, Any] = {
    "engine": "boltz",            # boltz | chai
    "workdir": ".",
    # Job name: ALL files of the run — config, engine inputs, SLURM/runner scripts, logs and
    # predictions (out_dir) — live in <workdir>/<jobname>/. Empty => directly in <workdir>.
    "jobname": "",
    "out_dir": "docking_output",
    "seed": 1,
    "use_msa_server": True,
    "accelerator": "gpu",         # gpu | cpu | tpu
    "devices": 1,
    "diffusion_samples": 1,       # boltz --diffusion_samples / chai --num-diffn-samples
    "recycling_steps": 3,         # boltz --recycling_steps / chai --num-trunk-recycles
    "sampling_steps": 200,        # boltz only
    "no_kernels": False,          # boltz only: kernels ON by default (needs the
                                  # boltz env from `chemflow boltz-env install`)
                                  # without the cuequivariance/trifast kernels (which
                                  # conflict with chai's torch<2.7). Set false in a
                                  # boltz-only env that has the kernels installed.
    "output_format": "pdb",       # boltz pdb|mmcif (chai writes mmCIF)
    "override": False,
    # Where the engines download model weights / CCD data. Default (None) uses
    # each engine's own default under $HOME (boltz: ~/.boltz, chai: ~/.cache),
    # which on HPC often hits the HOME disk quota. Point this at /data or scratch.
    # Env fallback: CHEMFLOW_MODEL_CACHE (or CHEMFLOW_BOLTZ_CACHE / BOLTZ_CACHE).
    "cache_dir": None,
    # --- Boltz-2 specific ---
    "boltz_command": "boltz",
    "predict_affinity": False,
    "affinity_binder": None,
    # --- Chai-1 specific ---
    "chai_command": "chai-lab",
    # --- DiffDock specific (diffusion docking into a receptor STRUCTURE) ---
    # DiffDock-L lives in its OWN env (chemflow diffdock-env install), not in alchemfold.
    # diffdock_command: null => "<diffdock env python> -m inference" run from diffdock_dir.
    "diffdock_command": None,
    "diffdock_python": None,       # interpreter of the diffdock env (auto-detected)
    "diffdock_dir": None,          # the cloned gcorso/DiffDock folder (auto-detected)
    "diffdock_trajectory": False,  # save the reverse-diffusion trajectory (multi-frame PDB per pose)
    "trajectories_dir": None,      # screens: one folder for all trajectories (default <workdir>/trajectories)
    "diffdock_batch_size": None,   # GPU batch over the poses of ONE ligand (DiffDock --batch_size, default 10); only helps when poses > batch
    "diffdock_dedupe_esm": True,   # embed the receptor once per run, not once per ligand (see diffdock/_shim.py)
    "diffdock_samples": None,      # poses per ligand (DiffDock --samples_per_complex); null => diffusion_samples if >1 else 10
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
    # Boltz library screening runs batched (one boltz call, one MSA) and writes
    # ALL poses into poses_dir and ALL scores into results_csv (no per-ligand
    # folders). Defaults: <workdir>/poses and <workdir>/results.csv.
    "poses_dir": None,
    "results_csv": None,
    "keep_raw": False,             # keep Boltz raw output + input YAMLs
    # --- optional binding-site pocket constraint ---
    # pocket:
    #   binder: L
    #   max_distance: 6.0
    #   force: false        # true = steering potential, not just a soft feature
    #   contacts: [{chain: A, residue: 123}, {chain: A, residue: 145}]
    "pocket": None,
    # --- other Boltz constraints, passed through in Boltz's own shape ---
    # constraints:
    #   - bond:                       # explicit covalent bond (e.g. a covalent ligand)
    #       atom1: [A, 123, SG]       # [chain, residue index, atom name]
    #       atom2: [L, 1, C12]
    #   - contact:                    # pairwise distance restraint
    #       token1: [A, 123]          # [chain, residue index] or [chain, atom name]
    #       token2: [L, 1]
    #       max_distance: 5.0
    #       force: false
    "constraints": [],
    # --- optional structural templates (Boltz-2 only; PROTEIN chains only) ---
    # Bias the receptor towards a known conformation (e.g. an activated/open
    # state the model would not otherwise sample). The ligand is never
    # templated, so its pose is still predicted.
    # templates:
    #   - cif: 7lpe.cif        # .cif (or .pdb on builds that accept it)
    #     chain_id: [A, B]     # chains of THIS config to template (default: best match)
    #     template_id: [A, B]  # matching chains inside the file (default: best match)
    #     force: true          # steering potential during diffusion, not just an input feature
    #     threshold: 2.0       # max backbone deviation in Angstrom; REQUIRED with force
    "templates": [],
}


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a docking YAML, merged over the defaults; resolve paths.

    ``workdir: "."`` (the default) means the **current working directory** — all
    outputs (``out_dir``, SLURM script, per-ligand folders) are written there, so
    running from a project directory keeps results out of the code/config dir.
    Input files (ligands, receptor, MSAs) still resolve relative to the config.
    """
    import os

    config_path = Path(path).expanduser().resolve()
    conf = _to_namespace(_deep_merge(DEFAULTS, _load_yaml(config_path)))

    base_dir = config_path.parent
    workdir = getattr(conf, "workdir", ".")
    if not workdir or str(workdir) == ".":
        conf.workdir = os.getcwd()
    else:
        conf.workdir = _resolve(base_dir, workdir)
    jobname = str(getattr(conf, "jobname", "") or "").strip()
    if jobname:
        # A config that already lives in <workdir>/<jobname>/ means that folder IS the job folder
        # (no <jobname>/<jobname> nesting); otherwise the job folder is <workdir>/<jobname>.
        in_job_folder = base_dir.name == jobname and (
            not workdir or str(workdir) == "." or Path(conf.workdir).resolve() == base_dir.parent)
        conf.workdir = str(base_dir if in_job_folder else Path(conf.workdir) / jobname)

    for key in ("receptor_pdb", "diffdock_model_dir", "diffdock_config", "ligand_file",
                "diffdock_dir", "diffdock_python"):
        value = getattr(conf, key, None)
        if value:
            setattr(conf, key, _resolve(base_dir, value))

    # Model/CCD cache: expand ~ and make absolute (relative to cwd), so the engine
    # downloads weights where there is room rather than under $HOME. If unset,
    # fall back to a /data (or scratch) location — never the HOME quota.
    cache = getattr(conf, "cache_dir", None) or default_cache_dir()
    if cache:
        conf.cache_dir = str(Path(cache).expanduser().absolute())
    for protein in getattr(conf, "proteins", []) or []:
        msa = getattr(protein, "msa", None)
        if msa and str(msa) not in {"empty", "server"}:
            protein.msa = _resolve(base_dir, msa)
    # Template structures resolve relative to the config, like every other input
    # file — the staged job runs from a different cwd.
    for template in getattr(conf, "templates", []) or []:
        cif = getattr(template, "cif", None)
        if cif:
            template.cif = _resolve(base_dir, cif)
    return conf


def default_cache_dir() -> str | None:
    """A non-HOME default for model/CCD weight downloads.

    HPC HOME directories have small disk quotas, so weights must not land under
    ``~``. Resolution order: ``CHEMFLOW_MODEL_CACHE`` / ``CHEMFLOW_BOLTZ_CACHE``
    / ``BOLTZ_CACHE`` env vars, then ``<base>/<user>/.alchemfold_cache`` for the
    first existing ``base`` in ``/data``, ``/scratch``, ``/gpfs/data``,
    ``/vf/users``. Returns None only if none of those exist (e.g. a laptop),
    letting the engine use its own default.
    """
    import getpass
    import os

    for var in ("CHEMFLOW_MODEL_CACHE", "CHEMFLOW_BOLTZ_CACHE", "BOLTZ_CACHE"):
        value = os.environ.get(var)
        if value:
            return str(Path(value).expanduser())
    try:
        user = os.environ.get("USER") or getpass.getuser()
    except Exception:
        user = None
    if user:
        for base in ("/data", "/scratch", "/gpfs/data", "/vf/users"):
            candidate = Path(base) / user
            if candidate.is_dir():
                return str(candidate / ".alchemfold_cache")
    return None


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
      * SDF (``.sdf``): parsed with RDKit; each record's title (``_Name``) is the
        id and the molecule is canonicalized to SMILES. Needs ``rdkit`` installed
        (present in the ``alchemfold`` env).
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
        elif inner == ".sdf":
            fmt = "sdf"
        else:
            fmt = "smiles"

    parsed: list[dict] = []
    if fmt == "sdf":
        import io as _io

        try:
            from rdkit import Chem  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ValueError(
                "reading an .sdf ligand file needs RDKit; install it in the "
                "'alchemfold' env (conda install -c conda-forge rdkit)"
            ) from exc
        supplier = Chem.ForwardSDMolSupplier(_io.BytesIO(text.encode("utf-8")))
        for n, mol in enumerate(supplier, start=1):
            if mol is None:
                continue
            raw_name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            smiles = Chem.MolToSmiles(mol)
            if not smiles:
                continue
            parsed.append({"id": _sanitize_id(raw_name, f"lig{n}"), "smiles": smiles})
    elif fmt == "csv":
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

    import re

    jobname = str(getattr(conf, "jobname", "") or "").strip()
    if jobname and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", jobname):
        raise ValueError("jobname may contain only letters, digits, '.', '_' and '-' "
                         f"(got {jobname!r})")

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
    _validate_constraints(conf)
    _validate_templates(conf)

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
    chain_ids = _chain_ids(conf)
    for contact in contacts:
        chain = str(getattr(contact, "chain", "")).strip()
        if chain not in chain_ids:
            raise ValueError(
                f"pocket contact chain '{chain}' is not a declared chain id {sorted(chain_ids)}"
            )
        residue = getattr(contact, "residue", None)
        if residue is None or str(residue).strip() == "":
            raise ValueError(
                "each pocket contact needs a 'residue': a positive index for a protein "
                "chain, or an atom name for a ligand chain"
            )
        # A residue INDEX must be positive; anything non-numeric is taken as an
        # atom name, which Boltz accepts when the chain is a ligand.
        try:
            index = int(str(residue).strip())
        except (TypeError, ValueError):
            continue
        if index < 1:
            raise ValueError(f"pocket contact residue index must be >= 1 (got {index})")


def _token_text(token) -> str:
    return ", ".join(str(part) for part in token)


def _validate_token(token, label: str, chain_ids: set[str], *, want_atom: bool) -> None:
    """Check one Boltz token: ``[chain, residue]`` or ``[chain, residue, atom]``."""
    if not isinstance(token, (list, tuple)):
        raise ValueError(
            f"{label} must be a list — "
            + ("[chain, residue, atom_name]" if want_atom else "[chain, residue_or_atom]")
        )
    expected = 3 if want_atom else 2
    if len(token) != expected:
        raise ValueError(
            f"{label} needs {expected} entries "
            + ("[chain, residue, atom_name]" if want_atom else "[chain, residue_or_atom]")
            + f", got {len(token)}: [{_token_text(token)}]"
        )
    chain = str(token[0]).strip()
    if chain not in chain_ids:
        raise ValueError(
            f"{label} chain '{chain}' is not a declared chain id {sorted(chain_ids)}"
        )
    residue = str(token[1]).strip()
    if not residue:
        raise ValueError(f"{label} needs a residue index (or atom name)")
    if want_atom:
        try:
            if int(residue) < 1:
                raise ValueError(f"{label} residue index must be >= 1 (got {residue})")
        except ValueError as exc:
            if "must be >= 1" in str(exc):
                raise
            raise ValueError(
                f"{label} residue must be an integer index for a bond, got '{residue}' "
                "(atom names go in the third position)"
            ) from None
        if not str(token[2]).strip():
            raise ValueError(f"{label} needs an atom name as its third entry")


def _validate_constraints(conf) -> None:
    """Check the pass-through ``constraints`` list (Boltz bond / contact / pocket)."""
    constraints = getattr(conf, "constraints", None) or []
    if not constraints:
        return
    engine = str(getattr(conf, "engine", "boltz")).strip().lower()
    if engine != "boltz":
        raise ValueError(
            f"bond/contact constraints are a Boltz-2 feature; engine '{engine}' does not "
            "support them (drop the 'constraints' block or set engine: boltz)"
        )
    chain_ids = _chain_ids(conf)
    known = {"bond", "contact", "pocket"}
    for index, entry in enumerate(constraints):
        where = f"constraints[{index}]"
        kinds = [k for k in known if getattr(entry, k, None) is not None]
        if not kinds:
            present = sorted(vars(entry)) if hasattr(entry, "__dict__") else []
            raise ValueError(
                f"{where} must be one of {sorted(known)} "
                f"(got {present or 'an empty entry'})"
            )
        if len(kinds) > 1:
            raise ValueError(f"{where} sets several constraint types at once: {kinds}")
        kind = kinds[0]
        body = getattr(entry, kind)

        if kind == "bond":
            for field in ("atom1", "atom2"):
                token = getattr(body, field, None)
                if token is None:
                    raise ValueError(f"{where} bond needs '{field}' [chain, residue, atom_name]")
                _validate_token(token, f"{where} bond.{field}", chain_ids, want_atom=True)
        elif kind == "contact":
            for field in ("token1", "token2"):
                token = getattr(body, field, None)
                if token is None:
                    raise ValueError(
                        f"{where} contact needs '{field}' [chain, residue_or_atom]")
                _validate_token(token, f"{where} contact.{field}", chain_ids, want_atom=False)
            max_distance = getattr(body, "max_distance", None)
            if max_distance is None:
                raise ValueError(f"{where} contact needs a 'max_distance' in Angstrom")
            if float(max_distance) <= 0:
                raise ValueError(f"{where} contact max_distance must be > 0 Angstrom")
        else:  # pocket written inline in Boltz's own shape
            if not str(getattr(body, "binder", "") or "").strip():
                raise ValueError(f"{where} pocket needs a 'binder' chain id")
            if not (getattr(body, "contacts", None) or []):
                raise ValueError(f"{where} pocket needs at least one contact")


def _as_id_list(value) -> list[str]:
    """Normalise a chain-id field: ``"A"`` / ``"A,B"`` / ``["A", "B"]`` -> list."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [tok.strip() for tok in value.replace(";", ",").split(",") if tok.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _validate_templates(conf) -> None:
    """Check the optional ``templates`` block (Boltz-2 only, protein chains only)."""
    templates = getattr(conf, "templates", None) or []
    if not templates:
        return
    engine = str(getattr(conf, "engine", "boltz")).strip().lower()
    if engine != "boltz":
        raise ValueError(
            f"structural templates are a Boltz-2 feature; engine '{engine}' does not "
            "support them (drop the 'templates' block or set engine: boltz)"
        )
    protein_ids = {str(p.id).strip() for p in (getattr(conf, "proteins", []) or [])}
    for index, template in enumerate(templates):
        where = f"templates[{index}]"
        cif = str(getattr(template, "cif", "") or "").strip()
        if not cif:
            raise ValueError(f"{where} needs a 'cif' path to the template structure")
        if not Path(cif).is_file():
            raise ValueError(f"{where} template file not found: {cif}")

        chain_ids = _as_id_list(getattr(template, "chain_id", None))
        for chain in chain_ids:
            if chain not in protein_ids:
                raise ValueError(
                    f"{where} chain_id '{chain}' is not a protein chain id "
                    f"{sorted(protein_ids)} — only protein chains can be templated"
                )
        template_ids = _as_id_list(getattr(template, "template_id", None))
        if template_ids and len(template_ids) != len(chain_ids):
            raise ValueError(
                f"{where} template_id has {len(template_ids)} entries but chain_id has "
                f"{len(chain_ids)} — they map one-to-one, so give both or neither"
            )

        force = bool(getattr(template, "force", False))
        threshold = getattr(template, "threshold", None)
        if force:
            if threshold is None:
                raise ValueError(
                    f"{where} sets force: true, which needs a 'threshold' (max backbone "
                    "deviation in Angstrom, e.g. 2.0)"
                )
            if float(threshold) <= 0:
                raise ValueError(f"{where} threshold must be > 0 Angstrom")
        elif threshold is not None and float(threshold) <= 0:
            raise ValueError(f"{where} threshold must be > 0 Angstrom")


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


# Named HPC presets, selectable GPU types and SLURM resources are user-
# customizable via a YAML file (see :mod:`alchemfold.hpc_config`). Presets may be
# single-node single-GPU (a plain walltime string) or multi-node / multi-GPU (a
# mapping). These imports resolve the current file contents.
from alchemfold.hpc_config import (  # noqa: E402
    DEFAULT_PARTITION,
    default_preset as _default_preset,
    gpu_types as _gpu_types,
    preset_resources as _preset_resources,
    presets as _presets,
    resource_defaults as _resource_defaults,
)


def stage_slurm(
    config_path: str | Path,
    task_directory: str | Path,
    *,
    preset: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    gpu_type: str | None = None,
    cpus_per_task: int | None = None,
    mem: str | None = None,
    conda_env: str | None = None,
) -> tuple[Path, Path]:
    """Write a SLURM submit script for one docking run.

    Resources come from the chosen ``preset`` (resolved via the customizable HPC
    config, :mod:`alchemfold.hpc_config`), which may be single-node single-GPU (a
    plain walltime string) or multi-node / multi-GPU (a mapping of
    ``time/nodes/ntasks_per_node/gpus_per_node/gpu_type/cpus_per_task/mem``).
    Any non-None argument overrides the preset/defaults. Returns
    ``(slurm_script, config_path)``.

    Note: AI co-folding (Boltz-2/Chai-1) is single-GPU inference, so multi-node
    presets do not speed up one docking run — they exist for cluster policies and
    for users who deliberately want them.
    """
    preset = preset or _default_preset()
    try:
        res = _preset_resources(preset)
    except KeyError:
        raise ValueError(
            f"unknown HPC preset {preset!r}; choose one of {sorted(_presets())}"
        )
    # Explicit args override the preset/defaults.
    partition = partition if partition is not None else res["partition"]
    account = account if account is not None else res["account"]
    gpu_type = gpu_type if gpu_type is not None else res["gpu_type"]
    cpus_per_task = cpus_per_task if cpus_per_task is not None else res["cpus_per_task"]
    mem = mem if mem is not None else res["mem"]
    conda_env = conda_env if conda_env is not None else res["conda_env"]
    nodes = int(res["nodes"])
    ntasks_per_node = int(res["ntasks_per_node"])
    gpus_per_node = int(res["gpus_per_node"])
    walltime = res["time"]

    config_path = Path(config_path).expanduser().resolve()
    task = Path(task_directory).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    script = task / "run_docking.slurm"

    gtype = str(gpu_type or "").strip()
    gres = ""
    if gpus_per_node > 0:
        gres = f"#SBATCH --gres=gpu:{gtype + ':' if gtype else ''}{gpus_per_node}\n"
    account_line = f"#SBATCH --account={account}\n" if str(account).strip() else ""
    env = str(conda_env or "").strip()
    activate = (
        'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
        f"conda activate {env}\n"
        if env
        else ""
    )
    script.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name=alchemfold_dock\n"
        f"#SBATCH --chdir={shlex.quote(str(task))}\n"
        f"#SBATCH --partition={partition}\n"
        f"{account_line}"
        f"#SBATCH --nodes={nodes}\n"
        f"#SBATCH --ntasks-per-node={ntasks_per_node}\n"
        f"#SBATCH --cpus-per-task={int(cpus_per_task)}\n"
        f"{gres}"
        f"#SBATCH --mem={mem}\n"
        f"#SBATCH --time={walltime}            # preset: {preset}\n"
        "#SBATCH --output=dock_%j.out\n"
        "set -euo pipefail\n"
        "trap 'echo \"[FAILED] line $LINENO: $BASH_COMMAND (exit $?)\" >&2' ERR\n"
        f"{activate}"
        f'exec {shlex.quote(sys.executable)} -m alchemfold.dock_run '
        f'--config {shlex.quote(str(config_path))}\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | 0o100)
    return script, config_path


def _parse_pocket_contacts(contacts, default_chain: str = "A"):
    """Normalise pocket contacts into ``[(chain, residue), ...]``.

    Accepts a list/tuple of ``(chain, residue)`` pairs, or strings in the forms
    ``"A:123"``, ``"A 123"``, or just ``"123"`` (uses ``default_chain``). A single
    comma/space/semicolon-separated string (e.g. ``"A:123, A:145"``) is also accepted.
    A numeric residue becomes an ``int`` (a residue index); anything else is kept
    as a string, which Boltz reads as an **atom name** on a ligand chain.
    Returns ``[]`` when nothing usable is given.
    """
    if not contacts:
        return []
    if isinstance(contacts, str):
        contacts = [tok for tok in contacts.replace(";", ",").split(",") if tok.strip()]
    out = []
    for item in contacts:
        if isinstance(item, (tuple, list)):
            chain, residue = item[0], item[1]
        else:
            s = str(item).strip()
            if not s:
                continue
            if ":" in s:
                chain, residue = s.split(":", 1)
            elif " " in s:
                chain, residue = s.split(None, 1)
            else:
                chain, residue = default_chain, s
        chain = str(chain).strip() or default_chain
        residue = str(residue).strip()
        if not residue:
            continue
        try:
            out.append((chain, int(residue)))
        except (TypeError, ValueError):
            out.append((chain, residue))   # atom name (ligand chain)
    return out


def _parse_token(text, *, want_atom: bool, default_chain: str = "A") -> list:
    """Parse ``"A:123"`` / ``"A:C12"`` / ``"A:123:SG"`` into a Boltz token list.

    With ``want_atom`` the token is a bond atom and must be
    ``chain:residue:atom_name``; otherwise it is a contact token,
    ``chain:residue_or_atom``. A bare value uses ``default_chain``.
    """
    parts = [p.strip() for p in str(text).replace("/", ":").split(":") if p.strip()]
    if not parts:
        raise ValueError(f"empty token: {text!r}")
    if want_atom:
        if len(parts) != 3:
            raise ValueError(
                f"bond atom must be CHAIN:RESIDUE:ATOM_NAME (got {text!r})")
        chain, residue, atom = parts
        try:
            return [chain, int(residue), atom]
        except ValueError:
            raise ValueError(
                f"bond atom residue must be an integer index (got {residue!r} in {text!r})"
            ) from None
    if len(parts) == 1:
        parts = [default_chain, parts[0]]
    if len(parts) != 2:
        raise ValueError(f"contact token must be CHAIN:RESIDUE_OR_ATOM (got {text!r})")
    chain, residue = parts
    try:
        return [chain, int(residue)]
    except ValueError:
        return [chain, residue]    # atom name


def _split_pair(text: str) -> list[str]:
    """Split ``"A:123,L:1"`` (or whitespace/semicolon separated) into two tokens."""
    parts = [p.strip() for p in str(text).replace(";", ",").replace(" ", ",").split(",")
             if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"expected two tokens separated by a comma (got {len(parts)} in {text!r})")
    return parts


def _parse_constraints(
    bonds=None,
    contacts=None,
    *,
    contact_max_distance: float = 5.0,
    contact_force: bool = False,
    default_chain: str = "A",
) -> list[dict]:
    """Build ``constraints`` entries from CLI-style ``"A:123:SG,L:1:C12"`` strings.

    A contact may carry its own distance as a third field
    (``"A:123,L:1,4.5"``); otherwise ``contact_max_distance`` applies.
    """
    out: list[dict] = []
    for spec in (bonds or []):
        atom1, atom2 = _split_pair(spec)
        out.append({"bond": {
            "atom1": _parse_token(atom1, want_atom=True, default_chain=default_chain),
            "atom2": _parse_token(atom2, want_atom=True, default_chain=default_chain),
        }})
    for spec in (contacts or []):
        fields = [p.strip() for p in str(spec).replace(";", ",").split(",") if p.strip()]
        distance = contact_max_distance
        if len(fields) == 3:
            *fields, tail = fields
            try:
                distance = float(tail)
            except ValueError:
                raise ValueError(
                    f"third field of --contact must be a distance in Angstrom "
                    f"(got {tail!r} in {spec!r})") from None
        if len(fields) != 2:
            raise ValueError(
                f"--contact takes TOKEN,TOKEN[,DISTANCE] (got {spec!r})")
        entry = {
            "token1": _parse_token(fields[0], want_atom=False, default_chain=default_chain),
            "token2": _parse_token(fields[1], want_atom=False, default_chain=default_chain),
            "max_distance": float(distance),
        }
        if contact_force:
            entry["force"] = True
        out.append({"contact": entry})
    return out


def _format_constraints_block(constraints: list[dict] | None) -> str:
    """Render the pass-through ``constraints:`` YAML block (or a commented example)."""
    if not constraints:
        return (
            "# --- other Boltz constraints (Boltz-2 only) -------------------------\n"
            "# constraints:\n"
            "#   - bond:                     # explicit covalent bond\n"
            "#       atom1: [A, 123, SG]     # [chain, residue index, atom name]\n"
            "#       atom2: [L, 1, C12]\n"
            "#   - contact:                  # pairwise distance restraint\n"
            "#       token1: [A, 123]        # [chain, residue index] or [chain, atom name]\n"
            "#       token2: [L, 1]\n"
            "#       max_distance: 5.0\n"
            "#       force: false            # true = steering potential\n"
        )
    lines = ["# Boltz constraints: covalent bonds and pairwise distance restraints.\n",
             "constraints:\n"]
    for entry in constraints:
        if "bond" in entry:
            bond = entry["bond"]
            lines.append("  - bond:\n")
            lines.append(f"      atom1: [{_token_text(bond['atom1'])}]"
                         "       # [chain, residue, atom name]\n")
            lines.append(f"      atom2: [{_token_text(bond['atom2'])}]\n")
        elif "contact" in entry:
            contact = entry["contact"]
            lines.append("  - contact:\n")
            lines.append(f"      token1: [{_token_text(contact['token1'])}]\n")
            lines.append(f"      token2: [{_token_text(contact['token2'])}]\n")
            lines.append(f"      max_distance: {float(contact['max_distance']):g}"
                         "          # Angstrom\n")
            lines.append(f"      force: {'true' if contact.get('force') else 'false'}"
                         "              # true = steering potential, not just a feature\n")
    return "".join(lines)


def parse_fasta(path, *, first_id: str = "A"):
    """Parse a FASTA/.fa file into ``[(chain_id, sequence), ...]``.

    Each ``>`` record becomes one protein chain; chain ids are assigned
    sequentially from ``first_id`` (A, B, C, …), ignoring the header text. A file
    with no ``>`` header is treated as a single raw sequence. Sequences are
    uppercased with whitespace and any trailing ``*`` stop codon removed.
    """
    records = _split_sequences(Path(path).read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"No sequence found in FASTA file: {path}")
    base = ord(first_id[:1].upper()) if first_id else ord("A")
    return [(chr(base + i), seq) for i, seq in enumerate(records)]


def parse_sequences_text(text, *, first_id: str = "A"):
    """Parse free-text sequence input into ``[(chain_id, sequence), ...]``.

    Accepts: FASTA text (``>`` headers → one chain each), several sequences
    separated by a blank line, or a single sequence (possibly line-wrapped).
    Chain ids are assigned A, B, C, … from ``first_id``. Returns ``[]`` for empty
    input.
    """
    records = _split_sequences(text)
    base = ord(first_id[:1].upper()) if first_id else ord("A")
    return [(chr(base + i), seq) for i, seq in enumerate(records)]


def _split_sequences(text: str) -> list[str]:
    """Split FASTA or blank-line-separated text into cleaned sequence strings.

    If any ``>`` header is present the text is parsed as FASTA (one record per
    header). Otherwise blocks separated by a blank line are treated as separate
    sequences, and lines within a block are joined (handles line-wrapped input).
    Each sequence is uppercased with whitespace and trailing ``*`` removed.
    """
    def _clean(parts: list[str]) -> str:
        return "".join(parts).replace(" ", "").replace("\t", "").replace("*", "").strip().upper()

    lines = (text or "").splitlines()
    has_header = any(ln.lstrip().startswith(">") for ln in lines)
    records: list[str] = []
    current: list[str] = []

    if has_header:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                seq = _clean(current)
                if seq:
                    records.append(seq)
                current = []
                continue
            current.append(line)
        seq = _clean(current)
        if seq:
            records.append(seq)
    else:
        # Blank line separates sequences; join wrapped lines within a block.
        for line in lines:
            if line.strip() == "":
                seq = _clean(current)
                if seq:
                    records.append(seq)
                current = []
            else:
                current.append(line.strip())
        seq = _clean(current)
        if seq:
            records.append(seq)
    return records



def _format_proteins_block(proteins) -> str:
    lines = ["proteins:\n"]
    for chain_id, seq in proteins:
        lines.append(f'  - id: "{chain_id}"\n')
        lines.append(f'    sequence: "{seq}"\n')
    return "".join(lines)


def _parse_templates(
    paths,
    *,
    chains=None,
    template_chains=None,
    force: bool = False,
    threshold: float | None = None,
) -> list[dict]:
    """Build ``templates`` entries from CLI-style arguments.

    ``paths`` is one path or a list of them; ``chains`` / ``template_chains``
    accept ``"A,B"`` or a list and apply to every template. ``force`` without an
    explicit ``threshold`` defaults to 2.0 A, which is tight enough to hold a
    conformational state the model would otherwise not sample.
    """
    if not paths:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    chain_ids = _as_id_list(chains)
    template_ids = _as_id_list(template_chains)
    out: list[dict] = []
    for path in paths:
        if not str(path).strip():
            continue
        entry: dict[str, Any] = {"cif": str(Path(str(path)).expanduser().resolve())}
        if chain_ids:
            entry["chain_id"] = list(chain_ids)
        if template_ids:
            entry["template_id"] = list(template_ids)
        if force:
            entry["force"] = True
            entry["threshold"] = float(threshold if threshold is not None else 2.0)
        elif threshold is not None:
            entry["threshold"] = float(threshold)
        out.append(entry)
    return out


def _format_templates_block(templates: list[dict] | None) -> str:
    """Render the ``templates:`` YAML block (or a commented placeholder)."""
    if not templates:
        return (
            "# --- structural templates (Boltz-2 only; PROTEIN chains only) -------\n"
            "# Bias the receptor towards a known conformation — e.g. an open/active\n"
            "# state the model does not sample on its own. Without `force` the\n"
            "# template is only an input feature the model may overrule; with it, a\n"
            "# potential steers the backbone during diffusion and `threshold` sets\n"
            "# how far (in Angstrom) it may drift. The ligand is never templated.\n"
            "# templates:\n"
            '#   - cif: "reference.cif"      # .cif (or .pdb where your build accepts it)\n'
            "#     chain_id: [A]             # chains of THIS config to template\n"
            "#     template_id: [A]          # matching chains in the file (default: best match)\n"
            "#     force: true               # steering potential, not just a feature\n"
            "#     threshold: 2.0            # max backbone deviation (A); REQUIRED with force\n"
        )
    lines = [
        "# Structural templates (Boltz-2 only; protein chains only). `force` turns\n",
        "# the template from an input feature into a steering potential during\n",
        "# diffusion; `threshold` is how far (A) the backbone may drift from it.\n",
        "templates:\n",
    ]
    for entry in templates:
        lines.append(f'  - cif: "{entry["cif"]}"\n')
        if entry.get("chain_id"):
            lines.append(f"    chain_id: [{', '.join(entry['chain_id'])}]"
                         "          # chains of THIS config to template\n")
        if entry.get("template_id"):
            lines.append(f"    template_id: [{', '.join(entry['template_id'])}]"
                         "       # matching chains inside the file\n")
        else:
            lines.append("    # template_id: [A]           # matching chains in the file "
                         "(default: best match)\n")
        if entry.get("force"):
            lines.append("    force: true                  # steering potential, not just a feature\n")
            lines.append(f"    threshold: {float(entry['threshold']):.1f}"
                         "               # max backbone deviation (A)\n")
        elif entry.get("threshold") is not None:
            lines.append(f"    threshold: {float(entry['threshold']):.1f}\n")
        else:
            lines.append("    # force: true                # add a steering potential "
                         "(then threshold is required)\n")
            lines.append("    # threshold: 2.0             # max backbone deviation (A)\n")
    return "".join(lines)


def default_yaml_text(
    *,
    engine: str = "boltz",
    protein_id: str = "A",
    ligand_id: str = "L",
    out_dir: str = "docking_output",
    sequence: str = "PUT_YOUR_PROTEIN_SEQUENCE_HERE",
    proteins: list | None = None,
    smiles: str = "CCO",
    with_pocket: bool = False,
    pocket_binder: str | None = None,
    pocket_max_distance: float = 6.0,
    pocket_contacts: list | None = None,
    pocket_force: bool = False,
    ligand_file: str | None = None,
    ligand_file_format: str = "auto",
    cache_dir: str | None = None,
    diffusion_samples: int = 1,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    no_kernels: bool = False,
    boltz_command: str | None = None,
    chai_command: str | None = None,
    use_msa_server: bool = True,
    predict_affinity: bool = False,
    affinity_binder: str | None = None,
    receptor_pdb: str | None = None,
    diffdock_samples: int | None = None,
    diffdock_steps: int | None = None,
    jobname: str = "",
    diffdock_trajectory: bool = False,
    diffdock_batch_size: int | None = None,
    templates: list | None = None,
    constraints: list | None = None,
) -> str:
    """Return a commented example docking config for the chosen engine.

    If ``ligand_file`` is given, the config docks the ligand(s) from that file
    (SMILES / .smi / .cxsmiles / CSV / .sdf) and the inline ligand is commented
    out; otherwise a single inline ligand with ``smiles`` is used.
    """
    if predict_affinity:
        binder = str(affinity_binder or ligand_id)
        affinity_block = (
            "predict_affinity: true          # Boltz-2 only; binding-affinity prediction\n"
            f'affinity_binder: "{binder}"            # ligand id to score (pIC50 = 6 - affinity_pred_value)\n'
        )
    else:
        affinity_block = (
            "predict_affinity: false         # Boltz-2 only; binding-affinity prediction\n"
            '# affinity_binder: "L"          # ligand id to score when predict_affinity: true\n'
        )
    parsed_contacts = _parse_pocket_contacts(pocket_contacts, protein_id)
    force_line = (
        f"  force: {'true' if pocket_force else 'false'}"
        "                   # true = steering potential during diffusion, "
        "not just a soft feature\n"
    )
    if parsed_contacts:
        # Real pocket constraint built from caller-supplied residues.
        binder = str(pocket_binder or ligand_id).strip() or ligand_id
        lines = [
            "pocket:                         # binding-site constraint\n",
            f'  binder: "{binder}"                  # which ligand binds the pocket\n',
            f"  max_distance: {float(pocket_max_distance):g}             "
            "# Angstrom (Boltz pocket constraint)\n",
            force_line,
            "  contacts:                     # residues lining the pocket\n",
        ]
        lines += [f"    - {{chain: {ch}, residue: {res}}}\n" for ch, res in parsed_contacts]
        pocket_block = "".join(lines)
    elif with_pocket:
        pocket_block = (
            "pocket:                         # optional binding-site constraint\n"
            f'  binder: "{ligand_id}"                  # which ligand binds the pocket\n'
            "  max_distance: 6.0             # Angstrom (Boltz pocket constraint)\n"
            + force_line
            + "  contacts:                     # residues lining the pocket\n"
            f"    - {{chain: {protein_id}, residue: 123}}\n"
            f"    - {{chain: {protein_id}, residue: 145}}\n"
        )
    else:
        pocket_block = (
            "# pocket:                       # optional binding-site constraint\n"
            f'#   binder: "{ligand_id}"\n'
            "#   max_distance: 6.0\n"
            "#   force: false                # true = steering potential\n"
            "#   contacts:\n"
            f"#     - {{chain: {protein_id}, residue: 123}}\n"
        )
    if ligand_file:
        ligand_block = (
            "# Dock the ligand(s) from an uploaded file (one run per entry, output\n"
            "# under <workdir>/ligands/<id>/). Supports SMILES/.smi, .cxsmiles, CSV,\n"
            "# and .sdf (plain or .bz2/.gz).\n"
            f'ligand_file: "{ligand_file}"\n'
            f'ligand_file_format: "{ligand_file_format}"   # auto | csv | smiles | cxsmiles | sdf\n'
            "\n"
            "# ligands:                      # (inline ligand disabled — using ligand_file)\n"
            f'#   - id: "{ligand_id}"\n'
            f'#     smiles: "{smiles}"\n'
        )
    else:
        ligand_block = (
            "ligands:\n"
            f'  - id: "{ligand_id}"\n'
            f'    smiles: "{smiles}"          # or use `ccd: "ATP"`\n'
            "\n"
            "# Dock a ligand FILE instead of the single ligand above — one run per\n"
            "# entry, output under <workdir>/ligands/<id>/. SMILES/.smi, .cxsmiles,\n"
            "# CSV ('smiles' column + optional 'name'), or .sdf; plain or .bz2/.gz:\n"
            '# ligand_file: "library.cxsmiles.bz2"\n'
            '# ligand_file_format: "auto"    # auto | csv | smiles | cxsmiles | sdf\n'
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
        'workdir: "."                    # working directory ("." = the current directory you run from)\n'
        + (f'jobname: "{jobname}"            # job folder: EVERYTHING (config, inputs, scripts, logs, predictions) '
           f"goes to <workdir>/{jobname}/\n" if jobname else "")
        + f'out_dir: "{out_dir}"            # predictions folder (inside the job folder when jobname is set)\n'
        'output_format: "pdb"            # pdb | mmcif (Chai writes mmCIF)\n'
        f"use_msa_server: {'true' if use_msa_server else 'false'}            "
        "# compute MSAs via the model's server (needs internet; false on offline nodes)\n"
        "seed: 1\n"
        'accelerator: "gpu"              # gpu | cpu | tpu\n'
        "devices: 1\n"
        f"diffusion_samples: {int(diffusion_samples)}            # predicted poses (chai: --num-diffn-samples)\n"
        f"recycling_steps: {int(recycling_steps)}               # chai: --num-trunk-recycles\n"
        f"sampling_steps: {int(sampling_steps)}             # Boltz only\n"
        f"no_kernels: {'true' if no_kernels else 'false'}                # Boltz only: true = pure-PyTorch "
        "(default; works without cuequivariance/trifast). false needs the kernels: "
        "`chemflow boltz-env install` (that env is then used automatically)\n"
        "override: false\n"
        + (f'cache_dir: "{cache_dir}"' + " " * max(1, 10 - len(str(cache_dir)))
           + "# model/CCD weights cache (boltz --cache / chai downloads); avoid HOME quota\n"
           if cache_dir else
           '# cache_dir: "/data/$USER/.alchemfold_cache"  # model/CCD weights cache; '
           "use /data or scratch, never HOME (disk quota)\n")
        + f"{affinity_block}"
        + (f'boltz_command: "{boltz_command}"'
           + (" " * max(1, 10 - len(str(boltz_command))))
           + "# Boltz-2 from its own env\n"
           if boltz_command else
           'boltz_command: "boltz"          # engine: boltz — found in the '
           "sibling conda env named 'boltz'\n")
        + (f'chai_command: "{chai_command}"'
           + (" " * max(1, 10 - len(str(chai_command))))
           + "# Chai-1 from its own env\n"
           if chai_command else
           'chai_command: "chai-lab"        # engine: chai — found in the '
           "sibling conda env named 'chai'\n")
        + "# --- DiffDock (engine: diffdock) docks into a receptor structure ---\n"
        + (f'receptor_pdb: "{receptor_pdb}"   # receptor structure DiffDock docks into\n'
           if receptor_pdb else
           '# receptor_pdb: "receptor.pdb"  # receptor structure (or omit to fold the sequence via ESMFold)\n')
        + "# DiffDock-L runs from its OWN env (chemflow diffdock-env install); paths are auto-detected:\n"
        '# diffdock_python: "/path/to/envs/diffdock/bin/python"\n'
        '# diffdock_dir: "/path/to/DiffDock"\n'
        + (f"diffdock_samples: {int(diffdock_samples)}            # poses per ligand (the best by confidence is kept)\n"
           if diffdock_samples else
           "# diffdock_samples: 10           # poses per ligand (confidence ranks them)\n")
        + (f"diffdock_inference_steps: {int(diffdock_steps)}     # denoising (reverse-diffusion) steps; default 20\n"
           if diffdock_steps else
           "# diffdock_inference_steps: 20   # denoising (reverse-diffusion) steps\n")
        + (f"diffdock_batch_size: {int(diffdock_batch_size)}       # GPU batch over one ligand's poses (default 10); raise only if poses > 10\n"
           if diffdock_batch_size else
           "# diffdock_batch_size: 10      # GPU batch over one ligand's poses (default 10); only helps when poses > batch\n")
        + ("diffdock_trajectory: true      # save the diffusion trajectory: rank<k>_reverseprocess.pdb (multi-frame, ligand only)\n"
           if diffdock_trajectory else
           "# diffdock_trajectory: true    # save the diffusion trajectory (multi-frame PDB per pose)\n")
        + '# diffdock_model_dir: "/path/to/DiffDock/workdir"\n'
        "\n"
        + _format_proteins_block(proteins or [(protein_id, sequence)])
        + "\n"
        f"{ligand_block}"
        "\n"
        f"{pocket_block}"
        "\n"
        + _format_constraints_block(constraints)
        + "\n"
        + _format_templates_block(templates)
    )
