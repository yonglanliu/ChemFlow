"""In-process protein-protein docking with PyRosetta (ternary search).

Easier to install than full Rosetta (``conda install pyrosetta`` with a license)
and called directly in Python, so there is no external binary path to configure.
Supports global search, local refinement, and an optional *linker-aware* distance
restraint between the two warhead attachment atoms (a restraint-based constraint,
without PatchDock).

Layering: ``load_config`` / ``validate_config`` / ``init_flags`` /
``write_constraint_file`` are pure (testable without PyRosetta). :func:`run`
imports PyRosetta lazily and performs the docking.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alchemfold.boltz_config import _deep_merge, _load_yaml, _resolve, _to_namespace

DEFAULTS: dict[str, Any] = {
    "workdir": ".",
    "out_dir": "pyrosetta_dock_output",
    "input_pdb": None,          # one PDB with both partners as separate chains
    "partners": None,           # "A_B": chain A docks against chain B
    "mode": "global",           # global | local
    "nstruct": 100,
    # Dock the proteins only (strip warhead HETATMs first) so no Rosetta ligand
    # .params are needed. The warheads are carried into each output model by
    # superposition, so linkability still works. Set false to keep ligands in the
    # pose (then ligand_params are required).
    "dock_proteins_only": True,
    "ligand_params": [],        # .params per warhead (only if dock_proteins_only=false)
    # FastRelax options (slow). pre_relax: relax the input structure ONCE before
    # docking (removes input strain → sensible total scores). post_relax: relax
    # each docked model after refinement (better totals/I_sc, much slower).
    "pre_relax": False,
    "post_relax": False,
    "seed": None,
    "extra_init": [],           # extra pyrosetta.init flags
    # Optional distance restraint between two PROTEIN residues (one per partner),
    # on their CA atoms, applied during docking. Works in proteins-only mode (no
    # ligand params); steers docking toward a chosen inter-protein arrangement.
    # restraint:
    #   res1: {chain: A, resseq: 123}   # a residue on partner 1 (e.g. near its warhead)
    #   res2: {chain: B, resseq: 456}   # a residue on partner 2
    #   distance: 10.0     # target CA-CA distance (A)
    #   stddev: 2.0        # harmonic width
    #   weight: 1.0        # constraint weight in the score
    #   atom: CA           # atom to restrain (default CA)
    "restraint": None,
}

SUPPORTED_MODES = {"global", "local"}


def load_config(path: str | Path) -> SimpleNamespace:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"PyRosetta dock config not found: {config_path}\n"
            "Create one with: chemflow protac pyrosetta-dock generate -o pyrosetta_dock.yaml")
    conf = _to_namespace(_deep_merge(DEFAULTS, _load_yaml(config_path)))
    base = config_path.parent
    workdir = getattr(conf, "workdir", ".")
    conf.workdir = str(base) if (not workdir or str(workdir) == ".") else _resolve(base, workdir)
    if getattr(conf, "input_pdb", None):
        conf.input_pdb = _resolve(base, conf.input_pdb)
    conf.ligand_params = [_resolve(base, p) for p in (getattr(conf, "ligand_params", None) or [])]
    conf._config_path = str(config_path)
    return conf


def validate_config(conf: SimpleNamespace) -> None:
    pdb = getattr(conf, "input_pdb", None)
    if not pdb:
        raise ValueError("input_pdb is required (one PDB with both partners as chains)")
    if not Path(str(pdb)).is_file():
        raise ValueError(f"input_pdb not found: {pdb}")
    partners = getattr(conf, "partners", None)
    if not partners or "_" not in str(partners):
        raise ValueError("partners is required, e.g. 'A_B'")
    if str(getattr(conf, "mode", "global")) not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {sorted(SUPPORTED_MODES)}")
    if int(getattr(conf, "nstruct", 0)) < 1:
        raise ValueError("nstruct must be >= 1")
    if not getattr(conf, "dock_proteins_only", True):
        for params in getattr(conf, "ligand_params", None) or []:
            if not Path(str(params)).is_file():
                raise ValueError(f"ligand_params file not found: {params}")
    restraint = getattr(conf, "restraint", None)
    if restraint is not None:
        _validate_restraint(restraint)


def _validate_restraint(restraint: SimpleNamespace) -> None:
    for key in ("res1", "res2"):
        res = getattr(restraint, key, None)
        if res is None:
            raise ValueError(f"restraint needs '{key}' (chain, resseq)")
        for field_name in ("chain", "resseq"):
            if getattr(res, field_name, None) in (None, ""):
                raise ValueError(f"restraint.{key} needs '{field_name}'")
    if float(getattr(restraint, "distance", 0)) <= 0:
        raise ValueError("restraint.distance must be > 0")
    if float(getattr(restraint, "stddev", 0)) <= 0:
        raise ValueError("restraint.stddev must be > 0")


def init_flags(conf: SimpleNamespace, out_dir: Path) -> list[str]:
    """Flags for ``pyrosetta.init`` (ligand params, seed)."""
    flags: list[str] = ["-mute", "all"]
    params = getattr(conf, "ligand_params", None) or []
    if params and not getattr(conf, "dock_proteins_only", True):
        flags += ["-extra_res_fa", *[str(p) for p in params]]
    if getattr(conf, "seed", None) is not None:
        flags += ["-run:constant_seed", "-run:jran", str(int(conf.seed))]
    extra = getattr(conf, "extra_init", None) or []
    if isinstance(extra, str):
        extra = shlex.split(extra)
    flags += [str(x) for x in extra]
    return flags


def strip_hetatm(pdb_text: str) -> str:
    """Keep only ATOM/TER/END records (drop ligands/waters) for params-free docking."""
    out = []
    for line in pdb_text.splitlines(keepends=True):
        rec = line[:6].strip()
        if rec in ("ATOM", "TER", "END"):
            out.append(line)
    if not out or not out[-1].startswith("END"):
        out.append("END\n")
    return "".join(out)


def _ca_and_hetatm(pdb_text: str):
    """Per-chain CA coords keyed by (resseq) and the chain's HETATM lines."""
    ca: dict[str, dict[str, tuple]] = {}
    het: dict[str, list[str]] = {}
    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        chain = line[21:22]
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if rec == "ATOM" and line[12:16].strip() == "CA":
            ca.setdefault(chain, {})[line[22:27].strip()] = xyz
        elif rec == "HETATM":
            het.setdefault(chain, []).append(line)
    return ca, het


def _apply_transform(line: str, rotation, shift) -> str:
    import numpy as np

    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    x, y, z = (rotation @ xyz) + shift
    return line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]


def transfer_ligands(original_pdb: str | Path, model_pdb: str | Path, out_pdb: str | Path) -> Path:
    """Carry each chain's warhead from ``original_pdb`` onto the docked ``model_pdb``.

    For every chain, superpose the original CA atoms onto the docked CA atoms
    (Kabsch) and apply that rigid transform to the chain's HETATM, so the warhead
    rides along with its docked protein. Writes the docked protein + transferred
    warheads to ``out_pdb``.
    """
    import numpy as np

    from chemflow.generative.reinvent.linker.fit3d import _kabsch

    orig_text = Path(original_pdb).read_text(encoding="utf-8", errors="replace")
    model_text = Path(model_pdb).read_text(encoding="utf-8", errors="replace")
    orig_ca, orig_het = _ca_and_hetatm(orig_text)
    model_ca, _ = _ca_and_hetatm(model_text)

    transferred: list[str] = []
    for chain, het_lines in orig_het.items():
        shared = sorted(set(orig_ca.get(chain, {})) & set(model_ca.get(chain, {})))
        if len(shared) < 3:
            continue  # need >=3 CA to define a rigid transform
        moving = np.array([orig_ca[chain][r] for r in shared])
        target = np.array([model_ca[chain][r] for r in shared])
        rotation, shift = _kabsch(moving, target)
        transferred += [_apply_transform(l, rotation, shift) for l in het_lines]

    out_pdb = Path(out_pdb)
    body = model_text if model_text.rstrip().endswith("END") else model_text
    # Insert warheads before a trailing END, else append.
    lines = [l for l in body.splitlines() if l.strip() != "END"]
    lines += transferred + ["END"]
    out_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_pdb


def run(conf: SimpleNamespace, *, log=print) -> Path:
    """Dock with PyRosetta; write models + score.csv. Returns the output dir.

    With ``dock_proteins_only`` (default), warhead HETATMs are stripped before
    docking (no Rosetta params needed) and carried back onto each model by
    superposition so linkability still works.

    NOTE: the PyRosetta calls are not exercised in CI (licensed dependency);
    verify the mover/API names against your PyRosetta version on first use.
    """
    out_dir = Path(conf.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else Path(conf.workdir) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    original_pdb = Path(conf.input_pdb)
    proteins_only = bool(getattr(conf, "dock_proteins_only", True))
    dock_input = original_pdb
    if proteins_only:
        dock_input = out_dir / "dock_input_proteins.pdb"
        dock_input.write_text(strip_hetatm(original_pdb.read_text(encoding="utf-8", errors="replace")),
                              encoding="utf-8")

    import pyrosetta  # noqa: PLC0415  (lazy: licensed optional dependency)
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.docking import (
        DockingLowRes, DockMCMProtocol, setup_foldtree,
    )
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.rigid import (
        RigidBodyPerturbMover, RigidBodyRandomizeMover,
        partner_downstream, partner_upstream,
    )
    from pyrosetta.rosetta.protocols.simple_moves import (
        ReturnSidechainMover, SwitchResidueTypeSetMover,
    )

    pyrosetta.init(" ".join(init_flags(conf, out_dir)))
    scorefxn_high = pyrosetta.create_score_function("ref2015")
    scorefxn_low = pyrosetta.create_score_function("interchain_cen")
    # A separate constraint-FREE score function for reporting total and I_sc, so
    # the restraint never pollutes the interface measurement (pulling partners
    # apart would otherwise blow up the constraint term and skew I_sc).
    scorefxn_report = pyrosetta.create_score_function("ref2015")
    start = pyrosetta.pose_from_file(str(dock_input))
    movable_jumps = pyrosetta.rosetta.utility.vector1_int()
    movable_jumps.append(1)
    setup_foldtree(start, str(conf.partners), movable_jumps)

    # Residue-residue distance restraint (one residue per partner, on CA by
    # default). Turn on the constraint term in BOTH score functions and re-apply
    # the constraint to the pose after each representation switch (switching clears
    # constraints). This is what actually steers the global search.
    restraint = getattr(conf, "restraint", None)
    if restraint is not None:
        weight = float(getattr(restraint, "weight", 1.0) or 1.0)
        scorefxn_low.set_weight(ScoreType.atom_pair_constraint, weight)
        scorefxn_high.set_weight(ScoreType.atom_pair_constraint, weight)
        atom_name = str(getattr(restraint, "atom", "CA") or "CA")

        def _apply_restraint(pose):
            info = pose.pdb_info()

            def atom_id(res):
                seqpos = info.pdb2pose(str(res.chain), int(res.resseq))
                if seqpos == 0:
                    raise ValueError(f"restraint residue {res.chain}{res.resseq} not in the pose")
                name = atom_name if pose.residue(seqpos).has(atom_name) else "CA"
                return AtomID(pose.residue(seqpos).atom_index(name), seqpos)

            func = HarmonicFunc(float(restraint.distance), float(restraint.stddev))
            pose.add_constraint(AtomPairConstraint(atom_id(restraint.res1), atom_id(restraint.res2), func))
    else:
        def _apply_restraint(pose):
            return None

    relax = FastRelax()
    relax.set_scorefxn(scorefxn_high)
    if bool(getattr(conf, "pre_relax", False)):
        log("Pre-relaxing the input structure (FastRelax)…")
        relax.apply(start)   # relax once; sidechains recovered from this below

    to_centroid = SwitchResidueTypeSetMover("centroid")
    to_fullatom = SwitchResidueTypeSetMover("fa_standard")
    recover_sidechains = ReturnSidechainMover(start)
    dock_lowres = DockingLowRes(scorefxn_low, movable_jumps)
    dock_highres = DockMCMProtocol(1, scorefxn_high, scorefxn_high)
    interface = InterfaceAnalyzerMover(1)
    interface.set_scorefunction(scorefxn_report)   # constraint-free
    is_global = str(conf.mode) == "global"
    if restraint is not None:
        log(f"Distance restraint: {restraint.res1.chain}{restraint.res1.resseq}–"
            f"{restraint.res2.chain}{restraint.res2.resseq} target {restraint.distance} A "
            f"(±{restraint.stddev}, weight {getattr(restraint, 'weight', 1.0)})")
    post_relax = bool(getattr(conf, "post_relax", False))

    rows = []
    for i in range(1, int(conf.nstruct) + 1):
        pose = start.clone()
        # Low-resolution (centroid) global search: randomize, then clash-tolerant
        # rigid-body docking. This is the stage that finds real bound arrangements.
        to_centroid.apply(pose)
        _apply_restraint(pose)        # constraint steers the (centroid) global search
        if is_global:
            RigidBodyRandomizeMover(pose, 1, partner_upstream).apply(pose)
            RigidBodyRandomizeMover(pose, 1, partner_downstream).apply(pose)
        else:
            RigidBodyPerturbMover(1, 8.0, 3.0).apply(pose)
        dock_lowres.apply(pose)
        # Back to full atom + restore the original side chains, then refine.
        to_fullatom.apply(pose)
        recover_sidechains.apply(pose)
        _apply_restraint(pose)        # re-add (the rep switch cleared constraints)
        dock_highres.apply(pose)
        if post_relax:
            relax.apply(pose)    # full-atom relax of the docked model

        total = float(scorefxn_report(pose))       # constraint-free energy
        try:
            interface.apply(pose)
            i_sc = float(interface.get_interface_dG())
        except Exception:
            i_sc = float("nan")
        model_pdb = out_dir / f"model_{i:04d}.pdb"
        pose.dump_pdb(str(model_pdb))
        if proteins_only:
            transfer_ligands(original_pdb, model_pdb, model_pdb)  # add warheads back
        rows.append((model_pdb.name, total, i_sc))
        log(f"[{i}/{conf.nstruct}] {model_pdb.name} total={total:.2f} I_sc={i_sc:.2f}")

    import csv

    with (out_dir / "score.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "total_score", "I_sc"])
        # Rank by interface score (I_sc); nan sinks to the bottom.
        writer.writerows(sorted(rows, key=lambda r: (r[2] != r[2], r[2])))
    log(f"Done. Models + score.csv in {out_dir} (rank by I_sc)")
    return out_dir


def default_yaml_text() -> str:
    return """# ChemFlow -> PyRosetta: in-process protein-protein docking for the
# PROTAC ternary search. Install with: conda install pyrosetta (needs a license).
#   chemflow protac pyrosetta-dock validate pyrosetta_dock.yaml
#   chemflow protac pyrosetta-dock run      pyrosetta_dock.yaml

out_dir: pyrosetta_dock_output
input_pdb: ternary_partners.pdb   # both partners as chains (see: chemflow protac merge-pdb)
partners: "A_B"                    # chain A docks against chain B
mode: global                      # global = search; local = refine a starting pose
nstruct: 100
# seed: 42

# Dock the proteins only (default): warhead HETATMs are stripped before docking
# so NO Rosetta ligand .params are needed, then carried back onto each model by
# superposition so linkability still works. Set false only if you have .params.
dock_proteins_only: true

# FastRelax (slow). pre_relax relaxes the input ONCE before docking (removes input
# strain so total scores are sensible); post_relax relaxes every docked model
# (cleaner totals/I_sc, much slower). Both off by default.
pre_relax: false
post_relax: false

# Only used when dock_proteins_only is false (then .params per warhead are
# required; PyRosetta does not ship molfile_to_params.py, so this needs full Rosetta).
# ligand_params:
#   - LIG.params
#   - VH1.params

# Optional distance restraint between two PROTEIN residues (one per partner), on
# their CA atoms, to steer docking toward a chosen arrangement. Works in
# proteins-only mode (no ligand params). Pick a residue on each protein near its
# warhead / the expected interface.
# restraint:
#   res1: {chain: A, resseq: 123}
#   res2: {chain: B, resseq: 456}
#   distance: 10.0
#   stddev: 2.0
#   weight: 1.0

# After it runs: rank score.csv by I_sc, then: chemflow protac linkability
"""
