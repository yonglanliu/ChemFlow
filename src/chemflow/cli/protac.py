"""``chemflow protac`` — PROTAC-specific workflows.

Ternary search via RosettaDock / PyRosetta (linker-aware restraint), linkability,
structure fixing, merge, screen, and RBFE setup.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _protac(args) -> int:
    try:
        return args.protac_func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


# --- RosettaDock (linker-free protein-protein search) -------------------------

def _rd_generate(args) -> int:
    from chemflow.protac.rosettadock import default_yaml_text

    output = Path(args.output).expanduser()
    if output.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {output} (use --force)", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(default_yaml_text(), encoding="utf-8")
    print(f"Wrote example RosettaDock config: {output}")
    return 0


def _rd_validate(args) -> int:
    from chemflow.protac.rosettadock import load_config, validate_config

    validate_config(load_config(args.config))
    print(f"Valid RosettaDock configuration: {args.config}")
    return 0


def _rd_stage(args) -> int:
    from chemflow.protac.rosettadock import load_config, stage, validate_config

    conf = load_config(args.config)
    validate_config(conf)
    runner, out_dir = stage(conf)
    print(f"Wrote runner: {runner}\nOutput dir: {out_dir}")
    return 0


def _rd_run(args) -> int:
    import shlex
    import subprocess

    from chemflow.protac.rosettadock import build_command, load_config, stage, validate_config

    conf = load_config(args.config)
    validate_config(conf)
    runner, out_dir = stage(conf)
    command = build_command(conf, out_dir)
    if args.dry_run:
        print(" ".join(shlex.quote(p) for p in command))
        return 0
    print("Running: " + " ".join(shlex.quote(p) for p in command))
    try:
        rc = subprocess.run(command, check=False).returncode
    except FileNotFoundError:
        print(f"Rosetta executable not found: {command[0]!r}. Install Rosetta and set "
              "rosetta_command.", file=sys.stderr)
        return 127
    if rc == 0:
        print(f"  -> models + score.sc in: {out_dir}")
        print("Next: rank + check linkability: chemflow protac screen …")
    return rc


# --- Linkability (rank a candidate complex before designing a linker) ---------

def _linkability(args) -> int:
    from chemflow.protac.linkability import assess_from_files

    result = assess_from_files(
        args.head1_file, args.atom1, args.head2_file, args.atom2,
        protein=args.protein, min_dist=args.min_dist, max_dist=args.max_dist,
    )
    print(f"attachment-atom distance: {result.distance:.2f} A")
    print(f"exit vectors point toward partner: head1 {result.toward1:+.2f}, head2 {result.toward2:+.2f} "
          "(cosine; positive = facing)")
    print(f"verdict: {result.verdict.upper()}" + (f" — {result.reason}" if result.reason else ""))
    print("\nLinkable complexes are good candidates to design a linker for: "
          "chemflow generate reinvent linker …")
    return 0 if result.linkable else 2


def add_protac_parser(subparsers) -> None:
    parser = subparsers.add_parser("protac", help="PROTAC workflows (ternary search/docking, ...)")
    groups = parser.add_subparsers(dest="protac_group", required=True)

    _add_rosettadock(groups)
    _add_pyrosetta_dock(groups)
    _add_screen(groups)
    _add_linkability(groups)
    _add_rbfe_setup(groups)
    _add_fix_structure(groups)
    _add_ligand_params(groups)
    _add_merge_pdb(groups)


# --- RBFE setup (linker designer output -> AlchemForge ligand-RBFE) ------------

def _rbfe_setup(args) -> int:
    from chemflow.protac.rbfe_setup import prepare_rbfe

    result = prepare_rbfe(args.poses, args.receptor, args.out, name=args.name)
    print(f"Prepared RBFE project with {result['n']} posed PROTACs:")
    print(f"  ligands: {result['ligands']}")
    print(f"  config:  {result['toml']}")
    print("\nReview the TOML (force field, charges, lambda windows), then run under AlchemForge:")
    print(f"  alchemforge rbfe all {result['toml']}        # fix-pdb -> charge -> plan -> stage + SLURM")
    print("  (triage first with MM-GBSA: alchemforge mmgbsa batch <md_runs_dir> --ligand-selection ...)")
    return 0


def _add_rbfe_setup(groups) -> None:
    rs = groups.add_parser(
        "rbfe-setup",
        help="Turn the linker designer's posed PROTAC SDF into an AlchemForge RBFE project",
    )
    rs.add_argument("--poses", required=True,
                    help="protac_poses.sdf exported by the linker designer (3D-posed PROTACs)")
    rs.add_argument("--receptor", required=True, help="receptor / ternary-complex PDB")
    rs.add_argument("--out", required=True, help="output RBFE project directory")
    rs.add_argument("--name", default="protac_rbfe", help="project name (default protac_rbfe)")
    rs.set_defaults(protac_func=_rbfe_setup, func=_protac)


def _fix_structure(args) -> int:
    from chemflow.protac.fix_structure import find_gaps, fix_with_pdbfixer, kic_refine, parse_loop_spec

    if args.list_gaps:
        gaps = find_gaps(Path(args.input).expanduser().read_text(errors="replace"))
        if not gaps:
            print("No internal chain-numbering gaps found (terminal gaps aren't detectable this way).")
            return 0
        print("Internal gaps (chain: after-before = length):")
        for g in gaps:
            print(f"  {g.chain}: {g.after_resseq}-{g.before_resseq}  (missing {g.length})")
        return 0

    if args.method == "pdbfixer":
        sequences = None
        if args.fasta or args.uniprot or args.sequence:
            from chemflow.protac.fix_structure import pdb_chains, resolve_sequences

            chains = pdb_chains(Path(args.input).expanduser().read_text(errors="replace"))
            sequences = resolve_sequences(args.fasta, args.uniprot, chains, args.sequence)
        out = fix_with_pdbfixer(
            args.input, args.output,
            keep_only_internal=not args.include_termini,
            max_gap=args.max_gap, add_hydrogens=args.add_hydrogens,
            sequences=sequences)
        print(f"PDBFixer wrote: {out}")
    else:  # kic
        from chemflow.protac.fix_structure import (
            fetch_uniprot_sequence, kic_build_missing, kic_refine, parse_loop_spec,
            read_fasta_records,
        )

        if args.fasta or args.uniprot or args.sequence:
            # build missing-density residues; align the structure to the best sequence
            candidates: list[str] = []
            for spec in args.sequence:
                candidates.append((spec.split("=", 1)[1] if "=" in spec else spec).upper())
            for spec in args.fasta:
                path = spec.split("=", 1)[1] if "=" in spec else spec
                candidates += [seq for _h, seq in read_fasta_records(path)]
            for spec in args.uniprot:
                acc = spec.split("=", 1)[1] if "=" in spec else spec
                candidates.append(fetch_uniprot_sequence(acc))
            out = kic_build_missing(args.input, args.output, candidates, chain=args.chain)
            print(f"Rosetta KIC built missing residues -> {out}")
        else:
            loops = [parse_loop_spec(s) for s in (args.loop or [])]
            if not loops:
                print("error: KIC needs either --fasta/--uniprot (build missing residues) "
                      "or --loop CHAIN:START-END (refine existing loops)", file=sys.stderr)
                return 1
            out = kic_refine(args.input, args.output, loops)
            print(f"Rosetta KIC refined loops -> {out}")
    print("Use the fixed structure as a docking input (merge-pdb -> pyrosetta-dock).")
    return 0


def _add_fix_structure(groups) -> None:
    fx = groups.add_parser(
        "fix-structure",
        help="Fill missing atoms/residues (PDBFixer) or rebuild loops (Rosetta KIC)",
    )
    fx.add_argument("--input", required=True, help="input PDB")
    fx.add_argument("-o", "--output", default="fixed.pdb")
    fx.add_argument("--method", choices=["pdbfixer", "kic"], default="pdbfixer")
    fx.add_argument("--list-gaps", action="store_true",
                    help="just report internal chain gaps and exit")
    # pdbfixer options
    fx.add_argument("--include-termini", action="store_true",
                    help="also model missing terminal residues (default: internal only)")
    fx.add_argument("--max-gap", type=int, default=None,
                    help="skip internal gaps longer than N residues (crude loops)")
    fx.add_argument("--add-hydrogens", action="store_true", help="add hydrogens (pdbfixer)")
    # reference sequence to fill missing residues when the PDB has no SEQRES
    fx.add_argument("--sequence", action="append", default=[],
                    help="paste a reference sequence: CHAIN=SEQ (or just SEQ if one chain)")
    fx.add_argument("--fasta", action="append", default=[],
                    help="reference sequence file to fill gaps: CHAIN=FILE (or just FILE if one chain)")
    fx.add_argument("--uniprot", action="append", default=[],
                    help="reference sequence by UniProt accession: CHAIN=ACC (or just ACC if one chain)")
    # kic options
    fx.add_argument("--loop", action="append", default=[],
                    help="loop to rebuild as CHAIN:START-END (repeatable; kic refine-only)")
    fx.add_argument("--chain", default=None,
                    help="which chain to build (kic; required only if the file has several)")
    fx.set_defaults(protac_func=_fix_structure, func=_protac)


def _screen(args) -> int:
    from chemflow.protac.screen import screen_models

    rows = screen_models(args.dir, args.atom1, args.atom2,
                         top=args.top, min_dist=args.min_dist, max_dist=args.max_dist,
                         models_dir=args.models_dir, linkable_dir=args.linkable_dir)
    print(f"{'model':<18}{'I_sc':>10}{'dist(A)':>10}  verdict")
    for r in rows:
        isc = "n/a" if r.i_sc is None else f"{r.i_sc:.2f}"
        dist = "n/a" if r.distance is None else f"{r.distance:.2f}"
        print(f"{r.model:<18}{isc:>10}{dist:>10}  {r.verdict}")
    linkable = [r for r in rows if r.verdict == "linkable"]
    base = Path(args.dir).expanduser()
    base_dir = base if base.is_dir() else base.parent
    link_dir = Path(args.linkable_dir).expanduser() if args.linkable_dir else base_dir / "linkable"
    print(f"\n{len(linkable)} of {len(rows)} models are linkable.")
    if linkable:
        print(f"Linkable model PDBs copied to: {link_dir}")
    print("Design linkers for those: chemflow generate reinvent linker …")
    return 0


def _add_screen(groups) -> None:
    sc = groups.add_parser(
        "screen",
        help="Rank docked models by I_sc and check warhead linkability in one step",
    )
    sc.add_argument("--dir", required=True,
                    help="output dir OR a score.csv path (with model_*.pdb alongside)")
    sc.add_argument("--models-dir", default=None,
                    help="where the model PDBs live, if not next to score.csv")
    sc.add_argument("--linkable-dir", default=None,
                    help="folder to copy the linkable model PDBs into (default: <dir>/linkable)")
    sc.add_argument("--atom1", required=True,
                    help="target warhead attachment atom as RESNAME:ATOMNAME (e.g. LIG:C7)")
    sc.add_argument("--atom2", required=True,
                    help="E3 warhead attachment atom as RESNAME:ATOMNAME (e.g. VH1:C3)")
    sc.add_argument("--top", type=int, default=20, help="screen the top N models by I_sc")
    sc.add_argument("--min-dist", type=float, default=3.0)
    sc.add_argument("--max-dist", type=float, default=20.0)
    sc.set_defaults(protac_func=_screen, func=_protac)


def _prd_generate(args) -> int:
    from chemflow.protac.pyrosetta_dock import default_yaml_text

    output = Path(args.output).expanduser()
    if output.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {output} (use --force)", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(default_yaml_text(), encoding="utf-8")
    print(f"Wrote example PyRosetta dock config: {output}")
    return 0


def _prd_validate(args) -> int:
    from chemflow.protac.pyrosetta_dock import load_config, validate_config

    validate_config(load_config(args.config))
    print(f"Valid PyRosetta dock configuration: {args.config}")
    return 0


def _prd_run(args) -> int:
    from chemflow.protac.pyrosetta_dock import load_config, run, validate_config

    conf = load_config(args.config)
    validate_config(conf)
    try:
        import pyrosetta  # noqa: F401
    except ImportError:
        print("PyRosetta is not installed. Install it with `conda install pyrosetta` "
              "(needs a free academic license from rosettacommons.org).", file=sys.stderr)
        return 127
    out_dir = run(conf)
    print(f"  -> models + score.csv in: {out_dir}")
    print("Next: rank + check linkability: chemflow protac screen …")
    return 0


def _add_pyrosetta_dock(groups) -> None:
    pd = groups.add_parser(
        "pyrosetta-dock",
        help="In-process protein-protein docking with PyRosetta (ternary search; "
             "optional linker-aware restraint)",
    )
    sub = pd.add_subparsers(dest="pyrosetta_dock_command", required=True)
    gen = sub.add_parser("generate", help="Write a commented example config")
    gen.add_argument("-o", "--output", default="pyrosetta_dock.yaml")
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(protac_func=_prd_generate, func=_protac)
    for name, fn, help_text in (
        ("validate", _prd_validate, "Validate a config (no PyRosetta needed)"),
        ("run", _prd_run, "Run docking with PyRosetta"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config")
        p.set_defaults(protac_func=fn, func=_protac)


def _merge_pdb(args) -> int:
    from chemflow.protac.merge_pdb import merge_pdbs

    result = merge_pdbs(args.pdb1, args.pdb2, args.output,
                        chain1=args.chain1, chain2=args.chain2,
                        drop_water=not args.keep_water)
    print(f"merged PDB: {result['out']}")
    print(f"partners: {result['partners']}  "
          f"(chain {result['chain1']} = {result['n_atoms1']} atoms, "
          f"chain {result['chain2']} = {result['n_atoms2']} atoms)")
    print(f"\nUse it in RosettaDock: input_pdb = {result['out']}, partners = {result['partners']}")
    return 0


def _add_merge_pdb(groups) -> None:
    mp = groups.add_parser(
        "merge-pdb",
        help="Merge two binary complexes into one two-chain PDB for docking",
    )
    mp.add_argument("--pdb1", required=True, help="target protein + warhead (PDB)")
    mp.add_argument("--pdb2", required=True, help="E3 ligase + warhead (PDB)")
    mp.add_argument("-o", "--output", default="ternary_partners.pdb")
    mp.add_argument("--chain1", default="A", help="chain id for pdb1 (default A)")
    mp.add_argument("--chain2", default="B", help="chain id for pdb2 (default B)")
    mp.add_argument("--keep-water", action="store_true", help="keep crystallographic waters")
    mp.set_defaults(protac_func=_merge_pdb, func=_protac)


def _ligand_params(args) -> int:
    from chemflow.protac.ligand_params import generate_params

    specs = list(args.ligand or [])
    if args.input and args.name:        # backward-compatible single form
        specs.append(f"{args.name}={args.input}")
    if not specs:
        print("error: give --ligand NAME=STRUCTURE (repeatable), or --name + --input",
              file=sys.stderr)
        return 1
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--ligand expects NAME=STRUCTURE, got {spec!r}")
        name, path = spec.split("=", 1)
        result = generate_params(path.strip(), name.strip(), args.out_dir,
                                 molfile_to_params=args.molfile_to_params, python=args.python)
        print(f"{result['name']}: params {result['params']}"
              + (f", renamed PDB {result['pdb']}" if result.get("pdb") else ""))
    print("\nOptional step: only needed if RosettaDock includes the warheads. "
          "Pass the .params files via ligand_params, and use the renamed PDB coordinates "
          "in your docking PDB.")
    return 0


def _add_ligand_params(groups) -> None:
    lp = groups.add_parser(
        "ligand-params",
        help="(optional) make Rosetta .params for one or both warheads via molfile_to_params.py",
    )
    lp.add_argument("--ligand", action="append", default=[],
                    help="warhead as NAME=STRUCTURE (e.g. LIG=target_warhead.sdf); repeat for both")
    lp.add_argument("--input", help="single-warhead form: structure (.sdf/.mol, 3D + explicit H)")
    lp.add_argument("--name", help="single-warhead form: 3-letter residue name, e.g. LIG")
    lp.add_argument("-o", "--out-dir", default="ligand_params")
    lp.add_argument("--molfile-to-params", default=None,
                    help="path to Rosetta's molfile_to_params.py "
                         "(default: found via $ROSETTA / $ROSETTA3 / $MOLFILE_TO_PARAMS or PATH)")
    lp.add_argument("--python", default=None, help="python to run the script with (default: current)")
    lp.set_defaults(protac_func=_ligand_params, func=_protac)


def _add_rosettadock(groups) -> None:
    rd = groups.add_parser("rosettadock",
                           help="Linker-free protein-protein docking for the ternary search")
    sub = rd.add_subparsers(dest="rosettadock_command", required=True)

    gen = sub.add_parser("generate", help="Write a commented example config")
    gen.add_argument("-o", "--output", default="rosettadock.yaml")
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(protac_func=_rd_generate, func=_protac)

    for name, func, help_text in (
        ("validate", _rd_validate, "Validate a config (no Rosetta needed)"),
        ("stage", _rd_stage, "Write the runner without running"),
        ("run", _rd_run, "Run RosettaDock"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config")
        if name == "run":
            p.add_argument("--dry-run", action="store_true",
                           help="Write runner and print the command without running")
        p.set_defaults(protac_func=func, func=_protac)


def _add_linkability(groups) -> None:
    lk = groups.add_parser(
        "linkability",
        help="Is a candidate complex linkable? (exit-vector distance/orientation of two warheads)",
    )
    lk.add_argument("--head1-file", required=True, help="posed target warhead (SDF/MOL/PDB)")
    lk.add_argument("--atom1", type=int, required=True, help="attachment atom index on head1")
    lk.add_argument("--head2-file", required=True, help="posed E3 warhead (SDF/MOL/PDB)")
    lk.add_argument("--atom2", type=int, required=True, help="attachment atom index on head2")
    lk.add_argument("--protein", action="append", default=[],
                    help="protein PDB(s) (optional; for picking exposed exit vectors)")
    lk.add_argument("--min-dist", type=float, default=3.0)
    lk.add_argument("--max-dist", type=float, default=20.0)
    lk.set_defaults(protac_func=_linkability, func=_protac)

