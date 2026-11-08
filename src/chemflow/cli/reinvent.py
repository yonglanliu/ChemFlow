"""``chemflow generate reinvent`` — REINVENT4 generative design.

    chemflow generate reinvent generate [--model M] [--run-type T] -o reinvent.yaml
    chemflow generate reinvent validate reinvent.yaml
    chemflow generate reinvent stage    reinvent.yaml     # write the REINVENT TOML
    chemflow generate reinvent run      reinvent.yaml [--dry-run]
    chemflow generate reinvent export   results.csv --top 100 [--sdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

from chemflow.generative.reinvent.config import (
    SUPPORTED_MODELS,
    SUPPORTED_RUN_TYPES,
    default_yaml_text,
    load_config,
    validate_config,
)


def _generate(args) -> int:
    text = default_yaml_text(args.model, args.run_type)
    output = Path(args.output).expanduser()
    if output.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {output} (use --force)", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote example REINVENT config: {output}")
    return 0


def _validate(args) -> int:
    validate_config(load_config(args.config))
    print(f"Valid REINVENT configuration: {args.config}")
    return 0


def _stage(args) -> int:
    from chemflow.generative.reinvent.run import stage_inputs

    conf = load_config(args.config)
    validate_config(conf)
    toml_path, _ = stage_inputs(conf)
    print(f"Wrote REINVENT TOML: {toml_path}")
    return 0


def _run(args) -> int:
    from chemflow.generative.reinvent.run import run_from_config

    return run_from_config(Path(args.config), dry_run=args.dry_run)


def _export(args) -> int:
    from chemflow.generative.reinvent.export import export_results

    result = export_results(
        args.csv, args.output, top=args.top, min_score=args.min_score, sdf=args.sdf
    )
    print(f"Exported {result['count']} molecules:")
    for key in ("smi", "csv", "sdf"):
        if key in result:
            print(f"  {key}: {result[key]}")
    if args.sdf:
        print(f"  3D-embedded: {result['sdf_count']}/{result['count']}")
    return 0


def _load_head_molecule(file_arg, smiles_arg, label):
    from chemflow.generative.reinvent.linker.heads import load_molecule, mol_from_smiles

    if file_arg and smiles_arg:
        raise ValueError(f"give either --{label}-file or --{label}-smiles, not both")
    if file_arg:
        return load_molecule(file_arg)
    if smiles_arg:
        return mol_from_smiles(smiles_arg)
    raise ValueError(f"--{label}-file or --{label}-smiles is required")


def _print_atoms(mol, label):
    print(f"{label}: {mol.GetNumAtoms()} atoms (RDKit indices; pick one that carries a hydrogen)")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        print(f"  {atom.GetIdx():4d}  {atom.GetSymbol():<2}  H={atom.GetTotalNumHs()}  "
              f"neighbours={[n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]}")


def _parse_model_spec(spec: str) -> dict:
    """``NAME=PATH`` or ``NAME=PATH:COLUMN`` -> {"name", "path", "column"}."""
    if "=" not in spec:
        raise ValueError(f"--model expects NAME=PATH[:COLUMN], got {spec!r}")
    name, rest = spec.split("=", 1)
    path, column = rest, "prediction"
    head, sep, tail = rest.rpartition(":")
    if sep and head and "/" not in tail and "\\" not in tail:
        path, column = head, tail
    return {"name": name.strip(), "path": path.strip(), "column": column.strip()}


def _linker(args) -> int:
    # Report expected user mistakes (bad path, bad atom, REINVENT failure) as a
    # single clean line instead of a Python traceback.
    try:
        return _linker_impl(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _linker_impl(args) -> int:
    from chemflow.generative.reinvent.linker.designer import (
        DesignOptions, design_linkers, export_design, table_columns, table_rows, write_warheads,
    )
    from chemflow.generative.reinvent.linker.fit3d import FitContext, exit_clearance, read_pdb_heavy_atoms
    from chemflow.generative.reinvent.linker.heads import make_head

    mol1 = _load_head_molecule(args.head1_file, args.head1_smiles, "head1")
    mol2 = _load_head_molecule(args.head2_file, args.head2_smiles, "head2")
    if args.list_atoms:
        _print_atoms(mol1, "head1")
        _print_atoms(mol2, "head2")
        return 0
    if args.atom1 is None or args.atom2 is None:
        raise ValueError("--atom1 and --atom2 are required (use --list-atoms to see the indices)")

    protein = read_pdb_heavy_atoms(args.protein) if args.protein else None
    head1 = make_head(mol1, args.atom1, name="head1", protein_xyz=protein)
    head2 = make_head(mol2, args.atom2, name="head2", protein_xyz=protein)
    print(f"head1 warhead: {head1.smiles}")
    print(f"head2 warhead: {head2.smiles}")
    for head in (head1, head2):
        clearance = exit_clearance(head, protein)
        if clearance is not None:
            print(f"  {head.name} exit-vector clearance: {clearance:.2f} A "
                  f"({'open' if clearance >= 3.0 else 'CROWDED: the linker would leave into the protein'})")

    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.dry_run:
        path = write_warheads(head1, head2, out_dir / "warheads.smi")
        print(f"Wrote LinkInvent input: {path}")
        return 0

    fit = None
    if args.fit_top > 0:
        if args.separate_frames:
            print("3D fit skipped: --separate-frames means the heads share no coordinate frame.")
        elif not (head1.has_3d and head2.has_3d):
            print("3D fit skipped: it needs 3D coordinates for both heads (give --head1-file/--head2-file).")
        else:
            fit = FitContext(head1=head1, head2=head2, protein_xyz=protein, same_frame=True)

    models = [_parse_model_spec(spec) for spec in (args.model or [])]
    if args.library:
        from chemflow.generative.reinvent.linker.library import design_from_library

        print(f"Searching linker library: {args.library}")
        candidates = design_from_library(
            head1, head2, args.library, models=models, fit=fit, fit_top=args.fit_top,
            min_length=args.min_length, max_length=args.max_length,
        )
        if not candidates:
            print("No PROTACs could be assembled from the library for these warheads.")
            return 1
    else:
        if not args.prior:
            raise ValueError("--prior (path to linkinvent.prior) is required, or use "
                             "--library to search a linker library instead")
        opts = DesignOptions(
            reinvent_command=args.reinvent_command, prior_file=args.prior, workdir=str(out_dir),
            out_dir=str(out_dir), num_smiles=args.num, device=args.device, seed=args.seed,
        )
        candidates = design_linkers(head1, head2, opts, models=models, fit=fit, fit_top=args.fit_top)
        if not candidates:
            print("REINVENT4 returned no valid linkers.")
            return 1
    files = export_design(candidates, out_dir, with_fit=fit is not None)

    columns = table_columns(candidates, with_fit=fit is not None)[:8]
    print("\n" + "  ".join(["id", "NLL"] + [label for _k, label, _f in columns]))
    for row in table_rows(candidates[:10], columns):
        print("  ".join([row[0], row[2]] + row[3:]))
    print(f"\n{len(candidates)} linkers written:")
    for key, path in files.items():
        print(f"  {key}: {path}")
    return 0


def _prepare(args) -> int:
    from chemflow.generative.reinvent.linker.pdb_prep import list_ligands, prepare_complex

    if not args.pdb and not args.pdb_file:
        print("error: give --pdb <ID> or --pdb-file <FILE>", file=sys.stderr)
        return 1
    if args.list:
        from chemflow.generative.reinvent.linker.pdb_prep import fetch_pdb

        path = args.pdb_file or fetch_pdb(args.pdb, args.out_dir)
        records = list_ligands(path, include_ions=args.include_ions)
        if not records:
            print("No non-water ligands found.")
            return 0
        print(f"Ligands in {path} (largest first — the PROTAC is usually biggest):")
        for rec in records:
            print(f"  {rec.label}")
        print("\nExtract with: --ligand RESNAME[:CHAIN] (repeat for both warheads)")
        return 0

    result = prepare_complex(
        pdb_id=args.pdb, pdb_file=args.pdb_file, out_dir=args.out_dir,
        ligands=args.ligand, keep_hetatm=args.keep_hetatm, use_templates=not args.no_template,
    )
    if not args.ligand:
        print(f"Downloaded: {result['pdb']}")
        print("Available ligands:")
        for rec in result["available"]:
            print(f"  {rec.label}")
        print("\nRe-run with --ligand RESNAME[:CHAIN] for each warhead to extract them + protein.")
        return 0
    print(f"protein: {result['protein']}")
    for path in result["ligands"]:
        print(f"ligand:  {path}")
    print("\nLoad these in the designer: Load Ligand 1 / Ligand 2 (the SDFs) + Load Protein (protein.pdb).")
    print("Check the extracted ligand structures before use (PDB coordinates carry no bond orders).")
    return 0


def add_reinvent_parser(generate_subparsers) -> None:
    parser = generate_subparsers.add_parser(
        "reinvent", help="REINVENT4 generative design (de novo, scaffold, linker, mol2mol)"
    )
    sub = parser.add_subparsers(dest="reinvent_command", required=True)

    gen = sub.add_parser("generate", help="Write a commented example config")
    gen.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default="reinvent")
    gen.add_argument("--run-type", choices=sorted(SUPPORTED_RUN_TYPES), default="sampling")
    gen.add_argument("-o", "--output", default="reinvent.yaml")
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(func=_generate)

    for name, func, help_text in (
        ("validate", _validate, "Validate a config (no REINVENT needed)"),
        ("stage", _stage, "Write the REINVENT TOML without running"),
        ("run", _run, "Run REINVENT4"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config")
        if name == "run":
            p.add_argument("--dry-run", action="store_true",
                           help="Write the TOML and print the command only")
        p.set_defaults(func=func)

    exp = sub.add_parser("export", help="Select top molecules; write .smi/.csv (and 3D .sdf)")
    exp.add_argument("csv", help="REINVENT output CSV")
    exp.add_argument("-o", "--output", default=None, help="Output prefix (default: <csv>_top)")
    exp.add_argument("--top", type=int, default=100)
    exp.add_argument("--min-score", type=float, default=None)
    exp.add_argument("--sdf", action="store_true", help="Also write 3D-embedded SDF (needs RDKit)")
    exp.set_defaults(func=_export)

    lk = sub.add_parser(
        "linker",
        help="PROTAC linker design: pick one atom on each head, LinkInvent proposes scored linkers",
    )
    for n in ("1", "2"):
        lk.add_argument(f"--head{n}-file", help=f"head {n} structure (.sdf/.mol/.mol2/.pdb; 3D enables the fit check)")
        lk.add_argument(f"--head{n}-smiles", help=f"head {n} as a SMILES (2D)")
        lk.add_argument(f"--atom{n}", type=int, help=f"atom index on head {n} where the linker attaches")
    lk.add_argument("--list-atoms", action="store_true", help="print atom indices for both heads and exit")
    lk.add_argument("--protein", action="append", default=[],
                    help="protein PDB (repeat for several); used for exit-vector clearance and clash checks")
    lk.add_argument("--separate-frames", action="store_true",
                    help="the two heads come from different structures (no 3D fit possible)")
    lk.add_argument("--prior", help="path to linkinvent.prior")
    lk.add_argument("--library", help="search a linker library file (.smi/.csv/.sdf of bifunctional "
                                      "linkers, e.g. Enamine) instead of generating with LinkInvent")
    lk.add_argument("--min-length", type=int, default=None,
                    help="library search: minimum linker length (bonds between attachment points)")
    lk.add_argument("--max-length", type=int, default=None,
                    help="library search: maximum linker length (bonds between attachment points)")
    lk.add_argument("--reinvent-command", default="reinvent", help="full path to the reinvent executable")
    lk.add_argument("--device", default="cpu")
    lk.add_argument("--num", type=int, default=200, help="linkers to sample")
    lk.add_argument("--seed", type=int, default=42)
    lk.add_argument("--model", action="append", default=[],
                    help="ChemFlow model score for each PROTAC: NAME=PATH[:COLUMN] (repeatable)")
    lk.add_argument("--fit-top", type=int, default=50,
                    help="run the 3D fit on the N best linkers by NLL (0 disables; it is the slow step)")
    lk.add_argument("-o", "--out-dir", default="linker_design")
    lk.add_argument("--dry-run", action="store_true", help="only write the LinkInvent warhead input")
    lk.set_defaults(func=_linker)

    prep = sub.add_parser(
        "prepare",
        help="Download a PDB complex and split it into protein.pdb + ligand SDFs for the designer",
    )
    prep.add_argument("--pdb", help="4-character PDB id to download from RCSB (e.g. 5T35)")
    prep.add_argument("--pdb-file", help="use a local PDB file instead of downloading")
    prep.add_argument("-o", "--out-dir", default="complex_prep")
    prep.add_argument("--list", action="store_true", help="list the ligands in the structure and exit")
    prep.add_argument("--include-ions", action="store_true", help="also list common ions/buffers")
    prep.add_argument("--ligand", action="append", default=[],
                      help="ligand to extract as RESNAME or RESNAME:CHAIN (repeat for both warheads)")
    prep.add_argument("--keep-hetatm", action="store_true",
                      help="keep metals/cofactors (non-water HETATM) in protein.pdb")
    prep.add_argument("--no-template", action="store_true",
                      help="skip downloading the ideal-SDF bond-order template (faster, less accurate)")
    prep.set_defaults(func=_prepare)
