"""Repair OpenFE transformation JSONs with mismatched cofactor charges.

OpenFE's RFE ``_validate_smcs`` rejects graph-isomorphic ``SmallMoleculeComponent``
objects whose partial charges differ ("Found SmallMoleculeComponents that are
isomorphic but with different charges"). This happens when an edge JSON was
generated before charge harmonization was applied: each cofactor copy carries its
own (conformer-dependent) AM1-BCC charges.

Rather than regenerating the whole ligand network, this tool loads an existing
transformation JSON, harmonizes the partial charges across isomorphic **cofactor**
copies in both end states (leaving the alchemical ligand / protein / solvent
untouched, so the atom mapping stays valid), and rewrites the JSON.

    # one edge, in place (keeps a .bak)
    python -m chemflow.fe.openfe.utils.repair_json edge.json

    # one edge, to a new file
    python -m chemflow.fe.openfe.utils.repair_json edge.json -o edge.fixed.json

    # every *_complex.json under a json/ directory, in place
    python -m chemflow.fe.openfe.utils.repair_json /path/to/job/json --glob '*_complex.json'
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

from chemflow.fe.openfe.utils.partial_charges import harmonize_isomorphic_charges


def _is_cofactor_key(key: str) -> bool:
    return "cofactor" in str(key).lower()


def repair_transformation(in_path, out_path=None, *, backup: bool = True) -> bool:
    """Harmonize isomorphic cofactor charges in one transformation JSON.

    Returns True if the file was changed (cofactors harmonized), False if there
    was nothing to do. Only components whose state key contains "cofactor" are
    touched; the alchemical ligand, protein and solvent are preserved by object
    identity so the transformation's mapping remains valid.
    """
    import openfe  # noqa: PLC0415  (ChemicalSystem / SmallMoleculeComponent)
    from gufe import Transformation  # noqa: PLC0415

    in_path = pathlib.Path(in_path)
    trans = Transformation.from_json(in_path)
    state_a, state_b = trans.stateA, trans.stateB

    # Collect the cofactor components across both states (dedupe by identity,
    # preserve order) so isomorphic copies are harmonized together.
    cofactors: list = []
    seen: set[int] = set()
    for state in (state_a, state_b):
        for key, comp in state.components.items():
            if _is_cofactor_key(key) and id(comp) not in seen:
                seen.add(id(comp))
                cofactors.append(comp)

    if len(cofactors) < 2:
        print(f"{in_path.name}: {len(cofactors)} cofactor(s); nothing to harmonize.")
        return False

    harmonized = harmonize_isomorphic_charges(cofactors, molecule_kind="cofactor")
    replacement = {id(orig): new for orig, new in zip(cofactors, harmonized)}
    if all(replacement[k] is orig for k, orig in zip((id(c) for c in cofactors), cofactors)):
        print(f"{in_path.name}: cofactors already consistent; no change.")
        return False

    def rebuild(state):
        new_components = {}
        for key, comp in state.components.items():
            repl = replacement.get(id(comp))
            new_components[key] = repl if (repl is not None) else comp
        return openfe.ChemicalSystem(new_components, name=state.name)

    new_trans = Transformation(
        stateA=rebuild(state_a),
        stateB=rebuild(state_b),
        protocol=trans.protocol,
        mapping=trans.mapping,
        name=trans.name,
    )

    out_path = pathlib.Path(out_path) if out_path else in_path
    if out_path == in_path and backup:
        shutil.copy2(in_path, in_path.with_suffix(in_path.suffix + ".bak"))
    new_trans.to_json(out_path)
    print(f"{in_path.name}: harmonized cofactor charges -> {out_path}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="chemflow.fe.openfe.utils.repair_json",
        description="Harmonize isomorphic cofactor partial charges in OpenFE edge JSON(s).",
    )
    parser.add_argument("path", type=pathlib.Path,
                        help="A transformation JSON, or a directory (use --glob).")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=None,
                        help="Output file (single-file mode only). Default: in place.")
    parser.add_argument("--glob", default="*.json",
                        help="When PATH is a directory, which files to repair (default: *.json).")
    parser.add_argument("--no-backup", action="store_true",
                        help="Do not write a .bak copy when editing in place.")
    args = parser.parse_args(argv)

    if args.path.is_dir():
        files = sorted(args.path.glob(args.glob))
        if not files:
            print(f"No files matching {args.glob!r} in {args.path}", file=sys.stderr)
            return 1
        changed = 0
        for f in files:
            try:
                changed += int(repair_transformation(f, backup=not args.no_backup))
            except Exception as exc:  # pragma: no cover - surface per-file errors
                print(f"{f.name}: ERROR {exc}", file=sys.stderr)
        print(f"Done. Repaired {changed}/{len(files)} file(s).")
        return 0

    repair_transformation(args.path, args.output, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
