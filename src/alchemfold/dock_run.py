"""AlchemFold docking runner — dispatches to the configured co-folding engine.

Run headlessly with::

    python -m alchemfold.dock_run --config docking.yaml
"""

from __future__ import annotations

import argparse
import copy
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

from alchemfold.dock_config import load_config, load_ligands_from_file, validate_config
from alchemfold.engines import get_engine


def _run_one(engine, conf, workdir: Path, *, dry_run: bool) -> int:
    inputs, out_dir = engine.write_inputs(conf, workdir)
    command = engine.build_command(conf, inputs, out_dir)
    if dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return 0
    for label, path in inputs.items():
        print(f"  {label}: {path}")
    print(f"Running: {' '.join(shlex.quote(p) for p in command)}")
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        print(f"  -> predictions in: {out_dir}")
    return result.returncode


def run_from_config(config_path, *, dry_run: bool = False) -> int:
    conf = load_config(config_path)
    validate_config(conf)
    engine = get_engine(conf.engine)

    workdir = Path(conf.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    ligand_file = getattr(conf, "ligand_file", None)
    if ligand_file:
        library = load_ligands_from_file(
            ligand_file, getattr(conf, "ligand_file_format", "auto")
        )
        print(f"Engine: {conf.engine}; screening {len(library)} ligand(s) from {ligand_file}")
        worst = 0
        for i, lig in enumerate(library, start=1):
            conf_i = copy.deepcopy(conf)
            conf_i.ligand_file = None
            conf_i.ligands = [SimpleNamespace(id=lig["id"], smiles=lig["smiles"])]
            # Scope pocket / affinity to the ligand being screened.
            if getattr(conf_i, "pocket", None):
                conf_i.pocket.binder = lig["id"]
            if getattr(conf_i, "predict_affinity", False):
                conf_i.affinity_binder = lig["id"]
            ligand_workdir = workdir / "ligands" / lig["id"]
            print(f"[{i}/{len(library)}] ligand {lig['id']}: {lig['smiles']}")
            rc = _run_one(engine, conf_i, ligand_workdir, dry_run=dry_run)
            worst = worst or rc
        if not dry_run and worst == 0:
            print(f"Screen complete. Per-ligand outputs under: {workdir / 'ligands'}")
            print("Feed a predicted complex into: "
                  "alchemforge openmm modeling build-soluble | build-membrane")
        return worst

    # Single complex (inline ligands).
    print(f"Engine: {conf.engine}")
    rc = _run_one(engine, conf, workdir, dry_run=dry_run)
    if not dry_run and rc == 0:
        print("Feed the predicted complex into: "
              "alchemforge openmm modeling build-soluble | build-membrane")
    return rc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AlchemFold docking (multi-engine co-folding: Boltz-2, Chai-1, DiffDock).",
    )
    parser.add_argument("--config", required=True, help="Docking YAML configuration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write engine inputs and print the command without running")
    args = parser.parse_args(argv)
    return run_from_config(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
