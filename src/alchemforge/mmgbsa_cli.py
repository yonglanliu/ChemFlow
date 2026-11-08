"""``alchemforge mmgbsa`` — single-trajectory MM-GBSA endpoint, one run or a batch.

    alchemforge mmgbsa generate [mmgbsa.yaml]       # write an example config
    alchemforge mmgbsa validate  CONFIG             # check inputs (no OpenMM)
    alchemforge mmgbsa stage     CONFIG             # write a runner script
    alchemforge mmgbsa run       CONFIG             # compute now (needs OpenMM + mdtraj)
    alchemforge mmgbsa batch     PARENT_DIR         # one config + SLURM array over many MD runs
    alchemforge mmgbsa aggregate BATCH_DIR          # collect + rank every candidate's dG

The config/validate/stage/batch/aggregate layers need no OpenMM; only ``run``
imports the heavy engine (:mod:`alchemforge.mmgbsa_run`).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _generate(args: argparse.Namespace) -> int:
    from alchemforge.mmgbsa_config import default_yaml_text

    target = Path(args.path or "mmgbsa.yaml").expanduser()
    if target.exists() and not args.force:
        raise ValueError(f"{target} already exists (use --force to overwrite)")
    target.write_text(default_yaml_text(ligand_selection=args.ligand_selection), encoding="utf-8")
    print(f"Wrote example MM-GBSA config: {target}")
    print("Edit the paths + ligand_selection, then: alchemforge mmgbsa validate " + str(target))
    return 0


def _validate(args: argparse.Namespace) -> int:
    from alchemforge.mmgbsa_config import load_config, validate_config

    conf_path = Path(args.config).expanduser().resolve()
    validate_config(load_config(conf_path))
    print(f"Valid MM-GBSA configuration: {args.config}")
    return 0


def _stage(args: argparse.Namespace) -> int:
    from alchemforge.mmgbsa_config import load_config, stage, validate_config

    conf_path = Path(args.config).expanduser().resolve()
    conf = load_config(conf_path)
    validate_config(conf)
    runner, _ = stage(conf_path, Path(getattr(conf, "workdir", conf_path.parent)))
    print(f"Wrote MM-GBSA runner: {runner}")
    print(f"Run it with: bash {runner} {conf_path}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from alchemforge.mmgbsa_config import load_config, validate_config

    conf_path = Path(args.config).expanduser().resolve()
    validate_config(load_config(conf_path))
    try:
        from alchemforge.mmgbsa_run import run_from_config
    except ImportError as exc:
        raise ValueError(
            f"MM-GBSA needs OpenMM + mdtraj (and openmmforcefields for ligand params): {exc}. "
            "Install them in the alchemforge env, or use 'stage' and run on a compute node."
        ) from exc
    run_from_config(conf_path)
    return 0


def _batch(args: argparse.Namespace) -> int:
    from alchemforge.mmgbsa_config import discover_runs, write_batch

    runs = discover_runs(args.parent, trajectory_glob=args.trajectory_glob)
    if not runs:
        raise ValueError(
            f"no MD runs found under {args.parent} (looked for */{args.trajectory_glob} "
            "with a sibling .pdb)"
        )
    base: dict = {}
    if args.config:
        from alchemforge.mmgbsa_config import load_config

        loaded = load_config(Path(args.config).expanduser().resolve())
        for key in ("ligand_selection", "strip_selection", "forcefields", "ligand_sdf",
                    "small_molecule_forcefield", "gb_model", "salt_concentration",
                    "solute_dielectric", "solvent_dielectric", "surface_area",
                    "surface_tension", "temperature", "platform", "start_frame",
                    "stride", "max_frames"):
            value = getattr(loaded, key, None)
            if value is not None:
                base[key] = value
    if args.ligand_selection:
        base["ligand_selection"] = args.ligand_selection
    if "ligand_selection" not in base:
        raise ValueError("provide --ligand-selection or a --config that sets ligand_selection")

    written = write_batch(
        runs, args.out or (Path(args.parent).expanduser().resolve() / "mmgbsa_batch"),
        base_config=base, partition=args.partition, cpus_per_task=args.cpus,
        memory=args.mem, walltime=args.time,
        modules=args.modules, conda_init=args.conda_init, conda_environment=args.conda_env,
    )
    print(f"Prepared {len(runs)} MM-GBSA jobs in {written['out_dir']}")
    print(f"Submit with: sbatch {written['array']}")
    print(f"Then rank with: alchemforge mmgbsa aggregate {written['out_dir']}")
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    from alchemforge.mmgbsa_config import aggregate

    rows = aggregate(args.batch_dir)
    if not rows:
        print(f"No mmgbsa_summary.csv found under {args.batch_dir} yet.")
        return 0
    print(f"{'candidate':<28}{'dG (kcal/mol)':>16}{'SEM':>10}{'frames':>8}")
    for row in rows:
        print(f"{row.get('candidate', ''):<28}{row.get('dG_mean', ''):>16}"
              f"{row.get('dG_sem', ''):>10}{row.get('n_frames', ''):>8}")
    print(f"\nRanking written to: {Path(args.batch_dir).expanduser().resolve() / 'mmgbsa_ranking.csv'}")
    return 0


def add_mmgbsa_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "mmgbsa",
        help="Single-trajectory MM-GBSA endpoint (binding free-energy triage)",
    )
    commands = parser.add_subparsers(dest="mmgbsa_command", required=True)

    gen = commands.add_parser("generate", help="Write an example MM-GBSA config")
    gen.add_argument("path", nargs="?", default=None, help="output YAML (default mmgbsa.yaml)")
    gen.add_argument("--ligand-selection", default="resname LIG", help="mdtraj selection for the ligand")
    gen.add_argument("--force", action="store_true", help="overwrite an existing file")
    gen.set_defaults(func=_generate)

    val = commands.add_parser("validate", help="Validate an MM-GBSA config")
    val.add_argument("config")
    val.set_defaults(func=_validate)

    stg = commands.add_parser("stage", help="Write a runner script for one config")
    stg.add_argument("config")
    stg.set_defaults(func=_stage)

    run = commands.add_parser("run", help="Compute MM-GBSA now (needs OpenMM + mdtraj)")
    run.add_argument("config")
    run.set_defaults(func=_run)

    batch = commands.add_parser("batch", help="Prepare configs + a SLURM array over many MD runs")
    batch.add_argument("parent", help="directory whose subfolders each hold an MD trajectory + PDB")
    batch.add_argument("--out", default=None, help="output directory (default <parent>/mmgbsa_batch)")
    batch.add_argument("--config", default=None, help="base config whose settings apply to every run")
    batch.add_argument("--ligand-selection", default=None, help="mdtraj selection for the ligand")
    batch.add_argument("--trajectory-glob", default="*production.dcd", help="trajectory filename glob")
    batch.add_argument("--partition", default="norm")
    batch.add_argument("--cpus", type=int, default=4)
    batch.add_argument("--mem", default="8G")
    batch.add_argument("--time", default="08:00:00")
    batch.add_argument("--modules", default="", help="HPC: 'module load' names, space/comma separated")
    batch.add_argument("--conda-init", default="", help="HPC: path to conda.sh to source in the job")
    batch.add_argument("--conda-env", default="", help="HPC: conda env to activate in the job")
    batch.set_defaults(func=_batch)

    agg = commands.add_parser("aggregate", help="Collect + rank every candidate's dG")
    agg.add_argument("batch_dir")
    agg.set_defaults(func=_aggregate)
