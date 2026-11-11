"""`alchemfold` CLI — co-folding (protein + ligand) structure prediction.

Standalone from the AlchemForge free-energy package. Currently backed by Boltz-2:

    alchemfold generate [-o cofold.yaml]          # commented example config
    alchemfold boltz validate CONFIG              # check inputs (no Boltz needed)
    alchemfold boltz stage    CONFIG              # write a runner script
    alchemfold boltz run      CONFIG [--dry-run]  # run boltz predict

The predicted complex (PDB/mmCIF) can then feed the OpenMM modeling steps
(`alchemforge openmm modeling build-soluble | build-membrane`) and FEP.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _hpc_preset_names(section: str = "docking") -> list[str]:
    from alchemfold.hpc_config import presets

    return list(presets(section).keys())


def _hpc_default_preset() -> str:
    from alchemfold.hpc_config import default_preset

    return default_preset()


def _hpc_gpu_types() -> list[str]:
    from alchemfold.hpc_config import gpu_types

    return gpu_types()


def _run_cmd(command: list[str], *, dry_run: bool) -> int:
    if dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, check=False).returncode


def _boltz_validate(args: argparse.Namespace) -> int:
    from alchemfold.boltz_config import load_config, validate_config

    conf_path = Path(args.config).expanduser().resolve()
    validate_config(load_config(conf_path))
    print(f"Valid Boltz co-folding configuration: {args.config}")
    return 0


def _boltz_stage(args: argparse.Namespace) -> int:
    from alchemfold.boltz_config import (
        load_config, stage as stage_boltz, validate_config,
    )
    conf_path = Path(args.config).expanduser().resolve()
    conf = load_config(conf_path)
    validate_config(conf)
    runner, _ = stage_boltz(conf_path, Path(getattr(conf, "workdir", conf_path.parent)))
    print(f"Wrote co-folding runner: {runner}")
    print(f"Run it with: bash {runner} {conf_path}")
    return 0


def _boltz_run(args: argparse.Namespace) -> int:
    conf_path = Path(args.config).expanduser().resolve()
    # Validate with the light config layer before importing the Boltz backend.
    from alchemfold.boltz_config import load_config, validate_config

    validate_config(load_config(conf_path))
    command = [sys.executable, "-m", "alchemfold.boltz_run", "--config", str(conf_path)]
    if args.dry_run:
        command.append("--dry-run")
    return _run_cmd(command, dry_run=False)


def _generate(args: argparse.Namespace) -> int:
    from alchemfold.boltz_config import default_yaml_text

    text = default_yaml_text(
        protein_id=args.protein_id,
        ligand_id=args.ligand_id,
        sequence=args.sequence or "PUT_YOUR_PROTEIN_SEQUENCE_HERE",
        smiles=args.smiles or "CCO",
        predict_affinity=not args.no_affinity,
    )
    output = Path(args.output) if args.output else Path("cofold_boltz.yaml")
    output = output.expanduser()
    if output.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {output} (use --force)",
              file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote commented co-folding config: {output}")
    print(f"Edit it, then: alchemfold boltz run {output}")
    return 0


def _dock_validate(args: argparse.Namespace) -> int:
    from alchemfold.dock_config import load_config, validate_config

    conf_path = Path(args.config).expanduser().resolve()
    conf = load_config(conf_path)
    validate_config(conf)
    print(f"Valid {conf.engine} docking configuration: {args.config}")
    return 0


def _dock_stage(args: argparse.Namespace) -> int:
    from alchemfold.dock_config import (
        load_config, stage as stage_dock, stage_slurm, validate_config,
    )
    conf_path = Path(args.config).expanduser().resolve()
    conf = load_config(conf_path)
    validate_config(conf)
    workdir = Path(getattr(conf, "workdir", conf_path.parent))
    if args.target == "hpc":
        script, _ = stage_slurm(
            conf_path, workdir,
            preset=args.hpc_preset,
            partition=args.partition,
            account=args.account,
            gpu_type=args.gpu_type,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            conda_env=args.conda_env,
        )
        print(f"Wrote SLURM submit script ({args.hpc_preset}): {script}")
        print(f"Submit it with: sbatch {script}")
    else:
        runner, _ = stage_dock(conf_path, workdir)
        print(f"Wrote docking runner: {runner}")
        print(f"Run it with: bash {runner} {conf_path}")
    return 0


def _dock_run(args: argparse.Namespace) -> int:
    conf_path = Path(args.config).expanduser().resolve()
    from alchemfold.dock_config import load_config, validate_config

    validate_config(load_config(conf_path))
    command = [sys.executable, "-m", "alchemfold.dock_run", "--config", str(conf_path)]
    if args.dry_run:
        command.append("--dry-run")
    return _run_cmd(command, dry_run=False)


def _dock_generate(args: argparse.Namespace) -> int:
    from alchemfold.dock_config import (
        default_cache_dir,
        default_yaml_text,
        parse_fasta,
        parse_sequences_text,
    )

    proteins = None
    if getattr(args, "sequence_file", None):
        proteins = parse_fasta(Path(args.sequence_file).expanduser(), first_id=args.protein_id)
        print(f"Read {len(proteins)} sequence(s) from {args.sequence_file}: "
              f"chains {', '.join(c for c, _ in proteins)}")
    elif args.sequence:
        # --sequence may hold one sequence, several (blank-line separated), or FASTA
        # text → one protein chain each.
        parsed = parse_sequences_text(args.sequence, first_id=args.protein_id)
        if len(parsed) > 1:
            proteins = parsed
            print(f"Parsed {len(parsed)} sequences → chains "
                  f"{', '.join(c for c, _ in parsed)}")
        elif len(parsed) == 1:
            args.sequence = parsed[0][1]  # cleaned single sequence (drop header/wrapping)

    text = default_yaml_text(
        engine=args.engine,
        protein_id=args.protein_id,
        ligand_id=args.ligand_id,
        out_dir=("predictions" if args.jobname else "docking_output"),
        jobname=(args.jobname or ""),
        sequence=args.sequence or "PUT_YOUR_PROTEIN_SEQUENCE_HERE",
        proteins=proteins,
        smiles=args.smiles or "CCO",
        with_pocket=args.pocket,
        pocket_binder=args.pocket_binder,
        pocket_max_distance=args.pocket_max_distance,
        pocket_contacts=args.pocket_contacts,
        # absolute: the config may live in a job subfolder, and paths resolve from the config's folder
        ligand_file=(str(Path(args.ligand_file).expanduser().resolve()) if args.ligand_file else None),
        ligand_file_format=args.ligand_file_format,
        cache_dir=args.cache_dir or default_cache_dir(),
        diffusion_samples=args.diffusion_samples,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        no_kernels=not args.kernels,
        use_msa_server=args.use_msa_server,
        predict_affinity=args.predict_affinity,
        affinity_binder=args.affinity_binder,
        receptor_pdb=(str(Path(args.receptor_pdb).expanduser().resolve()) if args.receptor_pdb else None),
        diffdock_samples=args.diffdock_samples if args.engine == "diffdock" else None,
        diffdock_steps=args.diffdock_steps if args.engine == "diffdock" else None,
        diffdock_trajectory=bool(args.diffdock_trajectory and args.engine == "diffdock"),
        diffdock_batch_size=args.diffdock_batch_size if args.engine == "diffdock" else None,
    )
    if args.output:
        output = Path(args.output)
    elif args.jobname:
        output = Path(args.jobname) / "docking.yaml"   # the config lives IN the job folder
    else:
        output = Path("docking.yaml")
    output = output.expanduser()
    if output.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {output} (use --force)", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote commented docking config ({args.engine}): {output}")
    print(f"Edit it, then: alchemfold dock run {output}")
    return 0


def _screen_generate(args: argparse.Namespace) -> int:
    from alchemfold.dock_config import parse_fasta, parse_sequences_text
    from alchemfold.screen_config import default_yaml_text

    proteins = None
    if getattr(args, "sequence_file", None):
        proteins = parse_fasta(Path(args.sequence_file).expanduser())
        print(f"Read {len(proteins)} sequence(s) from {args.sequence_file}: "
              f"chains {', '.join(c for c, _ in proteins)}")
    elif args.sequence:
        parsed = parse_sequences_text(args.sequence)
        if len(parsed) > 1:
            proteins = parsed
            print(f"Parsed {len(parsed)} sequences → chains "
                  f"{', '.join(c for c, _ in parsed)}")
        elif len(parsed) == 1:
            args.sequence = parsed[0][1]

    # SLURM resources: screening preset (env/hpc.yaml `screening:`) then CLI overrides.
    from alchemfold import hpc_config

    preset_name = args.hpc_preset or hpc_config.default_preset("screening")
    res = hpc_config.preset_resources(preset_name, "screening")
    slurm = {
        "partition": res["partition"], "account": res["account"], "nodes": res["nodes"],
        "gpus_per_task": res["gpus_per_node"], "gpu_type": res["gpu_type"],
        "cpus_per_task": res["cpus_per_task"], "mem": res["mem"], "time": res["time"],
        "conda_env": res["conda_env"],
    }
    if args.gpus_per_task is not None:
        slurm["gpus_per_task"] = args.gpus_per_task
    if args.gpu_type is not None:
        slurm["gpu_type"] = args.gpu_type
    n_shards = args.n_shards if args.n_shards is not None else res["shards"]
    lig_files = [str(Path(f).expanduser().resolve()) for f in (args.ligand_files or [])] or None
    if args.library == "custom" and not lig_files:
        print("--library custom requires --ligand-files", file=sys.stderr)
        return 1
    if args.library == "enamine_real" and args.max_ligands <= 0:
        print("Note: Enamine REAL is billions of ligands — set --max-ligands "
              "(e.g. 10000) or Validate will refuse it.", file=sys.stderr)

    text = default_yaml_text(
        engine=args.engine,
        library=args.library,
        sequence=args.sequence or "PUT_YOUR_PROTEIN_SEQUENCE_HERE",
        proteins=proteins,
        ligand_files=lig_files,
        max_ligands=args.max_ligands,
        sample=args.sample,
        n_shards=n_shards,
        shard_size=0 if n_shards else 1000,
        slurm=slurm,
        jobname=args.jobname or "",
        cleanup=args.cleanup,
        predict_affinity=args.predict_affinity and args.engine == "boltz",
        rank_metric=args.rank_metric,
        receptor_pdb=(str(Path(args.receptor_pdb).expanduser().resolve()) if args.receptor_pdb else None),
        diffdock_samples=args.diffdock_samples,
        diffdock_steps=args.diffdock_steps,
        diffdock_trajectory=bool(args.diffdock_trajectory),
        diffdock_batch_size=args.diffdock_batch_size,
    )
    print(f"SLURM preset {preset_name}: {n_shards or 'size-based'} shard(s) x "
          f"{slurm['gpus_per_task']} GPU(s)"
          + (f" ({slurm['gpu_type']})" if slurm["gpu_type"] else ""))
    if args.output:
        output = Path(args.output)
    elif args.jobname:
        output = Path(args.jobname) / "screen.yaml"   # config lives in the job folder
    else:
        output = Path("screen.yaml")
    output = output.expanduser()
    if output.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {output} (use --force)", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote commented screening config ({args.engine}): {output}")
    print(f"Edit it (add ligand_files + sequence), then: alchemfold screen stage {output}")
    return 0


def _screen_validate(args: argparse.Namespace) -> int:
    from alchemfold.screen_config import load_config, validate_config

    conf_path = Path(args.config).expanduser().resolve()
    validate_config(load_config(conf_path))
    print(f"Valid screening configuration: {args.config}")
    return 0


def _screen_stage(args: argparse.Namespace) -> int:
    from alchemfold.screen import stage

    return stage(Path(args.config).expanduser().resolve())


def _trajectory_fix(args: argparse.Namespace) -> int:
    from alchemfold.diffdock_traj import clean_file

    dst, info = clean_file(args.trajectory.expanduser(), args.output.expanduser() if args.output else None)
    print(f"Wrote {dst}: {info['frames_out']} frame(s) (from {info['frames_in']}), "
          f"{info['dropped_hydrogens']} frozen hydrogen(s) removed, "
          f"{'frames reordered to random start -> final pose' if info['reordered'] else 'frame order unchanged'}")
    return 0


def _gpu_test(args: argparse.Namespace) -> int:
    from alchemfold.gpu_test import run

    return run(args)


def _screen_clean(args: argparse.Namespace) -> int:
    from alchemfold.screen import cleanup

    return cleanup(args.config, force=args.force)


def _screen_rank(args: argparse.Namespace) -> int:
    from alchemfold.screen import rank

    return rank(Path(args.config).expanduser().resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alchemfold",
        description="Co-folding (protein + ligand) structure prediction and docking",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate", help="Write a fully-commented example co-folding config YAML",
    )
    generate.add_argument("-o", "--output", type=Path,
                          help="Output YAML path (default: cofold_boltz.yaml)")
    generate.add_argument("--protein-id", default="A", help="Protein chain id")
    generate.add_argument("--ligand-id", default="L", help="Ligand chain id")
    generate.add_argument("--sequence", default=None, help="Protein sequence to pre-fill")
    generate.add_argument("--smiles", default=None, help="Ligand SMILES to pre-fill")
    generate.add_argument("--no-affinity", action="store_true",
                          help="Do not enable Boltz-2 affinity prediction")
    generate.add_argument("--force", action="store_true",
                          help="Overwrite an existing file")
    generate.set_defaults(func=_generate)

    boltz = commands.add_parser("boltz", help="Boltz-2 co-folding engine")
    engine = boltz.add_subparsers(dest="boltz_command", required=True)

    validate = engine.add_parser("validate", help="Validate the co-folding config")
    validate.add_argument("config", type=Path, help="Co-folding YAML config")
    validate.set_defaults(func=_boltz_validate)

    stage_cmd = engine.add_parser("stage", help="Write a runner script")
    stage_cmd.add_argument("config", type=Path, help="Co-folding YAML config")
    stage_cmd.set_defaults(func=_boltz_stage)

    run = engine.add_parser("run", help="Run boltz predict now (requires Boltz-2)")
    run.add_argument("config", type=Path, help="Co-folding YAML config")
    run.add_argument("--dry-run", action="store_true",
                     help="Write Boltz input + print the command without running")
    run.set_defaults(func=_boltz_run)

    # --- Multi-engine docking (Boltz-2 / Chai-1) ---
    dock = commands.add_parser(
        "dock",
        help="Multi-engine docking / co-folding (choose Boltz-2 or Chai-1)",
    )
    dock_commands = dock.add_subparsers(dest="dock_command", required=True)

    dgen = dock_commands.add_parser(
        "generate", help="Write a commented docking config for a chosen engine"
    )
    dgen.add_argument("--engine", choices=["boltz", "chai", "diffdock"], default="boltz",
                      help="Co-folding / docking model (default: boltz)")
    dgen.add_argument("-o", "--output", type=Path,
                      help="Output YAML path (default: docking.yaml)")
    dgen.add_argument("--protein-id", default="A", help="Protein chain id")
    dgen.add_argument("--jobname", default=None,
                      help="Job name: EVERYTHING (config, engine inputs, scripts, logs, predictions) is "
                           "written under <workdir>/<jobname>/, config included (-o defaults to "
                           "<jobname>/docking.yaml)")
    dgen.add_argument("--ligand-id", default="L", help="Ligand chain id")
    dgen.add_argument("--receptor-pdb", default=None,
                      help="Receptor structure (PDB) — required for engine diffdock")
    dgen.add_argument("--diffdock-samples", type=int, default=None,
                      help="engine diffdock: poses sampled per ligand (default 10)")
    dgen.add_argument("--diffdock-steps", type=int, default=None,
                      help="engine diffdock: denoising (reverse-diffusion) steps (default 20)")
    dgen.add_argument("--diffdock-batch-size", type=int, default=None,
                      help="engine diffdock: GPU batch over one ligand's poses (default 10; only helps "
                           "when --diffdock-samples is larger)")
    dgen.add_argument("--diffdock-trajectory", action="store_true",
                      help="engine diffdock: save the diffusion trajectory of every pose "
                           "(rank<k>_reverseprocess.pdb, multi-frame, ligand only)")
    dgen.add_argument("--sequence", default=None, help="Protein sequence to pre-fill")
    dgen.add_argument("--sequence-file", default=None,
                      help="FASTA/.fa file to read the protein sequence(s) from "
                           "(each record becomes a chain A, B, C, …; overrides --sequence)")
    dgen.add_argument("--smiles", default=None, help="Ligand SMILES to pre-fill")
    dgen.add_argument("--ligand-file", default=None,
                      help="Dock ligand(s) from a file instead of --smiles "
                           "(SMILES/.smi, .cxsmiles, CSV, or .sdf; plain or .bz2/.gz)")
    dgen.add_argument("--ligand-file-format", default="auto",
                      choices=["auto", "csv", "smiles", "cxsmiles", "sdf"],
                      help="Format of --ligand-file (default: auto-detect by extension)")
    dgen.add_argument("--pocket", action="store_true",
                      help="Include a binding-site pocket constraint block (placeholder "
                           "residues unless --pocket-contacts is given)")
    dgen.add_argument("--pocket-contacts", default=None,
                      help="Binding-site pocket residues as 'CHAIN:RES' list, e.g. "
                           "'A:123,A:145' (bare numbers use the first protein chain). "
                           "Writes a real pocket constraint.")
    dgen.add_argument("--pocket-binder", default=None,
                      help="Ligand id that binds the pocket (default: the ligand id)")
    dgen.add_argument("--pocket-max-distance", type=float, default=6.0,
                      help="Pocket contact max distance in Angstrom (default: 6.0)")
    dgen.add_argument("--cache-dir", default=None,
                      help="Model/CCD weights cache (boltz --cache / chai CHAI_DOWNLOADS_DIR). "
                           "Point at /data or scratch to avoid the HOME disk quota.")
    dgen.add_argument("--diffusion-samples", type=int, default=1,
                      help="Predicted poses (boltz --diffusion_samples / chai --num-diffn-samples)")
    dgen.add_argument("--recycling-steps", type=int, default=3,
                      help="boltz --recycling_steps / chai --num-trunk-recycles")
    dgen.add_argument("--sampling-steps", type=int, default=200, help="boltz only")
    dgen.add_argument("--kernels", action=argparse.BooleanOptionalAction, default=False,
                      help="Enable Boltz-2's cuequivariance/trifast GPU kernels (default: off "
                           "→ pure-PyTorch; only turn on in a boltz-only env that has the "
                           "kernels installed, since they conflict with chai's torch<2.7)")
    dgen.add_argument("--predict-affinity", action="store_true",
                      help="Boltz-2 only: predict binding affinity (pIC50 = 6 - affinity_pred_value)")
    dgen.add_argument("--affinity-binder", default=None,
                      help="Ligand id to score when --predict-affinity (default: the ligand id)")
    dgen.add_argument("--use-msa-server", action=argparse.BooleanOptionalAction, default=True,
                      help="Compute MSAs via the model's server (needs internet; "
                           "use --no-use-msa-server on offline compute nodes)")
    dgen.add_argument("--force", action="store_true", help="Overwrite an existing file")
    dgen.set_defaults(func=_dock_generate)

    dval = dock_commands.add_parser("validate", help="Validate the docking config")
    dval.add_argument("config", type=Path, help="Docking YAML config")
    dval.set_defaults(func=_dock_validate)

    dstage = dock_commands.add_parser("stage", help="Write a runner script")
    dstage.add_argument("config", type=Path, help="Docking YAML config")
    dstage.add_argument("--target", choices=["localhost", "hpc"], default="localhost",
                        help="localhost: a plain bash runner. hpc: a SLURM submit "
                             "script (single-node single-GPU). (default: localhost)")
    _hpc_presets = list(_hpc_preset_names())
    dstage.add_argument("--hpc-preset", choices=_hpc_presets, default=_hpc_default_preset(),
                        help="HPC walltime preset (used with --target hpc). "
                             f"Choices: {', '.join(_hpc_presets)}")
    dstage.add_argument("--gpu-type", choices=_hpc_gpu_types(), default=None,
                        help="GPU type for --gres (used with --target hpc; overrides "
                             "the preset's gpu_type; '' = any)")
    dstage.add_argument("--partition", default=None, help="SLURM partition (--target hpc)")
    dstage.add_argument("--account", default=None, help="SLURM account (--target hpc)")
    dstage.add_argument("--cpus-per-task", type=int, default=None,
                        help="SLURM --cpus-per-task (--target hpc)")
    dstage.add_argument("--mem", default=None, help="SLURM --mem (--target hpc)")
    dstage.add_argument("--conda-env", default=None,
                        help="conda env to activate in the SLURM job (--target hpc)")
    dstage.set_defaults(func=_dock_stage)

    drun = dock_commands.add_parser("run", help="Run docking now (requires the engine)")
    drun.add_argument("config", type=Path, help="Docking YAML config")
    drun.add_argument("--dry-run", action="store_true",
                      help="Write engine inputs + print the command without running")
    drun.set_defaults(func=_dock_run)

    # --- Virtual screening (shard a library -> SLURM array -> rank) ---
    screen = commands.add_parser(
        "screen",
        help="Virtual screening: shard a ligand library, dock on a SLURM array, rank",
    )
    screen_commands = screen.add_subparsers(dest="screen_command", required=True)

    sgen = screen_commands.add_parser(
        "generate", help="Write a commented screening config for a chosen engine"
    )
    sgen.add_argument("--engine", choices=["boltz", "chai", "diffdock"], default="boltz")
    sgen.add_argument("--library", choices=["zinc20", "enamine_real", "custom"], default="zinc20",
                      help="Source ligand catalog (default: zinc20). zinc20/enamine_real "
                           "use the files under screen_db/ unless --ligand-files is given; "
                           "custom needs --ligand-files")
    sgen.add_argument("--receptor-pdb", default=None,
                      help="Receptor structure (PDB) — required for engine diffdock")
    sgen.add_argument("--diffdock-samples", type=int, default=10,
                      help="engine diffdock: poses sampled per ligand (default 10)")
    sgen.add_argument("--diffdock-steps", type=int, default=20,
                      help="engine diffdock: denoising (reverse-diffusion) steps (default 20)")
    sgen.add_argument("--diffdock-batch-size", type=int, default=None,
                      help="engine diffdock: GPU batch over one ligand's poses (default 10; only helps "
                           "when --diffdock-samples is larger)")
    sgen.add_argument("--diffdock-trajectory", action="store_true",
                      help="engine diffdock: save each ligand's best-pose diffusion trajectory "
                           "(multi-frame PDB) to <job>/trajectories/<id>.pdb")
    sgen.add_argument("--jobname", default=None,
                      help="Job name: everything (split library, SLURM scripts, poses, "
                           "results, ranking) is written under <workdir>/<jobname>/")
    sgen.add_argument("--predict-affinity", action=argparse.BooleanOptionalAction, default=False,
                      help="Boltz-2 only: predict binding affinity for every ligand "
                           "(adds affinity_pred_value + affinity_probability_binary)")
    sgen.add_argument("--rank-metric", default="auto",
                      choices=["auto", "affinity_probability_binary", "affinity_pred_value",
                               "confidence_score", "confidence", "iptm", "ptm", "complex_plddt"],
                      help="Score used to rank hits (default: auto)")
    sgen.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True,
                      help="After ranking, delete intermediate files (split library, shard "
                           "work folders) when all ligands finished (default: on)")
    sgen.add_argument("--ligand-files", nargs="+", default=None, metavar="FILE",
                      help="Library file(s) to screen (.smi/.csv/.cxsmiles, .bz2/.gz; "
                           "globs ok). Overrides the catalog's default files")
    sgen.add_argument("--max-ligands", type=int, default=0,
                      help="Dock only this many ligands, sampled from the library "
                           "(0 = all; REQUIRED for Enamine REAL, which is billions)")
    sgen.add_argument("--sample", choices=["random", "head"], default="random",
                      help="How to pick --max-ligands: random (uniform; decompresses the "
                           "whole library — slow on huge files) or head (first N, instant)")
    _scr_presets = list(_hpc_preset_names("screening"))
    sgen.add_argument("--hpc-preset", choices=_scr_presets, default=None,
                      help="SLURM screening preset from env/hpc.yaml (sets walltime, "
                           f"shards, GPUs/task). Choices: {', '.join(_scr_presets)}")
    sgen.add_argument("--n-shards", type=int, default=None,
                      help="Number of shards = parallel GPUs (SLURM array size); "
                           "overrides the preset")
    sgen.add_argument("--gpus-per-task", type=int, default=None,
                      help="GPUs per shard task (>1 enables Boltz DDP); overrides the preset")
    sgen.add_argument("--gpu-type", default=None,
                      help="GPU type for --gres, e.g. a100 ('' = any); overrides the preset")
    sgen.add_argument("-o", "--output", type=Path,
                      help="Output YAML path (default: <jobname>/screen.yaml, or screen.yaml "
                           "without --jobname)")
    sgen.add_argument("--sequence", default=None, help="Receptor sequence to pre-fill")
    sgen.add_argument("--sequence-file", default=None,
                      help="FASTA/.fa file to read the receptor sequence(s) from "
                           "(each record becomes a chain A, B, C, …; overrides --sequence)")
    sgen.add_argument("--force", action="store_true", help="Overwrite an existing file")
    sgen.set_defaults(func=_screen_generate)

    sval = screen_commands.add_parser("validate", help="Validate the screening config")
    sval.add_argument("config", type=Path, help="Screening YAML config")
    sval.set_defaults(func=_screen_validate)

    sstage = screen_commands.add_parser(
        "stage", help="Shard the library and write the SLURM array + ranking jobs"
    )
    sstage.add_argument("config", type=Path, help="Screening YAML config")
    sstage.set_defaults(func=_screen_stage)

    sclean = screen_commands.add_parser(
        "clean", help="Delete intermediate files (split library, shard work folders)"
    )
    sclean.add_argument("config", type=Path, help="Screening YAML config")
    sclean.add_argument("--force", action="store_true",
                        help="Clean even if some ligands failed / are missing")
    sclean.set_defaults(func=_screen_clean)

    srank = screen_commands.add_parser(
        "rank", help="Rank completed screening results and keep the top hits"
    )
    srank.add_argument("config", type=Path, help="Screening YAML config")
    srank.set_defaults(func=_screen_rank)

    # --- Repair an existing DiffDock trajectory (frozen hydrogens, frame order) ---
    tfix = commands.add_parser(
        "trajectory-fix",
        help="Clean a DiffDock rank<k>_reverseprocess.pdb: drop frozen hydrogens, fix frame order")
    tfix.add_argument("trajectory", type=Path, help="DiffDock trajectory PDB")
    tfix.add_argument("-o", "--output", type=Path, default=None,
                      help="Output PDB (default: <name>_clean.pdb next to the input)")
    tfix.set_defaults(func=_trajectory_fix)

    # --- DiffDock batch docking (backend of the active-learning DiffDock gate) ---
    from alchemfold import dock_batch

    dock_batch.add_parser(commands)

    # --- GPU / multi-GPU / tiny-example test (run inside a GPU allocation) ---
    gtest = commands.add_parser(
        "gpu-test",
        help="Check GPUs, multi-GPU srun launch, and optionally run a tiny built-in Boltz screen",
    )
    gtest.add_argument("--gpus", type=int, default=1,
                       help="GPUs per shard to test (>1 also probes the srun/NCCL launch)")
    gtest.add_argument("--boltz", action="store_true",
                       help="Also run a tiny real Boltz screen (4 ligands, 35-residue protein, "
                            "no MSA server) through the normal stage->run->rank path")
    gtest.add_argument("--diffdock", action="store_true",
                       help="Also run a tiny DiffDock-L screen (4 ligands, DiffDock's bundled 1a0q "
                            "receptor) through stage->run->rank; needs `chemflow diffdock-env "
                            "install` + warmup")
    gtest.add_argument("--affinity", action="store_true",
                       help="With --boltz: also exercise Boltz-2 affinity prediction")
    gtest.add_argument("--submit", action="store_true",
                       help="With --boltz: sbatch the staged array instead of running in the "
                            "current allocation (tests the real submit script)")
    gtest.add_argument("--jobname", default="gpu_test",
                       help="Job folder name for the example (default: gpu_test)")
    gtest.add_argument("--workdir", type=Path, default=Path("."),
                       help="Base folder; the example is written to <workdir>/<jobname>/")
    gtest.add_argument("--boltz-command", default=None, help=argparse.SUPPRESS)
    gtest.set_defaults(func=_gpu_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
