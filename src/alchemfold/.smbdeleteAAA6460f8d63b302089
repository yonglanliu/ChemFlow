"""Boltz-2 co-folding runner.

Translates an AlchemForge co-folding config into a Boltz input YAML and a
``boltz predict`` command, then runs it. Only stdlib + PyYAML are imported here;
the actual prediction needs the ``boltz`` package (Boltz-2) on the PATH.

Run headlessly with::

    python -m alchemfold.boltz_run --config cofold_boltz.yaml
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from alchemfold.boltz_config import load_config, validate_config


def build_boltz_input(conf) -> dict:
    """Build the Boltz-2 input mapping (version/sequences/properties)."""
    use_msa_server = bool(getattr(conf, "use_msa_server", True))
    sequences: list[dict] = []

    for protein in getattr(conf, "proteins", []) or []:
        entry = {
            "id": str(protein.id),
            "sequence": str(protein.sequence),
        }
        # When not using the MSA server, Boltz needs an explicit msa (or "empty").
        if not use_msa_server:
            entry["msa"] = str(getattr(protein, "msa", None) or "empty")
        sequences.append({"protein": entry})

    for ligand in getattr(conf, "ligands", []) or []:
        entry = {"id": str(ligand.id)}
        smiles = getattr(ligand, "smiles", None)
        ccd = getattr(ligand, "ccd", None)
        if smiles:
            entry["smiles"] = str(smiles)
        else:
            entry["ccd"] = str(ccd)
        sequences.append({"ligand": entry})

    boltz_input: dict = {"version": 1, "sequences": sequences}

    if getattr(conf, "predict_affinity", False):
        boltz_input["properties"] = [
            {"affinity": {"binder": str(conf.affinity_binder)}}
        ]
    return boltz_input


def write_inputs(conf) -> tuple[Path, Path]:
    """Write the Boltz input YAML into the work dir; return (yaml_path, out_dir)."""
    import yaml

    workdir = Path(conf.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    input_yaml = workdir / "boltz_input.yaml"
    input_yaml.write_text(
        yaml.safe_dump(build_boltz_input(conf), sort_keys=False), encoding="utf-8"
    )
    out_dir = Path(conf.out_dir)
    if not out_dir.is_absolute():
        out_dir = workdir / out_dir
    return input_yaml, out_dir


def build_command(conf, input_yaml: Path, out_dir: Path) -> list[str]:
    """Assemble the ``boltz predict`` command for this config."""
    command = shlex.split(str(getattr(conf, "boltz_command", None)
                              or os.environ.get("CHEMFLOW_BOLTZ_COMMAND", "boltz")))
    command += [
        "predict", str(input_yaml),
        "--out_dir", str(out_dir),
        "--output_format", str(conf.output_format),
        "--diffusion_samples", str(int(conf.diffusion_samples)),
        "--recycling_steps", str(int(conf.recycling_steps)),
        "--sampling_steps", str(int(conf.sampling_steps)),
        "--seed", str(int(conf.seed)),
        "--accelerator", str(conf.accelerator),
        "--devices", str(int(conf.devices)),
    ]
    if bool(getattr(conf, "use_msa_server", True)):
        command.append("--use_msa_server")
    if bool(getattr(conf, "override", False)):
        command.append("--override")
    cache = (getattr(conf, "cache_dir", None)
             or os.environ.get("CHEMFLOW_MODEL_CACHE")
             or os.environ.get("CHEMFLOW_BOLTZ_CACHE")
             or os.environ.get("BOLTZ_CACHE"))
    if not cache:
        from alchemfold.dock_config import default_cache_dir

        cache = default_cache_dir()
    if cache:
        command += ["--cache", str(Path(cache).expanduser())]
    return command


def run_from_config(config_path, *, dry_run: bool = False) -> int:
    """Load + validate the config, write Boltz inputs, and run ``boltz predict``."""
    conf = load_config(config_path)
    validate_config(conf)
    input_yaml, out_dir = write_inputs(conf)
    from alchemfold._exec import resolve_command

    command = resolve_command(build_command(conf, input_yaml, out_dir))

    if dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return 0

    print(f"Boltz input: {input_yaml}")
    print(f"Running: {' '.join(shlex.quote(p) for p in command)}")
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        print(f"Boltz co-folding complete. Predictions in: {out_dir}")
        print("Feed the predicted complex into: "
              "alchemforge openmm modeling build-soluble | build-membrane")
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Boltz-2 co-folding (protein + ligand joint structure prediction).",
    )
    parser.add_argument("--config", required=True, help="Co-folding YAML configuration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the Boltz input and print the command without running")
    args = parser.parse_args(argv)
    return run_from_config(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
