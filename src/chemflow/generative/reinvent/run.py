"""Run REINVENT4 from a ChemFlow YAML.

    python -m chemflow.generative.reinvent.run --config reinvent.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

from chemflow.generative.reinvent.config import build_toml, load_config, validate_config


def stage_inputs(conf: dict[str, Any]) -> tuple[Path, Path]:
    """Write the REINVENT TOML; return ``(toml_path, out_dir)``."""
    workdir = Path(conf["workdir"]).expanduser().resolve()
    out_dir = Path(conf["out_dir"])
    out_dir = out_dir if out_dir.is_absolute() else workdir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    toml_path = out_dir / f"reinvent_{conf['run_type']}.toml"
    toml_path.write_text(build_toml(conf, out_dir), encoding="utf-8")
    return toml_path, out_dir


def build_command(conf: dict[str, Any], toml_path: Path, out_dir: Path) -> list[str]:
    command = shlex.split(str(conf.get("reinvent_command") or "reinvent"))
    command += ["-l", str(out_dir / "reinvent.log")]
    if conf.get("seed") is not None:
        command += ["-s", str(int(conf["seed"]))]
    command.append(str(toml_path))
    return command


def find_results(conf: dict[str, Any], out_dir: Path) -> list[Path]:
    """Result CSVs REINVENT4 wrote (it can exit 0 even after logging errors)."""
    if conf["run_type"] == "sampling":
        candidates = [out_dir / "sampling.csv"]
    else:
        candidates = sorted(out_dir.glob("staged_learning*.csv"))
    return [p for p in candidates if p.is_file() and p.stat().st_size > 0]


def run_from_config(config_path, *, dry_run: bool = False) -> int:
    conf = load_config(config_path)
    validate_config(conf)
    toml_path, out_dir = stage_inputs(conf)
    command = build_command(conf, toml_path, out_dir)
    if dry_run:
        print(" ".join(shlex.quote(p) for p in command))
        return 0
    print(f"REINVENT4 {conf['model']} / {conf['run_type']}")
    print(f"  config: {toml_path}")
    print(f"Running: {' '.join(shlex.quote(p) for p in command)}")
    try:
        rc = subprocess.run(command, check=False).returncode
    except FileNotFoundError:
        from chemflow.generative.reinvent.locate import not_found_message

        print(not_found_message(command[0]))
        return 127
    if rc == 0:
        results = find_results(conf, out_dir)
        if not results:
            print(f"REINVENT4 exited without an error code but produced no result CSV in "
                  f"{out_dir}. Check {out_dir / 'reinvent.log'} for errors.")
            return 1
        for path in results:
            print(f"  -> results: {path}")
        print("Next: chemflow generate reinvent export <csv> --top 100 --sdf  "
              "(then alchemforge rbfe / alchemfold dock)")
    return rc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run REINVENT4 from a ChemFlow config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the REINVENT TOML and print the command without running")
    args = parser.parse_args(argv)
    return run_from_config(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
