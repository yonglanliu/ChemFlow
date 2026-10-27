"""Execute a directory of OpenFE transformation JSON files in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class OpenFEExecutionConfig:
    json_directory: Path
    output_directory: Path
    max_parallel: int = 3
    openfe_command: str = "openfe"
    resume: bool = True

    def validate(self) -> list[Path]:
        if not self.json_directory.is_dir():
            raise ValueError(
                f"OpenFE JSON directory does not exist: {self.json_directory}"
            )
        transformations = sorted(self.json_directory.glob("*.json"))
        if not transformations:
            raise ValueError(
                f"No transformation JSON files were found in {self.json_directory}"
            )
        if self.max_parallel < 1:
            raise ValueError("Maximum parallel transformations must be at least 1")
        if not self.openfe_command.strip():
            raise ValueError("OpenFE command cannot be empty")
        return transformations

    def payload(self) -> dict[str, object]:
        return {
            "json_directory": str(self.json_directory.resolve()),
            "output_directory": str(self.output_directory.resolve()),
            "max_parallel": self.max_parallel,
            "openfe_command": self.openfe_command,
            "resume": self.resume,
        }


def stage(config: OpenFEExecutionConfig, task_directory: Path) -> tuple[Path, Path]:
    config.validate()
    task_directory.mkdir(parents=True, exist_ok=True)
    config_path = task_directory / "openfe_execution.yaml"
    config_path.write_text(
        yaml.safe_dump(config.payload(), sort_keys=False), encoding="utf-8"
    )
    runner = task_directory / "run_openfe_transformations.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'CONFIG="${1:?Usage: $0 openfe_execution.yaml}"\n'
        'python -m alchemforge.openfe.execute --config "${CONFIG}"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, config_path


def _run_transformation(config: OpenFEExecutionConfig, source: Path) -> Path:
    destination = config.output_directory / source.stem
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        *shlex.split(config.openfe_command),
        "quickrun",
        str(source),
        "-d",
        str(destination),
        "-o",
        str(destination / "result.json"),
    ]
    if config.resume:
        command.append("--resume")
    print(f"Running {source.name}", flush=True)
    subprocess.run(command, check=True)
    print(f"Completed {source.name}", flush=True)
    return destination


def execute(config: OpenFEExecutionConfig) -> list[Path]:
    transformations = config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.max_parallel
    ) as executor:
        futures = [
            executor.submit(_run_transformation, config, source)
            for source in transformations
        ]
        return [future.result() for future in futures]


def load_config(path: Path) -> OpenFEExecutionConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.expanduser().resolve().parent

    def resolve_config_path(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else base / candidate

    return OpenFEExecutionConfig(
        json_directory=resolve_config_path(payload["json_directory"]),
        output_directory=resolve_config_path(payload["output_directory"]),
        max_parallel=int(payload.get("max_parallel", 3)),
        openfe_command=str(payload.get("openfe_command", "openfe")),
        resume=bool(payload.get("resume", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    execute(load_config(args.config))


if __name__ == "__main__":
    main()
