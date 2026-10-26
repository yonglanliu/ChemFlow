"""Prepare reusable GROMACS ligand files with Open Babel and ACPYPE."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LigandPreparationConfig:
    input_file: Path
    output_directory: Path
    charge_method: str = "bcc"
    atom_type: str = "gaff2"
    net_charge: int | None = None
    net_charges: tuple[int, ...] = ()
    resnames: tuple[str, ...] = ()
    obabel_command: str = "obabel"
    acpype_command: str = "acpype"

    def validate(self) -> None:
        if not self.input_file.is_file() or self.input_file.suffix.lower() not in {
            ".sdf",
            ".mol2",
        }:
            raise ValueError(
                f"Ligand input must be an existing SDF or MOL2: {self.input_file}"
            )
        if self.charge_method not in {"bcc", "user"}:
            raise ValueError("Charge method must be bcc or user")
        if self.atom_type not in {"gaff", "gaff2"}:
            raise ValueError("Atom type must be gaff or gaff2")
        if not self.obabel_command.strip() or not self.acpype_command.strip():
            raise ValueError("Open Babel and ACPYPE commands cannot be empty")
        for resname in self.resnames:
            if not re.fullmatch(r"[A-Za-z0-9]{1,5}", resname):
                raise ValueError(
                    "Residue names must contain 1-5 letters or numbers"
                )


def stage(config: LigandPreparationConfig, task_directory: Path) -> tuple[Path, Path]:
    config.validate()
    task_directory.mkdir(parents=True, exist_ok=True)
    manifest = task_directory / "gromacs_ligand_prepare.json"
    payload = asdict(config)
    payload["input_file"] = str(config.input_file.expanduser().resolve())
    payload["output_directory"] = str(config.output_directory.expanduser().resolve())
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    runner = task_directory / "run_gromacs_ligand_prepare.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'MANIFEST="${1:?Usage: $0 /path/to/gromacs_ligand_prepare.json}"\n'
        'python -m alchemforge.gromacs.ligand_prepare "${MANIFEST}"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, manifest


def _safe_name(name: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_.-")
    return cleaned or f"ligand_{index:03d}"


def _load_ligands(path: Path):
    from rdkit import Chem

    if path.suffix.lower() == ".sdf":
        molecules = [
            mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol
        ]
    else:
        # MOL2 file input
        molecule = Chem.MolFromMol2File(str(path), removeHs=False)
        molecules = [molecule] if molecule is not None else []
    if not molecules:
        raise ValueError(f"No valid ligand molecules found in {path}")
    return molecules


def _run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise ValueError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def prepare(config: LigandPreparationConfig) -> list[Path]:
    config.validate()
    if shutil.which(config.obabel_command) is None:
        raise ValueError(f"Open Babel command is unavailable: {config.obabel_command}")
    if shutil.which(config.acpype_command) is None:
        raise ValueError(f"ACPYPE command is unavailable: {config.acpype_command}")

    from rdkit import Chem

    molecules = _load_ligands(config.input_file)
    if config.net_charge is not None and len(molecules) != 1:
        raise ValueError("A net-charge override can only be used with one ligand")
    if config.net_charges and len(config.net_charges) != len(molecules):
        raise ValueError(
            "The number of assigned net charges must match the number of ligands"
        )
    if config.resnames and len(config.resnames) != len(molecules):
        raise ValueError(
            "The number of assigned residue names must match the number of ligands"
        )
    config.output_directory.mkdir(parents=True, exist_ok=True)
    prepared: list[Path] = []
    used_names: set[str] = set()
    used_resnames: set[str] = set()
    mapping_lines = ["index\tligand_name\tresname\tnet_charge\tdirectory"]

    for index, molecule in enumerate(molecules, start=1):
        source_name = molecule.GetProp("_Name") if molecule.HasProp("_Name") else ""
        name = _safe_name(source_name, index)
        if name in used_names:
            name = f"{name}_{index}"
        used_names.add(name)
        resname = (
            config.resnames[index - 1].upper()
            if config.resnames
            else (f"L{index:02d}" if len(molecules) > 1 else "LIG")
        )
        if resname in used_resnames:
            raise ValueError(
                f"Residue name {resname!r} is assigned more than once. Each "
                "compound needs a unique residue name."
            )
        used_resnames.add(resname)
        target = config.output_directory / resname
        if target.exists():
            raise ValueError(
                f"Residue output already exists: {target}. Choose a new output "
                "directory or move the existing result."
            )
        target.mkdir(parents=True)
        ligand_sdf = target / f"{resname}.sdf"
        writer = Chem.SDWriter(str(ligand_sdf))
        writer.write(molecule)
        writer.close()

        ligand_mol2 = target / f"{resname}.mol2"
        _run(
            [config.obabel_command, str(ligand_sdf), "-O", str(ligand_mol2)],
            target,
        )
        formal_charge = int(Chem.GetFormalCharge(molecule))
        if config.net_charges:
            charge = config.net_charges[index - 1]
        else:
            charge = (
                config.net_charge
                if config.net_charge is not None
                else formal_charge
            )
        _run(
            [
                config.acpype_command,
                "-i",
                str(ligand_mol2),
                "-b",
                resname,
                "-c",
                config.charge_method,
                "-a",
                config.atom_type,
                "-n",
                str(charge),
                "-o",
                "gmx",
            ],
            target,
        )
        acpype_directory = target / f"{resname}.acpype"
        expected = {
            "itp": acpype_directory / f"{resname}_GMX.itp",
            "gro": acpype_directory / f"{resname}_GMX.gro",
            "top": acpype_directory / f"{resname}_GMX.top",
        }
        missing = [str(path) for path in expected.values() if not path.is_file()]
        if missing:
            raise ValueError(
                "ACPYPE did not create expected files: " + ", ".join(missing)
            )
        for suffix, source in expected.items():
            shutil.copy2(source, target / f"{resname}.{suffix}")
        mapping_lines.append(f"{index}\t{name}\t{resname}\t{charge}\t{resname}")
        prepared.append(target)

    (config.output_directory / "ligand_manifest.tsv").write_text(
        "\n".join(mapping_lines) + "\n", encoding="utf-8"
    )
    return prepared


def run_manifest(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = LigandPreparationConfig(
        input_file=Path(payload["input_file"]),
        output_directory=Path(payload["output_directory"]),
        charge_method=str(payload["charge_method"]),
        atom_type=str(payload["atom_type"]),
        net_charge=payload.get("net_charge"),
        net_charges=tuple(int(value) for value in payload.get("net_charges", [])),
        resnames=tuple(str(value) for value in payload.get("resnames", [])),
        obabel_command=str(payload["obabel_command"]),
        acpype_command=str(payload["acpype_command"]),
    )
    prepared = prepare(config)
    print(f"Prepared {len(prepared)} ligand(s) in {config.output_directory}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        return run_manifest(args.manifest)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
