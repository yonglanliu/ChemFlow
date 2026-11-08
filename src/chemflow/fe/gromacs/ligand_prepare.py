"""Prepare reusable GROMACS ligand files with Open Babel and ACPYPE."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
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
    overwrite: bool = False

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
        'python -m chemflow.fe.gromacs.ligand_prepare "${MANIFEST}"\n',
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


def _set_mol2_resname(path: Path, resname: str) -> None:
    """Write ``resname`` as the substructure name of every atom in a MOL2.

    Open Babel names the residue ``UNL``/``UNL1``; ACPYPE takes the residue name
    of its outputs from here, so it must be set before ACPYPE runs.
    """
    section = ""
    out: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip().upper()
            out.append(line)
            continue
        fields = line.split()
        if section == "@<TRIPOS>ATOM" and len(fields) >= 8:
            fields[7] = resname
            out.append(" ".join(fields))
        elif section == "@<TRIPOS>SUBSTRUCTURE" and len(fields) >= 2:
            fields[1] = resname
            out.append(" ".join(fields))
        else:
            out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _set_itp_resname(path: Path, resname: str) -> None:
    """Rewrite the residue column of an ITP/TOP ``[ atoms ]`` block."""
    in_atoms = False
    out: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = re.fullmatch(r"\[\s*([^\]]+?)\s*\]", line.split(";", 1)[0].strip())
        if header:
            in_atoms = header.group(1).casefold() == "atoms"
            out.append(line)
            continue
        tokens = list(re.finditer(r"\S+", line.split(";", 1)[0]))
        if in_atoms and len(tokens) >= 5 and tokens[0].group().isdigit():
            token = tokens[3]
            line = line[: token.start()] + resname.ljust(len(token.group())) + line[token.end():]
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _set_gro_resname(path: Path, resname: str) -> None:
    """Set the residue name (columns 6-10) on every atom line of a GRO."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        count = int(lines[1].strip())
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid GRO file: {path}") from error
    for index in range(2, 2 + count):
        lines[index] = lines[index][:5] + f"{resname:<5}" + lines[index][10:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(command: list[str], cwd: Path, env: dict | None = None) -> None:
    # Capture output so a tool failure (e.g. acpype/antechamber/sqm) surfaces its
    # actual error instead of just an exit code.
    result = subprocess.run(command, cwd=cwd, check=False,
                            capture_output=True, text=True, env=env)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-25:])
        raise ValueError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}\n"
            f"--- tool output (last lines) ---\n{tail}"
        )


def _toolchain_env(*tool_paths: str) -> dict:
    """os.environ with each resolved tool's bin dir prepended to PATH.

    acpype shells out to AmberTools' ``antechamber``/``sqm``, found via PATH. When
    chemflow runs through ``<env>/bin/python`` without the conda env activated,
    ``<env>/bin`` is off PATH, so prepend the bin dirs of the resolved tools (and
    set AMBERHOME to the env prefix) to make the sub-tools findable.
    """
    env = dict(os.environ)
    extra: list[str] = []
    for tool in tool_paths:
        if tool and os.path.sep in tool:
            bindir = str(Path(tool).resolve().parent)
            if bindir not in extra:
                extra.append(bindir)
            env.setdefault("AMBERHOME", str(Path(bindir).parent))
    if extra:
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _resolve_executable(command: str) -> str | None:
    """Find ``command`` on PATH, or as a sibling of the running interpreter.

    When chemflow runs inside the GROMACS conda env, the tools live beside
    ``sys.executable`` (…/envs/gmxfe/bin) even if PATH wasn't fully activated.
    """
    import sys

    found = shutil.which(command)
    if found:
        return found
    sibling = Path(sys.executable).with_name(command)
    return str(sibling) if sibling.is_file() else None


def prepare(config: LigandPreparationConfig) -> list[Path]:
    config.validate()
    obabel = _resolve_executable(config.obabel_command)
    if obabel is None:
        raise ValueError(
            f"Open Babel command is unavailable: {config.obabel_command}. "
            "Run this in the GROMACS env (e.g. `conda run -n gmxfe chemflow rbfe assign-charge …`), "
            "install it (`chemflow gmx-env install`), or set obabel in chemflow_tools.yaml / "
            "$CHEMFLOW_OBABEL_COMMAND.")
    acpype = _resolve_executable(config.acpype_command)
    if acpype is None:
        raise ValueError(
            f"ACPYPE command is unavailable: {config.acpype_command}. "
            "Run this in the GROMACS env (`conda run -n gmxfe …`), install it "
            "(`chemflow gmx-env install`), or set acpype in chemflow_tools.yaml / "
            "$CHEMFLOW_ACPYPE_COMMAND.")
    config = replace(config, obabel_command=obabel, acpype_command=acpype)
    run_env = _toolchain_env(obabel, acpype)

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
            if config.overwrite:
                shutil.rmtree(target)
            else:
                raise ValueError(
                    f"Residue output already exists: {target}. Re-run with overwrite "
                    "(`chemflow rbfe assign-charge --overwrite …`), choose a new output "
                    "directory, or move the existing result."
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
            run_env,
        )
        # The residue name inside the files must equal the chosen name.
        _set_mol2_resname(ligand_mol2, resname)
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
            run_env,
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
        # Guarantee the output molecule's residue name matches the chosen one even
        # if ACPYPE/antechamber truncated or renamed it.
        _set_itp_resname(expected["itp"], resname)
        _set_itp_resname(expected["top"], resname)
        _set_gro_resname(expected["gro"], resname)
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
