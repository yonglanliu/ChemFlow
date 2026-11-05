"""Alanine scanning on top of the GROMACS resFEP pipeline.

For a selection of protein residues, this expands into one independent
residue -> ALA resFEP calculation per residue and aggregates the per-residue
ddG. Residues can be chosen explicitly (list and/or range) and/or by proximity
to the ligand (interface cutoff). Residues that are already ALA, or are GLY/PRO,
are skipped.

    alchemforge alanine-scan stage    scan.toml   # expand + stage one resFEP run per residue
    alchemforge alanine-scan submit   scan.toml   # submit every per-residue run
    alchemforge alanine-scan analyze  scan.toml   # aggregate per-residue ddG

The scan TOML is a resFEP configuration with the ``[mutation]`` section replaced
by a ``[scan]`` section, e.g.:

    [scan]
    chain = "A"
    residues = [40, 41, 42]          # explicit residue numbers (optional)
    residue_range = "45-55"          # inclusive range (optional)
    interface_cutoff_nm = 0.5        # add residues within this cutoff of the ligand (optional)
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

_SKIP_RESIDUES = {"ALA", "GLY", "PRO"}
# Standard amino acids that alanine scanning may mutate (ALA/GLY/PRO removed).
_STANDARD_AA = {
    "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "SER", "THR", "TRP", "TYR", "VAL", "ALA", "GLY", "PRO",
    # common protonation/charge variants pmx understands
    "HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYX",
}


# --------------------------------------------------------------------------- #
# PDB parsing (no external dependencies)
# --------------------------------------------------------------------------- #
def _parse_pdb(pdb_path: Path):
    """Return (residues, ligand_atoms).

    residues: dict[(chain, resnum)] -> (resname, [ (x,y,z), ... ])
    ligand_atoms is filled separately by the caller via resname match.
    """
    residues: dict[tuple[str, int], tuple[str, list[tuple[float, float, float]]]] = {}
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip().upper()
        chain = line[21].strip() or " "
        try:
            resnum = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        key = (chain, resnum)
        if key not in residues:
            residues[key] = (resname, [])
        residues[key][1].append((x, y, z))
    return residues


def _parse_range(spec: str) -> list[int]:
    spec = str(spec).strip()
    if "-" not in spec:
        raise ValueError(f"[scan].residue_range must be 'start-end' (got {spec!r})")
    start_s, end_s = spec.split("-", 1)
    start, end = int(start_s), int(end_s)
    if end < start:
        raise ValueError(f"[scan].residue_range end < start: {spec!r}")
    return list(range(start, end + 1))


def _interface_residues(
    residues: dict, ligand_resname: str, cutoff_nm: float
) -> set[tuple[str, int]]:
    cutoff_a = float(cutoff_nm) * 10.0  # nm -> Angstrom
    cutoff_sq = cutoff_a * cutoff_a
    ligand_coords: list[tuple[float, float, float]] = []
    for (_chain, _num), (resname, coords) in residues.items():
        if resname == ligand_resname.upper():
            ligand_coords.extend(coords)
    if not ligand_coords:
        raise ValueError(
            f"No ligand atoms with resname {ligand_resname!r} found for interface scan"
        )
    selected: set[tuple[str, int]] = set()
    for key, (resname, coords) in residues.items():
        if resname not in _STANDARD_AA:
            continue
        for ax, ay, az in coords:
            hit = False
            for lx, ly, lz in ligand_coords:
                if (ax - lx) ** 2 + (ay - ly) ** 2 + (az - lz) ** 2 <= cutoff_sq:
                    selected.add(key)
                    hit = True
                    break
            if hit:
                break
    return selected


def select_scan_residues(
    scan: dict, complex_pdb: Path, ligand_resname: str
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Return (selected, skipped) lists of (chain, resnum, resname).

    selected: residues to mutate to ALA; skipped: residues removed because they
    are ALA/GLY/PRO (or not standard amino acids).
    """
    residues = _parse_pdb(complex_pdb)
    chain = str(scan.get("chain", "")).strip()
    wanted: list[tuple[str, int]] = []

    explicit = list(scan.get("residues", []) or [])
    for num in explicit:
        wanted.append((chain, int(num)))
    if scan.get("residue_range"):
        wanted.extend((chain, n) for n in _parse_range(scan["residue_range"]))
    if scan.get("interface_cutoff_nm"):
        interface = _interface_residues(
            residues, ligand_resname, float(scan["interface_cutoff_nm"])
        )
        # Honor the chain filter when one is given; otherwise take all chains.
        wanted.extend(
            key for key in sorted(interface) if not chain or key[0] == chain
        )
    if not wanted:
        raise ValueError(
            "[scan] selected no residues; set residues, residue_range, and/or "
            "interface_cutoff_nm"
        )

    selected: list[tuple[str, int, str]] = []
    skipped: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for key in wanted:
        if key in seen:
            continue
        seen.add(key)
        if key not in residues:
            raise ValueError(f"Residue {key[0]}{key[1]} not found in {complex_pdb}")
        resname = residues[key][0]
        if resname in _SKIP_RESIDUES or resname not in _STANDARD_AA:
            skipped.append((key[0], key[1], resname))
        else:
            selected.append((key[0], key[1], resname))
    if not selected:
        raise ValueError(
            "Every selected residue was skipped (ALA/GLY/PRO or non-standard)"
        )
    return selected, skipped


# --------------------------------------------------------------------------- #
# Minimal TOML emitter (only the value types the resFEP schema uses)
# --------------------------------------------------------------------------- #
def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ValueError(f"Unsupported TOML value type: {type(value)!r}")


def _emit_toml(data: dict) -> str:
    lines: list[str] = []
    # scalars first (none expected at top level, but be safe)
    for key, value in data.items():
        if not isinstance(value, (dict, list)):
            lines.append(f"{key} = {_toml_value(value)}")
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for sub_key, sub_value in value.items():
                lines.append(f"{sub_key} = {_toml_value(sub_value)}")
        elif isinstance(value, list) and value and all(
            isinstance(item, dict) for item in value
        ):
            for item in value:
                lines.append(f"\n[[{key}]]")
                for sub_key, sub_value in item.items():
                    lines.append(f"{sub_key} = {_toml_value(sub_value)}")
    return "\n".join(lines).strip() + "\n"


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #
def _load_scan(path: Path) -> dict:
    import tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def _child_name(base_name: str, chain: str, resnum: int) -> str:
    chain_tag = chain if chain else "X"
    return f"{base_name}_{chain_tag}{resnum}A"


def _child_config_dict(scan_data: dict, base_name: str, chain: str, resnum: int) -> dict:
    child = copy.deepcopy(scan_data)
    child.pop("scan", None)
    child.setdefault("project", {})
    child["project"]["name"] = _child_name(base_name, chain, resnum)
    child["mutation"] = [{"chain": chain, "residue": resnum, "target": "ALA"}]
    return child


def stage(scan_path: Path, *, dry_run: bool = False) -> dict:
    """Expand the scan into per-residue resFEP child projects and stage them."""
    from alchemforge.gromacs.resFEP.config import ResFEPConfig
    from alchemforge.gromacs.resFEP.workflow import stage as resfep_stage

    scan_path = scan_path.expanduser().resolve()
    data = _load_scan(scan_path)
    scan = data.get("scan")
    if not isinstance(scan, dict):
        raise ValueError(f"{scan_path} has no [scan] section")
    base_name = str(data.get("project", {}).get("name", "alascan"))
    base_dir = scan_path.parent
    complex_pdb = Path(str(data.get("system", {}).get("complex", "")))
    if not complex_pdb.is_absolute():
        complex_pdb = (base_dir / complex_pdb).resolve()
    ligand_resname = str(data.get("ligand", {}).get("resname", ""))

    selected, skipped = select_scan_residues(scan, complex_pdb, ligand_resname)

    config_dir = base_dir / f"{base_name}_alascan" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    child_paths: list[Path] = []
    for chain, resnum, _resname in selected:
        child = _child_config_dict(data, base_name, chain, resnum)
        child_path = config_dir / f"{_child_name(base_name, chain, resnum)}.toml"
        child_path.write_text(_emit_toml(child), encoding="utf-8")
        child_paths.append(child_path)
        if not dry_run:
            config = ResFEPConfig.load(child_path, molecule_type="protein")
            config.validate_inputs(require_worker=False)
            resfep_stage(config)

    submit_script = base_dir / f"{base_name}_alascan" / "submit_alanine_scan.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for child_path in child_paths:
        lines.append(f'alchemforge resfep submit {child_path}')
    submit_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    submit_script.chmod(submit_script.stat().st_mode | 0o100)

    return {
        "selected": selected,
        "skipped": skipped,
        "child_configs": child_paths,
        "submit_script": submit_script,
    }


def _result_ddg(child_config: Path) -> float | None:
    """Best-effort read of a per-residue ddG from a staged child project."""
    from alchemforge.gromacs.resFEP.config import ResFEPConfig

    try:
        config = ResFEPConfig.load(child_config, molecule_type="protein")
    except (OSError, ValueError):
        return None
    for candidate in (
        config.project_dir / "results" / "ddg.txt",
        config.project_dir / "ddg.txt",
        config.project_dir / "results" / "summary.txt",
    ):
        if candidate.is_file():
            for token in candidate.read_text(encoding="utf-8", errors="replace").split():
                try:
                    return float(token)
                except ValueError:
                    continue
    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_stage(args: argparse.Namespace) -> int:
    result = stage(Path(args.config), dry_run=args.dry_run)
    print(f"Alanine scan: {len(result['selected'])} residue(s) to mutate -> ALA")
    for chain, resnum, resname in result["selected"]:
        print(f"  {chain}{resnum} {resname} -> ALA")
    if result["skipped"]:
        print("Skipped (ALA/GLY/PRO or non-standard):")
        for chain, resnum, resname in result["skipped"]:
            print(f"  {chain}{resnum} {resname}")
    print(f"Child configs: {result['child_configs'][0].parent}")
    print(f"Submit all with: bash {result['submit_script']}")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    import subprocess

    result = stage(Path(args.config), dry_run=False)
    script = result["submit_script"]
    if args.dry_run:
        print(f"bash {script}")
        return 0
    return subprocess.run(["bash", str(script)], check=False).returncode


def _cmd_analyze(args: argparse.Namespace) -> int:
    data = _load_scan(Path(args.config).expanduser().resolve())
    base_name = str(data.get("project", {}).get("name", "alascan"))
    base_dir = Path(args.config).expanduser().resolve().parent
    config_dir = base_dir / f"{base_name}_alascan" / "configs"
    if not config_dir.is_dir():
        raise ValueError(f"No staged scan found at {config_dir}; run `stage` first")
    rows = ["residue,ddg_kcal_per_mol"]
    for child in sorted(config_dir.glob("*.toml")):
        ddg = _result_ddg(child)
        rows.append(f"{child.stem},{'' if ddg is None else ddg}")
    out = base_dir / f"{base_name}_alascan" / "alanine_scan_ddg.csv"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(rows))
    return 0


def add_alanine_scan_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``alchemforge alanine-scan`` command."""
    parser = subparsers.add_parser(
        "alanine-scan",
        help="Alanine scanning (per-residue -> ALA resFEP runs) with GROMACS",
        description=(
            "Expand a residue selection into one resFEP residue->ALA run per "
            "residue, stage/submit them, and aggregate per-residue ddG."
        ),
    )
    commands = parser.add_subparsers(dest="alanine_scan_command", required=True)

    stage_cmd = commands.add_parser(
        "stage", help="Expand the scan and stage one resFEP run per residue"
    )
    stage_cmd.add_argument("config", type=Path)
    stage_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write child configs; do not stage the resFEP projects",
    )
    stage_cmd.set_defaults(func=_cmd_stage)

    submit = commands.add_parser("submit", help="Stage then submit every per-residue run")
    submit.add_argument("config", type=Path)
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=_cmd_submit)

    analyze = commands.add_parser(
        "analyze", help="Aggregate per-residue ddG into a CSV"
    )
    analyze.add_argument("config", type=Path)
    analyze.set_defaults(func=_cmd_analyze)
    return parser
