"""Shared validation and MDP settings for prepared GROMACS membrane systems."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_LIPID_RESNAMES = (
    "POPC",
    "POPE",
    "POPS",
    "POPG",
    "DOPC",
    "DOPE",
    "DOPS",
    "DPPC",
    "DMPC",
    "DLPC",
    "CHOL",
    "PSM",
    # Amber Lipid17/Lipid21 with charmmlipid2amber represents one
    # phosphatidylcholine
    # molecule as linked PA (glycerol/phosphate), PC (choline), and OL tails.
    "PA",
    "PC",
    "OL",
)

_AMBER_SPLIT_POPC_COMPONENTS = frozenset({"PA", "PC", "OL"})


def count_lipid_molecules_pdb(
    path: Path, lipid_resnames: tuple[str, ...] = DEFAULT_LIPID_RESNAMES
) -> int:
    """Count whole-residue and modular Amber lipids in a PDB coordinate file."""
    allowed = {name.upper() for name in lipid_resnames}
    residues: set[tuple[str, str, str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        extended = line[17:21].strip().upper()
        name = extended if line[20].strip() else line[17:20].strip().upper()
        if name in allowed:
            residues.add((line[21:22], line[22:26], line[26:27], name))
    whole = sum(
        1 for *_, name in residues if name not in _AMBER_SPLIT_POPC_COMPONENTS
    )
    components = {
        name: sum(1 for *_, observed in residues if observed == name)
        for name in _AMBER_SPLIT_POPC_COMPONENTS
        if name in allowed
    }
    split_popc = (
        min(components.values())
        if components.keys() == _AMBER_SPLIT_POPC_COMPONENTS
        and all(components.values())
        else 0
    )
    return whole + split_popc


def _positive_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two numbers")
    result = (float(value[0]), float(value[1]))
    if any(item <= 0 for item in result):
        raise ValueError(f"{label} values must be greater than zero")
    return result


@dataclass(frozen=True)
class MembraneSettings:
    enabled: bool
    normal: str
    lipid_resnames: tuple[str, ...]
    minimum_lipids: int
    barostat: str
    reference_pressure_bar: tuple[float, float]
    compressibility_bar: tuple[float, float]
    refcoord_scaling: str

    @classmethod
    def from_system(cls, system: dict[str, Any]) -> MembraneSettings:
        system_type = str(system.get("type", "soluble")).lower()
        if system_type not in {"soluble", "membrane"}:
            raise ValueError("[system].type must be 'soluble' or 'membrane'")
        normal = str(system.get("membrane_normal", "z")).lower()
        if system_type == "membrane" and normal != "z":
            raise ValueError(
                "GROMACS semi-isotropic coupling requires membrane_normal = 'z'"
            )
        raw_resnames = system.get("lipid_resnames", list(DEFAULT_LIPID_RESNAMES))
        if (
            not isinstance(raw_resnames, list)
            or not raw_resnames
            or not all(isinstance(item, str) and item.strip() for item in raw_resnames)
        ):
            raise ValueError("[system].lipid_resnames must be a non-empty string array")
        minimum = int(system.get("minimum_lipids", 20))
        if minimum < 1:
            raise ValueError("[system].minimum_lipids must be at least 1")
        scaling = str(system.get("refcoord_scaling", "com")).lower()
        if scaling not in {"no", "all", "com"}:
            raise ValueError("[system].refcoord_scaling must be no, all, or com")
        return cls(
            enabled=system_type == "membrane",
            normal=normal,
            lipid_resnames=tuple(item.strip().upper() for item in raw_resnames),
            minimum_lipids=minimum,
            barostat=str(system.get("barostat", "Parrinello-Rahman")),
            reference_pressure_bar=_positive_pair(
                system.get("reference_pressure_bar", [1.0, 1.0]),
                "[system].reference_pressure_bar",
            ),
            compressibility_bar=_positive_pair(
                system.get("compressibility_bar", [4.5e-5, 4.5e-5]),
                "[system].compressibility_bar",
            ),
            refcoord_scaling=scaling,
        )

    @property
    def system_type(self) -> str:
        return "membrane" if self.enabled else "soluble"

    def shell_variables(self) -> dict[str, str | int]:
        return {
            "SYSTEM_TYPE": self.system_type,
            "MEMBRANE_NORMAL": self.normal,
            "LIPID_RESNAMES": " ".join(self.lipid_resnames),
            "MINIMUM_LIPIDS": self.minimum_lipids,
            "MEMBRANE_BAROSTAT": self.barostat,
            "MEMBRANE_REF_P": " ".join(map(str, self.reference_pressure_bar)),
            "MEMBRANE_COMPRESSIBILITY": " ".join(map(str, self.compressibility_bar)),
            "MEMBRANE_REFCOORD_SCALING": self.refcoord_scaling,
        }


def validate_membrane_gro(path: Path, settings: MembraneSettings) -> int:
    """Validate periodic box data and return the number of lipid residues."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        atom_count = int(lines[1].strip())
        atoms = lines[2 : 2 + atom_count]
        box = [float(item) for item in lines[2 + atom_count].split()]
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot parse GROMACS coordinate file: {path}") from error
    if len(box) not in {3, 9} or any(value <= 0 for value in box[:3]):
        raise ValueError(f"Membrane system lacks valid periodic box vectors: {path}")
    whole_lipids = {
        (line[:5].strip(), line[5:10].strip().upper())
        for line in atoms
        if len(line) >= 10
        and line[5:10].strip().upper() in settings.lipid_resnames
        and line[5:10].strip().upper() not in _AMBER_SPLIT_POPC_COMPONENTS
    }
    component_counts = {
        name: len(
            {
                line[:5].strip()
                for line in atoms
                if len(line) >= 10 and line[5:10].strip().upper() == name
            }
        )
        for name in _AMBER_SPLIT_POPC_COMPONENTS
        if name in settings.lipid_resnames
    }
    split_popc = (
        min(component_counts.values())
        if component_counts.keys() == _AMBER_SPLIT_POPC_COMPONENTS
        and all(component_counts.values())
        else 0
    )
    lipid_count = len(whole_lipids) + split_popc
    if lipid_count < settings.minimum_lipids:
        raise ValueError(
            f"Membrane system {path} contains {lipid_count} recognized lipid molecules; "
            f"expected at least {settings.minimum_lipids}. Check lipid_resnames."
        )
    return lipid_count


def set_mdp_value(text: str, key: str, value: str) -> str:
    variants = {key, key.replace("-", "_"), key.replace("_", "-")}
    pattern = re.compile(
        rf"(?mi)^\s*(?:{'|'.join(re.escape(item) for item in variants)})\s*=.*$"
    )
    replacement = f"{key:<24} = {value}"
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text.rstrip() + f"\n{replacement}\n"


def apply_membrane_pressure(text: str, settings: MembraneSettings) -> str:
    """Apply the z-normal semi-isotropic pressure-coupling contract."""
    values = {
        "pcoupl": settings.barostat,
        "pcoupltype": "semiisotropic",
        "ref-p": " ".join(map(str, settings.reference_pressure_bar)),
        "compressibility": " ".join(map(str, settings.compressibility_bar)),
        "refcoord-scaling": settings.refcoord_scaling,
    }
    for key, value in values.items():
        text = set_mdp_value(text, key, value)
    return text
