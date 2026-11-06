"""Boltz-2 docking engine for AlchemFold.

Reuses the single-engine Boltz input/command builders and adds an optional
pocket (binding-site) constraint.
"""

from __future__ import annotations

from pathlib import Path

from alchemfold.boltz_run import build_boltz_input
from alchemfold.boltz_run import build_command as _boltz_build_command


def validate(conf) -> None:
    """Boltz-specific checks (affinity binder must be a declared ligand)."""
    if getattr(conf, "predict_affinity", False) and not getattr(conf, "ligand_file", None):
        binder = getattr(conf, "affinity_binder", None)
        ligand_ids = {str(l.id).strip() for l in (getattr(conf, "ligands", []) or [])}
        if not binder or str(binder) not in ligand_ids:
            raise ValueError(
                "predict_affinity is true but 'affinity_binder' is not a ligand id "
                f"{sorted(ligand_ids)}"
            )


def _pocket_constraint(conf) -> dict | None:
    pocket = getattr(conf, "pocket", None)
    if not pocket:
        return None
    contacts = [
        [str(c.chain), int(c.residue)]
        for c in (getattr(pocket, "contacts", []) or [])
    ]
    constraint = {"binder": str(pocket.binder), "contacts": contacts}
    max_distance = getattr(pocket, "max_distance", None)
    if max_distance is not None:
        constraint["max_distance"] = float(max_distance)
    return {"pocket": constraint}


def write_inputs(conf, workdir: Path) -> tuple[dict, Path]:
    import yaml

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    boltz_input = build_boltz_input(conf)
    constraint = _pocket_constraint(conf)
    if constraint is not None:
        boltz_input["constraints"] = [constraint]

    input_yaml = workdir / "boltz_input.yaml"
    input_yaml.write_text(yaml.safe_dump(boltz_input, sort_keys=False), encoding="utf-8")

    out_dir = Path(conf.out_dir)
    if not out_dir.is_absolute():
        out_dir = workdir / out_dir
    return {"input_yaml": input_yaml}, out_dir


def build_command(conf, inputs: dict, out_dir: Path) -> list[str]:
    return _boltz_build_command(conf, inputs["input_yaml"], out_dir)
