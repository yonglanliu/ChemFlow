"""Boltz-2 docking engine for AlchemFold.

Reuses the single-engine Boltz input/command builders and adds an optional
pocket (binding-site) constraint.
"""

from __future__ import annotations

from pathlib import Path

from alchemfold.boltz.run import build_boltz_input
from alchemfold.boltz.run import build_command as _boltz_build_command


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
    contacts = []
    for contact in (getattr(pocket, "contacts", []) or []):
        residue = str(getattr(contact, "residue", "")).strip()
        try:
            residue_value: object = int(residue)
        except ValueError:
            residue_value = residue      # atom name (ligand chain)
        contacts.append([str(contact.chain), residue_value])
    constraint = {"binder": str(pocket.binder), "contacts": contacts}
    max_distance = getattr(pocket, "max_distance", None)
    if max_distance is not None:
        constraint["max_distance"] = float(max_distance)
    if bool(getattr(pocket, "force", False)):
        constraint["force"] = True
    return {"pocket": constraint}


def _token(value) -> list:
    """Normalise one Boltz token to a plain list (numeric strings -> int)."""
    out = []
    for part in (value if isinstance(value, (list, tuple)) else [value]):
        text = str(part).strip()
        try:
            out.append(int(text))
        except ValueError:
            out.append(text)
    return out


def _extra_constraints(conf) -> list[dict]:
    """Pass-through ``bond`` / ``contact`` / inline ``pocket`` constraint entries."""
    entries: list[dict] = []
    for entry in (getattr(conf, "constraints", None) or []):
        bond = getattr(entry, "bond", None)
        contact = getattr(entry, "contact", None)
        pocket = getattr(entry, "pocket", None)
        if bond is not None:
            entries.append({"bond": {
                "atom1": _token(bond.atom1),
                "atom2": _token(bond.atom2),
            }})
        elif contact is not None:
            body: dict = {
                "token1": _token(contact.token1),
                "token2": _token(contact.token2),
            }
            max_distance = getattr(contact, "max_distance", None)
            if max_distance is not None:
                body["max_distance"] = float(max_distance)
            if bool(getattr(contact, "force", False)):
                body["force"] = True
            entries.append({"contact": body})
        elif pocket is not None:
            body = {"binder": str(pocket.binder),
                    "contacts": [_token(c) for c in (getattr(pocket, "contacts", []) or [])]}
            max_distance = getattr(pocket, "max_distance", None)
            if max_distance is not None:
                body["max_distance"] = float(max_distance)
            if bool(getattr(pocket, "force", False)):
                body["force"] = True
            entries.append({"pocket": body})
    return entries


def constraints(conf) -> list[dict] | None:
    """Every Boltz constraint for this config: the ``pocket:`` block plus extras."""
    out: list[dict] = []
    pocket = _pocket_constraint(conf)
    if pocket is not None:
        out.append(pocket)
    out.extend(_extra_constraints(conf))
    return out or None


def _template_entries(conf) -> list[dict] | None:
    """Boltz ``templates:`` entries (protein chains only; the ligand is never templated).

    Without ``force`` the template is an input feature the model may overrule;
    with it, a potential steers the backbone during diffusion and ``threshold``
    bounds the deviation in Angstrom.
    """
    from alchemfold.dock_config import _as_id_list

    entries: list[dict] = []
    for template in (getattr(conf, "templates", None) or []):
        cif = str(getattr(template, "cif", "") or "").strip()
        if not cif:
            continue
        # Absolute: the engine runs from the job folder, not the config's folder.
        entry: dict = {"cif": str(Path(cif).expanduser().resolve())}
        chain_ids = _as_id_list(getattr(template, "chain_id", None))
        if chain_ids:
            entry["chain_id"] = chain_ids
        template_ids = _as_id_list(getattr(template, "template_id", None))
        if template_ids:
            entry["template_id"] = template_ids
        if bool(getattr(template, "force", False)):
            entry["force"] = True
            threshold = getattr(template, "threshold", None)
            entry["threshold"] = float(threshold if threshold is not None else 2.0)
        elif getattr(template, "threshold", None) is not None:
            entry["threshold"] = float(template.threshold)
        entries.append(entry)
    return entries or None


def write_inputs(conf, workdir: Path) -> tuple[dict, Path]:
    import yaml

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    boltz_input = build_boltz_input(conf)
    all_constraints = constraints(conf)
    if all_constraints is not None:
        boltz_input["constraints"] = all_constraints
    templates = _template_entries(conf)
    if templates is not None:
        boltz_input["templates"] = templates

    input_yaml = workdir / "boltz_input.yaml"
    input_yaml.write_text(yaml.safe_dump(boltz_input, sort_keys=False), encoding="utf-8")

    out_dir = Path(conf.out_dir)
    if not out_dir.is_absolute():
        out_dir = workdir / out_dir
    return {"input_yaml": input_yaml}, out_dir


def build_command(conf, inputs: dict, out_dir: Path) -> list[str]:
    return _boltz_build_command(conf, inputs["input_yaml"], out_dir)
