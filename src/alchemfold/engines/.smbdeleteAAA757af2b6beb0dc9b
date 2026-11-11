"""DiffDock docking engine for AlchemFold.

DiffDock is a diffusion model that docks a ligand into a receptor **structure**
(unlike the sequence-based co-folding engines). It performs *blind* docking, so
it does not take a pocket constraint, and it does not predict binding affinity.

Inputs are written as DiffDock's protein-ligand CSV; the command runs DiffDock's
``inference`` entry point.

NOTE: DiffDock is distributed as a GitHub repository (not a PyPI package), and
its CLI flags vary between DiffDock / DiffDock-L releases. Set ``diffdock_command``
to however you invoke inference (e.g. ``python -m inference`` from the repo, or a
wrapper), and use ``run --dry-run`` to check the command for your checkout.
"""

from __future__ import annotations

import shlex
from pathlib import Path


def validate(conf) -> None:
    if getattr(conf, "predict_affinity", False):
        raise ValueError(
            "predict_affinity is only available with engine: boltz (Boltz-2). "
            "DiffDock does not predict affinity."
        )
    if getattr(conf, "pocket", None):
        raise ValueError(
            "DiffDock performs blind docking and does not accept a pocket "
            "constraint. Remove the 'pocket' block, or use engine: boltz/chai."
        )
    proteins = getattr(conf, "proteins", []) or []
    if len(proteins) != 1:
        raise ValueError("DiffDock docks into exactly one receptor chain")
    # A ligand_file is screened one ligand per run, so skip the inline-ligand
    # count/SMILES check here (the library is validated by the config layer).
    if not getattr(conf, "ligand_file", None):
        ligands = getattr(conf, "ligands", []) or []
        if len(ligands) != 1 or not getattr(ligands[0], "smiles", None):
            raise ValueError("DiffDock needs exactly one ligand given as 'smiles'")
    if not getattr(conf, "receptor_pdb", None) and not str(
        getattr(proteins[0], "sequence", "")
    ).strip():
        raise ValueError(
            "DiffDock needs a 'receptor_pdb' structure (or a protein 'sequence' "
            "to fold with ESMFold)"
        )


def write_inputs(conf, workdir: Path) -> tuple[dict, Path]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    protein = (getattr(conf, "proteins", []) or [])[0]
    ligand = (getattr(conf, "ligands", []) or [])[0]
    receptor_pdb = getattr(conf, "receptor_pdb", None) or ""
    sequence = "" if receptor_pdb else str(getattr(protein, "sequence", "") or "")
    name = str(getattr(protein, "id", "complex"))

    csv_path = workdir / "diffdock_input.csv"
    csv_path.write_text(
        "complex_name,protein_path,ligand_description,protein_sequence\n"
        f"{name},{receptor_pdb},{ligand.smiles},{sequence}\n",
        encoding="utf-8",
    )

    out_dir = Path(conf.out_dir)
    if not out_dir.is_absolute():
        out_dir = workdir / out_dir
    return {"protein_ligand_csv": csv_path}, out_dir


def build_command(conf, inputs: dict, out_dir: Path) -> list[str]:
    command = shlex.split(str(getattr(conf, "diffdock_command", "python -m inference")))
    command += [
        "--protein_ligand_csv", str(inputs["protein_ligand_csv"]),
        "--out_dir", str(out_dir),
        "--samples_per_complex", str(int(conf.diffusion_samples)),
        "--inference_steps", str(int(getattr(conf, "diffdock_inference_steps", 20))),
    ]
    config = getattr(conf, "diffdock_config", None)
    if config:
        command += ["--config", str(config)]
    model_dir = getattr(conf, "diffdock_model_dir", None)
    if model_dir:
        command += ["--model_dir", str(model_dir)]
    return command
