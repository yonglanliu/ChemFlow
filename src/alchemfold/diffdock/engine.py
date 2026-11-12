"""DiffDock-L docking engine for AlchemFold.

DiffDock is a diffusion model that docks a ligand into a receptor **structure** (unlike the
sequence-based co-folding engines). It does *blind* docking — no pocket constraint — and it
does not predict binding affinity; it outputs a pose **confidence** score.

DiffDock-L is NOT installed in the ``alchemfold`` env. It lives in its own conda env
(``chemflow diffdock-env install``, env file ``env/environment-diffdock.yaml``) because it needs
the PyTorch-Geometric stack. AlchemFold only builds DiffDock's input CSV and runs
``<diffdock-env python> -m inference`` from the DiffDock repo folder. Where that env and repo are
is found (first match wins) from:

  1. the config keys ``diffdock_python`` / ``diffdock_dir``
  2. ``$CHEMFLOW_DIFFDOCK_PYTHON`` / ``$CHEMFLOW_DIFFDOCK_DIR``
  3. ``~/.config/chemflow/diffdock.yaml`` (written by ``chemflow diffdock-env install``)
  4. a conda env named ``diffdock`` next to the running interpreter's env; ``<ChemFlow root>/DiffDock``

``diffdock_command`` (e.g. ``"/path/to/env/bin/python -m inference"``) overrides the interpreter.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

_LOCATION_FILE = Path.home() / ".config" / "chemflow" / "diffdock.yaml"
_NOT_INSTALLED = (
    "DiffDock is not installed (or not found). Install it in its own env with "
    "`chemflow diffdock-env install` (or the desktop Environments page), then run "
    "`chemflow diffdock-env warmup` once. To point at an existing install set "
    "`diffdock_python` + `diffdock_dir` in the config, or $CHEMFLOW_DIFFDOCK_PYTHON / "
    "$CHEMFLOW_DIFFDOCK_DIR."
)


def _recorded_location() -> dict:
    out: dict = {}
    try:
        for line in _LOCATION_FILE.read_text(encoding="utf-8").splitlines():
            m = re.match(r'\s*(dir|python)\s*:\s*"?([^"#]+?)"?\s*(#.*)?$', line)
            if m:
                out[m.group(1)] = m.group(2).strip()
    except OSError:
        pass
    return out


def _sibling_diffdock_python() -> str | None:
    parts = os.path.abspath(sys.executable).split(os.sep)
    if "envs" not in parts:
        return None
    root = os.sep.join(parts[: len(parts) - 1 - parts[::-1].index("envs") + 1])
    candidate = os.path.join(root, "diffdock", "bin", "python")
    return candidate if os.path.isfile(candidate) else None


def locate(conf) -> tuple[str | None, str | None]:
    """Return ``(python, repo_dir)`` of the DiffDock install, each ``None`` if unknown."""
    rec = _recorded_location()
    python = (getattr(conf, "diffdock_python", None)
              or os.environ.get("CHEMFLOW_DIFFDOCK_PYTHON")
              or rec.get("python") or _sibling_diffdock_python())
    repo = (getattr(conf, "diffdock_dir", None)
            or os.environ.get("CHEMFLOW_DIFFDOCK_DIR") or rec.get("dir"))
    if not repo:
        # <ChemFlow root>/DiffDock (sibling of src/) is where `chemflow diffdock-env install` puts it.
        cands = [Path(__file__).resolve().parents[3] / "DiffDock", Path.cwd() / "DiffDock",
                 Path.home() / "DiffDock"]
        repo = next((str(c) for c in cands if (c / "inference.py").is_file()), None)
    return (str(Path(python).expanduser()) if python else None,
            str(Path(repo).expanduser()) if repo else None)


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
    # A ligand_file is screened as a library, so skip the inline-ligand check here
    # (the library is validated by the config layer).
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
    import csv

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    protein = (getattr(conf, "proteins", []) or [])[0]
    ligand = (getattr(conf, "ligands", []) or [])[0]
    receptor_pdb = getattr(conf, "receptor_pdb", None) or ""
    sequence = "" if receptor_pdb else str(getattr(protein, "sequence", "") or "")
    name = str(getattr(protein, "id", "complex"))

    csv_path = workdir / "diffdock_input.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["complex_name", "protein_path", "ligand_description", "protein_sequence"])
        w.writerow([name, receptor_pdb, ligand.smiles, sequence])

    out_dir = Path(conf.out_dir)
    if not out_dir.is_absolute():
        out_dir = workdir / out_dir
    return {"protein_ligand_csv": csv_path}, out_dir


def write_library_csv(conf, library: list[dict], path: Path) -> Path:
    """One multi-row DiffDock input CSV for a whole ligand library (model loads once)."""
    import csv

    receptor_pdb = str(Path(getattr(conf, "receptor_pdb", None) or "").expanduser()) \
        if getattr(conf, "receptor_pdb", None) else ""
    protein = (getattr(conf, "proteins", []) or [None])[0]
    sequence = "" if receptor_pdb else str(getattr(protein, "sequence", "") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["complex_name", "protein_path", "ligand_description", "protein_sequence"])
        for lig in library:
            w.writerow([lig["id"], receptor_pdb, lig["smiles"], sequence])
    return path


def samples_per_complex(conf) -> int:
    n = getattr(conf, "diffdock_samples", None)
    if n:
        return int(n)
    ds = int(getattr(conf, "diffusion_samples", 1) or 1)
    return ds if ds > 1 else 10  # diffusion_samples defaults to 1 (Boltz); DiffDock wants ~10


def write_inference_config(conf, dest: Path) -> Path:
    """Write the DiffDock ``--config`` YAML carrying OUR samples / steps.

    DiffDock's ``inference.py`` loads ``--config`` (default ``default_inference_args.yaml``) AFTER
    parsing the command line and the YAML values OVERWRITE flags, so passing
    ``--samples_per_complex`` / ``--inference_steps`` alone is silently ignored. Instead copy the
    base config (``diffdock_config`` if set, else the repo's ``default_inference_args.yaml``) and
    override those keys in the copy.
    """
    import yaml

    _, repo = locate(conf)
    base = getattr(conf, "diffdock_config", None)
    candidates = [Path(base)] if base else []
    if repo:
        candidates.append(Path(repo) / "default_inference_args.yaml")
    data: dict = {}
    for cand in candidates:
        if cand.is_file():
            data = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
            break
    steps = int(getattr(conf, "diffdock_inference_steps", 20))
    data["samples_per_complex"] = samples_per_complex(conf)
    data["inference_steps"] = steps
    # DiffDock's own default is 20 scheduled steps with 19 performed (last step is noise-free).
    data["actual_steps"] = max(1, steps - 1)
    if getattr(conf, "diffdock_batch_size", None):
        # Batches the poses of ONE ligand on the GPU (ligands themselves are always processed one by one).
        data["batch_size"] = int(conf.diffdock_batch_size)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return dest


def build_command(conf, inputs: dict, out_dir: Path) -> list[str]:
    custom = getattr(conf, "diffdock_command", None)
    if custom:
        command = shlex.split(str(custom))
    else:
        python, _ = locate(conf)
        if not python:
            raise ValueError(_NOT_INSTALLED)
        if getattr(conf, "diffdock_dedupe_esm", True):
            # Launcher that embeds the receptor once per UNIQUE sequence (not once per ligand row).
            command = [python, str(Path(__file__).with_name("_shim.py"))]
        else:
            command = [python, "-m", "inference"]
    command += [
        "--protein_ligand_csv", str(inputs["protein_ligand_csv"]),
        "--out_dir", str(out_dir),
        "--samples_per_complex", str(samples_per_complex(conf)),
        "--inference_steps", str(int(getattr(conf, "diffdock_inference_steps", 20))),
    ]
    if getattr(conf, "diffdock_trajectory", False):
        # rank<k>_reverseprocess.pdb: one MODEL per denoising step (ligand only), per pose.
        command.append("--save_visualisation")
    # The generated config (not the command-line flags above) is what DiffDock actually uses.
    config = write_inference_config(
        conf, Path(inputs["protein_ligand_csv"]).parent / "diffdock_inference_args.yaml")
    command += ["--config", str(config)]
    model_dir = getattr(conf, "diffdock_model_dir", None)
    if model_dir:
        command += ["--model_dir", str(model_dir)]
    return command


def build_env(conf) -> dict:
    """Extra env for the DiffDock process: keep the ESM2 weights next to the repo (not in HOME)."""
    if os.environ.get("TORCH_HOME"):
        return {}
    _, repo = locate(conf)
    return {"TORCH_HOME": str(Path(repo) / "torch_cache")} if repo else {}


_TABLE_FILES = (".so3_omegas_array4.npy", ".so3_cdf_vals4.npy", ".so3_score_norms4.npy",
                ".so3_exp_score_norms4.npy", ".p.npy", ".score.npy")


def preflight(conf) -> list[str]:
    """One-time start-up work DiffDock will do on this run (so the user is told up front).

    A fresh install spends ~10 min before the first docking: it builds SO(2)/SO(3) lookup tables
    (written to files in DiffDock's working dir = the repo), downloads the DiffDock-L models and
    the ~2.5 GB ESM2 weights. `chemflow diffdock-env warmup` does all of it once. This matters most
    for SLURM arrays: tasks starting together on a fresh install all build — and write — the same
    table files at the same time.
    """
    _, repo = locate(conf)
    if not repo:
        return []
    r = Path(repo)
    msgs: list[str] = []
    if not all((r / f).is_file() for f in _TABLE_FILES):
        msgs.append(f"DiffDock's SO(2)/SO(3) lookup tables are not built yet in {r}: this run builds "
                    "them first (~5 min, CPU, one time). Array tasks starting together would all "
                    "write the same files — run `chemflow diffdock-env warmup` once BEFORE submitting.")
    model_dir = Path(getattr(conf, "diffdock_model_dir", None) or r / "workdir" / "v1.1" / "score_model")
    if not model_dir.exists():
        msgs.append("DiffDock-L models are not downloaded yet (needs internet): "
                    "run `chemflow diffdock-env warmup` once on a node with internet.")
    esm = Path(os.environ.get("TORCH_HOME") or r / "torch_cache") / "hub" / "checkpoints" / "esm2_t33_650M_UR50D.pt"
    if not esm.is_file():
        msgs.append("The ESM2 weights (~2.5 GB) are not cached yet: this run downloads them "
                    "(one time) — `chemflow diffdock-env warmup` does it ahead of time.")
    return msgs


def post_run(conf, out_dir: Path) -> None:
    """After a run: write a CLEAN trajectory next to each raw DiffDock ``rank<k>_reverseprocess.pdb``.

    The raw file has frozen hydrogens (long streaks) and out-of-order frames; the clean copy
    ``rank<k>_trajectory.pdb`` is chronological and heavy-atom only (see alchemfold.diffdock.traj).
    """
    if not getattr(conf, "diffdock_trajectory", False):
        return
    from alchemfold.diffdock.traj import clean_file

    n = 0
    for raw in sorted(Path(out_dir).glob("*/rank*_reverseprocess.pdb")):
        try:
            clean_file(raw, raw.with_name(raw.name.replace("_reverseprocess", "_trajectory")))
            n += 1
        except Exception as exc:  # never fail a finished docking over a viewer file
            print(f"  (could not clean trajectory {raw.name}: {exc})")
    if n:
        print(f"Wrote {n} cleaned trajectory file(s) (rank<k>_trajectory.pdb) in {out_dir}/<complex>/")


def build_cwd(conf) -> str | None:
    """``python -m inference`` must run from the DiffDock repo folder."""
    _, repo = locate(conf)
    if repo and not (Path(repo) / "inference.py").is_file():
        raise ValueError(f"diffdock_dir '{repo}' has no inference.py — {_NOT_INSTALLED}")
    return repo
