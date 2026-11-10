"""One-shot membrane assembly: build-membrane → relax-membrane (→ optional MD).

Chains the existing per-step configs end-to-end with a single command, wiring each
step's output structure into the next step's input so a membrane system is
assembled in one go:

    build-membrane  (tile + trim lipid patch, solvate, ionize)
        → relax-membrane  (staged restrained pre-equilibration)
        → md              (optional, unrestrained)

The underlying steps write both PDB and mmCIF, so the assembled system comes out
in both formats (mmCIF is OpenFE-ready and avoids the PDB >99,999-atom overflow).

This module only orchestrates subprocess calls to the existing step modules; the
heavy OpenMM work lives in ``membrane_build`` / ``membrane_relax`` / ``md_run``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_yaml(data: dict, path: Path) -> Path:
    import yaml

    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return Path(path)


def _run_module(module: str, config_path: Path, *, dry_run: bool, extra=None) -> None:
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    if extra:
        cmd += list(extra)
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"assembly step failed (exit {rc}): {module}")


def _chained_config(src_cfg: Path, new_input: Path) -> Path:
    """Write a temp copy of ``src_cfg`` (in its own dir, to preserve relative-path
    resolution) with ``input_pdb`` overridden to ``new_input`` (absolute)."""
    data = _load_yaml(src_cfg)
    data["input_pdb"] = str(Path(new_input).resolve())
    tmp = src_cfg.parent / f"_assemble_{src_cfg.stem}.yaml"
    return _write_yaml(data, tmp)


def _relax_production_pdb(relax_conf) -> Path:
    """Final relaxed PDB path from a loaded membrane-relax config."""
    stages = getattr(relax_conf, "stages", []) or []
    prod = next((s for s in stages if getattr(s, "name", "") == "production"),
                stages[-1] if stages else None)
    name = getattr(prod, "out_pdb", "relaxed_production.pdb") if prod else "relaxed_production.pdb"
    return Path(relax_conf.workdir) / name


def _soft_validate(validate, conf) -> None:
    """Validate a chained step, tolerating only the 'input_pdb' check (that input
    is produced by the previous step, so it won't exist yet)."""
    try:
        validate(conf)
    except ValueError as exc:
        if "input_pdb" in str(exc):
            return
        raise


def validate_configs(build_cfg, relax_cfg, md_cfg=None) -> None:
    """Validate the build config fully and the chained relax/md configs structurally."""
    from chemflow.fe.openmm_modeling import membrane_build_config as bc
    from chemflow.fe.openmm_modeling import membrane_relax_config as rc

    bc.validate_config(bc.load_config(Path(build_cfg).expanduser().resolve()))
    _soft_validate(rc.validate_config, rc.load_config(Path(relax_cfg).expanduser().resolve()))
    if md_cfg:
        from chemflow.fe.openmm_modeling import md_config as mc

        _soft_validate(mc.validate_config, mc.load_config(Path(md_cfg).expanduser().resolve()))


def assemble(build_cfg, relax_cfg, md_cfg=None, *, dry_run: bool = False) -> Path:
    """Run build → relax (→ md), chaining outputs to inputs. Returns the final PDB."""
    from chemflow.fe.openmm_modeling import membrane_build_config as bc
    from chemflow.fe.openmm_modeling import membrane_relax_config as rc

    build_cfg = Path(build_cfg).expanduser().resolve()
    relax_cfg = Path(relax_cfg).expanduser().resolve()

    bconf = bc.load_config(build_cfg)
    bc.validate_config(bconf)
    build_out = Path(bconf.workdir) / bconf.output_pdb

    rconf = rc.load_config(relax_cfg)
    _soft_validate(rc.validate_config, rconf)
    relax_out = _relax_production_pdb(rconf)

    md_cfg_path = None
    if md_cfg:
        from chemflow.fe.openmm_modeling import md_config as mc

        md_cfg_path = Path(md_cfg).expanduser().resolve()
        _soft_validate(mc.validate_config, mc.load_config(md_cfg_path))

    # 1) build the membrane
    print("=== [1/%d] build-membrane ===" % (3 if md_cfg_path else 2))
    _run_module("chemflow.fe.openmm_modeling.membrane_build", build_cfg, dry_run=dry_run)

    # 2) relax, with input chained to the build output
    print("=== [2/%d] relax-membrane ===" % (3 if md_cfg_path else 2))
    relax_run_cfg = relax_cfg if dry_run else _chained_config(relax_cfg, build_out)
    _run_module("chemflow.fe.openmm_modeling.membrane_relax", relax_run_cfg, dry_run=dry_run)

    final = relax_out
    # 3) optional MD, chained to the relaxed output
    if md_cfg_path:
        print("=== [3/3] md ===")
        md_run_cfg = md_cfg_path if dry_run else _chained_config(md_cfg_path, relax_out)
        _run_module("chemflow.fe.openmm_modeling.md_run", md_run_cfg, dry_run=dry_run)

    print(f"\nAssembled membrane system → {final}  (+ {final.with_suffix('.cif')})")
    if not md_cfg_path:
        print("Feed the .cif into OpenFE:  chemflow openfe rbfe ... (protein_pdb: <...>.cif)")
    return final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="One-shot membrane assembly: build → relax (→ md), chained.",
    )
    parser.add_argument("--build", required=True, help="membrane-build YAML config")
    parser.add_argument("--relax", required=True, help="membrane-relax YAML config")
    parser.add_argument("--md", default=None, help="optional MD YAML config")
    parser.add_argument("--dry-run", action="store_true",
                        help="print each step's command without running it")
    args = parser.parse_args(argv)
    assemble(args.build, args.relax, md_cfg=args.md, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
