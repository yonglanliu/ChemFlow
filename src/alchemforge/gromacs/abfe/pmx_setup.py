"""Call PMX's ABFE setup while honoring AlchemForge restraint parameters."""

from __future__ import annotations

import argparse
from functools import partial
from types import SimpleNamespace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-top", required=True)
    parser.add_argument("--protein-gro", required=True)
    parser.add_argument("--ligand-itp", required=True)
    parser.add_argument("--ligand-gro", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--distance-k", required=True, type=float)
    parser.add_argument("--angle-k", required=True, type=float)
    parser.add_argument("--dihedral-k", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    from pmx.alchemy import AbsRestraints
    from pmx.scripts import setup_abfe

    setup_abfe.AbsRestraints = partial(
        AbsRestraints,
        kbond=args.distance_k,
        kangle=args.angle_k,
        kdihedral=args.dihedral_k,
        T=args.temperature,
    )
    pmx_args = SimpleNamespace(
        pro_top=args.protein_top,
        lig_top=args.ligand_itp,
        pro_crd=args.protein_gro,
        lig_crd=args.ligand_gro,
        build=True,
        doublebox=False,
        longest_axis=False,
        keep_intra=False,
        lig_ids=None,
        pro_ids=None,
        restr_switch_on=True,
        seed=args.seed,
    )
    setup_abfe.main(pmx_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
