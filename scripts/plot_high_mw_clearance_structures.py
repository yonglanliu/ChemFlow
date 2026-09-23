#!/usr/bin/env python3
"""Render an atlas of high-molecular-weight structures in the CL-1 dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("dataset/curated/clearance_chembl_biogen")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "chembl_biogen_clearance_training_multitask.csv",
    )
    parser.add_argument("--threshold", type=float, default=700.0)
    parser.add_argument("--molecules-per-page", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "high_mw_structures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive.")
    if args.molecules_per_page < 1:
        raise ValueError("--molecules-per-page must be at least one.")

    frame = pd.read_csv(args.input).drop_duplicates("InChIKey").copy()
    frame["Molecule"] = frame["SMILES"].map(lambda value: Chem.MolFromSmiles(str(value)))
    invalid = int(frame["Molecule"].isna().sum())
    if invalid:
        print(f"Skipping {invalid} structures that RDKit could not parse.")
    frame = frame.loc[frame["Molecule"].notna()].copy()
    frame["MolecularWeight"] = frame["Molecule"].map(Descriptors.MolWt)
    selected = frame.loc[frame["MolecularWeight"].gt(args.threshold)].copy()
    selected = selected.sort_values(
        ["MolecularWeight", "Ligand_ID"], ascending=[False, True]
    ).reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No structures exceed {args.threshold:g} Da.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_columns = [
        "Ligand_ID",
        "SMILES",
        "InChIKey",
        "Source",
        "MolecularWeight",
        "Log10_HLM_CLint_mL_min_kg",
        "Log10_RLM_CLint_mL_min_kg",
        "Log10_MLM_CLint_mL_min_kg",
    ]
    index_path = args.output_dir / "cl1_mw_over_700_index.csv"
    selected[index_columns].to_csv(index_path, index=False)

    drawing_options = Draw.MolDrawOptions()
    drawing_options.legendFontSize = 20
    drawing_options.baseFontSize = 0.7
    pages: list[Image.Image] = []
    page_count = math.ceil(len(selected) / args.molecules_per_page)
    for page_index, start in enumerate(
        range(0, len(selected), args.molecules_per_page), start=1
    ):
        page = selected.iloc[start : start + args.molecules_per_page]
        legends = [
            f"{row.Ligand_ID} | MW={row.MolecularWeight:.1f}"
            for row in page.itertuples()
        ]
        image = Draw.MolsToGridImage(
            list(page["Molecule"]),
            molsPerRow=4,
            subImgSize=(420, 320),
            legends=legends,
            useSVG=False,
            drawOptions=drawing_options,
        ).convert("RGB")
        page_path = args.output_dir / f"cl1_mw_over_700_page_{page_index:02d}.png"
        image.save(page_path, dpi=(300, 300))
        pages.append(image)
        print(f"Saved page {page_index}/{page_count}: {page_path}")

    pdf_path = args.output_dir / "cl1_mw_over_700_structure_atlas.pdf"
    pages[0].save(
        pdf_path,
        save_all=True,
        append_images=pages[1:],
        resolution=300,
    )
    print(f"\nStructures rendered: {len(selected)}")
    print(f"CSV index: {index_path.resolve()}")
    print(f"Multipage PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
