#!/usr/bin/env python3
"""Render structures within a molecular-weight interval for clearance datasets."""

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
    parser.add_argument("--minimum-mw", type=float, default=600.0)
    parser.add_argument("--maximum-mw", type=float, default=700.0)
    parser.add_argument("--molecules-per-page", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "mw_600_700_structures",
    )
    parser.add_argument(
        "--cl1",
        type=Path,
        default=root / "chembl_biogen_clearance_training_multitask.csv",
    )
    parser.add_argument(
        "--cl2",
        type=Path,
        default=root / "expansionrx_clearance_train.csv",
    )
    parser.add_argument(
        "--cl-test",
        type=Path,
        default=root / "expansionrx_clearance_test.csv",
    )
    return parser.parse_args()


def load_dataset(path: Path, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(path).drop_duplicates("InChIKey").copy()
    frame.insert(0, "Dataset", dataset)
    frame["Molecule"] = frame["SMILES"].map(lambda value: Chem.MolFromSmiles(str(value)))
    frame = frame.loc[frame["Molecule"].notna()].copy()
    frame["MolecularWeight"] = frame["Molecule"].map(Descriptors.MolWt)
    return frame


def main() -> None:
    args = parse_args()
    if args.minimum_mw < 0 or args.maximum_mw <= args.minimum_mw:
        raise ValueError("The molecular-weight interval is invalid.")
    if args.molecules_per_page < 1:
        raise ValueError("--molecules-per-page must be at least one.")

    frame = pd.concat(
        [
            load_dataset(args.cl1, "CL-1"),
            load_dataset(args.cl2, "CL-2"),
            load_dataset(args.cl_test, "CL-Test"),
        ],
        ignore_index=True,
    )
    selected = frame.loc[
        frame["MolecularWeight"].between(
            args.minimum_mw, args.maximum_mw, inclusive="both"
        )
    ].copy()
    selected = selected.sort_values(
        ["MolecularWeight", "Dataset", "Ligand_ID"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            f"No structures are between {args.minimum_mw:g} and "
            f"{args.maximum_mw:g} Da."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"clearance_mw_{args.minimum_mw:g}_{args.maximum_mw:g}"
    index_columns = [
        "Dataset",
        "Ligand_ID",
        "SMILES",
        "InChIKey",
        "Source",
        "MolecularWeight",
        "Log10_HLM_CLint_mL_min_kg",
        "Log10_RLM_CLint_mL_min_kg",
        "Log10_MLM_CLint_mL_min_kg",
    ]
    index_path = args.output_dir / f"{stem}_index.csv"
    selected[index_columns].to_csv(index_path, index=False)

    drawing_options = Draw.MolDrawOptions()
    drawing_options.legendFontSize = 18
    drawing_options.baseFontSize = 0.7
    pages: list[Image.Image] = []
    page_count = math.ceil(len(selected) / args.molecules_per_page)
    for page_index, start in enumerate(
        range(0, len(selected), args.molecules_per_page), start=1
    ):
        page = selected.iloc[start : start + args.molecules_per_page]
        legends = [
            f"{row.Dataset} | {row.Ligand_ID} | MW={row.MolecularWeight:.1f}"
            for row in page.itertuples()
        ]
        image = Draw.MolsToGridImage(
            list(page["Molecule"]),
            molsPerRow=4,
            subImgSize=(450, 320),
            legends=legends,
            useSVG=False,
            drawOptions=drawing_options,
        ).convert("RGB")
        page_path = args.output_dir / f"{stem}_page_{page_index:02d}.png"
        image.save(page_path, dpi=(300, 300))
        pages.append(image)
        print(f"Saved page {page_index}/{page_count}: {page_path}")

    pdf_path = args.output_dir / f"{stem}_structure_atlas.pdf"
    pages[0].save(
        pdf_path,
        save_all=True,
        append_images=pages[1:],
        resolution=300,
    )
    print("\nStructures by dataset:")
    print(selected["Dataset"].value_counts().sort_index().to_string())
    print(f"Total structures: {len(selected)}")
    print(f"CSV index: {index_path.resolve()}")
    print(f"Multipage PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
