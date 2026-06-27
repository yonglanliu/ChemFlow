import pandas as pd
import numpy as np
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


def load_molecule_2d_database(database_file, smiles_col="SMILES", name_col=None):
    database_file = Path(database_file)
    suffix = database_file.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(database_file)
    elif suffix in [".tsv", ".txt"]:
        df = pd.read_csv(database_file, sep="\t")
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(database_file)
    elif suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(database_file), removeHs=False)
        rows = []

        for i, mol in enumerate(suppl):
            if mol is None:
                continue

            props = mol.GetPropsAsDict()
            rows.append(
                {
                    "Name": props.get("_Name", f"Mol_{i+1}"),
                    "SMILES": Chem.MolToSmiles(Chem.RemoveHs(mol)),
                    "Mol": mol,
                }
            )

        return pd.DataFrame(rows)

    else:
        raise ValueError(f"Unsupported database file format: {suffix}")

    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found.")

    mols = []
    valid_rows = []

    for _, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        row = row.to_dict()
        row["SMILES"] = Chem.MolToSmiles(mol)
        row["Mol"] = mol

        if name_col and name_col in row:
            row["Name"] = row[name_col]
        elif "Name" not in row:
            row["Name"] = f"Mol_{len(valid_rows) + 1}"

        valid_rows.append(row)

    return pd.DataFrame(valid_rows)

def load_molecule_3d_database(database_file, smiles_col="SMILES", name_col=None):
    database_file = Path(database_file)
    suffix = database_file.suffix.lower()

    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(database_file), removeHs=False)
        rows = []

        for i, mol in enumerate(suppl):
            if mol is None:
                continue

            props = mol.GetPropsAsDict()
            rows.append(
                {
                    "Name": props.get("_Name", f"Mol_{i+1}"),
                    "SMILES": Chem.MolToSmiles(Chem.RemoveHs(mol)),
                    "Mol": mol,
                }
            )

        return pd.DataFrame(rows)

    else:
        raise ValueError(f"Unsupported 3D database file format: {suffix}")