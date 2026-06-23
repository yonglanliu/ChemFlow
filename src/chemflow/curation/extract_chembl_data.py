# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import pandas as pd

from chembl_webresource_client.new_client import new_client
from src.config import CONFIG
from dataclasses import dataclass

# ============================================================
# Config
# ============================================================
@dataclass
class ChEMBLConfig:
    ligand_id_col: str = "molecule_chembl_id"
    structure_col: str = "SMILES"
    activity_col: str = "standard_value"

    standard_activity_col: str = "standard_value"
    standard_unit: str = "nM"

# ============================================================
# Target query
# ============================================================
def target_query_by_uniprot(uniprot_id):
    targets_api = new_client.target
    fields = CONFIG["chembl"]["target_query_fields"]

    targets = targets_api.get(
        target_components__accession=uniprot_id
    ).only(*fields)

    df = pd.DataFrame.from_records(targets)

    if df.empty:
        return pd.DataFrame()

    df["uniprot_id"] = uniprot_id

    return df


def fetch_bioactivity_data(target_chembl_id, query_fields, query_type, assay_fields):
    bioactivities_api = new_client.activity

    bioactivities = bioactivities_api.filter(
        target_chembl_id=target_chembl_id,
        type = query_type,
        relation = "=",  # Only exact matches
        assay_type__in=assay_fields # Binding and Functional assays
    ).only(*query_fields)

    df = pd.DataFrame.from_records(bioactivities)

    return df

def add_doi_data(df, cfg):
    documents_api = new_client.document
    document_ids = df[cfg.ligand_id_col].dropna().unique().tolist()
    if document_ids:
        documents = documents_api.filter(
            document_chembl_id__in=document_ids
        ).only("document_chembl_id", "doi")
        doi_df = pd.DataFrame.from_records(documents)
        if not doi_df.empty:
            df = df.merge(doi_df, on="document_chembl_id", how="left")
    return df

def add_compounds(df, cfg):
    compounds_api = new_client.molecule
    if cfg.ligand_id_col not in df.columns:
        df.reset_index(drop=True, inplace=True)
        return df.copy()
    compound_ids = df[cfg.ligand_id_col].dropna().unique().tolist()
    if compound_ids:
        compounds_provider = compounds_api.filter(
            molecule_chembl_id__in=compound_ids
        ).only(cfg.ligand_id_col, "molecule_structures")
        compound_df = pd.DataFrame.from_records(list(compounds_provider))
        if not compound_df.empty:

            def get_smiles(x):
                try:
                    return x["canonical_smiles"]
                except Exception:
                    return None

            compound_df["SMILES"] = compound_df["molecule_structures"].apply(get_smiles)
            compound_df = compound_df[[cfg.ligand_id_col, "SMILES"]].dropna()
            compound_df = compound_df.drop_duplicates(subset=[cfg.ligand_id_col], keep="first",)
            out_df = df.merge(compound_df, on=cfg.ligand_id_col, how="left")
        else:
            out_df = df.copy()
        out_df.reset_index(drop=True, inplace=True)
    if "SMILES" in out_df.columns:
        from src.utils.chem import safe_mol_wt
        out_df["mw"] = out_df["SMILES"].apply(safe_mol_wt)
    return out_df

if __name__ == "__main__":
    df = target_query_by_uniprot("P42336")
    print(df.head())