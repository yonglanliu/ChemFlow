# app/similarity_dashboard.py

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs


# -----------------------------
# Similarity Calculator
# -----------------------------
class SimilarityCalculator:
    def __init__(
        self,
        radius=2,
        nBits=2048,
        fp_type="ecfp",
        similarity_metric="tanimoto",
    ):
        fp_type = fp_type.lower()
        similarity_metric = similarity_metric.lower()

        if fp_type not in ["ecfp", "fcfp"]:
            raise ValueError("fp_type must be 'ecfp' or 'fcfp'")

        if similarity_metric not in ["tanimoto", "cosine", "euclidean"]:
            raise ValueError("similarity_metric must be tanimoto, cosine, or euclidean")

        self.radius = radius
        self.nBits = nBits
        self.useFeatures = fp_type == "fcfp"
        self.similarity_metric = similarity_metric

    def fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.radius,
            nBits=self.nBits,
            useFeatures=self.useFeatures,
        )

        arr = np.zeros((self.nBits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

    def compare(self, fp1, fp2):
        if fp1 is None or fp2 is None:
            return np.nan

        if self.similarity_metric == "tanimoto":
            intersection = np.logical_and(fp1, fp2).sum()
            union = np.logical_or(fp1, fp2).sum()
            return intersection / union if union > 0 else 0.0

        if self.similarity_metric == "cosine":
            norm1 = np.linalg.norm(fp1)
            norm2 = np.linalg.norm(fp2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(fp1, fp2) / (norm1 * norm2)

        if self.similarity_metric == "euclidean":
            return np.linalg.norm(fp1 - fp2)

    def search_dataframe(self, query_smiles, df, smiles_col="smiles"):
        query_fp = self.fingerprint(query_smiles)

        results = []

        for idx, row in df.iterrows():
            smi = row[smiles_col]
            fp = self.fingerprint(smi)
            score = self.compare(query_fp, fp)

            results.append(score)

        out = df.copy()
        out["similarity_score"] = results

        if self.similarity_metric == "euclidean":
            out = out.sort_values("similarity_score", ascending=True)
        else:
            out = out.sort_values("similarity_score", ascending=False)

        return out

    def similarity_matrix(self, df, smiles_col="smiles"):
        smiles = df[smiles_col].dropna().tolist()
        fps = [self.fingerprint(smi) for smi in smiles]

        n = len(fps)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.compare(fps[i], fps[j])

        return pd.DataFrame(matrix, index=smiles, columns=smiles)
