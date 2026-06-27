from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
    GraphDescriptors,
    rdFingerprintGenerator,
    rdShapeHelpers,
    rdMolAlign,
)


class SimilarityCalculator:
    def __init__(
        self,
        mode="2d_fingerprint",
        metric="tanimoto",
        radius=2,
        n_bits=2048,
        use_features=False,
    ):
        """
        mode:
            2d_fingerprint
            2d_descriptor
            topological_descriptor
            3d_shape

        metric:
            tanimoto, dice, cosine, euclidean, manhattan,
            mcconnaughey, shape_tanimoto, shape_protrude
        """
        self.mode = mode.lower()
        self.metric = metric.lower()
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features

        self.fpgen = self._get_morgan_generator(
            radius=self.radius,
            n_bits=self.n_bits,
            use_features=self.use_features,
        )

    # -----------------------------
    # Molecule handling
    # -----------------------------
    @staticmethod
    def mol_from_smiles(smiles):
        if pd.isna(smiles):
            return None
        mol = Chem.MolFromSmiles(str(smiles))
        return mol

    @staticmethod
    def mol_from_file(path):
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in [".sdf", ".sd"]:
            suppl = Chem.SDMolSupplier(str(path), removeHs=False)
            mols = [m for m in suppl if m is not None]
            return mols[0] if mols else None

        if suffix == ".mol":
            return Chem.MolFromMolFile(str(path), removeHs=False)

        if suffix == ".mol2":
            return Chem.MolFromMol2File(str(path), removeHs=False)

        if suffix == ".pdb":
            return Chem.MolFromPDBFile(str(path), removeHs=False)

        raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def prepare_3d_mol_from_smiles(smiles, seed=42):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        status = AllChem.EmbedMolecule(
            mol,
            AllChem.ETKDGv3(),
            randomSeed=seed,
        )

        if status != 0:
            return None

        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                pass

        return mol

    @staticmethod
    def has_3d_coordinates(mol):
        if mol is None or mol.GetNumConformers() == 0:
            return False

        conf = mol.GetConformer()
        return conf.Is3D()

    # -----------------------------
    # 2D fingerprints
    # -----------------------------
    @staticmethod
    def _get_morgan_generator(radius=2, n_bits=2048, use_features=False):
        if use_features:
            return rdFingerprintGenerator.GetMorganGenerator(
                radius=int(radius),
                fpSize=int(n_bits),
                atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
            )

        return rdFingerprintGenerator.GetMorganGenerator(
            radius=int(radius),
            fpSize=int(n_bits),
        )

    def calculate_fingerprint(self, smiles):
        mol = self.mol_from_smiles(smiles)
        if mol is None:
            return None
        return self.fpgen.GetFingerprint(mol)

    @staticmethod
    def fingerprint_to_array(fp, n_bits=2048):
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    # -----------------------------
    # 2D physicochemical descriptors
    # -----------------------------
    def calculate_2d_descriptors(self, smiles):
        mol = self.mol_from_smiles(smiles)
        if mol is None:
            return None

        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "RotatableBonds": Lipinski.NumRotatableBonds(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "FractionCsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        }

    def calculate_topological_descriptors(self, smiles):
        mol = self.mol_from_smiles(smiles)
        if mol is None:
            return None

        return {
            "Chi0": GraphDescriptors.Chi0(mol),
            "Chi1": GraphDescriptors.Chi1(mol),
            "Chi2n": GraphDescriptors.Chi2n(mol),
            "Chi3n": GraphDescriptors.Chi3n(mol),
            "Chi4n": GraphDescriptors.Chi4n(mol),
            "Chi0v": GraphDescriptors.Chi0v(mol),
            "Chi1v": GraphDescriptors.Chi1v(mol),
            "Chi2v": GraphDescriptors.Chi2v(mol),
            "Chi3v": GraphDescriptors.Chi3v(mol),
            "Chi4v": GraphDescriptors.Chi4v(mol),
            "Kappa1": GraphDescriptors.Kappa1(mol),
            "Kappa2": GraphDescriptors.Kappa2(mol),
            "Kappa3": GraphDescriptors.Kappa3(mol),
            "BalabanJ": GraphDescriptors.BalabanJ(mol),
            "BertzCT": GraphDescriptors.BertzCT(mol),
            "HallKierAlpha": GraphDescriptors.HallKierAlpha(mol),
        }

    @staticmethod
    def dict_to_array(desc):
        if desc is None:
            return None
        return np.array(list(desc.values()), dtype=np.float32)

    # -----------------------------
    # Similarity metrics
    # -----------------------------
    def compare_2d_fingerprints(self, fp1, fp2):
        if fp1 is None or fp2 is None:
            return np.nan

        if self.metric == "tanimoto":
            return DataStructs.TanimotoSimilarity(fp1, fp2)

        if self.metric == "dice":
            return DataStructs.DiceSimilarity(fp1, fp2)

        if self.metric == "cosine":
            return DataStructs.CosineSimilarity(fp1, fp2)

        if self.metric == "mcconnaughey":
            return DataStructs.McConnaugheySimilarity(fp1, fp2)

        arr1 = self.fingerprint_to_array(fp1, self.n_bits)
        arr2 = self.fingerprint_to_array(fp2, self.n_bits)

        return self.compare_vectors(arr1, arr2)

    def compare_vectors(self, v1, v2):
        if v1 is None or v2 is None:
            return np.nan

        v1 = np.asarray(v1, dtype=np.float32)
        v2 = np.asarray(v2, dtype=np.float32)

        if self.metric == "cosine":
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom == 0:
                return np.nan
            return float(np.dot(v1, v2) / denom)

        if self.metric == "euclidean":
            distance = np.linalg.norm(v1 - v2)
            return float(1.0 / (1.0 + distance))

        if self.metric == "manhattan":
            distance = np.sum(np.abs(v1 - v2))
            return float(1.0 / (1.0 + distance))

        raise ValueError(
            "For descriptor vectors, use metric='cosine', 'euclidean', or 'manhattan'."
        )

    def compare_3d_shape(self, mol1, mol2):
        if mol1 is None or mol2 is None:
            return np.nan

        if not self.has_3d_coordinates(mol1) or not self.has_3d_coordinates(mol2):
            return np.nan

        try:
            # Align mol2 to mol1 when possible
            try:
                rdMolAlign.GetO3A(mol2, mol1).Align()
            except Exception:
                pass

            if self.metric == "shape_tanimoto":
                distance = rdShapeHelpers.ShapeTanimotoDist(mol1, mol2)
                return float(1.0 - distance)

            if self.metric == "shape_protrude":
                distance = rdShapeHelpers.ShapeProtrudeDist(mol1, mol2)
                return float(1.0 - distance)

            raise ValueError(
                "For 3D shape mode, use metric='shape_tanimoto' or 'shape_protrude'."
            )

        except Exception:
            return np.nan

    # -----------------------------
    # Search
    # -----------------------------
    def search_dataframe(self, query, df, smiles_col="smiles"):
        scores = []

        if self.mode == "2d_fingerprint":
            query_fp = self.calculate_fingerprint(query)

            for smi in df[smiles_col]:
                fp = self.calculate_fingerprint(smi)
                scores.append(self.compare_2d_fingerprints(query_fp, fp))

        elif self.mode == "2d_descriptor":
            query_vec = self.dict_to_array(self.calculate_2d_descriptors(query))

            for smi in df[smiles_col]:
                vec = self.dict_to_array(self.calculate_2d_descriptors(smi))
                scores.append(self.compare_vectors(query_vec, vec))

        elif self.mode == "topological_descriptor":
            query_vec = self.dict_to_array(self.calculate_topological_descriptors(query))

            for smi in df[smiles_col]:
                vec = self.dict_to_array(self.calculate_topological_descriptors(smi))
                scores.append(self.compare_vectors(query_vec, vec))

        elif self.mode == "3d_shape":
            query_mol = self.prepare_3d_mol_from_smiles(query)

            for smi in df[smiles_col]:
                mol = self.prepare_3d_mol_from_smiles(smi)
                scores.append(self.compare_3d_shape(query_mol, mol))

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        out = df.copy()
        out["similarity_score"] = scores
        out = out.sort_values("similarity_score", ascending=False)

        return out.reset_index(drop=True)

    def search_3d_files(self, query_file, library_files):
        query_mol = self.mol_from_file(query_file)

        results = []

        for file in library_files:
            mol = self.mol_from_file(file)
            score = self.compare_3d_shape(query_mol, mol)

            results.append(
                {
                    "file": str(file),
                    "similarity_score": score,
                }
            )

        return (
            pd.DataFrame(results)
            .sort_values("similarity_score", ascending=False)
            .reset_index(drop=True)
        )

    # -----------------------------
    # Similarity matrix
    # -----------------------------
    def similarity_matrix(self, df, smiles_col="smiles"):
        smiles = df[smiles_col].dropna().tolist()
        n = len(smiles)
        matrix = np.zeros((n, n), dtype=np.float32)

        if self.mode == "2d_fingerprint":
            reps = [self.calculate_fingerprint(smi) for smi in smiles]
            compare_func = self.compare_2d_fingerprints

        elif self.mode == "2d_descriptor":
            reps = [
                self.dict_to_array(self.calculate_2d_descriptors(smi))
                for smi in smiles
            ]
            compare_func = self.compare_vectors

        elif self.mode == "topological_descriptor":
            reps = [
                self.dict_to_array(self.calculate_topological_descriptors(smi))
                for smi in smiles
            ]
            compare_func = self.compare_vectors

        elif self.mode == "3d_shape":
            reps = [self.prepare_3d_mol_from_smiles(smi) for smi in smiles]
            compare_func = self.compare_3d_shape

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        for i in range(n):
            for j in range(n):
                matrix[i, j] = compare_func(reps[i], reps[j])

        return pd.DataFrame(matrix, index=smiles, columns=smiles)