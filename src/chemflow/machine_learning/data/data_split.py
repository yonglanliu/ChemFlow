# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

from src.chemflow.featurization import smiles_to_fp


@dataclass
class SplitResult:
    X_train: np.ndarray
    X_valid: Optional[np.ndarray]
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: Optional[np.ndarray]
    y_test: np.ndarray
    train_indices: np.ndarray
    valid_indices: Optional[np.ndarray]
    test_indices: np.ndarray
    save_dir: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "X_train": self.X_train,
            "X_valid": self.X_valid,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_valid": self.y_valid,
            "y_test": self.y_test,
            "train_indices": self.train_indices,
            "valid_indices": self.valid_indices,
            "test_indices": self.test_indices,
            "save_dir": self.save_dir,
        }


class DataSplitter:
    SUPPORTED_METHODS = {"random", "stratified", "scaffold", "cluster", "butina"}

    def __init__(self, split_config: Optional[Dict[str, Any]] = None) -> None:
        if split_config is None:
            raise ValueError("split_config is required.")

        self.split_config = dict(split_config)

        if "split_method" not in self.split_config:
            raise ValueError("split_config must contain 'split_method'.")

        if "test_size" not in self.split_config:
            raise ValueError("split_config must contain 'test_size'.")

        self.test_size = float(self.split_config["test_size"])

        validation_size = self.split_config.get("validation_size", None)
        self.has_validation = validation_size is not None and str(validation_size).lower() not in {
            "",
            "none",
            "null",
        }

        self.validation_size = 0.0 if not self.has_validation else float(validation_size)

        self.random_seed = int(self.split_config.get("random_seed", 42))
        self.split_method = str(self.split_config["split_method"]).lower()

        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1. Got {self.test_size}")

        if not 0 <= self.validation_size < 1:
            raise ValueError(
                f"validation_size must be None or between 0 and 1. Got {validation_size}"
            )

        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be < 1.")

        if self.split_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"split_method must be one of {sorted(self.SUPPORTED_METHODS)}. "
                f"Got {self.split_method}"
            )

        self.train_size = 1.0 - self.test_size - self.validation_size

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        smiles: Optional[np.ndarray] = None,
    ) -> SplitResult:
        task_type = str(
            self.split_config.get(
                "task_type",
                self.split_config.get("task", "regression"),
            )
        ).lower()

        X = np.asarray(X)
        y = np.asarray(y)

        self._validate_same_length(X, y)

        if smiles is not None:
            smiles = np.asarray(smiles)
            self._validate_same_length(X, y, smiles)

        if self.split_method == "random":
            result = self.random_split(X, y)

        elif self.split_method == "stratified":
            if task_type != "classification":
                raise ValueError("Stratified split is only for classification.")
            result = self.stratified_split(X, y)

        elif self.split_method == "scaffold":
            if smiles is None:
                raise ValueError("smiles is required for scaffold split.")
            result = self.scaffold_split(X, y, smiles)

        elif self.split_method == "cluster":
            if smiles is None:
                raise ValueError("smiles is required for cluster split.")
            result = self.cluster_split(
                X=X,
                y=y,
                smiles=smiles,
                n_clusters=int(self.split_config.get("n_clusters", 20)),
                radius=int(self.split_config.get("fp_radius", 2)),
                n_bits=int(self.split_config.get("fp_n_bits", 2048)),
            )

        elif self.split_method == "butina":
            if smiles is None:
                raise ValueError("smiles is required for Butina split.")
            result = self.butina_split(
                X=X,
                y=y,
                smiles=smiles,
                cutoff=float(self.split_config.get("butina_cutoff", 0.4)),
                radius=int(self.split_config.get("fp_radius", 2)),
                n_bits=int(self.split_config.get("fp_n_bits", 2048)),
            )

        else:
            raise ValueError(f"Unsupported split_method: {self.split_method}")

        if self._as_bool(self.split_config.get("save_split_data", False)):
            extra_arrays = {}

            if smiles is not None:
                extra_arrays["smiles"] = smiles

            save_dir = self.split_config.get("save_dir", None)

            prefix_name = self.split_config.get(
                "prefix_name",
                self.split_config.get("split_name", None),
            )

            if save_dir is None or prefix_name is None:
                raise ValueError(
                    "Both save_dir and prefix_name/split_name must be provided "
                    "to save split data."
                )

            result = self.save_split(
                result=result,
                save_dir=save_dir,
                prefix_name=prefix_name,
                split_method=self.split_method,
                extra_arrays=extra_arrays,
                config=self.split_config,
            )

        return result

    def random_split(self, X: np.ndarray, y: np.ndarray) -> SplitResult:
        indices = np.arange(len(X))

        train_valid_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=True,
        )

        if not self.has_validation or self.validation_size == 0:
            return self._make_result(X, y, train_valid_idx, None, test_idx)

        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=self._validation_fraction(),
            random_state=self.random_seed,
            shuffle=True,
        )

        return self._make_result(X, y, train_idx, valid_idx, test_idx)

    def stratified_split(self, X: np.ndarray, y: np.ndarray) -> SplitResult:
        indices = np.arange(len(X))

        train_valid_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=True,
            stratify=y,
        )

        if not self.has_validation or self.validation_size == 0:
            return self._make_result(X, y, train_valid_idx, None, test_idx)

        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=self._validation_fraction(),
            random_state=self.random_seed,
            shuffle=True,
            stratify=y[train_valid_idx],
        )

        return self._make_result(X, y, train_idx, valid_idx, test_idx)

    def scaffold_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        smiles: np.ndarray,
    ) -> SplitResult:
        X, y, smiles = self._to_arrays(X, y, smiles)
        self._validate_same_length(X, y, smiles)

        scaffolds = []
        valid_original_indices = []

        for original_idx, smi in enumerate(smiles):
            scaffold = self._get_scaffold(str(smi))
            if scaffold is not None:
                scaffolds.append(scaffold)
                valid_original_indices.append(original_idx)

        if len(valid_original_indices) == 0:
            raise ValueError("No valid molecules after scaffold generation.")

        valid_original_indices = np.asarray(valid_original_indices, dtype=int)

        scaffold_to_local_indices: Dict[str, List[int]] = {}

        for local_idx, scaffold in enumerate(scaffolds):
            scaffold_to_local_indices.setdefault(scaffold, []).append(local_idx)

        return self.grouped_split(
            X=X[valid_original_indices],
            y=y[valid_original_indices],
            groups=list(scaffold_to_local_indices.values()),
            original_indices=valid_original_indices,
        )

    def cluster_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        smiles: np.ndarray,
        n_clusters: int = 20,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> SplitResult:
        X_valid, y_valid, _, fps, valid_original_indices = self._add_fingerprints(
            X,
            y,
            smiles,
            radius,
            n_bits,
        )

        if len(X_valid) < 2:
            raise ValueError("At least 2 valid molecules are required for cluster split.")

        n_clusters = min(max(2, int(n_clusters)), len(X_valid))
        dist_mat = self._tanimoto_distance_matrix(fps)

        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                linkage="average",
            )

        labels = clustering.fit_predict(dist_mat)

        cluster_to_local_indices: Dict[int, List[int]] = {}

        for local_idx, cluster_id in enumerate(labels):
            cluster_to_local_indices.setdefault(int(cluster_id), []).append(local_idx)

        return self.grouped_split(
            X=X_valid,
            y=y_valid,
            groups=list(cluster_to_local_indices.values()),
            original_indices=valid_original_indices,
        )

    def butina_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        smiles: np.ndarray,
        cutoff: float = 0.4,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> SplitResult:
        X_valid, y_valid, _, fps, valid_original_indices = self._add_fingerprints(
            X,
            y,
            smiles,
            radius,
            n_bits,
        )

        if len(fps) < 2:
            raise ValueError("At least 2 valid molecules are required for Butina split.")

        clusters = self._butina_cluster_fps(fps, cutoff)
        groups = [list(cluster) for cluster in clusters]

        return self.grouped_split(
            X=X_valid,
            y=y_valid,
            groups=groups,
            original_indices=valid_original_indices,
        )

    def grouped_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Sequence[Sequence[int]],
        original_indices: Optional[np.ndarray] = None,
    ) -> SplitResult:
        groups = [list(group) for group in groups if len(group) > 0]

        if len(groups) == 0:
            raise ValueError("No groups found for grouped split.")

        rng = random.Random(self.random_seed)
        rng.shuffle(groups)

        n_total = len(X)
        n_test_target = max(1, int(round(n_total * self.test_size)))
        n_valid_target = int(round(n_total * self.validation_size))

        train_local: List[int] = []
        valid_local: List[int] = []
        test_local: List[int] = []

        for group in groups:
            if len(test_local) < n_test_target:
                test_local.extend(group)
            elif self.validation_size > 0 and len(valid_local) < n_valid_target:
                valid_local.extend(group)
            else:
                train_local.extend(group)

        train_local_idx = np.asarray(train_local, dtype=int)
        valid_local_idx = (
            np.asarray(valid_local, dtype=int)
            if self.validation_size > 0
            else None
        )
        test_local_idx = np.asarray(test_local, dtype=int)

        self._validate_non_empty_split(
            train_local_idx,
            valid_local_idx,
            test_local_idx,
            "Grouped",
        )

        if original_indices is None:
            original_indices = np.arange(len(X), dtype=int)
        else:
            original_indices = np.asarray(original_indices, dtype=int)

        train_original_idx = original_indices[train_local_idx]
        valid_original_idx = (
            original_indices[valid_local_idx]
            if valid_local_idx is not None
            else None
        )
        test_original_idx = original_indices[test_local_idx]

        return SplitResult(
            X_train=X[train_local_idx],
            X_valid=X[valid_local_idx] if valid_local_idx is not None else None,
            X_test=X[test_local_idx],
            y_train=y[train_local_idx],
            y_valid=y[valid_local_idx] if valid_local_idx is not None else None,
            y_test=y[test_local_idx],
            train_indices=train_original_idx,
            valid_indices=valid_original_idx,
            test_indices=test_original_idx,
        )

    def save_split(
        self,
        result: SplitResult,
        save_dir: str | Path,
        prefix_name: str,
        split_method: str,
        extra_arrays: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> SplitResult:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        npz_path = save_path / f"{prefix_name}_split_data.npz"
        metadata_path = save_path / f"{prefix_name}_metadata.json"

        arrays_to_save: Dict[str, Any] = {
            "X_train": result.X_train,
            "X_test": result.X_test,
            "y_train": result.y_train,
            "y_test": result.y_test,
            "train_indices": result.train_indices,
            "test_indices": result.test_indices,
        }

        if result.X_valid is not None:
            arrays_to_save["X_valid"] = result.X_valid

        if result.y_valid is not None:
            arrays_to_save["y_valid"] = result.y_valid

        if result.valid_indices is not None:
            arrays_to_save["valid_indices"] = result.valid_indices

        for key, value in (extra_arrays or {}).items():
            if value is not None:
                arrays_to_save[key] = np.asarray(value)

        np.savez_compressed(npz_path, **arrays_to_save)

        metadata = {
            "prefix_name": prefix_name,
            "split_method": split_method,
            "test_size": self.test_size,
            "validation_size": None if not self.has_validation else self.validation_size,
            "train_size": self.train_size,
            "random_seed": self.random_seed,
            "n_train": int(len(result.train_indices)),
            "n_valid": int(len(result.valid_indices)) if result.valid_indices is not None else 0,
            "n_test": int(len(result.test_indices)),
            "npz_file": str(npz_path),
            "config": self._json_safe(config or {}),
        }

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        result.save_dir = str(save_path)
        return result

    def _make_result(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        valid_idx: Optional[np.ndarray],
        test_idx: np.ndarray,
    ) -> SplitResult:
        train_idx = np.asarray(train_idx, dtype=int)
        valid_idx = np.asarray(valid_idx, dtype=int) if valid_idx is not None else None
        test_idx = np.asarray(test_idx, dtype=int)

        self._validate_non_empty_split(train_idx, valid_idx, test_idx)

        return SplitResult(
            X_train=X[train_idx],
            X_valid=X[valid_idx] if valid_idx is not None else None,
            X_test=X[test_idx],
            y_train=y[train_idx],
            y_valid=y[valid_idx] if valid_idx is not None else None,
            y_test=y[test_idx],
            train_indices=train_idx,
            valid_indices=valid_idx,
            test_indices=test_idx,
        )

    def _validation_fraction(self) -> float:
        return self.validation_size / (1.0 - self.test_size)

    def _get_scaffold(self, smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            return None

        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )

    def _add_fingerprints(
        self,
        X: np.ndarray,
        y: np.ndarray,
        smiles: np.ndarray,
        radius: int = 2,
        n_bits: int = 2048,
    ):
        X, y, smiles = self._to_arrays(X, y, smiles)
        self._validate_same_length(X, y, smiles)

        fps = []
        valid_original_indices = []

        for original_idx, smi in enumerate(smiles):
            fp = smiles_to_fp(
                smiles=str(smi),
                radius=radius,
                n_bits=n_bits,
                use_features=False,
            )

            if fp is not None:
                fps.append(fp)
                valid_original_indices.append(original_idx)

        if len(valid_original_indices) == 0:
            raise ValueError("No valid molecules after SMILES parsing.")

        valid_original_indices = np.asarray(valid_original_indices, dtype=int)

        return (
            X[valid_original_indices],
            y[valid_original_indices],
            smiles[valid_original_indices],
            fps,
            valid_original_indices,
        )

    def _tanimoto_distance_matrix(self, fps: List) -> np.ndarray:
        n = len(fps)
        dist_mat = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            for j, sim in enumerate(sims):
                dist_mat[i, j] = 1.0 - float(sim)

        return dist_mat

    def _butina_cluster_fps(self, fps: List, cutoff: float = 0.4):
        n_fps = len(fps)
        dists = []

        for i in range(1, n_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1.0 - float(sim) for sim in sims])

        return Butina.ClusterData(
            dists,
            n_fps,
            cutoff,
            isDistData=True,
        )

    def _validate_same_length(self, *arrays: np.ndarray) -> None:
        lengths = [len(arr) for arr in arrays]

        if len(set(lengths)) != 1:
            raise ValueError(f"Arrays must have same length. Got lengths: {lengths}")

    def _validate_non_empty_split(
        self,
        train_idx: np.ndarray,
        valid_idx: Optional[np.ndarray],
        test_idx: np.ndarray,
        prefix_name: str = "Split",
    ) -> None:
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"{prefix_name} failed because train or test set is empty. "
                "Try increasing dataset size or reducing test_size."
            )

        if self.validation_size > 0 and (valid_idx is None or len(valid_idx) == 0):
            raise ValueError(
                f"{prefix_name} failed because validation set is empty. "
                "Try increasing dataset size or reducing validation_size."
            )

    def _to_arrays(self, *arrays: np.ndarray):
        return tuple(np.asarray(arr) for arr in arrays)

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if value is None:
            return False

        if isinstance(value, (int, float)):
            return bool(value)

        return str(value).strip().lower() in {"true", "yes", "y", "1"}

    def _json_safe(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, Path):
            return str(obj)

        return obj
