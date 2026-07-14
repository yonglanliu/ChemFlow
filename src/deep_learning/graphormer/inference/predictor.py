# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.deep_learning.graphormer.config import (
    GraphormerFinetuneClassificationConfig,
    GraphormerFinetuneRegressionConfig,
)
from src.deep_learning.graphormer.models.graphormer_finetune_model import (
    GraphormerFineTuneClassificationModel,
    GraphormerFineTuneRegressionModel,
)
from src.deep_learning.graphormer.modules.dataset import (
    GraphormerMoleculeDataset,
    featurize_and_cache_dataset,
)
from src.deep_learning.graphormer.modules.graphormer_featurizer import (
    GraphormerFeaturizer,
)
from src.deep_learning.graphormer.utils.data_collator import (
    graphormer_collate_fn,
)
from src.deep_learning.utils import namespace_to_dict

from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

def dict_to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{
                key: dict_to_namespace(item)
                for key, item in value.items()
            }
        )

    if isinstance(value, list):
        return [
            dict_to_namespace(item)
            for item in value
        ]

    return value


def config_get(
    config: Any,
    key: str,
    default: Any = None,
) -> Any:
    if config is None:
        return default

    if isinstance(config, Mapping):
        return config.get(key, default)

    return getattr(config, key, default)


def update_dataclass_from_config(
    target: Any,
    source: Any,
    *,
    strict: bool = False,
) -> Any:
    if not is_dataclass(target):
        raise TypeError(
            "target must be a dataclass instance, "
            f"got {type(target).__name__}."
        )

    target_fields = {
        item.name
        for item in fields(target)
    }

    if isinstance(source, Mapping):
        values = dict(source)
    elif is_dataclass(source):
        values = {
            item.name: getattr(source, item.name)
            for item in fields(source)
        }
    elif hasattr(source, "__dict__"):
        values = vars(source)
    else:
        raise TypeError(
            "Unsupported source config type: "
            f"{type(source).__name__}."
        )

    unknown_fields = []

    for name, value in values.items():
        if name in target_fields:
            setattr(target, name, value)
        else:
            unknown_fields.append(name)

    if strict and unknown_fields:
        raise ValueError(
            f"Unknown fields for {type(target).__name__}: "
            f"{unknown_fields}"
        )

    return target


def move_batch_to_device(
    batch: Any,
    device: torch.device,
) -> Any:
    if torch.is_tensor(batch):
        return batch.to(
            device,
            non_blocking=True,
        )

    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, tuple):
        return tuple(
            move_batch_to_device(value, device)
            for value in batch
        )

    if isinstance(batch, list):
        return [
            move_batch_to_device(value, device)
            for value in batch
        ]

    return batch


def select_device(
    requested_device: Optional[str] = None,
) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class GraphormerPredictor:
    """
    Graphormer inference helper.

    The predictor loads a full training checkpoint, reconstructs the
    fine-tuned model, creates an inference DataLoader, and returns
    regression values or classification probabilities.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = Path(
            checkpoint_path
        ).expanduser().resolve()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: "
                f"{self.checkpoint_path}"
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1, "
                f"got {threshold}."
            )

        self.device = select_device(device)
        self.threshold = float(threshold)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
        )

        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "Expected checkpoint to be a dictionary."
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Inference requires a full checkpoint containing "
                "'model_state_dict'. Adapter-only checkpoints are "
                "not sufficient by themselves."
            )

        checkpoint_config = checkpoint.get("config")

        if not checkpoint_config:
            raise KeyError(
                "Checkpoint does not contain its resolved config."
            )

        self.checkpoint = checkpoint
        self.full_config = checkpoint_config

        self.base_config = dict_to_namespace(
            checkpoint_config["BaseConfig"]
        )

        self.model_config_source = dict_to_namespace(
            checkpoint_config["GraphormerConfig"]
        )

        self.dataset_config = dict_to_namespace(
            checkpoint_config["DatasetConfig"]
        )

        # Your resolved config should also contain FeaturizerConfig.
        # If it is not currently saved, add it to resolved_config
        # in the Trainer.
        featurizer_config_data = checkpoint_config.get(
            "FeaturizerConfig"
        )

        if featurizer_config_data is None:
            raise KeyError(
                "Checkpoint config does not contain "
                "'FeaturizerConfig'. Add FeaturizerConfig to the "
                "Trainer's resolved_config before saving checkpoints."
            )

        self.featurizer_config = dict_to_namespace(
            featurizer_config_data
        )

        self.task = str(
            self.base_config.task
        ).lower()

        self.model = self._build_model()

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.featurizer = GraphormerFeaturizer(
            **namespace_to_dict(
                self.featurizer_config
            )
        )

        self.classification_type = (
            self._resolve_classification_type()
            if self.task == "classification"
            else None
        )

        print(
            f"Loaded checkpoint: {self.checkpoint_path}"
        )
        print(
            f"Task: {self.task}"
        )
        print(
            f"Device: {self.device}"
        )

        if self.classification_type is not None:
            print(
                "Classification type: "
                f"{self.classification_type}"
            )

    def _build_model(self) -> nn.Module:
        if self.task == "regression":
            model_config = (
                GraphormerFinetuneRegressionConfig()
            )

            model_config = update_dataclass_from_config(
                model_config,
                self.model_config_source,
            )

            self.model_config = model_config

            return GraphormerFineTuneRegressionModel(
                cfg=model_config,
            )

        if self.task == "classification":
            model_config = (
                GraphormerFinetuneClassificationConfig()
            )

            model_config = update_dataclass_from_config(
                model_config,
                self.model_config_source,
            )

            self.model_config = model_config

            return GraphormerFineTuneClassificationModel(
                cfg=model_config,
            )

        raise ValueError(
            f"Unsupported task: {self.task!r}."
        )

    def _resolve_classification_type(self) -> str:
        loss_type = str(
            self.model_config.loss_type
        ).lower()

        num_classes = int(
            self.model_config.num_classes
        )

        if loss_type == "bce":
            return "binary"

        if loss_type == "cross_entropy":
            if num_classes == 2:
                return "binary"

            if num_classes > 2:
                return "multiclass"

            raise ValueError(
                "cross_entropy requires num_classes >= 2, "
                f"got {num_classes}."
            )

        raise ValueError(
            "Unsupported classification loss type: "
            f"{loss_type!r}."
        )

    def build_loader(
        self,
        dataset: GraphormerMoleculeDataset,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> DataLoader:
        collate_fn = lambda samples: graphormer_collate_fn(
            samples,
            max_nodes=int(
                config_get(
                    self.dataset_config,
                    "max_nodes",
                    128,
                )
            ),
            multi_hop_max_dist=int(
                config_get(
                    self.dataset_config,
                    "multi_hop_max_dist",
                    5,
                )
            ),
            spatial_pos_max=int(
                config_get(
                    self.dataset_config,
                    "spatial_pos_max",
                    1024,
                )
            ),
        )

        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(num_workers),
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_fn,
        )

    @staticmethod
    def _forward_batch(
        model: nn.Module,
        batch: Any,
    ) -> Any:
        if isinstance(batch, dict):
            return model(
                batched_data=batch,
            )

        if isinstance(batch, (tuple, list)):
            return model(*batch)

        return model(batch)

    @staticmethod
    def _extract_predictions(
        outputs: Any,
    ) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return outputs

        if isinstance(outputs, Mapping):
            predictions = outputs.get(
                "predictions",
                outputs.get("logits"),
            )

            if predictions is not None:
                return predictions

        if hasattr(outputs, "predictions"):
            return outputs.predictions

        if hasattr(outputs, "logits"):
            return outputs.logits

        if isinstance(outputs, (tuple, list)):
            # During inference, some models may return:
            # (predictions,) or (loss, predictions).
            for value in reversed(outputs):
                if torch.is_tensor(value):
                    return value

        raise TypeError(
            "Could not extract predictions from model output."
        )

    @torch.inference_mode()
    def predict_loader(
        self,
        loader: DataLoader,
    ) -> pd.DataFrame:
        prediction_batches = []

        for batch in tqdm(
            loader,
            desc="Inference",
        ):
            batch = move_batch_to_device(
                batch,
                self.device,
            )

            outputs = self._forward_batch(
                self.model,
                batch,
            )

            predictions = self._extract_predictions(
                outputs
            )

            prediction_batches.append(
                predictions.detach().cpu()
            )

        if not prediction_batches:
            raise RuntimeError(
                "Inference loader produced no batches."
            )

        raw_predictions = torch.cat(
            prediction_batches,
            dim=0,
        )

        return self._format_predictions(
            raw_predictions
        )

    def _format_predictions(
        self,
        predictions: torch.Tensor,
    ) -> pd.DataFrame:
        if self.task == "regression":
            values = (
                predictions
                .reshape(-1)
                .to(dtype=torch.float32)
                .numpy()
            )

            return pd.DataFrame(
                {
                    "prediction": values,
                }
            )

        if self.classification_type == "binary":
            return self._format_binary_predictions(
                predictions
            )

        if self.classification_type == "multiclass":
            return self._format_multiclass_predictions(
                predictions
            )

        raise RuntimeError(
            "Classification type was not resolved."
        )

    def _format_binary_predictions(
        self,
        predictions: torch.Tensor,
    ) -> pd.DataFrame:
        loss_type = str(
            self.model_config.loss_type
        ).lower()

        if loss_type == "bce":
            if (
                predictions.ndim == 2
                and predictions.shape[1] == 1
            ):
                logits = predictions[:, 0]

            elif predictions.ndim == 1:
                logits = predictions

            else:
                raise ValueError(
                    "BCE binary inference expects predictions "
                    "with shape (N,) or (N, 1), got "
                    f"{tuple(predictions.shape)}."
                )

            positive_probabilities = torch.sigmoid(
                logits
            )

        elif loss_type == "cross_entropy":
            if (
                predictions.ndim != 2
                or predictions.shape[1] != 2
            ):
                raise ValueError(
                    "Binary cross-entropy inference expects "
                    "two logits per sample with shape (N, 2), "
                    f"got {tuple(predictions.shape)}."
                )

            probabilities = torch.softmax(
                predictions,
                dim=1,
            )

            positive_probabilities = probabilities[:, 1]

        else:
            raise ValueError(
                f"Unsupported binary loss type: {loss_type}."
            )

        positive_probabilities = (
            positive_probabilities
            .to(dtype=torch.float32)
            .numpy()
        )

        predicted_labels = (
            positive_probabilities >= self.threshold
        ).astype(np.int64)

        return pd.DataFrame(
            {
                "probability_negative": (
                    1.0 - positive_probabilities
                ),
                "probability_positive": (
                    positive_probabilities
                ),
                "predicted_label": predicted_labels,
            }
        )

    def _format_multiclass_predictions(
        self,
        predictions: torch.Tensor,
    ) -> pd.DataFrame:
        num_classes = int(
            self.model_config.num_classes
        )

        if predictions.ndim != 2:
            raise ValueError(
                "Multiclass predictions must have shape "
                f"(N, C), got {tuple(predictions.shape)}."
            )

        if predictions.shape[1] != num_classes:
            raise ValueError(
                "Prediction class dimension does not match "
                f"num_classes: {predictions.shape[1]} versus "
                f"{num_classes}."
            )

        probabilities = torch.softmax(
            predictions,
            dim=1,
        )

        predicted_labels = probabilities.argmax(
            dim=1
        )

        probabilities_np = (
            probabilities
            .to(dtype=torch.float32)
            .numpy()
        )

        result = {
            f"probability_class_{class_index}":
                probabilities_np[:, class_index]
            for class_index in range(num_classes)
        }

        result["predicted_label"] = (
            predicted_labels.numpy()
        )

        return pd.DataFrame(result)

    def predict_dataset(
        self,
        dataset: GraphormerMoleculeDataset,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        loader = self.build_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return self.predict_loader(loader)

    def predict_manifest(
        self,
        shard_paths: list[str | Path],
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        dataset = GraphormerMoleculeDataset(
            shard_paths=shard_paths,
        )

        return self.predict_dataset(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def save_predictions(
        self,
        predictions: pd.DataFrame,
        output_path: str | Path,
        input_frame: Optional[pd.DataFrame] = None,
    ) -> Path:
        output_path = Path(
            output_path
        ).expanduser().resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if input_frame is not None:
            if len(input_frame) != len(predictions):
                raise ValueError(
                    "Input DataFrame and prediction DataFrame "
                    "have different lengths: "
                    f"{len(input_frame)} versus "
                    f"{len(predictions)}."
                )

            output_frame = pd.concat(
                [
                    input_frame.reset_index(drop=True),
                    predictions.reset_index(drop=True),
                ],
                axis=1,
            )
        else:
            output_frame = predictions

        output_frame.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Saved predictions to: {output_path}"
        )

        return output_path
    def predict_smiles(
        self,
        smiles_list: list[str],
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        """
        Predict one or more raw SMILES strings.
        """
        dataset = GraphormerInferenceDataset(
            smiles_list=smiles_list,
            featurizer=self.featurizer,
        )

        loader = self.build_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return self.predict_loader(loader)

class GraphormerInferenceDataset(Dataset):
    """
    In-memory Graphormer dataset for raw SMILES inference.
    """

    def __init__(
        self,
        smiles_list: list[str],
        featurizer: Any,
    ) -> None:
        if not smiles_list:
            raise ValueError(
                "smiles_list cannot be empty."
            )

        self.smiles_list = [
            str(smiles).strip()
            for smiles in smiles_list
        ]

        self.features = []

        for index, smiles in enumerate(self.smiles_list):
            if not smiles:
                raise ValueError(
                    f"SMILES at index {index} is empty."
                )

            try:
                feature = self._featurize(
                    featurizer=featurizer,
                    smiles=smiles,
                )
            except Exception as error:
                raise ValueError(
                    f"Failed to featurize SMILES at index "
                    f"{index}: {smiles!r}"
                ) from error

            self.features.append(feature)

    @staticmethod
    def _featurize(
        featurizer: Any,
        smiles: str,
    ) -> Any:
        """
        Call the available Graphormer featurization interface.

        Keep only the branch matching your GraphormerFeaturizer API
        once its exact method is fixed.
        """
        if hasattr(featurizer, "featurize_smiles"):
            return featurizer.featurize_smiles(smiles)

        if hasattr(featurizer, "featurize"):
            return featurizer.featurize(smiles)

        if callable(featurizer):
            return featurizer(smiles)

        raise TypeError(
            "GraphormerFeaturizer must provide featurize_smiles(), "
            "featurize(), or __call__()."
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self,
        index: int,
    ) -> Any:
        return self.features[index]