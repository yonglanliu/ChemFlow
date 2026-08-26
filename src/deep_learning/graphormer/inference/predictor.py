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

from src.chemflow.machine_learning.data import dataset
from src.deep_learning.graphormer.config import (
    GraphormerFinetuneClassificationConfig,
    GraphormerFinetuneMultitaskConfig,
    GraphormerFinetuneRegressionConfig,
)
from src.deep_learning.graphormer.models.graphormer_finetune_model import (
    GraphormerFineTuneClassificationModel,
    GraphormerFineTuneRegressionModel,
)
from src.deep_learning.graphormer.models.graphormer_multitask_model import (
    GraphormerMultiTaskModel,
)

from src.deep_learning.graphormer.modules.dataset import (
    GraphormerMoleculeDataset,
    featurize_and_cache_dataset,
)
from src.deep_learning.graphormer.modules.data_pipeline import (
    _normalize_feature_types,
    _featurize_single_smiles,
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
        return [dict_to_namespace(item) for item in value]

    return value


def config_get(config: Any, key: str, default: Any = None) -> Any:
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

    target_fields = {item.name for item in fields(target)}

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

            if not strict:
                setattr(target, name, value)

    if strict and unknown_fields:
        raise ValueError(
            f"Unknown fields for {type(target).__name__}: "
            f"{unknown_fields}"
        )

    return target


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)

    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)

    if isinstance(batch, list):
        return [
            move_batch_to_device(value, device)
            for value in batch
        ]

    return batch



def graphormer_collate_with_extra_features(
    samples,
    *,
    max_nodes: int,
    multi_hop_max_dist: int,
    spatial_pos_max: int,
):
    """
    Use the standard Graphormer collator while preserving auxiliary
    descriptor/fingerprint tensors required by late-fusion models.
    """
    eligible_samples = [
        sample
        for sample in samples
        if getattr(sample, "x", None) is not None
        and int(sample.x.size(0)) <= int(max_nodes)
    ]

    batch = graphormer_collate_fn(
        samples,
        max_nodes=int(max_nodes),
        multi_hop_max_dist=int(multi_hop_max_dist),
        spatial_pos_max=int(spatial_pos_max),
    )

    if not isinstance(batch, dict):
        raise TypeError(
            "graphormer_collate_fn must return a dictionary."
        )

    def _stack_optional(attribute_name: str):
        present = [
            hasattr(sample, attribute_name)
            and getattr(sample, attribute_name) is not None
            for sample in eligible_samples
        ]

        if not any(present):
            return None

        if not all(present):
            raise RuntimeError(
                f"Only some samples contain '{attribute_name}'."
            )

        return torch.stack(
            [
                torch.as_tensor(
                    getattr(sample, attribute_name),
                    dtype=torch.float32,
                ).reshape(-1)
                for sample in eligible_samples
            ],
            dim=0,
        )

    descriptor_features = _stack_optional(
        "descriptor_features"
    )
    fingerprint_features = _stack_optional(
        "fingerprint_features"
    )

    if descriptor_features is not None:
        batch["descriptor_features"] = descriptor_features

    if fingerprint_features is not None:
        batch["fingerprint_features"] = fingerprint_features

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
        validation_predictions: Optional[np.ndarray | list[float] | tuple[float, ...]] = None,
        validation_targets: Optional[np.ndarray | list[float] | tuple[float, ...]] = None,
    ) -> None:

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")

        self.device = select_device(device)
        self.threshold = float(threshold)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, Mapping):
            raise TypeError("Expected checkpoint to be a dictionary.")

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Inference requires a full checkpoint containing "
                "'model_state_dict'. Adapter-only checkpoints are "
                "not sufficient by themselves."
            )

        checkpoint_config = checkpoint.get("config")

        if not checkpoint_config:
            raise KeyError("Checkpoint does not contain its resolved config.")

        self.checkpoint = checkpoint
        self.full_config = checkpoint_config

        self.base_config = dict_to_namespace(checkpoint_config["BaseConfig"])

        self.model_config_source = dict_to_namespace(checkpoint_config["GraphormerConfig"])

        self.dataset_config = dict_to_namespace(checkpoint_config["DatasetConfig"])

        # Your resolved config should also contain FeaturizerConfig.
        # If it is not currently saved, add it to resolved_config
        # in the Trainer.
        featurizer_config_data = checkpoint_config.get("FeaturizerConfig")

        if featurizer_config_data is None:
            raise KeyError(
                "Checkpoint config does not contain "
                "'FeaturizerConfig'. Add FeaturizerConfig to the "
                "Trainer's resolved_config before saving checkpoints."
            )

        self.featurizer_config = dict_to_namespace(featurizer_config_data)

        self.task = str(self.base_config.task).lower()

        self._recover_extra_feature_config_from_state_dict()

        self.model = self._build_model()
        print(f'use model: {type(self.model).__name__}')

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.featurizer = GraphormerFeaturizer(
            **namespace_to_dict(self.featurizer_config)
        )

        self.use_extra_features = bool(
            config_get(
                self.model_config,
                "use_extra_features",
                False,
            )
        )

        self.extra_feature_types = config_get(
            self.dataset_config,
            "extra_feature_types",
            None,
        )

        if self.extra_feature_types is not None:
            self.extra_feature_types = _normalize_feature_types(
                self.extra_feature_types
            )

        self.extra_feature_preprocessor = None
        self.extra_feature_preprocessor_path = None

        if self.use_extra_features:
            self._load_extra_feature_preprocessor()

        self.regression_calibration: dict[str, float] | None = None
        self.classification_type = (self._resolve_classification_type() if self.task == "classification" else None)

        if self.task == "regression" and validation_predictions is not None and validation_targets is not None:
            self.fit_regression_calibration(
                validation_predictions=validation_predictions,
                validation_targets=validation_targets,
            )

        print(f"Loaded checkpoint: {self.checkpoint_path}")
        print(f"Task: {self.task}")
        print(f"Device: {self.device}")

        if self.classification_type is not None:
            print(f"Classification type: {self.classification_type}")

    def _recover_extra_feature_config_from_state_dict(
        self,
    ) -> None:
        """
        Recover descriptor/fingerprint architecture from checkpoint tensors.

        This keeps older checkpoints usable when their saved GraphormerConfig
        does not contain the newer late-fusion fields.
        """
        state_dict = self.checkpoint["model_state_dict"]

        descriptor_input_key = (
            "descriptor_encoder.net.0.weight"
        )
        descriptor_output_key = (
            "descriptor_encoder.net.3.weight"
        )

        if descriptor_input_key in state_dict:
            input_weight = state_dict[
                descriptor_input_key
            ]
            output_weight = state_dict[
                descriptor_output_key
            ]

            descriptor_hidden_dim = int(
                input_weight.shape[0]
            )
            descriptor_dim = int(
                input_weight.shape[1]
            )
            descriptor_embedding_dim = int(
                output_weight.shape[0]
            )

            self.model_config_source.use_extra_features = True
            self.model_config_source.use_descriptors = True
            self.model_config_source.descriptor_dim = descriptor_dim
            self.model_config_source.descriptor_hidden_dim = (
                descriptor_hidden_dim
            )
            self.model_config_source.descriptor_embedding_dim = (
                descriptor_embedding_dim
            )

            if not hasattr(
                self.model_config_source,
                "descriptor_dropout",
            ):
                self.model_config_source.descriptor_dropout = 0.1

            if not hasattr(
                self.model_config_source,
                "descriptor_activation",
            ):
                self.model_config_source.descriptor_activation = "gelu"

            print(
                "Recovered descriptor branch from checkpoint:"
            )
            print(
                f"  descriptor_dim={descriptor_dim}"
            )
            print(
                f"  hidden_dim={descriptor_hidden_dim}"
            )
            print(
                f"  embedding_dim={descriptor_embedding_dim}"
            )
        else:
            self.model_config_source.use_descriptors = False

        fingerprint_input_key = (
            "fingerprint_encoder.net.0.weight"
        )
        fingerprint_output_key = (
            "fingerprint_encoder.net.3.weight"
        )

        if fingerprint_input_key in state_dict:
            input_weight = state_dict[
                fingerprint_input_key
            ]
            output_weight = state_dict[
                fingerprint_output_key
            ]

            fingerprint_hidden_dim = int(
                input_weight.shape[0]
            )
            fingerprint_dim = int(
                input_weight.shape[1]
            )
            fingerprint_embedding_dim = int(
                output_weight.shape[0]
            )

            self.model_config_source.use_extra_features = True
            self.model_config_source.use_fingerprint = True
            self.model_config_source.fingerprint_dim = fingerprint_dim
            self.model_config_source.fingerprint_hidden_dim = (
                fingerprint_hidden_dim
            )
            self.model_config_source.fingerprint_embedding_dim = (
                fingerprint_embedding_dim
            )

            if not hasattr(
                self.model_config_source,
                "fingerprint_dropout",
            ):
                self.model_config_source.fingerprint_dropout = 0.1

            if not hasattr(
                self.model_config_source,
                "fingerprint_activation",
            ):
                self.model_config_source.fingerprint_activation = "gelu"

            print(
                "Recovered fingerprint branch from checkpoint:"
            )
            print(
                f"  fingerprint_dim={fingerprint_dim}"
            )
            print(
                f"  hidden_dim={fingerprint_hidden_dim}"
            )
            print(
                f"  embedding_dim={fingerprint_embedding_dim}"
            )
        else:
            self.model_config_source.use_fingerprint = False

        if not (
            getattr(
                self.model_config_source,
                "use_descriptors",
                False,
            )
            or getattr(
                self.model_config_source,
                "use_fingerprint",
                False,
            )
        ):
            self.model_config_source.use_extra_features = False


    def _resolve_extra_feature_preprocessor_path(
        self,
    ) -> Path | None:
        candidates = []

        for key in (
            "extra_feature_preprocessor_path",
            "preprocessor_path",
        ):
            value = config_get(
                self.dataset_config,
                key,
                None,
            )
            if value:
                candidates.append(
                    Path(str(value)).expanduser()
                )

        checkpoint_path_value = self.checkpoint.get(
            "extra_feature_preprocessor_path"
        )

        if checkpoint_path_value:
            candidates.append(
                Path(
                    str(checkpoint_path_value)
                ).expanduser()
            )

        workdir = config_get(
            self.base_config,
            "workdir",
            None,
        )

        if workdir:
            workdir = Path(
                str(workdir)
            ).expanduser()

            candidates.extend(
                [
                    workdir
                    / "cache"
                    / "extra_feature_preprocessor.pt",
                    workdir
                    / "extra_feature_preprocessor.pt",
                ]
            )

        for candidate in candidates:
            candidate = candidate.resolve()

            if candidate.is_file():
                return candidate

        return None


    def _load_extra_feature_preprocessor(
        self,
    ) -> None:
        if self.extra_feature_types is None:
            raise ValueError(
                "The checkpoint uses extra molecular features, "
                "but DatasetConfig.extra_feature_types is missing."
            )

        path = self._resolve_extra_feature_preprocessor_path()

        if path is None:
            raise FileNotFoundError(
                "The checkpoint uses extra molecular features, but "
                "the fitted training preprocessor could not be found. "
                "Expected extra_feature_preprocessor.pt under the "
                "training workdir cache or a saved preprocessor path "
                "inside the checkpoint config."
            )

        try:
            preprocessor = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            preprocessor = torch.load(
                path,
                map_location="cpu",
            )

        self.extra_feature_preprocessor = preprocessor
        self.extra_feature_preprocessor_path = path

        print(
            f"Loaded extra-feature preprocessor: {path}"
        )
        print(
            f"Extra feature types: {self.extra_feature_types}"
        )
        print(
            "Descriptor branch: "
            f"{bool(config_get(self.model_config, 'use_descriptors', False))}"
        )
        print(
            "Fingerprint branch: "
            f"{bool(config_get(self.model_config, 'use_fingerprint', False))}"
        )


    def _build_model(self) -> nn.Module:
        if self.task == "regression":
            model_config = (GraphormerFinetuneRegressionConfig())
            model_config = update_dataclass_from_config(model_config, self.model_config_source)
            self.model_config = model_config
            return GraphormerFineTuneRegressionModel(cfg=model_config)

        if self.task == "classification":
            model_config = (GraphormerFinetuneClassificationConfig())
            model_config = update_dataclass_from_config(model_config, self.model_config_source)
            self.model_config = model_config
            return GraphormerFineTuneClassificationModel(cfg=model_config)
        
        if self.task == "multitask":
            model_config = (GraphormerFinetuneMultitaskConfig())
            model_config = update_dataclass_from_config(model_config, self.model_config_source,)
            self.model_config = model_config
            return GraphormerMultiTaskModel(cfg=model_config)
        
        raise ValueError(
            f"Unsupported task: {self.task!r}."
        )

    def _resolve_classification_type(self) -> str:
        loss_type = str(self.model_config.loss_type).lower()

        num_classes = int(self.model_config.num_classes)

        if loss_type == "bce":
            return "binary"

        if loss_type == "cross_entropy":
            if num_classes == 2:
                return "binary"

            if num_classes > 2:
                return "multiclass"

            raise ValueError(f"cross_entropy requires num_classes >= 2, got {num_classes}.")

        raise ValueError(f"Unsupported classification loss type: {loss_type!r}.")

    def build_loader(self, dataset: GraphormerMoleculeDataset, batch_size: int = 64, num_workers: int = 0) -> DataLoader:

        collate_fn = lambda samples: graphormer_collate_with_extra_features(
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
    def _forward_batch(model: nn.Module, batch: Any) -> Any:
        if isinstance(batch, dict):
            return model(batched_data=batch)

        if isinstance(batch, (tuple, list)):
            return model(*batch)

        return model(batch)

    @staticmethod
    def _extract_predictions(outputs: Any) -> torch.Tensor:

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

        raise TypeError("Could not extract predictions from model output.")

    @torch.inference_mode()
    def predict_loader(self, loader: DataLoader) -> pd.DataFrame:

        prediction_batches = []
        total_predictions = 0

        for batch_index, batch in enumerate(tqdm(loader, desc="Inference")):

            # Check input batch size BEFORE moving to device
            if isinstance(batch, Mapping):
                for key, value in batch.items():
                    if torch.is_tensor(value) and value.ndim > 0:
                        #print(f"Batch {batch_index}: "f"input key={key}, shape={tuple(value.shape)}")
                        break

            batch = move_batch_to_device(batch, self.device)

            outputs = self._forward_batch(self.model, batch)

            predictions = self._extract_predictions(outputs)

            if isinstance(predictions, Mapping):

                print(f"\nBatch {batch_index} task outputs:")

                for key, value in predictions.items():
                    print(f"    {key}: {tuple(value.shape)}")

                predictions = torch.cat(
                    [
                        predictions[f"task_{i}"]
                        for i in range(len(predictions))
                    ],
                    dim=1,
                )

            elif isinstance(predictions, (tuple, list)):
                predictions = torch.cat(
                    predictions,
                    dim=1,
                )

            elif not torch.is_tensor(predictions):
                raise TypeError(
                    "Predictions must be a tensor, a dict of tensors, "
                    f"or a tuple/list of tensors, got "
                    f"{type(predictions).__name__}."
                )

            # print(
            #     f"Batch {batch_index}: "
            #     f"prediction shape={tuple(predictions.shape)}"
            # )

            total_predictions += predictions.shape[0]

            prediction_batches.append(
                predictions.detach().cpu()
            )

        print(
            "\nTotal predictions before cat:",
            total_predictions,
        )

        if not prediction_batches:
            raise RuntimeError(
                "Inference loader produced no batches."
            )

        raw_predictions = torch.cat(
            prediction_batches,
            dim=0,
        )

        print(
            "Raw prediction shape:",
            tuple(raw_predictions.shape),
        )

        return self._format_predictions(
            raw_predictions
        )

    def fit_regression_calibration(
        self,
        validation_predictions: np.ndarray | list[float] | tuple[float, ...],
        validation_targets: np.ndarray | list[float] | tuple[float, ...],
        *,
        min_scale: float = 1e-6,
    ) -> dict[str, float]:
        # Calibration is fitted on the validation set by comparing model outputs to
        # the observed target values. The residuals r = y - y_hat capture the model's
        # systematic bias and spread.
        predictions = np.asarray(validation_predictions, dtype=np.float64).reshape(-1)
        targets = np.asarray(validation_targets, dtype=np.float64).reshape(-1)

        if predictions.shape != targets.shape:
            raise ValueError(
                "Validation predictions and targets must have the same length: "
                f"{predictions.shape} versus {targets.shape}."
            )

        valid_mask = np.isfinite(predictions) & np.isfinite(targets)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if predictions.size == 0:
            raise ValueError("No finite validation predictions and targets remain for calibration.")

        # Residuals = target - prediction. The median residual gives the average bias
        # of the model on validation data, while the median absolute deviation (MAD)
        # estimates the typical spread of that bias. Multiplying by 1.4826 converts
        # MAD to an estimate comparable to the standard deviation under a Gaussian
        # assumption. This makes the calibration robust to outliers.
        residuals = targets - predictions
        offset = float(np.median(residuals))
        abs_residuals = np.abs(residuals - offset)
        scale = float(np.median(abs_residuals) * 1.4826)

        # Fall back to the standard deviation if the robust estimate is unstable or
        # zero, which prevents degenerate calibration values when the validation set is
        # very small or unusually concentrated.
        if not np.isfinite(scale) or scale <= min_scale:
            scale = float(np.std(residuals, ddof=1) if residuals.size > 1 else min_scale)

        if not np.isfinite(scale) or scale <= min_scale:
            scale = min_scale

        calibration = {
            "offset": float(offset),
            "scale": float(scale),
            "n_samples": int(predictions.size),
        }

        self.regression_calibration = calibration
        return calibration

    def fit_calibration(
        self,
        validation_predictions: np.ndarray | list[float] | tuple[float, ...],
        validation_targets: np.ndarray | list[float] | tuple[float, ...],
        *,
        min_scale: float = 1e-6,
    ) -> dict[str, float]:
        return self.fit_regression_calibration(
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            min_scale=min_scale,
        )

    def _format_predictions(self, predictions: torch.Tensor) -> pd.DataFrame:
        if self.task == "regression":
            values = (predictions.reshape(-1).to(dtype=torch.float32).numpy())

            if self.regression_calibration is not None:
                # Apply the validation-derived correction to future predictions.
                # The offset shifts the raw regression output to remove the validation
                # bias, and the scale is used as a calibrated uncertainty estimate.
                offset = float(self.regression_calibration.get("offset", 0.0))
                scale = float(self.regression_calibration.get("scale", 1.0))
                calibrated_predictions = values + offset
                calibrated_std = np.full_like(calibrated_predictions, scale, dtype=np.float64)

                return pd.DataFrame(
                    {
                        "prediction": calibrated_predictions,
                        "prediction_std": calibrated_std,
                        "uncertainty": calibrated_std,
                        "prediction_lower": calibrated_predictions - 1.96 * calibrated_std,
                        "prediction_upper": calibrated_predictions + 1.96 * calibrated_std,
                    }
                )

            return pd.DataFrame({"prediction": values})

        if self.classification_type == "binary":
            return self._format_binary_predictions(predictions)

        if self.classification_type == "multiclass":
            return self._format_multiclass_predictions(predictions)

        if self.task == "multitask":
            return self._format_multitask_predictions(predictions)

        raise RuntimeError("Classification type was not resolved.")

    def _format_binary_predictions(self, predictions: torch.Tensor) -> pd.DataFrame:
        loss_type = str(self.model_config.loss_type).lower()

        if loss_type == "bce":
            if (predictions.ndim == 2 and predictions.shape[1] == 1):
                logits = predictions[:, 0]

            elif predictions.ndim == 1:
                logits = predictions

            else:
                raise ValueError(
                    "BCE binary inference expects predictions "
                    "with shape (N,) or (N, 1), got "
                    f"{tuple(predictions.shape)}."
                )

            positive_probabilities = torch.sigmoid(logits)

        elif loss_type == "cross_entropy":
            if (predictions.ndim != 2 or predictions.shape[1] != 2):
                raise ValueError(
                    "Binary cross-entropy inference expects "
                    "two logits per sample with shape (N, 2), "
                    f"got {tuple(predictions.shape)}."
                )
            probabilities = torch.softmax(predictions, dim=1)
            positive_probabilities = probabilities[:, 1]

        else:
            raise ValueError(f"Unsupported binary loss type: {loss_type}.")

        positive_probabilities = (positive_probabilities.to(dtype=torch.float32).numpy())

        predicted_labels = (positive_probabilities >= self.threshold).astype(np.int64)

        return pd.DataFrame(
            {
                "probability_negative": (1.0 - positive_probabilities),
                "probability_positive": (positive_probabilities),
                "predicted_label": predicted_labels,
            }
        )

    def _format_multiclass_predictions(
        self,
        predictions: torch.Tensor,
    ) -> pd.DataFrame:
        num_classes = int(self.model_config.num_classes)

        if predictions.ndim != 2:
            raise ValueError(f"Multiclass predictions must have shape (N, C), got {tuple(predictions.shape)}.")

        if predictions.shape[1] != num_classes:
            raise ValueError(f"Prediction class dimension does not match "
                             f"num_classes: {predictions.shape[1]} versus "
                             f"{num_classes}.")

        probabilities = torch.softmax(predictions, dim=1)
        predicted_labels = probabilities.argmax(dim=1)
        probabilities_np = (probabilities.to(dtype=torch.float32).numpy())
        result = {f"probability_class_{class_index}": probabilities_np[:, class_index] for class_index in range(num_classes)}
        result["predicted_label"] = (predicted_labels.numpy())

        return pd.DataFrame(result)

    def _format_multitask_predictions(
        self,
        predictions: torch.Tensor,
    ) -> pd.DataFrame:

        if not torch.is_tensor(predictions):
            raise TypeError(
                "Multitask predictions must be a Tensor, "
                f"got {type(predictions).__name__}."
            )

        num_tasks = int(self.model_config.num_tasks)

        if predictions.ndim != 2:
            raise ValueError(
                f"Multitask predictions must have shape (N, T), "
                f"got {tuple(predictions.shape)}."
            )

        if predictions.shape[1] != num_tasks:
            raise ValueError(
                "Prediction task dimension does not match num_tasks: "
                f"{predictions.shape[1]} versus {num_tasks}."
            )

        predictions = predictions.detach().cpu().float()

        result = {
            f"task_{task_index}": predictions[:, task_index].numpy()
            for task_index in range(num_tasks)
        }

        return pd.DataFrame(result)

    def predict_dataset(
        self,
        dataset: GraphormerMoleculeDataset,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        loader = self.build_loader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

        return self.predict_loader(loader)

    def predict_manifest(self, shard_paths: list[str | Path], batch_size: int = 64, num_workers: int = 0) -> pd.DataFrame:
        dataset = GraphormerMoleculeDataset(shard_paths=shard_paths)
        return self.predict_dataset(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    def save_predictions(self, predictions: pd.DataFrame, output_path: str | Path, input_frame: Optional[pd.DataFrame] = None) -> Path:
        output_path = Path(output_path).expanduser().resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if input_frame is not None:
            if len(input_frame) != len(predictions):
                raise ValueError(
                    "Input DataFrame and prediction DataFrame "
                    "have different lengths: "
                    f"{len(input_frame)} versus "
                    f"{len(predictions)}."
                )

            output_frame = pd.concat([input_frame.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
        else:
            output_frame = predictions

        output_frame.to_csv(output_path, index=False)

        print(f"Saved predictions to: {output_path}")

        return output_path
    
    def predict_smiles(self, smiles_list: list[str], batch_size: int = 64, num_workers: int = 0) -> pd.DataFrame:
        """
        Predict one or more raw SMILES strings.
        """
        dataset = GraphormerInferenceDataset(
            smiles_list=smiles_list,
            featurizer=self.featurizer,
            extra_feature_types=self.extra_feature_types,
            extra_feature_preprocessor=(
                self.extra_feature_preprocessor
            ),
            descriptor_dim=int(
                config_get(
                    self.model_config,
                    "descriptor_dim",
                    0,
                )
            ),
            use_descriptors=bool(
                config_get(
                    self.model_config,
                    "use_descriptors",
                    False,
                )
            ),
            use_fingerprint=bool(
                config_get(
                    self.model_config,
                    "use_fingerprint",
                    False,
                )
            ),
        )
        print("Input smiles:", len(smiles_list))
        print("Dataset size:", len(dataset))
        loader = self.build_loader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
        print("Loader dataset size:", len(loader.dataset))

        return self.predict_loader(loader)

class GraphormerInferenceDataset(Dataset):
    """
    In-memory Graphormer inference dataset.

    Extra molecular features are generated from SMILES and transformed
    using the exact preprocessor fitted on the training set.
    """

    def __init__(
        self,
        smiles_list: list[str],
        featurizer: Any,
        extra_feature_types=None,
        extra_feature_preprocessor=None,
        descriptor_dim: int = 0,
        use_descriptors: bool = False,
        use_fingerprint: bool = False,
    ) -> None:
        if not smiles_list:
            raise ValueError(
                "smiles_list cannot be empty."
            )

        self.smiles_list = [
            str(smiles).strip()
            for smiles in smiles_list
        ]

        self.extra_feature_types = (
            _normalize_feature_types(
                extra_feature_types
            )
            if extra_feature_types is not None
            else None
        )

        self.extra_feature_preprocessor = (
            extra_feature_preprocessor
        )

        self.descriptor_dim = int(
            descriptor_dim
        )

        self.use_descriptors = bool(
            use_descriptors
        )

        self.use_fingerprint = bool(
            use_fingerprint
        )

        use_extra_features = (
            self.use_descriptors
            or self.use_fingerprint
        )

        if (
            use_extra_features
            and self.extra_feature_types is None
        ):
            raise ValueError(
                "Extra-feature model is enabled, but "
                "extra_feature_types is missing."
            )

        if (
            use_extra_features
            and self.extra_feature_preprocessor is None
        ):
            raise ValueError(
                "Extra-feature model is enabled, but the "
                "training-fitted preprocessor is missing."
            )

        graph_features = []
        raw_extra_features = []

        for index, smiles in enumerate(
            self.smiles_list
        ):
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

            graph_features.append(
                feature
            )

            if use_extra_features:
                raw_extra = _featurize_single_smiles(
                    smiles,
                    self.extra_feature_types,
                )

                if raw_extra is None:
                    raise ValueError(
                        "Failed to generate extra molecular "
                        f"features at index {index}: {smiles!r}"
                    )

                raw_extra_features.append(
                    np.asarray(
                        raw_extra,
                        dtype=np.float32,
                    )
                )

        if use_extra_features:
            raw_matrix = np.vstack(
                raw_extra_features
            ).astype(
                np.float32,
                copy=False,
            )

            processed = (
                self.extra_feature_preprocessor
                .transform(raw_matrix)
            )

            processed = np.asarray(
                processed,
                dtype=np.float32,
            )

            if self.descriptor_dim < 0:
                raise ValueError(
                    "descriptor_dim must be >= 0."
                )

            if self.descriptor_dim > processed.shape[1]:
                raise ValueError(
                    f"descriptor_dim={self.descriptor_dim} exceeds "
                    f"processed feature dimension "
                    f"{processed.shape[1]}."
                )

            descriptor_matrix = None
            fingerprint_matrix = None

            if self.use_descriptors:
                descriptor_matrix = processed[
                    :,
                    : self.descriptor_dim,
                ]

                if (
                    descriptor_matrix.shape[1]
                    != self.descriptor_dim
                ):
                    raise ValueError(
                        "Descriptor feature dimension mismatch."
                    )

                print(
                    "Inference descriptor feature shape: "
                    f"{descriptor_matrix.shape}"
                )

            if self.use_fingerprint:
                fingerprint_matrix = processed[
                    :,
                    self.descriptor_dim :,
                ]

                print(
                    "Inference fingerprint feature shape: "
                    f"{fingerprint_matrix.shape}"
                )

            for index, feature in enumerate(
                graph_features
            ):
                if self.use_descriptors:
                    feature.descriptor_features = (
                        torch.as_tensor(
                            descriptor_matrix[index],
                            dtype=torch.float32,
                        )
                    )

                if self.use_fingerprint:
                    feature.fingerprint_features = (
                        torch.as_tensor(
                            fingerprint_matrix[index],
                            dtype=torch.float32,
                        )
                    )

        self.features = graph_features

    @staticmethod
    def _featurize(
        featurizer: Any,
        smiles: str,
    ) -> Any:
        if hasattr(
            featurizer,
            "featurize_smiles",
        ):
            return featurizer.featurize_smiles(
                smiles
            )

        if hasattr(
            featurizer,
            "featurize",
        ):
            return featurizer.featurize(
                smiles
            )

        if callable(featurizer):
            return featurizer(
                smiles
            )

        raise TypeError(
            "GraphormerFeaturizer must provide "
            "featurize_smiles(), featurize(), "
            "or __call__()."
        )

    def __len__(self) -> int:
        return len(
            self.features
        )

    def __getitem__(
        self,
        index: int,
    ) -> Any:
        return self.features[index]

