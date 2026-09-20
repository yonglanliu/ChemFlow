"""Checkpoint-contained applicability domain and validation calibration."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression


APPLICABILITY_VERSION = 6
MAX_EMBEDDING_DIMENSIONS = 128
DEFAULT_CALIBRATION_CONFIDENCE = 0.90
DEFAULT_OOD_CONFIDENCE = 0.95
DEFAULT_SIMILARITY_RADIUS = 2
DEFAULT_SIMILARITY_BITS = 2048
DEFAULT_LOCAL_MIN_SAMPLES = 20
DEFAULT_LOCAL_MAX_SAMPLES = 100
DEFAULT_LOCAL_PROFILE_RADIUS = 1.0
DEFAULT_LOCAL_MIN_FP_SIMILARITY = 0.30
DEFAULT_LOCAL_MIN_EMBEDDING_SIMILARITY = 0.50
_BYTE_BIT_COUNTS = np.unpackbits(
    np.arange(256, dtype=np.uint8).reshape(-1, 1), axis=1
).sum(axis=1)


def packed_morgan_fingerprints(
    smiles: Sequence[str],
    *,
    radius: int = DEFAULT_SIMILARITY_RADIUS,
    bits: int = DEFAULT_SIMILARITY_BITS,
) -> np.ndarray:
    """Return byte-packed Morgan fingerprints suitable for checkpoint storage."""
    if int(radius) < 1 or int(bits) < 64 or int(bits) % 8:
        raise ValueError("Morgan radius must be positive and bits a multiple of 8.")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius), fpSize=int(bits)
    )
    packed: list[np.ndarray] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Cannot fingerprint invalid SMILES: {value!r}")
        fingerprint = generator.GetFingerprint(molecule)
        packed.append(
            np.frombuffer(DataStructs.BitVectToBinaryText(fingerprint), dtype=np.uint8)
        )
    if not packed:
        return np.empty((0, int(bits) // 8), dtype=np.uint8)
    return np.stack(packed)


def _nearest_packed_tanimoto(
    query: np.ndarray, references: np.ndarray
) -> tuple[float, int]:
    intersection = _BYTE_BIT_COUNTS[np.bitwise_and(references, query)].sum(axis=1)
    union = _BYTE_BIT_COUNTS[np.bitwise_or(references, query)].sum(axis=1)
    scores = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union != 0,
    )
    nearest = int(np.argmax(scores))
    return float(scores[nearest]), nearest


def _packed_tanimoto_scores(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    intersection = _BYTE_BIT_COUNTS[np.bitwise_and(references, query)].sum(axis=1)
    union = _BYTE_BIT_COUNTS[np.bitwise_or(references, query)].sum(axis=1)
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union != 0,
    )


def _cosine_similarities(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    references = np.asarray(references, dtype=np.float32)
    query_norm = max(float(np.linalg.norm(query)), 1e-12)
    reference_norms = np.maximum(np.linalg.norm(references, axis=1), 1e-12)
    return (references @ query) / (reference_norms * query_norm)


def fit_ood_calibration(
    similarities: np.ndarray,
    embedding_distances: np.ndarray,
    *,
    confidence: float = DEFAULT_OOD_CONFIDENCE,
) -> dict[str, Any]:
    """Store validation nonconformity distributions for empirical OOD scoring."""
    similarities = np.asarray(similarities, dtype=np.float64)
    embedding_distances = np.asarray(embedding_distances, dtype=np.float64)
    valid = np.isfinite(similarities) & np.isfinite(embedding_distances)
    if not valid.any():
        raise ValueError("OOD calibration requires finite validation diagnostics.")
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("OOD confidence must be between 0.5 and 1.0.")
    fp_novelty = 1.0 - similarities[valid]
    distances = embedding_distances[valid]
    return {
        "method": "validation_empirical_tail",
        "confidence": float(confidence),
        "fp_novelty": fp_novelty.astype(np.float32).tolist(),
        "embedding_distance": distances.astype(np.float32).tolist(),
        "fp_similarity_threshold": float(
            np.quantile(similarities[valid], 1.0 - confidence, method="lower")
        ),
        "embedding_distance_threshold": float(
            np.quantile(distances, confidence, method="higher")
        ),
        "n_samples": int(valid.sum()),
    }


def _empirical_tail_score(values: np.ndarray, calibration: Sequence[float]) -> np.ndarray:
    """Return 1-p empirical upper-tail scores; larger values are more OOD-like."""
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(calibration, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    scores = np.full(values.shape, np.nan, dtype=np.float32)
    if reference.size == 0:
        return scores
    finite = np.isfinite(values)
    for index in np.flatnonzero(finite):
        p_value = (1.0 + np.count_nonzero(reference >= values[index])) / (
            reference.size + 1.0
        )
        scores[index] = 1.0 - p_value
    return scores


def extract_model_embeddings(model, dataloader) -> np.ndarray:
    """Extract mean-aggregated molecular fingerprints in dataloader order."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = model.transfer_batch_to_device(batch, device, 0)
            bmg, atom_descriptors, extra_descriptors = batch[:3]
            embedding = model.fingerprint(bmg, atom_descriptors, extra_descriptors)
            batches.append(embedding.detach().float().cpu())
    if not batches:
        return np.empty((0, 0), dtype=np.float32)
    return torch.cat(batches, dim=0).numpy()


def fit_embedding_projection(
    training_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    dimensions: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """Fit a compact PCA space and project reference embeddings into it."""
    training = np.asarray(training_embeddings, dtype=np.float32)
    validation = np.asarray(validation_embeddings, dtype=np.float32)
    if training.ndim != 2 or training.shape[0] == 0:
        raise ValueError("Training embeddings must be a non-empty matrix.")
    output_dimensions = min(int(dimensions), training.shape[0], training.shape[1])
    if output_dimensions < 1:
        raise ValueError("embedding_dimensions must be at least 1.")

    if training.shape[0] == 1:
        mean = training[0].copy()
        components = np.eye(training.shape[1], dtype=np.float32)[:output_dimensions]
        training_projected = (training - mean) @ components.T
        validation_projected = (validation - mean) @ components.T
    else:
        solver = "randomized" if output_dimensions < min(training.shape) else "full"
        projection = PCA(
            n_components=output_dimensions,
            svd_solver=solver,
            random_state=int(seed),
        )
        training_projected = projection.fit_transform(training)
        validation_projected = projection.transform(validation)
        mean = projection.mean_.astype(np.float32, copy=False)
        components = projection.components_.astype(np.float32, copy=False)

    metadata = {
        "mean": torch.from_numpy(mean.astype(np.float32, copy=False)),
        "components": torch.from_numpy(components.astype(np.float16)),
    }
    return (
        metadata,
        training_projected.astype(np.float16),
        validation_projected.astype(np.float16),
    )


def _finite_sample_quantile(residuals: np.ndarray, confidence: float) -> float:
    values = np.asarray(residuals, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    probability = min(1.0, math.ceil((values.size + 1) * confidence) / values.size)
    return float(np.quantile(values, probability, method="higher"))


def fit_validation_calibration(
    task: str,
    targets: np.ndarray,
    predictions: np.ndarray,
    target_names: Sequence[str],
    confidence: float,
) -> dict[str, Any]:
    """Fit regression residual or classification isotonic validation calibration."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.shape != predictions.shape:
        raise ValueError("Validation targets and predictions must have equal shapes.")

    fitted: dict[str, Any] = {
        "task": str(task),
        "confidence": float(confidence),
        "tasks": {},
    }
    for task_index, target_name in enumerate(target_names):
        observed = targets[:, task_index]
        predicted = predictions[:, task_index]
        valid = np.isfinite(observed) & np.isfinite(predicted)
        observed = observed[valid]
        predicted = predicted[valid]
        if observed.size == 0:
            continue

        if task == "regression":
            bias = float(np.median(observed - predicted))
            calibrated = predicted + bias
            radius = _finite_sample_quantile(
                np.abs(observed - calibrated), float(confidence)
            )
            fitted["tasks"][str(target_name)] = {
                "method": "median_bias_validation_residual",
                "bias": bias,
                "interval_radius": radius,
                "absolute_residuals": np.abs(observed - calibrated).astype(float).tolist(),
                "n_samples": int(observed.size),
            }
        else:
            if np.unique(observed).size < 2:
                continue
            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            calibrator.fit(predicted, observed)
            fitted["tasks"][str(target_name)] = {
                "method": "isotonic",
                "x_thresholds": calibrator.X_thresholds_.astype(float).tolist(),
                "y_thresholds": calibrator.y_thresholds_.astype(float).tolist(),
                "n_samples": int(observed.size),
            }
    return fitted


def fit_multitask_validation_calibration(
    task_types: Sequence[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    target_names: Sequence[str],
    confidence: float,
) -> dict[str, Any]:
    """Fit each target with its own regression or classification calibrator."""
    combined: dict[str, Any] = {
        "task": "mixed" if len(set(task_types)) > 1 else str(task_types[0]),
        "confidence": float(confidence),
        "task_types": dict(zip(target_names, task_types)),
        "tasks": {},
    }
    for index, (name, task_type) in enumerate(zip(target_names, task_types)):
        fitted = fit_validation_calibration(
            task_type,
            np.asarray(targets)[:, index : index + 1],
            np.asarray(predictions)[:, index : index + 1],
            [name],
            confidence,
        )
        combined["tasks"].update(fitted["tasks"])
    return combined


def apply_calibration(
    output: dict[str, np.ndarray],
    calibration: dict[str, Any] | None,
    original_names: Sequence[str],
    output_names: Sequence[str],
    task: str,
    threshold: float,
    confidence: float | None = None,
) -> None:
    """Append calibrated predictions and intervals to an inference result."""
    if not calibration:
        return
    task_calibration = calibration.get("tasks", {})
    confidence = float(
        calibration.get("confidence", DEFAULT_CALIBRATION_CONFIDENCE)
        if confidence is None
        else confidence
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError("calibration_confidence must be between 0 and 1.")
    coverage_label = int(round(confidence * 100))
    for original_name, output_name in zip(original_names, output_names):
        fitted = task_calibration.get(original_name)
        if not fitted:
            continue
        if task == "regression":
            values = np.asarray(output[output_name], dtype=np.float32)
            calibrated = values + float(fitted["bias"])
            residuals = fitted.get("absolute_residuals")
            radius = (
                _finite_sample_quantile(np.asarray(residuals), confidence)
                if residuals is not None
                else float(fitted["interval_radius"])
            )
            output[f"calibrated_{output_name}"] = calibrated
            output[f"{output_name}_lower_{coverage_label}"] = calibrated - radius
            output[f"{output_name}_upper_{coverage_label}"] = calibrated + radius
        else:
            values = np.asarray(output[f"prob_{output_name}"], dtype=np.float32)
            calibrated = np.interp(
                values,
                np.asarray(fitted["x_thresholds"], dtype=np.float32),
                np.asarray(fitted["y_thresholds"], dtype=np.float32),
            ).astype(np.float32)
            calibrated[~np.isfinite(values)] = np.nan
            output[f"calibrated_prob_{output_name}"] = calibrated
            classes = np.full(len(values), np.nan, dtype=np.float32)
            finite = np.isfinite(calibrated)
            classes[finite] = (calibrated[finite] >= threshold).astype(np.float32)
            output[f"calibrated_class_{output_name}"] = classes


def project_query_embeddings(
    embeddings: np.ndarray,
    projection: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    mean = np.asarray(projection["mean"], dtype=np.float32)
    components = np.asarray(projection["components"], dtype=np.float32)
    return (values - mean) @ components.T


def _cosine_nearest(
    query: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    query_unit = query / np.maximum(query_norm, 1e-12)
    reference_unit = reference / np.maximum(reference_norm, 1e-12)
    best_similarity = np.full(len(query), -np.inf, dtype=np.float32)
    best_index = np.zeros(len(query), dtype=np.int64)
    # Bound temporary matrix size for large batch-prediction/reference sets.
    for query_start in range(0, len(query_unit), 256):
        query_stop = min(query_start + 256, len(query_unit))
        block_best = np.full(query_stop - query_start, -np.inf, dtype=np.float32)
        block_index = np.zeros(query_stop - query_start, dtype=np.int64)
        for reference_start in range(0, len(reference_unit), 8192):
            reference_stop = min(reference_start + 8192, len(reference_unit))
            similarities = (
                query_unit[query_start:query_stop]
                @ reference_unit[reference_start:reference_stop].T
            )
            local_index = np.argmax(similarities, axis=1)
            local_best = similarities[
                np.arange(len(local_index)), local_index
            ]
            improved = local_best > block_best
            block_best[improved] = local_best[improved]
            block_index[improved] = reference_start + local_index[improved]
        best_similarity[query_start:query_stop] = block_best
        best_index[query_start:query_stop] = block_index
    return (1.0 - best_similarity).astype(np.float32), best_index


def applicability_diagnostics(
    smiles: Sequence[str],
    embeddings: np.ndarray,
    valid_indices: Sequence[int],
    payload: dict[str, Any] | None,
    *,
    embedding_dimensions: int | None = None,
    similarity_radius: int = DEFAULT_SIMILARITY_RADIUS,
    similarity_bits: int = DEFAULT_SIMILARITY_BITS,
    reference_task: str | None = None,
) -> dict[str, np.ndarray]:
    """Calculate training-set chemical and learned-space nearest neighbors."""
    if not payload or not valid_indices:
        return {}
    task_partitions = payload.get("training_by_task", {})
    training = (
        task_partitions.get(reference_task, payload.get("training", {}))
        if reference_task is not None and isinstance(task_partitions, dict)
        else payload.get("training", {})
    )
    reference_smiles = [str(value) for value in training.get("smiles", [])]
    reference_indices = list(
        training.get("original_index", range(len(reference_smiles)))
    )
    if not reference_smiles:
        return {}

    valid_indices_array = np.asarray(valid_indices, dtype=np.int64)
    valid_smiles = [str(smiles[index]) for index in valid_indices]
    size = len(smiles)
    radius = int(similarity_radius)
    bits = int(similarity_bits)
    if radius < 1:
        raise ValueError("similarity_radius must be at least 1.")
    if bits < 64:
        raise ValueError("similarity_bits must be at least 64.")
    stored_fingerprints = training.get("fingerprints")
    use_stored_fingerprints = (
        stored_fingerprints is not None
        and int(payload.get("similarity", {}).get("radius", radius)) == radius
        and int(payload.get("similarity", {}).get("bits", bits)) == bits
    )
    reference_packed = None
    if use_stored_fingerprints:
        reference_packed = np.asarray(stored_fingerprints, dtype=np.uint8)
        expected_shape = (len(reference_smiles), bits // 8)
        if reference_packed.shape != expected_shape:
            raise ValueError(
                "Stored training fingerprint shape does not match the checkpoint "
                f"metadata: {reference_packed.shape} != {expected_shape}."
            )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    reference_fingerprints = None
    if reference_packed is None:
        reference_fingerprints = [
            generator.GetFingerprint(Chem.MolFromSmiles(value))
            for value in reference_smiles
        ]
    similarities = np.full(size, np.nan, dtype=np.float32)
    similarity_neighbors = np.full(size, "", dtype=object)
    similarity_neighbor_indices = np.full(size, None, dtype=object)
    query_packed_by_index: dict[int, np.ndarray] = {}
    for output_index, value in zip(valid_indices, valid_smiles):
        query_fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(value))
        query_packed = np.frombuffer(
            DataStructs.BitVectToBinaryText(query_fingerprint), dtype=np.uint8
        )
        query_packed_by_index[int(output_index)] = query_packed
        if reference_packed is not None:
            score, nearest = _nearest_packed_tanimoto(
                query_packed, reference_packed
            )
        else:
            scores = DataStructs.BulkTanimotoSimilarity(
                query_fingerprint, reference_fingerprints
            )
            nearest = int(np.argmax(scores))
            score = float(scores[nearest])
        similarities[output_index] = score
        similarity_neighbors[output_index] = reference_smiles[nearest]
        similarity_neighbor_indices[output_index] = reference_indices[nearest]

    projected = project_query_embeddings(embeddings, payload["projection"])
    reference_embeddings = np.asarray(training["embeddings"], dtype=np.float32)
    available_dimensions = projected.shape[1]
    dimensions = (
        available_dimensions
        if embedding_dimensions is None
        else int(embedding_dimensions)
    )
    if not 1 <= dimensions <= available_dimensions:
        raise ValueError(
            "embedding_dimensions must be between 1 and the checkpoint maximum "
            f"({available_dimensions})."
        )
    projected = projected[:, :dimensions]
    reference_embeddings = reference_embeddings[:, :dimensions]
    distances_valid, embedding_indices = _cosine_nearest(
        projected, reference_embeddings
    )
    distances = np.full(size, np.nan, dtype=np.float32)
    embedding_neighbors = np.full(size, "", dtype=object)
    embedding_neighbor_indices = np.full(size, None, dtype=object)
    distances[valid_indices_array] = distances_valid
    embedding_neighbors[valid_indices_array] = np.asarray(
        reference_smiles, dtype=object
    )[embedding_indices]
    embedding_neighbor_indices[valid_indices_array] = np.asarray(
        reference_indices, dtype=object
    )[embedding_indices]
    output = {
        "train_max_tanimoto": similarities,
        "train_nearest_tanimoto_index": similarity_neighbor_indices,
        "train_nearest_tanimoto_smiles": similarity_neighbors,
        "train_embedding_cosine_distance": distances,
        "train_nearest_embedding_index": embedding_neighbor_indices,
        "train_nearest_embedding_smiles": embedding_neighbors,
    }
    ood_by_task = payload.get("ood_calibration", {})
    ood_calibration = (
        ood_by_task.get(reference_task)
        if reference_task is not None and isinstance(ood_by_task, dict)
        else None
    )
    if isinstance(ood_calibration, dict):
        fp_score = _empirical_tail_score(
            1.0 - similarities, ood_calibration.get("fp_novelty", [])
        )
        embedding_score = _empirical_tail_score(
            distances, ood_calibration.get("embedding_distance", [])
        )
        output["ood_fp_score"] = fp_score
        output["ood_embedding_score"] = embedding_score
        output["ood_score"] = np.fmax(fp_score, embedding_score)

    validation_by_task = payload.get("validation_by_task", {})
    validation = (
        validation_by_task.get(reference_task)
        if reference_task is not None and isinstance(validation_by_task, dict)
        else None
    )
    if isinstance(validation, dict) and validation.get("smiles"):
        settings = payload.get("local_calibration", {})
        min_samples = int(settings.get("min_samples", DEFAULT_LOCAL_MIN_SAMPLES))
        max_samples = int(settings.get("max_samples", DEFAULT_LOCAL_MAX_SAMPLES))
        min_fp_similarity = float(
            settings.get("min_fp_similarity", DEFAULT_LOCAL_MIN_FP_SIMILARITY)
        )
        min_embedding_similarity = float(
            settings.get(
                "min_embedding_similarity",
                DEFAULT_LOCAL_MIN_EMBEDDING_SIMILARITY,
            )
        )
        validation_targets = np.asarray(validation["targets"], dtype=np.float32).reshape(
            len(validation["smiles"]), -1
        )[:, 0]
        validation_predictions = np.asarray(
            validation["predictions"], dtype=np.float32
        ).reshape(len(validation["smiles"]), -1)[:, 0]
        local_count = np.zeros(size, dtype=np.int32)
        local_bias = np.full(size, np.nan, dtype=np.float32)
        local_radius = np.full(size, np.nan, dtype=np.float32)
        local_used = np.zeros(size, dtype=bool)
        validation_max_fp = np.full(size, np.nan, dtype=np.float32)
        validation_max_embedding = np.full(size, np.nan, dtype=np.float32)
        confidence = float(
            payload.get("calibration", {}).get(
                "confidence", DEFAULT_CALIBRATION_CONFIDENCE
            )
        )
        stored_validation_fingerprints = validation.get("fingerprints")
        validation_fingerprints = (
            np.asarray(stored_validation_fingerprints, dtype=np.uint8)
            if (
                stored_validation_fingerprints is not None
                and int(payload.get("similarity", {}).get("radius", radius)) == radius
                and int(payload.get("similarity", {}).get("bits", bits)) == bits
            )
            else packed_morgan_fingerprints(
                validation["smiles"], radius=radius, bits=bits
            )
        )
        validation_embeddings = np.asarray(
            validation.get("embeddings"), dtype=np.float32
        )[:, :dimensions]
        for valid_position, result_index in enumerate(valid_indices):
            fp_similarities = _packed_tanimoto_scores(
                query_packed_by_index[int(result_index)], validation_fingerprints
            )
            embedding_similarities = _cosine_similarities(
                projected[valid_position],
                validation_embeddings,
            )
            validation_max_fp[result_index] = float(np.max(fp_similarities))
            validation_max_embedding[result_index] = float(
                np.max(embedding_similarities)
            )
            comparable_indices = np.flatnonzero(
                (fp_similarities >= min_fp_similarity)
                & (embedding_similarities >= min_embedding_similarity)
                & np.isfinite(validation_targets)
                & np.isfinite(validation_predictions)
            )
            if comparable_indices.size > max_samples:
                combined_distance = (
                    (1.0 - fp_similarities[comparable_indices])
                    + (1.0 - embedding_similarities[comparable_indices]) / 2.0
                )
                order = np.argsort(combined_distance)[:max_samples]
                comparable_indices = comparable_indices[order]
            count = int(comparable_indices.size)
            local_count[result_index] = count
            if count < min_samples:
                continue
            signed_errors = (
                validation_targets[comparable_indices]
                - validation_predictions[comparable_indices]
            )
            bias = float(np.median(signed_errors))
            residuals = np.abs(signed_errors - bias)
            local_bias[result_index] = bias
            local_radius[result_index] = _finite_sample_quantile(
                residuals, confidence
            )
            local_used[result_index] = True
        output["local_calibration_count"] = local_count
        output["local_calibration_used"] = local_used
        output["local_calibration_bias"] = local_bias
        output["local_uncertainty_radius"] = local_radius
        output["validation_max_tanimoto"] = validation_max_fp
        output["validation_max_embedding_similarity"] = validation_max_embedding
    return output


def apply_local_calibration(
    output: dict[str, np.ndarray],
    original_names: Sequence[str],
    output_names: Sequence[str],
    confidence: float,
) -> None:
    """Override global regression calibration where local support is sufficient."""
    coverage_label = int(round(float(confidence) * 100))
    for original_name, output_name in zip(original_names, output_names):
        used_key = f"{original_name}_local_calibration_used"
        if used_key not in output or output_name not in output:
            continue
        used = np.asarray(output[used_key], dtype=bool)
        bias = np.asarray(
            output[f"{original_name}_local_calibration_bias"], dtype=np.float32
        )
        radius = np.asarray(
            output[f"{original_name}_local_uncertainty_radius"], dtype=np.float32
        )
        raw = np.asarray(output[output_name], dtype=np.float32)
        calibrated_key = f"calibrated_{output_name}"
        calibrated = np.asarray(
            output.get(calibrated_key, raw.copy()), dtype=np.float32
        ).copy()
        calibrated[used] = raw[used] + bias[used]
        output[calibrated_key] = calibrated
        lower_key = f"{output_name}_lower_{coverage_label}"
        upper_key = f"{output_name}_upper_{coverage_label}"
        lower = np.asarray(output.get(lower_key, calibrated), dtype=np.float32).copy()
        upper = np.asarray(output.get(upper_key, calibrated), dtype=np.float32).copy()
        lower[used] = calibrated[used] - radius[used]
        upper[used] = calibrated[used] + radius[used]
        output[lower_key] = lower
        output[upper_key] = upper


def apply_mc_dropout_intervals(
    output: dict[str, np.ndarray],
    output_names: Sequence[str],
    confidence: float,
) -> None:
    """Combine MC-dropout spread with validation-calibrated intervals.

    Validation residual intervals remain the coverage anchor.  The interval is
    widened only when the Gaussian MC-dropout radius is larger, avoiding a
    falsely narrow interval for a query with unusually unstable forward passes.
    """
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("calibration_confidence must be between 0 and 1.")
    coverage_label = int(round(confidence * 100))
    z_value = float(NormalDist().inv_cdf(0.5 + confidence / 2.0))
    for output_name in output_names:
        std_key = f"mc_std_{output_name}"
        lower_key = f"{output_name}_lower_{coverage_label}"
        upper_key = f"{output_name}_upper_{coverage_label}"
        if std_key not in output or lower_key not in output or upper_key not in output:
            continue
        standard_deviation = np.asarray(output[std_key], dtype=np.float32)
        lower = np.asarray(output[lower_key], dtype=np.float32).copy()
        upper = np.asarray(output[upper_key], dtype=np.float32).copy()
        center = np.asarray(
            output.get(f"calibrated_{output_name}", output[output_name]),
            dtype=np.float32,
        )
        residual_radius = np.maximum(center - lower, upper - center)
        mc_radius = z_value * standard_deviation
        combined_radius = np.fmax(residual_radius, mc_radius)
        finite = np.isfinite(center) & np.isfinite(combined_radius)
        lower[finite] = center[finite] - combined_radius[finite]
        upper[finite] = center[finite] + combined_radius[finite]
        output[lower_key] = lower
        output[upper_key] = upper
        output[f"{output_name}_mc_radius_{coverage_label}"] = mc_radius
