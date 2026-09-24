# Modified by ChemFlow: namespaced vendored imports; optional cuik_molmaker support.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The training function used in the finetuning task.
"""
import csv
import logging
import os
import json
import pickle  # noqa: F401  (kept; available if a caller wants to load legacy .pckl splits)
import sys
import time
from argparse import Namespace
from logging import Logger
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import wandb
except ImportError:
    wandb = None
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from chemflow.deep_learning.test_evaluation import (
    fingerprint_local_predictions,
    save_test_evaluation,
)

from chemflow.deep_learning.kermt.vendor.kermt.data import MolCollator
from chemflow.deep_learning.kermt.vendor.kermt.data import StandardScaler
from chemflow.deep_learning.kermt.vendor.kermt.util.loss import MTLLoss
from chemflow.deep_learning.kermt.vendor.kermt.util.metrics import get_metric_func
from chemflow.deep_learning.kermt.vendor.kermt.util.nn_utils import initialize_weights
from chemflow.deep_learning.kermt.vendor.kermt.util.scheduler import NoamLR
from chemflow.deep_learning.kermt.vendor.kermt.util.utils import build_optimizer, build_lr_scheduler, makedirs, load_checkpoint, get_loss_func, \
    save_checkpoint, build_model, get_ffn_layer_names, save_model_for_restart
from chemflow.deep_learning.kermt.vendor.kermt.util.utils import get_class_sizes, get_data, split_data, get_task_names
from chemflow.deep_learning.kermt.vendor.task.predict import predict, evaluate, evaluate_predictions


def _parameter_counts(named_parameters):
    """Return total, trainable, and frozen element counts for parameters."""
    parameters = [parameter for _, parameter in named_parameters]
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    return total, trainable, total - trainable


def _log_table(title: str, rows: list[tuple[str, object]], log) -> None:
    """Write a dependency-free fixed-width table to terminal and Slurm logs."""
    headers = ("Item", "Value")
    rendered = [(str(label), str(value)) for label, value in rows]
    label_width = max(len(headers[0]), *(len(label) for label, _ in rendered))
    value_width = max(len(headers[1]), *(len(value) for _, value in rendered))
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"

    log(title)
    log(border)
    log(f"| {headers[0]:<{label_width}} | {headers[1]:<{value_width}} |")
    log(border)
    for label, value in rendered:
        log(f"| {label:<{label_width}} | {value:<{value_width}} |")
    log(border)


def _log_model_audit(model, args: Namespace, checkpoint_path, debug) -> None:
    """Log checkpoint and post-freezing parameter state for a finetuning run."""
    named_parameters = list(model.named_parameters())
    encoder = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("kermt.")
    ]
    downstream = [
        (name, parameter)
        for name, parameter in named_parameters
        if not name.startswith("kermt.")
    ]
    total, trainable, frozen = _parameter_counts(named_parameters)
    encoder_total, encoder_trainable, encoder_frozen = _parameter_counts(encoder)
    head_total, head_trainable, head_frozen = _parameter_counts(downstream)
    trainable_percent = 100.0 * trainable / total if total else 0.0

    _log_table(
        "KERMT model audit:",
        [
            (
                "Checkpoint loaded successfully",
                "yes" if checkpoint_path is not None else "no; initialized from scratch",
            ),
            ("Checkpoint", checkpoint_path if checkpoint_path is not None else "none"),
            ("Total parameters", f"{total:,}"),
            ("Trainable parameters", f"{trainable:,} ({trainable_percent:.4f}%)"),
            ("Frozen parameters", f"{frozen:,}"),
            (
                "Encoder trainable",
                f"{encoder_trainable:,}/{encoder_total:,} "
                f"(frozen: {'yes' if encoder_total and encoder_trainable == 0 else 'no'})",
            ),
            (
                "Head/readout trainable",
                f"{head_trainable:,}/{head_total:,} "
                f"(frozen parameters: {head_frozen:,})",
            ),
            ("Encoder LR coefficient", f"{args.fine_tune_coff:g}"),
        ],
        debug,
    )


def _resolve_resume_checkpoint(args: Namespace, model_idx: int, save_dir: str) -> str:
    """Resolve one full-state checkpoint for the current fold and ensemble member."""
    configured = getattr(args, "resume_checkpoint", None)
    if not configured:
        checkpoint = Path(save_dir) / "last_checkpoint.pt"
    else:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            if args.num_folds != 1 or args.ensemble_size != 1:
                raise ValueError(
                    "A single resume_checkpoint file can only resume one fold and "
                    "one ensemble member. Provide the KERMT output directory instead."
                )
            checkpoint = candidate
        else:
            fold_num = int(getattr(args, "fold_num", 0))
            possibilities = (
                candidate / f"fold_{fold_num}" / f"model_{model_idx}" / "last_checkpoint.pt",
                candidate / f"model_{model_idx}" / "last_checkpoint.pt",
                candidate / "last_checkpoint.pt",
            )
            checkpoint = next((path for path in possibilities if path.is_file()), possibilities[0])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"KERMT resume checkpoint not found: {checkpoint}. Run once with "
            "resume=false or set TrainingConfig.resume_checkpoint correctly."
        )
    return str(checkpoint)


def _log_resume_audit(model, checkpoint_path: str, start_epoch: int, info) -> None:
    """Record successful full-state restoration and post-resume trainability."""
    total, trainable, frozen = _parameter_counts(list(model.named_parameters()))
    percentage = 100.0 * trainable / total if total else 0.0
    info("KERMT resume audit:")
    info("  Resume successful: yes")
    info(f"  Resume checkpoint: {checkpoint_path}")
    info(f"  Next epoch: {start_epoch}")
    info("  Model weights restored: yes")
    info("  Optimizer state restored: yes")
    info("  Scheduler state restored: yes")
    info(f"  Trainable parameters after resume: {trainable:,}/{total:,} ({percentage:.4f}%)")
    info(f"  Frozen parameters after resume: {frozen:,}")


def _update_early_stopping(
    value: float,
    best_value: float,
    bad_epochs: int,
    *,
    min_delta: float,
    minimize: bool,
) -> tuple[float, int, bool]:
    """Update an early-stopping state from one validation observation."""
    if not np.isfinite(value):
        return best_value, bad_epochs + 1, False
    improved = (
        value < best_value - min_delta
        if minimize
        else value > best_value + min_delta
    )
    if improved:
        return float(value), 0, True
    return best_value, bad_epochs + 1, False


def _mean_finite(values, fallback: float) -> float:
    """Return the finite mean of a scalar/dictionary checkpoint value."""
    candidates = values.values() if isinstance(values, dict) else [values]
    finite = [float(value) for value in candidates if np.isfinite(value)]
    return float(np.mean(finite)) if finite else fallback



def train(epoch, model, data, loss_func, mtl_loss, optimizer, scheduler,
          shared_dict, args: Namespace, n_iter: int = 0,
          logger: logging.Logger = None, world_size: int = 1):
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :return: The total number of iterations (training examples) trained on so far.
    """
    # debug = logger.debug if logger is not None else print
    model.train()

    # data.shuffle()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0


    mol_collator = MolCollator(shared_dict=shared_dict, args=args)

    num_workers = 0
    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=mol_collator)

    is_main = not dist.is_initialized() or dist.get_rank() == 0
    progress = tqdm(
        mol_loader,
        desc=f"Epoch {epoch + 1}/{args.epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        disable=not is_main,
    )
    for item in progress:
        _, batch, features_batch, mask, targets = item
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model. optimizer.zero_grad() clears the gradient of every
        # registered parameter, including mtl_loss.log_sigma (appended to the
        # optimizer's param groups when --use_mtl_loss). model.zero_grad() alone
        # would miss it, letting the MTL task-weight gradient accumulate.
        optimizer.zero_grad()
        preds = model(batch, features_batch)
        loss = loss_func(preds, targets) * class_weights * mask

        if mtl_loss is not None:
            # Per-task mean loss. Under DDP the normalization must use the GLOBAL per-task
            # valid-label count, not each rank's local count: DDP averages the per-rank
            # gradients (1/world_size), so dividing by a local count weights ranks equally
            # rather than per-label and systematically down-weights tasks whose labels land on
            # only a subset of ranks. So all-reduce the count (not the loss sum, which stays
            # local so each rank differentiates only its own samples), then scale by world_size
            # to cancel DDP's 1/world_size average -> the net normalization is the true global
            # per-label mean, matching a single-GPU run. Reduces to the old local mean when
            # world_size == 1 or per-rank counts are equal.
            task_mask_sum = mask.sum(axis=0)
            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(task_mask_sum, op=dist.ReduceOp.SUM)
            task_mask_sum = torch.clamp(task_mask_sum, min=1.0)  # Avoid division by zero
            task_losses = world_size * loss.sum(axis=0) / task_mask_sum
            loss = mtl_loss(task_losses)
        else:
            # Same global-count normalization for the single pooled loss (see the MTL branch).
            mask_sum = mask.sum()
            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(mask_sum, op=dist.ReduceOp.SUM)
            loss = world_size * loss.sum() / torch.clamp(mask_sum, min=1.0)

        loss_sum += loss.item()
        iter_count += args.batch_size

        cum_loss_sum += loss.item()
        cum_iter_count += 1

        loss.backward()

        # Under DDP, the model's gradients are all-reduced automatically by the
        # DDP wrapper. mtl_loss is a separate module (not DDP-wrapped) whose
        # log_sigma is registered in the optimizer, so its gradient must be
        # averaged across ranks manually to keep the learned task weights in sync.
        if world_size > 1 and mtl_loss is not None and dist.is_initialized():
            for p in mtl_loss.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size

        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += args.batch_size

        if is_main:
            current_lr = scheduler.get_lr()[-1]
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                mean=f"{cum_loss_sum / cum_iter_count:.4f}",
                lr=f"{current_lr:.1e}",
                refresh=False,
            )

        #if (n_iter // args.batch_size) % args.log_frequency == 0:
        #    lrs = scheduler.get_lr()
        #    loss_avg = loss_sum / iter_count
        #    loss_sum, iter_count = 0, 0
        #    lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
        # if cum_iter_count % 10 == 0:
        #     current_mem, peak_mem = get_memory_usage()
        #     print(f"After {cum_iter_count} iterations: {current_mem}MB, {peak_mem}MB")

    return n_iter, cum_loss_sum / cum_iter_count


def _write_epoch_history(path: str | Path, row: dict) -> None:
    """Write exactly one training-history row per epoch, including on resume."""
    output_path = Path(path)
    rows = []
    if output_path.is_file():
        rows = pd.read_csv(output_path).to_dict(orient="records")
    epoch = int(row["epoch"])
    rows = [previous for previous in rows if int(previous["epoch"]) != epoch]
    rows.append(row)
    pd.DataFrame(rows).sort_values("epoch").to_csv(output_path, index=False)


def _regression_metrics_by_task(preds, targets, task_names) -> dict[str, dict]:
    """Calculate ChemFlow's standard regression metrics for each KERMT task."""
    predictions = np.asarray(preds, dtype=float)
    observed = np.asarray(targets, dtype=float)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if observed.ndim == 1:
        observed = observed.reshape(-1, 1)

    result: dict[str, dict] = {}
    for task_index, task_name in enumerate(task_names):
        valid = np.isfinite(observed[:, task_index]) & np.isfinite(
            predictions[:, task_index]
        )
        y_true = observed[valid, task_index]
        y_pred = predictions[valid, task_index]
        metrics = {
            "n": int(valid.sum()),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "kendall": float("nan"),
        }
        if y_true.size:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        if y_true.size >= 2:
            metrics["r2"] = float(r2_score(y_true, y_pred))
            if np.unique(y_true).size > 1 and np.unique(y_pred).size > 1:
                for name, function in (
                    ("pearson", pearsonr),
                    ("spearman", spearmanr),
                    ("kendall", kendalltau),
                ):
                    try:
                        metrics[name] = float(function(y_true, y_pred).statistic)
                    except (ValueError, FloatingPointError):
                        pass
        result[str(task_name)] = metrics
    return result


def _write_task_metrics(path: str | Path, epoch: int, metrics_by_split) -> None:
    """Persist one row per epoch, split, and task without batch-level rows."""
    output_path = Path(path)
    rows = []
    if output_path.is_file():
        rows = pd.read_csv(output_path).to_dict(orient="records")
    rows = [previous for previous in rows if int(previous["epoch"]) != epoch]
    for split_name, task_metrics in metrics_by_split.items():
        for task_name, metrics in task_metrics.items():
            rows.append(
                {
                    "epoch": epoch,
                    "split": split_name,
                    "task": task_name,
                    **metrics,
                }
            )
    pd.DataFrame(rows).sort_values(["epoch", "split", "task"]).to_csv(
        output_path, index=False
    )


def run_training(args: Namespace, logger: Logger = None, return_val=False,
                 rank: int = 0, world_size: int = 1) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :param rank: DDP process rank (0 for single-process / non-DDP runs).
    :param world_size: number of DDP processes (1 for non-DDP runs).
    :return: A list of ensemble scores for each task.
    """
    # DDP is active only when a process group has been initialized (i.e. launched
    # with WORLD_SIZE>1 via `main.py finetune`, which spawns one process per GPU).
    # `python main.py finetune ...` without WORLD_SIZE keeps world_size=1,
    # is_distributed=False, and behaves exactly as before.
    is_distributed = dist.is_available() and dist.is_initialized()
    is_main = (rank == 0)

    if logger is not None and is_main:
        debug, info = logger.debug, logger.info
    else:
        # Non-rank-0 processes stay silent; single-process runs without a logger print.
        debug = info = (print if is_main else (lambda *a, **k: None))


    # pin GPU. Under DDP the device was already pinned to `rank` in ddp_setup;
    # only honor args.gpu for single-process runs.
    if args.cuda and not is_distributed and args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)

    # Default return value; overwritten with real scores on rank 0 after test eval.
    ensemble_scores = [float('nan')] * args.num_tasks

    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))
    val_smiles = val_data.smiles()
    scaled_val_targets = np.asarray(val_data.targets(), dtype=float)
    val_targets = (
        scaler.inverse_transform(scaled_val_targets).astype(float)
        if scaler is not None else scaled_val_targets
    )
    sum_val_preds = np.zeros((len(val_smiles), args.num_tasks))
    mc_test_draws = []
    mc_val_draws = []

    # Check if test data is blinded (no target columns)
    is_blinded_test = len(test_targets) == 0 or (len(test_targets) > 0 and len(test_targets[0]) == 0)

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Initialize TensorBoard writer if enabled (rank 0 only under DDP)
        if args.tensorboard and is_main:
            writer = SummaryWriter(save_dir)

        # Load/build model
        start_epoch = 0  # Default: start from epoch 0
        if getattr(args, "resume", False):
            selected_checkpoint = _resolve_resume_checkpoint(args, model_idx, save_dir)
            info(f'Resuming model {model_idx} from {selected_checkpoint}')
            model, loaded_ckpt_state = load_checkpoint(
                selected_checkpoint, current_args=args, logger=logger
            )
            required_resume_state = {"state_dict", "optimizer", "scheduler", "epoch"}
            missing_resume_state = sorted(
                required_resume_state.difference(loaded_ckpt_state)
            )
            if missing_resume_state:
                raise ValueError(
                    f"KERMT resume checkpoint {selected_checkpoint} is not a full "
                    f"restart checkpoint; missing: {missing_resume_state}. Use "
                    "last_checkpoint.pt rather than model.pt."
                )
        elif args.checkpoint_paths is not None:
            if len(args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            selected_checkpoint = args.checkpoint_paths[cur_model]
            debug(f'Loading model {cur_model} from {selected_checkpoint}')
            model, loaded_ckpt_state = load_checkpoint(
                selected_checkpoint, current_args=args, logger=logger
            )
        else:
            debug(f'Building model {model_idx}')
            model = build_model(model_idx=model_idx, args=args)
            loaded_ckpt_state = {}
            selected_checkpoint = None

        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

        if args.use_mtl_loss:
            mtl_loss = MTLLoss(args.num_tasks)
        else:
            mtl_loss = None

        debug(model)
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()
            if mtl_loss is not None:
                mtl_loss = mtl_loss.cuda()
        optimizer = build_optimizer(model, args)
        _log_model_audit(model, args, selected_checkpoint, debug)
        if args.use_mtl_loss:
            # Train log_sigma with same LR as task head (FFN params), not encoder
            optimizer.param_groups[1]['params'].append(mtl_loss.log_sigma)

        # Try to load optimizer state - only use start_epoch if optimizer loads successfully
        # (indicates resuming a finetune job vs starting fresh from pretrain checkpoint)
        optimizer_state_restored = False
        if selected_checkpoint is not None and "optimizer" in loaded_ckpt_state:
            try:
                info(f"Loading optimizer state from checkpoint: {selected_checkpoint}")
                optimizer.load_state_dict(loaded_ckpt_state["optimizer"])
                optimizer_state_restored = True
                # Only resume from checkpoint epoch if optimizer loaded successfully
                if "epoch" in loaded_ckpt_state:
                    start_epoch = loaded_ckpt_state["epoch"]
                    info(f"Resuming from epoch {start_epoch}")
            except (ValueError, RuntimeError) as error:
                if getattr(args, "resume", False):
                    raise RuntimeError(
                        "KERMT could not restore the optimizer state from "
                        f"{selected_checkpoint}: {error}"
                    ) from error
                info(f"Could not load optimizer state (model structure may differ): {error}")
                info("Starting fresh finetuning from epoch 0.")

        # Ensure a fresh model is available for evaluation if zero epochs are
        # requested. During resume, preserve the previous best model.pt until a
        # later epoch genuinely improves validation performance.
        if is_main and start_epoch == 0:
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Learning rate schedulers. Pass world_size so steps-per-epoch accounts
        # for the DistributedSampler sharding the data across ranks.
        scheduler = build_lr_scheduler(optimizer, args, world_size=world_size)
        # Only load scheduler state if we're resuming (start_epoch > 0 means optimizer loaded successfully)
        scheduler_state_restored = False
        if start_epoch > 0 and "scheduler" in loaded_ckpt_state:
            try:
                info(f"Loading scheduler state from checkpoint: {selected_checkpoint}")
                scheduler.load_state_dict(loaded_ckpt_state["scheduler"])
                scheduler_state_restored = True
            except (ValueError, KeyError, RuntimeError) as error:
                if getattr(args, "resume", False):
                    raise RuntimeError(
                        "KERMT could not restore the scheduler state from "
                        f"{selected_checkpoint}: {error}"
                    ) from error
                info(f"Could not load scheduler state: {error}")
                info("Starting with fresh scheduler state.")
        if getattr(args, "resume", False):
            if not optimizer_state_restored or not scheduler_state_restored:
                raise RuntimeError(
                    "KERMT resume did not restore both optimizer and scheduler state."
                )
            _log_resume_audit(model, selected_checkpoint, start_epoch, info)

        # Wrap in DDP for data-parallel training (optimizer was built on the
        # unwrapped model above, so its encoder/FFN param-group split is intact).
        # find_unused_parameters=True: the finetune model can carry parameters
        # not exercised by every forward (e.g. encoder sub-paths / heads retained
        # from the loaded pretrain checkpoint that the FFN task head does not use).
        # DDP would otherwise error ("expected to have finished reduction ...").
        if is_distributed:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        # Unwrapped handle for eval and checkpoint saving (portable, non-DDP state dict).
        core_model = model.module if is_distributed else model

        # Build data_loader. Under DDP, a DistributedSampler shards the train set
        # across ranks and owns the shuffling (loader shuffle must be False).
        shuffle = True
        mol_collator = MolCollator(shared_dict={}, args=args)
        # Use a separate name (train_loader) so train_data stays the underlying
        # dataset across ensemble iterations; reassigning it would double-wrap
        # the DataLoader (DataLoader(DataLoader(...))) on later ensemble models.
        if is_distributed:
            train_sampler = DistributedSampler(train_data, num_replicas=world_size,
                                               rank=rank, shuffle=True)
            # drop_last=False: DistributedSampler pads to an equal sample count per rank, so
            # every rank has the same number of batches and DDP stays in sync without dropping
            # data. drop_last=True would discard up to world_size*(batch_size-1) samples -- a
            # large fraction for small finetune sets on many GPUs -- and diverge from single-GPU.
            train_loader = DataLoader(train_data,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      collate_fn=mol_collator,
                                      sampler=train_sampler,
                                      drop_last=False)
        else:
            train_sampler = None
            train_loader = DataLoader(train_data,
                                      batch_size=args.batch_size,
                                      shuffle=shuffle,
                                      num_workers=0,
                                      collate_fn=mol_collator)
        # Run training
        saved_training_state = loaded_ckpt_state.get("training_state", {})
        if args.task_wise_checkpoint:
            best_score = {task: float('inf') if args.minimize_score else -float('inf') for task in args.task_names}
            curr_epoch_best_by_loss = {}
        else:
            best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch = {task: 0 for task in args.task_names}
        n_iter = int(saved_training_state.get("n_iter", 0))

        # Initialize validation losses
        if args.task_wise_checkpoint:
            min_val_loss = {task: float('inf') for task in args.task_names}
        else:
            min_val_loss = float('inf')
        if saved_training_state:
            best_score = saved_training_state.get("best_score", best_score)
            best_epoch = saved_training_state.get("best_epoch", best_epoch)
            min_val_loss = saved_training_state.get("min_val_loss", min_val_loss)
            debug(
                "Restored validation-selection state from the resume checkpoint "
                f"(best epoch: {best_epoch})."
            )
        early_stopping_patience = int(
            getattr(args, "early_stopping_patience", 0)
        )
        early_stopping_min_delta = float(
            getattr(args, "early_stopping_min_delta", 0.0)
        )
        early_stopping_monitor = str(
            getattr(args, "early_stopping_monitor", "metric")
        )
        early_stopping_minimize = (
            True if early_stopping_monitor == "loss" else bool(args.minimize_score)
        )
        early_stopping_state = saved_training_state.get("early_stopping", {})
        if (
            early_stopping_state
            and early_stopping_state.get("monitor") == early_stopping_monitor
        ):
            early_stopping_best = float(early_stopping_state["best_value"])
            early_stopping_bad_epochs = int(
                early_stopping_state.get("bad_epochs", 0)
            )
            info(
                "Restored early-stopping state: "
                f"best={early_stopping_best:.6f}, "
                f"epochs_without_improvement={early_stopping_bad_epochs}."
            )
        else:
            fallback = float("inf") if early_stopping_minimize else -float("inf")
            previous_best = (
                min_val_loss if early_stopping_monitor == "loss" else best_score
            )
            early_stopping_best = _mean_finite(previous_best, fallback)
            early_stopping_bad_epochs = 0
        if early_stopping_patience > 0:
            monitored_name = (
                "mean validation loss"
                if early_stopping_monitor == "loss"
                else f"mean validation {args.metric}"
            )
            info(
                f"Early stopping enabled: monitor={monitored_name}, "
                f"patience={early_stopping_patience}, "
                f"min_delta={early_stopping_min_delta:g}."
            )
        else:
            info("Early stopping disabled (patience=0).")
        for epoch in range(start_epoch, args.epochs):
            s_time = time.time()
            # Reshuffle the sharded train data differently each epoch (DDP).
            if is_distributed:
                train_sampler.set_epoch(epoch)
            n_iter, train_loss = train(
                epoch=epoch,
                model=model,
                data=train_loader,
                loss_func=loss_func,
                mtl_loss=mtl_loss,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                shared_dict=shared_dict,
                logger=logger,
                world_size=world_size
            )
            t_time = time.time() - s_time
            s_time = time.time()
            # Evaluate on the unwrapped model over the FULL val set. Under DDP, only rank 0
            # scores (evaluate() runs a plain, non-DDP forward on core_model, so it involves no
            # collectives and is safe to run on a single rank) -- this avoids every rank
            # redundantly scoring the whole val set, which is wasteful for large val sets. The
            # scores are then broadcast so every rank makes the same best-model / early-stop
            # decisions; only rank 0 writes checkpoints.
            if is_main:
                val_scores, val_loss, val_preds, val_targets = evaluate(
                    model=core_model,
                    data=val_data,
                    loss_func=loss_func,
                    num_tasks=args.num_tasks,
                    metric_func=metric_func,
                    batch_size=args.batch_size,
                    dataset_type=args.dataset_type,
                    scaler=scaler,
                    shared_dict=shared_dict,
                    logger=logger,
                    args=args,
                    return_predictions=True,
                )
            else:
                val_scores, val_loss = None, None
            if is_distributed:
                payload = [val_scores, val_loss]
                dist.broadcast_object_list(payload, src=0, device=torch.device(f"cuda:{rank}"))
                val_scores, val_loss = payload
            avg_val_loss = np.nanmean(val_loss)
            v_time = time.time() - s_time
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            early_stopping_value = float(
                avg_val_loss
                if early_stopping_monitor == "loss"
                else avg_val_score
            )
            (
                early_stopping_best,
                early_stopping_bad_epochs,
                _,
            ) = _update_early_stopping(
                early_stopping_value,
                early_stopping_best,
                early_stopping_bad_epochs,
                min_delta=early_stopping_min_delta,
                minimize=early_stopping_minimize,
            )
            should_stop = (
                early_stopping_patience > 0
                and early_stopping_bad_epochs >= early_stopping_patience
            )
            # Logged after lr step
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            train_eval_time = 0.0
            if is_main:
                train_eval_start = time.time()
                _, _, train_preds, train_targets = evaluate(
                    model=core_model,
                    data=train_data,
                    loss_func=loss_func,
                    num_tasks=args.num_tasks,
                    metric_func=metric_func,
                    batch_size=args.batch_size,
                    dataset_type=args.dataset_type,
                    scaler=scaler,
                    shared_dict=shared_dict,
                    logger=logger,
                    args=args,
                    return_predictions=True,
                )
                train_eval_time = time.time() - train_eval_start
                train_metrics = _regression_metrics_by_task(
                    train_preds, train_targets, args.task_names
                )
                val_metrics = _regression_metrics_by_task(
                    val_preds, val_targets, args.task_names
                )
            if is_distributed:
                # Non-zero ranks wait while rank zero evaluates the complete
                # training set with the unwrapped model. This prevents them
                # from entering the next DDP epoch early.
                dist.barrier()

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
            if is_main:
                print('Epoch: {:04d}'.format(epoch),
                      'loss_train: {:.6f}'.format(train_loss),
                      'loss_val: {:.6f}'.format(avg_val_loss),
                      f'{args.metric}_val: {avg_val_score:.4f}',
                      'cur_lr: {:.5f}'.format(scheduler.get_lr()[-1]),
                      't_time: {:.4f}s'.format(t_time),
                      'v_time: {:.4f}s'.format(v_time),
                      'train_eval_time: {:.4f}s'.format(train_eval_time),
                      flush=True)
                for split_label, split_metrics in (
                    ("Train", train_metrics),
                    ("Validation", val_metrics),
                ):
                    print(f"{split_label} metrics by task — epoch {epoch + 1}:")
                    for task_name, values in split_metrics.items():
                        print(
                            f"  {task_name} (n={values['n']}): "
                            f"RMSE={values['rmse']:.4f}, MAE={values['mae']:.4f}, "
                            f"R2={values['r2']:.4f}, Pearson={values['pearson']:.4f}, "
                            f"Spearman={values['spearman']:.4f}, "
                            f"Kendall={values['kendall']:.4f}",
                            flush=True,
                        )
                epoch_row = {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(avg_val_loss),
                    f"val_{args.metric}": float(avg_val_score),
                    "learning_rate": float(scheduler.get_lr()[-1]),
                    "train_time_seconds": float(t_time),
                    "validation_time_seconds": float(v_time),
                    "train_evaluation_time_seconds": float(train_eval_time),
                    "early_stopping_value": early_stopping_value,
                    "early_stopping_best": early_stopping_best,
                    "epochs_without_improvement": early_stopping_bad_epochs,
                }
                for task_name, val_score in zip(args.task_names, val_scores):
                    epoch_row[f"val_{task_name}_{args.metric}"] = float(val_score)
                for split_name, split_metrics in (
                    ("train", train_metrics),
                    ("val", val_metrics),
                ):
                    for task_name, values in split_metrics.items():
                        for metric_name, value in values.items():
                            epoch_row[
                                f"{split_name}_{task_name}_{metric_name}"
                            ] = value
                _write_epoch_history(
                    Path(save_dir) / "training_history.csv",
                    epoch_row,
                )
                _write_task_metrics(
                    Path(save_dir) / "training_task_metrics.csv",
                    epoch + 1,
                    {"train": train_metrics, "validation": val_metrics},
                )
            if args.wandb_project and is_main:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/time": t_time,
                    "val/time": v_time,
                    "val/loss": avg_val_loss,
                    f"val/{args.metric}": avg_val_score,
                    "cur_lr": scheduler.get_lr()[-1],
                }
                if args.show_individual_scores:
                    for task_name, val_score in zip(args.task_names, val_scores):
                        log_dict[f"val/{task_name}_{args.metric}"] = val_score
                wandb.log(log_dict)

            if args.tensorboard and is_main:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', avg_val_loss, epoch)
                writer.add_scalar(f'{args.metric}_val', avg_val_score, epoch)
                writer.add_scalar(
                    'early_stopping/epochs_without_improvement',
                    early_stopping_bad_epochs,
                    epoch,
                )

            # Always update min_val_loss as it is needed for HPO
            if args.task_wise_checkpoint:

                for itask, task_name in enumerate(args.task_names):
                    if val_loss[itask] < min_val_loss[task_name]:
                        curr_epoch_best_by_loss[task_name] = True
                        min_val_loss[task_name], best_epoch[task_name] = val_loss[itask], epoch
                    else:
                        curr_epoch_best_by_loss[task_name] = False
            else:
                if avg_val_loss < min_val_loss:
                    curr_epoch_best_by_loss = True
                    min_val_loss, best_epoch = avg_val_loss, epoch
                else:
                    curr_epoch_best_by_loss = False

            # Save model checkpoint if improved validation score.
            # Best-score tracking runs on all ranks (identical val metrics keep it
            # in sync); only rank 0 writes the checkpoint files (using core_model,
            # the unwrapped module, so the saved state dict is DDP-agnostic).
            if args.task_wise_checkpoint:
                if args.select_by_loss:
                    for task_name in args.task_names:
                        if curr_epoch_best_by_loss[task_name] and is_main:
                            print(f"Saving model {task_name} at epoch {epoch} with validation loss {min_val_loss[task_name]:.4f}")
                            save_checkpoint(os.path.join(save_dir, f'model_{task_name}.pt'), core_model, scaler, features_scaler, args)
                else:
                    for itask, task_name in enumerate(args.task_names):
                        task_val_score = val_scores[itask] if itask < len(val_scores) else avg_val_score
                        if args.minimize_score and task_val_score < best_score[task_name] or \
                                not args.minimize_score and task_val_score > best_score[task_name]:
                            best_score[task_name], best_epoch[task_name] = task_val_score, epoch
                            if is_main:
                                print(f"Saving model {task_name} at epoch {epoch} with validation score {best_score[task_name]:.4f}")
                                save_checkpoint(os.path.join(save_dir, f'model_{task_name}.pt'), core_model, scaler, features_scaler, args)
            else:
                if args.select_by_loss:
                    if curr_epoch_best_by_loss and is_main:
                        print(f"Saving model at epoch {epoch} with validation loss {min_val_loss:.4f}")
                        save_checkpoint(os.path.join(save_dir, 'model.pt'), core_model, scaler, features_scaler, args)
                else:
                    if args.minimize_score and avg_val_score < best_score or \
                            not args.minimize_score and avg_val_score > best_score:
                        best_score, best_epoch = avg_val_score, epoch
                        if is_main:
                            print(f"Saving model at epoch {epoch} with validation score {best_score:.4f}")
                            save_checkpoint(os.path.join(save_dir, 'model.pt'), core_model, scaler, features_scaler, args)
            if is_main:
                restart_state = {
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "min_val_loss": min_val_loss,
                    "n_iter": n_iter,
                    "early_stopping": {
                        "monitor": early_stopping_monitor,
                        "best_value": early_stopping_best,
                        "bad_epochs": early_stopping_bad_epochs,
                        "patience": early_stopping_patience,
                        "min_delta": early_stopping_min_delta,
                    },
                }
                save_model_for_restart(
                    os.path.join(save_dir, 'last_checkpoint.pt'),
                    core_model,
                    optimizer,
                    scheduler,
                    scaler,
                    features_scaler,
                    args,
                    epoch + 1,
                    training_state=restart_state,
                )
                checkpoint_interval = int(args.checkpoint_every_n_epochs)
                completed_epoch = epoch + 1
                if (
                    checkpoint_interval > 0
                    and completed_epoch % checkpoint_interval == 0
                ):
                    periodic_path = os.path.join(
                        save_dir,
                        f"epoch_{completed_epoch:04d}_checkpoint.pt",
                    )
                    save_model_for_restart(
                        periodic_path,
                        core_model,
                        optimizer,
                        scheduler,
                        scaler,
                        features_scaler,
                        args,
                        completed_epoch,
                        training_state=restart_state,
                    )
                    print(
                        f"Saved periodic checkpoint: {periodic_path}",
                        flush=True,
                    )
            if early_stopping_patience > 0 and is_main:
                print(
                    "Early stopping status: "
                    f"value={early_stopping_value:.6f}, "
                    f"best={early_stopping_best:.6f}, "
                    "epochs_without_improvement="
                    f"{early_stopping_bad_epochs}/{early_stopping_patience}",
                    flush=True,
                )
            if should_stop:
                if is_main:
                    info(
                        f"Early stopping triggered after epoch {epoch + 1}: "
                        f"no improvement greater than {early_stopping_min_delta:g} "
                        f"for {early_stopping_patience} consecutive epochs. "
                        "The best-validation model.pt is retained."
                    )
                break

        # Test-set evaluation + result writing happen on rank 0 only. All ranks
        # synchronize first; non-zero ranks then skip to the next ensemble member
        # (or fall out of the loop). Model weights are identical across ranks, so
        # rank 0's test scores are representative.
        if is_distributed:
            dist.barrier()
        if not is_main:
            continue

        ensemble_scores = 0.0

        # Evaluate on test set using model with best validation score
        if args.select_by_loss:
            if args.task_wise_checkpoint:
                for task_name in args.task_names:
                    info(f'Model {model_idx} best val loss = {min_val_loss[task_name]:.6f} on epoch {best_epoch[task_name]}')
            else:
                info(f'Model {model_idx} best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
        else:
            if args.task_wise_checkpoint:
                for task_name in args.task_names:
                    info(f'Model {model_idx} best validation {args.metric} = {best_score[task_name]:.6f} on epoch {best_epoch[task_name]}')
            else:
                info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

        if args.task_wise_checkpoint:
            test_preds = np.zeros((len(test_data), args.num_tasks))
            test_scores = []
            _loss_func = None if is_blinded_test else loss_func
            for itask, task_name in enumerate(args.task_names):
                print(f"{itask=}, {task_name=}")
                task_model, _ = load_checkpoint(os.path.join(save_dir, f'model_{task_name}.pt'), cuda=args.cuda, logger=logger)
                test_preds_task, _ = predict(
                    model=task_model,
                    data=test_data,
                    loss_func=_loss_func,
                    batch_size=args.batch_size,
                    logger=logger,
                    shared_dict=shared_dict,
                    scaler=scaler,
                    args=args
                )
                if not is_blinded_test:
                    test_scores_task = evaluate_predictions(
                        preds=test_preds_task,
                        targets=test_targets,
                        num_tasks=args.num_tasks,
                        metric_func=metric_func,
                        dataset_type=args.dataset_type,
                        logger=logger
                    )
                    test_scores.append(test_scores_task[itask])
                test_preds[:, itask] = np.array(test_preds_task)[:, itask]
            if is_blinded_test:
                test_scores = [float('nan')] * args.num_tasks
        else:
            model, _ = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)
            test_preds, _ = predict(
                model=model,
                data=test_data,
                loss_func=None if is_blinded_test else loss_func,
                batch_size=args.batch_size,
                logger=logger,
                shared_dict=shared_dict,
                scaler=scaler,
                args=args
            )
            val_preds, _ = predict(
                model=model,
                data=val_data,
                loss_func=None,
                batch_size=args.batch_size,
                logger=logger,
                shared_dict=shared_dict,
                scaler=scaler,
                args=args,
            )
            sum_val_preds += np.asarray(val_preds, dtype=float)
            for _ in range(int(getattr(args, 'test_mc_dropout_samples', 0))):
                mc_test, _ = predict(
                    model=model, data=test_data, loss_func=None,
                    batch_size=args.batch_size, logger=logger,
                    shared_dict=shared_dict, scaler=scaler, args=args,
                    mc_dropout=True,
                )
                mc_val, _ = predict(
                    model=model, data=val_data, loss_func=None,
                    batch_size=args.batch_size, logger=logger,
                    shared_dict=shared_dict, scaler=scaler, args=args,
                    mc_dropout=True,
                )
                mc_test_draws.append(np.asarray(mc_test, dtype=float))
                mc_val_draws.append(np.asarray(mc_val, dtype=float))
            if not is_blinded_test:
                test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=test_targets,
                    num_tasks=args.num_tasks,
                    metric_func=metric_func,
                    dataset_type=args.dataset_type,
                    logger=logger
                )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds, dtype=float)

        if not is_blinded_test and not args.task_wise_checkpoint:
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

        if not is_blinded_test:
            # Average test score
            avg_test_score = np.nanmean(test_scores)
            info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')

            if args.show_individual_scores:
                # Individual test scores
                for task_name, test_score in zip(args.task_names, test_scores):
                    info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
        else:
            info(f'Model {model_idx} test: Blinded data - skipping metric evaluation')

        # Evaluate ensemble on test set
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        if not is_blinded_test:
            ensemble_scores = evaluate_predictions(
                preds=avg_test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            # Output with both predictions and targets
            ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
            ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
            data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
            test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
            test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))

            # Average ensemble score
            avg_ensemble_test_score = np.nanmean(ensemble_scores)
            info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

            # Individual ensemble scores
            if args.show_individual_scores:
                for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                    info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
            if args.wandb_project:
                for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                    wandb.summary[f"test/{task_name}_{args.metric}"] = ensemble_score
        else:
            # Blinded test data - output predictions only
            ensemble_scores = [float('nan')] * args.num_tasks
            test_result = pd.DataFrame(avg_test_preds, index=test_smiles, columns=args.task_names)
            test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))
            info(f'Ensemble test: Blinded data - predictions saved to test_result.csv')

        # Close TensorBoard writer
        if args.tensorboard:
            writer.close()

    if (
        is_main and not is_blinded_test and not args.task_wise_checkpoint
        and int(getattr(args, 'test_mc_dropout_samples', 0)) >= 2
    ):
        direct_test = sum_test_preds / args.ensemble_size
        direct_val = sum_val_preds / args.ensemble_size
        stacked_test = np.stack(mc_test_draws, axis=0)
        stacked_val = np.stack(mc_val_draws, axis=0)
        mc_test = stacked_test.mean(axis=0)
        mc_test_sd = stacked_test.std(axis=0, ddof=1)
        mc_val = stacked_val.mean(axis=0)
        prediction_output = fingerprint_local_predictions(
            train_smiles=train_data.smiles(),
            validation_smiles=val_smiles,
            validation_truths=val_targets,
            validation_predictions=mc_val,
            test_smiles=test_smiles,
            direct_predictions=direct_test,
            mc_predictions=mc_test,
            mc_standard_deviations=mc_test_sd,
            targets=args.task_names,
            confidence=float(getattr(args, 'test_calibration_confidence', 0.90)),
        )
        output_root = Path(args.save_dir).parent
        summary = {
            'backend': 'KERMT',
            'selection': 'best_validation_checkpoint',
            'models': [
                str(Path(args.save_dir) / f'model_{index}' / 'model.pt')
                for index in range(args.ensemble_size)
            ],
            'targets': list(args.task_names),
        }
        _, summary = save_test_evaluation(
            workdir=output_root,
            smiles=test_smiles,
            truths=np.asarray(test_targets, dtype=float),
            targets=args.task_names,
            task_types=['regression'] * len(args.task_names),
            predictions=prediction_output,
            summary=summary,
            confidence=float(getattr(args, 'test_calibration_confidence', 0.90)),
        )
        (output_root / 'test_metrics.json').write_text(
            json.dumps(summary, indent=2) + '\n', encoding='utf-8'
        )

    if return_val:
        return ensemble_scores, min_val_loss
    else:
        return ensemble_scores


def load_data(args, debug, logger):
    """
    load the training data.
    :param args:
    :param debug:
    :param logger:
    :return:
    """
    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)
    else:
        args.features_dim = 0
    shared_dict = {}
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')
    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path, logger=logger)
    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type,
                                                     sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    #if args.save_smiles_splits:
    #    save_splits(args, test_data, train_data, val_data)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None
    args.train_data_size = len(train_data)
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        val_targets = val_data.targets()
        scaled_val_targets = scaler.transform(val_targets).tolist()
        val_data.set_targets(scaled_val_targets)
    else:
        scaler = None
    return features_scaler, scaler, shared_dict, test_data, train_data, val_data


def save_splits(args, test_data, train_data, val_data):
    """
    Save the splits.
    :param args:
    :param test_data:
    :param train_data:
    :param val_data:
    :return:
    """
    with open(args.data_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        lines_by_smiles = {}
        indices_by_smiles = {}
        for i, line in enumerate(reader):
            smiles = line[0]
            lines_by_smiles[smiles] = line
            indices_by_smiles[smiles] = i

    all_split_indices = []
    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles'])
            for smiles in dataset.smiles():
                writer.writerow([smiles])
        with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for smiles in dataset.smiles():
                writer.writerow(lines_by_smiles[smiles])
        split_indices = []
        for smiles in dataset.smiles():
            split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
        all_split_indices.append(split_indices)
    with open(os.path.join(args.save_dir, 'split_indices.json'), 'w') as f:
        json.dump(all_split_indices, f)
    return writer
