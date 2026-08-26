from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn
from src.deep_learning.fine_tune.lora import LoRALinear
from ..modules import GraphormerGraphEncoder
from src.deep_learning.sharing.adaptor import HardSharingMTL, SoftSharingMTL
import torch.nn.functional as F
from src.deep_learning.graphormer.modules.loss import MultiTaskGaussianNLLLoss, MultiTaskLaplaceNLLLoss


class ExtraFeatureEncoder(nn.Module):
    """
    Encode molecular descriptors / fingerprints before late fusion
    with the Graphormer representation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                f"input_dim must be > 0, got {input_dim}."
            )

        if output_dim <= 0:
            raise ValueError(
                f"output_dim must be > 0, got {output_dim}."
            )

        if hidden_dim is None:
            hidden_dim = max(
                output_dim,
                min(512, input_dim),
            )

        activation = activation.lower().strip()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                "Expected 'relu', 'gelu', or 'silu'."
            )

        layers = [
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        ]

        if use_layer_norm:
            layers.append(
                nn.LayerNorm(output_dim)
            )

        self.net = nn.Sequential(*layers)

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(
                    module.weight
                )
                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        if features is None:
            raise ValueError(
                "features cannot be None."
            )

        if features.ndim == 1:
            features = features.unsqueeze(0)

        if features.ndim != 2:
            raise ValueError(
                "features must have shape "
                "(batch_size, feature_dim), "
                f"got {tuple(features.shape)}."
            )

        if features.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"got {features.size(-1)}."
            )

        return self.net(
            features.float()
        )


def _get_batch_value(
    batched_data: Any,
    key: str,
):
    if isinstance(batched_data, dict):
        return batched_data.get(key)

    return getattr(
        batched_data,
        key,
        None,
    )


class FusedGraphormerEncoder(nn.Module):
    """
    Wrapper used by HardSharingMTL / SoftSharingMTL.

    It runs Graphormer first, then optionally appends encoded
    descriptor and fingerprint representations, and returns the
    fused representation.

    HardSharingMTL / SoftSharingMTL can therefore continue to treat
    this as their shared encoder.
    """

    def __init__(
        self,
        graph_encoder: nn.Module,
        descriptor_encoder: nn.Module | None,
        fingerprint_encoder: nn.Module | None,
        use_descriptors: bool,
        use_fingerprint: bool,
        fusion_dim: int,
    ) -> None:
        super().__init__()

        self.graph_encoder = graph_encoder
        self.descriptor_encoder = descriptor_encoder
        self.fingerprint_encoder = fingerprint_encoder

        self.use_descriptors = bool(
            use_descriptors
        )
        self.use_fingerprint = bool(
            use_fingerprint
        )

        self.fusion_dim = int(
            fusion_dim
        )

    def forward(
        self,
        batched_data,
        perturb: torch.Tensor | None = None,
        **kwargs,
    ):
        encoder_output = self.graph_encoder(
            batched_data=batched_data,
            perturb=perturb,
            **kwargs,
        )

        if (
            isinstance(encoder_output, tuple)
            and len(encoder_output) >= 2
        ):
            graph_rep = encoder_output[1]
        else:
            raise TypeError(
                "Graphormer encoder must return "
                "(encoder_output, graph_rep)."
            )

        representations = [
            graph_rep
        ]

        if self.use_descriptors:
            descriptor_features = _get_batch_value(
                batched_data,
                "descriptor_features",
            )

            if descriptor_features is None:
                raise KeyError(
                    "Descriptor branch is enabled, but "
                    "'descriptor_features' is missing "
                    "from the batch."
                )

            descriptor_features = descriptor_features.to(
                device=graph_rep.device,
                dtype=graph_rep.dtype,
            )

            descriptor_rep = self.descriptor_encoder(
                descriptor_features
            )

            representations.append(
                descriptor_rep
            )

        if self.use_fingerprint:
            fingerprint_features = _get_batch_value(
                batched_data,
                "fingerprint_features",
            )

            if fingerprint_features is None:
                raise KeyError(
                    "Fingerprint branch is enabled, but "
                    "'fingerprint_features' is missing "
                    "from the batch."
                )

            fingerprint_features = fingerprint_features.to(
                device=graph_rep.device,
                dtype=graph_rep.dtype,
            )

            fingerprint_rep = self.fingerprint_encoder(
                fingerprint_features
            )

            representations.append(
                fingerprint_rep
            )

        fused_rep = torch.cat(
            representations,
            dim=-1,
        )

        if fused_rep.size(-1) != self.fusion_dim:
            raise ValueError(
                f"Expected fused representation dim "
                f"{self.fusion_dim}, got "
                f"{fused_rep.size(-1)}."
            )

        # Return the same two-value style expected from Graphormer.
        return encoder_output[0], fused_rep


class GraphormerMultiTaskModel(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        print("Initializing GraphormerMultiTaskModel...")
        # ============================================================
        # Graphormer backbone
        # ============================================================

        self.encoder = GraphormerGraphEncoder(
            num_atoms=cfg.num_atoms,
            num_in_degree=cfg.num_in_degree,
            num_out_degree=cfg.num_out_degree,
            num_edges=cfg.num_edges,
            num_spatial=cfg.num_spatial,
            num_edge_dis=cfg.num_edge_dis,
            edge_type=cfg.edge_type,
            multi_hop_max_dist=cfg.multi_hop_max_dist,
            num_encoder_layers=cfg.num_encoder_layers,
            embedding_dim=cfg.encoder_embed_dim,
            ffn_embedding_dim=cfg.ffn_embedding_dim,
            num_attention_heads=cfg.encoder_attention_heads,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation_dropout=cfg.activation_dropout,
            layerdrop=getattr(cfg, "layerdrop", 0.0),
            encoder_normalize_before=cfg.encoder_normalize_before,
            pre_layernorm=cfg.pre_layernorm,
            apply_graphormer_init=cfg.apply_graphormer_init,
            activation_fn=cfg.activation_fn,
            embed_scale=getattr(cfg, "embed_scale", None),
            freeze_layer_indices=getattr(cfg, "freeze_layer_indices", None),
            traceable=getattr(cfg, "traceable", False),
            last_state_only=getattr(cfg, "last_state_only", False),
            use_quant_noise=getattr(cfg, "use_quant_noise", False),
            q_noise=getattr(cfg, "q_noise", 0.0),
            qn_block_size=getattr(cfg, "qn_block_size", 8),
        )


        # ============================================================
        # Load pretrained Graphormer backbone
        # ============================================================

        pretrained_path = getattr(cfg, "pretrained_path", None)

        if pretrained_path is not None:
            pretrained_path = Path(pretrained_path).expanduser().resolve()

            if not pretrained_path.exists():
                raise FileNotFoundError(f"Pretrained checkpoint does not exist: {pretrained_path}")

            self.load_pretrained_parameters(pretrained_path)
            
        # ============================================================
        # Freeze pretrained backbone
        # ============================================================

        # LoRA fine-tuning normally freezes the original backbone.
        freeze_encoder = getattr(cfg, "freeze_encoder", True)
        use_lora = getattr(cfg, "use_lora", False)

        if freeze_encoder or use_lora:
            self.freeze_encoder()

        # ============================================================
        # Add LoRA after freezing the original parameters
        # ============================================================

        if use_lora:
            self.apply_lora(cfg)  # change the encoder in place

        # ============================================================
        # Descriptor / fingerprint late-fusion branches
        # ============================================================

        self.use_extra_features = bool(
            getattr(
                cfg,
                "use_extra_features",
                False,
            )
        )

        self.use_descriptors = bool(
            getattr(
                cfg,
                "use_descriptors",
                False,
            )
        )

        self.use_fingerprint = bool(
            getattr(
                cfg,
                "use_fingerprint",
                False,
            )
        )

        if not self.use_extra_features:
            self.use_descriptors = False
            self.use_fingerprint = False

        self.descriptor_encoder = None
        self.fingerprint_encoder = None

        self.descriptor_dim = 0
        self.fingerprint_dim = 0

        self.descriptor_embedding_dim = 0
        self.fingerprint_embedding_dim = 0

        fusion_dim = int(
            cfg.encoder_embed_dim
        )

        print(
            f"Graphormer representation dim: "
            f"{cfg.encoder_embed_dim}"
        )

        # ------------------------------------------------------------
        # Descriptor branch
        # ------------------------------------------------------------
        if self.use_descriptors:
            self.descriptor_dim = int(
                getattr(
                    cfg,
                    "descriptor_dim",
                )
            )

            descriptor_hidden_dim = int(
                getattr(
                    cfg,
                    "descriptor_hidden_dim",
                    64,
                )
            )

            self.descriptor_embedding_dim = int(
                getattr(
                    cfg,
                    "descriptor_embedding_dim",
                    32,
                )
            )

            descriptor_dropout = float(
                getattr(
                    cfg,
                    "descriptor_dropout",
                    0.1,
                )
            )

            descriptor_activation = str(
                getattr(
                    cfg,
                    "descriptor_activation",
                    "gelu",
                )
            )

            self.descriptor_encoder = (
                ExtraFeatureEncoder(
                    input_dim=self.descriptor_dim,
                    hidden_dim=descriptor_hidden_dim,
                    output_dim=self.descriptor_embedding_dim,
                    dropout=descriptor_dropout,
                    activation=descriptor_activation,
                    use_layer_norm=True,
                )
            )

            fusion_dim += (
                self.descriptor_embedding_dim
            )

            print(
                "Descriptor branch enabled: "
                f"{self.descriptor_dim} -> "
                f"{descriptor_hidden_dim} -> "
                f"{self.descriptor_embedding_dim}"
            )
        else:
            print(
                "Descriptor branch disabled."
            )

        # ------------------------------------------------------------
        # Fingerprint branch
        # ------------------------------------------------------------
        if self.use_fingerprint:
            self.fingerprint_dim = int(
                getattr(
                    cfg,
                    "fingerprint_dim",
                )
            )

            fingerprint_hidden_dim = int(
                getattr(
                    cfg,
                    "fingerprint_hidden_dim",
                    512,
                )
            )

            self.fingerprint_embedding_dim = int(
                getattr(
                    cfg,
                    "fingerprint_embedding_dim",
                    128,
                )
            )

            fingerprint_dropout = float(
                getattr(
                    cfg,
                    "fingerprint_dropout",
                    0.1,
                )
            )

            fingerprint_activation = str(
                getattr(
                    cfg,
                    "fingerprint_activation",
                    "gelu",
                )
            )

            self.fingerprint_encoder = (
                ExtraFeatureEncoder(
                    input_dim=self.fingerprint_dim,
                    hidden_dim=fingerprint_hidden_dim,
                    output_dim=self.fingerprint_embedding_dim,
                    dropout=fingerprint_dropout,
                    activation=fingerprint_activation,
                    use_layer_norm=True,
                )
            )

            fusion_dim += (
                self.fingerprint_embedding_dim
            )

            print(
                "Fingerprint branch enabled: "
                f"{self.fingerprint_dim} -> "
                f"{fingerprint_hidden_dim} -> "
                f"{self.fingerprint_embedding_dim}"
            )
        else:
            print(
                "Fingerprint branch disabled."
            )

        self.fusion_dim = int(
            fusion_dim
        )

        print(
            f"Final fused representation dim: "
            f"{self.fusion_dim}"
        )

        # Wrap Graphormer so HardSharingMTL / SoftSharingMTL
        # receive fused representations transparently.
        self.fused_encoder = FusedGraphormerEncoder(
            graph_encoder=self.encoder,
            descriptor_encoder=self.descriptor_encoder,
            fingerprint_encoder=self.fingerprint_encoder,
            use_descriptors=self.use_descriptors,
            use_fingerprint=self.use_fingerprint,
            fusion_dim=self.fusion_dim,
        )

        self.adaptor_gate = getattr(cfg, "adaptor_gate", True)
        self.adaptor_gate_fn = getattr(cfg, "adaptor_gate_fn", "tanh")

        adaptor_kwargs = {
            "gate": self.adaptor_gate,
            "gate_fn": self.adaptor_gate_fn
            }

        sharing_type = getattr(cfg, "sharing_type", "hard").lower()

        if sharing_type not in {"hard", "soft"}:
            raise ValueError(f"Unsupported sharing_type: {sharing_type}. "
                "Expected 'hard' or 'soft'."
            )

        # Multi-task adapters/heads operate on the fused representation.
        feature_dim = self.fusion_dim

        num_adapters = getattr(cfg, "num_adapters", None)
        task_groups = getattr(cfg, "task_groups", None)

        if sharing_type == "hard":
            self.multi_task_model = HardSharingMTL(
                num_targets=cfg.num_tasks,
                shared_encoder=self.fused_encoder,
                dim=feature_dim,
                adaptor_bottleneck_dim=cfg.adaptor_bottleneck_dim,
                adaptor_dropout=cfg.adaptor_dropout,
                adaptor_activation=F.relu,
                adaptor_kwargs=adaptor_kwargs,
                num_adapters=num_adapters,
                task_groups=task_groups,
            )
        else:
            self.multi_task_model = SoftSharingMTL(
                num_targets=cfg.num_tasks,
                shared_encoder=self.fused_encoder,
                dim=feature_dim,
                adaptor_bottleneck_dim=cfg.adaptor_bottleneck_dim,
                adaptor_dropout=cfg.adaptor_dropout,
                adaptor_activation=F.relu,
                adaptor_kwargs=adaptor_kwargs,
                num_adapters=num_adapters,
                task_groups=task_groups,
            )

        task_weights = getattr(cfg, "task_weights", None)
        if task_weights is None:
            task_weight_method = getattr(cfg, "task_weight_method", "sqr_inverse")
            custom_task_weights = getattr(cfg, "custom_task_weights", None)
            task_weight_args = getattr(cfg, "task_weight_args", None)

            if task_weight_args is not None and isinstance(task_weight_args, dict):
                task_weights = task_weight_args.get("weights")
                if task_weight_method == "customed" and task_weights is None:
                    task_weights = custom_task_weights
            elif custom_task_weights is not None and task_weight_method in {"customed", "customized"}:
                task_weights = custom_task_weights

            if task_weights is None:
                task_weights = [1.0] * cfg.num_tasks

        self.register_buffer(
            "task_weights",
            torch.tensor(
                task_weights,
                dtype=torch.float32,
            ),
        )

        loss_type = getattr(cfg, "loss_type", "huber").lower()
        print(f"Using loss_type: {loss_type} for multi-task learning.")

        if loss_type == "mse": # MSE loss is also known as L2 loss
            self.loss_fn = nn.MSELoss(reduction="mean")
        elif loss_type == "mae":  # MAE (mean absolute error) is also known as L1 loss
            self.loss_fn = nn.L1Loss(reduction="mean")
        elif loss_type == "huber": # Huber loss is less sensitive to outliers than MSE and L1
            self.loss_fn = nn.HuberLoss(reduction="mean")
        elif loss_type == "laplace_nll": # Laplace loss is another name for L1 loss
            self.loss_fn = MultiTaskLaplaceNLLLoss(num_tasks=cfg.num_tasks)
        elif loss_type == "gaussian_nll": # Gaussian negative log-likelihood loss
            self.loss_fn = MultiTaskGaussianNLLLoss(num_tasks=cfg.num_tasks)
        else:
            raise ValueError(
                f"Unsupported loss_type: {loss_type}. "
                "Expected 'mse', 'mae', 'huber', 'laplace_nll', or 'gaussian_nll'."
            )

        self.print_model_summary()

    def forward(self, batched_data, perturb: torch.Tensor | None = None, **kwargs) -> dict[str, torch.Tensor]:
        """
        Predict all tasks for one batch.

        Parameters
        ----------
        batched_data:
            Batched Graphormer input.

        perturb:
            Optional perturbation passed to the Graphormer encoder.

        **kwargs:
            Additional keyword arguments forwarded to the encoder.

        return:
            Dictionary of task predictions, keyed by task name.
        """
        outputs = self.multi_task_model.forward_all(batched_data, perturb=perturb, **kwargs)
        if isinstance(batched_data, dict):
            y = batched_data.get("y")
        else:
            y = getattr(batched_data, "y", None)
        loss = None
        if y is not None:
            loss, task_losses = self.compute_multitask_loss(
                outputs=outputs,
                targets=y,
                loss_fn=self.loss_fn,
                task_weights=self.task_weights,
            )
        else: 
            task_losses = None
            loss = None
        
        predictions = torch.cat(
            [
                outputs[f"task_{i}"]
                for i in range(self.cfg.num_tasks)
            ],
            dim=1,
        )

        out_dict = {"predictions": predictions}
        if loss is not None:
            out_dict["loss"] = loss
        if task_losses is not None:
            out_dict["task_losses"] = task_losses

        return out_dict

    def forward_task(
        self,
        batched_data,
        task: int | str,
        perturb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict one task.
        """
        return self.multi_task_model.forward_task(
            batched_data,
            task=task,
            perturb=perturb,
            **kwargs,
        )

    def compute_multitask_loss(self, outputs: Mapping[str, torch.Tensor], targets: torch.Tensor, loss_fn: nn.Module, task_weights: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Compute loss for multi-task predictions.

        Parameters
        ----------
        outputs:
            Dictionary of task predictions:

                {
                    "task_0": Tensor[B, 1],
                    "task_1": Tensor[B, 1],
                    ...
                }

        targets:
            Target tensor with shape:

                [B, num_tasks]

            Missing labels can be represented by NaN.

        loss_fn:
            Loss function such as:

                nn.MSELoss()
                nn.L1Loss()
                nn.HuberLoss()

        task_weights:
            Optional tensor with shape [num_tasks].

            Example:
                tensor([1.0, 2.0, 0.5])

        Returns
        -------
        torch.Tensor
            Scalar multi-task loss.
        torch.Tensor
            Tensor of individual task losses.
        """

        ## Check inputs
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs must be a mapping from task names to tensors.")

        if not torch.is_tensor(targets):
            raise TypeError("targets must be a torch.Tensor.")

        if targets.ndim != 2:
            raise ValueError(
                f"targets must have shape [batch_size, num_tasks], "
                f"but got {tuple(targets.shape)}."
            )

        num_tasks = targets.shape[1]

        if len(outputs) != num_tasks:
            raise ValueError(
                f"Number of model outputs ({len(outputs)}) does not "
                f"match number of target tasks ({num_tasks})."
            )

        if task_weights is not None:
            if not torch.is_tensor(task_weights):
                raise TypeError("task_weights must be a torch.Tensor.")

            if task_weights.ndim != 1:
                raise ValueError("task_weights must be one-dimensional.")

            if task_weights.numel() != num_tasks:
                raise ValueError(
                    f"Expected {num_tasks} task weights, "
                    f"but got {task_weights.numel()}."
                )

            task_weights = task_weights.to(device=targets.device, dtype=targets.dtype)

        total_loss = None
        task_losses = []
        total_weight = 0.0

        for task_index in range(num_tasks):

            task_name = f"task_{task_index}"

            if task_name not in outputs:
                raise KeyError(f"Missing prediction for '{task_name}'.")

            prediction = outputs[task_name]

            if not torch.is_tensor(prediction):
                raise TypeError(f"Prediction for '{task_name}' must be a tensor.")

            # ----------------------------------------------------
            # Convert [B, 1] -> [B]
            # ----------------------------------------------------
            if prediction.ndim == 2 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)

            if prediction.ndim != 1:
                raise ValueError(
                    f"Prediction for '{task_name}' must have shape "
                    f"[B] or [B, 1], but got {tuple(prediction.shape)}."
                )

            target = targets[:, task_index]

            if prediction.shape[0] != target.shape[0]:
                raise ValueError(
                    f"Batch-size mismatch for '{task_name}': "
                    f"prediction has {prediction.shape[0]} samples, "
                    f"target has {target.shape[0]}."
                )

            # ----------------------------------------------------
            # Ignore missing labels
            # ----------------------------------------------------
            valid_mask = torch.isfinite(target)

            if not valid_mask.any():
                # No labels for this task in this batch.
                continue

            prediction_valid = prediction[valid_mask]
            target_valid = target[valid_mask]

            # ----------------------------------------------------
            # Per-task loss
            # ----------------------------------------------------
            if isinstance(loss_fn, (MultiTaskGaussianNLLLoss, MultiTaskLaplaceNLLLoss)):
                task_loss = loss_fn(prediction_valid, target_valid, task_index,)
            else:
                task_loss = loss_fn(prediction_valid, target_valid)

            if task_loss.ndim != 0:
                raise ValueError(
                    "loss_fn must return a scalar tensor. "
                    "Use a reduction such as reduction='mean'."
                )
            task_losses.append(
                task_loss.detach()
            )
            # ----------------------------------------------------
            # Optional task weighting
            # ----------------------------------------------------

            if task_weights is None:
                weight = 1.0
            else:
                weight = task_weights[task_index]

            weighted_loss = task_loss * weight

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

            if torch.is_tensor(weight):
                total_weight = total_weight + weight
            else:
                total_weight = total_weight + float(weight)

        # --------------------------------------------------------
        # Make sure at least one task has valid labels
        # --------------------------------------------------------

        if total_loss is None:
            raise ValueError("No valid target labels were found in this batch.")

        # Average across valid tasks / weights
        average_loss = total_loss / total_weight

        task_losses = torch.stack(task_losses) if task_losses else None

        return average_loss, task_losses
    
    def soft_sharing_regularization_loss(
        self,
        reduction: str = "mean",
        normalize_by_numel: bool = True,
    ) -> torch.Tensor:
        """
        Return encoder regularization for soft sharing.

        Hard sharing has only one encoder, so its regularization is zero.
        """
        if isinstance(self.multi_task_model, SoftSharingMTL):
            return self.multi_task_model.encoder_regularization_loss(
                reduction=reduction,
                normalize_by_numel=normalize_by_numel,
            )

        reference_parameter = next(self.multi_task_model.parameters())
        return reference_parameter.new_zeros(())

    def freeze_encoder(self) -> None:
        """Freeze all original Graphormer backbone parameters."""

        print("Freezing Graphormer encoder parameters...")

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        print("Graphormer encoder parameters are frozen.")

    def apply_lora(self, cfg: Any) -> None:
        """Inject LoRA modules into the Graphormer encoder."""

        print("Applying LoRA to the Graphormer encoder...")

        lora_target = getattr(
            cfg,
            "lora_target",
            "attention",
        ).lower()

        valid_targets = {"attention", "ffn", "all"}

        if lora_target not in valid_targets:
            raise ValueError(
                f"Unsupported lora_target: {lora_target}. "
                f"Expected one of {sorted(valid_targets)}."
            )

        if lora_target in {"attention", "all"}:
            result = self.add_lora_to_attention_layers(
                self.encoder,
                r=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                use_k_proj=cfg.apply_lora_to_k_proj,
            )

            # Support functions that either mutate in place or return
            # the modified model.
            if result is not None:
                self.encoder = result

            print("LoRA added to attention layers.")

        if lora_target in {"ffn", "all"}:
            result = self.add_lora_to_ffn_layers(
                self.encoder,
                r=cfg.lora_ffn_r,
                alpha=cfg.lora_ffn_alpha,
                dropout=cfg.lora_dropout,
                use_fc2=cfg.lora_use_fc2,
            )

            if result is not None:
                self.encoder = result

            print("LoRA added to FFN layers.")

        self._verify_lora_parameters()

    def _verify_lora_parameters(self) -> None:
        """Ensure that injected LoRA parameters are trainable."""

        lora_parameters = [
            (name, parameter)
            for name, parameter in self.encoder.named_parameters()
            if "lora_" in name.lower()
        ]

        if not lora_parameters:
            raise RuntimeError(
                "LoRA was requested, but no LoRA parameters were "
                "found in the encoder. Check the target module names."
            )

        for _, parameter in lora_parameters:
            parameter.requires_grad = True

        print(
            f"Found {len(lora_parameters)} trainable LoRA "
            f"parameter tensors."
        )

    def load_pretrained_parameters(
        self,
        pretrained_path: str | Path,
    ) -> None:
        """
        Load only the GraphormerGraphEncoder backbone parameters.

        Expected pretrained checkpoint keys may look like:

            encoder.graph_encoder.layers.0...
            encoder.graph_encoder.graph_node_feature...

        Current encoder keys look like:

            layers.0...
            graph_node_feature...
        """

        pretrained_path = Path(
            pretrained_path
        ).expanduser().resolve()

        print(
            f"Loading pretrained parameters from "
            f"{pretrained_path}..."
        )

        checkpoint = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._remove_ddp_prefix(state_dict)

        backbone_prefixes = (
            "encoder.graph_encoder.",
            "graph_encoder.",
            "encoder.",
        )

        encoder_state_dict = None
        selected_prefix = None

        for prefix in backbone_prefixes:
            matched_state = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }

            if matched_state:
                encoder_state_dict = matched_state
                selected_prefix = prefix
                break

        # The checkpoint may already contain raw encoder keys.
        if encoder_state_dict is None:
            current_encoder_keys = set(
                self.encoder.state_dict().keys()
            )

            encoder_state_dict = {
                key: value
                for key, value in state_dict.items()
                if key in current_encoder_keys
            }

            selected_prefix = "<none>"

        if not encoder_state_dict:
            raise RuntimeError(
                "No Graphormer encoder parameters were found in "
                f"checkpoint: {pretrained_path}"
            )

        print(
            f"Using checkpoint prefix: {selected_prefix}"
        )
        print(
            f"Found {len(encoder_state_dict):,} encoder tensors."
        )

        load_result = self.encoder.load_state_dict(
            encoder_state_dict,
            strict=False,
        )

        self._print_load_result(
            missing_keys=load_result.missing_keys,
            unexpected_keys=load_result.unexpected_keys,
        )

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        """Extract a model state dictionary from common formats."""

        if not isinstance(checkpoint, dict):
            return checkpoint

        for key in (
            "model_state_dict",
            "model",
            "state_dict",
        ):
            candidate = checkpoint.get(key)

            if isinstance(candidate, dict):
                return candidate

        # The checkpoint itself may already be a state dictionary.
        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint

        raise KeyError(
            "Could not find a model state dictionary in the "
            "checkpoint. Expected one of: model_state_dict, "
            "model, or state_dict."
        )

    @staticmethod
    def _remove_ddp_prefix(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Remove the DDP `module.` prefix when present."""

        return {
            (
                key[len("module."):]
                if key.startswith("module.")
                else key
            ): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _print_load_result(
        missing_keys: list[str],
        unexpected_keys: list[str],
    ) -> None:
        """Print checkpoint loading diagnostics."""

        if missing_keys:
            print(
                f"Missing encoder keys ({len(missing_keys)}):"
            )

            for key in missing_keys:
                print(f"  {key}")
        else:
            print("No missing encoder keys.")

        if unexpected_keys:
            print(
                f"Unexpected encoder keys "
                f"({len(unexpected_keys)}):"
            )

            for key in unexpected_keys:
                print(f"  {key}")
        else:
            print("No unexpected encoder keys.")

    def print_model_summary(self) -> None:
        """Print total and trainable parameter counts."""

        total_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        trainable_ratio = (
            100.0 * trainable_parameters / total_parameters
            if total_parameters > 0
            else 0.0
        )

        print("GraphormerMultiTaskModel initialized.")
        print(
            f"Extra features enabled: "
            f"{self.use_extra_features}"
        )
        print(
            f"Descriptor branch enabled: "
            f"{self.use_descriptors}"
        )
        print(
            f"Fingerprint branch enabled: "
            f"{self.use_fingerprint}"
        )
        print(
            f"Final fused representation dim: "
            f"{self.fusion_dim}"
        )
        print(f"Total parameters: {total_parameters:,}")
        print(
            f"Trainable parameters: "
            f"{trainable_parameters:,} "
            f"({trainable_ratio:.4f}%)"
        )

        print("Trainable parameter names:")

        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                print(
                    f"  {name}: {tuple(parameter.shape)}"
                )

    def add_lora_to_attention_layers(
        self,
        model: nn.Module,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_k_proj: bool = False,
    ) -> nn.Module:
        """
        Add LoRA to Graphormer attention projection layers.

        Expected Graphormer structure:
            model.layers[i].self_attn.q_proj
            model.layers[i].self_attn.k_proj
            model.layers[i].self_attn.v_proj
        """

        if not hasattr(model, "layers"):
            raise AttributeError(
                f"{type(model).__name__} has no attribute 'layers'. "
                "Expected a GraphormerGraphEncoder."
            )

        num_modified = 0

        for layer_idx, layer in enumerate(model.layers):
            if not hasattr(layer, "self_attn"):
                raise AttributeError(
                    f"Encoder layer {layer_idx} has no attribute "
                    "'self_attn'."
                )

            attention = layer.self_attn

            if not hasattr(attention, "q_proj"):
                raise AttributeError(
                    f"Layer {layer_idx} self_attn has no q_proj."
                )

            if not hasattr(attention, "v_proj"):
                raise AttributeError(
                    f"Layer {layer_idx} self_attn has no v_proj."
                )

            attention.q_proj = LoRALinear(
                attention.q_proj,
                r=r,
                alpha=alpha,
                dropout=dropout,
            )

            attention.v_proj = LoRALinear(
                attention.v_proj,
                r=r,
                alpha=alpha,
                dropout=dropout,
            )

            num_modified += 2

            if use_k_proj:
                if not hasattr(attention, "k_proj"):
                    raise AttributeError(
                        f"Layer {layer_idx} self_attn has no k_proj."
                    )

                attention.k_proj = LoRALinear(
                    attention.k_proj,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )

                num_modified += 1

        print(
            f"Added LoRA to {num_modified} attention projection "
            f"layers across {len(model.layers)} Graphormer layers."
        )

        return model

    def add_lora_to_ffn_layers(
        self,
        model: nn.Module,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_fc2: bool = False,
    ) -> nn.Module:
        """
        Add LoRA to Graphormer feed-forward layers.

        Expected structure:
            model.layers[i].fc1
            model.layers[i].fc2
        """

        if not hasattr(model, "layers"):
            raise AttributeError(
                f"{type(model).__name__} has no attribute 'layers'."
            )

        num_modified = 0

        for layer_idx, layer in enumerate(model.layers):
            if not hasattr(layer, "fc1"):
                raise AttributeError(
                    f"Encoder layer {layer_idx} has no fc1."
                )

            layer.fc1 = LoRALinear(
                layer.fc1,
                r=r,
                alpha=alpha,
                dropout=dropout,
            )

            num_modified += 1

            if use_fc2:
                if not hasattr(layer, "fc2"):
                    raise AttributeError(
                        f"Encoder layer {layer_idx} has no fc2."
                    )

                layer.fc2 = LoRALinear(
                    layer.fc2,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )

                num_modified += 1

        print(
            f"Added LoRA to {num_modified} FFN layers across "
            f"{len(model.layers)} Graphormer layers."
        )

        return model