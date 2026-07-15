from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..modules import GraphormerGraphEncoder, RegressionHead, ClassificationHead
from src.deep_learning.fine_tune.lora import LoRALinear

class GraphormerFineTuneRegressionModel(nn.Module):
    """
    Graphormer model for downstream regression fine-tuning.

    Architecture:
        GraphormerGraphEncoder
            -> graph representation (CLS token)
            -> regression head
            -> continuous prediction
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()

        self.cfg = cfg
        print("Initializing GraphormerFineTuneRegressionModel...")
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
            freeze_layer_indices=getattr(
                cfg,
                "freeze_layer_indices",
                None,
            ),
            traceable=getattr(cfg, "traceable", False),
            last_state_only=getattr(
                cfg,
                "last_state_only",
                False,
            ),
            use_quant_noise=getattr(
                cfg,
                "use_quant_noise",
                False,
            ),
            q_noise=getattr(cfg, "q_noise", 0.0),
            qn_block_size=getattr(cfg, "qn_block_size", 8),
        )

        # ============================================================
        # Regression head
        # ============================================================

        # The input dimension must match graph_rep:
        # graph_rep shape = (batch_size, encoder_embed_dim)
        self.regression_head = RegressionHead(
            hidden_dim=cfg.encoder_embed_dim,
            intermediate_dim=cfg.head_intermediate_dim,
            dropout=cfg.head_dropout,
        )

        self._initialize_regression_head_weights()

        # ============================================================
        # Load pretrained Graphormer backbone
        # ============================================================

        pretrained_path = getattr(cfg, "pretrained_path", None)

        if pretrained_path is not None:
            pretrained_path = Path(pretrained_path).expanduser().resolve()

            if not pretrained_path.exists():
                raise FileNotFoundError(
                    f"Pretrained checkpoint does not exist: "
                    f"{pretrained_path}"
                )

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
            self.apply_lora(cfg)

        self.print_model_summary()

        self.loss_fn = nn.L1Loss()

    def forward(
        self,
        batched_data,
        perturb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            batched_data:
                Graphormer input batch.

            perturb:
                Optional perturbation tensor.

        Returns:
            Classification logits.

            Multi-class classification:
                shape (batch_size, num_classes)

            Binary classification with one logit:
                shape (batch_size,) or (batch_size, 1)
        """

        _, graph_rep = self.encoder(
            batched_data=batched_data,
            perturb=perturb,
            **kwargs,
        )

        # graph_rep: (batch_size, encoder_embed_dim)
        predictions = self.regression_head(graph_rep)

        # For single-target regression, optionally remove the final
        # dimension: (B, 1) -> (B,)
        if predictions.ndim == 2 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)

        if isinstance(batched_data, dict):
            y = batched_data.get("y")
        else:
            y = getattr(batched_data, "y", None)
            
        loss = None

        if y is not None:
            y = y.to(
                device=predictions.device,
                dtype=predictions.dtype,
            )
            # Normalize single-target label shape:
            # (B, 1) -> (B,)
            if y.ndim == 2 and y.size(-1) == 1:
                y = y.squeeze(-1)
            if predictions.shape != y.shape:
                raise ValueError(
                    f"Predictions shape {tuple(predictions.shape)} does not "
                    f"match target shape {tuple(y.shape)}."
                )

            loss = self.compute_loss(predictions=predictions,targets=y,)

        out_dict = {
            "predictions": predictions,
        }

        if graph_rep is not None:
            out_dict["graph_rep"] = graph_rep

        if loss is not None:
            out_dict["loss"] = loss

        return out_dict
    
    def compute_loss(self, predictions, targets):
        loss = self.loss_fn(predictions, targets)
        return loss

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

    def _initialize_regression_head_weights(self) -> None:
        """Initialize task-specific regression head parameters."""

        print("Initializing regression head weights...")

        for module in self.regression_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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

        print("GraphormerFineTuneRegressionModel initialized.")
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


class GraphormerFineTuneClassificationModel(nn.Module):
    """
    Graphormer model for downstream classification fine-tuning.

    Architecture:
        GraphormerGraphEncoder
            -> graph representation (CLS token)
            -> classification head
            -> class prediction
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()

        self.cfg = cfg
        print("Initializing GraphormerFineTuneClassificationModel...")
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
            freeze_layer_indices=getattr(
                cfg,
                "freeze_layer_indices",
                None,
            ),
            traceable=getattr(cfg, "traceable", False),
            last_state_only=getattr(
                cfg,
                "last_state_only",
                False,
            ),
            use_quant_noise=getattr(
                cfg,
                "use_quant_noise",
                False,
            ),
            q_noise=getattr(cfg, "q_noise", 0.0),
            qn_block_size=getattr(cfg, "qn_block_size", 8),
        )

        # ============================================================
        # Classification head
        # ============================================================

        # The input dimension must match graph_rep:
        # graph_rep shape = (batch_size, encoder_embed_dim)
        self.classification_head = ClassificationHead(
            hidden_dim=cfg.encoder_embed_dim,
            intermediate_dim=cfg.head_intermediate_dim,
            num_classes=cfg.num_classes,
            dropout=cfg.head_dropout,
        )
        if cfg.num_classes < 1:
            raise ValueError(
                f"num_classes must be >= 1 for classification, "
                f"got {cfg.num_classes}."
            )
        print(f'Number of classes: {cfg.num_classes}')

        self._initialize_classification_head_weights()

        # ============================================================
        # Load pretrained Graphormer backbone
        # ============================================================

        pretrained_path = getattr(cfg, "pretrained_path", None)

        if pretrained_path is not None:
            pretrained_path = Path(pretrained_path).expanduser().resolve()

            if not pretrained_path.exists():
                raise FileNotFoundError(
                    f"Pretrained checkpoint does not exist: "
                    f"{pretrained_path}"
                )

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
            self.apply_lora(cfg)

        self.print_model_summary()

        loss_type = cfg.loss_type.lower()

        if loss_type == "bce":
            if cfg.num_classes != 1:
                raise ValueError(
                    "BCEWithLogitsLoss expects num_classes=1, "
                    f"but got num_classes={cfg.num_classes}."
                )

            positive_weight = getattr(
                cfg,
                "positive_weight",
                None,
            )

            if positive_weight is not None:
                positive_weight_tensor = torch.tensor(
                    [positive_weight],
                    dtype=torch.float32,
                )
            else:
                positive_weight_tensor = None

            self.register_buffer(
                "positive_weight",
                positive_weight_tensor,
            )

            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.positive_weight,
            )

        elif loss_type == "cross_entropy":
            if cfg.num_classes < 2:
                raise ValueError(
                    "CrossEntropyLoss expects num_classes >= 2, "
                    f"but got num_classes={cfg.num_classes}."
                )

            class_weights = getattr(
                cfg,
                "class_weights",
                None,
            )

            if class_weights is not None:
                if len(class_weights) != cfg.num_classes:
                    raise ValueError(
                        f"class_weights must contain exactly "
                        f"{cfg.num_classes} values, but got "
                        f"{len(class_weights)}."
                    )

                class_weight_tensor = torch.tensor(
                    class_weights,
                    dtype=torch.float32,
                )
            else:
                class_weight_tensor = None

            self.register_buffer(
                "class_weights",
                class_weight_tensor,
            )

            self.loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights,
            )

        else:
            raise ValueError(
                f"Unsupported loss_type: {cfg.loss_type}. "
                "Expected 'bce' or 'cross_entropy'."
            )

    def forward(
        self,
        batched_data,
        perturb: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        _, graph_rep = self.encoder(
            batched_data=batched_data,
            perturb=perturb,
            **kwargs,
        )

        logits = self.classification_head(graph_rep)

        if isinstance(batched_data, dict):
            targets = batched_data.get("y")
        else:
            targets = getattr(batched_data, "y", None)

        loss = None

        if targets is not None:
            targets = targets.to(logits.device)

            loss = self.compute_loss(logits=logits, targets=targets,)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "graph_rep": graph_rep,
        }

        if loss is not None:
            output["loss"] = loss

        return output

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.loss_type.lower() == "bce":
            # logits: (B, 1) or (B,)
            # targets: (B, 1) or (B,)
            logits = logits.squeeze(-1)

            if targets.ndim == 2 and targets.size(-1) == 1:
                targets = targets.squeeze(-1)

            targets = targets.to(dtype=logits.dtype)

            if logits.shape != targets.shape:
                raise ValueError(
                    f"For BCE, logits shape {tuple(logits.shape)} "
                    f"must match targets shape {tuple(targets.shape)}."
                )

            return self.loss_fn(logits, targets)

        if self.cfg.loss_type.lower() == "cross_entropy":
            # logits: (B, C)
            # targets: (B,)
            if targets.ndim == 2 and targets.size(-1) == 1:
                targets = targets.squeeze(-1)

            targets = targets.long()

            if logits.ndim != 2:
                raise ValueError(
                    "CrossEntropyLoss expects logits with shape "
                    f"(batch_size, num_classes), got {tuple(logits.shape)}."
                )

            if targets.ndim != 1:
                raise ValueError(
                    "CrossEntropyLoss expects targets with shape "
                    f"(batch_size,), got {tuple(targets.shape)}."
                )

            if logits.size(0) != targets.size(0):
                raise ValueError(
                    f"Batch-size mismatch: logits have "
                    f"{logits.size(0)} samples, but targets have "
                    f"{targets.size(0)}."
                )

            return self.loss_fn(
                logits,
                targets,
            )

        raise RuntimeError(
            f"Unexpected loss type: {self.cfg.loss_type}"
        )

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

    def _initialize_classification_head_weights(self) -> None:
        """Initialize task-specific classification head parameters."""

        print("Initializing classification head weights...")

        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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

        print("GraphormerFineTuneRegressionModel initialized.")
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
