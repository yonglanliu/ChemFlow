"""
For multi-task learning, we can share the embedding layer and the encoder layer across different tasks. This module provides a way to share the embedding layer and the encoder layer.
Two sharing strategies are provided:
1. Hard sharing: 
    - Tasks share the exact same parameters in early layers, then split into task-specific heads for final predictions. 
    - The shared layers learn common representations while specialized layers handle task-specific features.
2. Soft sharing: 
    - Each task has its own separate parameters, but they're encouraged to be similar through regularization terms. 
    - Tasks influence each other indirectly through shared constraints rather than shared weights.


How Hard Sharing Works: 
    - Start with a common backbone network that all tasks use. 
    - Add task-specific branches or heads on top of the shared layers. 
    - During training, gradients from all tasks flow through the shared layers, forcing them to learn generalizable features.

How Soft Sharing Works: 
    - Create separate networks for each task but add penalty terms that encourage similar parameters across tasks. 
    - Tasks can diverge when needed but are pulled toward similarity by regularization constraints.
    - By keeping weights close, one task’s learning nudges the other in the right direction. Regularizing weights across tasks keeps them grounded in generalizable features, not just task-specific quirks.

When Hard Sharing Works Better:
--> Similar Task Domains: When tasks are closely related like sentiment analysis and emotion detection, shared representations make sense.
--> Limited Data Per Task: Hard sharing acts as regularization, preventing overfitting when individual tasks have small datasets.
--> Resource Constraints: Single shared model requires less memory and computation than multiple separate models.
--> Feature Commonality: When tasks benefit from similar low-level features like edge detection in computer vision.

When Soft Sharing Works Better:
--> Task Conflicts: When tasks have competing requirements that would hurt shared representations. Translation and summarization might need different text encodings.
--> Different Data Distributions: Tasks from different domains where forced sharing could degrade performance on individual tasks.
--> Unequal Task Importance: When some tasks are more critical and shouldn't be compromised by sharing constraints.
--> Varying Task Complexity: Complex tasks might need more parameters while simple tasks need fewer, making equal sharing suboptimal.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from src.deep_learning.sharing.adaptor import ResidualAdaptor
from src.deep_learning.graphormer.modules.graphormer_encoder import GraphormerGraphEncoder
from itertools import combinations



class HardSharingMTL(nn.Module):
    """
    Hard parameter sharing multi-task learning model.

    All tasks share the same encoder. Each task has its own adaptor and
    prediction head.

    Parameters
    ----------
    num_targets:
        Number of tasks.
    shared_encoder:
        Shared feature encoder.
    dim:
        Dimension of the shared encoder output.
    adaptor_bottleneck:
        Bottleneck dimension of each task-specific adaptor.
    dropout:
        Dropout probability used by the adaptors.
    adaptor_cls:
        Adaptor module class. It must return an nn.Module when initialized.
    adaptor_activation:
        Activation function or activation module class used by the adaptor.
    adaptor_kwargs:
        Additional keyword arguments passed to the adaptor constructor.
    """

    def __init__(
        self,
        num_targets: int,
        shared_encoder: nn.Module,
        dim: int = 512,
        adaptor_bottleneck: int = 32,
        dropout: float = 0.1,
        adaptor_cls: Callable[..., nn.Module] = ResidualAdaptor,
        adaptor_activation: Union[
            Callable[[torch.Tensor], torch.Tensor],
            type[nn.Module],
        ] = nn.ReLU,
        adaptor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if num_targets <= 0:
            raise ValueError(
                f"num_targets must be positive, but got {num_targets}."
            )

        if dim <= 0:
            raise ValueError(f"dim must be positive, but got {dim}.")

        if adaptor_bottleneck <= 0:
            raise ValueError(
                "adaptor_bottleneck must be positive, "
                f"but got {adaptor_bottleneck}."
            )

        self.shared_encoder = shared_encoder
        self.dim = int(dim)
        self.num_targets = int(num_targets)

        adaptor_kwargs = dict(adaptor_kwargs or {})

        self.adaptors = nn.ModuleList(
            [
                adaptor_cls(
                    dim=self.dim,
                    bottleneck=adaptor_bottleneck,
                    dropout=dropout,
                    activation=adaptor_activation,
                    **adaptor_kwargs,
                )
                for _ in range(self.num_targets)
            ]
        )

        self.heads = nn.ModuleList(
            [
                nn.Linear(self.dim, 1)
                for _ in range(self.num_targets)
            ]
        )

        self._initialize_heads()

    def _initialize_heads(self) -> None:
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def _parse_task_index(
        self,
        task: Union[int, str],
    ) -> int:
        """
        Convert an integer task index or a name such as 'task_0'
        into a valid task index.
        """
        if isinstance(task, int):
            task_index = task

        elif isinstance(task, str):
            if not task.startswith("task_"):
                raise ValueError(
                    f"Invalid task name '{task}'. "
                    "Expected a name such as 'task_0'."
                )

            index_string = task.removeprefix("task_")

            try:
                task_index = int(index_string)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid task name '{task}'. "
                    "The part after 'task_' must be an integer."
                ) from exc

        else:
            raise TypeError(
                "task must be an integer index or a string such as "
                f"'task_0', but got {type(task).__name__}."
            )

        if not 0 <= task_index < self.num_targets:
            raise IndexError(
                f"Task index {task_index} is out of range. "
                f"Expected an index between 0 and "
                f"{self.num_targets - 1}."
            )

        return task_index

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the shared representation.
        """
        shared_features = self.shared_encoder(x)

        if not isinstance(shared_features, torch.Tensor):
            raise TypeError(
                "shared_encoder must return a torch.Tensor, "
                f"but returned {type(shared_features).__name__}."
            )

        if shared_features.shape[-1] != self.dim:
            raise ValueError(
                "The last dimension of the shared encoder output must "
                f"be {self.dim}, but got {shared_features.shape[-1]}."
            )

        return shared_features

    def forward_task(
        self,
        shared_features: torch.Tensor,
        task: Union[int, str],
    ) -> torch.Tensor:
        """
        Run one task-specific adaptor and prediction head.
        """
        task_index = self._parse_task_index(task)

        task_features = self.adaptors[task_index](shared_features)
        output = self.heads[task_index](task_features)

        return output

    def forward(
        self,
        x: torch.Tensor,
        task: Union[int, str],
    ) -> torch.Tensor:
        shared_features = self.encode(x)
        return self.forward_task(shared_features, task)

    def forward_all(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Return predictions for every task.

        This is useful when one batch contains labels for all tasks.
        """
        shared_features = self.encode(x)

        outputs = {
            f"task_{task_index}": self.heads[task_index](
                self.adaptors[task_index](shared_features)
            )
            for task_index in range(self.num_targets)
        }

        return outputs


class SoftSharingMTL(nn.Module):
    """
    Soft parameter sharing multi-task learning model.

    Each task has its own encoder and prediction head. Optionally, each task
    can also have its own adaptor.

    Soft sharing is implemented by adding a regularization loss that encourages
    corresponding parameters of different task-specific encoders to remain
    similar.

    Parameters
    ----------
    num_targets:
        Number of prediction tasks.

    encoder_cls:
        Encoder class used to construct one independent encoder per task.

    dim:
        Dimension of the encoder output.

    adaptor_bottleneck:
        Bottleneck dimension of each optional task-specific adaptor.

    dropout:
        Dropout probability used by the adaptor.

    adaptor_cls:
        Optional adaptor class. If None, no adaptor is used.

    adaptor_activation:
        Activation passed to the adaptor.

    encoder_kwargs:
        Additional keyword arguments passed to encoder_cls.

    adaptor_kwargs:
        Additional keyword arguments passed to adaptor_cls.
    """

    def __init__(
        self,
        num_targets: int,
        encoder_cls: Callable[..., nn.Module],
        dim: int = 512,
        adaptor_bottleneck: int = 32,
        dropout: float = 0.1,
        adaptor_cls: Optional[Callable[..., nn.Module]] = None,
        adaptor_activation: Union[
            Callable[[torch.Tensor], torch.Tensor],
            type[nn.Module],
        ] = nn.ReLU,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        adaptor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if num_targets <= 0:
            raise ValueError(
                f"num_targets must be positive, but got {num_targets}."
            )

        if dim <= 0:
            raise ValueError(
                f"dim must be positive, but got {dim}."
            )

        if adaptor_bottleneck <= 0:
            raise ValueError(
                "adaptor_bottleneck must be positive, "
                f"but got {adaptor_bottleneck}."
            )

        self.dim = int(dim)
        self.num_targets = int(num_targets)
        self.adaptor_cls = adaptor_cls

        encoder_kwargs = dict(encoder_kwargs or {})
        adaptor_kwargs = dict(adaptor_kwargs or {})

        self.encoders = nn.ModuleList(
            [
                encoder_cls(
                    dim=self.dim,
                    **encoder_kwargs,
                )
                for _ in range(self.num_targets)
            ]
        )

        if self.adaptor_cls is not None:
            self.adaptors = nn.ModuleList(
                [
                    self.adaptor_cls(
                        dim=self.dim,
                        bottleneck=adaptor_bottleneck,
                        dropout=dropout,
                        activation=adaptor_activation,
                        **adaptor_kwargs,
                    )
                    for _ in range(self.num_targets)
                ]
            )
        else:
            self.adaptors = None

        self.heads = nn.ModuleList(
            [
                nn.Linear(self.dim, 1)
                for _ in range(self.num_targets)
            ]
        )

        self._initialize_heads()

    def _initialize_heads(self) -> None:
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def _parse_task_index(
        self,
        task: Union[int, str],
    ) -> int:
        """
        Convert an integer task index or a name such as ``task_0``
        into a valid task index.
        """
        if isinstance(task, bool):
            raise TypeError(
                "task must be an integer index or a string such as "
                "'task_0', not a boolean."
            )

        if isinstance(task, int):
            task_index = task

        elif isinstance(task, str):
            if not task.startswith("task_"):
                raise ValueError(
                    f"Invalid task name '{task}'. "
                    "Expected a name such as 'task_0'."
                )

            index_string = task.removeprefix("task_")

            try:
                task_index = int(index_string)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid task name '{task}'. "
                    "The part after 'task_' must be an integer."
                ) from exc

        else:
            raise TypeError(
                "task must be an integer index or a string such as "
                f"'task_0', but got {type(task).__name__}."
            )

        if not 0 <= task_index < self.num_targets:
            raise IndexError(
                f"Task index {task_index} is out of range. "
                f"Expected an index between 0 and "
                f"{self.num_targets - 1}."
            )

        return task_index

    def _validate_features(
        self,
        task_features: torch.Tensor,
    ) -> None:
        """
        Validate the output of a task-specific encoder.
        """
        if not isinstance(task_features, torch.Tensor):
            raise TypeError(
                "encoder must return a torch.Tensor, "
                f"but returned {type(task_features).__name__}."
            )

        if task_features.ndim == 0:
            raise ValueError(
                "Encoder output must have at least one dimension."
            )

        if task_features.shape[-1] != self.dim:
            raise ValueError(
                "The last dimension of the encoder output must "
                f"be {self.dim}, but got {task_features.shape[-1]}."
            )

    def encode(
        self,
        x: torch.Tensor,
        task: Union[int, str],
    ) -> torch.Tensor:
        """
        Compute one task-specific representation.
        """
        task_index = self._parse_task_index(task)

        task_features = self.encoders[task_index](x)
        self._validate_features(task_features)

        return task_features

    def forward_task(
        self,
        x: torch.Tensor,
        task: Union[int, str],
    ) -> torch.Tensor:
        """
        Run one task-specific encoder, optional adaptor, and head.
        """
        task_index = self._parse_task_index(task)

        task_features = self.encode(x, task_index)

        if self.adaptors is not None:
            task_features = self.adaptors[task_index](task_features)

        output = self.heads[task_index](task_features)

        return output

    def forward(
        self,
        x: torch.Tensor,
        task: Union[int, str],
    ) -> torch.Tensor:
        return self.forward_task(x, task)

    def forward_all(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Return predictions for every task.

        This assumes that the same input tensor can be processed by every
        task-specific encoder.
        """
        outputs: Dict[str, torch.Tensor] = {}

        for task_index in range(self.num_targets):
            outputs[f"task_{task_index}"] = self.forward_task(
                x=x,
                task=task_index,
            )

        return outputs

    def encoder_regularization_loss(
        self,
        reduction: str = "mean",
        normalize_by_numel: bool = True,
    ) -> torch.Tensor:
        """
        Compute pairwise L2 regularization between task-specific encoders.

        For encoders E_i and E_j, the regularization is approximately:

            sum over i < j of ||theta_i - theta_j||_2^2

        Parameters
        ----------
        reduction:
            ``"mean"`` averages over encoder pairs.
            ``"sum"`` sums over encoder pairs.

        normalize_by_numel:
            If True, divide each parameter difference by the number of
            elements in that parameter tensor. This prevents large layers from
            dominating the regularization loss.

        Returns
        -------
        torch.Tensor
            Scalar regularization loss.
        """
        if reduction not in {"mean", "sum"}:
            raise ValueError(
                "reduction must be either 'mean' or 'sum', "
                f"but got '{reduction}'."
            )

        if self.num_targets < 2:
            reference_parameter = next(self.encoders[0].parameters())
            return reference_parameter.new_zeros(())

        pair_losses = []

        for encoder_i, encoder_j in combinations(self.encoders, 2):
            params_i = dict(encoder_i.named_parameters())
            params_j = dict(encoder_j.named_parameters())

            if params_i.keys() != params_j.keys():
                raise ValueError(
                    "All encoders must have the same parameter names "
                    "and architecture for parameter-level soft sharing."
                )

            pair_loss = None

            for parameter_name in params_i:
                parameter_i = params_i[parameter_name]
                parameter_j = params_j[parameter_name]

                if parameter_i.shape != parameter_j.shape:
                    raise ValueError(
                        f"Parameter '{parameter_name}' has inconsistent "
                        f"shapes: {tuple(parameter_i.shape)} and "
                        f"{tuple(parameter_j.shape)}."
                    )

                # Compute the L2 distance between corresponding parameters：sum((theta_i - theta_j)^2)
                parameter_loss = torch.sum(
                    (parameter_i - parameter_j) ** 2
                )

                if normalize_by_numel:
                    parameter_loss = (
                        parameter_loss / parameter_i.numel()
                    )

                if pair_loss is None:
                    pair_loss = parameter_loss
                else:
                    pair_loss = pair_loss + parameter_loss

            if pair_loss is not None:
                pair_losses.append(pair_loss)

        if not pair_losses:
            reference_parameter = next(self.encoders[0].parameters())
            return reference_parameter.new_zeros(())

        regularization_loss = torch.stack(pair_losses)

        if reduction == "mean":
            return regularization_loss.mean()

        return regularization_loss.sum()


if __name__ == "__main__":
    # Example usage
    from torch import nn

    encoder_cls = GraphormerGraphEncoder
    adaptor_cls = ResidualAdaptor

    num_tasks = 3
    input_dim = 10
    model = SoftSharingMTL(
        num_targets=num_tasks,
        encoder_cls=encoder_cls,
        dim=input_dim,
        adaptor_cls=adaptor_cls,
    )

    # Create a dummy input tensor
    x = torch.randn(5, input_dim)

    # Forward pass for all tasks
    outputs = model.forward_all(x)
    for task_name, output in outputs.items():
        print(f"{task_name}: {output.shape}")

    # Compute regularization loss
    reg_loss = model.encoder_regularization_loss()
    print(f"Regularization loss: {reg_loss.item()}")