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

import copy
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from src.deep_learning.graphormer.modules.graphormer_encoder import GraphormerGraphEncoder
from itertools import combinations



import torch
from torch import nn
import torch.nn.functional as F

from typing import Any, Callable, Dict, Optional, Union


# ============================================================
# Residual Adaptor
# ============================================================

class ResidualAdaptor(nn.Module):
    """
    Lightweight task-specific residual adaptor.

    The adaptor learns a small task-specific delta:

        z -> down -> activation -> dropout -> up -> delta

    and adds it back to the input:

        output = LayerNorm(z + delta)

    Optionally, a learnable scalar gate can control the
    contribution of the residual delta.

    Parameters
    ----------
    dim:
        Input and output feature dimension.

    bottleneck:
        Hidden bottleneck dimension of the adaptor.

    dropout:
        Dropout probability.

    activation:
        Activation function applied after the down projection.

    gate:
        Whether to use a learnable scalar gate.

    gate_fn:
        Gating function. Supported values are:
        - "tanh"
        - "sigmoid"
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int = 32,
        dropout: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        gate: bool = False,
        gate_fn: str = "tanh", # "sigmoid" or "tanh"
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be positive, but got {dim}.")

        if bottleneck <= 0:
            raise ValueError(f"bottleneck must be positive, but got {bottleneck}.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), but got {dropout}.")

        if gate_fn not in {"tanh", "sigmoid"}:
            raise ValueError(f"Unsupported gate function '{gate_fn}'. Expected 'tanh' or 'sigmoid'.")

        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.down = nn.Linear(self.dim, self.bottleneck)
        self.up = nn.Linear(self.bottleneck, self.dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(self.dim)

        self.activation = activation

        self.gate = bool(gate)
        self.gate_fn = gate_fn

        if self.gate:
            if self.gate_fn == "tanh":
                # tanh(0) = 0
                # Residual branch initially contributes nothing.
                initial_alpha = 0.0

            else:
                # sigmoid(-5) ~= 0.0067
                # Starts close to zero rather than sigmoid(0)=0.5.
                initial_alpha = -5.0

            self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Initialize adaptor parameters.
        """

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)

        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _compute_gate(self) -> torch.Tensor:
        """
        Compute the learnable scalar gate.
        """

        if not self.gate:
            raise RuntimeError("_compute_gate() was called while gating is disabled.")

        if self.gate_fn == "tanh":
            return torch.tanh(self.alpha)

        if self.gate_fn == "sigmoid":
            return torch.sigmoid(self.alpha)

        raise RuntimeError(f"Unsupported gate function: {self.gate_fn}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual adaptor.

        Parameters
        ----------
        z:
            Input tensor with shape (..., dim).

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as z.
        """

        if z.shape[-1] != self.dim:
            raise ValueError(f"The last dimension of the adaptor input must be {self.dim}, but got {z.shape[-1]}.")

        out = self.down(z)

        out = self.activation(out)

        out = self.dropout(out)

        delta = self.up(out)

        if self.gate:
            gate = self._compute_gate()
            delta = gate * delta

        return self.ln(z + delta)


# ============================================================
# Hard Parameter Sharing MTL
# ============================================================

class HardSharingMTL(nn.Module):
    """
    Hard parameter sharing multi-task learning model.

    All tasks share the same encoder:

    One task to one specific adaptor and head:

                    x
                    |
                    v
                shared encoder
                    |
        +---------------------------+
        |             |             |
        v             v             v
    adaptor_0     adaptor_1     adaptor_2
        |             |             |
        v             v             v
      head_0        head_1        head_2
        |             |             |
        v             v             v
      task_0        task_1        task_2


    Parameters
    ----------
    num_targets:
        Number of prediction tasks.

    shared_encoder:
        Already-initialized shared encoder.

        The encoder may return either:

            shared_features

        or:

            (something_else, shared_features)

    dim:
        Feature dimension produced by the shared encoder.

    adaptor_bottleneck_dim:
        Bottleneck dimension of each task-specific adaptor.

    adaptor_dropout:
        Dropout probability used by the adaptors.

    adaptor_activation:
        Activation function used by each adaptor.

    adaptor_kwargs:
        Additional keyword arguments passed to ResidualAdaptor.

        Example:

            {
                "gate": True,
                "gate_fn": "tanh",
            }
    """

    def __init__(
        self,
        num_targets: int,
        shared_encoder: nn.Module,
        dim: int = 512,
        adaptor_bottleneck_dim: int = 32,
        adaptor_dropout: float = 0.1,
        adaptor_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        adaptor_kwargs: Optional[Dict[str, Any]] = None,
        num_adapters: Optional[int] = None,
        task_groups: Optional[list[list[int]]] = None,
    ) -> None:
        super().__init__()

        if num_targets <= 0:
            raise ValueError(f"num_targets must be positive, but got {num_targets}.")

        if dim <= 0:
            raise ValueError(f"dim must be positive, but got {dim}.")

        if adaptor_bottleneck_dim <= 0:
            raise ValueError(f"adaptor_bottleneck_dim must be positive, but got {adaptor_bottleneck_dim}.")

        if not isinstance(shared_encoder, nn.Module):
            raise TypeError(f"shared_encoder must be an instance of nn.Module, but got {type(shared_encoder).__name__}.")

        self.num_targets = int(num_targets)
        self.dim = int(dim)
        self.task_groups = self._resolve_task_groups(num_targets, num_adapters, task_groups)
        self.num_adapters = len(self.task_groups)
        self.task_to_group = {
            task_index: group_index
            for group_index, group in enumerate(self.task_groups)
            for task_index in group
        }

        print(
            "Task grouping for adaptor sharing: "
            + "; ".join(
                f"adaptor_{group_index} -> tasks {group}"
                for group_index, group in enumerate(self.task_groups)
            )
        )

        # One encoder is shared across all tasks.
        self.shared_encoder = shared_encoder

        adaptor_kwargs = dict(adaptor_kwargs or {})

        # ----------------------------------------------------
        # Group-specific adaptors
        # ----------------------------------------------------

        self.adaptors = nn.ModuleList(
            [
                ResidualAdaptor(
                    dim=self.dim,
                    bottleneck=adaptor_bottleneck_dim,
                    dropout=adaptor_dropout,
                    activation=adaptor_activation,
                    **adaptor_kwargs,
                )
                for _ in range(self.num_adapters)
            ]
        )

        # ----------------------------------------------------
        # Task-specific prediction heads within each group
        # ----------------------------------------------------

        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.dim, 1)
                        for _ in range(len(group))
                    ]
                )
                for group in self.task_groups
            ]
        )

        self._initialize_heads()

    def _resolve_task_groups(
        self,
        num_targets: int,
        num_adapters: Optional[int] = None,
        task_groups: Optional[list[list[int]]] = None,
    ) -> list[list[int]]:
        """Build the grouping layout for tasks."""

        if task_groups is not None:
            groups = [list(map(int, group)) for group in task_groups]
        elif num_adapters is not None and num_adapters > 0:
            base_size, extra = divmod(num_targets, num_adapters)
            groups = []
            start = 0
            for group_index in range(num_adapters):
                group_size = base_size + (1 if group_index < extra else 0)
                stop = start + group_size
                if stop > num_targets:
                    stop = num_targets
                groups.append(list(range(start, stop)))
                start = stop
            if start != num_targets:
                groups.append(list(range(start, num_targets)))
        else:
            groups = [[task_index] for task_index in range(num_targets)]

        flat_groups = [task_index for group in groups for task_index in group]

        if sorted(flat_groups) != list(range(num_targets)):
            raise ValueError(
                "task_groups must cover every task index exactly once. "
                f"Received task_groups={groups} for num_targets={num_targets}."
            )

        return groups

    # ========================================================
    # Initialization
    # ========================================================

    def _initialize_heads(self) -> None:
        """
        Initialize task-specific prediction heads.
        """
        for group_heads in self.heads:
            if isinstance(group_heads, nn.ModuleList):
                for head in group_heads:
                    nn.init.xavier_uniform_(head.weight)
                    nn.init.zeros_(head.bias)
            else:
                nn.init.xavier_uniform_(group_heads.weight)
                nn.init.zeros_(group_heads.bias)

    # ========================================================
    # Task handling
    # ========================================================

    def _parse_task_index(self, task: Union[int, str]) -> int:
        """
        Convert a task specification into an integer index.

        Examples
        --------
        0
            -> 0

        "task_0"
            -> 0
        """

        if isinstance(task, int):
            task_index = task

        elif isinstance(task, str):

            if not task.startswith( "task_"):
                raise ValueError(f"Invalid task name '{task}'. ""Expected something like 'task_0'.")

            index_string = (task.removeprefix("task_"))

            try:
                task_index = int(index_string)

            except ValueError as exc:
                raise ValueError(f"Invalid task name '{task}'. The part after 'task_' must be an integer.") from exc

        else:
            raise TypeError(
                "task must be either an integer "
                "or a string such as 'task_0', "
                f"but got {type(task).__name__}."
            )

        if not (0 <= task_index < self.num_targets):
            raise IndexError(
                f"Task index {task_index} is out of range. "
                "Expected an index between "
                f"0 and {self.num_targets - 1}."
            )

        return task_index

    # ========================================================
    # Shared encoder
    # ========================================================

    def encode(self, x: torch.Tensor, perturb: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute the shared representation.

        The shared encoder may return either:

            Tensor

        or:

            (something, Tensor)

        In the second case, the second element is interpreted
        as the shared feature representation.
        """

        encoder_output = (self.shared_encoder(x, perturb=perturb, **kwargs))

        # ----------------------------------------------------
        # Handle encoder outputs
        # ----------------------------------------------------
        if isinstance(encoder_output, (tuple, list)):
            if len(encoder_output) != 2:
                raise ValueError(
                    "Expected shared_encoder to return either "
                    "a Tensor or a tuple/list of length 2, "
                    f"but got {len(encoder_output)} outputs."
                )

            _, shared_features = (encoder_output)  # graphormer has two outputs, the second one is the graph representation

        else:
            shared_features = (encoder_output)

        # ----------------------------------------------------
        # Validate output
        # ----------------------------------------------------
        if not isinstance(shared_features, torch.Tensor):
            raise TypeError(
                "shared_encoder must produce a torch.Tensor, "
                "but produced "
                f"{type(shared_features).__name__}."
            )

        if shared_features.shape[-1] != self.dim:
            raise ValueError(
                "The last dimension of shared encoder output "
                f"must be {self.dim}, "
                f"but got {shared_features.shape[-1]}."
            )

        return shared_features

    # ========================================================
    # One task
    # ========================================================

    def forward_task(self, shared_features: torch.Tensor, task: Union[int, str]) -> torch.Tensor:
        """
        Apply one task-specific adaptor and head.

        Parameters
        ----------
        shared_features:
            Shared encoder representation.

        task:
            Task index or task name such as "task_0".
        """

        task_index = (self._parse_task_index(task))
        group_index = self.task_to_group[task_index]
        local_index = self.task_groups[group_index].index(task_index)

        task_features = (self.adaptors[group_index](shared_features))

        output = (self.heads[group_index][local_index](task_features))

        return output

    # ========================================================
    # Default forward: one task
    # ========================================================

    def forward(self, x: torch.Tensor, task: Union[int, str], perturb: Optional[torch.Tensor] = None, **kwargs,) -> torch.Tensor:
        """
        Predict one task.

        Example
        -------
        output = model(
            x,
            task=0,
        )
        """

        shared_features = (self.encode(x, perturb=perturb, **kwargs,))

        return self.forward_task(shared_features, task)

    # ========================================================
    # All tasks
    # ========================================================

    def forward_all(self, x: torch.Tensor, perturb: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor,]:

        """
        Predict all tasks using one shared encoder pass.

        Example
        -------
        outputs = model.forward_all(x)

        outputs["task_0"]
        outputs["task_1"]
        ...
        """

        shared_features = (self.encode(x, perturb=perturb, **kwargs))

        outputs = {}

        for group_index, group_tasks in enumerate(self.task_groups):
            task_features = (self.adaptors[group_index](shared_features))

            for local_index, task_index in enumerate(group_tasks):
                output = (self.heads[group_index][local_index](task_features))
                outputs[f"task_{task_index}"] = output

        return outputs


# ============================================================
# Soft Parameter Sharing MTL
# ============================================================

class SoftSharingMTL(nn.Module):
    """
    Soft parameter sharing multi-task learning model.

    Each task has its own independent encoder, adaptor,
    and prediction head.

    All encoders start from the same initialized encoder
    parameters through deepcopy:

        encoder_0
        encoder_1
        encoder_2
        ...

    During training, parameter-level L2 regularization
    encourages the task-specific encoders to remain similar:

        L_total = L_task + lambda_soft * L_regularization

    Architecture
    ------------

                x
        ________|________
       |        |        |
       v        v        v
    encoder0 encoder1 encoder2
       |        |        |
       v        v        v
    adaptor0 adaptor1 adaptor2
       |        |        |
       v        v        v
     head0    head1    head2


    Parameters
    ----------
    num_targets:
        Number of prediction tasks.

    shared_encoder:
        An already initialized encoder instance.

        Independent copies of this encoder are created for
        every task using copy.deepcopy().

    dim:
        Dimension of encoder output.

    adaptor_bottleneck_dim:
        Bottleneck dimension of each task-specific adaptor.

    adaptor_dropout:
        Dropout probability used by adaptors.

    adaptor_activation:
        Activation function used by adaptors.

    adaptor_kwargs:
        Additional arguments passed to ResidualAdaptor.

        Example:

            {
                "gate": True,
                "gate_fn": "tanh",
            }
    """

    def __init__(
        self,
        num_targets: int,
        shared_encoder: nn.Module,
        dim: int = 512,
        adaptor_bottleneck_dim: int = 32,
        adaptor_dropout: float = 0.1,
        adaptor_activation: Callable[
            [torch.Tensor],
            torch.Tensor,
        ] = F.relu,
        adaptor_kwargs: Optional[
            Dict[str, Any]
        ] = None,
        num_adapters: Optional[int] = None,
        task_groups: Optional[list[list[int]]] = None,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if num_targets <= 0:
            raise ValueError(
                "num_targets must be positive, "
                f"but got {num_targets}."
            )

        if dim <= 0:
            raise ValueError(
                f"dim must be positive, but got {dim}."
            )

        if adaptor_bottleneck_dim <= 0:
            raise ValueError(
                "adaptor_bottleneck_dim must be positive, "
                f"but got {adaptor_bottleneck_dim}."
            )

        if not isinstance(
            shared_encoder,
            nn.Module,
        ):
            raise TypeError(
                "shared_encoder must be an instance of nn.Module, "
                f"but got {type(shared_encoder).__name__}."
            )

        self.num_targets = int(
            num_targets
        )

        self.dim = int(
            dim
        )

        self.task_groups = self._resolve_task_groups(num_targets, num_adapters, task_groups)
        self.num_adapters = len(self.task_groups)
        self.task_to_group = {
            task_index: group_index
            for group_index, group in enumerate(self.task_groups)
            for task_index in group
        }

        adaptor_kwargs = dict(
            adaptor_kwargs or {}
        )

        # ----------------------------------------------------
        # Independent task-specific encoders
        # ----------------------------------------------------
        #
        # Important:
        #
        # deepcopy means all encoders START with identical
        # parameters but subsequently train independently.
        #
        # This is different from hard sharing.
        # ----------------------------------------------------

        self.encoders = nn.ModuleList(
            [
                copy.deepcopy(
                    shared_encoder
                )
                for _ in range(
                    self.num_targets
                )
            ]
        )

        # ----------------------------------------------------
        # Group-specific adaptors
        # ----------------------------------------------------

        self.adaptors = nn.ModuleList(
            [
                ResidualAdaptor(
                    dim=self.dim,
                    bottleneck=adaptor_bottleneck_dim,
                    dropout=adaptor_dropout,
                    activation=adaptor_activation,
                    **adaptor_kwargs,
                )
                for _ in range(
                    self.num_adapters
                )
            ]
        )

        # ----------------------------------------------------
        # Task-specific prediction heads within each group
        # ----------------------------------------------------

        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(
                            self.dim,
                            1,
                        )
                        for _ in range(
                            len(group)
                        )
                    ]
                )
                for group in self.task_groups
            ]
        )

        self._initialize_heads()

    def _resolve_task_groups(
        self,
        num_targets: int,
        num_adapters: Optional[int] = None,
        task_groups: Optional[list[list[int]]] = None,
    ) -> list[list[int]]:
        """Build the grouping layout for tasks."""

        if task_groups is not None:
            groups = [list(map(int, group)) for group in task_groups]
        elif num_adapters is not None and num_adapters > 0:
            base_size, extra = divmod(num_targets, num_adapters)
            groups = []
            start = 0
            for group_index in range(num_adapters):
                group_size = base_size + (1 if group_index < extra else 0)
                stop = start + group_size
                if stop > num_targets:
                    stop = num_targets
                groups.append(list(range(start, stop)))
                start = stop
            if start != num_targets:
                groups.append(list(range(start, num_targets)))
        else:
            groups = [[task_index] for task_index in range(num_targets)]

        flat_groups = [task_index for group in groups for task_index in group]
        if sorted(flat_groups) != list(range(num_targets)):
            raise ValueError(
                "task_groups must cover every task index exactly once. "
                f"Received task_groups={groups} for num_targets={num_targets}."
            )

        return groups

    # ========================================================
    # Head initialization
    # ========================================================

    def _initialize_heads(
        self,
    ) -> None:

        for group_heads in self.heads:
            if isinstance(group_heads, nn.ModuleList):
                for head in group_heads:
                    nn.init.xavier_uniform_(
                        head.weight
                    )

                    nn.init.zeros_(
                        head.bias
                    )
            else:
                nn.init.xavier_uniform_(
                    group_heads.weight
                )

                nn.init.zeros_(
                    group_heads.bias
                )

    # ========================================================
    # Task parsing
    # ========================================================

    def _parse_task_index(
        self,
        task: Union[int, str],
    ) -> int:
        """
        Convert:

            0
            "task_0"

        into integer task index 0.
        """

        # bool is technically a subclass of int in Python,
        # so explicitly reject it.
        if isinstance(
            task,
            bool,
        ):
            raise TypeError(
                "task must be an integer index or a string "
                "such as 'task_0', not a boolean."
            )

        if isinstance(
            task,
            int,
        ):
            task_index = task

        elif isinstance(
            task,
            str,
        ):

            if not task.startswith(
                "task_"
            ):
                raise ValueError(
                    f"Invalid task name '{task}'. "
                    "Expected something like 'task_0'."
                )

            index_string = (
                task.removeprefix(
                    "task_"
                )
            )

            try:
                task_index = int(
                    index_string
                )

            except ValueError as exc:
                raise ValueError(
                    f"Invalid task name '{task}'. "
                    "The part after 'task_' must "
                    "be an integer."
                ) from exc

        else:
            raise TypeError(
                "task must be an integer index or a string "
                "such as 'task_0', "
                f"but got {type(task).__name__}."
            )

        if not (
            0
            <= task_index
            < self.num_targets
        ):
            raise IndexError(
                f"Task index {task_index} is out of range. "
                "Expected an index between "
                f"0 and {self.num_targets - 1}."
            )

        return task_index

    # ========================================================
    # Encoder output validation
    # ========================================================

    def _validate_features(
        self,
        task_features: torch.Tensor,
    ) -> None:

        if not isinstance(
            task_features,
            torch.Tensor,
        ):
            raise TypeError(
                "encoder must produce a torch.Tensor, "
                "but produced "
                f"{type(task_features).__name__}."
            )

        if task_features.ndim == 0:
            raise ValueError(
                "Encoder output must have at least "
                "one dimension."
            )

        if task_features.shape[-1] != self.dim:
            raise ValueError(
                "The last dimension of encoder output "
                f"must be {self.dim}, "
                f"but got {task_features.shape[-1]}."
            )

    # ========================================================
    # Encode one task
    # ========================================================

    def encode(
        self,
        x: torch.Tensor,
        task: Union[int, str],
        perturb: Optional[
            torch.Tensor
        ] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute one task-specific encoder representation.
        """

        task_index = (
            self._parse_task_index(
                task
            )
        )

        encoder_output = (
            self.encoders[
                task_index
            ](
                x,
                perturb=perturb,
                **kwargs,
            )
        )

        # ----------------------------------------------------
        # Support encoders returning:
        #
        # Tensor
        #
        # OR
        #
        # (inner_states, graph_representation)
        # ----------------------------------------------------

        if isinstance(
            encoder_output,
            (tuple, list),
        ):

            if len(encoder_output) != 2:
                raise ValueError(
                    "Expected encoder to return either "
                    "a Tensor or a tuple/list of length 2, "
                    f"but got {len(encoder_output)} outputs."
                )

            _, task_features = (
                encoder_output
            )

        else:
            task_features = (
                encoder_output
            )

        self._validate_features(
            task_features
        )

        return task_features

    # ========================================================
    # One task forward
    # ========================================================

    def forward_task(
        self,
        x: torch.Tensor,
        task: Union[int, str],
        perturb: Optional[
            torch.Tensor
        ] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run one task-specific encoder, adaptor, and head.
        """

        task_index = (
            self._parse_task_index(
                task
            )
        )

        group_index = self.task_to_group[task_index]
        local_index = self.task_groups[group_index].index(task_index)

        task_features = (
            self.encode(
                x=x,
                task=task_index,
                perturb=perturb,
                **kwargs,
            )
        )

        task_features = (
            self.adaptors[
                group_index
            ](
                task_features
            )
        )

        output = (
            self.heads[
                group_index
            ][
                local_index
            ](
                task_features
            )
        )

        return output

    # ========================================================
    # Default forward: one task
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        task: Union[int, str],
        perturb: Optional[
            torch.Tensor
        ] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict one task.

        Example
        -------
        output = model(
            x,
            task=0,
        )
        """

        return self.forward_task(
            x=x,
            task=task,
            perturb=perturb,
            **kwargs,
        )

    # ========================================================
    # Forward all tasks
    # ========================================================

    def forward_all(
        self,
        x: torch.Tensor,
        perturb: Optional[
            torch.Tensor
        ] = None,
        **kwargs,
    ) -> Dict[
        str,
        torch.Tensor,
    ]:
        """
        Predict every task.

        Unlike HardSharingMTL, each task runs its own encoder,
        so this performs num_targets encoder forward passes.
        """

        outputs: Dict[
            str,
            torch.Tensor,
        ] = {}

        for task_index in range(
            self.num_targets
        ):

            outputs[
                f"task_{task_index}"
            ] = self.forward_task(
                x=x,
                task=task_index,
                perturb=perturb,
                **kwargs,
            )

        return outputs

    # ========================================================
    # Soft-sharing regularization
    # ========================================================

    def encoder_regularization_loss(
        self,
        reduction: str = "mean",
        normalize_by_numel: bool = True,
    ) -> torch.Tensor:
        """
        Compute pairwise L2 regularization between
        task-specific encoder parameters.

        For encoder pair (i, j):

            L_ij = sum_k ||theta_i,k - theta_j,k||^2

        where k indexes corresponding parameters.

        Parameters
        ----------
        reduction:
            "mean":
                Average regularization over encoder pairs.

            "sum":
                Sum regularization over encoder pairs.

        normalize_by_numel:
            If True, each parameter tensor's contribution
            is divided by its number of elements.

            This prevents very large parameter matrices from
            dominating the regularization loss.

        Returns
        -------
        torch.Tensor
            Scalar regularization loss.
        """

        if reduction not in {
            "mean",
            "sum",
        }:
            raise ValueError(
                "reduction must be either 'mean' or 'sum', "
                f"but got '{reduction}'."
            )

        # ----------------------------------------------------
        # No pair exists with only one task
        # ----------------------------------------------------

        if self.num_targets < 2:

            reference_parameter = next(
                self.encoders[
                    0
                ].parameters()
            )

            return (
                reference_parameter
                .new_zeros(())
            )

        pair_losses = []

        # ----------------------------------------------------
        # Compare every encoder pair
        #
        # Example with three encoders:
        #
        # (0, 1)
        # (0, 2)
        # (1, 2)
        # ----------------------------------------------------

        for (
            encoder_i,
            encoder_j,
        ) in combinations(
            self.encoders,
            2,
        ):

            params_i = dict(
                encoder_i.named_parameters()
            )

            params_j = dict(
                encoder_j.named_parameters()
            )

            # All task encoders should have identical architecture.
            if (
                params_i.keys()
                != params_j.keys()
            ):
                raise ValueError(
                    "All encoders must have the same "
                    "parameter names and architecture "
                    "for parameter-level soft sharing."
                )

            pair_loss = None

            # ------------------------------------------------
            # Compare corresponding parameters
            # ------------------------------------------------

            for parameter_name in params_i:

                parameter_i = (
                    params_i[
                        parameter_name
                    ]
                )

                parameter_j = (
                    params_j[
                        parameter_name
                    ]
                )

                if (
                    parameter_i.shape
                    != parameter_j.shape
                ):
                    raise ValueError(
                        f"Parameter '{parameter_name}' "
                        "has inconsistent shapes: "
                        f"{tuple(parameter_i.shape)} and "
                        f"{tuple(parameter_j.shape)}."
                    )

                # --------------------------------------------
                # Squared L2 parameter distance
                #
                # sum((theta_i - theta_j)^2)
                # --------------------------------------------

                parameter_loss = torch.sum(
                    (
                        parameter_i
                        - parameter_j
                    ) ** 2
                )

                if normalize_by_numel:

                    parameter_loss = (
                        parameter_loss
                        / parameter_i.numel()
                    )

                if pair_loss is None:

                    pair_loss = (
                        parameter_loss
                    )

                else:

                    pair_loss = (
                        pair_loss
                        + parameter_loss
                    )

            if pair_loss is not None:
                pair_losses.append(
                    pair_loss
                )

        # ----------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------

        if not pair_losses:

            reference_parameter = next(
                self.encoders[
                    0
                ].parameters()
            )

            return (
                reference_parameter
                .new_zeros(())
            )

        regularization_loss = (
            torch.stack(
                pair_losses
            )
        )

        if reduction == "mean":
            return (
                regularization_loss.mean()
            )

        return regularization_loss.sum()