"""Active-learning virtual screening for ChemFlow.

A Deep-Docking-style loop: a Chemprop surrogate scores the whole (sharded) library
each round, and the Glide oracle only docks the batch selected by acquisition.
See :mod:`chemflow.active_learning.workflow`.
"""

from chemflow.active_learning.acquisition import (  # noqa: F401
    Candidate,
    enrichment_factor,
    select_batch,
    top_hits,
)
from chemflow.active_learning.workflow import load_config, run_loop  # noqa: F401
