from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    shard_paths: list[Path]


def assign_shards_to_workers(shard_paths: list[Path], num_workers: int) -> list[WorkerAssignment]:
    """
    Assign shard files to workers using round-robin assignment.

    Example
    -------
    8 shards, 3 workers:

        worker 0 -> shard 0, shard 3, shard 6
        worker 1 -> shard 1, shard 4, shard 7
        worker 2 -> shard 2, shard 5
    """

    if num_workers < 1:
        raise ValueError("num_workers must be greater than 0.")

    assignments = [
        WorkerAssignment(
            worker_id=worker_id,
            shard_paths=[],
        )
        for worker_id in range(num_workers)
    ]

    # Dataclass is frozen, so construct mutable temporary lists.
    worker_shards: list[list[Path]] = [[] for _ in range(num_workers)]

    for shard_id, shard_path in enumerate(shard_paths):
        worker_id = shard_id % num_workers

        worker_shards[worker_id].append(Path(shard_path))

    return [
        WorkerAssignment(
            worker_id=worker_id,
            shard_paths=worker_shards[worker_id],
        )
        for worker_id in range(num_workers)
    ]