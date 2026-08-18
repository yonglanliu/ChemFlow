


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class WorkerShard:
    worker_id: int
    indices: list[int]


class DataDistributor:
    def __init__(self, num_workers: int, drop_last: bool = False):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        self.num_workers = num_workers
        self.drop_last = drop_last

    def distribute(self, dataset: Sequence[Any]) -> list[WorkerShard]:
        dataset_size = len(dataset)

        if dataset_size == 0:
            return []

        indices = list(range(dataset_size))

        if self.drop_last:
            usable_size = (dataset_size // self.num_workers) * self.num_workers

            indices = indices[:usable_size]

        shards = [
            WorkerShard(
                worker_id=worker_id,
                indices=indices[worker_id::self.num_workers],
            )
            for worker_id in range(self.num_workers)
        ]

        return shards