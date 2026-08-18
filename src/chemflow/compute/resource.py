from dataclasses import dataclass
import math
import logging
logger = logging.getLogger(__name__)

@dataclass
class ComputeResource:
    num_workers: int
    cpus_per_worker: int = 1
    gpus_per_worker: int = 0
    memory_gb_per_worker: float | None = None

class ShardPlanner:
    def __init__(
        self,
        min_records_per_shard: int = 1_000,
        max_records_per_shard: int = 50_000,
        shards_per_worker: int = 2,
    ):
        self.min_records_per_shard = min_records_per_shard
        self.max_records_per_shard = max_records_per_shard
        self.shards_per_worker = shards_per_worker

    def plan(self, dataset_size: int, resource: ComputeResource) -> int:

        if dataset_size <= 0:
            logger.error(
                "dataset_size must be greater than 0. "
                "Returning 1 shard to avoid division by zero."
            )
            raise ValueError( "dataset_size must be greater than 0.")

        if resource.num_workers <= 0:
            logger.error(
                "num_workers must be greater than 0. "
                "Returning 1 shard to avoid division by zero."
            )
            raise ValueError("num_workers must be greater than 0.")

        # Start with multiple shards per worker for load balancing
        num_shards = (resource.num_workers * self.shards_per_worker)

        # Avoid shards that are excessively large
        required_shards = math.ceil(dataset_size / self.max_records_per_shard)

        num_shards = max(num_shards, required_shards)

        # Avoid creating lots of tiny shards
        max_reasonable_shards = max(1, dataset_size // self.min_records_per_shard)

        num_shards = min(num_shards, max_reasonable_shards)

        # Never create more shards than records
        return min(num_shards, dataset_size)