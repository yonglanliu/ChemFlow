from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class DataBaseConfig:
    input_path: Path
    output_path: Path
    output_file_prefix: str
    structure_column: str
    name_column: str | None = None
    data_size: int | None = None


@dataclass(kw_only=True)
class DataPartitionConfig(DataBaseConfig):
    num_shards: int = 1


@dataclass(kw_only=True)
class DataDistributionConfig:
    num_workers: int = 1
    drop_last: bool = False


@dataclass(kw_only=True)
class DataCollectionConfig:
    sort_by: str | None = None
    descending: bool = True
    top_k_ratio: float | None = 0.1


@dataclass(kw_only=True)
class DataPipelineConfig(
    DataPartitionConfig,
    DataDistributionConfig,
    DataCollectionConfig,
):
    pass