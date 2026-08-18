from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from src.chemflow.data.config import DataCollectionConfig

import logging

logger = logging.getLogger(__name__)



class ResultCollector:

    """
    Collect, merge, rank, and optionally select a top fraction
    of predictions from multiple workers.
    """

    def __init__(self, sort_by: str | None=None, descending: bool = True, top_k_ratio: float = 0.1) -> None:

        self.sort_by = sort_by
        self.descending = descending
        self.top_k_ratio = top_k_ratio

        if self.top_k_ratio is not None:
            if not 0 < self.top_k_ratio <= 1:
                logger.error("top_k_ratio must be in the range (0, 1].")
                raise ValueError("top_k_ratio must be in the range (0, 1].")

    def collect(self, results: list[Any]) -> pd.DataFrame:
        
        """
        Collect results returned by workers.

        Supported worker outputs
        ------------------------
        - pandas DataFrame
        - CSV file Path
        - Parquet file Path
        """

        frames: list[pd.DataFrame] = []

        for result in results:
            if result is None:
                continue

            frame = self._load_result(result)
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)

        merged = self._sort_results(merged)
        merged = self._select_top_k(merged)

        return merged.reset_index(drop=True)


    def _load_result(self, result: Any) -> pd.DataFrame:

        if isinstance(result, pd.DataFrame):
            return result

        if isinstance(result, Exception):
            raise RuntimeError(f"A worker failed: {result}")

        if isinstance(result, (str, Path)):
            path = Path(result)

            if not path.exists():
                raise FileNotFoundError(f"Worker result file not found: {path}")

            suffix = path.suffix.lower()

            if suffix == ".csv":
                return pd.read_csv(path)

            if suffix == ".parquet":
                return pd.read_parquet(path)

            raise ValueError(f"Unsupported result format: {suffix}")

        raise TypeError(f"Unsupported worker result type: {type(result).__name__}")

    def _sort_results(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        if self.sort_by is None:
            return dataframe

        if self.sort_by not in dataframe.columns:
            raise KeyError(
                f"Sort column '{self.sort_by}' "
                f"not found in result columns: "
                f"{list(dataframe.columns)}"
            )

        return dataframe.sort_values(by=self.sort_by, ascending=not self.descending)

    def _select_top_k(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        if self.top_k_ratio is None:
            return dataframe

        top_k = int(len(dataframe) * self.top_k_ratio)

        # Ensure at least one result is returned
        # when the dataframe is non-empty.
        top_k = max(1, top_k)

        return dataframe.head(top_k)
