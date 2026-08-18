from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from rdkit import Chem
import csv
import logging

class DatasetPartitioner:
    """Split molecular datasets into a fixed number of physical shard files."""

    SUPPORTED_FORMATS = {
        ".sdf",
        ".smi",
        ".csv",
        ".parquet",
    }

    def __init__(self, input_path: Path | str, output_dir: Path | str, num_shards: int, output_file_prefix: str, logger: logging.Logger) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.num_shards = num_shards
        self.output_file_prefix = output_file_prefix 
        self.logger = logger
        self._validate()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate(self) -> None:
        if not self.input_path.exists():
            self.logger.error(f"Input file not found: {self.input_path}")
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        if not self.input_path.is_file():
            self.logger.error(f"Input path must be a file: {self.input_path}")
            raise ValueError(f"Input path must be a file: {self.input_path}")

        if self.num_shards <= 0:
            self.logger.error("num_shards must be greater than 0.")
            raise ValueError("num_shards must be greater than 0.")

        suffix = self.input_path.suffix.lower()

        if suffix not in self.SUPPORTED_FORMATS:
            self.logger.error(
                f"Unsupported file format '{suffix}'. "
                f"Supported formats are: "
                f"{', '.join(sorted(self.SUPPORTED_FORMATS))}."
            )
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                f"Supported formats are: "
                f"{', '.join(sorted(self.SUPPORTED_FORMATS))}."
            )

    def split(self) -> list[Path]:
        """Split the input dataset according to its file format."""

        suffix = self.input_path.suffix.lower()

        splitters = {
            ".sdf": self._split_sdf,
            ".smi": self._split_smi,
            ".csv": self._split_csv,
            ".parquet": self._split_parquet,
        }

        return splitters[suffix]()

    # ============================================================
    # SDF
    # ============================================================

    def _split_sdf(self) -> list[Path]:
        """Split an SDF file into approximately equal shards."""
        self.logger.info(f"Splitting SDF file: {self.input_path}")
        
        # First pass: count valid molecules
        supplier = Chem.ForwardSDMolSupplier(str(self.input_path))
        dataset_size = sum(mol is not None for mol in supplier)

        self.logger.info(f"Total valid molecules in SDF: {dataset_size}")

        if dataset_size == 0:
            return []

        shard_size = math.ceil(dataset_size / self.num_shards)
        self.logger.info(f"Splitting into {self.num_shards} shards, each with up to {shard_size} molecules.")

        # Second pass: actually write molecules
        supplier = Chem.ForwardSDMolSupplier(str(self.input_path))

        output_paths: list[Path] = []

        writer = None
        shard_id = 0
        count = 0

        try:
            for mol in supplier:
                if mol is None:
                    continue

                if count % shard_size == 0:
                    if writer is not None:
                        writer.close()

                    output_path = self._get_output_path(shard_id=shard_id, suffix=".sdf",)
                    output_paths.append(output_path)
                    writer = Chem.SDWriter(str(output_path))
                    shard_id += 1

                writer.write(mol)
                count += 1

        finally:
            if writer is not None:
                writer.close()

        return output_paths

    # ============================================================
    # SMILES
    # ============================================================

    def _split_smi(self) -> list[Path]:
        """Split a SMILES file into approximately equal shards. The output format is .csv"""
        self.logger.info(f"Splitting SMILES file: {self.input_path}")

        # First pass: count non-empty records
        with self.input_path.open("r", encoding="utf-8") as input_file:
            dataset_size = sum(1 for line in input_file if line.strip())
        self.logger.info(f"Total non-empty records in SMILES file: {dataset_size}")

        if dataset_size == 0:
            return []

        shard_size = math.ceil(dataset_size / self.num_shards)
        self.logger.info(f"Splitting into {self.num_shards} shards, each with up to {shard_size} records.")

        output_paths: list[Path] = []

        writer = None
        shard_id = 0
        count = 0

        try:
            with self.input_path.open("r", encoding="utf-8") as input_file:

                for line in input_file:
                    if not line.strip():
                        continue

                    if count % shard_size == 0:
                        if writer is not None:
                            writer.close()

                        output_path = self._get_output_path(shard_id=shard_id, suffix=".csv")
                        output_paths.append(output_path)
                        writer = output_path.open("w", newline="", encoding="utf-8")
                        csv_writer = csv.writer(writer)

                        # header
                        csv_writer.writerow(["Ligand ID", "SMILES"])

                        shard_id += 1

                    # First token = SMILES
                    # Everthing after it = compound name
                    parts = line.split(maxsplit=1)
                    smiles = parts[0]
                    ligand_id = (parts[1] if len(parts)>1 else f"ligand_{count}")

                    csv_writer.writerow([ligand_id, smiles])
                    count += 1

        finally:
            if writer is not None:
                writer.close()

        return output_paths

    # ============================================================
    # CSV
    # ============================================================

    def _split_csv(self) -> list[Path]:
        """Split a CSV file into approximately equal shards."""

        dataframe = pd.read_csv(self.input_path)
        dataset_size = len(dataframe)

        if dataset_size == 0:
            return []

        shard_size = math.ceil(dataset_size / self.num_shards)

        output_paths: list[Path] = []

        for shard_id, start in enumerate(range(0, dataset_size, shard_size)):
            end = min(start + shard_size, dataset_size)
            chunk = dataframe.iloc[start:end]
            output_path = self._get_output_path(shard_id=shard_id, suffix=".csv")
            chunk.to_csv(output_path, index=False)
            output_paths.append(output_path)

        return output_paths

    # ============================================================
    # Parquet
    # ============================================================

    def _split_parquet(self) -> list[Path]:
        """
        Split a Parquet dataset into approximately equal CSV shard files.

        Each output shard is written in CSV format so that
        downstream workers use a unified tabular interface.
        """
        self.logger.info(f"Splitting Parquet file: {self.input_path}")
        dataframe = pd.read_parquet(self.input_path)

        dataset_size = len(dataframe)
        self.logger.info(f"Total records in Parquet file: {dataset_size}")

        if dataset_size == 0:
            return []

        shard_size = math.ceil(dataset_size / self.num_shards)
        self.logger.info(f"Splitting into {self.num_shards} shards, each with up to {shard_size} records.")

        output_paths: list[Path] = []

        for shard_id, start in enumerate(range(0, dataset_size, shard_size)):

            end = min(start + shard_size, dataset_size)
            chunk = dataframe.iloc[start:end]
            output_path = self._get_output_path(shard_id=shard_id, suffix=".csv")
            chunk.to_csv(output_path, index=False)
            output_paths.append(output_path)

        return output_paths

    # ============================================================
    # Utilities
    # ============================================================
    def _get_output_path(self, shard_id: int, suffix: str) -> Path:
        self.logger.info(f"Generating output path for shard {shard_id} with suffix '{suffix}'")
        return (self.output_dir / f"{self.output_file_prefix}_{shard_id:04d}{suffix}")