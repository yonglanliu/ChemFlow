from __future__ import annotations

from concurrent.futures import (
    ProcessPoolExecutor,
    Future,
    as_completed,
)
from pathlib import Path
from typing import Any, Callable


class WorkerExecutor:
    """
    Execute the same worker function over multiple shard files
    using multiple processes.
    """

    def __init__(self, num_workers: int, fail_fast: bool = True) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be greater than 0.")

        self.num_workers = num_workers
        self.fail_fast = fail_fast

    def run(self, worker_fn: Callable[[Path, Path], Any], inputs: list[Path], outdir: Path, query_name: str) -> list[Any]:
        """
        Run worker_fn on all shard files.

        worker_fn must have the signature:

            worker_fn(
                shard_path: Path,
                output_path: Path,
            )

        Parameters
        ----------
        worker_fn
            Function executed for each shard.

        inputs
            List of shard paths.
        outdir
            Directory where worker output files will be written.
        Returns
        -------
        list[Any]
            Worker results in the same order as the input shards.
        """

        if not inputs:
            return []

        futures: dict[Future, int] = {}
        results: list[Any | None] = [None for _ in inputs]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:

            for shard_id, shard_path in enumerate(inputs):
                shard_path = Path(shard_path)
                output_path = self._build_output_path(shard_path=shard_path, query_name=query_name, outdir=outdir)
                future = executor.submit(worker_fn, shard_path, output_path)
                futures[future] = shard_id

            for future in as_completed(futures):
                shard_id = futures[future]

                try:
                    result = future.result()

                except Exception as exc:
                    if self.fail_fast:
                        # Cancel jobs which have not started yet.
                        for pending_future in futures:
                            pending_future.cancel()

                        raise RuntimeError(
                            f"Worker failed while processing "
                            f"shard {shard_id}: "
                            f"{inputs[shard_id]}"
                        ) from exc

                    result = exc

                results[shard_id] = result

        return results

    def _build_output_path(self, shard_path: Path, query_name: str, outdir: Path) -> Path:
        """
        Build output path for one worker result.

        The default output format is CSV.
        """

        return (outdir / f"{shard_path.stem}_for_{query_name}_out.csv")