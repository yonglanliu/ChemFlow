from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    log_dir: Path | str,
    log_name: str = "chemflow.log",
    level: int = logging.INFO,
) -> logging.Logger:

    log_dir = Path(log_dir)
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = log_dir / log_name

    logger = logging.getLogger()

    logger.setLevel(level)

    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # log file
    file_handler = logging.FileHandler(
        log_file,
        mode="w",
    )
    file_handler.setFormatter(formatter)

    # console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger