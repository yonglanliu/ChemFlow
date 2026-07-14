import os
import random
import numpy as np
import json
import torch
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Optional

# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def namespace_to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}

    if isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [namespace_to_dict(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    return obj

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    raise TypeError(
        f"{type(obj)} is not JSON serializable."
    )

def save_json(
    data,
    path: str | Path,
):
    path = Path(path)

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
            default=_json_default,
        )


# ============================================================
# Utils
# ============================================================

def get_resume_path(training_config: SimpleNamespace, checkpoint_dir: Path) -> Optional[Path]:
    """
    Resume options in TOML:

    resume = true                         -> resume from checkpoints/last_model.pt
    resume_checkpoint = "/path/to/last_model.pt" -> resume from explicit checkpoint
    """
    resume_checkpoint = getattr(training_config, "resume_checkpoint", None)
    resume = bool(getattr(training_config, "resume", False))

    if resume_checkpoint is not None and str(resume_checkpoint).strip().lower() not in ["", "none", "false"]:
        return Path(resume_checkpoint).expanduser().resolve()

    if resume:
        return checkpoint_dir / "last_model.pt"

    return None