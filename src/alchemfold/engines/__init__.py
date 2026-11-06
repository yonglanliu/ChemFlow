"""Co-folding engine registry for AlchemFold docking.

Each engine module exposes three functions sharing the same docking config:

    validate(conf)                       -> raise on engine-specific misconfig
    write_inputs(conf, workdir)          -> (inputs: dict[str, Path], out_dir: Path)
    build_command(conf, inputs, out_dir) -> list[str]   # the shell command to run
"""

from __future__ import annotations

from types import ModuleType

from alchemfold.engines import boltz, chai, diffdock

ENGINES: dict[str, ModuleType] = {
    "boltz": boltz,
    "chai": chai,
    "diffdock": diffdock,
}

SUPPORTED_ENGINES = frozenset(ENGINES)


def get_engine(name: str) -> ModuleType:
    try:
        return ENGINES[str(name)]
    except KeyError:
        raise ValueError(
            f"Unknown docking engine: {name!r}; choose one of {sorted(ENGINES)}"
        ) from None
