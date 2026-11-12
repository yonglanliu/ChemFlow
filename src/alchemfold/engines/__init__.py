"""Docking engine registry.

Each engine lives in its OWN subpackage — :mod:`alchemfold.boltz`,
:mod:`alchemfold.chai`, :mod:`alchemfold.diffdock` — holding its adapter
(``engine.py``) plus whatever else only that engine needs (runner, batch
screening, trajectory tools). Each adapter exposes the same three functions
over the shared docking config:

    validate(conf)                       -> raise on engine-specific misconfig
    write_inputs(conf, workdir)          -> (inputs: dict[str, Path], out_dir: Path)
    build_command(conf, inputs, out_dir) -> list[str]   # the shell command to run

Each also runs in its own conda env (boltz / chai / diffdock); see
:func:`alchemfold._exec.prefer_envs_for`.
"""

from __future__ import annotations

from types import ModuleType

from alchemfold.boltz import engine as boltz
from alchemfold.chai import engine as chai
from alchemfold.diffdock import engine as diffdock

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
