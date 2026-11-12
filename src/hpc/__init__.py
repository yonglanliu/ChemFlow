"""Shared HPC layer: GPU info and per-module SLURM presets.

A top-level package beside ``chemflow`` / ``alchemfold`` / ``alchemforge``,
because every one of them needs the same two things and all three had grown
their own copy: GPU detection, and "which SLURM resources should this job ask
for".

Presets are per MODULE. Each module has a file in :mod:`hpc.presets`
(``docking.yaml``, ``screening.yaml``, ``openfe.yaml``, ``gromacs.yaml``,
``openmm.yaml``, ``mmgbsa.yaml``) holding its own named presets and defaults —
a docking run and an RBFE campaign want very different walltimes. A site can
override any of it from one user file without touching the package:

    1. ``$CHEMFLOW_HPC_CONFIG``
    2. ``<PROJECT_ROOT>/env/hpc.yaml``
    3. ``~/.config/chemflow/hpc.yaml``

Typical use::

    from hpc import presets

    names = presets.names("openfe")                 # for a dropdown
    res   = presets.resources("openfe", names[0])   # canonical Resources
    res.gres()                                      # 'gpu:a100:1'

Modules keep their own script renderers; this package only decides the numbers
and renders the individual ``#SBATCH`` fragments they share.
"""

from __future__ import annotations

from hpc.resources import Resources, normalize_keys

__all__ = ["Resources", "normalize_keys", "presets", "slurm", "gpu"]
