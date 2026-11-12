"""GPU/run banner for AlchemFold runs.

The implementation now lives in the shared :mod:`hpc.gpu` so the CLI, the FE
modules and the generated SLURM scripts all print the same report. This module
stays as AlchemFold's import point (``boltz.run`` / ``dock_run`` call
``print_run_info``).
"""

from __future__ import annotations

from hpc.gpu import NO_BF16 as _NO_BF16
from hpc.gpu import gpu_report, print_run_info

__all__ = ["gpu_report", "print_run_info", "_NO_BF16"]
