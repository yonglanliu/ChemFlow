"""Utilities for preserving user-supplied small-molecule partial charges."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

import numpy as np
from openfe.protocols.openmm_utils.charge_generation import (
    bulk_assign_partial_charges,
)


def has_complete_partial_charges(molecule: Any) -> bool:
    """Return whether every atom has a finite stored partial charge.

    ``SmallMoleculeComponent.to_openff()`` converts an SDF atom property list
    named ``PartialCharge`` into ``Molecule.partial_charges``.  An uncharged SDF
    produces ``None`` instead.  An all-zero array is also treated as uncharged,
    matching OpenFE protocol behavior; detailed charge-quality checks remain in
    the charge QC step.
    """

    try:
        off_molecule = molecule.to_openff()
        charges = off_molecule.partial_charges
    except (AttributeError, TypeError, ValueError):
        return False

    if charges is None:
        return False

    values = getattr(charges, "m", getattr(charges, "magnitude", charges))
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False

    return (
        array.size == off_molecule.n_atoms
        and bool(np.all(np.isfinite(array)))
        and bool(np.any(array != 0.0))
    )


def _component_key(molecule: Any) -> tuple[str, str]:
    """Build a charge-independent key used to restore bulk result ordering."""

    return (str(getattr(molecule, "name", "")), str(molecule.smiles))


def ensure_partial_charges(
    molecules: Iterable[Any],
    *,
    molecule_kind: str,
    method: str,
    toolkit_backend: str,
    generate_n_conformers: int | None,
    nagl_model: str | None,
    processors: int = 1,
) -> list[Any]:
    """Keep complete stored charges and generate charges only where missing.

    Returned components retain the same order as the input.  This matters when
    OpenFE's multiprocessing bulk helper finishes molecules out of order.
    """

    result = list(molecules)
    missing_indices = [
        index
        for index, molecule in enumerate(result)
        if not has_complete_partial_charges(molecule)
    ]
    preserved = len(result) - len(missing_indices)

    if not missing_indices:
        print(
            f"Using stored partial charges for all {len(result)} "
            f"{molecule_kind} molecule(s); skipping charge generation."
        )
        return result

    print(
        f"Preserving stored partial charges for {preserved}/{len(result)} "
        f"{molecule_kind} molecule(s); generating charges for "
        f"{len(missing_indices)}."
    )
    missing = [result[index] for index in missing_indices]
    generated = bulk_assign_partial_charges(
        molecules=missing,
        overwrite=False,
        method=method,
        toolkit_backend=toolkit_backend,
        generate_n_conformers=generate_n_conformers,
        nagl_model=nagl_model,
        processors=processors,
    )

    # The multiprocessing branch in OpenFE 1.12 collects futures as they
    # complete.  Match results back to inputs so ligand selection stays stable.
    generated_by_key: dict[tuple[str, str], deque[Any]] = defaultdict(deque)
    for molecule in generated:
        generated_by_key[_component_key(molecule)].append(molecule)

    for index in missing_indices:
        key = _component_key(result[index])
        try:
            result[index] = generated_by_key[key].popleft()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                "Charge generation returned a molecule that could not be "
                "matched to its input."
            ) from exc

    return result
