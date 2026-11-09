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


def harmonize_isomorphic_charges(components: list, *, molecule_kind: str = "cofactor") -> list:
    """Give graph-isomorphic SmallMoleculeComponents one shared partial-charge set.

    OpenFE's RFE protocol rejects isomorphic ``SmallMoleculeComponent``s whose
    partial charges differ ("Found SmallMoleculeComponents that are isomorphic but
    with different charges"). Cofactors supplied as several copies, or charged
    per-conformer, commonly trip this. Group the components by graph isomorphism and
    copy the first member's partial charges onto the rest, remapped through the
    isomorphism's atom map. Order is preserved; components without an isomorphic
    duplicate are returned unchanged.
    """
    comps = list(components)
    if len(comps) < 2:
        return comps

    import numpy as np  # noqa: PLC0415
    from openff.toolkit import Molecule  # noqa: PLC0415
    from openff.units import unit as offunit  # noqa: PLC0415

    def _iso(a, b):
        # Use OpenFF's default (strict) isomorphism — fast and discriminating.
        # Fully-relaxed matching (ignoring bond order/charge/stereo) is MUCH
        # slower for symmetric molecules (automorphism blow-up), so we keep the
        # default here; the atom remap below handles charge-array alignment.
        return Molecule.are_isomorphic(a, b, return_atom_map=True)

    offs = [c.to_openff() for c in comps]
    out = list(comps)
    used = [False] * len(comps)
    harmonized = 0
    for i in range(len(offs)):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, len(offs)):
            if used[j]:
                continue
            is_iso, _ = _iso(offs[i], offs[j])
            if is_iso:
                group.append(j)
        for k in group:
            used[k] = True
        if len(group) < 2:
            continue
        rep_idx = next((k for k in group if offs[k].partial_charges is not None), None)
        if rep_idx is None:
            continue
        rep = offs[rep_idx]
        rep_charges = rep.partial_charges
        for k in group:
            if k == rep_idx:
                continue
            mol = offs[k]
            # atom_map maps mol (member) atom indices -> rep atom indices.
            is_iso, mol_to_rep = _iso(mol, rep)
            if not is_iso or not mol_to_rep:
                continue
            try:
                # Reorder the member's atoms into the representative's ordering so
                # the per-atom charge arrays line up index-for-index (OpenFE
                # compares charges by atom index). Geometry is remapped too.
                remapped = mol.remap(mol_to_rep, current_to_new=True)
                remapped.partial_charges = rep_charges
                out[k] = type(comps[k]).from_openff(remapped)
            except Exception:
                # Fallback: assign chemically-equivalent charges in the member's
                # own atom order (works if OpenFE remaps before comparing).
                rep_vals = rep_charges.m_as(offunit.elementary_charge)
                new = np.zeros(mol.n_atoms)
                for mol_atom, rep_atom in mol_to_rep.items():
                    new[mol_atom] = rep_vals[rep_atom]
                mol.partial_charges = new * offunit.elementary_charge
                out[k] = type(comps[k]).from_openff(mol)
            harmonized += 1

    if harmonized:
        print(
            f"Harmonized partial charges across {harmonized} isomorphic "
            f"{molecule_kind} copy/copies (OpenFE requires identical charges "
            f"for isomorphic components)."
        )
    return out
