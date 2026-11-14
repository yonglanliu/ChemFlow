"""Helpers behind the OpenFE workflows.

The workflow entry points stay one level up — ``config``, ``workflow``,
``workflow_defaults``, ``execute`` and the ``rbfe`` / ``abfe`` / ``septop``
packages. Everything here is supporting machinery those call into:

* ``partial_charges``, ``charge``, ``charge_qc``, ``charge_report`` — assigning,
  checking and reporting ligand partial charges
* ``cofactor_from_pdb`` — build cofactor copies from one charged template and
  strip those residues from the protein
* ``system_preflight`` — refuse an input whose box or geometry would fail later
* ``prepare_system`` — build and equilibrate the system as part of the run
* ``membrane_assembly`` — membrane-specific assembly helpers
* ``network_edit``, ``network_view`` — inspect and edit the ligand network
* ``repair_json`` — fix charges in already-written transformation JSONs

Import submodules directly (``from chemflow.fe.openfe.utils.partial_charges
import ...``); nothing is re-exported here, so importing this package stays
cheap and free of OpenFE/RDKit imports.
"""
