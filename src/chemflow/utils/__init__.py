"""Small shared helpers.

``safe_mol_wt`` is re-exported lazily: importing it pulls in RDKit, and several
callers of this package (the OpenFE setup path, for one) only want the
structure helpers and may run in environments where RDKit is absent or slow to
import. PEP 562 keeps ``from chemflow.utils import safe_mol_wt`` working while
deferring the cost to first use.
"""

__all__ = ["safe_mol_wt"]


def __getattr__(name):
    if name == "safe_mol_wt":
        from chemflow.utils.chem import safe_mol_wt

        return safe_mol_wt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
