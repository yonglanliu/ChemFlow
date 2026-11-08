"""Click-an-atom PROTAC linker design on top of REINVENT4's LinkInvent.

Layers (none import Qt, so the CLI, desktop apps and tests share them):

* :mod:`.heads`    – turn "the atom I clicked" on each ligand into a LinkInvent
                     warhead SMILES (``*`` marks the exit vector).
* :mod:`.results`  – parse LinkInvent's output CSV into candidates.
* :mod:`.scores`   – linker length/flexibility, whole-PROTAC properties and
                     ChemFlow-model predictions (RDKit based, each score fails
                     soft to ``None``).
* :mod:`.fit3d`    – optional 3D pocket fit: keep both ligands at their poses,
                     build the linker around them, report clashes and strain.
* :mod:`.designer` – orchestration: sample linkers, score them, rank them.
"""
