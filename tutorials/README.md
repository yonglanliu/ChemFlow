# Tutorials

Step-by-step guides for ChemFlow / AlchemForge workflows.

| Tutorial | What it covers |
| --- | --- |
| [Running on HPC](HPC_GUIDE.md) | Setting up and running the free-energy (OpenFE, GROMACS/PMX, OpenMM), PyRosetta and REINVENT4 workflows on a SLURM GPU cluster — environments, job submission, and GPU notes. |
| [Membrane system prep for OpenFE](MEMBRANE_PREP.md) | Fixing a Maestro/Desmond membrane PDB so it loads into OpenFE (`ProteinMembraneComponent`) — adding missing hydrogens, box vectors, and solving common `from_pdb_file` failures. |

> Module-specific READMEs live next to their code under `src/chemflow/...` and `src/alchemforge/`.
