# Tutorials

Step-by-step guides for ChemFlow / AlchemForge workflows.

| Tutorial | What it covers |
| --- | --- |
| [Running on HPC](HPC_GUIDE.md) | Setting up and running the free-energy (OpenFE, GROMACS/PMX, OpenMM), PyRosetta and REINVENT4 workflows on a SLURM GPU cluster — environments, job submission, and GPU notes. |
| [Membrane system prep for OpenFE](MEMBRANE_PREP.md) | Fixing a Maestro/Desmond membrane PDB so it loads into OpenFE (`ProteinMembraneComponent`) — adding missing hydrogens, box vectors, and solving common `from_pdb_file` failures. |
| [Configuring the 3D viewer](3D_VIEWER.md) | Setting up the AlchemFold binding-pose viewer — vendoring 3Dmol.js for offline use (`chemflow vendor-3dmol`), enabling WebGL (SwiftShader) on headless HPC nodes, loading a pose, and reading the affinity/pIC50 scores. |

> Module-specific READMEs live next to their code under `src/chemflow/...` and `src/alchemforge/`.
