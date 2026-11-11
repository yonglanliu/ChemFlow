# Tutorials

Step-by-step guides for ChemFlow / AlchemForge workflows.

| Tutorial | What it covers |
| --- | --- |
| [Running on HPC](HPC_GUIDE.md) | Setting up and running the free-energy (OpenFE, GROMACS/PMX, OpenMM), PyRosetta and REINVENT4 workflows on a SLURM GPU cluster — environments, job submission, and GPU notes. |
| [Membrane system prep for OpenFE](MEMBRANE_PREP.md) | Fixing a Maestro/Desmond membrane PDB so it loads into OpenFE (`ProteinMembraneComponent`) — adding missing hydrogens, box vectors, and solving common `from_pdb_file` failures. |
| [Membrane input (`sol.cif`) for OpenFE RBFE](OPENFE_MEMBRANE_INPUT.md) | Saving an equilibrated Maestro/Desmond membrane as a wrapped mmCIF OpenFE can run — avoid `trj_align`/`trj_unwrap`, export mmCIF (not PDB, >99,999-atom overflow), wrap + assert `span ≤ box`; fixes the step-0 `Potential energy is NaN` crash. |
| [Wrapping a membrane with `alchemforge wrap-pbc`](MEMBRANE_WRAP_PBC.md) | One-command alternative: read the chorus box from Schrödinger, then `alchemforge wrap-pbc` rotates the tilted box, makes molecules whole, and centers the protein+membrane — also fixes the step-0 `Potential energy is NaN` crash. |
| [Configuring the 3D viewer](3D_VIEWER.md) | Setting up the AlchemFold binding-pose viewer — vendoring 3Dmol.js for offline use (`chemflow vendor-3dmol`), enabling WebGL (SwiftShader) on headless HPC nodes, loading a pose, and reading the affinity/pIC50 scores. |
| [Active learning virtual screening](ACTIVE_LEARNING_SCREENING.md) | Screening an ultra-large library (Enamine REAL, 10⁹–10¹⁰) with a Chemprop surrogate + Glide/VSW/Vina docking oracle — `chemflow active-learn` scaffold/seed/run, VSW license & host sizing, UCB acquisition, HPC resumability, and escalating hits to MM-GBSA/FEP. |

> Module-specific READMEs live next to their code under `src/chemflow/...` and `src/alchemforge/`.
