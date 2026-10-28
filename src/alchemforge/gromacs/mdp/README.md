# Equilibration — Membrane RBFE System

This covers energy minimization, NVT, and NPT equilibration for a membrane-embedded protein–ligand–cofactor system, prior to FEP production. Each stage has an AMBER and a CHARMM variant.

## Files

| Stage                              | AMBER                   | CHARMM                   |
| ---------------------------------- | ----------------------- | ------------------------ |
| EM (energy minimization)           | `em_AMBER.mdp`        | `em_CHARMM.mdp`        |
| NVT (restrained, fixed volume)     | `nvt_equil_AMBER.mdp` | `nvt_equil_CHARMM.mdp` |
| NPT (restrained, pressure-coupled) | `npt_equil_AMBER.mdp` | `npt_equil_CHARMM.mdp` |

Use the column matching your force field throughout — do not mix AMBER and CHARMM settings within the same pipeline. The van der Waals treatment in particular is fit specifically to each force field family; using the wrong one produces a subtly different effective potential than the parameters were validated against.

## Run order

```bash
# 1. Energy minimization
gmx grompp -f em_AMBER.mdp -c system_assembled.gro -p topol.top \
    -n index.ndx -o em.tpr -maxwarn 1
gmx mdrun -deffnm em -v

# 2. NVT
gmx grompp -f nvt_equil_AMBER.mdp -c em.gro -r em.gro -p topol.top \
    -n index.ndx -o nvt.tpr
gmx mdrun -deffnm nvt -v

# 3. NPT (continues from NVT)
gmx grompp -f npt_equil_AMBER.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top \
    -n index.ndx -o npt.tpr
gmx mdrun -deffnm npt -v
```

---

## EM stage — what it does and why

### `integrator = steep`

Steepest descent — not real dynamics, just moves atoms downhill in energy at each step. Robust and simple, good for resolving severe initial clashes, which are common right after system assembly (ligand insertion into an existing membrane structure, cofactor placement, solvation). There's no concept of temperature or physically meaningful timestep here.

### `emtol = 200.0`

Convergence criterion (kJ/mol/nm) — minimization stops once the maximum force on any atom drops below this value. 200 is a reasonably loose, standard target sufficient to remove gross clashes before moving to dynamics; a much tighter value isn't necessary since the subsequent restrained NVT stage will continue relaxing the system.

### `emstep = 0.01`

Initial step size (nm) for steepest descent. GROMACS adjusts this automatically as minimization proceeds; this is just the starting value.

### `nsteps = 10000`

Maximum number of minimization steps allowed. Set higher than what a typical soluble-protein system would need, since membrane systems (freshly assembled lipid + inserted ligand) tend to have more severe initial steric clashes that take longer to resolve.

### `define = -DPOSRES -DPOSRES_LIG -DPOSRES_COFACTOR`

Same restraint logic as the NVT/NPT stages — protein, ligand, and cofactor held in place even during minimization, so the minimizer resolves local clashes (e.g., lipid tails overlapping the newly-placed ligand) without letting the whole assembly collapse toward a different, unintended arrangement. Lipids remain unrestrained.

### Neighbor searching, electrostatics, vdW, constraints

Identical in form and purpose to the NVT/NPT stages described below — same cutoff-scheme, same force-field-specific vdW modifier and cutoff distances (AMBER: `Potential-shift`/1.0 nm; CHARMM: `Force-switch`/1.2 nm), same `h-bonds`/LINCS constraint treatment. These need to match what you'll use in the rest of the pipeline so the minimized structure is consistent with the potential energy surface it will next be equilibrated under.

### Output settings

Lighter-weight than the dynamics stages (`nstenergy`/`nstlog = 100`, `nstxout-compressed = 1000`) since minimization is typically short and you mainly just want to confirm convergence, not produce a trajectory for analysis. Full-precision `.trr` output remains disabled, same reasoning as the dynamics stages.

### No thermostat, barostat, or velocity settings

Minimization has none of `tcoupl`, `pcoupl`, `gen-vel`, `tc-grps`, `comm_grps`, etc. — there are no velocities or temperature/pressure concepts in steepest descent, so these parameters simply don't apply at this stage.

### Checking convergence

```bash
gmx energy -f em.edr -o potential.xvg
# select "Potential" — should plateau/flatten by the end of the run
```

If the potential energy is still dropping sharply at `nsteps`, consider increasing `nsteps` rather than proceeding with an incompletely minimized structure into NVT.

---

## NVT stage — what it does and why

### Identical across AMBER and CHARMM

- **Position restraints** (`-DPOSRES -DPOSRES_LIG -DPOSRES_COFACTOR`) — hold protein, ligand, and cofactor in place while water and lipid relax around them. Lipids are intentionally left unrestrained so they can pack naturally.
- **Run length** (`nsteps = 150000` at `dt = 0.001`, i.e. 150 ps) — membrane systems need longer early equilibration than soluble proteins because lipid packing (area-per-lipid relaxation) is slow.
- **Timestep** (`dt = 0.001`, 1 fs) — smaller than the eventual 2 fs production timestep, for stability while large initial forces from system assembly are still relaxing.
- **Output frequencies** — standard logging/trajectory intervals for QC inspection, not used in the free energy analysis itself. Full-precision `.trr` output is disabled (`nstxout/nstvout/nstfout = 0`) since the compressed `.xtc` is sufficient and `.cpt` checkpoints handle restarts.
- **Constraints** (`h-bonds` / `LINCS`) — removes fast H-bond vibrations so a 2 fs timestep is stable later; not force-field-specific.
- **Thermostat** (`V-rescale`) — generates a correct canonical ensemble, unlike plain Berendsen.
- **Two-group thermostat/COM split** (`SOLU_MEMB SOLV`) — a lipid bilayer and bulk water can show different effective local temperature/relaxation behavior, so they're coupled to the thermostat and corrected for center-of-mass drift independently rather than as one group. This is a **membrane requirement**, not a force-field one.
- **Velocity generation** (`gen-vel = yes`, `gen_temp = 300`) — appropriate for this first equilibration stage, since there are no prior velocities to continue from.
- **`pcoupl = no`** — NVT is fixed-volume by definition; no barostat active.

### Differs by force field

| Parameter        | AMBER               | CHARMM           | Reason                                |
| ---------------- | ------------------- | ---------------- | ------------------------------------- |
| `rlist`        | 1.0 nm              | 1.2 nm           | Matches each FF's validated cutoff    |
| `rcoulomb`     | 1.0 nm              | 1.2 nm           | Same                                  |
| `vdw-modifier` | `Potential-shift` | `Force-switch` | See below                             |
| `rvdw_switch`  | not used            | 1.0 nm           | Only meaningful with `Force-switch` |
| `rvdw`         | 1.0 nm              | 1.2 nm           | Matches each FF's convention          |

**`Potential-shift` (AMBER)** shifts the LJ potential curve down by a constant so it equals exactly zero at `rvdw`, removing the energy discontinuity but not the force discontinuity. Simpler and cheaper; standard pairing for AMBER-family force fields (ff19SB, GAFF2, Lipid17/21).

**`Force-switch` (CHARMM)** smoothly switches both energy and force to zero over `[rvdw_switch, rvdw]`, removing both discontinuities at slightly higher cost. Standard for CHARMM36/36m and CHARMM36 lipids.

---

## NPT stage — what changes from NVT, and why

### `continuation = yes`

Tells GROMACS this run continues from a previous stage (NVT) rather than starting fresh. The input structure's bond lengths are already constraint-consistent and the input velocities are already equilibrated, so GROMACS should **not** re-initialize/reset them the way it would for a brand-new start. This is paired with `gen-vel = no` below — the two settings go together.

### `gen-vel = no`

Unlike the NVT stage, this run does **not** generate fresh Maxwell-Boltzmann velocities. It reads real velocities from the NVT checkpoint (`-t nvt.cpt`), continuing the dynamics smoothly rather than restarting with an artificial velocity distribution.

### `dt = 0.002`

Now at the full production timestep (2 fs). By this point the system has already relaxed past its most unstable initial state during the smaller-timestep NVT stage, so the larger, more efficient timestep is safe.

### `nsteps = 250000`

500 ps at 2 fs — longer than the NVT stage, giving the barostat time to bring the box dimensions (including membrane area-per-lipid) toward their equilibrium values.

### Pressure coupling (new in this stage)

```
pcoupl           = berendsen
pcoupltype       = semiisotropic
tau_p            = 5.0
ref_p            = 1.0   1.0
compressibility  = 4.5e-5  4.5e-5
```

- **`pcoupl = berendsen`** — used here specifically because this is an *early* equilibration stage. Berendsen coupling doesn't generate a strictly correct NPT ensemble on its own, but its strong, non-oscillatory relaxation is good at quickly damping large initial pressure fluctuations without overshooting. **Do not use Berendsen for production** — switch to `C-rescale` (or `Parrinello-Rahman`) for later equilibration sub-stages and production, once the system is closer to its true equilibrium.
- **`pcoupltype = semiisotropic`** — couples pressure independently in the lateral (xy, membrane plane) and normal (z, membrane normal) directions, rather than scaling the whole box uniformly. A bilayer's area and thickness equilibrate differently and need independent control; using `isotropic` here would force them to scale together, distorting both. This is a **membrane requirement**, identical for AMBER and CHARMM.
- **`ref_p = 1.0 1.0`** — target pressure (bar), one value per coupled direction (xy, z). Both are standard atmospheric pressure; the two numbers correspond to the two independently-coupled directions, not different targets.
- **`compressibility = 4.5e-5 4.5e-5`** — isothermal compressibility of water (bar⁻¹), used for both directions; tells the barostat how much volume change to expect per unit pressure change.
- **`tau_p = 5.0`** — pressure coupling time constant (ps); longer than the temperature time constant since pressure equilibrates more slowly, particularly lateral membrane relaxation.

### Everything else

Position restraints, constraints, thermostat choice and groups, COM-removal groups, and the force-field-specific vdW/cutoff block all carry over unchanged from the NVT stage — see the NVT section above.

---

## Required companion files

Both stages reference the custom groups `SOLU_MEMB` and `SOLV`, which are **not** GROMACS built-ins. Generate an index file before running either stage:

```bash
gmx make_ndx -f system.gro -o index.ndx
# Inside make_ndx:
#   combine Protein + Ligand + Cofactor + Lipid residue groups -> name it SOLU_MEMB
#   combine SOL + ion groups -> name it SOLV
#   q
```

Supply it at every `grompp` call with `-n index.ndx`. Omitting this will cause `grompp` to fail with a "group not found" error.

## QC checkpoint before moving on

After NPT completes, confirm the membrane has actually equilibrated before trusting the structure for production:

```bash
gmx energy -f npt.edr -o box_dims.xvg
# select Box-X, Box-Y, Box-Z
# Box-X and Box-Y should stabilize together (lateral isotropy within the membrane plane)
# Box-Z should stabilize independently (bilayer thickness)
```

If box dimensions haven't plateaued, extend the NPT stage (or add further restraint-ramping sub-stages at progressively lower force constants) before proceeding to production.

## Next step

Once NPT is stable, continue with further restraint-ramping NPT sub-stages (force constants typically stepped down: 1000 → 500 → 200 → 50 → 0 kJ/mol/nm²) until restraints are fully or near-fully released, then proceed to **per-lambda-window FEP equilibration and production** — see the next section.

---

# Per-Lambda-Window FEP Files

Once the base system is assembled and minimally stable, every lambda window needs its **own** complete EM → NVT → NPT → Production chain, run entirely at that window's own alchemical state. You cannot reuse one window's equilibrated structure as the production starting point for a different window — see the note below for why.

## Files (templates — `{LAMBDA_STATE}` placeholder)

| Stage | AMBER                 |
| ----- | --------------------- |
| EM    | `em_FEP_AMBER.mdp`  |
| NVT   | `nvt_FEP_AMBER.mdp` |
| NPT   | `npt_FEP_AMBER.mdp` |

(CHARMM equivalents follow the same pattern — swap in the CHARMM vdW block from the EM/NVT/NPT files above: `rlist/rcoulomb/rvdw = 1.2`, `vdw-modifier = Force-switch`, `rvdw_switch = 1.0`.)

## Why each window needs its own full chain

Each lambda window experiences genuinely different forces — at λ=0 the ligand is fully interacting and the surrounding water/lipid has relaxed around its real size and charge; at an intermediate λ the ligand may be partially decoupled via soft-core, meaning its effective size/charge is different and the environment should relax differently around it; near full decoupling, water or lipid may want to occupy space the ligand previously excluded. Equilibrating once (e.g. at λ=0) and then switching `init-lambda-state` only for production means production starts far from that window's true equilibrium, contaminating early sampling with non-equilibrium relaxation rather than genuine Boltzmann-sampled data. `free-energy = yes` must therefore be active with the same `init-lambda-state` through EM, NVT, NPT, and production for a given window — this is what actually turns on the soft-core potentials and lambda-dependent forces that the system needs to equilibrate under.

What *can* be shared: the overall system assembly (protein, membrane, cofactor, water, ions) is built once. What can *optionally* speed things up without weakening correctness: using one window's finished equilibration as the starting structure for an adjacent window's own equilibration (since neighboring windows are often similar) — but that adjacent window still must run its own complete equilibration at its own lambda state before its production data is trusted; this is a smarter starting guess, not a substitute for equilibrating.

## What's new in these files vs. the base EM/NVT/NPT files

```ini
free-energy               = yes
init-lambda-state         = {LAMBDA_STATE}   ; set once per window, identical across
                                               ; EM/NVT/NPT/production for that window
delta-lambda              = 0                 ; fixed-lambda per window, not slow-growth
calc-lambda-neighbors     = -1                ; required for MBAR (computes energies at
                                               ; all states, not just neighbors)

coul-lambdas  = ...   ; electrostatic scaling schedule — identical across ALL windows'
vdw-lambdas   = ...   ; files; only init-lambda-state (the index into these vectors)
bonded-lambdas= ...   ; changes between windows
restraint-lambdas = ...

sc-alpha      = 0.5   ; soft-core parameters — prevent singularities at partially-
sc-power      = 1     ; decoupled/overlapping atom states; must be present at every
sc-sigma      = 0.3   ; stage where free-energy = yes is set, including EM
sc-coul       = no
sc-r-power    = 6

couple-moltype  =      ; blank: this system uses a pmx-style hybrid topology (A/B
couple-lambda0  = vdw-q ; states encoded directly in the ligand .itp), not GROMACS's
couple-lambda1  = vdw-q ; built-in molecule-decoupling mechanism
couple-intramol = no
```

The lambda vectors, soft-core block, and `couple-*` settings must be byte-identical across every window's EM/NVT/NPT/production files — the only line that changes per window is `init-lambda-state`.

## Generating one full set of files per window

```python
# generate_lambda_mdps.py
n_windows = 13
stages = ["em_FEP_AMBER.mdp", "nvt_FEP_AMBER.mdp", "npt_FEP_AMBER.mdp"]

for lam in range(n_windows):
    outdir = f"lambda_{lam}"
    os.makedirs(outdir, exist_ok=True)
    for stage_file in stages:
        with open(stage_file) as f:
            content = f.read()
        content = content.replace("{LAMBDA_STATE}", str(lam))
        out_name = stage_file.replace("_FEP_AMBER", "")
        with open(f"{outdir}/{out_name}", "w") as out:
            out.write(content)
```

This produces `lambda_0/em.mdp`, `lambda_0/nvt.mdp`, `lambda_0/npt.mdp`, `lambda_1/em.mdp`, … `lambda_12/npt.mdp` — 13 directories × 3 files, each internally consistent with one `init-lambda-state`.

## Run order per window

```bash
cd lambda_5   # example: window 5

gmx grompp -f em.mdp -c ../system_assembled.gro -p ../topol.top \
    -n ../index.ndx -o em.tpr -maxwarn 1
gmx mdrun -deffnm em -v

gmx grompp -f nvt.mdp -c em.gro -r em.gro -p ../topol.top \
    -n ../index.ndx -o nvt.tpr
gmx mdrun -deffnm nvt -v

gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p ../topol.top \
    -n ../index.ndx -o npt.tpr
gmx mdrun -deffnm npt -v
```

Repeat identically for every `lambda_N` directory (can be parallelized as a SLURM array job across windows, since each window's chain is fully independent of the others), then proceed to production using `npt.gro`/`npt.cpt` as the starting point for that window.
