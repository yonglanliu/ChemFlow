"""Quick OpenMM and GROMACS GPU smoke tests."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GPUCheckResult:
    """Result from one GPU backend smoke test."""

    name: str
    passed: bool
    detail: str


def _command_output(
    command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_openmm_gpu(platform_name: str, device_index: str, steps: int) -> GPUCheckResult:
    """Create and advance a minimal OpenMM context on a GPU platform."""

    try:
        import openmm
        from openmm import unit
    except ImportError as error:
        return GPUCheckResult("OpenMM", False, f"ImportError: {error}")

    try:
        available = [
            openmm.Platform.getPlatform(index).getName()
            for index in range(openmm.Platform.getNumPlatforms())
        ]
        platform = openmm.Platform.getPlatformByName(platform_name)

        system = openmm.System()
        system.addParticle(12.0)
        system.addParticle(12.0)
        bond = openmm.HarmonicBondForce()
        bond.addBond(
            0,
            1,
            0.3 * unit.nanometer,
            1000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
        )
        system.addForce(bond)
        integrator = openmm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        properties = {"DeviceIndex": device_index, "Precision": "mixed"}
        context = openmm.Context(system, integrator, platform, properties)
        context.setPositions(
            [openmm.Vec3(0.0, 0.0, 0.0), openmm.Vec3(0.3, 0.0, 0.0)]
            * unit.nanometer
        )
        integrator.step(steps)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        selected_device = platform.getPropertyValue(context, "DeviceIndex")
        precision = platform.getPropertyValue(context, "Precision")
        return GPUCheckResult(
            "OpenMM",
            True,
            f"{platform_name} device {selected_device}, {precision} precision; "
            f"energy={energy}; available={','.join(available)}",
        )
    except (openmm.OpenMMException, OSError, RuntimeError, ValueError) as error:
        return GPUCheckResult("OpenMM", False, f"{type(error).__name__}: {error}")


def _write_argon_fixture(directory: Path, atom_count: int = 1000) -> tuple[Path, Path, Path]:
    """Write a self-contained, nonphysical argon system for GPU diagnostics."""

    gro = directory / "argon.gro"
    topology = directory / "argon.top"
    mdp = directory / "quick_gpu_test.mdp"

    side = 10
    spacing_nm = 0.35
    lines = ["AlchemForge GPU smoke-test system", str(atom_count)]
    for zero_index in range(atom_count):
        x_index = zero_index % side
        y_index = (zero_index // side) % side
        z_index = zero_index // (side * side)
        atom_index = zero_index + 1
        lines.append(
            f"{atom_index:5d}{'ARG':<5}{'AR':>5}{atom_index:5d}"
            f"{(x_index + 1) * spacing_nm:8.3f}"
            f"{(y_index + 1) * spacing_nm:8.3f}"
            f"{(z_index + 1) * spacing_nm:8.3f}"
        )
    lines.append("   4.20000   4.20000   4.20000")
    gro.write_text("\n".join(lines) + "\n")

    topology.write_text(
        """[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
1         2          no         1.0      1.0

[ atomtypes ]
; name  at.num  mass    charge  ptype  sigma  epsilon
AR      18      39.948  0.0     A      0.340  0.997

[ moleculetype ]
; name  nrexcl
ARG     0

[ atoms ]
; nr  type  resnr  residue  atom  cgnr  charge  mass
1     AR    1      ARG      AR    1     0.0     39.948

[ system ]
AlchemForge argon GPU smoke test

[ molecules ]
ARG  1000
"""
    )
    mdp.write_text(
        """integrator               = md
dt                       = 0.002
nsteps                   = 100
cutoff-scheme            = Verlet
nstlist                  = 10
rlist                    = 1.0
coulombtype              = Cut-off
rcoulomb                 = 1.0
vdwtype                  = Cut-off
rvdw                     = 1.0
tcoupl                   = no
pcoupl                   = no
constraints              = none
gen-vel                  = yes
gen-temp                 = 300
gen-seed                 = 2026
nstxout                  = 0
nstvout                  = 0
nstfout                  = 0
nstenergy                = 10
nstlog                   = 10
"""
    )
    return gro, topology, mdp


def test_gromacs_gpu(
    gromacs_command: str,
    tpr: Path | None,
    steps: int,
    threads: int,
    pme_target: str,
    keep_workdir: bool,
) -> GPUCheckResult:
    """Build or consume a TPR and force short-range work onto a GPU."""

    command = shlex.split(gromacs_command)
    if not command or shutil.which(command[0]) is None:
        return GPUCheckResult("GROMACS", False, f"command not found: {gromacs_command}")

    version = _command_output([*command, "--version"])
    if version.returncode != 0:
        return GPUCheckResult("GROMACS", False, version.stdout.strip())
    gpu_line = next(
        (line.strip() for line in version.stdout.splitlines() if "GPU support:" in line),
        "GPU support: not reported",
    )
    if "disabled" in gpu_line.casefold() or "none" in gpu_line.casefold():
        return GPUCheckResult("GROMACS", False, gpu_line)

    workdir = Path(tempfile.mkdtemp(prefix="alchemforge_gpu_test_"))
    try:
        selected_tpr = tpr.resolve() if tpr is not None else workdir / "quick_gpu_test.tpr"
        if tpr is not None and not selected_tpr.is_file():
            return GPUCheckResult("GROMACS", False, f"TPR not found: {selected_tpr}")
        if tpr is None:
            gro, topology, mdp = _write_argon_fixture(workdir)
            grompp = _command_output(
                [
                    *command,
                    "grompp",
                    "-f",
                    str(mdp),
                    "-c",
                    str(gro),
                    "-p",
                    str(topology),
                    "-o",
                    str(selected_tpr),
                    "-maxwarn",
                    "0",
                ],
                cwd=workdir,
            )
            if grompp.returncode != 0:
                detail = "\n".join(grompp.stdout.splitlines()[-20:])
                return GPUCheckResult("GROMACS", False, f"grompp failed:\n{detail}")

        mdrun = [
            *command,
            "mdrun",
            "-s",
            str(selected_tpr),
            "-deffnm",
            "gpu_smoke_test",
            "-nsteps",
            str(steps),
            "-ntmpi",
            "1",
            "-ntomp",
            str(threads),
            "-nb",
            "gpu",
            "-pme",
            pme_target,
        ]
        run = _command_output(mdrun, cwd=workdir)
        log_path = workdir / "gpu_smoke_test.log"
        log = log_path.read_text(errors="replace") if log_path.is_file() else ""
        combined = f"{run.stdout}\n{log}"
        markers = (
            "GPU selected for this run",
            "Mapping of GPU IDs",
            "on the GPU",
        )
        offloaded = any(marker in combined for marker in markers)
        if run.returncode != 0 or not offloaded:
            tail = "\n".join(combined.splitlines()[-30:])
            reason = "mdrun failed" if run.returncode else "GPU offload was not confirmed"
            return GPUCheckResult("GROMACS", False, f"{reason}; {gpu_line}\n{tail}")
        location = f"; files kept in {workdir}" if keep_workdir else ""
        return GPUCheckResult(
            "GROMACS",
            True,
            f"{gpu_line}; {steps} steps completed with GPU offload{location}",
        )
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _run_gpu_test(args: argparse.Namespace) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"CUDA_VISIBLE_DEVICES={visible}")
    results: list[GPUCheckResult] = []
    if not args.skip_openmm:
        results.append(
            test_openmm_gpu(args.openmm_platform, args.device_index, args.steps)
        )
    if not args.skip_gromacs:
        results.append(
            test_gromacs_gpu(
                args.gmx,
                args.tpr,
                args.steps,
                args.threads,
                args.pme,
                args.keep_workdir,
            )
        )
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    passed = bool(results) and all(result.passed for result in results)
    print("GPU TEST: PASSED" if passed else "GPU TEST: FAILED")
    return 0 if passed else 1


def add_gpu_test_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the quick GPU diagnostic command."""

    parser = subparsers.add_parser(
        "gpu-test",
        help="Run quick OpenMM and GROMACS GPU smoke tests",
    )
    parser.add_argument(
        "--tpr",
        type=Path,
        help="Existing GROMACS TPR to test; otherwise build a temporary argon system",
    )
    parser.add_argument("--gmx", default="gmx", help="GROMACS command")
    parser.add_argument(
        "--openmm-platform",
        choices=("CUDA", "OpenCL"),
        default="CUDA",
        help="OpenMM GPU platform (default: CUDA)",
    )
    parser.add_argument("--device-index", default="0", help="OpenMM device index")
    parser.add_argument("--steps", type=int, default=100, help="Steps per test")
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="GROMACS OpenMP threads (default: SLURM_CPUS_PER_TASK or 1)",
    )
    parser.add_argument(
        "--pme",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="GROMACS PME target (default: cpu)",
    )
    parser.add_argument("--skip-openmm", action="store_true")
    parser.add_argument("--skip-gromacs", action="store_true")
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the temporary GROMACS diagnostic files",
    )
    parser.set_defaults(func=_run_gpu_test)

