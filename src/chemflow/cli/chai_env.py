"""``chemflow chai-env`` — Chai-1 in its own conda env.

One env per docking engine: ``boltz`` (``chemflow boltz-env install``),
``chai`` (here) and ``diffdock`` (``chemflow diffdock-env install``). The shared
``alchemfold`` env is retired — the AlchemFold CLI itself runs from the
``chemflow`` env and resolves each engine's executable from the engine's own env.

    chemflow chai-env install --cuda 12.6
    chemflow chai-env status

Chai-1 pins ``torch<2.7``, which is exactly why it cannot share an env with
Boltz-2's cuEquivariance kernels.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

_DEFAULT_ENV = "chai"
_PACKAGES = ("chai_lab",)
_CONDA_EXTRAS = ("pyyaml", "rdkit>=2024.03", "numpy", "scipy",
                 "setuptools>=80", "wheel")
_LINUX_RUNTIME = ("libgcc-ng>=12", "libstdcxx-ng>=12")


def _conda_tool(prefer_mamba: bool = True) -> str | None:
    for tool in (("mamba", "conda") if prefer_mamba else ("conda", "mamba")):
        if shutil.which(tool):
            return tool
    return None


def install_commands(env: str, *, tool: str, channel: str = "conda-forge",
                     python_version: str = "3.11",
                     packages: tuple[str, ...] = _PACKAGES,
                     conda_extras: tuple[str, ...] = _CONDA_EXTRAS,
                     torch_index: str | None = None,
                     exists: bool = False) -> list[list[str]]:
    """Commands that build the Chai-1 env (pure, for testing)."""
    commands: list[list[str]] = []
    if not exists:
        commands.append([tool, "create", "-y", "-n", env, "-c", channel,
                         f"python={python_version}", *conda_extras])
    elif conda_extras:
        commands.append([tool, "install", "-y", "-n", env, "-c", channel,
                         *conda_extras])
    pip_pkgs = [tool, "run", "-n", env, "pip", "install", *packages]
    if torch_index:
        pip_pkgs += ["--extra-index-url", torch_index]
    commands.append(pip_pkgs)
    return commands


def _env_exists(tool: str, env: str) -> bool:
    return subprocess.run([tool, "run", "-n", env, "python", "-c", ""],
                          capture_output=True).returncode == 0


def _env_bin(tool: str, env: str, exe: str) -> str | None:
    probe = subprocess.run(
        [tool, "run", "-n", env, "python", "-c",
         "import shutil,sys; print(shutil.which(sys.argv[1]) or '')", exe],
        capture_output=True, text=True)
    return probe.stdout.strip() or None


_PROBE_SOURCE = r'''
import importlib, json
report = {}
for name in ("chai_lab", "torch"):
    try:
        module = importlib.import_module(name)
        report[name] = {"ok": True,
                        "version": str(getattr(module, "__version__", "") or "")}
    except BaseException as exc:
        report[name] = {"ok": False, "err": f"{type(exc).__name__}: {exc}"[:300]}
if report["torch"]["ok"]:
    import torch
    ok = bool(torch.cuda.is_available())
    report["torch"]["cuda"] = str(torch.version.cuda or "")
    report["torch"]["available"] = ok
    report["torch"]["device"] = torch.cuda.get_device_name(0) if ok else ""
print("CHEMFLOW_CHAI_PROBE " + json.dumps(report))
'''


def _probe(tool: str, env: str) -> dict:
    import json

    result = subprocess.run([tool, "run", "-n", env, "python", "-c", _PROBE_SOURCE],
                            capture_output=True, text=True)
    for line in (result.stdout or "").splitlines():
        if line.startswith("CHEMFLOW_CHAI_PROBE "):
            try:
                return json.loads(line.split(" ", 1)[1])
            except ValueError:
                break
    print(f"could not inspect env '{env}':\n"
          f"{(result.stderr or '').strip()[:500]}", file=sys.stderr)
    return {}


def _print_report(report: dict, env: str) -> None:
    if not report:
        return
    chai, torch_info = report.get("chai_lab", {}), report.get("torch", {})
    for label, info in (("Chai-1", chai), ("torch", torch_info)):
        if info.get("ok"):
            extra = f"  {info['version']}" if info.get("version") else ""
            if label == "torch" and info.get("cuda"):
                extra += f"  (CUDA {info['cuda']})"
            print(f"  {label:10s}-> importable ✓{extra}")
        else:
            print(f"  {label:10s}-> NOT importable")
            if info.get("err"):
                print(f"  {'':10s}   {info['err']}")
    if torch_info.get("ok"):
        if torch_info.get("available"):
            print(f"  {'GPU':10s}-> {torch_info['device']}")
        else:
            print(f"  {'GPU':10s}-> not visible from here "
                  "(expected on a login node; re-check on a GPU node)")
    print(f"\n  Chai-1 usable: {'yes' if chai.get('ok') else 'no'} (env '{env}')")


def _install(args) -> int:
    tool = args.tool or _conda_tool()
    if not tool:
        print("error: neither mamba nor conda found on PATH. On HPC, load conda "
              "first, then retry; otherwise install Miniforge.", file=sys.stderr)
        return 1

    torch_index = None
    if args.cuda:
        torch_index = ("https://download.pytorch.org/whl/cu"
                       + args.cuda.replace(".", ""))

    preamble: list[list[str]] = []
    exists = _env_exists(tool, args.env)
    if args.recreate and exists:
        preamble.append([tool, "env", "remove", "-y", "-n", args.env])
        exists = False
    elif exists:
        print(f"Environment '{args.env}' already exists — installing/updating "
              "into it (use --recreate for a clean rebuild).")

    commands = preamble + install_commands(
        args.env, tool=tool, channel=args.channel,
        python_version=args.python_version,
        conda_extras=_CONDA_EXTRAS + (_LINUX_RUNTIME if sys.platform == "linux" else ()),
        torch_index=torch_index, exists=exists)
    for command in commands:
        print("$ " + " ".join(command))
        if args.dry_run:
            continue
        rc = subprocess.run(command, check=False).returncode
        if rc != 0:
            print(f"error: command failed (exit {rc}): {' '.join(command)}",
                  file=sys.stderr)
            return rc
    if args.dry_run:
        return 0

    print()
    _print_report(_probe(tool, args.env), args.env)
    path = _env_bin(tool, args.env, "chai-lab")
    if path:
        print("\nThe AlchemFold CLI finds this automatically (sibling env named "
              f"'{args.env}'). To be explicit, set in the docking config:")
        print(f'    chai_command: "{path}"')
    print("\nTip: point CHAI_DOWNLOADS_DIR at /data or scratch — Chai-1 "
          "downloads its weights on first use and they are large.")
    return 0


def _status(args) -> int:
    tool = args.tool or _conda_tool()
    print(f"Chai-1 environment: {args.env}")
    if not tool:
        print("  (no conda/mamba on PATH to inspect it)")
        return 0
    _print_report(_probe(tool, args.env), args.env)
    path = _env_bin(tool, args.env, "chai-lab")
    if path:
        print(f'\n  chai_command: "{path}"')
    return 0


def add_chai_env_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "chai-env", help="Create the Chai-1 conda env (one env per docking engine)")
    commands = parser.add_subparsers(dest="chai_env_command", required=True)

    install = commands.add_parser("install", help="Create the env and install chai_lab")
    install.add_argument("--env", default=_DEFAULT_ENV,
                         help=f"conda env name (default {_DEFAULT_ENV})")
    install.add_argument("--channel", default="conda-forge")
    install.add_argument("--python-version", default="3.11")
    install.add_argument("--cuda", default=None,
                         help="CUDA version for the GPU Torch build, e.g. 12.6 "
                              "(Chai-1 pins torch<2.7)")
    install.add_argument("--recreate", action="store_true",
                         help="remove the env first and rebuild it cleanly")
    install.add_argument("--tool", default=None,
                         help="mamba | conda (default: mamba if present)")
    install.add_argument("--dry-run", action="store_true",
                         help="print the commands only")
    install.set_defaults(func=_install)

    status = commands.add_parser("status", help="Show whether Chai-1 is importable")
    status.add_argument("--env", default=_DEFAULT_ENV,
                        help=f"conda env name (default {_DEFAULT_ENV})")
    status.add_argument("--tool", default=None, help="mamba | conda")
    status.set_defaults(func=_status)
