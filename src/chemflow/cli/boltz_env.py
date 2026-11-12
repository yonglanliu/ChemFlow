"""``chemflow boltz-env`` — a Boltz-2 env WITH the optimized GPU kernels.

Separate from the AlchemFold env on purpose. The kernels (cuEquivariance, and on
older Boltz the legacy ``trifast``) need a recent PyTorch, while ``chai_lab``
pins ``torch<2.7``. Installing them there upgrades torch and breaks Chai-1, so
Boltz gets its own env and AlchemFold calls into it via ``boltz_command``.

Note on ``trifast``: Boltz-2 replaced it with cuEquivariance (the old
``--no_trifast`` flag became ``--no_kernels``). On a current Boltz it is simply
not installed, and ``status`` checks whether the installed Boltz references it
at all before calling that a problem.

    chemflow boltz-env install --cuda 12.6
    chemflow boltz-env status

Then in the docking / screening YAML::

    boltz_command: "/path/to/conda/envs/boltz/bin/boltz"
    no_kernels: false
"""

from __future__ import annotations

import shutil
import subprocess
import sys

_DEFAULT_ENV = "boltz"
# The [cuda] extra is what pins cuequivariance-torch + trifast at versions Boltz
# has tested together — more reliable than choosing them independently.
_PACKAGES = ("boltz[cuda]",)
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
    """Commands that build the Boltz+kernels env (pure, for testing)."""
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
    path = probe.stdout.strip()
    return path or None


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

    # The whole point of this env is the kernels — verify rather than assume.
    report = _probe(tool, args.env)
    boltz_path = _env_bin(tool, args.env, "boltz") or f"<{args.env}>/bin/boltz"
    print()
    _print_report(report, args.env)
    print("\nUse it from the AlchemFold config without switching envs — set:")
    print("    no_kernels: false")
    print("AlchemFold prefers a sibling env named 'boltz' when the kernels are on, "
          "so a path is only needed for a differently-named env:")
    print(f"    boltz_command: \"{boltz_path}\"")
    return 0


# Run inside the target env: one probe, so the report is consistent and cheap.
_PROBE_SOURCE = r'''
import importlib, importlib.util, json, pathlib

def mod(name):
    try:
        m = importlib.import_module(name)
    except BaseException as exc:
        return {"ok": False, "err": f"{type(exc).__name__}: {exc}"[:300]}
    return {"ok": True, "version": str(getattr(m, "__version__", "") or "")}

report = {name: mod(name) for name in
          ("boltz", "cuequivariance_torch", "trifast")}

# Is trifast REACHABLE in the installed boltz, or just dead code? Boltz 2.2.1
# still ships `_trifast_attn` in triangular_attention/primitives.py, but nothing
# calls it — triangle attention and triangle multiplication both go to
# cuequivariance_torch. A plain grep for the string therefore cries wolf; parse
# instead, and only call it required if the import is at module level or the
# helper holding it is referenced somewhere.
usage, detail = None, ""
try:
    import ast

    spec = importlib.util.find_spec("boltz")
    root = (pathlib.Path(list(spec.submodule_search_locations)[0])
            if spec and spec.submodule_search_locations else None)
    sources = {}
    if root is not None:
        for path in root.rglob("*.py"):
            try:
                sources[path] = path.read_text(errors="ignore")
            except BaseException:
                pass

    def imports_trifast(node):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            return False
        return any(n.split(".")[0] == "trifast" for n in names if n)

    helpers, module_level = set(), False
    trees = {}
    for path, text in sources.items():
        if "trifast" not in text:
            continue
        try:
            trees[path] = tree = ast.parse(text)
        except SyntaxError:
            usage = "unknown"
            continue
        for node in tree.body:                      # top-level import => hard dep
            if imports_trifast(node):
                module_level = True
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(imports_trifast(sub) for sub in ast.walk(node)):
                    helpers.add(node.name)

    if not trees and usage is None:
        usage = "absent"
    elif module_level:
        usage, detail = "required", "imported at module level"
    elif helpers:
        callers = []
        for path, text in sources.items():
            if not any(h in text for h in helpers):
                continue
            try:
                tree = trees.get(path) or ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                name = (node.id if isinstance(node, ast.Name)
                        else node.attr if isinstance(node, ast.Attribute) else None)
                if name in helpers:
                    callers.append(f"{path.name}:{getattr(node, 'lineno', '?')}")
        usage = "required" if callers else "unused"
        detail = (", ".join(sorted(set(callers))[:3]) if callers
                  else "defined but never called: " + ", ".join(sorted(helpers)))
    elif usage is None:
        usage = "unknown"
except BaseException as exc:
    usage, detail = None, f"{type(exc).__name__}: {exc}"[:200]
report["trifast_usage"] = usage
report["trifast_detail"] = detail

try:
    import torch
    cuda_ok = bool(torch.cuda.is_available())
    report["torch"] = {
        "ok": True, "version": torch.__version__,
        "cuda": str(torch.version.cuda or ""), "available": cuda_ok,
        "device": (torch.cuda.get_device_name(0) if cuda_ok else ""),
    }
except BaseException as exc:
    report["torch"] = {"ok": False, "err": f"{type(exc).__name__}: {exc}"[:300]}

print("CHEMFLOW_BOLTZ_PROBE " + json.dumps(report))
'''


def _probe(tool: str, env: str) -> dict:
    """Import-check the env from the inside; returns {} if it could not run."""
    import json

    result = subprocess.run([tool, "run", "-n", env, "python", "-c", _PROBE_SOURCE],
                            capture_output=True, text=True)
    for line in (result.stdout or "").splitlines():
        if line.startswith("CHEMFLOW_BOLTZ_PROBE "):
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
    boltz, cueq = report.get("boltz", {}), report.get("cuequivariance_torch", {})
    trifast, torch_info = report.get("trifast", {}), report.get("torch", {})

    def line(label: str, info: dict, suffix: str = "") -> None:
        if info.get("ok"):
            state = "importable ✓" + (f"  {info['version']}" if info.get("version") else "")
        else:
            state = "NOT importable"
        print(f"  {label:15s}-> {state}{suffix}")
        if not info.get("ok") and info.get("err") and not suffix:
            print(f"  {'':15s}   {info['err']}")

    line("Boltz-2", boltz)
    line("cuEquivariance", cueq)
    # trifast is the previous-generation triangle-attention kernel. Boltz 2.2.1
    # still ships a `_trifast_attn` helper but never calls it (cuEquivariance
    # handles both triangle attention and triangle multiplication), so the probe
    # distinguishes "mentioned" from "actually reachable".
    usage = report.get("trifast_usage")
    detail = report.get("trifast_detail") or ""
    notes = {
        "absent": "  (not referenced by this Boltz)",
        "unused": "  (legacy — present but never called)",
        "required": "  REQUIRED: this Boltz calls it",
        "unknown": "  (could not tell whether this Boltz uses it)",
    }
    line("trifast", trifast, notes.get(usage, "  (legacy kernel; optional)"))
    if detail and not trifast.get("ok") and usage in ("unused", "required"):
        print(f"  {'':15s}   {detail}")

    if torch_info.get("ok"):
        cuda = torch_info.get("cuda") or "cpu build"
        print(f"  {'torch':15s}-> {torch_info['version']}  (CUDA {cuda})")
        if torch_info.get("available"):
            print(f"  {'GPU':15s}-> {torch_info['device']}")
        else:
            print(f"  {'GPU':15s}-> not visible from here "
                  "(expected on a login node; re-check on a GPU node)")
    elif torch_info:
        print(f"  {'torch':15s}-> NOT importable\n"
              f"  {'':15s}   {torch_info.get('err', '')}")

    kernels_ok = cueq.get("ok") and (usage != "required" or trifast.get("ok"))
    if kernels_ok:
        print(f"\n  Kernels available: yes — run with no_kernels: false.")
    else:
        print(f"\n  Kernels NOT available in '{env}' — keep no_kernels: true "
              "(pure PyTorch) until this is fixed.")


def _status(args) -> int:
    tool = args.tool or _conda_tool()
    print(f"Boltz (kernels) environment: {args.env}")
    if not tool:
        print("  (no conda/mamba on PATH to inspect it)")
        return 0
    _print_report(_probe(tool, args.env), args.env)
    path = _env_bin(tool, args.env, "boltz")
    if path:
        print(f"\n  boltz_command: \"{path}\"\n  no_kernels: false")
    return 0


def _torch_ok(version: str) -> bool:
    """True if this torch is new enough for the legacy trifast kernel (>=2.6)."""
    try:
        major, minor = (int(p) for p in version.split(".")[:2])
    except ValueError:
        return True
    return (major, minor) >= (2, 6)


def add_boltz_env_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "boltz-env",
        help="Create a Boltz-2 env WITH the GPU kernels (separate from alchemfold)")
    commands = parser.add_subparsers(dest="boltz_env_command", required=True)

    install = commands.add_parser(
        "install", help="Create the env and install boltz[cuda] (kernels included)")
    install.add_argument("--env", default=_DEFAULT_ENV,
                         help=f"conda env name (default {_DEFAULT_ENV})")
    install.add_argument("--channel", default="conda-forge")
    install.add_argument("--python-version", default="3.11")
    install.add_argument("--cuda", default=None,
                         help="CUDA version for the GPU Torch build, e.g. 12.6. "
                              "Omit to take whatever `boltz[cuda]` resolves to "
                              "(currently a CUDA-13 Torch, which needs a recent "
                              "driver); pin it if the cluster driver is older.")
    install.add_argument("--recreate", action="store_true",
                         help="remove the env first and rebuild it cleanly")
    install.add_argument("--tool", default=None,
                         help="mamba | conda (default: mamba if present)")
    install.add_argument("--dry-run", action="store_true",
                         help="print the commands only")
    install.set_defaults(func=_install)

    status = commands.add_parser(
        "status", help="Show whether Boltz and its kernels are importable")
    status.add_argument("--env", default=_DEFAULT_ENV,
                        help=f"conda env name (default {_DEFAULT_ENV})")
    status.add_argument("--tool", default=None, help="mamba | conda")
    status.set_defaults(func=_status)
