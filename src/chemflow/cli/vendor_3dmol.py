"""``chemflow vendor-3dmol`` — fetch 3Dmol-min.js for the offline desktop viewer.

The desktop binding-pose viewer (``Molecule3DView``) renders with 3Dmol.js. On an
HPC compute node there is usually no internet, so the CDN fallback fails and the
3D panel stays blank. Run this once on a node that HAS internet (e.g. a Biowulf
login / data-transfer node) to vendor the library into a cache the viewer loads
from offline::

    chemflow vendor-3dmol

It writes ``~/.cache/chemflow/3dmol/3Dmol-min.js`` by default (also discoverable
via ``CHEMFLOW_3DMOL_PATH``). Use ``--project`` to vendor it inside the repo
instead (``<repo>/.chemflow/vendor/3dmol/3Dmol-min.js``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Primary + fallback sources for the minified bundle.
_URLS = (
    "https://3Dmol.org/build/3Dmol-min.js",
    "https://cdn.jsdelivr.net/npm/3dmol/build/3Dmol-min.js",
)


def _default_dest() -> Path:
    return Path.home() / ".cache" / "chemflow" / "3dmol" / "3Dmol-min.js"


def _project_dest() -> Path | None:
    try:
        from chemflow.config import PROJECT_ROOT

        return Path(PROJECT_ROOT) / ".chemflow" / "vendor" / "3dmol" / "3Dmol-min.js"
    except Exception:
        return None


def _download(url: str, timeout: int = 60) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "chemflow-vendor-3dmol"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted CDN)
        return resp.read()


def _run(args) -> int:
    dest = Path(args.dest).expanduser() if args.dest else (
        _project_dest() if args.project else _default_dest()
    )
    if dest is None:
        print("error: could not resolve a destination (no project root).", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for url in _URLS:
        try:
            print(f"Downloading {url} ...")
            data = _download(url)
            if len(data) < 100_000:  # the real bundle is ~1-2 MB; guard against error pages
                raise ValueError(f"suspiciously small download ({len(data)} bytes)")
            dest.write_bytes(data)
            print(f"Wrote {dest} ({len(data) // 1024} KB)")
            print("The desktop binding-pose viewer will now render offline.")
            return 0
        except Exception as exc:  # try the next mirror
            last_err = exc
            print(f"  failed: {exc}", file=sys.stderr)
    print(
        "error: could not download 3Dmol-min.js from any source. Run this on a node "
        "with internet, or download the file manually and point CHEMFLOW_3DMOL_PATH at it.",
        file=sys.stderr,
    )
    print(f"  last error: {last_err}", file=sys.stderr)
    return 1


def add_vendor_3dmol_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "vendor-3dmol",
        help="Download 3Dmol-min.js for the offline desktop binding-pose viewer",
    )
    parser.add_argument("--dest", default=None,
                        help="Explicit output path for 3Dmol-min.js "
                             "(default: ~/.cache/chemflow/3dmol/3Dmol-min.js)")
    parser.add_argument("--project", action="store_true",
                        help="Vendor inside the repo (<repo>/.chemflow/vendor/3dmol/) "
                             "instead of the user cache")
    parser.set_defaults(func=_run)
