"""Legacy ``src`` namespace retained for loading older ChemFlow artifacts."""

from pathlib import Path


_COMPATIBILITY_ROOT = Path(__file__).resolve().parent / "src"
if _COMPATIBILITY_ROOT.is_dir():
    __path__.append(str(_COMPATIBILITY_ROOT))
