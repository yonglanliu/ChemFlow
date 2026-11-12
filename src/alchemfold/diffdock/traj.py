"""Clean DiffDock reverse-diffusion trajectories (``rank<k>_reverseprocess.pdb``).

DiffDock's trajectory writer has two quirks that make the raw file look broken in PyMOL / VMD:

1. **Frozen hydrogens.** Each frame holds ALL atoms of the ligand (heavy atoms + hydrogens), but
   DiffDock only moves the heavy atoms — the hydrogens keep the reference-conformer coordinates (near
   the origin) in every frame. Heavy atom -> hydrogen bonds then stretch ~50 A across the box, which
   draws as long streaks from the pocket out of the protein. Fix: drop hydrogens that never move.
2. **Frame order.** The frames are written as [reference conformer, original position, random start,
   FINAL pose (it overwrites the first denoising step), denoising steps 1..N-1]. Fix: reorder to
   random start -> steps -> final and drop the two reference frames.

Also writes the CONECT records once, after the last frame (standard multi-model PDB layout),
restricted to the atoms that remain. Pure standard library.
"""

from __future__ import annotations

import re
from pathlib import Path

_MODEL_RE = re.compile(r"^MODEL[^\n]*\n(.*?)^ENDMDL", re.S | re.M)


def split_frames(text: str) -> list[str]:
    return [m.group(1) for m in _MODEL_RE.finditer(text)]


def _atom_lines(frame: str) -> list[str]:
    return [ln for ln in frame.splitlines() if ln.startswith(("ATOM", "HETATM"))]


def _is_hydrogen(line: str) -> bool:
    element = line[76:78].strip()
    if not element:  # fall back to the atom name when the element column is empty
        element = line[12:16].strip().lstrip("0123456789")[:1]
    return element.upper() == "H"


def _xyz(line: str) -> tuple[str, str, str]:
    return line[30:38], line[38:46], line[46:54]


def clean_trajectory(text: str, *, reorder: bool = True, drop_frozen_h: bool = True) -> tuple[str, dict]:
    """Return ``(cleaned_text, info)``; ``info`` reports what was changed.

    Files that do not look like DiffDock output are only touched where it is clearly safe
    (hydrogens are dropped only if they do not move at all while the heavy atoms do).
    """
    frames = split_frames(text)
    info = {"frames_in": len(frames), "frames_out": len(frames), "dropped_hydrogens": 0,
            "reordered": False}
    if not frames:
        return text, info
    atoms = [_atom_lines(f) for f in frames]
    same_size = len({len(a) for a in atoms}) == 1

    order = list(range(len(frames)))
    if reorder and len(frames) >= 5 and "CONECT" in frames[0] and same_size:
        order = [2] + list(range(4, len(frames))) + [3]   # random start, steps 1..N-1, final
        info["reordered"] = True

    keep = list(range(len(atoms[0])))
    if drop_frozen_h and same_size:
        n = len(atoms[0])

        def moves(i: int) -> bool:
            return len({_xyz(a[i]) for a in atoms}) > 1

        heavy_moves = any(moves(i) for i in range(n) if not _is_hydrogen(atoms[0][i]))
        if heavy_moves:
            keep = [i for i in range(n) if not (_is_hydrogen(atoms[0][i]) and not moves(i))]
            info["dropped_hydrogens"] = n - len(keep)

    kept_serials = set()
    out: list[str] = []
    for out_idx, k in enumerate(order, start=1):
        lines = [atoms[k][i] for i in keep] if same_size else atoms[k]
        if out_idx == 1:
            kept_serials = {int(ln[6:11]) for ln in lines if ln[6:11].strip().isdigit()}
        out.append(f"MODEL {out_idx:>8d}\n" + "\n".join(lines) + "\nENDMDL\n")

    conect: list[str] = []
    for ln in text.splitlines():            # DiffDock puts them inside frame 0; a cleaned file after the last frame
        if ln.startswith("CONECT"):
            nums = [int(x) for x in ln.split()[1:] if x.isdigit()]
            if nums and nums[0] in kept_serials:
                partners = [x for x in nums[1:] if x in kept_serials]
                if partners:
                    conect.append("CONECT" + "".join(f"{x:5d}" for x in [nums[0], *partners]))
    info["frames_out"] = len(order)
    return "".join(out) + ("\n".join(conect) + "\n" if conect else "") + "END\n", info


def clean_file(src: str | Path, dst: str | Path | None = None) -> tuple[Path, dict]:
    """Clean ``src`` into ``dst`` (default ``<stem>_clean.pdb`` next to it)."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.stem + "_clean.pdb")
    cleaned, info = clean_trajectory(src.read_text(encoding="utf-8", errors="replace"))
    dst.write_text(cleaned, encoding="utf-8")
    return dst, info
