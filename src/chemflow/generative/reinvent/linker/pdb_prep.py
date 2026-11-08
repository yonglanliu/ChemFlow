"""Download a structure from the PDB and split it into protein + ligand files,
ready to load into the linker designer.

A PROTAC ternary complex gives you correctly-posed warheads in one coordinate
frame, which is what the 3D pocket-fit check needs. This module:

* fetches ``<ID>.pdb`` from RCSB (``fetch_pdb``),
* lists the non-water HETATM ligands so you can see what is there (``list_ligands``),
* writes the protein to ``protein.pdb`` (``write_protein``),
* extracts a chosen ligand to its own SDF (``extract_ligand``), using the PDB
  chemical component's *ideal* SDF as a bond-order template when available
  (PDB coordinate records carry no bond orders).

Network and RDKit are used lazily, so importing this module is cheap and the
list/parse helpers work without RDKit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RCSB_STRUCTURE = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_LIGAND_IDEAL = "https://files.rcsb.org/ligands/download/{resname}_ideal.sdf"

# Residues that are almost never the ligand of interest.
WATERS = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SPC"}
COMMON_IONS = {
    "NA", "CL", "K", "MG", "ZN", "CA", "MN", "FE", "CU", "NI", "CO", "CD",
    "SO4", "PO4", "NO3", "ACT", "EDO", "GOL", "PEG", "DMS", "IOD", "BR", "FMT",
}


@dataclass
class LigandRecord:
    resname: str
    chain: str
    resseq: str
    n_atoms: int

    @property
    def label(self) -> str:
        return f"{self.resname} (chain {self.chain}, res {self.resseq}, {self.n_atoms} atoms)"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ChemFlow/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (trusted RCSB)
        data = response.read()
    if not data:
        raise RuntimeError(f"empty download from {url}")
    dest.write_bytes(data)
    return dest


def fetch_pdb(pdb_id: str, dest_dir: str | Path, *, timeout: float = 60.0) -> Path:
    """Download ``<ID>.pdb`` from RCSB into ``dest_dir``."""
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError(f"not a 4-character PDB id: {pdb_id!r}")
    dest = Path(dest_dir) / f"{pdb_id}.pdb"
    try:
        return _download(RCSB_STRUCTURE.format(pdb_id=pdb_id), dest, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"could not download {pdb_id} from RCSB: {exc}") from exc


def fetch_ligand_template(resname: str, dest_dir: str | Path, *, timeout: float = 60.0) -> Path | None:
    """Download the chemical component's ideal SDF (bond-order template), or None."""
    resname = resname.strip().upper()
    dest = Path(dest_dir) / f"{resname}_ideal.sdf"
    try:
        return _download(RCSB_LIGAND_IDEAL.format(resname=resname), dest, timeout=timeout)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parsing (pure Python)
# ---------------------------------------------------------------------------

def _hetatm_key(line: str) -> tuple[str, str, str]:
    resname = line[17:20].strip()
    chain = line[21:22].strip()
    resseq = line[22:26].strip() + line[26:27].strip()  # include insertion code
    return resname, chain, resseq


def list_ligands(pdb_path: str | Path, *, include_ions: bool = False) -> list[LigandRecord]:
    """Non-water HETATM ligands, largest first (the PROTAC is usually biggest)."""
    counts: dict[tuple[str, str, str], int] = {}
    with Path(pdb_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            resname, chain, resseq = _hetatm_key(line)
            if resname in WATERS:
                continue
            if not include_ions and resname in COMMON_IONS:
                continue
            counts[(resname, chain, resseq)] = counts.get((resname, chain, resseq), 0) + 1
    records = [LigandRecord(r, c, s, n) for (r, c, s), n in counts.items()]
    records.sort(key=lambda rec: rec.n_atoms, reverse=True)
    return records


def write_protein(pdb_path: str | Path, out: str | Path, *, keep_hetatm: bool = False) -> Path:
    """Write protein atoms to ``out`` (ATOM records; waters always dropped)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with Path(pdb_path).open("r", encoding="utf-8", errors="replace") as src, \
            out.open("w", encoding="utf-8") as dst:
        for line in src:
            record = line[:6].strip()
            if record == "ATOM" or (keep_hetatm and record == "HETATM"
                                    and line[17:20].strip() not in WATERS):
                dst.write(line)
                kept += 1
            elif record == "TER":
                dst.write(line)
        dst.write("END\n")
    if kept == 0:
        raise ValueError(f"no protein ATOM records found in {pdb_path}")
    return out


def _ligand_pdb_block(pdb_path: str | Path, resname: str, chain: str | None, resseq: str | None) -> str:
    resname = resname.strip().upper()
    lines = []
    with Path(pdb_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            r, c, s = _hetatm_key(line)
            if r.upper() != resname:
                continue
            if chain is not None and c != chain:
                continue
            if resseq is not None and s != str(resseq):
                continue
            lines.append(line)
    if not lines:
        raise ValueError(f"ligand {resname!r} not found in {pdb_path}"
                         + (f" (chain {chain})" if chain else ""))
    return "".join(lines) + "END\n"


# ---------------------------------------------------------------------------
# Ligand extraction (RDKit)
# ---------------------------------------------------------------------------

def extract_ligand(
    pdb_path: str | Path,
    resname: str,
    out_dir: str | Path,
    *,
    chain: str | None = None,
    resseq: str | None = None,
    template_sdf: str | Path | None = None,
) -> Path:
    """Extract one ligand to ``<resname>.sdf`` with 3D coords.

    PDB coordinates have no bond orders. If ``template_sdf`` (the component's
    ideal SDF) is given, bond orders are transferred onto the extracted pose with
    ``AssignBondOrdersFromTemplate``; otherwise RDKit guesses by proximity and the
    result should be checked.
    """
    from rdkit import Chem

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    block = _ligand_pdb_block(pdb_path, resname, chain, resseq)

    mol = Chem.MolFromPDBBlock(block, removeHs=False, proximityBonding=True)
    if mol is None:
        raise ValueError(f"RDKit could not read ligand {resname} from the PDB coordinates")

    assigned = False
    if template_sdf and Path(template_sdf).is_file():
        try:
            from rdkit.Chem import AllChem

            template = next(iter(Chem.SDMolSupplier(str(template_sdf), removeHs=False)), None)
            if template is not None:
                mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
                assigned = True
        except Exception:
            assigned = False

    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        pass
    out_sdf = out_dir / f"{resname.upper()}.sdf"
    name = f"{resname.upper()}{'' if assigned else ' (bond orders guessed - verify)'}"
    writer = Chem.SDWriter(str(out_sdf))
    try:
        mol.SetProp("_Name", name)
        writer.write(mol)
    finally:
        writer.close()
    return out_sdf


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def prepare_complex(
    *,
    pdb_id: str | None = None,
    pdb_file: str | Path | None = None,
    out_dir: str | Path,
    ligands: list[str] | None = None,
    keep_hetatm: bool = False,
    use_templates: bool = True,
) -> dict:
    """Download/parse a complex; write protein.pdb and the chosen ligand SDFs.

    ``ligands`` entries are ``RESNAME`` or ``RESNAME:CHAIN``. Returns
    ``{"pdb": ..., "protein": ..., "ligands": [paths], "available": [LigandRecord]}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pdb_file:
        pdb_path = Path(pdb_file)
        if not pdb_path.is_file():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    elif pdb_id:
        pdb_path = fetch_pdb(pdb_id, out_dir)
    else:
        raise ValueError("give either pdb_id or pdb_file")

    available = list_ligands(pdb_path)
    result = {"pdb": pdb_path, "available": available, "protein": None, "ligands": []}
    if not ligands:
        return result

    result["protein"] = write_protein(pdb_path, out_dir / "protein.pdb", keep_hetatm=keep_hetatm)
    for spec in ligands:
        resname, _, chain = spec.partition(":")
        template = fetch_ligand_template(resname, out_dir) if use_templates else None
        path = extract_ligand(pdb_path, resname, out_dir, chain=chain or None, template_sdf=template)
        result["ligands"].append(path)
    return result
