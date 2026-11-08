"""Repair input structures before PROTAC docking: fill missing atoms/residues.

Two methods, matching the two situations:

* ``pdbfixer`` — PDBFixer (OpenMM) adds missing heavy atoms and short missing
  residues. Fast, but inserted loops are crude (roughly extended). Fine for
  missing atoms and short loops away from the interface.
* ``kic`` — Rosetta loop modeling (kinematic closure) via PyRosetta. Rigorous
  geometry for interface / binding-pocket loops. Slower. The loop residues must
  already exist in the pose, so the usual recipe is: ``pdbfixer`` to insert them
  (extended), then ``kic`` to rebuild them properly.

Gap detection and loop-spec parsing are pure; the PDBFixer / PyRosetta calls are
lazy imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"

_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL", "U": "SEC", "O": "PYL",
}
_THREE_TO_ONE = {v: k for k, v in _ONE_TO_THREE.items()}


@dataclass
class Gap:
    chain: str
    after_resseq: int       # last present residue before the gap
    before_resseq: int      # first present residue after the gap

    @property
    def length(self) -> int:
        return self.before_resseq - self.after_resseq - 1


def find_gaps(pdb_text: str) -> list[Gap]:
    """Internal chain breaks inferred from jumps in residue numbering.

    Only ``ATOM`` records are considered (protein), and only within a chain.
    Terminal missing residues cannot be seen from numbering, so are not reported.
    """
    last: dict[str, int] = {}
    seen_pairs: set[tuple[str, int]] = set()
    gaps: list[Gap] = []
    for line in pdb_text.splitlines():
        if line[:6].strip() != "ATOM":
            continue
        chain = line[21:22]
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        key = (chain, resseq)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if chain in last and resseq > last[chain] + 1:
            gaps.append(Gap(chain, last[chain], resseq))
        last[chain] = max(resseq, last.get(chain, resseq))
    return gaps


def parse_loop_spec(spec: str) -> tuple[str, int, int]:
    """``A:45-52`` -> ('A', 45, 52)."""
    if ":" not in spec or "-" not in spec.split(":", 1)[1]:
        raise ValueError(f"loop spec must be CHAIN:START-END, got {spec!r}")
    chain, rng = spec.split(":", 1)
    start_s, end_s = rng.split("-", 1)
    start, end = int(start_s), int(end_s)
    if not chain.strip() or start > end:
        raise ValueError(f"invalid loop spec: {spec!r}")
    return chain.strip(), start, end


# ---------------------------------------------------------------------------
# Reference sequences (FASTA file or UniProt), used to fill missing residues
# when the PDB has no SEQRES (e.g. a structure saved from PyMOL).
# ---------------------------------------------------------------------------

def read_fasta_sequence(path: str | Path) -> str:
    """First record's sequence from a FASTA/.fa file (letters only, upper-case)."""
    records = read_fasta_records(path)
    if not records:
        raise ValueError(f"no sequence found in {path}")
    return records[0][1]


def read_fasta_records(path: str | Path) -> list[tuple[str, str]]:
    """All (header, sequence) records from a FASTA file (sequences upper-cased)."""
    records: list[tuple[str, str]] = []
    header, seq = None, []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq).upper()))
            header, seq = line[1:].strip(), []
        elif line.strip():
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq).upper()))
    return [(h, s) for h, s in records if s]


def observed_residues(pdb_text: str, chain: str | None = None) -> tuple[str, list[tuple[int, str]]]:
    """(chain, [(resseq, one_letter), ...]) for protein residues, in order.

    Reads one CA per residue. If ``chain`` is None, uses the single chain present
    (errors if more than one).
    """
    chains: dict[str, list[tuple[int, str]]] = {}
    for line in pdb_text.splitlines():
        if line[:6].strip() != "ATOM" or line[12:16].strip() != "CA":
            continue
        ch = line[21:22]
        one = _THREE_TO_ONE.get(line[17:20].strip().upper(), "X")
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        chains.setdefault(ch, []).append((resseq, one))
    if chain is None:
        if len(chains) != 1:
            raise ValueError(f"structure has chains {sorted(chains)}; specify which one")
        chain = next(iter(chains))
    if chain not in chains:
        raise ValueError(f"chain {chain!r} not found (have {sorted(chains)})")
    return chain, chains[chain]


def best_offset(observed: list[tuple[int, str]], seq: str) -> tuple[int, int]:
    """Offset o (seq_index = resseq + o, 1-based) best matching observed to seq.

    Returns (offset, match_count). Tries every offset that keeps all observed
    residues inside the sequence.
    """
    if not observed:
        return 0, 0
    first = observed[0][0]
    last = observed[-1][0]
    best = (0, -1)
    for o in range(1 - first, len(seq) - last + 1):
        matches = sum(1 for r, aa in observed if seq[r + o - 1] == aa)
        if matches > best[1]:
            best = (o, matches)
    return best


def choose_sequence(candidates: list[str], observed: list[tuple[int, str]],
                    *, min_coverage: float = 0.8) -> tuple[str, int]:
    """Pick the candidate sequence + offset that best matches the observed residues."""
    scored = []
    for seq in candidates:
        if not seq:
            continue
        offset, matches = best_offset(observed, seq)
        scored.append((matches, seq, offset))
    if not scored:
        raise ValueError("no candidate sequences to align")
    matches, seq, offset = max(scored, key=lambda t: t[0])
    coverage = matches / len(observed) if observed else 0.0
    if coverage < min_coverage:
        raise ValueError(
            f"best sequence matches only {matches}/{len(observed)} observed residues "
            f"({coverage:.0%}); the FASTA/UniProt likely isn't this chain. "
            "Pick the record for this chain, or check the numbering.")
    return seq, offset



def uniprot_fasta_url(accession: str) -> str:
    return UNIPROT_FASTA.format(accession=accession.strip())


def fetch_uniprot_sequence(accession: str, *, timeout: float = 30.0) -> str:
    """Fetch a sequence from UniProt by accession (e.g. P12345)."""
    import urllib.request

    url = uniprot_fasta_url(accession)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ChemFlow/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"could not fetch UniProt {accession}: {exc}") from exc
    lines = [l for l in text.splitlines() if not l.startswith(">")]
    seq = "".join(lines).strip().upper()
    if not seq:
        raise ValueError(f"UniProt {accession} returned no sequence")
    return seq


def parse_chain_spec(spec: str, chains: list[str]) -> tuple[str, str]:
    """``CHAIN=VALUE`` -> (chain, value); bare ``VALUE`` allowed if one chain."""
    if "=" in spec:
        chain, value = spec.split("=", 1)
        return chain.strip(), value.strip()
    if len(chains) == 1:
        return chains[0], spec.strip()
    raise ValueError(
        f"{spec!r}: with multiple chains ({chains}), use CHAIN=VALUE (e.g. A={spec})")


def resolve_sequences(fasta_specs, uniprot_specs, chains: list[str],
                      sequence_specs=None) -> dict[str, str]:
    """Build ``{chain: one_letter_sequence}`` from --sequence, --fasta and --uniprot."""
    sequences: dict[str, str] = {}
    for spec in sequence_specs or []:
        chain, seq = parse_chain_spec(spec, chains)
        sequences[chain] = seq.upper().replace(" ", "")
    for spec in fasta_specs or []:
        chain, path = parse_chain_spec(spec, chains)
        sequences[chain] = read_fasta_sequence(path)
    for spec in uniprot_specs or []:
        chain, acc = parse_chain_spec(spec, chains)
        sequences[chain] = fetch_uniprot_sequence(acc)
    return sequences


def pdb_chains(pdb_text: str) -> list[str]:
    """Protein chain ids present (ATOM records), in order of first appearance."""
    seen = []
    for line in pdb_text.splitlines():
        if line[:6].strip() == "ATOM":
            chain = line[21:22]
            if chain not in seen:
                seen.append(chain)
    return seen


# ---------------------------------------------------------------------------
# PDBFixer
# ---------------------------------------------------------------------------

def fix_with_pdbfixer(
    input_pdb: str | Path,
    out_pdb: str | Path,
    *,
    add_missing_residues: bool = True,
    keep_only_internal: bool = True,
    max_gap: int | None = None,
    add_hydrogens: bool = False,
    ph: float = 7.0,
    sequences: dict[str, str] | None = None,
) -> Path:
    """Fill missing heavy atoms (and optionally short missing residues) with PDBFixer.

    ``keep_only_internal`` drops terminal missing-residue stretches (disordered
    tails) and ``max_gap`` skips internal gaps longer than N (crude loops).
    ``sequences`` ({chain: one-letter}) supplies the reference sequence when the
    PDB has no SEQRES (e.g. saved from PyMOL), so missing residues can be filled.
    """
    try:
        from openmm.app import PDBFile  # noqa: PLC0415
        from pdbfixer import PDBFixer  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "PDBFixer/OpenMM not installed in this environment. Install them with:\n"
            "  conda install -c conda-forge openmm pdbfixer\n"
            f"(original import error: {exc})") from exc

    input_pdb, out_pdb = Path(input_pdb), Path(out_pdb)
    fixer = PDBFixer(filename=str(input_pdb))
    if sequences:
        _apply_reference_sequences(fixer, sequences)
    fixer.findMissingResidues()
    if not add_missing_residues:
        fixer.missingResidues = {}
    else:
        chains = list(fixer.topology.chains())
        filtered = {}
        for (chain_idx, res_idx), residues in fixer.missingResidues.items():
            chain = chains[chain_idx]
            n_res = len(list(chain.residues()))
            is_terminal = res_idx == 0 or res_idx == n_res
            if keep_only_internal and is_terminal:
                continue
            if max_gap is not None and len(residues) > max_gap:
                continue
            filtered[(chain_idx, res_idx)] = residues
        fixer.missingResidues = filtered
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if add_hydrogens:
        fixer.addMissingHydrogens(ph)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with out_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle)
    return out_pdb


def _apply_reference_sequences(fixer, sequences: dict[str, str]) -> None:
    """Give PDBFixer a reference sequence per chain (when the PDB lacks SEQRES)."""
    from pdbfixer.pdbfixer import Sequence  # noqa: PLC0415

    chain_ids = [c.id for c in fixer.topology.chains()]
    built = []
    for chain_id, one_letter in sequences.items():
        if chain_id not in chain_ids:
            raise ValueError(f"sequence given for chain {chain_id!r}, not in the structure {chain_ids}")
        try:
            three = [_ONE_TO_THREE[a] for a in one_letter]
        except KeyError as exc:
            raise ValueError(f"unknown amino-acid letter {exc} in sequence for chain {chain_id}") from None
        built.append(Sequence(chain_id, three))
    fixer.sequences = built


# ---------------------------------------------------------------------------
# Rosetta loop modeling (KIC) via PyRosetta
# ---------------------------------------------------------------------------

def kic_refine(
    input_pdb: str | Path,
    out_pdb: str | Path,
    loops: list[tuple[str, int, int]],
    *,
    log=print,
) -> Path:
    """Rebuild the given loops with Rosetta KIC (loop residues must already exist).

    ``loops`` are ``(chain, start_resseq, end_resseq)`` in PDB numbering. Run
    PDBFixer first if the residues are missing. Not exercised in CI — verify the
    mover/API names against your PyRosetta version.
    """
    import pyrosetta  # noqa: PLC0415
    from pyrosetta.rosetta.protocols.loops import Loop, Loops, add_cutpoint_variants
    from pyrosetta.rosetta.protocols.loops.loop_closure.kinematic_closure import KinematicMover
    from pyrosetta.rosetta.protocols.loops.loop_mover.perturb import LoopMover_Perturb_KIC
    from pyrosetta.rosetta.protocols.loops.loop_mover.refine import LoopMover_Refine_KIC

    if not loops:
        raise ValueError("no loops given to kic_refine")
    pyrosetta.init("-mute all")
    pose = pyrosetta.pose_from_file(str(input_pdb))
    info = pose.pdb_info()

    rosetta_loops = Loops()
    for chain, start, end in loops:
        s = info.pdb2pose(chain, int(start))
        e = info.pdb2pose(chain, int(end))
        if s == 0 or e == 0:
            raise ValueError(f"loop {chain}:{start}-{end} not found in the pose")
        cut = (s + e) // 2
        rosetta_loops.add_loop(Loop(s, e, cut))
        log(f"KIC loop: {chain}:{start}-{end} -> pose {s}-{e} (cut {cut})")

    from pyrosetta.rosetta.protocols.loops import set_single_loop_fold_tree  # noqa: PLC0415
    for loop in rosetta_loops:
        set_single_loop_fold_tree(pose, loop)
    add_cutpoint_variants(pose)

    LoopMover_Perturb_KIC(rosetta_loops).apply(pose)
    LoopMover_Refine_KIC(rosetta_loops).apply(pose)

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    pose.dump_pdb(str(out_pdb))
    return out_pdb


def _contiguous_ranges(indices: set[int], total: int) -> list[tuple[int, int]]:
    """Missing index ranges within 1..total, given the set of present indices."""
    ranges: list[tuple[int, int]] = []
    start = None
    for i in range(1, total + 1):
        if i not in indices:
            start = i if start is None else start
        elif start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, total))
    return ranges


def kic_build_missing(
    input_pdb: str | Path,
    out_pdb: str | Path,
    candidate_sequences: list[str],
    *,
    chain: str | None = None,
    max_loop_length: int = 15,
    log=print,
) -> Path:
    """Build missing-density residues with KIC (pure PyRosetta, no PDBFixer).

    Aligns the structure's observed residues to the best-matching candidate
    sequence (handles multi-record FASTA and arbitrary numbering), models only the
    resolved span, and builds each short internal gap with KIC one loop at a time.
    Loops longer than ``max_loop_length`` are skipped (use co-folding). Single
    chain only. Experimental; run step by step.
    """
    import pyrosetta  # noqa: PLC0415
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.protocols.loops import (
        Loop, Loops, add_cutpoint_variants, fold_tree_from_loops,
        set_extended_torsions, set_single_loop_fold_tree,
    )
    from pyrosetta.rosetta.protocols.loops.loop_mover.perturb import LoopMover_Perturb_KIC
    from pyrosetta.rosetta.protocols.loops.loop_mover.refine import LoopMover_Refine_KIC
    from pyrosetta.rosetta.protocols.simple_moves import (
        ReturnSidechainMover, SwitchResidueTypeSetMover,
    )

    pdb_text = Path(input_pdb).read_text(encoding="utf-8", errors="replace")
    chain, observed = observed_residues(pdb_text, chain)
    seq, offset = choose_sequence(candidate_sequences, observed)
    first, last = observed[0][0], observed[-1][0]
    # sequence positions (1-based) spanned by the resolved residues
    seq_first, seq_last = first + offset, last + offset
    log(f"[1] chain {chain}: resolved residues {first}-{last} aligned to sequence "
        f"positions {seq_first}-{seq_last} (offset {offset}); modelling this span only")

    pyrosetta.init("-mute all")
    span_seq = seq[seq_first - 1:seq_last]
    full = pyrosetta.pose_from_sequence(span_seq)   # pose idx 1..len(span); pose_idx(r) = r - first + 1

    log("[2] threading known coordinates")
    exp = pyrosetta.pose_from_file(str(input_pdb))
    info = exp.pdb_info()
    present_idx: set[int] = set()
    for i in range(1, exp.total_residue() + 1):
        if info.chain(i) != chain:
            continue
        idx = info.number(i) - first + 1
        if idx < 1 or idx > full.total_residue():
            continue
        er, fr = exp.residue(i), full.residue(idx)
        for ai in range(1, er.natoms() + 1):
            name = er.atom_name(ai).strip()
            if fr.has(name):
                full.set_xyz(AtomID(fr.atom_index(name), idx), er.xyz(ai))
        present_idx.add(idx)
    log(f"    threaded {len(present_idx)}; {full.total_residue() - len(present_idx)} internal missing")

    # Collect the short internal gaps as loops (skip long ones — use co-folding).
    loops = Loops()
    loop_objs = []
    for start, end in _contiguous_ranges(present_idx, full.total_residue()):
        length = end - start + 1
        pdb_lo, pdb_hi = start + first - 1, end + first - 1
        if length > max_loop_length:
            log(f"[3] SKIP gap {pdb_lo}-{pdb_hi} ({length} residues): too long for KIC "
                "(use co-folding for this one)")
            continue
        loop_start, loop_end = start - 1, end + 1
        cut = (loop_start + loop_end) // 2
        loop = Loop(loop_start, loop_end, cut, 0.0, True)   # extended=True -> rebuild
        loops.add_loop(loop)
        loop_objs.append(loop)
        log(f"[3] gap {pdb_lo}-{pdb_hi} ({length} residues) -> loop pose {loop_start}-{loop_end} cut {cut}")

    if not loop_objs:
        log("[!] no loops built (all gaps too long or none internal). "
            "Consider co-folding the full protein with Boltz/Chai.")
    else:
        # Fold tree + cutpoints + idealised extended loops (use kept Python refs,
        # not iteration of the Rosetta Loops container).
        log("[3b] setting loop fold tree")
        if len(loop_objs) == 1:
            set_single_loop_fold_tree(full, loop_objs[0])
        else:
            ft = pyrosetta.rosetta.core.kinematics.FoldTree()
            fold_tree_from_loops(full, loops, ft)
            full.fold_tree(ft)
        log("[3c] adding cutpoint variants")
        add_cutpoint_variants(full)
        log("[3d] idealising loop(s) extended")
        for loop in loop_objs:
            set_extended_torsions(full, loop)

        starting_fa = full.clone()
        log(f"[4] KIC centroid build of {len(loop_objs)} loop(s)…")
        SwitchResidueTypeSetMover("centroid").apply(full)
        LoopMover_Perturb_KIC(loops).apply(full)
        log("[5] back to full atom; KIC refine…")
        SwitchResidueTypeSetMover("fa_standard").apply(full)
        ReturnSidechainMover(starting_fa).apply(full)   # restore non-loop sidechains
        LoopMover_Refine_KIC(loops).apply(full)
    built = len(loop_objs)

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    full.dump_pdb(str(out_pdb))
    log(f"[done] built {built} loop(s); wrote {out_pdb}")
    return out_pdb
