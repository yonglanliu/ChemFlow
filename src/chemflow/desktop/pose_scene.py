"""Pose-viewer scene logic (pure Python — no Qt, so it is unit-testable).

Different engines return different objects:

* **Boltz-2 / Chai-1** co-fold the protein *with* the ligand -> ONE complex file
  (PDB / mmCIF: protein ATOM + ligand HETATM).
* **DiffDock** only moves the ligand -> a ligand-only pose (``.sdf``) and the receptor stays the
  separate ``receptor_pdb`` you supplied; its optional diffusion trajectory is a multi-frame,
  ligand-only PDB.

This module classifies a file, builds the list of objects the 3D viewer renders, puts DiffDock
trajectory frames in chronological order, scans a job folder for results, and reads the scores.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

_WATER = {"HOH", "WAT", "H2O", "DOD"}
_IONS = {"NA", "CL", "K", "MG", "ZN", "CA", "MN", "FE", "CU", "CO", "NI", "CD", "HG", "SO4", "PO4"}
LIGAND_EXT = {".sdf", ".mol", ".mol2"}
PDB_EXT = {".pdb", ".ent"}
CIF_EXT = {".cif", ".mmcif", ".mcif"}

# Roles a loaded object can have in the scene.
COMPLEX, RECEPTOR, LIGAND, TRAJECTORY = "complex", "receptor", "ligand", "trajectory"


def fmt_for(path: str | Path) -> str:
    """3Dmol model format for a file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in CIF_EXT:
        return "mmcif"
    if suffix in {".sdf", ".mol"}:
        return "sdf"
    if suffix == ".mol2":
        return "mol2"
    if suffix == ".xyz":
        return "xyz"
    return "pdb"


def _scan_atoms(text: str, mmcif: bool) -> tuple[int, int, int]:
    """(protein-ish ATOM count, ligand HETATM count, MODEL count) for a PDB / mmCIF text."""
    atom = het = models = 0
    for line in text.splitlines():
        if mmcif:
            if line.startswith("ATOM "):
                atom += 1
            elif line.startswith("HETATM "):
                parts = line.split()
                resname = parts[5] if len(parts) > 5 else ""
                if resname not in _WATER and resname not in _IONS:
                    het += 1
            continue
        tag = line[:6]
        if tag.startswith("MODEL"):
            models += 1
        elif tag.startswith("ATOM"):
            atom += 1
        elif tag.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname not in _WATER and resname not in _IONS:
                het += 1
    return atom, het, models


def classify_text(text: str, suffix: str) -> str:
    """complex | receptor | ligand | trajectory for a structure given as text + extension."""
    suffix = suffix.lower()
    if suffix in LIGAND_EXT:
        return LIGAND
    atom, het, models = _scan_atoms(text, mmcif=suffix in CIF_EXT)
    if atom > 0 and het > 0:
        return COMPLEX
    if atom > 0:
        return RECEPTOR
    if het > 0:
        return TRAJECTORY if models >= 3 else LIGAND
    return COMPLEX  # unknown content: let the viewer try to render it as-is


def classify(path: str | Path) -> str:
    p = Path(path)
    return classify_text(p.read_text(encoding="utf-8", errors="replace"), p.suffix)


_MODEL_RE = re.compile(r"^MODEL.*?\n(.*?)^ENDMDL", re.S | re.M)


def split_frames(pdb_text: str) -> list[str]:
    """The MODEL ... ENDMDL blocks of a multi-model PDB (without the MODEL/ENDMDL lines)."""
    return [m.group(1) for m in _MODEL_RE.finditer(pdb_text)]


def join_frames(frames: list[str]) -> str:
    return "".join(f"MODEL {i + 1}\n{f.rstrip()}\nENDMDL\n" for i, f in enumerate(frames))


def diffdock_chronological(pdb_text: str) -> tuple[str, str]:
    """Reorder a DiffDock ``rank<k>_reverseprocess.pdb`` into time order.

    DiffDock writes: [0] reference conformer (the only frame with CONECT), [1] original conformer
    position, [2] random starting pose, [3] FINAL pose (it overwrites the first denoising step),
    [4..] denoising steps 1..N-1. Played as written the animation jumps start -> final -> steps.
    Chronological order is ``[2] + [4..] + [3]`` and the two reference frames are dropped.
    Returns ``(text, note)``; files that do not look like DiffDock's layout are returned unchanged.
    """
    frames = split_frames(pdb_text)
    same_size = len({sum(1 for ln in f.splitlines() if ln.startswith(("HETATM", "ATOM"))) for f in frames}) == 1
    if len(frames) >= 5 and "CONECT" in frames[0] and same_size:
        ordered = [frames[2]] + frames[4:] + [frames[3]]
        return join_frames(ordered), (f"DiffDock trajectory: {len(ordered)} frames, random start -> final pose "
                                      "(reference frames dropped, order fixed)")
    return pdb_text, f"{len(frames)} frame(s), shown in file order" if frames else ""


def _load_cleaner():
    """alchemfold's trajectory cleaner.

    Imported normally when alchemfold is on the path, else loaded by file path
    so the viewer still works from a bare repo checkout.
    """
    import importlib.util

    try:
        from alchemfold.diffdock.traj import clean_trajectory

        return clean_trajectory
    except Exception:
        pass

    src = Path(__file__).resolve().parents[2] / "alchemfold" / "diffdock" / "traj.py"
    if not src.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_alchemfold_diffdock_traj", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod.clean_trajectory


def read_object(path: str | Path, role: str | None = None) -> dict:
    """Load a file into a scene object ``{data, fmt, role, name}``."""
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8", errors="replace")
    kind = role or classify_text(text, p.suffix)
    if kind == TRAJECTORY and p.suffix.lower() in PDB_EXT | {""}:
        cleaner = _load_cleaner()
        if cleaner is not None:
            text, _ = cleaner(text)          # chronological order + frozen hydrogens removed
        else:
            text, _ = diffdock_chronological(text)
    return {"data": text, "fmt": fmt_for(p), "role": kind, "name": p.name}


def build_scene(pose: str | None = None, receptor: str | None = None,
                trajectory: str | None = None) -> tuple[list[dict], list[str]]:
    """Objects to render + human-readable notes, from whichever files the user has.

    * a complex (Boltz/Chai)              -> [complex]
    * a ligand pose + a receptor          -> [receptor, ligand]
    * a trajectory (+ pose) + a receptor  -> [receptor, ligand pose, trajectory]
    A ligand-only file with no receptor is still shown (alone) with a note.
    """
    notes: list[str] = []
    rec_obj = None
    if receptor:
        if Path(receptor).expanduser().is_file():
            rec_obj = read_object(receptor, RECEPTOR)
        else:
            notes.append(f"Receptor not found: {receptor}")

    shown: list[dict] = []
    for path, forced in ((pose, None), (trajectory, TRAJECTORY)):
        if not path:
            continue
        if not Path(path).expanduser().is_file():
            notes.append(f"File not found: {path}")
            continue
        shown.append(read_object(path, forced))

    has_complex = any(o["role"] == COMPLEX for o in shown)
    for o in shown:
        if o["role"] in (LIGAND, TRAJECTORY) and rec_obj is None:
            notes.append(f"{o['name']} is ligand-only — choose the receptor PDB it was docked into "
                         "to see it in the pocket.")
        if o["role"] == TRAJECTORY:
            n = len(split_frames(o["data"]))
            notes.append(f"Trajectory: {n} frame(s) — use Play or the slider." if n else
                         "Trajectory file has no MODEL frames.")
    if rec_obj is not None and has_complex:
        notes.append("A complex was given, so the separate receptor is not drawn.")
        rec_obj = None
    objects = ([rec_obj] if rec_obj is not None else []) + shown
    return objects, notes


# ------------------------------------------------------------------ job-folder scan
def _read_yaml_value(path: Path, key: str) -> str:
    """Top-level ``key: value`` from a small YAML file (stdlib-only; no nesting needed)."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(rf'\s*{re.escape(key)}\s*:\s*"?([^"#\n]+?)"?\s*(#.*)?$', line)
            if m:
                return m.group(1).strip()
    except OSError:
        pass
    return ""


def _job_receptor(job: Path) -> str:
    for name in ("screen.yaml", "docking.yaml"):
        v = _read_yaml_value(job / name, "receptor_pdb")
        if v:
            return v
    return ""


def _fnum(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def scan_results(job_dir: str | Path) -> list[dict]:
    """Find poses in a job folder: ``[{label, pose, receptor, trajectory, scores}]`` (best first).

    Looks (in order) at ``ranking.csv`` / ``results.csv`` / ``results/shard_*.csv`` from screens, then
    at ``predictions/`` of single dockings (Boltz/Chai ``*_model_0.pdb|cif``; DiffDock
    ``rank1*.sdf`` + ``rank1_reverseprocess.pdb``).
    """
    job = Path(job_dir).expanduser()
    receptor = _job_receptor(job)
    out: list[dict] = []

    tables = [job / "ranking.csv", job / "results.csv", *sorted((job / "results").glob("shard_*.csv"))]
    for table in tables:
        if not table.is_file():
            continue
        with table.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pose = row.get("pose") or row.get("structure") or ""
                if not pose or (row.get("status") not in (None, "", "ok")):
                    continue
                scores = {k: _fnum(v) for k, v in row.items()
                          if _fnum(v) is not None and k not in ("rank",)}
                label = str(row.get("ligand_id") or Path(pose).stem)
                if scores.get("rank") is None and row.get("rank"):
                    label = f"#{row['rank']} {label}"
                out.append({"label": label, "pose": pose, "receptor": receptor,
                            "trajectory": row.get("trajectory") or "", "scores": scores})
        if out:
            return out  # the ranking (or the first non-empty table) wins

    pred = job / "predictions"
    if pred.is_dir():
        # Boltz / Chai: one complex per prediction folder
        for cplx in sorted(list(pred.rglob("*_model_0.pdb")) + list(pred.rglob("*_model_0.cif"))
                           + list(pred.rglob("pred.model_idx_0.cif"))):
            out.append({"label": cplx.parent.name or cplx.stem, "pose": str(cplx), "receptor": "",
                        "trajectory": "", "scores": {}})
        # DiffDock: per-complex folder with ranked ligand SDFs
        for d in sorted(p for p in pred.iterdir() if p.is_dir()):
            ranked = sorted(d.glob("rank1_confidence*.sdf")) or sorted(d.glob("rank1.sdf"))
            if ranked:
                m = re.search(r"confidence(-?\d+\.?\d*)", ranked[0].name)
                traj = d / "rank1_reverseprocess.pdb"
                out.append({"label": d.name, "pose": str(ranked[0]), "receptor": receptor,
                            "trajectory": str(traj) if traj.is_file() else "",
                            "scores": {"confidence": float(m.group(1))} if m else {}})
    return out


# ------------------------------------------------------------------ scores
def diffdock_confidence_text(c: float) -> str:
    """DiffDock README guide for the TOP pose's confidence (a raw, unbounded model logit)."""
    level = "high" if c > 0 else ("moderate" if c > -1.5 else "low")
    return (f"DiffDock confidence {c:.2f} -> {level} confidence "
            "(guide: > 0 high, -1.5..0 moderate, < -1.5 low; shift down for large ligands / apo "
            "structures). It scores the POSE, not the binding affinity.")


def confidence_from_name(path: str | Path) -> float | None:
    m = re.search(r"confidence(-?\d+\.?\d*)", Path(path).name)
    return float(m.group(1)) if m else None


def json_scores(structure_path: str | Path, keys: set[str]) -> dict[str, float]:
    """Numeric scores from JSONs beside a Boltz/Chai structure (``keys`` = names to keep)."""
    start = Path(structure_path).expanduser()
    found: dict[str, float] = {}
    for d in (start.parent, start.parent.parent):
        if not d.is_dir():
            continue
        for jf in sorted(d.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                for k, v in data.items():
                    if str(k).lower() in keys and isinstance(v, (int, float)) and str(k).lower() not in found:
                        found[str(k).lower()] = float(v)
    return found


# ------------------------------------------------------------------ viewer JavaScript
# Rendered by 3Dmol.js inside the QtWebEngine page (molecule3d_view.py). Kept here, as a string,
# so it can be exercised under Node with a mock $3Dmol (no Qt needed).
VIEWER_JS = r"""
var _viewer=null, _traj=null, _nFrames=0;
function _ensure(){
  if(typeof $3Dmol==='undefined'){return null;}
  if(!_viewer){_viewer=$3Dmol.createViewer('viewer',{backgroundColor:'0x0b1622'});}
  return _viewer;
}
window.showMessage=function(t){var m=document.getElementById('msg');if(m){m.textContent=t;m.style.display='block';}};
window.clearModel=function(){
  var v=_ensure(); if(v){ if(v.pauseAnimate){v.pauseAnimate();} v.clear(); v.render(); }
  _traj=null; _nFrames=0;
};
// objs: [{data, fmt, role}] with role = complex | receptor | ligand | trajectory.
//  complex    protein cartoon + ligand sticks (Boltz / Chai: one file)
//  receptor   protein cartoon (DiffDock: the separate receptor PDB)
//  ligand     sticks (a docked pose, e.g. DiffDock's SDF)
//  trajectory multi-frame ligand-only PDB -> animated sticks (DiffDock reverse diffusion)
window.renderScene=function(objs, autoplay){
  var v=_ensure();
  if(!v){window.showMessage('3Dmol.js not found. Run  chemflow vendor-3dmol  on a node with internet '+
    '(or set CHEMFLOW_3DMOL_PATH), then reopen this page.');return -1;}
  var msg=document.getElementById('msg'); if(msg){msg.style.display='none';}
  if(v.pauseAnimate){v.pauseAnimate();}
  v.clear(); _traj=null; _nFrames=0;
  var items=[];
  try{
    for(var i=0;i<objs.length;i++){
      var o=objs[i], m;
      if(o.role==='trajectory'){
        m=v.addModelsAsFrames(o.data,o.fmt);
        _traj=m; _nFrames=(m.getNumFrames?m.getNumFrames():(v.getNumFrames?v.getNumFrames():0));
      } else { m=v.addModel(o.data,o.fmt); }
      items.push({id:m.getID(), role:o.role});
    }
  }catch(e){window.showMessage('Parse error: '+e);return -1;}
  var protId=null, ligId=null;
  for(var j=0;j<items.length;j++){
    var it=items[j];
    if(it.role==='complex'){
      v.setStyle({model:it.id},{cartoon:{color:'spectrum'}});
      v.setStyle({model:it.id,hetflag:true},{stick:{colorscheme:'greenCarbon',radius:0.18}});
      v.addStyle({model:it.id,hetflag:true},{sphere:{scale:0.28}});
      if(protId===null){protId=it.id;}
    } else if(it.role==='receptor'){
      v.setStyle({model:it.id},{cartoon:{color:'spectrum',opacity:0.9}});
      if(protId===null){protId=it.id;}
    } else if(it.role==='ligand'){
      v.setStyle({model:it.id},{stick:{colorscheme:'greenCarbon',radius:0.2}});
      if(ligId===null){ligId=it.id;}
    } else if(it.role==='trajectory'){
      v.setStyle({model:it.id},{stick:{colorscheme:'cyanCarbon',radius:0.15}});
      if(ligId===null){ligId=it.id;}
    }
  }
  // show the pocket: residues within 5 A of the ligand as thin sticks
  if(protId!==null && ligId!==null){
    try{v.addStyle({model:protId,within:{distance:5,sel:{model:ligId}}},
                   {stick:{colorscheme:'whiteCarbon',radius:0.1}});}catch(e){}
  }
  try{ if(ligId!==null){v.zoomTo({model:ligId});} else {v.zoomTo({hetflag:true});} }catch(e){ v.zoomTo(); }
  v.render();
  if(_traj && _nFrames>1 && autoplay){ window.playTraj(); }
  return _nFrames;
};
window.playTraj=function(){ if(_viewer&&_traj){ _viewer.animate({interval:150,loop:'forward',reps:0}); } };
window.pauseTraj=function(){ if(_viewer&&_viewer.pauseAnimate){ _viewer.pauseAnimate(); } };
window.setTrajFrame=function(i){ if(_viewer&&_traj){ _viewer.pauseAnimate(); _viewer.setFrame(i); _viewer.render(); } };
window.trajFrames=function(){ return _nFrames; };
"""
