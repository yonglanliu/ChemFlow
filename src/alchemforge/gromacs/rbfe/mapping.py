"""OpenFE-powered ligand network planning for GROMACS RBFE projects."""

from __future__ import annotations

import csv
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import LigandRBFEConfig


@dataclass(frozen=True)
class NetworkPlan:
    ligand_count: int
    edge_count: int
    graphml: Path
    edges_csv: Path


@dataclass(frozen=True)
class LigandNetworkSettings:
    """Ligand-only settings shared by OpenFE and GROMACS workflows."""

    project_dir: Path
    ligand_library: Path
    network_method: str = "lomap"
    atom_mapper: str = "lomap"
    max3d_angstrom: float = 1.0
    element_change: bool = False
    require_cycle_covering: bool = True
    radial_hub: str | None = None
    redundant_networks: int = 2
    pmx_command: str = "pmx"
    pmx_mapping_mode: str = "stage"
    pmx_h2h_polar: bool = True


def existing_network_plan(config: LigandRBFEConfig) -> NetworkPlan:
    """Load a previously planned or desktop-edited network."""
    from rdkit import Chem

    network_dir = config.project_dir / "network"
    edges_csv = network_dir / "edges.csv"
    graphml = network_dir / "ligand_network.graphml"
    if not edges_csv.is_file():
        raise ValueError(f"Existing network edge table was not found: {edges_csv}")
    with edges_csv.open(newline="", encoding="utf-8") as handle:
        edges = list(csv.DictReader(handle))
    if not edges:
        raise ValueError("Existing network contains no transformations")
    edges = normalize_edge_ids(network_dir, edges)
    ligands = {
        molecule.GetProp("_Name").strip()
        for molecule in Chem.SDMolSupplier(str(config.ligand_library), removeHs=False)
        if molecule is not None and molecule.HasProp("_Name")
    }
    return NetworkPlan(len(ligands), len(edges), graphml, edges_csv)


def _safe_name(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not result:
        raise ValueError(f"Invalid ligand name: {name!r}")
    return result


def readable_edge_id(
    ligand_a: str, ligand_b: str, used_ids: set[str] | None = None
) -> str:
    """Build a stable, filesystem-safe edge name from both ligand names."""
    left = _safe_name(ligand_a)
    right = _safe_name(ligand_b)
    base = f"edge_{left}_to_{right}"
    if len(base.encode("utf-8")) > 240:
        import hashlib

        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:10]
        base = f"edge_{left[:100]}_to_{right[:100]}_{digest}"
    used = used_ids or set()
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def normalize_edge_ids(
    network_dir: Path, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Rename legacy numbered network artifacts to ligand-based edge IDs."""
    used: set[str] = set()
    renamed: list[dict[str, str]] = []
    replacements: dict[str, str] = {}
    for row in rows:
        updated = dict(row)
        old = row.get("edge_id", "").strip()
        new = readable_edge_id(row["ligand_a"], row["ligand_b"], used)
        used.add(new)
        updated["edge_id"] = new
        renamed.append(updated)
        if old and old != new:
            replacements[old] = new
    if not replacements:
        return renamed

    mappings = network_dir / "mappings"
    for old, new in replacements.items():
        for source in (mappings / f"{old}.json", mappings / old):
            if not source.exists():
                continue
            destination = source.with_name(
                f"{new}.json" if source.suffix == ".json" else new
            )
            if destination.exists():
                raise ValueError(
                    f"Cannot rename legacy mapping {source.name}: "
                    f"destination already exists: {destination.name}"
                )
            source.rename(destination)
            if destination.suffix == ".json":
                text = destination.read_text(encoding="utf-8")
                for previous, replacement in replacements.items():
                    text = text.replace(previous, replacement)
                destination.write_text(text, encoding="utf-8")

    script = network_dir / "run_pmx_mappings.sh"
    if script.is_file():
        text = script.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        script.write_text(text, encoding="utf-8")

    edges_csv = network_dir / "edges.csv"
    fields = list(rows[0])
    with edges_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(renamed)
    return renamed


def plan_network(config: LigandRBFEConfig | LigandNetworkSettings) -> NetworkPlan:
    """Create a network plus auditable, zero-based atom mappings for its edges."""
    import openfe
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(config.ligand_library), removeHs=False)
    rdkit_ligands = []
    names: set[str] = set()
    invalid_records = []
    for index, molecule in enumerate(supplier, start=1):
        if molecule is None:
            invalid_records.append(index)
            continue
        name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
        if not name:
            raise ValueError(f"Ligand record {index} has no _Name")
        if name in names:
            raise ValueError(f"Duplicate ligand _Name in SDF: {name}")
        names.add(name)
        rdkit_ligands.append((name, molecule))
    if invalid_records:
        raise ValueError(
            "RDKit could not read ligand record(s): "
            + ", ".join(map(str, invalid_records))
        )
    if len(rdkit_ligands) < 2:
        raise ValueError("The ligand SDF must contain at least two valid molecules")

    ligands = [
        openfe.SmallMoleculeComponent.from_rdkit(molecule)
        for _, molecule in rdkit_ligands
    ]
    if config.atom_mapper in {"lomap", "pmx"}:
        mapper = openfe.setup.LomapAtomMapper(
            threed=True,
            max3d=config.max3d_angstrom,
            element_change=config.element_change,
        )
    else:
        mapper = openfe.setup.KartografAtomMapper(
            atom_max_distance=config.max3d_angstrom
        )
    scorer = openfe.lomap_scorers.default_lomap_score
    planners = openfe.ligand_network_planning
    if config.network_method == "lomap":
        network = planners.generate_lomap_network(
            ligands=ligands,
            mappers=[mapper],
            scorer=scorer,
            require_cycle_covering=config.require_cycle_covering,
        )
    elif config.network_method == "minimal_spanning":
        network = planners.generate_minimal_spanning_network(
            ligands=ligands, mappers=[mapper], scorer=scorer
        )
    elif config.network_method == "minimal_redundant":
        network = planners.generate_minimal_redundant_network(
            ligands=ligands,
            mappers=[mapper],
            scorer=scorer,
            mst_num=config.redundant_networks,
        )
    else:
        by_name = {ligand.name: ligand for ligand in ligands}
        if config.radial_hub not in by_name:
            raise ValueError(
                f"Radial hub {config.radial_hub!r} is not a ligand _Name in the SDF"
            )
        hub = by_name[config.radial_hub]
        network = planners.generate_radial_network(
            ligands=[ligand for ligand in ligands if ligand is not hub],
            central_ligand=hub,
            mappers=[mapper],
            scorer=scorer,
        )

    network_dir = config.project_dir / "network"
    mappings_dir = network_dir / "mappings"
    ligand_dir = config.project_dir / "input" / "ligands"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    ligand_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.ligand_library, network_dir / "ligands.sdf")
    graphml = network_dir / "ligand_network.graphml"
    graphml.write_text(network.to_graphml(), encoding="utf-8")

    for name, molecule in rdkit_ligands:
        safe_name = _safe_name(name)
        writer = Chem.SDWriter(str(ligand_dir / f"{safe_name}.sdf"))
        writer.write(molecule)
        writer.close()
        Chem.MolToPDBFile(molecule, str(ligand_dir / f"{safe_name}.pdb"))

    pmx_script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by AlchemForge. OpenFE/LoMap selected the network edges;",
        "# PMX atomMapping generates the final mapping for each selected pair.",
    ]

    ordered_edges = sorted(
        network.edges, key=lambda edge: (edge.componentA.name, edge.componentB.name)
    )
    edges_csv = network_dir / "edges.csv"
    with edges_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "edge_id",
                "ligand_a",
                "ligand_b",
                "mapped_atoms",
                "lomap_score",
            ),
        )
        writer.writeheader()
        used_edge_ids: set[str] = set()
        for edge in ordered_edges:
            edge_id = readable_edge_id(
                edge.componentA.name, edge.componentB.name, used_edge_ids
            )
            used_edge_ids.add(edge_id)
            edge_dir = config.project_dir / "edges" / edge_id
            edge_dir.mkdir(parents=True, exist_ok=True)
            atom_mapping = {
                int(atom_a): int(atom_b)
                for atom_a, atom_b in edge.componentA_to_componentB.items()
            }
            score = float(scorer(edge))
            mapped_atoms: int | str = len(atom_mapping)
            if config.atom_mapper == "pmx":
                mapping_dir = mappings_dir / edge_id
                mapping_dir.mkdir(parents=True, exist_ok=True)
                ligand_a = ligand_dir / f"{_safe_name(edge.componentA.name)}.pdb"
                ligand_b = ligand_dir / f"{_safe_name(edge.componentB.name)}.pdb"
                command = [
                    *shlex.split(config.pmx_command),
                    "atomMapping",
                    "-i1",
                    str(ligand_a),
                    "-i2",
                    str(ligand_b),
                    "-o1",
                    str(mapping_dir / "pairs1.dat"),
                    "-o2",
                    str(mapping_dir / "pairs2.dat"),
                ]
                if config.pmx_h2h_polar:
                    command.append("--H2Hpolar")
                command.extend(
                    [
                        "-opdbm1",
                        str(mapping_dir / "mapped1.pdb"),
                        "-opdbm2",
                        str(mapping_dir / "mapped2.pdb"),
                    ]
                )
                pmx_script_lines.extend(["", f"# {edge_id}", shlex.join(command)])
                if config.pmx_mapping_mode == "run":
                    try:
                        subprocess.run(
                            command,
                            check=True,
                            text=True,
                            capture_output=True,
                        )
                    except FileNotFoundError as error:
                        raise ValueError(
                            f"PMX command was not found: {command[0]!r}. "
                            "Choose the stage action to generate commands for later."
                        ) from error
                    except subprocess.CalledProcessError as error:
                        detail = (error.stderr or error.stdout or "").strip()
                        raise ValueError(
                            f"PMX atomMapping failed for {edge_id}: {detail}"
                        ) from error
                    pairs = mapping_dir / "pairs1.dat"
                    mapped_atoms = sum(
                        1
                        for line in pairs.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.lstrip().startswith(("#", ";"))
                    )
                else:
                    mapped_atoms = "pending"
                payload = {
                    "edge_id": edge_id,
                    "ligand_a": edge.componentA.name,
                    "ligand_b": edge.componentB.name,
                    "mapper": "pmx atomMapping",
                    "mapping_status": config.pmx_mapping_mode,
                    "command": command,
                    "pairs1": str(mapping_dir / "pairs1.dat"),
                    "pairs2": str(mapping_dir / "pairs2.dat"),
                    "mapped1": str(mapping_dir / "mapped1.pdb"),
                    "mapped2": str(mapping_dir / "mapped2.pdb"),
                    "lomap_network_score": score,
                    "note": "LoMap selected/scored this edge; PMX supplies its final atom mapping.",
                }
            else:
                payload = {
                    "edge_id": edge_id,
                    "ligand_a": edge.componentA.name,
                    "ligand_b": edge.componentB.name,
                    "mapper": config.atom_mapper,
                    "index_base": 0,
                    "componentA_to_componentB": atom_mapping,
                    "lomap_score": score,
                    "note": (
                        "OpenFE/RDKit atom indices. Verify atom ordering before "
                        "using this mapping in a PMX or GROMACS topology."
                    ),
                }
            writer.writerow(
                {
                    "edge_id": edge_id,
                    "ligand_a": edge.componentA.name,
                    "ligand_b": edge.componentB.name,
                    "mapped_atoms": mapped_atoms,
                    "lomap_score": f"{score:.6f}",
                }
            )
            (mappings_dir / f"{edge_id}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )

    if config.atom_mapper == "pmx":
        pmx_script = network_dir / "run_pmx_mappings.sh"
        pmx_script.write_text("\n".join(pmx_script_lines) + "\n", encoding="utf-8")
        pmx_script.chmod(pmx_script.stat().st_mode | 0o100)

    (network_dir / "network_manifest.json").write_text(
        json.dumps(
            {
                "format": "alchemforge-ligand-network-v1",
                "ligands_sdf": "ligands.sdf",
                "graphml": "ligand_network.graphml",
                "edges_csv": "edges.csv",
                "network_method": config.network_method,
                "atom_mapper": config.atom_mapper,
                "max3d_angstrom": config.max3d_angstrom,
                "element_change": config.element_change,
                    "pmx_command": getattr(config, "pmx_command", None),
                    "pmx_mapping_mode": getattr(
                        config, "pmx_mapping_mode", None
                    ),
                    "pmx_h2h_polar": getattr(config, "pmx_h2h_polar", None),
                "ligand_count": len(ligands),
                "edge_count": len(ordered_edges),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return NetworkPlan(len(ligands), len(ordered_edges), graphml, edges_csv)


def add_network_edge(
    network_dir: str | Path, ligand_a: str, ligand_b: str
) -> dict[str, str]:
    """Map and add one new transformation to a portable network bundle."""
    import openfe
    from rdkit import Chem

    directory = Path(network_dir).expanduser().resolve()
    graphml = directory / "ligand_network.graphml"
    edges_csv = directory / "edges.csv"
    manifest_path = directory / "network_manifest.json"
    ligand_sdf = directory / "ligands.sdf"
    for required in (graphml, edges_csv, manifest_path, ligand_sdf):
        if not required.is_file():
            raise ValueError(f"Network bundle is missing: {required.name}")
    if ligand_a == ligand_b:
        raise ValueError("Select two different ligands")

    with edges_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if any({row["ligand_a"], row["ligand_b"]} == {ligand_a, ligand_b} for row in rows):
        raise ValueError(f"{ligand_a!r} and {ligand_b!r} are already connected")

    network = openfe.LigandNetwork.from_graphml(graphml.read_text(encoding="utf-8"))
    components = {component.name: component for component in network.nodes}
    if ligand_a not in components or ligand_b not in components:
        raise ValueError("Selected ligand is not present in the network GraphML")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapper_name = str(manifest.get("atom_mapper", "lomap")).lower()
    max3d = float(manifest.get("max3d_angstrom", 1.0))
    if mapper_name in {"lomap", "pmx"}:
        mapper = openfe.setup.LomapAtomMapper(
            threed=True,
            max3d=max3d,
            element_change=bool(manifest.get("element_change", False)),
        )
    elif mapper_name == "kartograf":
        mapper = openfe.setup.KartografAtomMapper(atom_max_distance=max3d)
    else:
        raise ValueError(f"Unsupported mapper in network manifest: {mapper_name}")
    scorer = openfe.lomap_scorers.default_lomap_score
    candidates = list(
        mapper.suggest_mappings(components[ligand_a], components[ligand_b])
    )
    if not candidates:
        raise ValueError(
            f"The {mapper_name} mapper found no mapping between {ligand_a!r} "
            f"and {ligand_b!r}"
        )
    scored = [(float(scorer(mapping)), mapping) for mapping in candidates]
    score, mapping = max(scored, key=lambda item: item[0])
    atom_mapping = {
        int(left): int(right)
        for left, right in mapping.componentA_to_componentB.items()
    }
    if not atom_mapping:
        raise ValueError("The selected mapper returned an empty atom mapping")

    edge_id = readable_edge_id(
        ligand_a,
        ligand_b,
        {row.get("edge_id", "") for row in rows},
    )
    mappings_dir = directory / "mappings"
    mappings_dir.mkdir(exist_ok=True)
    mapped_atoms: int | str = len(atom_mapping)

    if mapper_name == "pmx":
        molecules = {
            molecule.GetProp("_Name").strip(): molecule
            for molecule in Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
            if molecule is not None and molecule.HasProp("_Name")
        }
        ligand_dir = directory / "ligands"
        ligand_dir.mkdir(exist_ok=True)
        pdb_a = ligand_dir / f"{_safe_name(ligand_a)}.pdb"
        pdb_b = ligand_dir / f"{_safe_name(ligand_b)}.pdb"
        Chem.MolToPDBFile(molecules[ligand_a], str(pdb_a))
        Chem.MolToPDBFile(molecules[ligand_b], str(pdb_b))
        mapping_dir = mappings_dir / edge_id
        mapping_dir.mkdir(exist_ok=True)
        command = [
            *shlex.split(str(manifest.get("pmx_command", "pmx"))),
            "atomMapping",
            "-i1",
            str(pdb_a),
            "-i2",
            str(pdb_b),
            "-o1",
            str(mapping_dir / "pairs1.dat"),
            "-o2",
            str(mapping_dir / "pairs2.dat"),
        ]
        if bool(manifest.get("pmx_h2h_polar", True)):
            command.append("--H2Hpolar")
        command.extend(
            [
                "-opdbm1",
                str(mapping_dir / "mapped1.pdb"),
                "-opdbm2",
                str(mapping_dir / "mapped2.pdb"),
            ]
        )
        mode = str(manifest.get("pmx_mapping_mode", "stage"))
        script = directory / "run_pmx_mappings.sh"
        if not script.is_file():
            script.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8"
            )
        with script.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# {edge_id}\n{shlex.join(command)}\n")
        script.chmod(script.stat().st_mode | 0o100)
        if mode == "run":
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as error:
                detail = getattr(error, "stderr", "") or str(error)
                raise ValueError(
                    f"PMX mapping failed for {edge_id}: {detail}"
                ) from error
            mapped_atoms = sum(
                1
                for line in (mapping_dir / "pairs1.dat")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip() and not line.lstrip().startswith(("#", ";"))
            )
        else:
            mapped_atoms = "pending"
        payload = {
            "edge_id": edge_id,
            "ligand_a": ligand_a,
            "ligand_b": ligand_b,
            "mapper": "pmx atomMapping",
            "mapping_status": mode,
            "command": command,
            "pairs1": str(mapping_dir / "pairs1.dat"),
            "pairs2": str(mapping_dir / "pairs2.dat"),
            "mapped1": str(mapping_dir / "mapped1.pdb"),
            "mapped2": str(mapping_dir / "mapped2.pdb"),
            "lomap_network_score": score,
        }
    else:
        payload = {
            "edge_id": edge_id,
            "ligand_a": ligand_a,
            "ligand_b": ligand_b,
            "mapper": mapper_name,
            "index_base": 0,
            "componentA_to_componentB": atom_mapping,
            "lomap_score": score,
        }

    row = {
        "edge_id": edge_id,
        "ligand_a": ligand_a,
        "ligand_b": ligand_b,
        "mapped_atoms": str(mapped_atoms),
        "lomap_score": f"{score:.6f}",
    }
    updated_network = openfe.LigandNetwork(
        edges=[*network.edges, mapping], nodes=network.nodes
    )
    graphml.write_text(updated_network.to_graphml(), encoding="utf-8")
    with edges_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*rows, row])
    (mappings_dir / f"{edge_id}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    manifest["edge_count"] = len(rows) + 1
    manifest["edited"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return row
