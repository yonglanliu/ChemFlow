"""Build and equilibrate the solvated system as part of the OpenFE prep.

Historically the membrane system was built by a separate run of
``chemflow openmm modeling assemble`` and then named in the OpenFE config by
hand, along with ``complex_box_vectors``. Nothing checked that the two agreed,
and when they didn't, OpenFE mapped molecules into the wrong periodic cell and
the run died much later with a NaN inside a lambda window.

With a ``preparation`` block in the OpenFE config, ``create_json`` runs the
build/relax itself and takes both the structure and the box from its output, so
they cannot disagree. Equilibrating here — once, on the real system, with its
real force field — also removes any need to relax the alchemical hybrid later.

Config::

    preparation:
      enabled: true
      build_config: membrane_build.yaml     # relative to this config; omit BOTH
      relax_config: membrane_relax.yaml     # to have defaults generated for you
      md_config: null                       # optional production MD
      strip_resnames: [LIG]                 # remove the build ligand; OpenFE adds its own
      reference_ligand: true                # put a ligand in the pocket before packing lipids
                                            # true/"largest" = the biggest in the network
                                            # (carves room for the whole transformation),
                                            # "first", or a molecule name
      reference_ligand_resname: LIG         # ...as this residue; auto-added to strip_resnames
      dry_run: false                        # print the build/relax commands and stop
      allow_short_relax: false              # permit a non-"full" relax on a fresh bilayer

The build starts from a bare, OPM-oriented protein: ``membrane_build`` tiles a
lipid patch around it, trims to protein-XY plus padding, then solvates and
ionises. So the RBFE run takes a protein the way a soluble run does, and the
bilayer never has to be produced or described by hand.
"""

from __future__ import annotations

from pathlib import Path


def _resolve(base: Path, value) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _build_mode(build_cfg: Path) -> str:
    """``build`` (pack a bilayer from scratch) or ``existing`` (solvate only)."""
    try:
        from chemflow.fe.openmm_modeling import membrane_build_config as bc

        return str(getattr(bc.load_config(build_cfg), "mode", "build")).lower()
    except Exception:
        return "build"


def _report_relax(relax_cfg: Path, from_scratch: bool, allow_short: bool) -> None:
    """Print the equilibration ladder, and refuse a short one on a fresh bilayer.

    A bilayer packed from scratch has lipids in arbitrary contact and a protein
    that has never seen them. It needs the whole ladder — minimise, cold
    restrained NVT, barostat on, heating ramp, restraint release, unrestrained
    production — not the short re-settle path meant for a membrane that was
    already equilibrated. OpenFE's own per-window equilibration cannot make up
    the difference, so getting this wrong is silent and expensive.
    """
    from chemflow.fe.openmm_modeling import membrane_relax_config as rc

    conf = rc.load_config(relax_cfg)
    protocol = str(getattr(conf, "protocol", "full")).lower()
    stages = list(getattr(conf, "stages", []) or [])

    total = 0.0
    print(f"  relax protocol: {protocol}  ({len(stages)} stage(s))", flush=True)
    for stage in stages:
        name = str(getattr(stage, "name", getattr(stage, "title", "?")))
        duration = getattr(stage, "simulation_time", None)
        if duration is None:
            schedule = getattr(stage, "temperature_schedule", None) or []
            ramp = sum(float(getattr(step, "duration", 0.0) or 0.0) for step in schedule)
            duration = ramp or None
        restraints = getattr(stage, "restraints", None)
        held = "unrestrained" if restraints in (False, None) else "restrained"
        if duration:
            total += float(duration)
            print(f"      {name:24s} {float(duration):8.1f} ps   {held}", flush=True)
        else:
            print(f"      {name:24s} {'minimise':>11s}", flush=True)
    print(f"  total equilibration MD: {total:.0f} ps", flush=True)

    if from_scratch and protocol != "full" and not allow_short:
        raise ValueError(
            f"the membrane is being packed from scratch (build mode 'build') but "
            f"{relax_cfg.name} uses the '{protocol}' relaxation protocol.\n"
            "  A freshly packed bilayer needs the full ladder — minimisation, cold "
            "restrained NVT, barostat on, heating ramp, restraint release, then "
            "unrestrained production. The short path is for a membrane that was "
            "already equilibrated and only needs to re-settle.\n"
            "  Set protocol: full in the relax config, or "
            "preparation.allow_short_relax: true if you really mean it."
        )


def _sdf_records(path: Path) -> list[str]:
    """Split an SDF into complete records (each ending in the ``$$$$`` line)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    records, current = [], []
    for line in text.splitlines(keepends=True):
        current.append(line)
        if line.strip() == "$$$$":
            records.append("".join(current))
            current = []
    if any(ln.strip() for ln in current):            # trailing record without $$$$
        records.append("".join(current) + "$$$$\n")
    return records


def _record_title(record: str) -> str:
    """First line of an SDF record — the molecule name, when it has one."""
    return record.splitlines()[0].strip() if record.splitlines() else ""


def _record_heavy_atoms(record: str) -> int:
    """Heavy-atom count from a V2000 counts line; 0 if it cannot be read."""
    lines = record.splitlines()
    if len(lines) < 4:
        return 0
    try:
        total = int(lines[3][:3])
    except ValueError:
        return 0
    heavy = 0
    for line in lines[4:4 + total]:
        # atom block: x y z symbol ...
        parts = line.split()
        if len(parts) >= 4 and parts[3] != "H":
            heavy += 1
    return heavy or total


def _pick_reference(records: list[str], wanted) -> int:
    """Index of the reference molecule: named, largest, or first.

    ``largest`` is the default because the pocket has to accommodate the whole
    alchemical network, not just one end of it. Carving it with a small ligand
    leaves the region covered only by the larger ligand's unique atoms free for
    lipid and water, which then collide with those atoms once the hybrid is
    built.
    """
    if isinstance(wanted, str) and wanted.lower() not in {
            "true", "first", "1", "yes", "largest"}:
        for index, record in enumerate(records):
            if _record_title(record) == wanted:
                return index
        raise ValueError(
            f"reference ligand '{wanted}' is not in the ligand SDF; "
            f"available: {[_record_title(r) for r in records]}"
        )
    if isinstance(wanted, str) and wanted.lower() == "first":
        return 0
    sizes = [_record_heavy_atoms(r) for r in records]
    return int(max(range(len(records)), key=lambda i: sizes[i])) if sizes else 0


def merge_ligand_sdf(conf, prep, build_cfg: Path, destination: Path) -> Path | None:
    """One SDF covering every non-standard residue the build has to parameterize.

    OpenMM needs a force-field template for each non-standard residue in the
    input structure, and ``membrane_build`` gets those by registering the
    molecules in its ``ligand_sdf``. The alchemical ligands and the cofactor
    are declared separately in the OpenFE config, so without this the cofactor
    is missing and the build dies with ``No template found for residue (4DY)``
    after the force field is already half built.

    The two sources are treated differently, because the built system contains
    different numbers of each:

    * ``ligand_sdf`` holds every ligand in the alchemical network, but only
      **one** of them is placed in the structure (the reference pose). Just
      that one is taken — the others would each cost an antechamber run for a
      template nothing uses.
    * ``cofactor_template_sdf`` / ``cofactor_sdf`` hold cofactors that are all
      present in the structure, so **all** of them are taken.
    """
    import yaml

    wanted = getattr(prep, "reference_ligand", None) if prep is not None else None
    named = wanted if isinstance(wanted, str) and wanted.lower() not in {
        "true", "first", "1", "yes"} else None
    try:
        build_data = yaml.safe_load(build_cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        build_data = {}

    def _resolve_sdf(value):
        if not value:
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (build_cfg.parent / path).resolve()
        return path if path.is_file() else None

    # (path, take_all)
    sources: list[tuple[Path, bool]] = []
    seen: set[Path] = set()
    for value, take_all in ((build_data.get("ligand_sdf"), False),
                            (getattr(conf, "ligand_sdf", None), False),
                            (getattr(conf, "cofactor_template_sdf", None), True),
                            (getattr(conf, "cofactor_sdf", None), True)):
        path = _resolve_sdf(value)
        if path is not None and path not in seen:
            seen.add(path)
            sources.append((path, take_all))

    if not sources:
        return None

    records: list[str] = []
    for path, take_all in sources:
        found = _sdf_records(path)
        if take_all:
            chosen = found
            note = f"all {len(found)}"
        else:
            index = _pick_reference(found, wanted)
            chosen = [found[index]]
            size = _record_heavy_atoms(found[index])
            note = (f"1 of {len(found)}"
                    + (f" ({_record_title(chosen[0])}, {size} heavy atoms"
                       f"{', largest' if len(found) > 1 and named is None else ''})"
                       if _record_title(chosen[0]) else ""))
        print(f"    {note} molecule(s) from {path.name}", flush=True)
        records.extend(chosen)

    if not records:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(records), encoding="utf-8")
    print(f"  force-field ligands: {len(records)} molecule(s) -> {destination}",
          flush=True)
    return destination


def _config_with_ligands(cfg_path: Path, ligand_sdf: Path) -> Path:
    """Copy ``cfg_path`` beside itself with ``ligand_sdf`` overridden."""
    import yaml

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if str(data.get("ligand_sdf") or "") == str(ligand_sdf):
        return cfg_path
    data["ligand_sdf"] = str(ligand_sdf)
    out = cfg_path.parent / f"_prep_{cfg_path.stem}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def insert_reference_ligand(conf, prep, build_cfg: Path) -> tuple[Path, str] | None:
    """Put a reference ligand into the build input so the pocket stays occupied.

    The lipid packing has no reason to avoid an empty binding site — and for a
    site that opens onto the bilayer it will actively fill it. Whatever gets
    packed in there survives the relaxation, and OpenFE then inserts its own
    ligands on top of it. So a representative ligand goes in before the build
    and comes out again (via ``strip_resnames``) before OpenFE sees the system.

    The coordinates come from the alchemical ``ligand_sdf``, which is already
    posed in the receptor frame. Returns ``(new_input_pdb, resname)``.
    """
    wanted = getattr(prep, "reference_ligand", None)
    if not wanted:
        return None

    import yaml
    from openmm.app import Modeller, PDBFile

    resname = str(getattr(prep, "reference_ligand_resname", "LIG")).strip()[:4] or "LIG"

    sdf = getattr(conf, "ligand_sdf", None)
    if not sdf or not Path(str(sdf)).is_file():
        raise ValueError(
            "preparation.reference_ligand is set but ligand_sdf is missing — "
            "the reference pose is taken from there"
        )

    from openff.toolkit import Molecule

    molecules = Molecule.from_file(str(sdf), allow_undefined_stereo=True)
    molecules = molecules if isinstance(molecules, list) else [molecules]
    # Select exactly as merge_ligand_sdf does — the molecule placed in the
    # pocket and the molecule given a force-field template must be the same.
    index = _pick_reference(_sdf_records(Path(str(sdf))), wanted)
    chosen = molecules[min(index, len(molecules) - 1)]
    heavy = sum(1 for a in chosen.atoms if a.symbol != "H")
    if len(molecules) > 1:
        others = max(
            (sum(1 for a in m.atoms if a.symbol != "H") for m in molecules), default=heavy)
        if heavy < others:
            print(f"  NOTE: the reference ligand has {heavy} heavy atoms but the "
                  f"network's largest has {others} — the pocket will be carved too "
                  "small for part of the transformation.", flush=True)

    build_data = yaml.safe_load(build_cfg.read_text(encoding="utf-8")) or {}
    input_pdb = Path(str(build_data.get("input_pdb", "")))
    if not input_pdb.is_absolute():
        input_pdb = (build_cfg.parent / input_pdb).resolve()
    if not input_pdb.is_file():
        raise ValueError(f"the build config's input_pdb does not exist: {input_pdb}")

    pdb = PDBFile(str(input_pdb))
    topology = chosen.to_topology()
    modeller = Modeller(pdb.topology, pdb.positions)
    before = modeller.topology.getNumResidues()
    modeller.add(topology.to_openmm(), topology.get_positions().to_openmm())
    for index, residue in enumerate(modeller.topology.residues()):
        if index >= before:
            residue.name = resname

    out = input_pdb.with_name(f"{input_pdb.stem}_with_{resname.lower()}.pdb")
    with out.open("w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle)
    print(f"  reference ligand: {chosen.name or '(first molecule)'} from "
          f"{Path(sdf).name} -> {out.name} as residue {resname} "
          f"({chosen.n_atoms} atoms)", flush=True)
    return out, resname


def _config_with_input(cfg_path: Path, input_pdb: Path) -> Path:
    """Copy ``cfg_path`` beside itself with ``input_pdb`` overridden."""
    import yaml

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    data["input_pdb"] = str(input_pdb)
    out = cfg_path.parent / f"_prep_{cfg_path.stem.removeprefix('_prep_')}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def _config_with_binding_pose(cfg_path: Path, prep) -> Path:
    """Force ``preserve_binding_pose`` on for a free-energy starting structure.

    The stock relaxation ladder releases every restraint at stage 6, which is
    right for a plain membrane equilibration but wrong here: a free-energy run
    starts from the pose in the ligand SDF, and the alchemical mapping is built
    against that geometry. If the solute drifts during the last two stages the
    ligands have to be re-aligned before the network can be planned.

    This used to be set only when :func:`_autogenerate_configs` wrote a fresh
    relax config, so a supplied ``relax_config`` — or simply a second run that
    reused the generated one — silently released the restraints instead. Now it
    applies to whatever config this module drives.

    Precedence: ``preparation.preserve_binding_pose`` (what the desktop sets)
    beats the relax config's own key, which in turn beats the default of on.
    """
    import yaml

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    requested = getattr(prep, "preserve_binding_pose", None)
    source = "preparation.preserve_binding_pose"

    if requested is None:
        if "preserve_binding_pose" in data:
            requested, source = bool(data["preserve_binding_pose"]), cfg_path.name
        else:
            requested, source = True, "default for free-energy prep"

    if requested:
        print(f"  preserve_binding_pose: true ({source}) — backbone, sidechain "
              "and ligand stay restrained through stage 6 and production",
              flush=True)
    else:
        print(f"  preserve_binding_pose: false ({source}) — the solute is "
              "released at stage 6, so the ligands may need re-aligning before "
              "the network is planned", flush=True)

    if bool(data.get("preserve_binding_pose")) == requested \
            and "preserve_binding_pose" in data:
        return cfg_path

    data["preserve_binding_pose"] = requested
    out = cfg_path.parent / f"_prep_{cfg_path.stem.removeprefix('_prep_')}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def _autogenerate_configs(conf, prep, base: Path) -> tuple[Path, Path]:
    """Write default build + relax configs when none were given.

    They are ordinary commented YAML, written next to the OpenFE config so you
    can edit and re-run without regenerating. Values that the OpenFE config
    already states (solvent model, ions, force field, platform) are carried
    over, so the two do not quietly disagree.
    """
    import yaml

    from chemflow.fe.openmm_modeling import membrane_build_config as bc
    from chemflow.fe.openmm_modeling import membrane_relax_config as rc

    protein = str(getattr(conf, "protein_pdb", "") or "")
    if not protein:
        raise ValueError(
            "preparation.enabled is true with no build_config, so one is generated "
            "from protein_pdb — but protein_pdb is not set"
        )
    platform = str(getattr(conf, "platform", "CUDA") or "CUDA")
    lipid = str(getattr(prep, "lipid_type", "POPC") or "POPC")

    build_path = base / "membrane_build.yaml"
    relax_path = base / "membrane_relax.yaml"

    if not build_path.exists():
        build_path.write_text(
            bc.default_yaml_text(protein, platform=platform, lipid_type=lipid),
            encoding="utf-8")
        data = yaml.safe_load(build_path.read_text(encoding="utf-8")) or {}
        for key, value in (
            ("small_molecule_forcefield", getattr(conf, "small_molecule_forcefield", None)),
            ("solvent_model", getattr(conf, "solvent_model", None)),
            ("positive_ion", getattr(conf, "positive_ion", None)),
            ("negative_ion", getattr(conf, "negative_ion", None)),
            ("ionic_strength", getattr(conf, "ion_concentration", None)),
            ("neutralize", getattr(conf, "neutralize", None)),
        ):
            if value is not None:
                data[key] = value
        build_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(f"  generated build config: {build_path.name} "
              f"(lipid {lipid}, membrane_center_z 0.0)", flush=True)
    else:
        print(f"  reusing existing {build_path.name}", flush=True)

    if not relax_path.exists():
        relax_path.write_text(rc.default_yaml_text(platform=platform), encoding="utf-8")
        data = yaml.safe_load(relax_path.read_text(encoding="utf-8")) or {}
        # A free-energy starting structure must keep the pocket and the docked
        # pose where the ligand SDF expects them, so the solute stays restrained
        # to the end instead of being released at stage 6.
        data["preserve_binding_pose"] = True
        for key, value in (
            ("small_molecule_forcefield", getattr(conf, "small_molecule_forcefield", None)),
        ):
            if value is not None:
                data[key] = value
        relax_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(f"  generated relax config: {relax_path.name} (full staged protocol, "
              "solute restrained throughout)", flush=True)
    else:
        print(f"  reusing existing {relax_path.name}", flush=True)

    return build_path, relax_path


def _check_opm_frame(conf, build_cfg: Path) -> None:
    """A generated build config assumes membrane_center_z = 0, i.e. OPM frame.

    Getting this wrong packs the lipid patch past the end of the protein and
    the build dies in minimisation with no obvious cause, so say so here.
    """
    import yaml

    data = yaml.safe_load(build_cfg.read_text(encoding="utf-8")) or {}
    centre = float((data.get("build") or {}).get("membrane_center_z", 0.0) or 0.0)
    pdb = data.get("input_pdb") or getattr(conf, "protein_pdb", None)
    if not pdb or not Path(str(pdb)).is_file():
        return
    try:
        from chemflow.utils.structure import load_structure

        _, positions = load_structure(str(pdb))
    except Exception as exc:
        # Never block the build on a diagnostic — but never fail silently either.
        print(f"  (could not check the membrane centre against {Path(pdb).name}: "
              f"{type(exc).__name__}: {exc})", flush=True)
        return

    z = positions[:, 2]
    lo, hi = float(z.min()), float(z.max())
    print(f"  input z range {lo:.2f} .. {hi:.2f} nm, "
          f"membrane_center_z {centre:.3f} nm", flush=True)
    if not (lo < centre < hi):
        raise ValueError(
            f"membrane_center_z is {centre:.3f} nm but the structure spans "
            f"{lo:.2f} .. {hi:.2f} nm, so the bilayer would be built outside the "
            "protein entirely.\n  0.0 is correct only for an OPM-oriented structure. "
            "If this protein came out of an existing membrane system, measure the "
            "midplane from that system (midpoint of the two lipid phosphate planes) "
            "and set build.membrane_center_z to it — the desktop's Measure button on "
            "the OpenMM Modeling page does this."
        )


def prepare_system(conf, config_dir: str | Path) -> str | None:
    """Run the configured build/relax and return the structure OpenFE should use.

    Returns ``None`` when no ``preparation`` block is enabled, leaving
    ``protein_pdb`` exactly as the user set it.
    """
    prep = getattr(conf, "preparation", None)
    if prep is None or not bool(getattr(prep, "enabled", False)):
        return None

    base = Path(config_dir).expanduser().resolve()
    build_cfg = _resolve(base, getattr(prep, "build_config", None))
    relax_cfg = _resolve(base, getattr(prep, "relax_config", None))
    md_cfg = _resolve(base, getattr(prep, "md_config", None))

    missing = [name for name, path in (("build_config", build_cfg),
                                       ("relax_config", relax_cfg)) if path is None]
    if missing:
        # Both unset is the normal "just build it" case — generate defaults from
        # what the OpenFE config already says. One of the two set is ambiguous.
        if len(missing) == 1:
            raise ValueError(
                f"preparation sets {'relax_config' if missing[0] == 'build_config' else 'build_config'} "
                f"but not {missing[0]} — set both, or neither to generate both"
            )
        build_cfg, relax_cfg = _autogenerate_configs(conf, prep, base)

    for label, path in (("build_config", build_cfg), ("relax_config", relax_cfg),
                        ("md_config", md_cfg)):
        if path is not None and not path.is_file():
            raise ValueError(f"preparation.{label} not found: {path}")

    _check_opm_frame(conf, build_cfg)

    from chemflow.fe.openmm_modeling.assemble import assemble

    dry_run = bool(getattr(prep, "dry_run", False))
    print("=" * 62, flush=True)
    print("System preparation: build + relax before the free-energy setup", flush=True)

    # Every non-standard residue in the input needs a force-field template, and
    # the ligands and cofactor are declared in separate keys. Merge them so the
    # build and relax both see the full set.
    merged = merge_ligand_sdf(conf, prep, build_cfg,
                              build_cfg.parent / "_prep_ff_ligands.sdf")
    if merged is not None:
        build_cfg = _config_with_ligands(build_cfg, merged)
        relax_cfg = _config_with_ligands(relax_cfg, merged)
        if md_cfg is not None:
            md_cfg = _config_with_ligands(md_cfg, merged)

    # A free-energy starting structure must hold the pose to the end, whatever
    # relax config we were handed — generated, reused, or supplied by hand.
    relax_cfg = _config_with_binding_pose(relax_cfg, prep)

    # Keep the binding site occupied while the lipids pack around the protein.
    strip = getattr(prep, "strip_resnames", None)
    if isinstance(strip, str):
        strip = [token for token in strip.replace(",", " ").split() if token]
    strip = list(strip or [])
    reference = insert_reference_ligand(conf, prep, build_cfg)
    if reference is not None:
        reference_pdb, reference_resname = reference
        build_cfg = _config_with_input(build_cfg, reference_pdb)
        # It must come out again: OpenFE inserts its own ligands, and leaving
        # this one in would put two copies at the same coordinates.
        if reference_resname not in strip:
            strip.append(reference_resname)

    print(f"  build : {build_cfg}", flush=True)
    print(f"  relax : {relax_cfg}", flush=True)
    if md_cfg:
        print(f"  md    : {md_cfg}", flush=True)
    _report_relax(
        relax_cfg,
        from_scratch=_build_mode(build_cfg) == "build",
        allow_short=bool(getattr(prep, "allow_short_relax", False)),
    )
    if dry_run:
        print("  DRY RUN — printing the steps, running nothing", flush=True)

    final = Path(assemble(build_cfg, relax_cfg, md_cfg, dry_run=dry_run))
    if dry_run:
        print(f"Dry run complete; the real run would produce {final}", flush=True)
        print("=" * 62, flush=True)
        raise SystemExit(0)
    # assemble writes both a PDB and an mmCIF; OpenFE reads either, and the
    # mmCIF carries the cell unambiguously, so prefer it when present.
    structure = final.with_suffix(".cif")
    if not structure.is_file():
        structure = final
    if not structure.is_file():
        raise RuntimeError(
            f"system preparation finished but produced no structure at {final} "
            "— check the build/relax logs above"
        )

    # The membrane is usually built around a representative ligand, to carve the
    # pocket. OpenFE inserts its own ligands as alchemical components, so that
    # copy has to go — otherwise the ligand exists twice, superimposed, which
    # ends as a NaN in the first lambda window. `strip` was assembled above and
    # already includes the reference ligand, if one was inserted.
    if strip:
        from chemflow.fe.openfe.utils.cofactor_from_pdb import strip_residues

        stripped = structure.with_name(structure.stem + "_noligand" + structure.suffix)
        strip_residues(str(structure), list(strip), str(stripped))
        print(f"Stripped {list(strip)} from the prepared system -> {stripped.name} "
              "(OpenFE inserts the ligands itself)", flush=True)
        structure = stripped

    print(f"Prepared system: {structure}", flush=True)
    print("=" * 62, flush=True)
    return str(structure)
