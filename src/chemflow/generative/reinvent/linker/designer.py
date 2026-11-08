"""Orchestration: two clicked heads -> LinkInvent -> scored, ranked linker list.

This is the single entry point used by the CLI and by both desktop dialogs.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from chemflow.generative.reinvent.config import DEFAULTS, _deep_merge, build_toml, validate_config
from chemflow.generative.reinvent.linker.fit3d import FitContext, fit_candidates
from chemflow.generative.reinvent.linker.heads import HeadSpec, warhead_line
from chemflow.generative.reinvent.linker.results import LinkerCandidate, read_candidates
from chemflow.generative.reinvent.linker.scores import (
    FIT_COLUMNS,
    LINKER_COLUMNS,
    PROPERTY_COLUMNS,
    format_value,
    score_candidates,
)
from chemflow.generative.reinvent.run import build_command, find_results, stage_inputs


@dataclass
class DesignOptions:
    reinvent_command: str             # full path to the `reinvent` executable
    prior_file: str                   # linkinvent.prior
    workdir: str
    out_dir: str = "linker_design"
    num_smiles: int = 200             # linkers sampled for this warhead pair
    device: str = "cpu"
    seed: int | None = 42
    sample_strategy: str = "multinomial"


def write_warheads(head1: HeadSpec, head2: HeadSpec, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(warhead_line(head1, head2) + "\n", encoding="utf-8")
    return path


def sample_linkers(
    head1: HeadSpec,
    head2: HeadSpec,
    opts: DesignOptions,
    *,
    log: Callable[[str], None] = print,
) -> list[LinkerCandidate]:
    """Run LinkInvent sampling for the warhead pair; return candidates by NLL."""
    workdir = Path(opts.workdir).expanduser().resolve()
    out_dir = Path(opts.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else workdir / out_dir
    warheads = write_warheads(head1, head2, out_dir / "warheads.smi")

    conf: dict[str, Any] = _deep_merge(DEFAULTS, {
        "model": "linkinvent",
        "run_type": "sampling",
        "reinvent_command": opts.reinvent_command,
        "prior_file": str(Path(opts.prior_file).expanduser()),
        "smiles_file": str(warheads),
        "num_smiles": int(opts.num_smiles),
        "device": opts.device,
        "seed": opts.seed,
        "sample_strategy": opts.sample_strategy,
        "workdir": str(workdir),
        "out_dir": str(out_dir),
    })
    validate_config(conf)
    toml_path, out = stage_inputs(conf)
    command = build_command(conf, toml_path, out)
    log(f"warheads: {warhead_line(head1, head2)}")
    log("running: " + " ".join(command))
    log("Loading the LinkInvent model — this can take a minute on the first run…")

    # Force the child to flush line-by-line so the GUI shows live progress instead
    # of nothing until the process ends (REINVENT block-buffers stdout otherwise).
    import os as _os

    child_env = dict(_os.environ)
    child_env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, env=child_env,
        )
    except FileNotFoundError as exc:
        from chemflow.generative.reinvent.locate import not_found_message

        raise RuntimeError(not_found_message(command[0])) from exc
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-20:]
        log(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"REINVENT4 exited with code {rc}:\n" + "\n".join(tail))

    results = find_results(conf, out)
    if not results:
        raise RuntimeError(
            f"REINVENT4 produced no result file; see {out / 'reinvent.log'}"
        )
    candidates = read_candidates(results[0])
    log(f"{len(candidates)} unique valid linkers")
    return candidates


def design_linkers(
    head1: HeadSpec,
    head2: HeadSpec,
    opts: DesignOptions,
    *,
    models: Sequence[dict[str, Any]] | None = None,
    fit: FitContext | None = None,
    fit_top: int = 50,
    log: Callable[[str], None] = print,
    progress: Callable[[int, int], None] | None = None,
) -> list[LinkerCandidate]:
    """Sample, score and (optionally) 3D-check linkers for the two heads.

    ``fit_top`` limits the 3D fit to the best-ranked candidates by NLL, since it
    is the slow step. Candidates beyond that keep ``fit_*`` scores of ``None``.
    """
    candidates = sample_linkers(head1, head2, opts, log=log)
    log("scoring...")
    score_candidates(candidates, models=models)
    if fit is not None and fit_top > 0:
        subset = candidates[:fit_top]
        log(f"3D pocket fit for the top {len(subset)} linkers...")
        fit_candidates(subset, fit, progress=progress)
        placed = sum(1 for c in subset if c.scores.get("fit_ok") is True)
        clashing = sum(1 for c in subset if c.scores.get("fit_ok") is False)
        not_eval = sum(1 for c in subset if c.scores.get("fit_ok") is None)
        log(f"3D fit: {placed} placed clash-free, {clashing} placed with clashes, "
            f"{not_eval} could not be evaluated (of {len(subset)}).")
        if not_eval:
            from collections import Counter

            reasons = Counter(c.notes[-1] for c in subset
                              if c.scores.get("fit_ok") is None and c.notes)
            for reason, n in reasons.most_common(3):
                log(f"  not evaluated ({n}x): {reason}")
            if not reasons:
                log("  (no reason recorded — likely the two warheads are not in a "
                    "common coordinate frame; load the posed warheads from one "
                    "ternary-complex model, not two separate files.)")
    return candidates


# ---------------------------------------------------------------------------
# Tables / export
# ---------------------------------------------------------------------------

def table_columns(candidates: Sequence[LinkerCandidate], *, with_fit: bool) -> list[tuple[str, str, str]]:
    """(key, label, format) for every column that has data, incl. model scores."""
    columns = list(LINKER_COLUMNS) + list(PROPERTY_COLUMNS)
    model_keys = sorted({k for c in candidates for k in c.scores if k.startswith("model:")})
    columns += [(k, k.split(":", 1)[1], "{:.3f}") for k in model_keys]
    if with_fit:
        columns += list(FIT_COLUMNS)
    return columns


def table_rows(candidates: Sequence[LinkerCandidate], columns: Sequence[tuple[str, str, str]]) -> list[list[str]]:
    rows = []
    for c in candidates:
        rows.append([c.id, c.linker_smiles, format_value(c.nll, "{:.2f}")] +
                    [format_value(c.scores.get(key), fmt) for key, _label, fmt in columns])
    return rows


def export_design(candidates: Sequence[LinkerCandidate], out_dir: str | Path, *, with_fit: bool) -> dict[str, Path]:
    """Write ``linkers.csv`` (all scores), ``protacs.smi`` and, if present, 3D poses."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    columns = table_columns(candidates, with_fit=with_fit)

    csv_path = out / "linkers.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "linker_smiles", "protac_smiles", "nll"] + [label for _k, label, _f in columns])
        for c in candidates:
            writer.writerow([c.id, c.linker_smiles, c.protac_smiles, "" if c.nll is None else c.nll] +
                            ["" if c.scores.get(k) is None else c.scores.get(k) for k, _l, _f in columns])

    smi_path = out / "protacs.smi"
    smi_path.write_text("".join(f"{c.protac_smiles} {c.id}\n" for c in candidates), encoding="utf-8")
    result = {"csv": csv_path, "smi": smi_path}

    posed = [c for c in candidates if c.fit_molblock]
    if posed:
        sdf_path = out / "protac_poses.sdf"
        with sdf_path.open("w", encoding="utf-8") as handle:
            for c in posed:
                block = c.fit_molblock.rstrip("\n")
                handle.write(f"{block}\n>  <id>\n{c.id}\n\n>  <linker_smiles>\n{c.linker_smiles}\n\n$$$$\n")
        result["sdf"] = sdf_path
    return result
