"""Configuration layer for REINVENT4 (no REINVENT / torch import).

One YAML drives every REINVENT4 model and run type:

* ``model``    : reinvent (de novo), libinvent (scaffold decoration / R-group),
                 linkinvent (linker design), mol2mol (molecule optimisation),
                 pepinvent (peptides)
* ``run_type`` : ``sampling`` (draw molecules from a prior) or
                 ``staged_learning`` (reinforcement learning against a score)

:func:`build_toml` translates the config to the TOML REINVENT4 consumes.
ChemFlow-trained predictors can be plugged into the RL reward through
``chemflow_scorers`` (see :mod:`chemflow.generative.reinvent.scorer`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SUPPORTED_MODELS = {"reinvent", "libinvent", "linkinvent", "mol2mol", "pepinvent"}
SUPPORTED_RUN_TYPES = {"sampling", "staged_learning"}
# Models whose sampling / RL needs an input SMILES file (scaffolds, warhead
# pairs, seed molecules, ...).
MODELS_NEEDING_INPUT = {"libinvent", "linkinvent", "mol2mol", "pepinvent"}

DEFAULT_PRIORS = {
    "reinvent": "priors/reinvent_pubchem.prior",
    "libinvent": "priors/libinvent.prior",
    "linkinvent": "priors/linkinvent.prior",
    "mol2mol": "priors/pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior",
    "pepinvent": "priors/pepinvent.prior",
}

DEFAULTS: dict[str, Any] = {
    "model": "reinvent",
    "run_type": "sampling",
    "reinvent_command": "reinvent",
    "workdir": ".",
    "out_dir": "reinvent_output",
    "device": "cuda:0",
    "seed": None,
    "prior_file": None,
    "agent_file": None,           # RL only; defaults to prior_file
    "smiles_file": None,          # scaffolds / warheads / seed molecules
    # --- sampling ---
    "num_smiles": 1000,
    "sample_strategy": "multinomial",
    "temperature": 1.0,
    "unique_molecules": True,
    "randomize_smiles": True,
    # --- reinforcement learning ---
    "batch_size": 100,
    "use_checkpoint": False,
    "purge_memories": False,
    "learning_strategy": {"type": "dap", "sigma": 128, "rate": 0.0001},
    "diversity_filter": {
        "type": "IdenticalMurckoScaffold",
        "bucket_size": 25,
        "minscore": 0.4,
        "minsimilarity": 0.4,
        "penalty_multiplier": 0.5,
    },
    # [{chkpt_file, termination, max_score, min_steps, max_steps,
    #   scoring: {type, components: [{name, weight, params, transform}]}}]
    "stages": [],
    # ChemFlow predictors appended to every stage's scoring components:
    # [{name, model, weight, column, python, transform}]
    "chemflow_scorers": [],
    # Raw extra keys merged into the TOML [parameters] table.
    "parameters_extra": {},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"REINVENT config must be a YAML mapping: {path}")
    return data


def _resolve(base_dir: Path, value: Any) -> Any:
    if not value:
        return value
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return str(candidate)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a REINVENT YAML merged over the defaults; resolve file paths.

    Returned as a plain dict (the TOML builder needs ordered, nested data).
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"REINVENT config not found: {config_path}\n"
            "Create one with: chemflow generate reinvent generate -o reinvent.yaml "
            "(or pass the full path to an existing file)."
        )
    conf = _deep_merge(DEFAULTS, _load_yaml(config_path))
    # ``stages`` / ``chemflow_scorers`` are lists, so they replace (not merge).
    base_dir = config_path.parent

    workdir = conf.get("workdir") or "."
    conf["workdir"] = str(base_dir) if str(workdir) == "." else _resolve(base_dir, workdir)

    for key in ("prior_file", "agent_file", "smiles_file"):
        if conf.get(key):
            conf[key] = _resolve(base_dir, conf[key])
    for scorer in conf.get("chemflow_scorers") or []:
        if scorer.get("model"):
            scorer["model"] = _resolve(base_dir, scorer["model"])
    conf["_config_path"] = str(config_path)
    return conf


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _stage_components(conf: dict[str, Any], stage: dict[str, Any]) -> list[dict]:
    comps = list(((stage.get("scoring") or {}).get("components")) or [])
    return comps + list(conf.get("chemflow_scorers") or [])


def validate_config(conf: dict[str, Any]) -> None:
    """Validate a loaded config (no REINVENT needed)."""
    model = str(conf.get("model"))
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"model must be one of {sorted(SUPPORTED_MODELS)} (got {model!r})")
    run_type = str(conf.get("run_type"))
    if run_type not in SUPPORTED_RUN_TYPES:
        raise ValueError(
            f"run_type must be one of {sorted(SUPPORTED_RUN_TYPES)} (got {run_type!r})"
        )

    prior = conf.get("prior_file")
    if not prior:
        raise ValueError("prior_file is required (e.g. priors/reinvent.prior from the REINVENT4 repo)")
    if not Path(str(prior)).exists():
        raise ValueError(f"prior_file not found: {prior}")
    agent = conf.get("agent_file")
    if agent and not Path(str(agent)).exists():
        raise ValueError(f"agent_file not found: {agent}")

    smiles_file = conf.get("smiles_file")
    if model in MODELS_NEEDING_INPUT and not smiles_file:
        raise ValueError(f"model '{model}' requires smiles_file (scaffolds / warheads / seed molecules)")
    if smiles_file and not Path(str(smiles_file)).exists():
        raise ValueError(f"smiles_file not found: {smiles_file}")

    if int(conf.get("num_smiles", 0)) < 1:
        raise ValueError("num_smiles must be >= 1")
    if int(conf.get("batch_size", 0)) < 1:
        raise ValueError("batch_size must be >= 1")

    for scorer in conf.get("chemflow_scorers") or []:
        _validate_chemflow_scorer(scorer)

    if run_type == "staged_learning":
        stages = conf.get("stages") or []
        if not stages:
            raise ValueError("staged_learning requires at least one entry in 'stages'")
        seen: set[str] = set()
        for i, stage in enumerate(stages, start=1):
            if not stage.get("chkpt_file"):
                raise ValueError(f"stage {i} needs a 'chkpt_file'")
            if stage["chkpt_file"] in seen:
                raise ValueError(f"duplicate chkpt_file: {stage['chkpt_file']}")
            seen.add(stage["chkpt_file"])
            if int(stage.get("max_steps", 0)) < 1:
                raise ValueError(f"stage {i} needs max_steps >= 1")
            components = _stage_components(conf, stage)
            if not components:
                raise ValueError(f"stage {i} has no scoring components")
            for comp in components:
                name = str(comp.get("name", "")).strip()
                if not name:
                    raise ValueError(f"stage {i}: every scoring component needs a 'name'")
                weight = comp.get("weight", 1.0)
                if not isinstance(weight, (int, float)) or weight <= 0:
                    raise ValueError(f"stage {i}: component '{name}' weight must be > 0")


def _validate_chemflow_scorer(scorer: dict[str, Any]) -> None:
    name = str(scorer.get("name", "")).strip()
    if not name:
        raise ValueError("each chemflow_scorers entry needs a 'name'")
    model = scorer.get("model")
    if not model:
        raise ValueError(f"chemflow scorer '{name}' needs a 'model' (ChemFlow model pickle)")
    if not Path(str(model)).exists():
        raise ValueError(f"chemflow scorer '{name}': model not found: {model}")
    weight = scorer.get("weight", 1.0)
    if not isinstance(weight, (int, float)) or weight <= 0:
        raise ValueError(f"chemflow scorer '{name}' weight must be > 0")


# ---------------------------------------------------------------------------
# TOML emission (tiny emitter: avoids a tomli_w dependency)
# ---------------------------------------------------------------------------

def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _flat_items(prefix: str, mapping: dict[str, Any]):
    """Yield ``dotted.key = value`` pairs, flattening nested dicts."""
    for key, value in mapping.items():
        if value is None:
            continue
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _flat_items(dotted, value)
        else:
            yield dotted, value


def _kv_lines(mapping: dict[str, Any], prefix: str = "") -> list[str]:
    return [f"{k} = {_toml_value(v)}" for k, v in _flat_items(prefix, mapping)]


def chemflow_scorer_endpoint(scorer: dict[str, Any]) -> dict[str, Any]:
    """REINVENT ``ExternalProcess`` endpoint that calls the ChemFlow scorer."""
    args = [
        "-m", "chemflow.generative.reinvent.scorer",
        "--model", str(scorer["model"]),
    ]
    if scorer.get("column"):
        args += ["--column", str(scorer["column"])]
    endpoint: dict[str, Any] = {
        "name": scorer["name"],
        "weight": scorer.get("weight", 1.0),
        "params": {
            "executable": str(scorer.get("python") or sys.executable),
            "args": " ".join(_shell_quote(a) for a in args),
            "property": "score",
        },
    }
    if scorer.get("transform"):
        endpoint["transform"] = dict(scorer["transform"])
    return endpoint


def _shell_quote(arg: str) -> str:
    import shlex

    return shlex.quote(arg)


def _group_components(conf: dict[str, Any], stage: dict[str, Any]) -> list[tuple[str, list[dict]]]:
    """Return ``[(component_name, endpoints), ...]`` for one stage.

    Endpoints that share a component name are merged (e.g. several QED
    endpoints). Each ChemFlow scorer becomes its OWN ``ExternalProcess``
    component, because REINVENT4 allows only one executable + arguments per
    ExternalProcess component.
    """
    grouped: dict[str, list[dict]] = {}
    for comp in ((stage.get("scoring") or {}).get("components")) or []:
        endpoint: dict[str, Any] = {
            "name": comp.get("endpoint_name") or comp["name"],
            "weight": comp.get("weight", 1.0),
        }
        if comp.get("params"):
            endpoint["params"] = dict(comp["params"])
        if comp.get("transform"):
            endpoint["transform"] = dict(comp["transform"])
        grouped.setdefault(str(comp["name"]), []).append(endpoint)
    result = list(grouped.items())
    for scorer in conf.get("chemflow_scorers") or []:
        result.append(("ExternalProcess", [chemflow_scorer_endpoint(scorer)]))
    return result


def build_toml(conf: dict[str, Any], out_dir: str | Path | None = None) -> str:
    """Translate a validated config into REINVENT4 TOML text."""
    out = Path(out_dir or conf.get("out_dir") or "reinvent_output")
    if not out.is_absolute():
        out = Path(conf.get("workdir") or ".") / out
    run_type = conf["run_type"]
    lines: list[str] = [
        "# Generated by chemflow (chemflow generate reinvent) - edit the YAML, not this file.",
        f'run_type = "{run_type}"',
        f'device = "{conf.get("device", "cuda:0")}"',
        f'json_out_config = "{out / ("_" + run_type + ".json")}"',
    ]
    if run_type == "staged_learning":
        lines.append(f'tb_logdir = "{out / "tb_logs"}"')
    lines += ["", "[parameters]"]

    params: dict[str, Any] = {}
    if run_type == "sampling":
        params.update({
            "model_file": conf["prior_file"],
            "output_file": str(out / "sampling.csv"),
            "num_smiles": int(conf["num_smiles"]),
            "unique_molecules": bool(conf["unique_molecules"]),
            "randomize_smiles": bool(conf["randomize_smiles"]),
        })
        if conf.get("smiles_file"):
            params["smiles_file"] = conf["smiles_file"]
        if conf["model"] in MODELS_NEEDING_INPUT:
            params["sample_strategy"] = conf["sample_strategy"]
        if conf["model"] == "mol2mol":
            params["temperature"] = float(conf["temperature"])
    else:
        params.update({
            "summary_csv_prefix": str(out / "staged_learning"),
            "use_checkpoint": bool(conf["use_checkpoint"]),
            "purge_memories": bool(conf["purge_memories"]),
            "prior_file": conf["prior_file"],
            "agent_file": conf.get("agent_file") or conf["prior_file"],
            "batch_size": int(conf["batch_size"]),
            "randomize_smiles": bool(conf["randomize_smiles"]),
        })
        if conf.get("smiles_file"):
            params["smiles_file"] = conf["smiles_file"]
        if conf["model"] == "mol2mol":
            params["sample_strategy"] = conf["sample_strategy"]
            params["temperature"] = float(conf["temperature"])
    params.update(conf.get("parameters_extra") or {})
    lines += _kv_lines(params)

    if run_type == "staged_learning":
        if conf.get("learning_strategy"):
            lines += ["", "[learning_strategy]"] + _kv_lines(conf["learning_strategy"])
        if conf.get("diversity_filter"):
            lines += ["", "[diversity_filter]"] + _kv_lines(conf["diversity_filter"])
        for stage in conf.get("stages") or []:
            lines += ["", "[[stage]]"]
            stage_keys = {
                "chkpt_file": str(out / stage["chkpt_file"]),
                "termination": stage.get("termination", "simple"),
                "max_score": stage.get("max_score", 1.0),
                "min_steps": stage.get("min_steps", 0),
                "max_steps": stage["max_steps"],
            }
            lines += _kv_lines(stage_keys)
            scoring_type = (stage.get("scoring") or {}).get("type", "geometric_mean")
            lines += ["", "[stage.scoring]", f'type = "{scoring_type}"']
            for comp_name, endpoints in _group_components(conf, stage):
                lines += ["", "[[stage.scoring.component]]", f"[stage.scoring.component.{comp_name}]"]
                for endpoint in endpoints:
                    lines += ["", f"[[stage.scoring.component.{comp_name}.endpoint]]"]
                    lines += _kv_lines(endpoint)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Example config
# ---------------------------------------------------------------------------

def default_yaml_text(model: str = "reinvent", run_type: str = "sampling") -> str:
    """Commented example YAML for a model / run type."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"model must be one of {sorted(SUPPORTED_MODELS)}")
    if run_type not in SUPPORTED_RUN_TYPES:
        raise ValueError(f"run_type must be one of {sorted(SUPPORTED_RUN_TYPES)}")
    needs_input = model in MODELS_NEEDING_INPUT
    input_hint = {
        "libinvent": "scaffolds with attachment points, e.g. [*:0]c1ccccc1[*:1]",
        "linkinvent": "warhead pairs, e.g. [*:0]c1ccccc1|[*:1]c1ccncc1",
        "mol2mol": "seed molecules to optimise, one SMILES per line",
        "pepinvent": "masked peptide inputs",
    }.get(model, "")
    text = f"""# ChemFlow -> REINVENT4 configuration.
# chemflow generate reinvent validate <this.yaml>
# chemflow generate reinvent run <this.yaml> [--dry-run]

model: {model}                 # reinvent | libinvent | linkinvent | mol2mol | pepinvent
run_type: {run_type}          # sampling | staged_learning (reinforcement learning)
reinvent_command: reinvent     # executable from the REINVENT4 conda env
out_dir: reinvent_output
device: cuda:0                 # cuda:0 | cpu
# seed: 42

# Pretrained model weights: download from https://doi.org/10.5281/zenodo.15641296
# (they are not in the REINVENT4 repo) and use the FULL path here.
prior_file: {DEFAULT_PRIORS[model]}
"""
    if needs_input:
        text += f"smiles_file: input.smi          # {input_hint}\n"
    if run_type == "sampling":
        text += """
num_smiles: 1000               # (per input line for lib/link/mol2mol)
sample_strategy: multinomial   # mol2mol: multinomial | beamsearch
unique_molecules: true
randomize_smiles: true
"""
    else:
        text += """
batch_size: 100
learning_strategy: {type: dap, sigma: 128, rate: 0.0001}
diversity_filter:
  type: IdenticalMurckoScaffold
  bucket_size: 25
  minscore: 0.4
  minsimilarity: 0.4
  penalty_multiplier: 0.5

# Plug ChemFlow-trained predictors into the reward. Needs `chemflow`
# importable in `python` (default: the interpreter running this command).
# chemflow_scorers:
#   - name: logD
#     model: models/logd_model.pkl
#     column: prediction          # or prob_class_1 for classifiers
#     weight: 1.0
#     transform: {type: double_sigmoid, low: 1.0, high: 3.0, coef_div: 3.0, coef_si: 10.0, coef_se: 10.0}

stages:
  - chkpt_file: stage1.chkpt
    termination: simple
    max_score: 0.7
    min_steps: 25
    max_steps: 200
    scoring:
      type: geometric_mean
      components:
        - {name: QED, weight: 1.0}
"""
    return text
