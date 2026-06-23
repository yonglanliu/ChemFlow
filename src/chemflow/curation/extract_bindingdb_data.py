# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import time
from dataclasses import dataclass
import pandas as pd
from src.config import CONFIG
import requests

# ============================================================
# Config
# ============================================================
@dataclass
class BDBConfig:
    pause_sec: float = 0.25
    retries: int = 4
    connect_timeout_sec: int = 30
    read_timeout_sec: int = 300
    cutoff: int = 1000

    ligand_id_col: str = "monomerid"
    structure_col: str = "smile"
    activity_col: str = "affinity"
    activity_type_col: str = "affinity_type"

    standard_activity_col: str = "standard_value"
    standard_unit: str = "nM"

def get_default_index(options, target):
    return options.index(target) if target in options else 0


def clean_numeric_activity(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(">", "", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace("=", "", regex=False)
        .str.replace("~", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def convert_to_nM(values, unit):
    unit_factor = {
        "nM": 1.0,
        "uM": 1000.0,
        "µM": 1000.0,
        "μM": 1000.0,
        "M": 1_000_000_000.0,
        "pM": 0.001,
    }

    factor = unit_factor.get(unit, 1.0)
    return values * factor

# ============================================================
# BindingDB request and parser
# ============================================================
def fetch_bindingdb_by_uniprot(uniprot_id, cfg):
    url = CONFIG["BindingDB"]["api_url"]

    params = {
        "uniprot": uniprot_id,
        "cutoff": cfg.cutoff,
        "response": "application/json",
    }

    headers = {
        "User-Agent": "BindingDBDashboard/1.0",
    }

    for attempt in range(1, cfg.retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=(cfg.connect_timeout_sec, cfg.read_timeout_sec),
            )

            response.raise_for_status()

            data = response.json()

            root_key = next(iter(data.keys()))
            resp = data[root_key]

            possible_keys = [
                "bdb.affinities",
                "affinities",
                "affinity",
            ]

            rows = None

            for key in possible_keys:
                if key in resp:
                    rows = resp[key]
                    break

            if rows is None:
                raise KeyError(
                    f"Cannot find affinities in response. Keys={list(resp.keys())}"
                )

            if isinstance(rows, dict):
                if "affinity" in rows:
                    rows = rows["affinity"]
                elif "bdb.affinity" in rows:
                    rows = rows["bdb.affinity"]

            if not isinstance(rows, list):
                rows = [rows]

            df = pd.DataFrame(rows)

            time.sleep(cfg.pause_sec)

            return df

        except Exception as e:
            wait = min(8.0, 0.5 * (2 ** (attempt - 1)))
            print(
                f"BindingDB request failed for {uniprot_id} "
                f"({attempt}/{cfg.retries}): {e}. Retrying in {wait:.1f}s."
            )
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch BindingDB data for {uniprot_id}.")