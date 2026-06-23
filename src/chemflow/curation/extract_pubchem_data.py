from dataclasses import dataclass
import io
import time
import requests
import pandas as pd
from typing import Iterable, List
from src.config import CONFIG

BASE = CONFIG["pubchem"]["base_url"]

@dataclass
class PipelineConfig:
    pause_sec: float = 0.25
    batch_size: int = 200
    properties: str = "CanonicalSMILES,IsomericSMILES,InChI,InChIKey"
    retries: int = 4
    timeout_sec: int = 180
    ligand_id_col: str = "CID"


def run_request(url, retries=4, timeout_sec=180, pause_sec=0.25):
    headers = {"User-Agent": "PubChemPipeline/1.0"}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout_sec,
            )
            response.raise_for_status()

            if not response.text.strip():
                raise RuntimeError("Empty PubChem response.")

            time.sleep(pause_sec)
            return response.text

        except Exception as e:
            wait = min(8.0, 0.5 * (2 ** (attempt - 1)))
            print(
                f"Request failed ({attempt}/{retries}): {e}. "
                f"Retrying in {wait:.1f}s."
            )
            time.sleep(wait)

    raise RuntimeError(f"Request failed after {retries} attempts: {url}")

def chunked(xs: List[int], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def fetch_pubchem_target_assays(uniprot_id, cfg):
    url = f"{BASE}/assay/target/accession/{uniprot_id}/concise/CSV"

    text = run_request(
        url,
        retries=cfg.retries,
        timeout_sec=cfg.timeout_sec,
        pause_sec=cfg.pause_sec,
    )

    df = pd.read_csv(io.StringIO(text))

    if "Target Accession" in df.columns:
        df = df[df["Target Accession"].astype(str).str.strip() == uniprot_id].copy()

    return df


def add_compounds(cfg: PipelineConfig, cids: Iterable[int]) -> pd.DataFrame:
    cid_list = sorted({int(c) for c in cids if pd.notna(c)})

    if not cid_list:
        return pd.DataFrame(columns=["CID"] + cfg.properties.split(","))

    frames = []

    for batch in chunked(cid_list, cfg.batch_size):
        cid_str = ",".join(map(str, batch))

        url = (
            f"{BASE}/compound/cid/"
            f"{cid_str}/property/"
            f"{cfg.properties}/CSV"
        )

        text = run_request(
            url,
            retries=cfg.retries,
            timeout_sec=cfg.timeout_sec,
            pause_sec=cfg.pause_sec,
        )

        df_props = pd.read_csv(io.StringIO(text))

        df_props["CID"] = pd.to_numeric(
            df_props["CID"],
            errors="coerce",
        ).astype("Int64")

        frames.append(df_props)

    out = pd.concat(frames, ignore_index=True).dropna(subset=["CID"])
    out["CID"] = out["CID"].astype(int)
    out = out.drop_duplicates(subset=["CID"])

    return out
