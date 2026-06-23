import io
import tomllib
from pathlib import Path

import pandas as pd
import requests
from src.config import CONFIG


def query_gene(gene: str) -> pd.DataFrame:
    config = CONFIG["uniprot"]

    url = config["url"]
    timeout = config["timeout"]
    fields_string = ",".join(config["fields"])

    query_string = f"gene:{gene.strip()}"

    params = {
        "query": query_string,
        "format": "tsv",
        "fields": fields_string,
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        if len(response.text.strip().split("\n")) <= 1:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(response.text), sep="\t")
        df["query_gene"] = gene.strip()

        return df

    except requests.RequestException as e:
        print(f"UniProt API error for {gene}: {e}")
        return pd.DataFrame()

    except Exception as e:
        print(f"Unexpected error for {gene}: {e}")
        return pd.DataFrame()



# if __name__ == "__main__":
#     test_gene = "BRCA1"
#     result_df = query_gene(test_gene)
#     print(result_df.head())