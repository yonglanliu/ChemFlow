# src/config.py

from pathlib import Path
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parent

with open(PROJECT_ROOT / "conf.toml", "rb") as f:
    CONFIG = tomllib.load(f)


# from src.conf import UNIPROT

# url = UNIPROT["url"]
# timeout = UNIPROT["timeout"]
# fields = ",".join(UNIPROT["fields"])

# if __name__ == "__main__":
#     print(Path(__file__).resolve())
#     print(Path(__file__).resolve().parent)
#     print(Path(__file__).resolve().parents[1])
#     print(Path(__file__).resolve().parents[2])