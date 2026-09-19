"""Install the offline Ketcher distribution used by ChemFlow Desktop."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src.desktop.widgets.ketcher_editor import KETCHER_VERSION


DOWNLOAD_URL = (
    "https://lifescience.opensource.epam.com/downloads/ketcher/"
    f"ketcher-v{KETCHER_VERSION}/ketcher-standalone-{KETCHER_VERSION}.zip"
)


def installation_root() -> Path:
    return Path.home() / ".cache" / "chemflow" / f"ketcher-{KETCHER_VERSION}"


def main() -> None:
    destination = installation_root()
    index = destination / "standalone" / "index.html"
    if index.is_file():
        print(f"Ketcher {KETCHER_VERSION} is already installed at {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chemflow-ketcher-") as temporary:
        archive = Path(temporary) / "ketcher.zip"
        print(f"Downloading Ketcher {KETCHER_VERSION}…")
        urllib.request.urlretrieve(DOWNLOAD_URL, archive)
        extracted = Path(temporary) / "extracted"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        if not (extracted / "standalone" / "index.html").is_file():
            raise RuntimeError("The downloaded Ketcher archive has an unexpected layout.")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(extracted), str(destination))
    print(f"Installed Ketcher {KETCHER_VERSION} at {destination}")


if __name__ == "__main__":
    main()
