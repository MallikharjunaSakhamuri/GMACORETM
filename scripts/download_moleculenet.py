"""Download the MoleculeNet benchmark CSV files from the DeepChem mirror.

Example
-------
    python scripts/download_moleculenet.py --datasets bace bbbp clintox esol qm9
    python scripts/download_moleculenet.py --all
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gmacore.data.moleculenet import BENCHMARKS


def download(url: str, destination: str) -> None:
    """Fetch a file, transparently decompressing a gzip payload."""
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)

    if url.endswith(".gz"):
        archive = destination + ".gz"
        urllib.request.urlretrieve(url, archive)
        with gzip.open(archive, "rb") as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target)
        os.remove(archive)
    else:
        urllib.request.urlretrieve(url, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MoleculeNet benchmarks.")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--root", default="data/moleculenet")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = parser.parse_args()

    names = sorted(BENCHMARKS) if args.all else [name.lower() for name in args.datasets]
    if not names:
        raise SystemExit("Specify --datasets or --all.")

    for name in names:
        if name not in BENCHMARKS:
            print(f"Skipping unknown benchmark '{name}'")
            continue

        spec = BENCHMARKS[name]
        destination = os.path.join(args.root, spec.filename)

        if os.path.exists(destination) and not args.force:
            print(f"{spec.name}: already present at {destination}")
            continue
        if spec.url is None:
            print(f"{spec.name}: no download URL registered, obtain the file manually")
            continue

        print(f"{spec.name}: downloading to {destination}")
        try:
            download(spec.url, destination)
        except Exception as error:  # noqa: BLE001
            print(f"{spec.name}: download failed ({error}).")
            print("  Retrieve the file manually from https://moleculenet.org/datasets")


if __name__ == "__main__":
    main()
