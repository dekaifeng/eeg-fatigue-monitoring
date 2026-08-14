"""Download and verify the Phase 2 public EEG dataset."""

from __future__ import annotations

import argparse

from eeg_fatigue.datasets import FIGSHARE_FILENAME, download_figshare_dataset, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"data/raw/{FIGSHARE_FILENAME}")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    path = download_figshare_dataset(arguments.output, overwrite=arguments.overwrite)
    print(f"verified {path} sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
