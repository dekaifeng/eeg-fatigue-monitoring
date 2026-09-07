"""Reproduce public EEG evidence in a fresh directory with an exact run manifest."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from eeg_fatigue.datasets import FIGSHARE_FILENAME, FIGSHARE_SHA256, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw") / FIGSHARE_FILENAME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extended", action="store_true", help="also run robustness and epoch replay"
    )
    args = parser.parse_args()
    if sha256_file(args.dataset) != FIGSHARE_SHA256:
        raise ValueError("dataset checksum does not match the declared Figshare v3 artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    phases = [("baseline", "eeg_fatigue.real_cli", ["--config", "configs/figshare.yaml"])]
    if args.extended:
        phases += [
            (
                "robustness",
                "eeg_fatigue.robustness_cli",
                [
                    "--study-config",
                    "configs/figshare.yaml",
                    "--robustness-config",
                    "configs/robustness.yaml",
                ],
            ),
            (
                "streaming",
                "eeg_fatigue.streaming_cli",
                [
                    "--study-config",
                    "configs/figshare.yaml",
                    "--streaming-config",
                    "configs/streaming.yaml",
                ],
            ),
        ]
    manifest = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset_sha256": FIGSHARE_SHA256,
        "dataset_doi": "10.6084/m9.figshare.14273687.v3",
        "status": "running",
        "commands": [],
        "completed": [],
    }
    manifest_path = args.output / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    shutil.copytree(root / "configs", args.output / "configs")
    shutil.copy(root / "data/README.md", args.output / "DATASET_ATTRIBUTION.md")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output / "requirements.txt").write_text(freeze, encoding="utf-8")
    try:
        for name, module, flags in phases:
            command = [
                sys.executable,
                "-m",
                module,
                "--dataset",
                str(args.dataset.resolve()),
                *flags,
                "--output",
                str((args.output / name).resolve()),
            ]
            manifest["commands"].append(command)
            save()
            with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest["completed"].append(name)
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["file_sha256"] = {
            str(path.relative_to(args.output)): sha256_file(path)
            for path in sorted(args.output.rglob("*"))
            if path.is_file() and path != manifest_path
        }
        save()


if __name__ == "__main__":
    main()
