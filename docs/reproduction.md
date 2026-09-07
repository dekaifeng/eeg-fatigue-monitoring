# Public-data evidence bundle

The fast CI verifies synthetic-data behavior. The separate **Public EEG
reproduction** workflow downloads and checksum-verifies the declared Figshare
v3 file, then runs the full subject-held-out baseline. Enable `extended` manually
to include robustness and epoch replay. Its artifact contains per-epoch outputs,
logs, exact installed dependencies, configuration copies, Git SHA, output hashes
and dataset attribution. The raw MATLAB dataset is not included.

Local equivalent from the repository root:

```bash
python -m pip install -e '.[dev]'
eeg-fatigue-download
python scripts/reproduce_public.py --output tmp/public-evidence
# Or use a new directory for the longer study:
python scripts/reproduce_public.py --output tmp/extended-evidence --extended
```

Output directories must be new, preserving published and previous evidence.
Inspect `manifest.json`: `completed` means every selected subprocess exited
successfully; failed runs retain their logs and completed-phase list. Timing
measurements remain specific to their recorded host. Dependency snapshots are
evidence of the run environment, not a promise of bitwise cross-platform output.

These runs reuse the existing Figshare population. They are independent software
reproductions, not validation on a second dataset or continuous real-time EEG.
