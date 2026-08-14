# Phase 4: accelerated epoch replay and fault handling

## Purpose and boundary

This phase tests whether the Phase 2 classifier can be wrapped in a small, auditable monitoring
state machine. It does not claim live EEG acquisition, continuous-time replay, a validated alarm,
or hard real-time execution. The Figshare artifact contains independently published three-second
epochs, so their stored order is used only to make a deterministic accelerated demonstration.

## Monitor behavior

The monitor starts in `NORMAL`. Three consecutive probabilities greater than or equal to 0.65
enter `DROWSY`; two consecutive probabilities less than or equal to 0.35 return it to `NORMAL`.
Invalid or unavailable data immediately enter `DATA_FAULT`, reset both counters, and never produce
a drowsiness alarm. The next valid probability recovers to `NORMAL` before normal hysteresis
continues. Unit tests cover entry, exit, fault, recovery, ordering, and subject-boundary reset.

The complete 2,022-epoch replay produces 40 state transitions and 872 alarm-active epochs. These
counts verify deterministic state-machine execution only. They are not event sensitivity, false
alarm rate, or detection delay because the source does not establish continuous timing.

## Missing-channel sensitivity

Each evaluated condition removes a seeded random set of channels independently in every epoch.
For one through eight missing channels, every missing waveform is replaced with the sample-wise
median of the available channels in the same epoch. The spectral features and the complete
11-fold leave-one-subject-out evaluation are then regenerated.

| Missing channels | Status | Balanced accuracy | ROC AUC | Mean absolute probability shift |
|---:|---|---:|---:|---:|
| 0 | evaluated | 0.6853 | 0.7671 | 0.0000 |
| 1 | evaluated | 0.6839 | 0.7676 | 0.0276 |
| 4 | evaluated | 0.6832 | 0.7609 | 0.0399 |
| 8 | evaluated | 0.6709 | 0.7552 | 0.0546 |
| 16 | `DATA_FAULT` | not evaluated | not evaluated | not evaluated |

The eight-channel condition loses 0.0144 balanced-accuracy points relative to the unchanged
baseline. This is a controlled robustness result, not evidence that the same policy is safe for
physical electrode failure.

## Computational latency

The benchmark trains the fixed random forest on subjects 2-11 and times feature extraction plus
probability inference for 150 subject-1 epochs after five warm-up calls.

| Statistic | Latency |
|---|---:|
| Median | 26.25 ms |
| p95 | 27.00 ms |
| p99 | 28.02 ms |
| Maximum | 57.55 ms |

The maximum is retained rather than removed as an outlier. Measurements exclude file loading, EEG
acquisition, transport, visualization, and operating-system scheduling. They characterize only the
tested Python implementation and host.

## Reproduction

After downloading the checksum-pinned dataset described in `data/README.md`, run:

```bash
bash scripts/verify_streaming.sh
```

The command writes the event log, latency samples, missing-channel metrics, three PNG figures, and
an animated GIF under `results/streaming/`. The release versions only the compact summary,
missing-channel metrics, robustness figure, and GIF; detailed and redundant outputs remain ignored.
