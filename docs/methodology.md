# Methodology

## Phase 1 objective

Phase 1 validates the software pipeline with deterministic synthetic EEG. It does not estimate
performance on people. Alert and fatigued epochs contain explicit, documented spectral shifts,
subject-specific amplitude and frequency variation, sensor noise, and sparse motion-like pulses.

## Signal processing

Eight channels are sampled at 128 Hz in non-overlapping four-second epochs. Epochs exceeding the
configured peak-to-peak threshold are rejected before a fourth-order Butterworth 1–40 Hz band-pass
filter is applied through MNE-Python. Welch power spectral density estimates support delta, theta,
alpha, and beta relative power; theta/beta and (theta+alpha)/beta ratios; spatial dispersion;
spectral entropy; and RMS amplitude.

## Evaluation protocol

The primary model is standardized logistic regression. A prior-only dummy classifier provides a
sanity-check baseline and a fixed random forest tests a nonlinear conventional model. Complete
subjects are assigned to folds: no epoch from a held-out subject can enter that fold's training set.
Phase 1 uses five-fold `GroupKFold`; Phase 2 uses leave-one-subject-out evaluation. Balanced
accuracy, F1, and ROC AUC are reported per fold and as mean ± population standard deviation.

The random seed, configuration, per-fold metrics, and out-of-fold predictions are all retained for
auditability.

## Phase 2 public-data protocol

The Figshare v3 artifact is an author-prepared balanced subset, not raw acquisition data. Its 2,022
three-second samples were extracted from a sustained-attention lane-departure experiment. Labels
are based on subject-relative local and 90-second global reaction times; moderate states are excluded.
The publisher reports that the source had already undergone filtering, manual blink rejection,
EEGLAB AAR processing, and resampling to 128 Hz. Consequently, Phase 2 does not apply another
band-pass filter or a second amplitude-based rejection step. It checks that all samples are finite,
calculates channel-agnostic spectral summaries, and evaluates all 11 subjects with
leave-one-subject-out folds. Amplitude-threshold sensitivity is reserved for a separately reported
robustness experiment rather than silently changing the publisher's balanced subset.

## Phase 3 validation hierarchy

Phase 3 keeps the 11-subject leave-one-subject-out loop as the outer evaluation boundary.

- Artifact thresholds are sensitivity conditions rather than model-selection choices.
- Feature groups are specified before evaluation and compared under the same outer folds.
- Percentile bootstrap intervals resample the 11 subject-fold scores, not individual EEG windows.
- The exact sign-flip test enumerates every sign assignment of subject-level improvement over
  balanced-accuracy chance (0.5).
- Random-forest sigmoid calibration is fitted from grouped out-of-fold predictions generated only
  from the outer training subjects. The held-out subject is used once for final measurement.
- Logistic-regression `C` is selected with an inner five-fold `GroupKFold` among outer training
  subjects. The full outer training set is then refitted before evaluating the untouched subject.

Calibration is evaluated with Brier loss, log loss, ECE, and reliability curves. Brier and log loss
combine probability reliability with discrimination, so they are interpreted alongside ECE and
the curve rather than as pure calibration measures.

## Phase 4 replay and fault policy

The Phase 2 random-forest baseline and its leave-one-subject-out predictions remain unchanged.
Published epoch identifiers define a deterministic accelerated replay order within each subject;
the monitor is reset between subjects. A drowsy alarm requires three consecutive probabilities at
or above 0.65. It clears after two consecutive probabilities at or below 0.35. Missing or
non-finite predictions enter a separate `DATA_FAULT` state instead of being interpreted as normal
or drowsy.

For sensitivity analysis, a seeded mask removes the same number of randomly selected channels from
each epoch. Up to eight missing channels are replaced sample-by-sample with the median across that
epoch's available channels, after which features and all LOSO predictions are regenerated. More
than eight missing channels cause rejection without inference. The condition with no removed
channels is the unchanged Phase 2 baseline.

The latency microbenchmark trains on subjects 2-11 and measures 150 subject-1 epochs. Each timed
operation includes feature extraction from one three-second epoch and random-forest
`predict_proba`. Five untimed warm-up operations precede measurement. Loading the MATLAB file,
acquiring or transmitting EEG, and operating-system scheduling guarantees are outside the timed
scope.
