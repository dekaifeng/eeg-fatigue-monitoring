# Limitations and safety

- Phase 1 uses synthetic EEG with deliberately separable spectral signatures.
- Synthetic scores measure pipeline regression behavior, not human fatigue-detection performance.
- No clinical, workplace, driver, or safety-critical decision should use this software.
- Amplitude-only artifact rejection cannot handle all eye, muscle, motion, or electrode artifacts.
- The current features and classifier are baselines, not a validated physiological biomarker.
- Cross-subject evaluation reduces subject leakage but does not establish external validity.
- No hardware acquisition, end-to-end online latency, calibration drift, or demographic bias has
  been tested.
- The Phase 2 Figshare artifact is a balanced, author-selected derivative; it does not represent the
  natural prevalence of drowsiness and cannot estimate deployment predictive values.
- The extracted MATLAB artifact does not include channel labels. Phase 2 therefore uses only
  channel-agnostic aggregates and does not make scalp-region or electrode-specific claims.
- Reaction-time-derived labels quantify task behavior under a simulator protocol; they are not
  clinical sleep-stage annotations or a general definition of fatigue.
- Phase 3 confidence intervals and sign-flip tests operate on only 11 subject folds. They quantify
  uncertainty inside this selected dataset and are not a population-level validation study.
- Threshold and feature-ablation comparisons are exploratory and are not corrected for multiple
  comparisons; the unchanged no-rejection baseline remains primary.
- Pooled calibration curves can hide subject-level miscalibration. Mean improvements are small and
  inconsistent across subjects, so calibrated probabilities are not suitable for operational risk
  decisions.
- The Phase 4 replay uses independently published three-second epochs. Their stored order is not
  established as a continuous recording, so alarm transitions and active-epoch counts are software
  demonstrations rather than detection-delay or event-level performance estimates.
- Missing-channel masks are simulated, independent across epochs, and repaired by a simple median
  policy. They do not reproduce electrode impedance changes, correlated sensor failures, or motion
  artifacts.
- The latency benchmark measures feature extraction and inference on one Ubuntu workstation only.
  It excludes acquisition, transport, storage, concurrent load, and hard real-time guarantees; one
  57.55 ms maximum was observed among 150 trials.
