# Phase 2: public-data baseline

## Scope

The experiment uses all 2,022 finite samples in the checksum-pinned Figshare v3 artifact: 1,011
alert and 1,011 drowsy windows from 11 subjects. It neither redistributes the dataset nor performs
a second filtering or amplitude-rejection pass on the publisher's preprocessed, balanced subset.

Each model is fixed before evaluation. All samples from one subject are held out together, and the
reported aggregate is the unweighted mean across the 11 leave-one-subject-out folds. Standardization
for logistic regression is fitted inside each training fold. No held-out fold is used for feature
selection, threshold selection, or hyperparameter tuning.

## Aggregate metrics

| Model | Balanced accuracy | F1 | ROC AUC |
|---|---:|---:|---:|
| Prior-only dummy | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.635 ± 0.099 | 0.619 ± 0.142 | 0.689 ± 0.109 |
| Random forest | **0.685 ± 0.099** | **0.685 ± 0.119** | **0.767 ± 0.103** |

The random forest is the best observed fixed baseline, but performance is heterogeneous: balanced
accuracy ranges from 0.559 for subject 7 to 0.882 for subject 9. Logistic regression falls below
chance for subject 2 (0.477). These results support the limited conclusion that the handcrafted
spectral summaries contain cross-subject signal in this selected dataset; they do not support a
claim of reliable monitoring for every new person.

## Interpretation boundary

The publisher's associated paper reports a compact single-channel CNN and a different modeling
pipeline. Its reported accuracy is not treated as a directly comparable target here because this
project reports balanced accuracy, averages subjects equally, aggregates 30 unnamed channels, and
uses fixed handcrafted-feature baselines. The current result is deliberately a transparent
baseline for later ablation and calibration work, not a reproduction claim.

The dataset itself is balanced by construction after reaction-time thresholding and selection.
Consequently, these results cannot estimate real-world drowsiness prevalence, positive predictive
value, operational false-alarm burden, clinical sleep stage, or driving safety.

See [data provenance](../data/README.md), [methodology](methodology.md), and
[limitations](limitations.md).
