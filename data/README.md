# Public dataset

Phase 2 uses **EEG driver drowsiness dataset, version 3**, published by Jian Cui on
Figshare under CC BY 4.0:

- Article: <https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687>
- DOI: <https://doi.org/10.6084/m9.figshare.14273687.v3>
- File ID: `30707285`
- Expected size: `180829497` bytes
- SHA-256: `53cc4ef14b1343f7f3fb5322dd2b541c031c6c2297bddf1817aed04dc687a6a4`

The dataset contains 2,022 balanced alert/drowsy samples from 11 subjects. Each sample
contains 30 EEG channels, 384 time points, and represents three seconds at 128 Hz. The
repository does not redistribute this 172 MiB file. Download and verify it with:

```bash
eeg-fatigue-download
```

The publisher extracted these samples from Cao et al.'s sustained-attention driving-task
dataset. Cui et al. describe the preparation and labeling protocol: the source data had
already been filtered and artifact-processed; three-second windows before lane deviation
were downsampled to 128 Hz; alert/drowsy labels were derived from subject-relative local
and global reaction-time thresholds; intermediate states were excluded; and the published
v3 subset was balanced within every subject.

Required attribution:

1. Cui, J. (2021). *EEG driver drowsiness dataset* (Version 3) [Data set]. Figshare.
   <https://doi.org/10.6084/m9.figshare.14273687.v3>
2. Cui, J., Lan, Z., Liu, Y., Li, R., Li, F., Sourina, O., & Müller-Wittig, W. (2021).
   A Compact and Interpretable Convolutional Neural Network for Cross-Subject Driver
   Drowsiness Detection from Single-Channel EEG. *Methods*, 202,
   <https://doi.org/10.1016/j.ymeth.2021.04.017>.
3. Cao, Z., Chuang, C.-H., King, J.-T., & Lin, C.-T. (2019). Multi-channel EEG recordings
   during a sustained-attention driving task. *Scientific Data*, 6, 19.
   <https://doi.org/10.1038/s41597-019-0027-4>.

The downloaded artifact is excluded by `.gitignore`; its CC BY 4.0 license is separate
from this repository's MIT-licensed code.
