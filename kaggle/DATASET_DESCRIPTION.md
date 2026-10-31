# InGesture - Eight-Class Inertial Gesture Dataset

## Overview

This Kaggle dataset republishes the public **InGesture v5** data for an
eight-class gesture recognition project using TF-C and DINOv3 encoders.
It is not a newly self-collected private smartphone dataset.

Original source: [InGesture v5](https://doi.org/10.17632/fdxst56tcj.5).
Dataset attribution/license: CC BY 4.0; retain the original attribution.
Device: wrist-worn WT901BLECL5 IMU. Nominal sample rate: approximately 200 Hz.

## Dataset Structure and Verification

| Item | Verified value in the project copy |
|---|---:|
| Subjects | 50 |
| Gesture classes | 8 |
| Recognized segmented recordings | 5,345 |
| X/Y/Z sample rows in these recordings | 7,958,400 |
| Generated overlapping windows (206 samples, stride 103) | 68,854 |
| Recordings too short to generate a window | 9 |

The recording count includes only files matching a recognized
`subject_*/gesture__*.csv` pattern. The archive can contain additional original
recordings or auxiliary files; those are not added to these segmented counts.
Window counts describe the project's preprocessing, not extra independent
recordings stored in the original data.

```text
InGesture/
  subject_1/
    fluid_intake__*.csv
    answering_cell_phone__*.csv
    ...
  subject_2/
  ...
  subject_50/
```

## Columns and Annotations

| Field | Meaning |
|---|---|
| time | Original time values; the visualization does not infer their units |
| x, y, z | Acceleration axes used by the project, in g |
| gx, gy, gz | Gyroscope columns present in original files; unused by the current model |
| Gesture label | Inferred from the segmented filename, not an extra per-row label column |

| ID | Gesture |
|---:|---|
| 0 | other |
| 1 | fluid_intake |
| 2 | answering_cell_phone |
| 3 | scratching_head |
| 4 | passing_hand_over_face |
| 5 | adjusting_glasses_touching_temples |
| 6 | holding_chin |
| 7 | stretching_with_hands_behind_neck |

`other` is a genuine target class, not a void/ignore label. Labels apply at
recording level. Exact gesture onset/offset annotations are not inferred.

## Visualization Notebook

The companion **InGesture Dataset Visualization for TF-C and DINOv3** notebook
contains inventory/format checks, seeded sample selection, X/Y/Z traces,
signal-label-window alignment, FFT, examples across subjects, coverage,
class/split distributions, duration/quality, gesture mean ± SD profiles,
normalization/correlation and initial DINOv3 adapter images.

The updated notebook also includes statistical-feature PCA, normalized channel
RMS, STFT plots and exact recording-duplicate checks, inspired by the public
[Sensor Logger notebook](https://www.kaggle.com/code/vuvandat000099/sensor-logger-five-gesture-dataset-visualization).
All plots are recomputed for InGesture's eight labels and three acceleration
axes. The notebook has 20 executed code cells and 14 plot outputs.

In the audited local copy, 85 recordings repeat an earlier cleaned float32
X/Y/Z array. No exact recording hash is shared across the subject splits.
The published counts above include duplicate files; this is not a claim of
5,345 unique signal arrays. Inspect duplicate_recordings.csv before creating
a deduplicated data version. Window-level and approximate duplicates are not
covered by this recording audit.

The notebook distinguishes the full inventory from the illustrative sample
pool (up to 12 distinct training subjects per class, seed 42). Selected paths
are exported in `sample_pool.csv`. All plots use actual data.

After publishing the companion notebook, add its actual Kaggle URL here.
The prepared notebook file alone is not a published Kaggle notebook link.

## Preprocessing and Evaluation Notes

The project splits by subject (32 train / 8 validation / 10 test), fits clipping
and normalization on train only, then creates 206-sample windows with stride
103. Different splits share no subjects. Do not split overlapping windows
randomly across train and test.

`other` accounts for approximately 75.50% of generated windows. Report macro
F1, balanced accuracy and per-class recall alongside accuracy. The data comes
from a wrist IMU; generalization to smartphone recordings needs separate
validation. This dataset visualization does not establish model accuracy.

## References

- Original dataset: https://doi.org/10.17632/fdxst56tcj.5
- Project: https://github.com/Tiemanh/TFC-Dino_V3
- Presentation structure inspired by: https://www.kaggle.com/code/loanhviet/sentinel-2-rpg-dataset-visualization
