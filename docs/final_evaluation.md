# neuroBrain — Final Evaluation

## Experiment

- Selected checkpoint: Epoch 18
- Stored checkpoint metric: 0.7628627273
- Validation patients: 97
- Random seed: 42
- ROI size: (96, 96, 96)
- Sliding-window overlap: 0.25

## Final Patient-Level Validation Results

| Metric | Mean Dice | Std | Min | Max |
|---|---:|---:|---:|---:|
| Background | 0.9988 | 0.0007 | 0.9964 | 0.9998 |
| Edema | 0.7061 | 0.1401 | 0.2466 | 0.9096 |
| Non-enhancing tumor | 0.4738 | 0.2450 | 0.0000 | 0.9537 |
| Enhancing tumor | 0.7106 | 0.2375 | 0.0000 | 0.9348 |
| Whole Tumor (WT) | 0.8450 | 0.1106 | 0.3540 | 0.9679 |
| Tumor Core (TC) | 0.7352 | 0.2028 | 0.0769 | 0.9690 |
| Enhancing Tumor (ET) | 0.7106 | 0.2375 | 0.0000 | 0.9348 |
| Mean Tumor Dice | 0.6302 | 0.1436 | 0.1426 | 0.8690 |


## Metric Definitions

- Background = class 0.
- Edema = class 1.
- Non-enhancing tumor = class 2.
- Enhancing tumor = class 3.
- Whole Tumor (WT) = labels 1 + 2 + 3.
- Tumor Core (TC) = labels 2 + 3.
- Enhancing Tumor (ET) = label 3.
- Mean Tumor Dice = arithmetic mean of class 1, class 2, and class 3 Dice.

The checkpoint-selection metric is reported separately from Mean Tumor Dice because checkpoint selection uses the composite WT/TC/ET regions.

These results represent the fixed 97-patient validation set and are not hidden test-set results.

## Inference

Inference used MONAI sliding-window inference:

- ROI: 96 × 96 × 96
- Sliding-window batch size: 1
- Overlap: 0.25

## Persistence

The patient-level and aggregate evaluation artifacts are also backed up to the project's Hugging Face checkpoint repository.
