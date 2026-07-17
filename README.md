# neuroBrain
# Brain Tumor MRI Segmentation

A research-oriented deep learning project for brain tumor segmentation using the Medical Segmentation Decathlon (MSD) dataset.

## Objectives

- Explore and analyze 3D brain MRI data
- Build an efficient segmentation pipeline
- Train and evaluate a deep learning model
- Compare lightweight approaches suitable for limited computational resources

## Dataset

Medical Segmentation Decathlon - Brain Tumour Dataset </br>
http://medicaldecathlon.com/dataaws/ </br>
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar

## Dataset Overview

The dataset contains multimodal MRI scans with expert-annotated tumor segmentation masks.

The first exploratory visualization compares the four MRI modalities together with the corresponding segmentation mask.

![MRI Modalities](figures/01_modalities_overview.png)

---

## MRI Preprocessing

Before training a deep learning model, MRI intensity distributions were analyzed to determine an appropriate preprocessing strategy.

### Research Questions

- Are MRI intensities standardized across modalities?
- How much of each MRI volume consists of background voxels?
- What normalization strategy is appropriate for this dataset?

### Key Findings

- MRI modalities exhibit different intensity distributions.
- Histograms reveal a dominant spike at zero intensity due to background voxels.
- Including background voxels biases intensity statistics.
- Therefore, Z-score normalization is performed using only non-zero voxels.

### Implementation

A reusable preprocessing module was implemented in:

src/preprocessing.py

which performs non-zero voxel normalization for each MRI modality.

### Normalization Example

![Normalization Comparison](figures/02_normalization_comparison.png)


## Dataset Analysis

### MRI Modalities

![MRI Modalities](figures/01_modalities_overview.png)

---

### Intensity Normalization

![Normalization](figures/02_normalization_comparison.png)

---

### Single Patient Class Distribution

![Single Patient](figures/03_single_patient_class_distribution.png)

---

### Log Scale Distribution

![Log Distribution](figures/04_single_patient_class_distribution_log.png)

---

### Dataset-wide Class Distribution

![Dataset Distribution](figures/05_dataset_class_distribution.png)

## Project Structure

```text
neuroBrain/
├── docs/
├── figures/
├── models/
├── notebooks/
├── results/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
└── Task01_BrainTumour/
```

## Status

## Project Progress

### Milestone 1 — Dataset Exploration & Preprocessing

- [x] Repository setup
- [x] Dataset download and inspection
- [x] MRI modality visualization
- [x] Intensity distribution analysis
- [x] Non-zero voxel normalization
- [x] Single-patient tumor class distribution
- [x] Dataset-wide class imbalance analysis
- [x] Reusable preprocessing module
- [x] Centralized project configuration

### Upcoming

- [ ] PyTorch Dataset
- [ ] MONAI transforms
- [ ] 3D U-Net implementation
- [ ] Model training
- [ ] Evaluation
- [ ] Inference

