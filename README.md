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

## Repository Structure

```
notebooks/
src/
figures/
results/
README.md
```

## Status

## Project Progress

-  Repository setup [done]
-  Dataset exploration [done]
-  MRI modality visualization [done]
-  Preprocessing [InProgress...]
-  Baseline 3D U-Net
-  Model evaluation
-  Final report
