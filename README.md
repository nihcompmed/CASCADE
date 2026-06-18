# CASCADE: Chaotic Attractor Sensitivity for Cardiac Anomaly Detection

## Overview

This repository contains the implementation of **CASCADE (Chaotic Attractor Sensitivity for Cardiac Anomaly Detection)**, an online, personalized framework for ECG arrhythmia forecasting based on **Dynamical Systems Machine Learning (DynML)**.

CASCADE models normal cardiac dynamics using entropy-tuned chaotic reservoirs and detects arrhythmias as failures of short-term predictability. Rather than treating arrhythmia detection as a conventional classification problem, the framework formulates it as the detection of dynamical regime transitions through online prediction.

The repository accompanies the manuscript:

**From Chaos to Care: Personalized AI for Early Cardiac Arrhythmia Warning**

---

# Repository Organization

## Core CASCADE (DynML) Models

### MIT-BIH Arrhythmia Database

- `DynML_PCA_online_selectedPatients.py`
- `DynML_selectedPatients_lead_I_online_detection-PCA-w1-10-multiParaSet_seeds.py`

Implementation of the proposed CASCADE framework using DynML with PCA-based preprocessing and online anomaly detection.

### Icentia11k Dataset

- `DynML_icentia11kdata_selectedPatients_PCA_online.py`

CASCADE implementation for external validation on the Icentia11k dataset.

---

## Baseline Models

The repository includes three baseline forecasting models evaluated under the same online prediction framework.

### Multi-Layer Perceptron (MLP)

- `MLP_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py`
- `MLP_icentia11kdata_selectedPatients_PCA_online.py`

### Long Short-Term Memory (LSTM)

- `LSTM_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py`
- `LSTM_icentia11kdata_selectedPatients_PCA_online.py`

### Temporal Convolutional Network (TCN)

- `TCN_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py`
- `TCN_icentia11kdata_selectedPatients_PCA_online.py`

---

## Data Generation and Experiment Scripts

These scripts generate prediction results used throughout the manuscript.

### DynML

- `generator_script_multiPara_seeds.py`
- `generator_script_different_seeds_N.py`
- `generator_script_DynML_multiPara_seeds_icentiaData.py`

### Baseline Models

- `generator_script_MLP.py`
- `generator_script_LSTM.py`
- `generator_script_TCN.py`

### Icentia11k Baselines

- `generator_script_MLP_icentiaData.py`
- `generator_script_LSTM_icentiaData.py`
- `generator_script_TCN_icentiaData.py`

### Runtime Analysis

- `generator_DynML_timimg.py`
- `timing_cascade.py`

Used for computational performance and timing analyses.

---

# Figure Generation Notebooks

Each notebook reproduces one or more figures from the manuscript.

| Notebook | Figures |
|----------|---------|
| `Fig_3_beat_length_selection.ipynb` | Figure 3 |
| `Fig_4_MITBIH_data_patients_selection.ipynb` | Figure 4 |
| `Fig_5_PCA_data_total_variance_selected_patients.ipynb` | Figure 5 |
| `Fig_6_PCA_data_Patient106.ipynb` | Figure 6 |
| `Fig_7_error_histogram_validation_set.ipynb` | Figure 7 |
| `Fig_9-11,13,14,17_SuppFig_1-3_MultiMethod_comparision.ipynb` | Figures 9–11, 13, 14, 17 and Supplementary Figures 1–3 |
| `Fig_15,16,18,22_SuppFig_15-18_statistical_rigor_entropy_early_detection_results.ipynb` | Figures 15, 16, 18, 22 and Supplementary Figures 15–18 |
| `Fig_20,21_SuppFig_14_topo_figure_multipara-newtopocalculation.ipynb` | Figures 20, 21 and Supplementary Figure 14 |
| `SuppFig_4_Icentia11K_beatlength_selection.ipynb` | Supplementary Figure 4 |
| `SuppFig_7-9_error_histogram_validation_set.ipynb` | Supplementary Figures 7–9 |
| `SuppFig_11_Arrhythmia type composition per patient alongside peak F1.ipynb` | Supplementary Figure 11 |
| `SuppFig_19_bootstap_ci_fast.ipynb` | Supplementary Figure 19 |
| `SuppFig_20_DynML_timimg_figure.ipynb` | Supplementary Figure 20 |
| `Fi_12_SuppFig_10,12,13_HDBSCAN_mitbih_clusterMetrices.ipynb` | Figure 12 and Supplementary Figures 10, 12, 13 |

---

## Utility Notebook

- `Incentia_Check_for_enough_data.ipynb`

Used for selecting eligible patients from the Icentia11k dataset.

---

# Workflow

The typical workflow is:

1. Prepare ECG datasets (MIT-BIH or Icentia11k).
2. Generate PCA-transformed beat windows.
3. Train DynML and baseline forecasting models.
4. Perform online prediction and anomaly detection.
5. Generate manuscript figures using the corresponding notebooks.
6. Compute statistical analyses, entropy analyses, and timing benchmarks.

---

# Features

- Personalized patient-specific modeling
- Online beat-by-beat arrhythmia forecasting
- Chaotic reservoir computing (DynML)
- Entropy-guided reservoir design
- Probabilistic anomaly detection
- Comparative benchmarking against MLP, LSTM, and TCN
- External validation on Icentia11k
- Statistical evaluation with bootstrap confidence intervals
- Topological entropy analysis
- Runtime and computational efficiency analysis

---

# Datasets

Experiments were performed using:

- MIT-BIH Arrhythmia Database
- Icentia11k ECG Dataset

Please obtain the datasets from their original sources and organize them according to the paths used in the scripts.

---

