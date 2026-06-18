# CASCADE: Chaotic Attractor Sensitivity for Cardiac Anomaly Detection
## Overview

This repository contains the implementation and experiments for CASCADE (Chaotic Attractor Sensitivity for Cardiac Anomaly Detection), an online, patient-specific framework for ECG arrhythmia forecasting.

The method leverages entropy-tuned chaotic reservoir computing (DynML) to model short-term ECG dynamics and identify anomalies as failures of predictability. The method reframes arrhythmia detection as a dynamical regime transition problem using Dynamical Systems Machine Learning (DynML) and compares it with standard machine learning models including MLP, LSTM, and TCN.

The system is evaluated on the MIT-BIH Arrhythmia Database with preprocessing, PCA-based dimensionality reduction, and online prediction-based anomaly forecasting.

This repository accompanies the manuscript:
**"From Chaos to Care: Personalized AI for Early Cardiac Arrhythmia Warning"**

---

## Dataset Information

### MIT-BIH Raw Dataset

The repository includes:
- mit-bih-arrhythmia-database-1.0.0.zip

This contains the original ECG records downloaded from PhysioNet:
MIT-BIH Arrhythmia Database (PhysioNet ECG dataset)

It includes:
- Raw ECG waveform signals
- Beat annotations
- Multiple patient recordings

---

### Processed Numpy Dataset

- mitdb_data_full.zip

This contains preprocessed data:
- Converted .npy files for all patients
- Beat-segmented ECG signals
- Feature-aligned data used in ML/DL pipelines

This dataset is used for:
- Training
- Validation
- Online detection experiments

---

## Repository Structure

### Core Models

- DynML_PCA_online_selectedPatients.py  
  Main CASCADE implementation using DynML + PCA + online detection

- MLP_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py  
  MLP baseline model

- LSTM_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py  
  LSTM baseline model

- TCN_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py  
  TCN baseline model

---

### Data Generation / Training Pipelines

- generator_script_DynML_different_seeds_N.py  
  DynML reservoir simulations across different seeds and configurations

- generator_script_MLP.py  
  MLP training and prediction pipeline

- generator_script_LSTM.py  
  LSTM training and prediction pipeline

- generator_script_TCN.py  
  TCN training and prediction pipeline

---

### Analysis / Notebooks

- MultiMethod_comparision.ipynb  
  Comparison of DynML, MLP, LSTM, TCN

- PCA_data_Patient106.ipynb  
  Patient-specific PCA visualization

- PCA_data_total_variance_selected_patients.ipynb  
  Variance analysis across selected patients

- beat_length_selection.ipynb  
  Beat segmentation optimization

- error_histogram_validation_set.ipynb  
  Error distribution analysis

- MITBIH_data_patients_selection.ipynb  
  Patient selection from MIT-BIH dataset

---

## Methodology Summary

1. Data Preprocessing  
   - ECG extraction from MIT-BIH dataset  
   - Beat segmentation and normalization   

2. Dimensionality Reduction  
   - PCA applied to ECG features  
   - Retains dominant dynamical structure  

3. Models  
   - DynML (chaotic reservoir computing)  
   - MLP baseline  
   - LSTM baseline  
   - TCN baseline  

4. Online Detection  
   - Sequential prediction of ECG beats  
   - Error-based anomaly detection  
   - Detection triggered by deviation from normal dynamics  

---

## Key Idea

CASCADE treats arrhythmia detection as:

A breakdown of predictability in a nonlinear dynamical system.

Anomalies are detected when:
- Prediction error increases consistently  
- The system deviates from learned normal dynamics  

---

## Key Features

- Online beat-by-beat prediction  
- Dynamical Systems Machine Learning (DynML)  
- Entropy/chaos-based reservoir dynamics  
- Patient-specific modeling  
- Trained only on normal beats  
- Detection via prediction error  
- Multi-model benchmarking framework  

---

## Comparisons

Models evaluated:
- DynML
- MLP
- LSTM
- TCN

Evaluated across:
- Multiple patients
- Different prediction horizons
- Online detection setting

---



# New writeup here



# CASCADE: Chaotic Attractor Sensitivity for Cardiac Anomaly Detection

## Overview

This repository contains the full implementation, data generation pipelines, and analysis notebooks for CASCADE (Chaotic Attractor Sensitivity for Cardiac Anomaly Detection), an online, personalized framework for ECG arrhythmia detection based on Dynamical Systems Machine Learning (DynML).

CASCADE reframes arrhythmia detection as a dynamical regime transition problem: anomalies are identified as failures of short-term predictability, quantified via statistically significant deviations between predicted and observed cardiac dynamics relative to patient-specific baselines.

This repository accompanies the manuscript:
**"From Chaos to Care: Personalized AI for Early Cardiac Arrhythmia Warning"**
Suvankar Halder, Christopher M. Kim, Vipul Periwal
Laboratory of Biological Modeling, NIDDK, National Institutes of Health

---

## Dataset Information

### MIT-BIH Arrhythmia Database

Standard benchmark ECG dataset from PhysioNet, used for primary evaluation. Includes raw waveforms, beat annotations, and multi-patient recordings sampled at 360 Hz.

Download: https://physionet.org/content/mitdb/1.0.0/

### Icentia11k Dataset

Large-scale ambulatory ECG database comprising continuous single-lead recordings from 11,000 patients at 250 Hz, used for external validation.

Download: https://physionet.org/content/icentia11k-continuous-ecg/1.0/

Preprocessed `.npy` files (beat-segmented, feature-aligned) used across all pipelines are generated by the core model scripts listed below.

---

## Repository Structure

### Core Models — MIT-BIH

- DynML_PCA_online_selectedPatients.py
  Main CASCADE implementation: DynML + PCA + online anomaly detection (Figs. 9–11, 13, 14, 17)

- MLP_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py
  MLP baseline with sequential PCA updating and online detection

- LSTM_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py
  LSTM baseline

- TCN_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py
  TCN baseline

---

### Core Models — Icentia11k (External Validation)

- DynML_icentia11kdata_selectedPatients_PCA_online.py
  CASCADE applied to Icentia11k dataset (Fig. 19, SuppFigs. 5–6)

- MLP_icentia11kdata_selectedPatients_PCA_online.py
  MLP baseline for Icentia11k

- LSTM_icentia11kdata_selectedPatients_PCA_online.py
  LSTM baseline for Icentia11k

- TCN_icentia11kdata_selectedPatients_PCA_online.py
  TCN baseline for Icentia11k

---

### Extended DynML

- DynML_selectedPatients_lead_I_online_detection-PCA-w1-10-multiParaSet_seeds.py
  DynML runs across multiple reservoir parameter sets and seeds for extended entropy-regime analysis

---

### Data Generation / Job Scripts — MIT-BIH

These scripts are designed for batch/cluster execution (e.g., SLURM swarm jobs) to generate result files across patients, seeds, and hyperparameter configurations.

- generator_script_different_seeds_N.py
  DynML runs across reservoir sizes (N) and random seeds

- generator_script_multiPara_seeds.py
  DynML runs across multiple reservoir parameter regimes and seeds (entropy sweep; Figs. 20–22)

- generator_script_MLP.py
  MLP training and prediction pipeline

- generator_script_LSTM.py
  LSTM training and prediction pipeline

- generator_script_TCN.py
  TCN training and prediction pipeline

- generator_DynML_timimg.py
  DynML inference timing measurements (SuppFig. 20)

- timing_cascade.py
  Per-sample latency and memory profiling across reservoir sizes

---

### Data Generation / Job Scripts — Icentia11k

- generator_script_DynML_multiPara_seeds_icentiaData.py
  DynML multi-parameter regime sweep for Icentia11k

- generator_script_MLP_icentiaData.py
  MLP pipeline for Icentia11k

- generator_script_LSTM_icentiaData.py
  LSTM pipeline for Icentia11k

- generator_script_TCN_icentiaData.py
  TCN pipeline for Icentia11k

---

### Analysis Notebooks

#### Patient and Data Selection

- Fig_3_beat_length_selection.ipynb
  Inter-beat interval distributions; justification of 180-sample window (Fig. 3)

- Fig_4_MITBIH_data_patients_selection.ipynb
  Patient inclusion criteria and beat count distributions (Fig. 4)

- Incentia_Check_for_enough_data.ipynb
  Icentia11k patient selection and data sufficiency checks

- SuppFig_4_Icentia11K_beatlength_selection.ipynb
  Beat length selection for Icentia11k (SuppFig. 4)

#### Dimensionality Reduction

- Fig_5_PCA_data_total_variance_selected_patients.ipynb
  PCA variance explained across patients and input window lengths (Fig. 5)

- Fig_6_PCA_data_Patient106.ipynb
  3D PCA projections of normal vs. arrhythmic beat windows for Patient 106 (Fig. 6)

#### Prediction Error Analysis

- Fig_7_error_histogram_validation_set.ipynb
  Gaussian structure of validation prediction errors and Q–Q plots (Fig. 7)

- SuppFig_7-9_error_histogram_validation_set.ipynb
  Per-patient and time-resolved normality analysis of prediction errors (SuppFigs. 7–9)

#### Model Comparison and Detection Performance

- Fig_9-11,13,14,17_SuppFig_1-3_MultiMethod_comparision.ipynb
  Online F1 score trajectories across DynML, MLP, LSTM, and TCN; all patients and prediction horizons (Figs. 9–11, 13, 14, 17; SuppFigs. 1–3)

#### Morphological Separability

- Fig_12_SuppFig_10,12,13_HDBSCAN_mitbih_clusterMetrices.ipynb
  HDBSCAN clustering, PCA overlap metrics, silhouette scores, and waveform difference analysis (Fig. 12; SuppFigs. 10, 12, 13)

- SuppFig_11_Arrhythmia type composition per patient alongside peak F1.ipynb
  Arrhythmia type breakdown per patient alongside peak F1 scores (SuppFig. 11)

#### Statistical Analysis and Entropy

- Fig_15,16,18,22_SuppFig_15-18_statistical_rigor_entropy_early_detection_results.ipynb
  Wilcoxon signed-rank tests, AUC-F1 distributions, fastest-method analysis, and entropy vs. detection speed (Figs. 15, 16, 18, 22; SuppFigs. 15–18)

- Fig_20,21_SuppFig_14_topo_figure_multipara-newtopocalculation.ipynb
  Topological entropy sweep across reservoir regimes; co-variation with AUC-F1 (Figs. 20–21; SuppFig. 14)

#### Bootstrap and Timing

- SuppFig_19_bootstap_ci_fast.ipynb
  Bootstrap confidence intervals (2,000 resamples) for F1 estimates across patients (SuppFig. 19)

- SuppFig_20_DynML_timimg_figure.ipynb
  Inference latency, training time, and model memory across reservoir sizes (SuppFig. 20)

---

## Methodology Summary

1. ECG Data Processing
   - Beat segmentation into 180-sample (500 ms) windows preceding each R-peak
   - Normal and arrhythmic beat split per patient
   - Training on normal beats only

2. Dimensionality Reduction
   - Patient-specific 3D PCA fitted exclusively on normal training beats
   - Explains >92% variance across all patients and window lengths

3. Dynamical Reservoir (DynML)
   - Ensemble of Rössler oscillators spanning stable-to-chaotic parameter regimes
   - Input windows initialize reservoir states; terminal states form high-dimensional embeddings
   - Reservoir complexity controlled via topological entropy

4. Linear Readout and Prediction
   - Trained on normal beats only
   - Generates one-step-ahead ECG predictions online without retraining

5. Probabilistic Anomaly Detection
   - Prediction errors modeled as Gaussian from validation data
   - Cumulative log-likelihoods monitored per beat
   - Beats flagged when likelihood drops below patient-specific 5th-percentile threshold

---

## Key Results

- CASCADE (DynML) significantly outperforms MLP, TCN, and LSTM under extended prediction horizons (AUC-F1: +0.067 to +0.107; all p < 0.01)
- Topological entropy monotonically controls reservoir complexity (Spearman r = 1.00) and correlates with detection accuracy and speed
- Higher-entropy reservoirs achieve earlier threshold crossing (Spearman r = −0.035 with time-to-F1 ≥ 0.70; p < 10⁻³)
- External validation on Icentia11k (50 patients) replicates MIT-BIH findings without dataset-specific modification
- Model weights: 0.8–17.0 KB; suitable for wearable deployment after ODE solver optimization

---

## Citation

If you use this code or data, please cite:

Halder S, Kim CM, Periwal V. From Chaos to Care: Personalized AI for Early Cardiac Arrhythmia Warning. [manuscript in preparation]

---

## Funding

Supported by the Intramural Research Program of the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK, ZIA DK075091-13), National Institutes of Health.

The findings and conclusions presented are those of the authors and do not necessarily reflect the views of the NIH or the U.S. Department of Health and Human Services.

---

## Data Availability and Code

All code is publicly available at: https://github.com/nihcompmed/CASCADE

ECG data are available from PhysioNet:
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- Icentia11k Dataset: https://physionet.org/content/icentia11k-continuous-ecg/1.0/