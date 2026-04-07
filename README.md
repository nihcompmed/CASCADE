# CASCADE: Chaotic Attractor Sensitivity for Cardiac Anomaly Detection
## Overview

This repository contains the implementation and experiments for CASCADE (Chaotic Attractor Sensitivity for Cardiac Anomaly Detection), an online, patient-specific framework for ECG arrhythmia detection.

The method leverages entropy-tuned chaotic reservoir computing (DynML) to model short-term ECG dynamics and identify anomalies as failures of predictability. The method reframes arrhythmia detection as a dynamical regime transition problem using Dynamical Systems Machine Learning (DynML) and compares it with standard machine learning models including MLP, LSTM, and TCN.

The system is evaluated on the MIT-BIH Arrhythmia Database with preprocessing, PCA-based dimensionality reduction, and online prediction-based anomaly detection.

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
