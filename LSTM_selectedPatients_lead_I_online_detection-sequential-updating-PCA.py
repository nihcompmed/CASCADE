#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import norm
import wfdb
import pandas as pd
import time
import random
import json  # for storing arrays as strings
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.decomposition import PCA
from numpy.random import default_rng
import torch
import torch.nn as nn
import torch.optim as optim

# =======================================================
# Fixed Parameters
# =======================================================

# ====== LOAD REAL ECG ======
fs = 360  # Sampling frequency in Hz
data_folder = 'mitdb_data_full'
annotation_folder = 'mit-bih-arrhythmia-database-1.0.0'

NORMAL_BEATS = ['.', 'N', 'L', 'R', 'e', 'j']
ARRHYTHMIC_BEATS = ['A', 'a', 'J', 'S', 'V', 'E', 'F']

# ====== CONFIG ======
epochs = 100
lr = 1e-3
num_layers = 1

pca_comp = 3

# ====== Parameters ======
beat_length = 180  # samples before the beat
num_train_beats = 500
num_val_beats = 100

num_test_norm = 100
num_test_arr = 100

# ====== FUNCTION: Extract beat segments ======
def get_segments(signal, ann_samples, ann_symbols, beat_length, target_symbols):
    segments = []
    indices = []
    for s, sym in zip(ann_samples, ann_symbols):
        if sym in target_symbols and s >= beat_length:
            seg = signal[s - beat_length:s]
            segments.append(seg)
            indices.append(s)
    return segments, indices

# Convert numpy arrays inside online_results so they become JSON-serializable
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj
    
# ====== BEAT-BASED DATA PREP (no continuous time) ======
def sliding_data(segment, input_len, pred_len):
    """
    Generate input-output pairs (X, Y) for one 1D segment.
    Returns:
        X_y : (input_len x num_windows)
        Y   : (pred_len x num_windows)
    """
    X_y, Y = [], []
    for t_idx in range(len(segment) - input_len - pred_len + 1):
        X_y.append(segment[t_idx : t_idx + input_len])
        Y.append(segment[t_idx + input_len : t_idx + input_len + pred_len])
    return np.array(X_y).T, np.array(Y).T

# ====== Define LSTM model ======
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden_dim)
        out = self.tanh(out[:, -1, :])   # Take last time step + Tanh activation
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used:", device)

# =======================================================
# Parameters for the loops (can be modified as per requirement)
# =======================================================
#selected_patients = [
#    '215','213','209','203','210','116','222','233',
#    '118','223','221','214','200','228','201','208',
#    '119','207','106'
#]
#random_seeds = [42, 46, 2] # Random seeds

record = sys.argv[1]
data_seed = int(sys.argv[2])
res_seed = int(sys.argv[3])
input_len = int(sys.argv[4])
batch_size = int(sys.argv[5])
hidden_dim = int(sys.argv[6])

prediction_lengths = [1, 10] # Pred_len

# =======================================================
# CHECK IF RESULTS ALREADY EXIST (SKIP FULL EXPERIMENT)
# =======================================================

save_dir = f"results_all_selected_patients_LSTM"
os.makedirs(save_dir, exist_ok=True)

results_csv = os.path.join(
    save_dir,
    f"results_summary_LSTM_online_detection_PCA_all_patient_{record}_data_seed_{data_seed}_res_seed_{res_seed}_input_len_{input_len}_batch_size_{batch_size}_hidden_dim_{hidden_dim}.csv"
)

if os.path.exists(results_csv):
    df_existing = pd.read_csv(results_csv)

    df_existing["Patient"] = df_existing["Patient"].astype(str).str.strip()
    df_existing["Data_seed"] = df_existing["Data_seed"].astype(int)
    df_existing["Reservoir_seed"] = df_existing["Reservoir_seed"].astype(int)
    df_existing["input_len"] = df_existing["input_len"].astype(int)
    df_existing["Pred_len"] = df_existing["Pred_len"].astype(int)
    df_existing["batch_size"] = df_existing["batch_size"].astype(int)
    df_existing["hidden_dim"] = df_existing["hidden_dim"].astype(int)

    existing_configs = set(
        zip(
            df_existing["Patient"],
            df_existing["Data_seed"],
            df_existing["Reservoir_seed"],
            df_existing["input_len"],
            df_existing["Pred_len"],
            df_existing["batch_size"],
            df_existing["hidden_dim"]
        )
    )
else:
    existing_configs = set()

print(f"=========> Running for patient {record}")

# ====== Load annotations ======
record_path = os.path.join(annotation_folder, record)
ann = wfdb.rdann(record_path, extension='atr')

# ====== Load signal ======
signal_path = os.path.join(data_folder, f'{record}.npy')
signal = np.load(signal_path)[:, 0]  # Lead I only

# ============ REPRODUCIBLITY BLOCK ==============
# -------------------------------
# Separate seeds
# -------------------------------
data_seed = data_seed              # FIXED for reproducibility
reservoir_seed = res_seed       # this will vary per run

rng_data = default_rng(data_seed)
rng_res  = default_rng(reservoir_seed)

# ====== Extract normal and arrhythmic beats ======
normal_segments, normal_indices = get_segments(signal, ann.sample, ann.symbol, beat_length, NORMAL_BEATS)
arr_segments, arr_indices = get_segments(signal, ann.sample, ann.symbol, beat_length, ARRHYTHMIC_BEATS)

# ---- Basic numbers ----
num_total_norm = len(normal_segments)

# ---- Random split for normal beats ----
all_norm_indices = np.arange(num_total_norm)

# Shuffle the indices to randomize selection
rng_data.shuffle(all_norm_indices)

# Choose random validation indices *within* normal beats
val_indices = all_norm_indices[:num_val_beats]
train_indices = all_norm_indices[num_val_beats:num_val_beats + num_train_beats]
test_norm_indices = all_norm_indices[num_val_beats + num_train_beats:
                                     num_val_beats + num_train_beats + num_test_norm]

# ---- Create splits ----
train_segments = [normal_segments[i] for i in train_indices]
val_segments = [normal_segments[i] for i in val_indices]
test_segments = [normal_segments[i] for i in test_norm_indices] + arr_segments[:num_test_arr]

# ====== Convert to numpy arrays ======
train_segments = np.array(train_segments)
val_segments = np.array(val_segments)
test_segments = np.array(test_segments)

# In[2]:

# ============ RUNNING EXPERIMRENTS ==============

for pred_len in prediction_lengths:



    # ====== CONCATENATE ALL SEGMENTS FOR TRAINING SINGLE S (NO TIME) ======
    all_X_y_raw = []
    all_Y_raw = []
    
    for seg_idx, seg in enumerate(train_segments):
        if len(seg) < input_len + pred_len:
            print(f"Skipping segment {seg_idx}, too short for sliding window")
            continue
    
        # ====== SLIDING WINDOW ======
        X_y_raw, Y_raw = sliding_data(seg, input_len, pred_len)
    
        if X_y_raw.size == 0 or Y_raw.size == 0:
            print(f"Skipping segment {seg_idx}, sliding window returned empty arrays")
            continue
    
        # Collect raw (unscaled) data for global normalization later
        all_X_y_raw.append(X_y_raw)
        all_Y_raw.append(Y_raw)
    
    # ====== CONCATENATE RAW WINDOWS ACROSS ALL SEGMENTS ======
    X_y_all_raw = np.hstack(all_X_y_raw)  # (input_len, total_samples)
    Y_all_raw   = np.hstack(all_Y_raw)    # (pred_len, total_samples)
    
    # print(f"Raw training shapes: X_y_all_raw={X_y_all_raw.shape}, Y_all_raw={Y_all_raw.shape}")
    
    # PCA-transform the input windows
    pca_train = PCA(n_components=pca_comp)
    X_y_all_pca = pca_train.fit_transform(X_y_all_raw.T)   # shape: (samples, 3) 3 for 3 components
    
    # DSRN requires (features × samples)
    X_y_all_norm = X_y_all_pca.T    # shape: (3, samples)
    
    # Output remains ECG windows
    Y_all_norm = Y_all_raw      # shape: (samples, pred_len)
    
    # print(f"PCA training data shapes: X_y_all_norm = {X_y_all_norm.shape}, Output data shape (Y_all_norm) = {Y_all_norm.shape}")
    
    # ====== Prepare data for LSTM ======
    # Training arrays: X_y_all_norm (input_len, total_samples)
    # We reshape to (total_samples, input_len, 1) for LSTM
    X_lstm = X_y_all_norm.T.reshape(-1, pca_comp, 1)
    Y_lstm = Y_all_norm.T  # (total_samples, pred_len)
    
    X_t = torch.from_numpy(X_lstm).float().to(device)
    Y_t = torch.from_numpy(Y_lstm).float().to(device)
    
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
    # ====== Initialize model, loss, optimizer ======
    model = LSTMRegressor(input_dim=1, hidden_dim=hidden_dim, output_dim=pred_len, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # ====== Train LSTM ======
    model.train()
    for epoch in range(epochs):

        # tic = time.time()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(dataset)
        # toc = time.time()
        # print(f'time taken for one epoch {round(toc - tic,4)}')

        if (epoch % 10 == 0) or (epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss:.6e}")
    
    print("\nLSTM training complete.")
                
    # ====== PREDICTION ON EACH SEGMENT (LSTM) ======
    model.eval()
    all_predictions = []
    all_true = []
    
    for seg_idx, seg in enumerate(train_segments):
        if len(seg) < input_len + pred_len:
            continue
    
        predictions = []
        start = input_len
    
        while start + pred_len <= len(seg):
            # Prepare one input window (shape: (1, input_len))
            y_in = seg[start - input_len:start].reshape(1, input_len)
    
            # --- PCA transform the input window ---
            y_in_norm = pca_train.transform(y_in)   # shape: (1, pca_comp)
    
            # Reshape for LSTM (batch=1, seq_len=20, input_dim=1)
            x_t = torch.from_numpy(y_in_norm.reshape(1, pca_comp, 1)).float().to(device)
    
            # Predict
            with torch.no_grad():
                y_pred_norm = model(x_t).cpu().numpy().reshape(1, pred_len)
    
            # Inverse transform
            y_pred = y_pred_norm.flatten()
    
            predictions.extend(y_pred)
            start += pred_len
    
        # Align prediction length with true segment
        pred_seq = np.array(predictions[:len(seg) - input_len])
        true_seq = seg[input_len:input_len + len(pred_seq)]
    
        all_predictions.append(pred_seq)
        all_true.append(true_seq)
    
    print("Prediction done for all segments (LSTM).")

    r_values = []
    mse_values = []
    rmse_values = []
    
    for pred_seq, true_seq in zip(all_predictions, all_true):
        # Do NOT slice true_seq again
        if len(true_seq) != len(pred_seq):
            continue  # skip if lengths mismatch
    
        # ---- Compute metrics ----
        if np.std(pred_seq) > 0 and np.std(true_seq) > 0:
            r = np.corrcoef(true_seq, pred_seq)[0, 1]        # Pearson correlation coefficient
            mse = mean_squared_error(true_seq, pred_seq)    # Mean squared error
            rmse = np.sqrt(mse)                              # Root mean squared error
    
            r_values.append(r)
            mse_values.append(mse)
            rmse_values.append(rmse)
        else:
            print("Skipping segment due to zero std (constant values)")
    
    # ---- Average metrics ----
    avg_r = np.mean(r_values)
    avg_mse = np.mean(mse_values)
    avg_rmse = np.mean(rmse_values)
    
    # ====== VALIDATION PREDICTIONS (LSTM) ======
    model.eval()
    val_predictions = []
    val_true = []
    
    for seg_idx, seg in enumerate(val_segments):
        if len(seg) < input_len + pred_len:
            continue
    
        predictions = []
    
        start = input_len
        while start + pred_len <= len(seg):
            # Prepare one input window (shape: (1, input_len))
            y_in = seg[start - input_len:start].reshape(1, input_len)
    
            # --- PCA transform the input window ---
            y_in_norm = pca_train.transform(y_in)   # shape: (1, pca_comp)
    
            # Reshape for LSTM (batch=1, seq_len=20, input_dim=1)
            x_t = torch.from_numpy(y_in_norm.reshape(1, pca_comp, 1)).float().to(device)
    
            # Predict using trained LSTM
            with torch.no_grad():
                y_pred_norm = model(x_t).cpu().numpy().reshape(1, pred_len)
    
            # Inverse transform
            y_pred = y_pred_norm.flatten()
    
            predictions.extend(y_pred)
            start += pred_len
    
        # Align predicted and true sequences
        pred_seq = np.array(predictions[:len(seg) - input_len])
        true_seq = seg[input_len:input_len + len(pred_seq)]
    
        val_predictions.append(pred_seq)
        val_true.append(true_seq)
    
    print("Validation prediction done for all segments (LSTM).")
    
    # ====== Compute validation metrics ======
    r_values, mse_values, rmse_values = [], [], []
    for pred_seq, true_seq in zip(val_predictions, val_true):
        if len(pred_seq) != len(true_seq):
            continue
        if np.std(pred_seq) > 0 and np.std(true_seq) > 0:
            r = np.corrcoef(true_seq, pred_seq)[0, 1]
            mse = mean_squared_error(true_seq, pred_seq)
            rmse = np.sqrt(mse)
            r_values.append(r)
            mse_values.append(mse)
            rmse_values.append(rmse)
    
    avg_r_val = np.mean(r_values)
    avg_mse_val = np.mean(mse_values)
    avg_rmse_val = np.mean(rmse_values)
    
    # ====== TEST PREDICTIONS (LSTM) ======
    model.eval()
    test_predictions = []
    test_true = []
    
    TEST_BEATS = NORMAL_BEATS + ARRHYTHMIC_BEATS
    
    for seg_idx, seg in enumerate(test_segments):
        if len(seg) < input_len + pred_len:
            continue
    
        predictions = []
    
        start = input_len
        while start + pred_len <= len(seg):
            # Prepare one input window (1 × input_len)
            y_in = seg[start - input_len:start].reshape(1, input_len)
    
            # --- PCA transform the input window ---
            y_in_norm = pca_train.transform(y_in)   # shape: (1, pca_comp)
    
            # Reshape for LSTM (batch=1, seq_len=20, input_dim=1)
            x_t = torch.from_numpy(y_in_norm.reshape(1, pca_comp, 1)).float().to(device)
    
            # Predict with trained LSTM
            with torch.no_grad():
                y_pred_norm = model(x_t).cpu().numpy().reshape(1, pred_len)
    
            # Inverse transform
            y_pred = y_pred_norm.flatten()
    
            predictions.extend(y_pred)
            start += pred_len
    
        # Align predictions with true sequence
        pred_seq = np.array(predictions[:len(seg) - input_len])
        true_seq = seg[input_len:input_len + len(pred_seq)]
    
        test_predictions.append(pred_seq)
        test_true.append(true_seq)
    
    print("Test prediction done for all segments (LSTM).")

    # ====== Compute metrics ======
    r_values, mse_values, rmse_values = [], [], []
    
    for pred_seq, true_seq in zip(test_predictions, test_true):
        if len(pred_seq) != len(true_seq):
            continue
        if np.std(pred_seq) > 0 and np.std(true_seq) > 0:
            r = np.corrcoef(true_seq, pred_seq)[0, 1]
            mse = mean_squared_error(true_seq, pred_seq)
            rmse = np.sqrt(mse)
            r_values.append(r)
            mse_values.append(mse)
            rmse_values.append(rmse)
    
    avg_r_test = np.mean(r_values)
    avg_mse_test = np.mean(mse_values)
    avg_rmse_test = np.mean(rmse_values)

    # ====== COMPUTE mu_i and sigma_i from validation predictions ======
    L = beat_length - input_len
    all_val_preds = np.array([p[:L] for p in val_predictions])
    
    mu_i = np.mean(all_val_preds, axis=0)
    sigma_i = np.std(all_val_preds, axis=0)
    
    dx = 1 / 360  # time interval (seconds)
    
    # ====== STEP 1: Compute thresholds from validation normal beats ======
    window_lengths = list(range(1, beat_length - input_len + 1))  # 1 to 160 points
    threshold_dict = {}  # store threshold for each window size
    
    for w in window_lengths:
        log_probs_normal_w = []
    
        for pred_seq in val_predictions:
            # truncate or pad to w
            if len(pred_seq) < w:
                pred_seq_pad = np.pad(pred_seq, (0, w - len(pred_seq)), 'edge')
            else:
                pred_seq_pad = pred_seq[:w]
    
            pdf_vals = norm.pdf(pred_seq_pad, loc=mu_i[:w], scale=sigma_i[:w] + 1e-8)
            log_p = np.sum(np.log(pdf_vals + 1e-10) + np.log(dx))
            log_probs_normal_w.append(log_p)
    
        # threshold = 5th percentile of validation normals
        threshold_dict[w] = np.percentile(log_probs_normal_w, 5)
    
    print(" Thresholds computed for all window lengths.")

    # ====== GENERATE GROUND TRUTH LABELS FOR TEST SET ======
    # Test segments contain first 5 normal + 10 arrhythmic beats
    # We can create labels directly:
    # 0 = normal, 1 = arrhythmic
    test_labels = [0] * num_test_norm + [1] * num_test_arr  # length must match len(test_segments)
    
    
    # =======================================================
    # PREPARE STORAGE
    # =======================================================
    num_segs = len(test_segments)
    
    # cumulative log p for each test segment
    cum_log_p = np.zeros(num_segs)     # log(p1 * p2 * ... * pt) = sum(log p_i)
    
    # anomaly flag for each segment (per time)
    anomaly_flags_over_time = []       # list of lists length beat_length
    accuracy_over_time = []
    precision_over_time = []
    recall_over_time = []
    f1_over_time = []
    online_results = []
    
    # final anomaly flag for each segment at current t
    current_flags = np.zeros(num_segs, dtype=int)

    # =======================================================
    # REAL-TIME LOOP
    # =======================================================
    model.eval()
    for t in range(input_len, beat_length):
    
        # for this timepoint, evaluate all segments
        for seg_idx, seg in enumerate(test_segments):
    
            # ========= 1) GET INPUT WINDOW =========
            # Prepare one input window (1 × input_len)
            y_in = seg[t - input_len : t].reshape(1, input_len)
            # --- PCA transform the input window ---
            y_in_norm = pca_train.transform(y_in)   # shape: (1, pca_comp)

            # Reshape for LSTM (batch=1, seq_len=20, input_dim=1)
            x_t = torch.from_numpy(y_in_norm.reshape(1, pca_comp, 1)).float().to(device)

            # ========= 2) PREDICT NEXT POINT =========
            # Predict with trained LSTM
            with torch.no_grad():
                y_pred_norm = model(x_t).cpu().numpy().reshape(1, pred_len)
    
            # Inverse transform to original scale
            y_pred = y_pred_norm.flatten()         
            pred_val = y_pred[0]
            obs_val  = seg[t]    # true value at time t
    
            # ========= 3) REFERENCE DISTRIBUTION (from train) =========
            mu_t = mu_i[t - input_len]
            sigma_t = sigma_i[t - input_len]
    
            # ========= 4) COMPUTE LOG PDF (your logic) =========
            pdf_val = norm.pdf(pred_val, loc=mu_t, scale=sigma_t + 1e-8)
            log_p = np.log(pdf_val + 1e-10) + np.log(dx)
    
            # update cumulative log p
            cum_log_p[seg_idx] += log_p
    
            # ========= 5) CHECK ANOMALY THRESHOLD =========
            threshold_val = threshold_dict[t - input_len + 1]
            # compare cumulative log p
            if cum_log_p[seg_idx] < threshold_val:
                current_flags[seg_idx] = 1
            else:
                current_flags[seg_idx] = 0           
    
        # =======================================================
        # AFTER PROCESSING ALL SEGMENTS AT TIME t:
        # COMPUTE ACCURACY / F1 / PRECISION / RECALL
        # =======================================================
        ann_true = np.array(test_labels)     # true arrhythmia labels
        ann_pred = current_flags.copy()
    
        acc = accuracy_score(ann_true, ann_pred)
        pre = precision_score(ann_true, ann_pred, zero_division=0)
        rec = recall_score(ann_true, ann_pred, zero_division=0)
        f1 = f1_score(ann_true, ann_pred, zero_division=0)
    
        accuracy_over_time.append(acc)
        precision_over_time.append(pre)
        recall_over_time.append(rec)
        f1_over_time.append(f1)
    
        anomaly_flags_over_time.append(current_flags.copy())
    
        # Store only the current values (not entire lists)
        online_results.append({
            "time": t,
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1,
            "flags": current_flags.copy()
        })

    online_results_serializable = convert_numpy(online_results)


    # ====== PREPARE METRICS DICTIONARY ======
    results_dict = {

        # ---- CONFIGS ----
        "Patient": record,
        "Data_seed": data_seed,
        "Reservoir_seed": reservoir_seed,
        "input_len": input_len,
        "Pred_len": pred_len,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        
        "Epochs": epochs,
        "Learning_rate": lr,
    
        # ---- TRAINING ----
        "Train_MSE": avg_mse,
        "Train_r": avg_r,
        "Train_RMSE": avg_rmse,
    
        # ---- VALIDATION ----
        "Val_MSE": avg_mse_val if 'avg_mse_val' in locals() else np.nan,
        "Val_r": avg_r_val if 'avg_r_val' in locals() else np.nan,
        "Val_RMSE": avg_rmse_val if 'avg_rmse_val' in locals() else np.nan,
    
        # ---- TEST ----
        "Test_MSE": avg_mse_test if 'avg_mse_test' in locals() else np.nan,
        "Test_r": avg_r_test if 'avg_r_test' in locals() else np.nan,
        "Test_RMSE": avg_rmse_test if 'avg_rmse_test' in locals() else np.nan,
        
        # ---- ONLINE DETECTION ----
        # ---- Online results saved as JSON string ----
        "Online_results": json.dumps(online_results_serializable),
    }
    
    # ====== CREATE CSV IF NOT EXISTS ======
    if not os.path.exists(results_csv):
        df_init = pd.DataFrame(columns=list(results_dict.keys()))
        df_init.to_csv(results_csv, index=False)
    
    # ====== READ EXISTING CSV ======
    df_existing = pd.read_csv(results_csv)
    
    # Append new row
    df_result = pd.DataFrame([results_dict])
    df_combined = pd.concat([df_existing, df_result], ignore_index=True)
    df_combined.to_csv(results_csv, index=False)
    print(f" Results saved/updated in CSV")


# In[ ]:



