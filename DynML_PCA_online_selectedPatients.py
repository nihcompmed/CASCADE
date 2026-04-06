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
steps = 1

# Epsilon values for entropy calculations
# eps_values = [0.9, 0.5, 0.1, 0.05, 1e-2, 1e-4, 1e-6, 1e-10] # Finer granularity
# eps_values = [1e-6] # take lower value of epsilon where the topological entropy is already saturated for low no of reservoirs
eps_values = [0.9]

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

# ====== DSRN SYSTEM ======
def rossler_system(t, X, params, N):
    X = X.reshape(N, 3)
    dxdt = (1 / params[:, 3]) * (-X[:, 1] - X[:, 2])
    dydt = (1 / params[:, 3]) * (X[:, 0] + params[:, 0] * X[:, 1])
    dzdt = (1 / params[:, 3]) * (params[:, 1] + X[:, 2] * (X[:, 0] - params[:, 2]))
    return np.stack([dxdt, dydt, dzdt], axis=1).flatten()

# ============ COMPUTE ENTROPY ==============
def compute_topological_entropy(trajectory, eps_values):
    entropy_estimates = {}
    for eps in eps_values:
        unique_orbits = set()
        for point in trajectory:
            rounded_point = tuple(np.round(point / eps))  
            unique_orbits.add(rounded_point)
        N_n_eps = len(unique_orbits)
        entropy_estimates[eps] = np.log(N_n_eps) / len(trajectory)  
    return entropy_estimates

# ====== Modified compute_phi to accept fixed params ======
def compute_phi(X_norm, N, steps, R=None, params=None):
    M, D = X_norm.shape

    # Use fixed params if passed, otherwise generate new ones
    if params is None:
        raise ValueError("Please pass fixed params for reproducible experiments")

    params_used = params[:N]  # take only the first N reservoirs

    # Compute mean and std per parameter
    param_means = np.mean(params_used, axis=0)  # shape: (4,)
    param_stds  = np.std(params_used, axis=0)   # shape: (4,)

    if R is None:
        R = np.random.uniform(-1, 1, size=(3 * N, M))

    phi = np.zeros((3 * N, D))
    entropy_all = []

    for i in range(D):
        u = R @ X_norm[:, i]
        # t_eval = np.linspace(0, 100, 1000 * steps)
        t_eval = np.linspace(0, 20, 100 * steps)
        sol = solve_ivp(rossler_system, [t_eval[0], t_eval[-1]], u,
                        t_eval=t_eval, args=(params_used, N), rtol=1e-6, atol=1e-9)
        phi[:, i] = sol.y[:, -1]

        trajectory = sol.y.T
        trajectory = (trajectory - trajectory.mean(axis=0)) / trajectory.std(axis=0)
        entropy_values = compute_topological_entropy(trajectory, eps_values)
        entropy_all.append(np.mean(list(entropy_values.values())))

    mean_entropy = np.mean(entropy_all)
    
    # Return phi, R, params used, mean entropy, and param statistics
    return phi, R, params_used, mean_entropy, param_means, param_stds

# ====== Train DSRN with fixed parameter distribution ======
def train_dsrn(X_norm, Y_norm, N, steps, params_fixed):
    # Pass fixed params to compute_phi
    phi, R, params_used, _, param_means, param_stds = compute_phi(X_norm, N, steps, params=params_fixed)
    
    # Compute linear mapping S
    S, _, _, _ = np.linalg.lstsq(phi.T, Y_norm.T, rcond=None)
    
    return S.T, R, params_used, param_means, param_stds


# ====== Predict DSRN with fixed parameter distribution ======
def predict_dsrn(X_norm, S, R, params_fixed, N, steps):
    # Use the same fixed params
    phi, _, _, entropy_values, _, _ = compute_phi(X_norm, N, steps, R=R, params=params_fixed)
    return S @ phi, entropy_values

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
N = int(sys.argv[5])


prediction_lengths = [1, 10] # Pred_len

# =======================================================
# CHECK IF RESULTS ALREADY EXIST (SKIP FULL EXPERIMENT)
# =======================================================

save_dir = f"results_all_selected_patients_w1-10_N"
os.makedirs(save_dir, exist_ok=True)

results_csv = os.path.join(
    save_dir,
    f"results_summary_online_detection_PCA_all_patient_{record}_data_seed_{data_seed}_res_seed_{res_seed}_input_len_{input_len}_reservoirs_{N}.csv"
)

if os.path.exists(results_csv):
    df_existing = pd.read_csv(results_csv)

    df_existing["Patient"] = df_existing["Patient"].astype(str).str.strip()
    df_existing["Data_seed"] = df_existing["Data_seed"].astype(int)
    df_existing["Reservoir_seed"] = df_existing["Reservoir_seed"].astype(int)
    df_existing["input_len"] = df_existing["input_len"].astype(int)
    df_existing["Pred_len"] = df_existing["Pred_len"].astype(int)
    df_existing["N"] = df_existing["N"].astype(int)

    existing_configs = set(
        zip(
            df_existing["Patient"],
            df_existing["Data_seed"],
            df_existing["Reservoir_seed"],
            df_existing["input_len"],
            df_existing["Pred_len"],
            df_existing["N"]
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

# ====== Generate parameter distribution that spans stable → chaotic ======

max_N = 200  # total reservoirs
tau_range = (0.8, 2.0)

# Define the extreme ends
stable_range = [(0.02, 0.07),  (0.02, 0.07),  (6.0, 7.0)]
chaotic_range = [(0.18, 0.22), (0.18, 0.22), (5.5, 5.9)]

# ====== Create hybrid parameter ranges ======
# For example, interpolate 30% stable → 70% chaotic
mix_ratio = 1   # 0 = purely stable, 1 = purely chaotic

hybrid_range = []
for (s_low, s_high), (c_low, c_high) in zip(stable_range, chaotic_range):
    low  = s_low  + mix_ratio * (c_low  - s_low)
    high = s_high + mix_ratio * (c_high - s_high)
    hybrid_range.append((low, high))

# ====== Generate parameters from this blended range ======
params_fixed = np.array([
    [rng_res.uniform(*r) for r in hybrid_range] + [rng_res.uniform(*tau_range)]
    for _ in range(max_N)
])

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

    # PCA-transform the input windows
    pca_train = PCA(n_components=3)
    X_y_all_pca = pca_train.fit_transform(X_y_all_raw.T)   # shape: (samples, 3) 3 for 3 components

    # DSRN requires (features × samples)
    X_y_all_norm = X_y_all_pca.T    # shape: (3, samples)

    # Output remains ECG windows
    Y_all_norm = Y_all_raw      # shape: (samples, pred_len)

    # ------------------------------
    # Skip if already computed
    # ------------------------------

    if (str(record), int(data_seed), int(reservoir_seed), int(input_len), int(pred_len), int(N)) in existing_configs:
        print(f"Skipping existing config: Patient={record}, Data_seed={data_seed}, Reservoir_seed = {reservoir_seed}, input_len={input_len}, Pred_len={pred_len}, N={N}")
        continue
            
    print(f"Running experiment for N = {N}")

    # ====== TRAIN SINGLE DSRN ======
    S, R, params, param_means, param_stds = train_dsrn(X_y_all_norm, Y_all_norm, N, steps, params_fixed)
    print(f"--> Training done for Patient={record}, Data_seed={data_seed}, Reservoir_seed = {reservoir_seed}, input_len={input_len}, Pred_len={pred_len}, N={N}.")

    # ====== PREDICTION ON EACH SEGMENT (NO TIME) ======
    all_predictions = []
    all_true = []
    all_entropy_values = []  # store average entropy from each segment
    
    for seg_idx, seg in enumerate(train_segments):
        if len(seg) < input_len + pred_len:
            continue
    
        predictions = []
        segment_entropy_sum = 0.0
        segment_entropy_count = 0
    
        start = input_len
        while start + pred_len <= len(seg):
            # input = last `input_len` true values
            y_in = seg[start - input_len:start].reshape(-1, 1)

            # --- PCA transform the input window ---
            y_in_pca = pca_train.transform(y_in.T).T   # shape (3,1)


            # --- predict in raw scale ---
            y_pred_raw, entropy_values = predict_dsrn(
                y_in_pca, S, R, params, N, steps
            )
            
            # no inverse transform needed
            y_pred = y_pred_raw.flatten()

            predictions.extend(y_pred)
    
            # --- accumulate entropy ---
            if isinstance(entropy_values, dict):
                avg_entropy = np.mean(list(entropy_values.values()))
                segment_entropy_sum += avg_entropy
                segment_entropy_count += 1
            elif isinstance(entropy_values, (list, np.ndarray)):
                segment_entropy_sum += np.sum(entropy_values)
                segment_entropy_count += len(entropy_values)
            else:
                segment_entropy_sum += entropy_values
                segment_entropy_count += 1
    
            start += pred_len
    
        # ====== Average entropy for this segment ======
        if segment_entropy_count > 0:
            avg_entropy_seg = segment_entropy_sum / segment_entropy_count
            all_entropy_values.append(avg_entropy_seg)
    
        # ====== Align predictions with true values ======
        pred_seq = np.array(predictions[:len(seg) - input_len])
        true_seq = seg[input_len:input_len + len(pred_seq)]
    
        all_predictions.append(pred_seq)
        all_true.append(true_seq)
    
    # ====== Compute overall average entropy ======
    if all_entropy_values:
        overall_avg_entropy = np.mean(all_entropy_values)
    
    save_overall_avg_entropy = overall_avg_entropy
    
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

    # ====== VALIDATION PREDICTIONS (NO TIME) ======
    val_predictions = []
    val_true = []
    val_entropy_values = []
    
    for seg_idx, seg in enumerate(val_segments):
        if len(seg) < input_len + pred_len:
            continue
    
        predictions = []
        segment_entropy_sum = 0.0
        segment_entropy_count = 0
    
        start = input_len
        while start + pred_len <= len(seg):
            # ====== Input sequence ======
            y_in = seg[start - input_len:start].reshape(-1, 1)

            # --- PCA transform the input window ---
            y_in_pca = pca_train.transform(y_in.T).T   # shape (3,1)


            # --- predict in raw scale ---
            y_pred_raw, entropy_values = predict_dsrn(
                y_in_pca, S, R, params, N, steps
            )
            
            # no inverse transform needed
            y_pred = y_pred_raw.flatten()

            predictions.extend(y_pred)
    
            # ====== Accumulate entropy ======
            if isinstance(entropy_values, dict):
                avg_entropy = np.mean(list(entropy_values.values()))
                segment_entropy_sum += avg_entropy
                segment_entropy_count += 1
            elif isinstance(entropy_values, (list, np.ndarray)):
                segment_entropy_sum += np.sum(entropy_values)
                segment_entropy_count += len(entropy_values)
            else:
                segment_entropy_sum += entropy_values
                segment_entropy_count += 1
    
            start += pred_len
    
        # ====== Average entropy for this segment ======
        if segment_entropy_count > 0:
            avg_entropy_seg = segment_entropy_sum / segment_entropy_count
            val_entropy_values.append(avg_entropy_seg)
    
        # ====== Align predictions with true values ======
        pred_seq = np.array(predictions[:len(seg) - input_len])
        true_seq = seg[input_len:input_len + len(pred_seq)]
    
        val_predictions.append(pred_seq)
        val_true.append(true_seq)
    
    # ====== Compute overall validation entropy ======
    if val_entropy_values:
        overall_val_entropy = np.mean(val_entropy_values)
        print(f"Validation Average Topological Entropy: {overall_val_entropy:.6f}")
    else:
        print("No entropy values computed for validation set.")
    
    # ====== Compute metrics ======
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
    
        # threshold = minimum log probability of validation normals
        threshold_dict[w] = np.percentile(log_probs_normal_w, 5)
    
    print(" Thresholds computed for all window lengths (using minimum value).")

    # ====== TEST PREDICTIONS (NO TIME) ======
    test_predictions = []
    test_true = []
    test_entropy_all = []  # store (symbol, entropy) per segment
    
    TEST_BEATS = NORMAL_BEATS + ARRHYTHMIC_BEATS
    
    for seg_idx, seg in enumerate(test_segments):
        if len(seg) < input_len + pred_len:
            continue
    
        predictions = []
        segment_entropy_sum = 0.0
        segment_entropy_count = 0
    
        start = input_len
        while start + pred_len <= len(seg):
            # ====== Input sequence ======
            y_in = seg[start - input_len:start].reshape(-1, 1)

            # --- PCA transform the input window ---
            y_in_pca = pca_train.transform(y_in.T).T   # shape (3,1)


            # --- predict in raw scale ---
            y_pred_raw, entropy_values = predict_dsrn(
                y_in_pca, S, R, params, N, steps
            )
            
            # no inverse transform needed
            y_pred = y_pred_raw.flatten()

            predictions.extend(y_pred)
    
            # Accumulate entropy
            if isinstance(entropy_values, dict):
                avg_entropy = np.mean(list(entropy_values.values()))
                segment_entropy_sum += avg_entropy
                segment_entropy_count += 1
            elif isinstance(entropy_values, (list, np.ndarray)):
                segment_entropy_sum += np.sum(entropy_values)
                segment_entropy_count += len(entropy_values)
            else:
                segment_entropy_sum += entropy_values
                segment_entropy_count += 1
    
            start += pred_len
    
        # Average entropy for this segment
        if segment_entropy_count > 0:
            avg_entropy_seg = segment_entropy_sum / segment_entropy_count
            # Use a placeholder symbol for test segments if you want to differentiate
            sym = 'N' if seg_idx < len(normal_segments[num_train_beats + num_val_beats:num_train_beats + num_val_beats + num_test_norm]) else 'A'
            test_entropy_all.append((sym, avg_entropy_seg))
    
        # Align predictions with true
        pred_seq = np.array(predictions[:len(seg) - input_len])
        true_seq = seg[input_len:input_len + len(pred_seq)]
    
        test_predictions.append(pred_seq)
        test_true.append(true_seq)
    
    # ====== Compute overall entropy ======
    if test_entropy_all:
        overall_test_entropy = np.mean([e for _, e in test_entropy_all])
    
        normal_entropy = np.mean([e for sym, e in test_entropy_all if sym == 'N'])
        arr_entropy = np.mean([e for sym, e in test_entropy_all if sym == 'A'])
    else:
        print("No entropy values computed for test set.")
    
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
    for t in range(input_len, beat_length):
    
        # for this timepoint, evaluate all segments
        for seg_idx, seg in enumerate(test_segments):
    
            # ========= 1) GET INPUT WINDOW =========
            y_in = seg[t - input_len : t].reshape(-1, 1)

            # --- PCA transform the input window ---
            y_in_pca = pca_train.transform(y_in.T).T   # shape (3,1)
    
            # ========= 2) PREDICT NEXT POINT =========
            # --- predict in raw scale ---
            y_pred_raw, entropy_values = predict_dsrn(
                y_in_pca, S, R, params, N, steps
            )
            # no inverse transform needed
            y_pred = y_pred_raw.flatten()[0]
            pred_val = y_pred
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
        "Patient": record,
        "Data_seed": data_seed,
        "Reservoir_seed": reservoir_seed,
        "input_len": input_len,
        "Pred_len": pred_len,
        "N": N,

        # ---- TRAINING ----
        "Train_MSE": avg_mse,
        "Train_r": avg_r,
        "Train_RMSE": avg_rmse,
        "Avg_Topological_Entropy": save_overall_avg_entropy,
    
        # ---- VALIDATION ----
        "Val_MSE": avg_mse_val if 'avg_mse_val' in locals() else np.nan,
        "Val_r": avg_r_val if 'avg_r_val' in locals() else np.nan,
        "Val_RMSE": avg_rmse_val if 'avg_rmse_val' in locals() else np.nan,
        "Val_Entropy": overall_val_entropy if 'overall_val_entropy' in locals() else np.nan,
    
        # ---- TEST ----
        "Test_MSE": avg_mse_test if 'avg_mse_test' in locals() else np.nan,
        "Test_r": avg_r_test if 'avg_r_test' in locals() else np.nan,
        "Test_RMSE": avg_rmse_test if 'avg_rmse_test' in locals() else np.nan,
        "Test_Entropy": overall_test_entropy if 'overall_test_entropy' in locals() else np.nan,
        
        # ---- MODEL PARAMETERS ----
        "Param_Means": json.dumps(param_means.tolist()) if 'param_means' in locals() else "[]",
        "Param_Stds":  json.dumps(param_stds.tolist()) if 'param_stds' in locals() else "[]",
    
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

    # ====== CHECK IF N EXISTS ======
    key_cols = ["Patient", "Data_seed", "Reservoir_seed", "input_len", "Pred_len", "N"]
    
    exists = (
        (df_existing["Patient"] == record) &
        (df_existing["Data_seed"] == data_seed) &
        (df_existing["Reservoir_seed"] == reservoir_seed) &
        (df_existing["input_len"] == input_len) &
        (df_existing["Pred_len"] == pred_len) &
        (df_existing["N"] == N)
    ).any()

    # ====== CHECK IF THIS CONFIGURATION EXISTS ======
    if exists:
        print(
            f"Entry already exists for "
            f"Patient={record}, Data_seed={data_seed}, Reservoir_seed={reservoir_seed}, input_len={input_len}, Pred_len={pred_len}, N={N}. "
            "Skipping save."
        )
    else:
        df_result = pd.DataFrame([results_dict])
        df_combined = pd.concat([df_existing, df_result], ignore_index=True)
        df_combined.to_csv(results_csv, index=False)
        print(
            f"Results saved for "
            f"Patient={record}, Data_seed={data_seed}, Reservoir_seed={reservoir_seed}, input_len={input_len}, Pred_len={pred_len}, N={N}"
        )
    
    # ====== LOAD CSV ======
    if os.path.exists(results_csv):
        df_loaded = pd.read_csv(results_csv)
        print(f"\n Loaded CSV: {results_csv}")
        
        # Format all floats to 6 decimal places
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(df_loaded)
    else:
        print(f" CSV does not exist: {results_csv}")
      


    # In[ ]:





