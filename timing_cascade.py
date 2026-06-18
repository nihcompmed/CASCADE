#!/usr/bin/env python
# coding: utf-8

# =======================================================
# TIMING VERSION OF MAIN SCRIPT
# Changes from main script:
#   1. compute_phi_fast: endpoint-only (t_eval=[20.0]) + NO entropy
#   2. time.perf_counter() wrappers around training and real-time loop
#   3. saves timing + all existing metrics to results_timing/
#   4. hardware info saved for Biowulf node identification
# Everything else is IDENTICAL to main script.
# =======================================================

import sys
import numpy as np
import os
import time
import platform
import socket
import tracemalloc
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import norm
import wfdb
import pandas as pd
import json
from sklearn.decomposition import PCA
from numpy.random import default_rng

# =======================================================
# ARGS — same as main script
# =======================================================

record    = sys.argv[1]
data_seed = int(sys.argv[2])
res_seed  = int(sys.argv[3])
input_len = int(sys.argv[4])
N         = int(sys.argv[5])

print("Args: record={} data_seed={} res_seed={} input_len={} N={}".format(
    record, data_seed, res_seed, input_len, N))

# =======================================================
# CONFIG — identical to main script
# =======================================================

fs              = 360
data_folder     = 'mitdb_data_full'
annotation_folder = 'mit-bih-arrhythmia-database-1.0.0'

NORMAL_BEATS     = ['.', 'N', 'L', 'R', 'e', 'j']
ARRHYTHMIC_BEATS = ['A', 'a', 'J', 'S', 'V', 'E', 'F']

steps           = 1
beat_length     = 180
num_train_beats = 500
num_val_beats   = 100
num_test_norm   = 100
num_test_arr    = 100

prediction_lengths = [1, 10]

# CHANGED: save to separate timing dir
save_dir = "results_timing"
os.makedirs(save_dir, exist_ok=True)

results_csv = os.path.join(
    save_dir,
    "timing_{}_data_seed_{}_res_seed_{}_input_len_{}_reservoirs_{}.csv".format(
        record, data_seed, res_seed, input_len, N)
)

if os.path.exists(results_csv):
    print("Already exists, skipping: {}".format(results_csv))
    sys.exit(0)

# =======================================================
# HARDWARE INFO (Biowulf node capture)
# =======================================================

def get_hardware_info():
    info = {}
    info["hostname"]       = socket.gethostname()
    info["platform"]       = platform.platform()
    info["python_version"] = platform.python_version()
    info["numpy_version"]  = np.__version__
    info["cpu_logical"]    = os.cpu_count()
    info["cpu_physical"]   = None
    info["cpu_freq_mhz"]   = None
    info["ram_total_gb"]   = None
    try:
        import psutil
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["cpu_logical"]  = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        info["cpu_freq_mhz"] = freq.current if freq else None
        info["ram_total_gb"] = psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    info["cpu_model"] = line.split(':')[1].strip()
                    break
        if "cpu_model" not in info:
            info["cpu_model"] = platform.processor()
    except Exception:
        info["cpu_model"] = platform.processor()
    return info

hw = get_hardware_info()
print("Biowulf: {}  CPU: {}".format(hw["hostname"], hw.get("cpu_model","")))

# =======================================================
# HELPERS — identical to main script
# =======================================================

def get_segments(signal, ann_samples, ann_symbols, beat_length, target_symbols):
    segments, indices = [], []
    for s, sym in zip(ann_samples, ann_symbols):
        if sym in target_symbols and s >= beat_length:
            seg = signal[s - beat_length:s]
            segments.append(seg)
            indices.append(s)
    return segments, indices

def sliding_data(segment, input_len, pred_len):
    X_y, Y = [], []
    for t_idx in range(len(segment) - input_len - pred_len + 1):
        X_y.append(segment[t_idx : t_idx + input_len])
        Y.append(segment[t_idx + input_len : t_idx + input_len + pred_len])
    return np.array(X_y).T, np.array(Y).T

def rossler_system(t, X, params, N):
    X    = X.reshape(N, 3)
    dxdt = (1/params[:,3])*(-X[:,1]-X[:,2])
    dydt = (1/params[:,3])*(X[:,0]+params[:,0]*X[:,1])
    dzdt = (1/params[:,3])*(params[:,1]+X[:,2]*(X[:,0]-params[:,2]))
    return np.stack([dxdt, dydt, dzdt], axis=1).flatten()

# CHANGED 1: endpoint-only, no entropy — everything else same as compute_phi
def compute_phi_fast(X_norm, N, steps, R=None, params=None):
    M, D        = X_norm.shape
    params_used = params[:N]
    if R is None:
        R = np.random.uniform(-1, 1, size=(3*N, M))
    phi = np.zeros((3*N, D))
    t_eval = np.linspace(0, 20, 10)   # only 10 points instead of 100
    for i in range(D):
        u   = R @ X_norm[:, i]
        sol = solve_ivp(rossler_system,
                        [0, 20.0], u,
                        t_eval=t_eval,
                        args=(params_used, N),
                        rtol=1e-6, atol=1e-9)
        phi[:, i] = sol.y[:, -1]      # take only last point
    return phi, R, params_used

def train_dsrn_fast(X_norm, Y_norm, N, steps, params_fixed):
    phi, R, params_used = compute_phi_fast(X_norm, N, steps, params=params_fixed)
    S, _, _, _ = np.linalg.lstsq(phi.T, Y_norm.T, rcond=None)
    return S.T, R, params_used

def predict_dsrn_fast(X_norm, S, R, params_fixed, N, steps):
    phi, _, _ = compute_phi_fast(X_norm, N, steps, R=R, params=params_fixed)
    return S @ phi   # no entropy returned

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32,   np.int64)):   return int(obj)
    if isinstance(obj, dict):  return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [convert_numpy(v) for v in obj]
    return obj

# =======================================================
# LOAD DATA — identical to main script
# =======================================================

record_path = os.path.join(annotation_folder, record)
ann         = wfdb.rdann(record_path, extension='atr')
signal_path = os.path.join(data_folder, '{}.npy'.format(record))
signal      = np.load(signal_path)[:, 0]

reservoir_seed = res_seed
rng_data = default_rng(data_seed)
rng_res  = default_rng(reservoir_seed)

normal_segments, _  = get_segments(signal, ann.sample, ann.symbol, beat_length, NORMAL_BEATS)
arr_segments,    _  = get_segments(signal, ann.sample, ann.symbol, beat_length, ARRHYTHMIC_BEATS)

num_total_norm  = len(normal_segments)
all_norm_indices = np.arange(num_total_norm)
rng_data.shuffle(all_norm_indices)

val_indices       = all_norm_indices[:num_val_beats]
train_indices     = all_norm_indices[num_val_beats:num_val_beats+num_train_beats]
test_norm_indices = all_norm_indices[num_val_beats+num_train_beats:
                                     num_val_beats+num_train_beats+num_test_norm]

train_segments = np.array([normal_segments[i] for i in train_indices])
val_segments   = np.array([normal_segments[i] for i in val_indices])
test_segments  = np.array([normal_segments[i] for i in test_norm_indices] +
                            arr_segments[:num_test_arr])

max_N     = 200
tau_range = (0.8, 2.0)
stable_range  = [(0.02,0.07),(0.02,0.07),(6.0,7.0)]
chaotic_range = [(0.18,0.22),(0.18,0.22),(5.5,5.9)]
mix_ratio     = 1

hybrid_range = []
for (s_low, s_high), (c_low, c_high) in zip(stable_range, chaotic_range):
    hybrid_range.append((s_low+mix_ratio*(c_low-s_low),
                         s_high+mix_ratio*(c_high-s_high)))

params_fixed = np.array([
    [rng_res.uniform(*r) for r in hybrid_range] + [rng_res.uniform(*tau_range)]
    for _ in range(max_N)
])

# =======================================================
# MAIN LOOP — identical structure to main script
# =======================================================

all_results = []

for pred_len in prediction_lengths:

    # Build training windows — identical to main script
    all_X_y_raw, all_Y_raw = [], []
    for seg in train_segments:
        if len(seg) < input_len + pred_len:
            continue
        X_y_raw, Y_raw = sliding_data(seg, input_len, pred_len)
        if X_y_raw.size == 0 or Y_raw.size == 0:
            continue
        all_X_y_raw.append(X_y_raw)
        all_Y_raw.append(Y_raw)

    X_y_all_raw = np.hstack(all_X_y_raw)
    Y_all_raw   = np.hstack(all_Y_raw)
    n_train_windows = X_y_all_raw.shape[1]

    pca_train   = PCA(n_components=3)
    X_y_all_pca = pca_train.fit_transform(X_y_all_raw.T).T
    Y_all_norm  = Y_all_raw

    # =======================================================
    # CHANGED 2: time the training
    # =======================================================
    t_train_start = time.perf_counter()
    S, R, params = train_dsrn_fast(X_y_all_pca, Y_all_norm, N, steps, params_fixed)
    train_time_s  = time.perf_counter() - t_train_start

    # Validation — identical to main script
    val_predictions = []
    for seg in val_segments:
        if len(seg) < input_len + pred_len:
            continue
        predictions = []
        start = input_len
        while start + pred_len <= len(seg):
            y_in     = seg[start-input_len:start].reshape(-1,1)
            y_in_pca = pca_train.transform(y_in.T).T
            y_pred   = predict_dsrn_fast(y_in_pca, S, R, params, N, steps)
            predictions.extend(y_pred.flatten())
            start += pred_len
        pred_seq = np.array(predictions[:len(seg)-input_len])
        val_predictions.append(pred_seq)

    L       = beat_length - input_len
    all_val_preds = np.array([p[:L] for p in val_predictions])
    mu_i    = np.mean(all_val_preds, axis=0)
    sigma_i = np.std(all_val_preds,  axis=0)
    dx      = 1 / 360

    threshold_dict = {}
    for w in range(1, beat_length-input_len+1):
        log_probs = []
        for pred_seq in val_predictions:
            pp = pred_seq[:w] if len(pred_seq)>=w \
                 else np.pad(pred_seq,(0,w-len(pred_seq)),'edge')
            log_probs.append(np.sum(
                np.log(norm.pdf(pp, mu_i[:w], sigma_i[:w]+1e-8)+1e-10)
                + np.log(dx)))
        threshold_dict[w] = np.percentile(log_probs, 5)

    # Test labels — identical to main script
    test_labels = [0]*num_test_norm + [1]*num_test_arr
    num_segs    = len(test_segments)

    # =======================================================
    # CHANGED 3: time per-sample inference inside real-time loop
    # =======================================================
    cum_log_p    = np.zeros(num_segs)
    current_flags = np.zeros(num_segs, dtype=int)
    online_results = []

    per_sample_times = []   # time for one predict_dsrn_fast call
    beat_times       = []   # time for one full time-step across all segments

    t_loop_start = time.perf_counter()

    for t in range(input_len, beat_length):

        t_step_start = time.perf_counter()

        for seg_idx, seg in enumerate(test_segments):

            y_in     = seg[t-input_len:t].reshape(-1,1)
            y_in_pca = pca_train.transform(y_in.T).T

            # time one predict call
            t_pred_start = time.perf_counter()
            y_pred_raw   = predict_dsrn_fast(y_in_pca, S, R, params, N, steps)
            per_sample_times.append((time.perf_counter()-t_pred_start)*1000)

            pred_val = y_pred_raw.flatten()[0]
            mu_t     = mu_i[t-input_len]
            sigma_t  = sigma_i[t-input_len]

            pdf_val = norm.pdf(pred_val, loc=mu_t, scale=sigma_t+1e-8)
            log_p   = np.log(pdf_val+1e-10) + np.log(dx)
            cum_log_p[seg_idx] += log_p

            threshold_val = threshold_dict[t-input_len+1]
            current_flags[seg_idx] = int(cum_log_p[seg_idx] < threshold_val)

        beat_times.append((time.perf_counter()-t_step_start)*1000)

        ann_true = np.array(test_labels)
        ann_pred = current_flags.copy()

        acc = accuracy_score(ann_true, ann_pred)
        pre = precision_score(ann_true, ann_pred, zero_division=0)
        rec = recall_score(ann_true, ann_pred, zero_division=0)
        f1  = f1_score(ann_true, ann_pred, zero_division=0)

        online_results.append({
            "time": t, "accuracy": acc,
            "precision": pre, "recall": rec, "f1": f1,
            "flags": current_flags.copy()
        })

    total_loop_time_ms = (time.perf_counter()-t_loop_start)*1000

    # AUC-F1 and peak F1
    f1_series = [r["f1"] for r in online_results]
    auc_f1    = float(np.trapezoid(f1_series) / len(f1_series))
    peak_f1   = float(np.max(f1_series))
    final_f1  = float(f1_series[-1])
    final_prec = float(online_results[-1]["precision"])
    final_rec  = float(online_results[-1]["recall"])
    final_acc  = float(online_results[-1]["accuracy"])

    S_kb      = S.nbytes / 1024
    R_kb      = R.nbytes / 1024
    params_kb = params_fixed[:N].nbytes / 1024
    pca_kb    = (pca_train.components_.nbytes + pca_train.mean_.nbytes) / 1024
    total_kb  = S_kb + R_kb + params_kb + pca_kb

    # Memory — single prediction call only
    tracemalloc.start()
    seg_mem   = test_segments[0]
    y_in_mem  = seg_mem[0:input_len].reshape(-1, 1)
    y_pca_mem = pca_train.transform(y_in_mem.T).T
    _         = predict_dsrn_fast(y_pca_mem, S, R, params, N, steps)
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kb = peak_kb / 1024

    # Collect all results
    all_results.append({
        # identifiers
        "Patient":      record,
        "Data_seed":    data_seed,
        "Reservoir_seed": reservoir_seed,
        "input_len":    input_len,
        "Pred_len":     pred_len,
        "N":            N,
        "n_train_windows": n_train_windows,

        # hardware
        "hostname":       hw["hostname"],
        "cpu_model":      hw.get("cpu_model", ""),
        "cpu_logical":    hw.get("cpu_logical"),
        "cpu_physical":   hw.get("cpu_physical"),
        "cpu_freq_mhz":   hw.get("cpu_freq_mhz"),
        "ram_total_gb":   hw.get("ram_total_gb"),
        "platform":       hw["platform"],
        "python_version": hw["python_version"],
        "numpy_version":  hw["numpy_version"],

        # CHANGED: timing results
        "train_time_s":          train_time_s,
        "latency_ep_mean_ms":    float(np.mean(per_sample_times)),
        "latency_ep_std_ms":     float(np.std(per_sample_times)),
        "realtime_req_ms":       1000/fs,
        "ep_feasible":           float(np.mean(per_sample_times)) < 1000/fs,
        "beat_step_mean_ms":     float(np.mean(beat_times)),
        "beat_step_std_ms":      float(np.std(beat_times)),
        "total_loop_time_ms":    total_loop_time_ms,
        "beat_duration_ms":      beat_length * 1000 / fs,
        "latency_ratio":         total_loop_time_ms / (num_segs * beat_length * 1000/fs),

        # detection metrics — same as main script
        "final_f1":        final_f1,
        "final_precision": final_prec,
        "final_recall":    final_rec,
        "final_accuracy":  final_acc,
        "auc_f1":          auc_f1,
        "peak_f1":         peak_f1,
        "Online_results":  json.dumps(convert_numpy(online_results)),

        # memory
        "S_kb":           S_kb,
        "R_kb":           R_kb,
        "params_kb":      params_kb,
        "pca_kb":         pca_kb,
        "total_model_kb": total_kb,
        "peak_ram_kb":    peak_kb,
    })

    print("pred_len={} train={:.2f}s  EP={:.3f}ms  F1={:.3f}  AUC-F1={:.3f}".format(
        pred_len, train_time_s,
        float(np.mean(per_sample_times)),
        final_f1, auc_f1))

# Save
pd.DataFrame(all_results).to_csv(results_csv, index=False)
print("Saved: {}".format(results_csv))