"""
================================================================================
Walk-Forward Validation med 4D Grid (theta_i, theta_c, c_b, tau)
================================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

INPUT_FILE = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Walk-Forward Inställningar ──────────────────────────────────────────────
TRAIN_WINDOW_YEARS = 20
STEP_MONTHS        = 12

# ── 4D Gridparametrar ─────────────────────────────────────────────────────────
THETA_I_GRID = np.round(np.linspace(-5, 5, 20), 2)
THETA_C_GRID = np.round(np.linspace(-5, 5, 20), 2)
BUFFER_GRID  = np.round(np.linspace( 0.0, 1.0, 20), 2)
MIN_DUR_GRID = np.arange(1, 7, 1)

REGIME_EXPANSION   = 0    
REGIME_SLOWDOWN    = 1    
REGIME_CONTRACTION = 2    
REGIME_RECOVERY    = 3    

def load_data(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = [p for p in path.iterdir() if p.suffix.lower() in {".xlsx", ".xls"}]
        if len(files) == 0:
            raise FileNotFoundError(f"No Excel file found in input directory: {path}")
        if len(files) > 1:
            raise ValueError(f"Multiple Excel files found in input directory: {path}: {[p.name for p in files]}")
        path = files[0]

    df = pd.read_excel(path, sheet_name="MoM")
    df = df.rename(columns={"DatumA1:F1": "Datum", "CPI MoM (%)": "CPI_MoM", "INDPRO MoM (%)": "INDPRO_MoM", "NBER": "NBER"})
    df = df[["Datum", "INDPRO_MoM", "CPI_MoM", "NBER"]].copy()
    df["Datum"] = pd.to_datetime(df["Datum"])
    df = df.sort_values("Datum").reset_index(drop=True)
    df["INDPRO_SMA3"] = df["INDPRO_MoM"].rolling(window=3, min_periods=3).mean()
    df["CPI_SMA3"]    = df["CPI_MoM"].rolling(window=3, min_periods=3).mean()
    df = df.dropna(subset=["INDPRO_SMA3", "CPI_SMA3", "NBER"]).reset_index(drop=True)
    df["NBER"] = df["NBER"].astype(int)
    return df

def compute_direction(series_values, sigma, theta, c_b):
    n = len(series_values)
    direction = np.zeros(n, dtype=np.int8)
    thr = theta * sigma
    upper_flip, lower_flip = thr + c_b * sigma, thr - c_b * sigma
    current = 1
    for i in range(n):
        if not np.isnan(series_values[i]):
            current = 1 if series_values[i] > thr else -1
            break
    for t in range(n):
        v = series_values[t]
        if np.isnan(v):
            direction[t] = current
            continue
        if current == 1 and v < lower_flip: current = -1
        elif current == -1 and v > upper_flip: current = 1
        direction[t] = current
    return direction

def classify_regime(indpro_dir, cpi_dir):
    regime = np.empty_like(indpro_dir, dtype=np.int8)
    regime[(indpro_dir > 0) & (cpi_dir > 0)] = REGIME_EXPANSION
    regime[(indpro_dir < 0) & (cpi_dir > 0)] = REGIME_SLOWDOWN
    regime[(indpro_dir < 0) & (cpi_dir < 0)] = REGIME_CONTRACTION
    regime[(indpro_dir > 0) & (cpi_dir < 0)] = REGIME_RECOVERY
    return regime

def regime_to_raw_signal(regime: np.ndarray) -> np.ndarray:
    """
    Recession styrs helt av att INDPRO faller.
    Larmar vid både Slowdown (INDPRO ner, CPI upp) och Contraction (INDPRO ner, CPI ner).
    """
    return np.isin(regime, [REGIME_CONTRACTION, REGIME_SLOWDOWN]).astype(np.int8)

def apply_signal_min_duration(signal, tau):
    if tau <= 1: return signal.copy()
    n = len(signal)
    accepted = np.empty(n, dtype=np.int8)
    current = int(signal[0])
    accepted[0] = current
    candidate, run = current, 1
    for i in range(1, n):
        s = int(signal[i])
        if s == candidate: run += 1
        else: candidate, run = s, 1
        if candidate != current and run >= tau: current = candidate
        accepted[i] = current
    return accepted

def run_pipeline(indpro_sma3, cpi_sma3, sigma_indpro, sigma_cpi, theta_i, theta_c, c_b, tau):
    indpro_dir = compute_direction(indpro_sma3, sigma_indpro, theta_i, c_b)
    cpi_dir    = compute_direction(cpi_sma3, sigma_cpi, theta_c, c_b)
    regime     = classify_regime(indpro_dir, cpi_dir)
    return apply_signal_min_duration(regime_to_raw_signal(regime), tau)

def compute_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tot = tp + tn + fp + fn
    return {
        "Accuracy": (tp + tn) / tot if tot else np.nan,
        "Precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "Recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    }

def compute_confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp

def evaluate_combo(df, sig_i, sig_c, ti, tc, cb, tau):
    sig = run_pipeline(df["INDPRO_SMA3"].to_numpy(), df["CPI_SMA3"].to_numpy(), sig_i, sig_c, ti, tc, cb, tau)
    m = compute_metrics(df["NBER"].to_numpy().astype(int), sig)
    m["theta_i"], m["theta_c"], m["c_b"], m["tau"] = ti, tc, cb, tau
    return m

def run_grid_search(df_train, sig_i, sig_c):
    rows = [evaluate_combo(df_train, sig_i, sig_c, ti, tc, cb, tau)
            for tau in MIN_DUR_GRID for ti in THETA_I_GRID for tc in THETA_C_GRID for cb in BUFFER_GRID]
    return pd.DataFrame(rows)

def run_walk_forward_cv(df, train_years, step_months):
    min_date, max_date = df["Datum"].min(), df["Datum"].max()
    train_start = min_date
    fold_idx = 0
    fold_results, pooled_preds = [], []

    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end  = train_end + pd.DateOffset(months=step_months)
        if train_end >= max_date: break
        
        df_train = df[(df["Datum"] >= train_start) & (df["Datum"] < train_end)]
        df_test  = df[(df["Datum"] >= train_end) & (df["Datum"] < test_end)]
        if df_test.empty: break

        sig_i = df_train["INDPRO_SMA3"].std(ddof=1)
        sig_c = df_train["CPI_SMA3"].std(ddof=1)

        res_train = run_grid_search(df_train, sig_i, sig_c)
        best = res_train.loc[res_train["F1"].idxmax()]
        ti_opt, tc_opt, cb_opt, tau_opt = float(best["theta_i"]), float(best["theta_c"]), float(best["c_b"]), int(best["tau"])

        test_signal = run_pipeline(df_test["INDPRO_SMA3"].to_numpy(), df_test["CPI_SMA3"].to_numpy(),
                                   sig_i, sig_c, ti_opt, tc_opt, cb_opt, tau_opt)
        
        test_m = compute_metrics(df_test["NBER"].to_numpy().astype(int), test_signal)
        
        fold_results.append({
            "Fold": fold_idx, "Train_Start": train_start, "Test_Start": train_end, "Test_End": test_end,
            "theta_i": ti_opt, "theta_c": tc_opt, "c_b": cb_opt, "tau": tau_opt,
            "val_F1": test_m["F1"], "val_Precision": test_m["Precision"], "val_Recall": test_m["Recall"]
        })
        
        df_test = df_test.copy()
        df_test["Signal"] = test_signal
        pooled_preds.append(df_test)

        print(f"Fold {fold_idx:02d}: {train_end.date()} -> {test_end.date() - pd.Timedelta(days=1)} | "
              f"ti={ti_opt:+.2f}, tc={tc_opt:+.2f}, cb={cb_opt:.2f}, tau={tau_opt} | Val F1: {test_m['F1']:.3f}")

        train_start += pd.DateOffset(months=step_months)
        fold_idx += 1

    return pd.DataFrame(fold_results), pd.concat(pooled_preds, ignore_index=True)

def plot_parameter_stability(df_folds, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    folds = df_folds["Fold"]
    ax.plot(folds, df_folds["theta_i"], label="Theta_i (INDPRO)", marker="o")
    ax.plot(folds, df_folds["theta_c"], label="Theta_c (CPI)", marker="s")
    ax.plot(folds, df_folds["c_b"], label="Buffert c_b", marker="^")
    ax.set_title("Parameterstabilitet över tid (Walk-Forward)")
    ax.set_xlabel("Fold index")
    ax.set_ylabel("Parametervärde (sigma)")
    ax.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    df = load_data(INPUT_FILE)
    df_folds, pooled = run_walk_forward_cv(df, TRAIN_WINDOW_YEARS, STEP_MONTHS)
    
    # Exportera OOS-datan till CSV för uppsatsen!
    csv_path_data = OUTPUT_DIR / "uppdrag1_4d_pooled_oos_data.csv"
    pooled.to_csv(csv_path_data, index=False)
    
    # Exportera parametrarna per fold
    csv_path_folds = OUTPUT_DIR / "uppdrag1_4d_walkforward_folds.csv"
    df_folds.to_csv(csv_path_folds, index=False)

    # Poolade metrics i terminalen
    y_true = pooled["NBER"].to_numpy().astype(int)
    y_pred = pooled["Signal"].to_numpy().astype(int)
    tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred)
    
    print("\n" + "=" * 70)
    print("POOLADE OOS-METRICS (Endast Contraction = Recession)")
    print("=" * 70)
    print(f"  True Positives  : {tp}")
    print(f"  False Positives : {fp}")
    print(f"  True Negatives  : {tn}")
    print(f"  False Negatives : {fn}")
    print(f"  F1-Score        : {metrics['F1']:.4f}")
    print(f"  Recall          : {metrics['Recall']:.4f}")
    print(f"  Precision       : {metrics['Precision']:.4f}")
    print("=" * 70)
    print(f"Sparade OOS-data till: {csv_path_data}")

    plot_parameter_stability(df_folds, OUTPUT_DIR / "uppdrag1_4d_parameter_stability.png")

if __name__ == "__main__":
    main()