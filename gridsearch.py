"""
================================================================================
4D Grid Search: Separat theta_i och theta_c
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

SPLIT_DATE = pd.Timestamp("2030-01-01")

# ── 4D Gridparametrar ─────────────────────────────────────────────────────────
THETA_I_GRID = np.round(np.linspace(-5, 5, 20), 2)  # Regimgräns INDPRO
THETA_C_GRID = np.round(np.linspace(-5, 5, 20), 2)  # Regimgräns CPI
BUFFER_GRID  = np.round(np.linspace( 0.0, 1.0, 20), 2)   # Gemensam buffert
MIN_DUR_GRID = np.arange(1, 7, 1)


def load_data(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = [p for p in path.iterdir() if p.suffix.lower() in {".xlsx", ".xls"}]
        if len(files) == 0:
            raise FileNotFoundError(f"No Excel file found in input directory: {path}")
        if len(files) > 1:
            raise ValueError(f"Multiple Excel files found in input directory: {path}: {[p.name for p in files]}")
        path = files[0]

    df = pd.read_excel(path, sheet_name="MoM")
    df = df.rename(columns={
        "DatumA1:F1": "Datum",
        "CPI MoM (%)": "CPI_MoM",
        "INDPRO MoM (%)": "INDPRO_MoM",
        "NBER": "NBER",
    })
    df = df[["Datum", "INDPRO_MoM", "CPI_MoM", "NBER"]].copy()
    df["Datum"] = pd.to_datetime(df["Datum"])
    df = df.sort_values("Datum").reset_index(drop=True)
    df["INDPRO_SMA3"] = df["INDPRO_MoM"].rolling(window=3, min_periods=3).mean()
    df["CPI_SMA3"]    = df["CPI_MoM"].rolling(window=3, min_periods=3).mean()
    df = df.dropna(subset=["INDPRO_SMA3", "CPI_SMA3", "NBER"]).reset_index(drop=True)
    df["NBER"] = df["NBER"].astype(int)
    return df

def compute_direction(series_values: np.ndarray, sigma: float, theta: float, c_b: float) -> np.ndarray:
    n = len(series_values)
    direction = np.zeros(n, dtype=np.int8)
    thr = theta * sigma
    upper_flip = thr + c_b * sigma
    lower_flip = thr - c_b * sigma

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
        if current == 1 and v < lower_flip:
            current = -1
        elif current == -1 and v > upper_flip:
            current = 1
        direction[t] = current
    return direction

REGIME_EXPANSION   = 0    
REGIME_SLOWDOWN    = 1    
REGIME_CONTRACTION = 2    
REGIME_RECOVERY    = 3    

def classify_regime(indpro_dir: np.ndarray, cpi_dir: np.ndarray) -> np.ndarray:
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

def apply_signal_min_duration(signal: np.ndarray, tau: int) -> np.ndarray:
    if tau <= 1: return signal.copy()
    n = len(signal)
    accepted = np.empty(n, dtype=np.int8)
    current = int(signal[0])
    accepted[0] = current
    candidate = current
    run = 1
    for i in range(1, n):
        s = int(signal[i])
        if s == candidate: run += 1
        else:
            candidate = s
            run = 1
        if candidate != current and run >= tau:
            current = candidate
        accepted[i] = current
    return accepted

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total else np.nan
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def run_pipeline(indpro_sma3, cpi_sma3, sigma_indpro, sigma_cpi, theta_i, theta_c, c_b, tau) -> np.ndarray:
    indpro_dir = compute_direction(indpro_sma3, sigma_indpro, theta_i, c_b)
    cpi_dir    = compute_direction(cpi_sma3, sigma_cpi, theta_c, c_b)
    regime     = classify_regime(indpro_dir, cpi_dir)
    raw_signal = regime_to_raw_signal(regime)
    return apply_signal_min_duration(raw_signal, tau)

def evaluate_combo(df, sigma_indpro, sigma_cpi, theta_i, theta_c, c_b, tau) -> dict:
    signal = run_pipeline(df["INDPRO_SMA3"].to_numpy(), df["CPI_SMA3"].to_numpy(),
                          sigma_indpro, sigma_cpi, theta_i, theta_c, c_b, tau)
    m = compute_metrics(df["NBER"].to_numpy().astype(int), signal)
    m["theta_i"] = float(theta_i)
    m["theta_c"] = float(theta_c)
    m["c_b"]     = float(c_b)
    m["tau"]     = int(tau)
    return m

def run_grid_search(df_train, sigma_indpro, sigma_cpi) -> pd.DataFrame:
    n_total = len(THETA_I_GRID) * len(THETA_C_GRID) * len(BUFFER_GRID) * len(MIN_DUR_GRID)
    print(f"Kör 4D Grid: {n_total} kombinationer...")
    rows = []
    for tau in MIN_DUR_GRID:
        for theta_i in THETA_I_GRID:
            for theta_c in THETA_C_GRID:
                for c_b in BUFFER_GRID:
                    rows.append(evaluate_combo(df_train, sigma_indpro, sigma_cpi, theta_i, theta_c, c_b, int(tau)))
    df = pd.DataFrame(rows)
    return df[["theta_i", "theta_c", "c_b", "tau", "TP", "FP", "FN", "TN", "Accuracy", "Precision", "Recall", "F1"]]

def plot_heatmaps_theta_i_vs_theta_c(results, best_tau, best_cb, out_path):
    subset = results[(results["tau"] == best_tau) & (results["c_b"] == best_cb)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, metric, cmap in zip(axes.flat, ["Accuracy", "Precision", "Recall", "F1"], ["viridis", "viridis", "viridis", "magma"]):
        pivot = subset.pivot(index="theta_i", columns="theta_c", values=metric)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax, annot_kws={"size": 8})
        ax.set_title(f"{metric} (tau={best_tau}, c_b={best_cb})", fontsize=13)
        ax.set_xlabel("Regimgräns CPI (theta_c)")
        ax.set_ylabel("Regimgräns INDPRO (theta_i)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_signal_vs_nber(df, signal, title, out_path):
    fig, ax = plt.subplots(figsize=(16, 5))
    nber, dates = df["NBER"].to_numpy(), df["Datum"].to_numpy()
    in_rec, start = False, None
    for i in range(len(dates)):
        if nber[i] == 1 and not in_rec:
            start, in_rec = dates[i], True
        elif nber[i] == 0 and in_rec:
            ax.axvspan(start, dates[i], color="gray", alpha=0.3, zorder=0)
            in_rec = False
    if in_rec: ax.axvspan(start, dates[-1], color="gray", alpha=0.3, zorder=0)
    ax.plot(dates, signal, color="crimson", linewidth=1.3, label="Modellsignal")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    df = load_data(INPUT_FILE)
    df_train = df[df["Datum"] < SPLIT_DATE].reset_index(drop=True)
    df_val   = df[df["Datum"] >= SPLIT_DATE].reset_index(drop=True)
    
    sigma_indpro = df_train["INDPRO_SMA3"].std(ddof=1)
    sigma_cpi    = df_train["CPI_SMA3"].std(ddof=1)

    results_train = run_grid_search(df_train, sigma_indpro, sigma_cpi)
    best = results_train.loc[results_train["F1"].idxmax()]
    ti_opt, tc_opt, cb_opt, tau_opt = float(best["theta_i"]), float(best["theta_c"]), float(best["c_b"]), int(best["tau"])

    print(f"\nOPTIMALT: theta_i={ti_opt}, theta_c={tc_opt}, c_b={cb_opt}, tau={tau_opt} | F1={best['F1']:.4f}")

    val_signal = run_pipeline(df_val["INDPRO_SMA3"].to_numpy(), df_val["CPI_SMA3"].to_numpy(),
                              sigma_indpro, sigma_cpi, ti_opt, tc_opt, cb_opt, tau_opt)
    val_m = compute_metrics(df_val["NBER"].to_numpy().astype(int), val_signal)
    print(f"OOS F1 (Validering): {val_m['F1']:.4f}\n")

    plot_heatmaps_theta_i_vs_theta_c(results_train, tau_opt, cb_opt, OUTPUT_DIR / "uppdrag1_4d_heatmaps.png")
    full_signal = run_pipeline(df["INDPRO_SMA3"].to_numpy(), df["CPI_SMA3"].to_numpy(),
                               sigma_indpro, sigma_cpi, ti_opt, tc_opt, cb_opt, tau_opt)
    plot_signal_vs_nber(df, full_signal, f"Signal (Contraction=1) | ti={ti_opt}, tc={tc_opt}", OUTPUT_DIR / "uppdrag1_4d_timeline.png")

if __name__ == "__main__":
    main()