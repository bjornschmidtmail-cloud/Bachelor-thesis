"""
Integration av första ordningens Markov-modell med S&P 500-avkastning.

Datakällor (läses från lokala filer):
  - sp500_inflationadj.csv/xlsx  : Real S&P 500 prisnivå (månadsdata)
  - Markovseries_clean.xlsx      : Konjunkturregimer (header på rad 4)
  - TB3MS.csv                    : 3-month T-bill, discount basis (FRED)
  - CPIAUCSL.csv                 : CPI All Urban Consumers SA (FRED)

Real T-bill beräknas via Fisher:
  r_real = (1 + r_nom_monthly) / (1 + pi_monthly) - 1

Strategier som utvärderas:
  1. Buy and Hold                  - benchmark
  2. Naive State Persistence       - 100% S&P om förra månadens regim var
                                     Expansion/Recovery, annars T-bill.
                                     ANVÄNDER INTE Markov-modellen - bara
                                     regim-klassificeringen. Ingår som
                                     naiv baseline.
  3. Markov Expected Return        - binär: in/ut baserat på
                                     sum_j P[s_{t-1}, j] * r_bar[j] > 0
  4. Markov Weighted               - mjuk vikt proportionell mot
                                     P-viktat E[r_excess]
  5. Markov Multi-step (h=3,6,12)  - använder P^h istället för P^1

Alla Markov-strategier använder in-sample by-regime means och
State_{t-1} som input (shift(1) för att undvika look-ahead bias).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import os
import sys

STATE_ORDER = ["Expansion", "Slowdown", "Contraction", "Recovery"]

# README-aligned standardfilnamn
README_MARKOV_FILE = "Markovseries_clean.xlsx"
README_SP500_FILE  = "sp500_inflationadj.csv"
README_CPI_FILE    = "CPIAUCSL.csv"
README_TBILL_FILE  = "TB3MS.csv"


# ===========================================================================
# ROBUST FILSÖKNING
# ===========================================================================

def _find_file(filename: str, search_dirs: list[str]) -> str:
    """
    Letar efter filnamnet i en lista av kataloger och returnerar första träffen.
    Försöker även vanliga filnamns-varianter (t.ex. CPIAUCSL__1_.csv).
    """
    candidates = [filename]
    base, ext = os.path.splitext(filename)
    candidates += [f"{base}__1_{ext}", f"{base} (1){ext}", f"{base}_1{ext}"]

    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for name in candidates:
            full = os.path.join(d, name)
            if os.path.isfile(full):
                return full

    searched = "\n  ".join(
        f"{d}/{n}" for d in search_dirs if d for n in candidates
    )
    raise FileNotFoundError(
        f"Hittade inte '{filename}'.\nLetade på:\n  {searched}"
    )


def _resolve_sp500(search_dirs: list[str]) -> str:
    """S&P-datat kan vara .csv eller .xlsx - testar båda."""
    for ext in (".csv", ".xlsx"):
        try:
            return _find_file(f"sp500_inflationadj{ext}", search_dirs)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "Hittade inte sp500_inflationadj.csv eller .xlsx i:\n  "
        + "\n  ".join(d for d in search_dirs if d)
    )


# ===========================================================================
# 1) DATAINLÄSNING
# ===========================================================================

def load_readme_input_files(
    markov_file: str = README_MARKOV_FILE,
    sp500_file: str = README_SP500_FILE,
    cpi_file: str = README_CPI_FILE,
    tb3ms_file: str = README_TBILL_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Laddar rådata exakt enligt README-filnamnen och rätt Pandas-funktion.

    - Markovseries_clean.xlsx -> pd.read_excel(..., skiprows=3)
    - sp500_inflationadj.csv  -> pd.read_csv(...)
      (ändra till pd.read_excel(...) om lokalt använder .xlsx)
    - CPIAUCSL.csv            -> pd.read_csv(...)
    - TB3MS.csv               -> pd.read_csv(...)
    """
    # 1. Ladda regimdata (Excel)
    markov_df = pd.read_excel(markov_file, skiprows=3, engine="openpyxl")

    # 2. Ladda S&P 500 (CSV eller Excel om användaren uttryckligen anger .xlsx)
    if str(sp500_file).lower().endswith(".xlsx"):
        sp500_df = pd.read_excel(sp500_file, engine="openpyxl")
    else:
        sp500_df = pd.read_csv(sp500_file)

    # 3. Ladda CPI (Inflation)
    cpi_df = pd.read_csv(cpi_file)

    # 4. Ladda 3-månaders T-Bill (Riskfri ränta)
    tb3ms_df = pd.read_csv(tb3ms_file)

    return markov_df, sp500_df, cpi_df, tb3ms_df


def load_sp500_real(path_or_df: str | pd.DataFrame) -> pd.DataFrame:
    """Läser real S&P 500-prisnivå från CSV/XLSX eller från redan inläst DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        path = path_or_df
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xlsx":
            df = pd.read_excel(path, engine="openpyxl")
        else:
            df = pd.read_csv(path)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    date_col  = next(c for c in df.columns if "date" in c.lower())
    price_col = next(c for c in df.columns if any(
        k in c.lower() for k in ("value", "price", "close")))
    df = df[[date_col, price_col]].rename(
        columns={date_col: "Date", price_col: "Price"})
    df["Date"]  = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    df["Return"] = np.log(df["Price"] / df["Price"].shift(1))
    return df


def load_markov_states(path_or_df: str | pd.DataFrame) -> pd.DataFrame:
    """Läser Markovseries_clean.xlsx, rubriker på rad 4 (skiprows=3/header=3), eller från redan inläst DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_excel(path_or_df, engine="openpyxl", header=3)
    df.columns = [str(c).strip() for c in df.columns]
    date_col  = next(c for c in df.columns if "date" in c.lower())
    state_col = next(c for c in df.columns if "state" in c.lower())
    df = df[[date_col, state_col]].rename(
        columns={date_col: "Date", state_col: "State"})
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce") + pd.offsets.MonthEnd(0)
    df["State"] = df["State"].astype(str).str.strip()
    df = df.dropna(subset=["State"])
    invalid = set(df["State"]) - set(STATE_ORDER)
    if invalid:
        raise ValueError(f"Okända tillstånd: {invalid}")
    return df.reset_index(drop=True)


def load_real_tbill(tbill_csv: str, cpi_csv: str) -> pd.Series:
    """
    Beräknar månadsvis real riskfri ränta via Fisher:
      r_real = (1 + r_nom_monthly) / (1 + pi_monthly) - 1
    """
    if isinstance(tbill_csv, pd.DataFrame):
        tb_df = tbill_csv.copy()
        tb_df["observation_date"] = pd.to_datetime(tb_df["observation_date"])
    else:
        tb_df = pd.read_csv(tbill_csv, parse_dates=["observation_date"])
    tb = tb_df.set_index("observation_date")["TB3MS"] / 100.0
    tb.index = tb.index + pd.offsets.MonthEnd(0)
    r_nom_monthly = (1.0 + tb) ** (1.0 / 12.0) - 1.0

    if isinstance(cpi_csv, pd.DataFrame):
        cpi = cpi_csv.copy()
        cpi["observation_date"] = pd.to_datetime(cpi["observation_date"])
    else:
        cpi = pd.read_csv(cpi_csv, parse_dates=["observation_date"])
    cpi = cpi.set_index("observation_date")["CPIAUCSL"]
    cpi.index = cpi.index + pd.offsets.MonthEnd(0)
    inflation = cpi.pct_change()

    joined = pd.concat(
        [r_nom_monthly.rename("r_nom"), inflation.rename("pi")], axis=1
    ).dropna()
    rf_real = (1.0 + joined["r_nom"]) / (1.0 + joined["pi"]) - 1.0
    rf_real.name = "r_f_real"
    return rf_real


def merge_data(
    sp500_df:  pd.DataFrame,
    states_df: pd.DataFrame,
    rf_real:   pd.Series,
) -> pd.DataFrame:
    """Slår samman på gemensamt datum (månadsslut)."""
    sp = sp500_df.set_index("Date")
    st = states_df.set_index("Date")
    rf = rf_real.copy()
    rf.index = pd.DatetimeIndex(rf.index)

    df = sp.join(st[["State"]], how="inner")
    df = df.join(rf.rename("r_f_real"), how="inner")
    df = df.dropna(subset=["Return", "State", "r_f_real"]).copy()
    df["Excess"] = df["Return"] - df["r_f_real"]
    df = df.reset_index().rename(columns={"index": "Date"})
    return df


# ===========================================================================
# 2) BY-REGIME STATISTIK
# ===========================================================================

def by_regime_statistics(
    df: pd.DataFrame,
    states: list = STATE_ORDER,
) -> pd.DataFrame:
    """Nyckeltal per regim."""
    rows = []
    for s in states:
        sub = df[df["State"] == s]
        if sub.empty:
            rows.append({"State": s, "n": 0})
            continue

        r = sub["Return"]
        e = sub["Excess"]

        mean_m   = r.mean()
        vol_m    = r.std(ddof=1)
        std_e    = e.std(ddof=1)
        sharpe_m = e.mean() / std_e if std_e > 0 else np.nan

        equity      = (1.0 + r).cumprod()
        running_max = equity.cummax()
        max_dd      = (equity / running_max - 1.0).min()

        rows.append({
            "State":        s,
            "n":            int(len(sub)),
            "mean_m":       mean_m,
            "vol_m":        vol_m,
            "mean_ann":     mean_m * 12.0,
            "vol_ann":      vol_m * np.sqrt(12.0),
            "sharpe_ann":   sharpe_m * np.sqrt(12.0) if not np.isnan(sharpe_m) else np.nan,
            "max_drawdown": max_dd,
            "hit_rate":     (r > 0).mean(),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# 3) FORWARD-LOOKING REGIME-SANNOLIKHETER
# ===========================================================================

def forward_regime_probabilities(
    P: np.ndarray,
    current_state: str,
    horizon: int,
    states: list = STATE_ORDER,
) -> pd.DataFrame:
    """P(X_{t+h} = s | X_t = current_state) för h = 1..horizon."""
    idx = states.index(current_state)
    n   = len(states)
    e_t = np.zeros(n); e_t[idx] = 1.0

    rows = []
    P_h  = np.eye(n)
    for h in range(1, horizon + 1):
        P_h  = P_h @ P
        dist = e_t @ P_h
        rows.append({"h": h, **{s: dist[i] for i, s in enumerate(states)}})

    return pd.DataFrame(rows)


def forward_probabilities_all_starts(
    P: np.ndarray,
    horizon: int,
    states: list = STATE_ORDER,
) -> dict[str, pd.DataFrame]:
    return {s: forward_regime_probabilities(P, s, horizon, states) for s in states}


# ===========================================================================
# 4) P-VIKTAT VÄNTEVÄRDE
# ===========================================================================

def markov_expected_excess_by_from_state(
    df: pd.DataFrame,
    P: np.ndarray,
    states: list = STATE_ORDER,
    horizon: int = 1,
) -> dict[str, float]:
    """E[r_excess, t+h | s_{t-1}] = (e_s @ P^h) . mean_excess_vector"""
    mean_excess = {}
    for s in states:
        sub = df[df["State"] == s]
        mean_excess[s] = sub["Excess"].mean() if len(sub) > 0 else 0.0
    mu_e = np.array([mean_excess[s] for s in states])

    P_h = np.linalg.matrix_power(P, horizon)

    expected = {}
    for i, s in enumerate(states):
        dist = P_h[i, :]
        expected[s] = float(dist @ mu_e)
    return expected


# ===========================================================================
# 5) ALLOCATION-STRATEGIER
# ===========================================================================

def strategy_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """100% S&P 500 hela perioden."""
    return df.set_index("Date")["Return"].copy().rename("Buy and Hold")


def strategy_naive_state_persistence(
    df: pd.DataFrame,
    risk_on_states: tuple = ("Expansion", "Recovery"),
) -> pd.Series:
    """NAIV baseline: 100% S&P om State_{t-1} ∈ risk_on_states."""
    d = df.set_index("Date").copy()
    d["State_lag"] = d["State"].shift(1)
    risk_on = d["State_lag"].isin(risk_on_states)
    out = np.where(risk_on, d["Return"], d["r_f_real"])
    return pd.Series(out, index=d.index, name="Naive State Persistence")


def strategy_markov_expected_return(
    df: pd.DataFrame,
    P: np.ndarray,
    states: list = STATE_ORDER,
) -> pd.Series:
    """MARKOV binär: S&P om sum_j P[s_{t-1}, j] * r_bar[j] > 0."""
    expected_by_state = markov_expected_excess_by_from_state(df, P, states, horizon=1)
    signal = {s: (expected_by_state[s] > 0) for s in states}

    d = df.set_index("Date").copy()
    d["State_lag"] = d["State"].shift(1)
    d["risk_on"]   = d["State_lag"].map(signal).fillna(False)
    out = np.where(d["risk_on"], d["Return"], d["r_f_real"])
    return pd.Series(out, index=d.index, name="Markov Expected Return")


def strategy_markov_weighted(
    df: pd.DataFrame,
    P: np.ndarray,
    states: list = STATE_ORDER,
) -> pd.Series:
    """MARKOV mjuk vikt: w = clip(E[r_excess] / scale, 0, 1)."""
    expected_by_state = markov_expected_excess_by_from_state(df, P, states, horizon=1)
    vals = np.array([expected_by_state[s] for s in states])
    scale = np.max(np.abs(vals))
    weights = np.clip(vals / scale, 0.0, 1.0) if scale > 0 else np.zeros(len(states))
    weight_map = {s: float(weights[i]) for i, s in enumerate(states)}

    d = df.set_index("Date").copy()
    d["State_lag"] = d["State"].shift(1)
    d["weight"]    = d["State_lag"].map(weight_map).fillna(0.0)
    out = d["weight"] * d["Return"] + (1.0 - d["weight"]) * d["r_f_real"]
    return pd.Series(out.values, index=d.index, name="Markov Weighted")


def strategy_markov_multistep(
    df: pd.DataFrame,
    P: np.ndarray,
    horizon: int,
    states: list = STATE_ORDER,
) -> pd.Series:
    """MARKOV multi-step: använder P^h istället för P^1."""
    expected_by_state = markov_expected_excess_by_from_state(df, P, states, horizon=horizon)
    signal = {s: (expected_by_state[s] > 0) for s in states}

    d = df.set_index("Date").copy()
    d["State_lag"] = d["State"].shift(1)
    d["risk_on"]   = d["State_lag"].map(signal).fillna(False)
    out = np.where(d["risk_on"], d["Return"], d["r_f_real"])
    return pd.Series(out, index=d.index, name=f"Markov Multi-step (h={horizon})")


# ===========================================================================
# 6) STRATEGI-UTVÄRDERING
# ===========================================================================

def evaluate_strategy(
    returns: pd.Series,
    rf: pd.Series,
    name: str = "Strategy",
) -> dict:
    """Annualiserad mean, vol, Sharpe, max DD, total return, final equity."""
    r      = returns.dropna()
    excess = r - rf.reindex(r.index).fillna(0.0)

    mean_m = r.mean()
    vol_m  = r.std(ddof=1)
    std_e  = excess.std(ddof=1)
    sharpe = (excess.mean() / std_e * np.sqrt(12.0)) if std_e > 0 else np.nan

    equity      = (1.0 + r).cumprod()
    running_max = equity.cummax()
    max_dd      = (equity / running_max - 1.0).min()

    return {
        "Strategy":     name,
        "n_months":     int(len(r)),
        "mean_ann":     mean_m * 12.0,
        "vol_ann":      vol_m * np.sqrt(12.0),
        "sharpe_ann":   sharpe,
        "max_drawdown": max_dd,
        "total_return": equity.iloc[-1] - 1.0,
        "final_equity": equity.iloc[-1],
        "equity_curve": equity,
    }


def compare_strategies(
    df: pd.DataFrame,
    P: np.ndarray,
    multistep_horizons: tuple = (3, 6, 12),
) -> tuple[pd.DataFrame, dict, dict]:
    """Kör alla strategier."""
    rf = df.set_index("Date")["r_f_real"]

    strategies = {
        "Buy and Hold":            strategy_buy_and_hold(df),
        "Naive State Persistence": strategy_naive_state_persistence(df),
        "Markov Expected Return":  strategy_markov_expected_return(df, P),
        "Markov Weighted":         strategy_markov_weighted(df, P),
    }
    for h in multistep_horizons:
        strategies[f"Markov Multi-step h={h}"] = strategy_markov_multistep(df, P, h)

    results       = []
    equity_curves = {}
    for name, ret in strategies.items():
        ev = evaluate_strategy(ret, rf, name=name)
        equity_curves[name] = ev.pop("equity_curve")
        results.append(ev)

    signal_details = {}
    for h in (1,) + multistep_horizons:
        signal_details[f"h={h}"] = markov_expected_excess_by_from_state(df, P, horizon=h)

    return pd.DataFrame(results), equity_curves, signal_details


# ===========================================================================
# 7) MAIN-RUNNER
# ===========================================================================

def run_sp500_integration(
    sp500_path:         str,
    states_xlsx:        str,
    tbill_csv:          str,
    cpi_csv:            str,
    P_first_order:      np.ndarray,
    horizon:            int = 12,
    multistep_horizons: tuple = (3, 6, 12),
) -> dict:

    print("=" * 70)
    print("DATAINLÄSNING")
    print("=" * 70)

    # Ladda först råfilerna exakt enligt README-filnamn och rätt Pandas-funktioner.
    markov_raw, sp500_raw, cpi_raw, tb3ms_raw = load_readme_input_files(
        markov_file=states_xlsx,
        sp500_file=sp500_path,
        cpi_file=cpi_csv,
        tb3ms_file=tbill_csv,
    )

    sp = load_sp500_real(sp500_raw)
    print(f"S&P 500 real:  {len(sp)} obs, "
          f"{sp['Date'].min().date()} – {sp['Date'].max().date()}")

    states = load_markov_states(markov_raw)
    print(f"Markov states: {len(states)} obs, "
          f"{states['Date'].min().date()} – {states['Date'].max().date()}")
    print(f"  Fördelning: {states['State'].value_counts().to_dict()}")

    rf_real = load_real_tbill(tb3ms_raw, cpi_raw)
    print(f"Real T-bill:   {len(rf_real)} obs, "
          f"{rf_real.index.min().date()} – {rf_real.index.max().date()}")
    print(f"  Mean annual: {rf_real.mean()*12:.4f}  "
          f"(min: {rf_real.min():.4f}, max: {rf_real.max():.4f})")

    df = merge_data(sp, states, rf_real)
    print(f"\nKombinerad:    {len(df)} månader, "
          f"{df['Date'].min().date()} – {df['Date'].max().date()}")

    print("\n" + "=" * 70)
    print("BY-REGIME STATISTIK  (real log-avkastning, real T-bill)")
    print("=" * 70)
    stats = by_regime_statistics(df)
    print(stats.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"FORWARD-LOOKING REGIME-SANNOLIKHETER  (horizon = {horizon} månader)")
    print("=" * 70)
    forwards = forward_probabilities_all_starts(P_first_order, horizon)
    for s, fdf in forwards.items():
        print(f"\nFrån {s}:")
        print(fdf.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("MARKOV-VIKTADE E[r_excess] GIVET FROM-STATE  (per horizon)")
    print("=" * 70)
    for h in (1,) + multistep_horizons:
        exp = markov_expected_excess_by_from_state(df, P_first_order, horizon=h)
        print(f"\nHorizon h = {h}:")
        for s, v in exp.items():
            sign = "+" if v >= 0 else "-"
            print(f"  From {s:12s}: E[r_excess] = {v:+.5f}  (signal: {sign})")

    print("\n" + "=" * 70)
    print("STRATEGI-JÄMFÖRELSE")
    print("=" * 70)
    summary, equity_curves, signal_details = compare_strategies(
        df, P_first_order, multistep_horizons=multistep_horizons
    )
    print(summary.round(4).to_string(index=False))

    return {
        "df":               df,
        "by_regime_stats":  stats,
        "forwards":         forwards,
        "strategy_summary": summary,
        "equity_curves":    equity_curves,
        "signal_details":   signal_details,
    }


# ===========================================================================
# 8) SPARA TILL EXCEL
# ===========================================================================

def save_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "sp500_integration_results.xlsx")

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        results["df"].to_excel(writer, sheet_name="MergedData", index=False)
        results["by_regime_stats"].round(6).to_excel(
            writer, sheet_name="ByRegimeStats", index=False)
        for s, fdf in results["forwards"].items():
            fdf.round(6).to_excel(writer, sheet_name=f"Forward_{s[:8]}", index=False)
        results["strategy_summary"].round(6).to_excel(
            writer, sheet_name="StrategySummary", index=False)
        pd.DataFrame(results["equity_curves"]).to_excel(
            writer, sheet_name="EquityCurves")

        sig_rows = []
        for h_label, exp in results["signal_details"].items():
            for s, v in exp.items():
                sig_rows.append({"horizon": h_label, "from_state": s,
                                 "E_r_excess": v,
                                 "signal": "risk-on" if v > 0 else "risk-off"})
        pd.DataFrame(sig_rows).to_excel(
            writer, sheet_name="MarkovSignals", index=False)

    print(f"Excel sparat: {out}")


# ===========================================================================
# 9) VISUALISERINGAR
# ===========================================================================

def make_visualizations(results: dict, out_dir: str):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib.patches import Patch
    os.makedirs(out_dir, exist_ok=True)

    stats         = results["by_regime_stats"]
    colors_state  = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]
    regime_colors = dict(zip(STATE_ORDER, colors_state))

    # 1. By-regime nyckeltal
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (col, title) in zip(axes, [
        ("mean_ann",     "Annualiserad medelavkastning"),
        ("vol_ann",      "Annualiserad volatilitet"),
        ("sharpe_ann",   "Annualiserad Sharpe"),
        ("max_drawdown", "Max Drawdown"),
    ]):
        vals = stats[col].values
        bars = ax.bar(stats["State"], vals,
                      color=colors_state, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.set_xticks(range(len(stats["State"])))
        ax.set_xticklabels(stats["State"], rotation=20, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.003 if v >= 0 else -0.012),
                    f"{v:.1%}", ha="center", fontsize=8)
    fig.suptitle("S&P 500 real avkastning per konjunkturregim (1947–2026)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "by_regime_stats.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2. Forward-looking sannolikheter
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (s, fdf) in zip(axes.flatten(), results["forwards"].items()):
        for k, col in enumerate(STATE_ORDER):
            ax.plot(fdf["h"], fdf[col], label=col,
                    color=colors_state[k], linewidth=2)
        ax.set_title(f"Start: {s}", fontsize=10)
        ax.set_xlabel("Månader framåt")
        ax.set_ylabel("Sannolikhet")
        ax.set_ylim(0, 1)
        ax.axhline(0.785, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Forward-looking regime-sannolikheter (P^h)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "forward_probabilities.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3. Equity-kurvor
    strat_colors = {
        "Buy and Hold":             "#607D8B",
        "Naive State Persistence":  "#9E9E9E",
        "Markov Expected Return":   "#2196F3",
        "Markov Weighted":          "#4CAF50",
        "Markov Multi-step h=3":    "#FF9800",
        "Markov Multi-step h=6":    "#F44336",
        "Markov Multi-step h=12":   "#9C27B0",
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, eq in results["equity_curves"].items():
        lw = 2.2 if name == "Buy and Hold" else 1.5
        ls = "-" if "Markov" in name or name == "Buy and Hold" else "--"
        ax.plot(eq.index, eq.values,
                label=name, color=strat_colors.get(name, "black"),
                linewidth=lw, linestyle=ls)
    ax.set_yscale("log")
    ax.set_title("Equity-kurvor 1947–2026 (log-skala, start = 1.0)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Datum"); ax.set_ylabel("Equity (log)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equity_curves.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 4. Regim-färgad S&P 500
    df = results["df"].set_index("Date")
    fig, ax = plt.subplots(figsize=(14, 5))
    prev_state = None
    seg_start  = df.index[0]
    for date, row in df.iterrows():
        if row["State"] != prev_state:
            if prev_state is not None:
                seg = df.loc[seg_start:date, "Price"]
                ax.fill_between(seg.index, 0, seg.values,
                                color=regime_colors[prev_state], alpha=0.25)
                ax.plot(seg.index, seg.values,
                        color=regime_colors[prev_state], linewidth=0.7)
            seg_start  = date
            prev_state = row["State"]
    seg = df.loc[seg_start:, "Price"]
    ax.fill_between(seg.index, 0, seg.values,
                    color=regime_colors[prev_state], alpha=0.25)
    ax.plot(seg.index, seg.values,
            color=regime_colors[prev_state], linewidth=0.7)
    legend_elements = [Patch(facecolor=c, alpha=0.6, label=s)
                       for s, c in regime_colors.items()]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title("Real S&P 500 färgad efter konjunkturregim",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Datum"); ax.set_ylabel("Prisnivå (real, log)")
    ax.set_yscale("log"); ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sp500_regime_colored.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 5. Strategi-jämförelse
    summary = results["strategy_summary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (col, title) in zip(axes, [
        ("mean_ann",     "Annualiserad medelavkastning"),
        ("sharpe_ann",   "Annualiserad Sharpe"),
        ("max_drawdown", "Max Drawdown"),
    ]):
        vals = summary[col].values
        names = summary["Strategy"].values
        bar_colors = [strat_colors.get(n, "gray") for n in names]
        bars = ax.bar(range(len(names)), vals, color=bar_colors,
                      alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.003 if v >= 0 else -0.012),
                    f"{v:.1%}", ha="center", fontsize=7)
    fig.suptitle("Strategi-jämförelse (1947–2026, real avkastning)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "strategy_comparison.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    for f in ["by_regime_stats.png", "forward_probabilities.png",
              "equity_curves.png", "sp500_regime_colored.png",
              "strategy_comparison.png"]:
        print(f"  Sparad: {f}")


# ===========================================================================
# 10) KÖRNING
# ===========================================================================

if __name__ == "__main__":
    # Filsökning: letar i skriptets mapp, current working directory,
    # ev. "uploads"-mapp, och /mnt/user-data/uploads. Fungerar oavsett var
    # skriptet körs ifrån.
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _cwd         = os.getcwd()
    _search_dirs = [_script_dir, _cwd,
                    os.path.join(_cwd, "uploads"),
                    "/mnt/user-data/uploads"]
    _search_dirs = list(dict.fromkeys(_search_dirs))

    try:
        first_order_path = _find_file("first_order_results.xlsx", _search_dirs)
        # README-namn: använd exakt dessa filnamn lokalt.
        try:
            sp500_path = _find_file(README_SP500_FILE, _search_dirs)
        except FileNotFoundError:
            # Tillåt .xlsx-lokal kopia om användaren sparat om filen.
            sp500_path = _find_file("sp500_inflationadj.xlsx", _search_dirs)
        states_path      = _find_file(README_MARKOV_FILE, _search_dirs)
        tbill_path       = _find_file(README_TBILL_FILE, _search_dirs)
        cpi_path         = _find_file(README_CPI_FILE, _search_dirs)
    except FileNotFoundError as err:
        print("FEL: " + str(err), file=sys.stderr)
        print("\nSökvägarna som användes:", file=sys.stderr)
        for d in _search_dirs:
            print(f"  - {d}", file=sys.stderr)
        sys.exit(1)

    print("FILER SOM HITTADES:")
    for label, path in [("first_order_results", first_order_path),
                        ("sp500",              sp500_path),
                        ("states",             states_path),
                        ("TB3MS",              tbill_path),
                        ("CPIAUCSL",           cpi_path)]:
        print(f"  {label:22s} -> {path}")
    print()

    P_df = pd.read_excel(
        first_order_path,
        sheet_name="P_Smoothed_a0.5",
        index_col=0,
    ).reindex(index=STATE_ORDER, columns=STATE_ORDER)
    P = P_df.values

    results = run_sp500_integration(
        sp500_path         = sp500_path,
        states_xlsx        = states_path,
        tbill_csv          = tbill_path,
        cpi_csv            = cpi_path,
        P_first_order      = P,
        horizon            = 12,
        multistep_horizons = (3, 6, 12),
    )

    out_dir = os.path.join(_script_dir, "output_sp500")
    save_results(results, out_dir)
    make_visualizations(results, out_dir)
