import numpy as np
import pandas as pd

# PyDTMC
from pydtmc import MarkovChain

# ------------------------------------------------------------
# ------------------------------------------------------------
# Håll samma ordning överallt i projektet:
STATE_ORDER = ["Expansion", "Slowdown", "Contraction", "Recovery"]


# ------------------------------------------------------------
#
# ------------------------------------------------------------
def load_markov_series(xlsx_path: str) -> pd.DataFrame:
    """
    Läser Excel-filen 'Markovseries clean.xlsx'.

    Viktigt:
    - Rubrikerna ligger på rad 4 i filen -> header=3
    - Kolumnerna förväntas vara: Dates, States
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl", header=3)
    df.columns = ["Date", "State"]

    df = df.dropna(subset=["State"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["State"] = df["State"].astype(str).str.strip()

    # Säkerställ att endast kända tillstånd finns
    invalid_states = sorted(set(df["State"]) - set(STATE_ORDER))
    if invalid_states:
        raise ValueError(
            f"Okända tillstånd hittades i serien: {invalid_states}. "
            f"Förväntade tillstånd: {STATE_ORDER}"
        )

    return df


# ------------------------------------------------------------
# 3) RÅA FIRST-ORDER COUNTS
# ------------------------------------------------------------
def first_order_counts(sequence, states=STATE_ORDER) -> np.ndarray:
    """
    Räknar råa övergångar:
    counts[i, j] = antal gånger state_i -> state_j observeras
    """
    idx = {state: i for i, state in enumerate(states)}
    n = len(states)

    counts = np.zeros((n, n), dtype=float)

    for s_t, s_next in zip(sequence[:-1], sequence[1:]):
        i = idx[s_t]
        j = idx[s_next]
        counts[i, j] += 1.0

    return counts


# ------------------------------------------------------------
# 4) FIRST-ORDER TRANSITION MATRIX
# ------------------------------------------------------------
def first_order_transition_matrix(counts: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """
    Skattar first-order transition matrix från counts.

    alpha = 0.0  -> ingen smoothing (ren MLE)
    alpha > 0.0 -> additiv Laplace/Dirichlet-lik smoothing

    Formellt:
    P_ij = (count_ij + alpha) / sum_j(count_ij + alpha)
    """
    counts_smoothed = counts + alpha
    row_sums = counts_smoothed.sum(axis=1, keepdims=True)

    P = np.divide(
        counts_smoothed,
        row_sums,
        out=np.zeros_like(counts_smoothed),
        where=row_sums != 0
    )

    return P


# ------------------------------------------------------------
# 5) HJÄLPFUNKTIONER
# ------------------------------------------------------------
def as_dataframe(matrix: np.ndarray, states=STATE_ORDER) -> pd.DataFrame:
    """Returnerar en snygg DataFrame med labels."""
    return pd.DataFrame(matrix, index=states, columns=states)


def expected_duration_from_diagonal(P: np.ndarray, states=STATE_ORDER) -> pd.Series:
    """
    Enkel in-house durationindikator:
    E[duration_i] ≈ 1 / (1 - p_ii)

    Detta är ett praktiskt komplement till PyDTMC-analysen.
    """
    durations = {}
    for i, state in enumerate(states):
        p_ii = P[i, i]
        if np.isclose(p_ii, 1.0):
            durations[state] = np.inf
        else:
            durations[state] = 1.0 / (1.0 - p_ii)
    return pd.Series(durations, name="Expected Duration (months)")


# ------------------------------------------------------------
# 6) PYDTMC-ANALYS
# ------------------------------------------------------------
def analyze_with_pydtmc(P: np.ndarray, states=STATE_ORDER, horizon: int = 12) -> dict:
    """
    Bygger en PyDTMC-kedja och returnerar centrala analysobjekt.
    """
    mc = MarkovChain(P, states)

    analysis = {
        "mc": mc,
        "is_ergodic": mc.is_ergodic,
        "recurrent_states": mc.recurrent_states,
        "transient_states": mc.transient_states,
        "steady_states": mc.steady_states,  # lista av stationära fördelningar
        "entropy_rate": mc.entropy_rate,
        "expected_transitions_horizon": pd.DataFrame(
            mc.expected_transitions(horizon),
            index=states,
            columns=states
        )
    }

    return analysis


# ------------------------------------------------------------
# 7) MAIN-RUNNER: UTAN OCH MED SMOOTHING
# ------------------------------------------------------------
def run_first_order_analysis(
    xlsx_path: str,
    alpha_smoothed: float = 0.5,
    horizon: int = 12
):
    # Läs in data
    df = load_markov_series(xlsx_path)
    sequence = df["State"].tolist()

    # Råa counts
    counts = first_order_counts(sequence, states=STATE_ORDER)

    # Utan smoothing (MLE)
    P_mle = first_order_transition_matrix(counts, alpha=0.0)

    # Med smoothing
    P_smoothed = first_order_transition_matrix(counts, alpha=alpha_smoothed)

    # DataFrames
    counts_df = as_dataframe(counts, states=STATE_ORDER)
    P_mle_df = as_dataframe(P_mle, states=STATE_ORDER)
    P_smoothed_df = as_dataframe(P_smoothed, states=STATE_ORDER)

    # Enkla durationsmått
    dur_mle = expected_duration_from_diagonal(P_mle, states=STATE_ORDER)
    dur_smoothed = expected_duration_from_diagonal(P_smoothed, states=STATE_ORDER)

    # PyDTMC-analys
    analysis_mle = analyze_with_pydtmc(P_mle, states=STATE_ORDER, horizon=horizon)
    analysis_smoothed = analyze_with_pydtmc(P_smoothed, states=STATE_ORDER, horizon=horizon)

    # Utskrifter
    print("=" * 80)
    print("DATAÖVERSIKT")
    print("=" * 80)
    print(f"Antal observationer: {len(df)}")
    print(f"Startdatum: {df['Date'].min().date()}")
    print(f"Slutdatum:  {df['Date'].max().date()}")
    print("\nState counts:")
    print(df["State"].value_counts())

    print("\n" + "=" * 80)
    print("RÅA FIRST-ORDER COUNTS")
    print("=" * 80)
    print(counts_df)

    print("\nRad-summor i counts:")
    print(counts_df.sum(axis=1))

    print("\n" + "=" * 80)
    print("FIRST-ORDER TRANSITION MATRIX (UTAN SMOOTHING / MLE)")
    print("=" * 80)
    print(P_mle_df.round(4))

    print("\nRad-summor (ska vara 1.0):")
    print(P_mle_df.sum(axis=1).round(6))

    print("\nFörväntad duration (enkel in-house approximation):")
    print(dur_mle.round(3))

    print("\n" + "=" * 80)
    print(f"FIRST-ORDER TRANSITION MATRIX (MED SMOOTHING, alpha={alpha_smoothed})")
    print("=" * 80)
    print(P_smoothed_df.round(4))

    print("\nRad-summor (ska vara 1.0):")
    print(P_smoothed_df.sum(axis=1).round(6))

    print("\nFörväntad duration (enkel in-house approximation):")
    print(dur_smoothed.round(3))

    print("\n" + "=" * 80)
    print("PYDTMC-ANALYS: UTAN SMOOTHING / MLE")
    print("=" * 80)
    print("Ergodisk:", analysis_mle["is_ergodic"])
    print("Recurrent states:", analysis_mle["recurrent_states"])
    print("Transient states:", analysis_mle["transient_states"])
    print("Steady states:", analysis_mle["steady_states"])
    print("Entropy rate:", analysis_mle["entropy_rate"])
    print(f"\nExpected transitions över {horizon} steg:")
    print(analysis_mle["expected_transitions_horizon"].round(4))

    print("\n" + "=" * 80)
    print(f"PYDTMC-ANALYS: MED SMOOTHING, alpha={alpha_smoothed}")
    print("=" * 80)
    print("Ergodisk:", analysis_smoothed["is_ergodic"])
    print("Recurrent states:", analysis_smoothed["recurrent_states"])
    print("Transient states:", analysis_smoothed["transient_states"])
    print("Steady states:", analysis_smoothed["steady_states"])
    print("Entropy rate:", analysis_smoothed["entropy_rate"])
    print(f"\nExpected transitions över {horizon} steg:")
    print(analysis_smoothed["expected_transitions_horizon"].round(4))

    return {
        "df": df,
        "counts": counts,
        "counts_df": counts_df,
        "P_mle": P_mle,
        "P_mle_df": P_mle_df,
        "P_smoothed": P_smoothed,
        "P_smoothed_df": P_smoothed_df,
        "dur_mle": dur_mle,
        "dur_smoothed": dur_smoothed,
        "analysis_mle": analysis_mle,
        "analysis_smoothed": analysis_smoothed,
    }


# ------------------------------------------------------------
# 8) SPARA RESULTAT TILL FIL
# ------------------------------------------------------------
def save_results(results: dict, out_dir: str, alpha_smoothed: float = 0.5, horizon: int = 12):
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Excel med flera flikar
    xlsx_out = os.path.join(out_dir, "first_order_results.xlsx")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        results["counts_df"].to_excel(writer, sheet_name="Counts")
        results["P_mle_df"].round(6).to_excel(writer, sheet_name="P_MLE")
        results["P_smoothed_df"].round(6).to_excel(writer, sheet_name=f"P_Smoothed_a{alpha_smoothed}")
        results["dur_mle"].to_frame().to_excel(writer, sheet_name="Duration_MLE")
        results["dur_smoothed"].to_frame().to_excel(writer, sheet_name="Duration_Smoothed")

        ss_mle = results["analysis_mle"]["steady_states"]
        if ss_mle is not None:
            pd.DataFrame(ss_mle, columns=STATE_ORDER).to_excel(writer, sheet_name="SteadyState_MLE", index=False)

        ss_sm = results["analysis_smoothed"]["steady_states"]
        if ss_sm is not None:
            pd.DataFrame(ss_sm, columns=STATE_ORDER).to_excel(writer, sheet_name="SteadyState_Smoothed", index=False)

        results["analysis_mle"]["expected_transitions_horizon"].round(6).to_excel(
            writer, sheet_name=f"ExpTransitions_MLE_{horizon}steps"
        )
        results["analysis_smoothed"]["expected_transitions_horizon"].round(6).to_excel(
            writer, sheet_name=f"ExpTransitions_Sm_{horizon}steps"
        )

    print(f"Excel sparat: {xlsx_out}")

    # Textrapport
    txt_out = os.path.join(out_dir, "first_order_summary.txt")
    df = results["df"]
    a_mle = results["analysis_mle"]
    a_sm = results["analysis_smoothed"]
    lines = [
        "=" * 70,
        "FIRST-ORDER MARKOV – SAMMANFATTNING",
        "=" * 70,
        f"Observationer : {len(df)}",
        f"Startdatum    : {df['Date'].min().date()}",
        f"Slutdatum     : {df['Date'].max().date()}",
        "",
        "--- MLE ---",
        f"Ergodisk      : {a_mle['is_ergodic']}",
        f"Recurrent     : {a_mle['recurrent_states']}",
        f"Transient     : {a_mle['transient_states']}",
        f"Steady states : {a_mle['steady_states']}",
        f"Entropy rate  : {a_mle['entropy_rate']}",
        "",
        f"--- Smoothed (alpha={alpha_smoothed}) ---",
        f"Ergodisk      : {a_sm['is_ergodic']}",
        f"Recurrent     : {a_sm['recurrent_states']}",
        f"Transient     : {a_sm['transient_states']}",
        f"Steady states : {a_sm['steady_states']}",
        f"Entropy rate  : {a_sm['entropy_rate']}",
    ]
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Textrapport sparad: {txt_out}")


# ------------------------------------------------------------
# 9) VISUALISERINGAR (matplotlib + networkx, pydtmc 8.x har inga plot-metoder)
# ------------------------------------------------------------
def make_visualizations(results: dict, out_dir: str, alpha_smoothed: float = 0.5, horizon: int = 12):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import networkx as nx
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    mc_mle = results["analysis_mle"]["mc"]
    mc_sm  = results["analysis_smoothed"]["mc"]
    pairs  = [(mc_mle, "MLE"), (mc_sm, f"Smoothed_a{alpha_smoothed}")]

    # -- 1. Heatmap av transition matrix --
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, P_df, label in [
        (axes[0], results["P_mle_df"], "MLE"),
        (axes[1], results["P_smoothed_df"], f"Smoothed α={alpha_smoothed}"),
    ]:
        im = ax.imshow(P_df.values, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(STATE_ORDER)))
        ax.set_yticks(range(len(STATE_ORDER)))
        ax.set_xticklabels(STATE_ORDER, rotation=30, ha="right")
        ax.set_yticklabels(STATE_ORDER)
        ax.set_title(f"Transition Matrix – {label}")
        for i in range(len(STATE_ORDER)):
            for j in range(len(STATE_ORDER)):
                ax.text(j, i, f"{P_df.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transition_matrix_heatmap.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Sparad: transition_matrix_heatmap.png")

    # -- 2. Transition graph (networkx DiGraph från pydtmc) --
    pos = {"Expansion": (1, 1), "Slowdown": (0, 0), "Contraction": (1, -1), "Recovery": (2, 0)}
    for mc, label in pairs:
        G = mc.to_graph()
        fig, ax = plt.subplots(figsize=(8, 6))
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1800, node_color="#4C72B0", alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=9, font_weight="bold")
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=[w * 5 for w in edge_weights],
            alpha=0.7,
            edge_color=edge_weights,
            edge_cmap=cm.Blues,
            connectionstyle="arc3,rad=0.1",
            arrows=True, arrowsize=20,
        )
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges() if G[u][v]["weight"] > 0.05}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)
        ax.set_title(f"Transition Graph – {label}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"graph_{label}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Sparad: graph_{label}.png")

    # -- 3. Steady-state bar chart --
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (mc, label) in zip(axes, pairs):
        ss = np.array(mc.steady_states[0])
        bars = ax.bar(STATE_ORDER, ss, color="#4C72B0", alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title(f"Steady-State Distribution – {label}")
        for bar, v in zip(bars, ss):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "steady_states.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Sparad: steady_states.png")

    # -- 4. Redistribution från varje startstate --
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for mc, label in pairs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        for ax, start_state in zip(axes, STATE_ORDER):
            dist_series = mc.redistribute(horizon, start_state, output_last=False)
            dist_array = np.array(dist_series)   # shape (horizon+1, n_states)
            steps = range(len(dist_array))
            for k, (state, color) in enumerate(zip(STATE_ORDER, colors)):
                ax.plot(steps, dist_array[:, k], label=state, color=color, linewidth=2)
            ax.set_title(f"Start: {start_state}")
            ax.set_xlabel("Steg")
            ax.set_ylabel("P(state)")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
        fig.suptitle(f"Redistribution över {horizon} steg – {label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"redistributions_{label}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Sparad: redistributions_{label}.png")


# ------------------------------------------------------------
# 10) KÖRNING
# ------------------------------------------------------------
if __name__ == "__main__":
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _alpha = 0.5
    _horizon = 12
    _out = _os.path.join(_here, "output")

    results = run_first_order_analysis(
        xlsx_path=_os.path.join(_here, "Markovseries clean.xlsx"),
        alpha_smoothed=_alpha,
        horizon=_horizon,
    )
    save_results(results, out_dir=_out, alpha_smoothed=_alpha, horizon=_horizon)
    make_visualizations(results, out_dir=_out, alpha_smoothed=_alpha, horizon=_horizon)
