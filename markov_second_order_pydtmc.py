"""
Andra ordningens Markov-modell för konjunkturcykler via STATE EMBEDDING.

Metod
-----
En andra ordningens Markovkedja på tillståndsrymden S kan representeras
som en första ordningens Markovkedja på den embeddade rymden S x S, där
varje "embedded state" är ett par (s_{t-1}, s_t). Detta gör att vi kan
analysera modellen med PyDTMC precis som en vanlig DTMC.

Tillåtna övergångar i den embeddade kedjan:
    (a, b) -> (c, d) endast om b == c.
Övriga övergångar är STRUKTURELLA nollor (inte estimeringsnollor) och
ska aldrig smoothas över.

För |S| = 4 har vi 16 embedded states och 64 tillåtna övergångar.
"""

import numpy as np
import pandas as pd
from itertools import product

from pydtmc import MarkovChain


# ------------------------------------------------------------
# 1) FAST ORDNING PÅ TILLSTÅND
# ------------------------------------------------------------
STATE_ORDER = ["Expansion", "Slowdown", "Contraction", "Recovery"]

# Embedded states: alla par (a, b) i lexikografisk ordning baserad på STATE_ORDER
EMBEDDED_STATES = [f"{a}|{b}" for a, b in product(STATE_ORDER, STATE_ORDER)]
EMBEDDED_PAIRS = list(product(STATE_ORDER, STATE_ORDER))  # samma ordning, som tuples


# ------------------------------------------------------------
# 2) LÄS IN DEN RENA MARKOVSERIEN
# ------------------------------------------------------------
def load_markov_series(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, engine="openpyxl", header=3)
    df.columns = ["Date", "State"]

    df = df.dropna(subset=["State"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["State"] = df["State"].astype(str).str.strip()

    invalid_states = sorted(set(df["State"]) - set(STATE_ORDER))
    if invalid_states:
        raise ValueError(
            f"Okända tillstånd hittades i serien: {invalid_states}. "
            f"Förväntade tillstånd: {STATE_ORDER}"
        )
    return df


# ------------------------------------------------------------
# 3) ANDRA ORDNINGENS COUNTS PÅ GRUNDRYMDEN
# ------------------------------------------------------------
def second_order_counts(sequence, states=STATE_ORDER) -> np.ndarray:
    """
    Räknar trillingar:
    counts[i, j, k] = antal gånger sekvensen (state_i, state_j, state_k) observeras.

    Notera: detta är counts på grundrymden S, INTE på embedded space.
    Konverteringen till embedded transition counts sker i nästa funktion.
    """
    idx = {state: i for i, state in enumerate(states)}
    n = len(states)

    counts = np.zeros((n, n, n), dtype=float)

    for s_a, s_b, s_c in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
        i = idx[s_a]
        j = idx[s_b]
        k = idx[s_c]
        counts[i, j, k] += 1.0

    return counts


# ------------------------------------------------------------
# 4) BYGG DEN EMBEDDADE TRANSITION-MATRISEN
# ------------------------------------------------------------
def embedded_transition_matrix(
    counts3: np.ndarray,
    alpha: float = 0.0,
    states=STATE_ORDER,
) -> np.ndarray:
    """
    Konverterar trilling-counts till en transitionsmatris på embedded state space.

    Definition:
        Embedded state (a, b) representerar "föregående var a, nuvarande är b".
        Övergång (a, b) -> (b, c) har sannolikheten P(c | a, b),
        skattad som count(a, b, c) / sum_c count(a, b, c).

    Smoothing:
        alpha appliceras endast på TILLÅTNA efterföljare, dvs. par som börjar med b.
        Strukturella nollor (övergångar (a,b) -> (c,d) med b != c) förblir exakt 0.

    Returnerar:
        P av form (16, 16) där rad- och kolumnordningen följer EMBEDDED_STATES.
    """
    n = len(states)
    n_emb = n * n
    P = np.zeros((n_emb, n_emb), dtype=float)

    # Snabb mappning: embedded index -> (i, j)
    # Vi använder samma ordning som EMBEDDED_PAIRS = product(STATE_ORDER, STATE_ORDER),
    # vilket innebär: embedded_idx(i, j) = i * n + j
    def emb(i, j):
        return i * n + j

    for i in range(n):           # föregående tillstånd a
        for j in range(n):       # nuvarande tillstånd b
            row = emb(i, j)
            # Tillåtna efterföljande embedded states är (j, k) för alla k
            row_counts = counts3[i, j, :] + alpha       # shape (n,)
            row_sum = row_counts.sum()
            if row_sum > 0:
                probs = row_counts / row_sum
            else:
                # Inget observerat och ingen smoothing - lämna raden 0.
                # Detta hanteras i validate_and_repair() nedan.
                probs = np.zeros(n)

            for k in range(n):   # framtida tillstånd c
                col = emb(j, k)
                P[row, col] = probs[k]

    return P


def validate_and_repair_rows(P: np.ndarray, states=STATE_ORDER) -> np.ndarray:
    """
    Reparerar rader vars summa är 0 (uppstår när alpha = 0 och en kontext (a, b)
    aldrig observerats). Vi sätter då en uniform fördelning över tillåtna
    efterföljare (b, k) för k in states. Detta är en defensiv åtgärd så att
    PyDTMC accepterar matrisen som stokastisk.

    Returnerar en kopia (modifierar inte input).
    """
    n = len(states)
    P = P.copy()
    row_sums = P.sum(axis=1)
    bad = np.where(np.abs(row_sums) < 1e-12)[0]
    for row in bad:
        i, j = divmod(row, n)
        # Tillåtna efterföljare: (j, k) för k = 0..n-1
        for k in range(n):
            col = j * n + k
            P[row, col] = 1.0 / n
    return P


# ------------------------------------------------------------
# 5) HJÄLPFUNKTIONER
# ------------------------------------------------------------
def as_embedded_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    """DataFrame med EMBEDDED_STATES som rad- och kolumnetiketter."""
    return pd.DataFrame(matrix, index=EMBEDDED_STATES, columns=EMBEDDED_STATES)


def expected_duration_second_order(
    counts3: np.ndarray,
    alpha: float = 0.0,
    states=STATE_ORDER,
) -> pd.DataFrame:
    """
    Andra ordningens duration:
    Givet att vi varit i tillstånd b föregående steg och är i b nu (dvs. embedded
    state (b, b)), vad är förväntad återstående duration i b?

    Approximation: 1 / (1 - P(b | b, b)).

    Vi rapporterar också durationen givet alla möjliga föregångare a för att
    visa hur historik påverkar persistens (en av huvudpoängerna med 2:a ordningen).
    """
    n = len(states)
    rows = []
    for i, a in enumerate(states):       # föregående
        for j, b in enumerate(states):   # nuvarande
            row_counts = counts3[i, j, :] + alpha
            tot = row_counts.sum()
            if tot <= 0:
                p_stay = np.nan
                dur = np.nan
            else:
                p_stay = row_counts[j] / tot   # P(c = b | a, b) -> stanna kvar
                dur = np.inf if np.isclose(p_stay, 1.0) else 1.0 / (1.0 - p_stay)
            rows.append({
                "Prev (a)": a,
                "Curr (b)": b,
                "P(stay in b | a, b)": p_stay,
                "Expected duration": dur,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 6) PYDTMC-ANALYS PÅ DEN EMBEDDADE KEDJAN
# ------------------------------------------------------------
def analyze_with_pydtmc_embedded(
    P_embedded: np.ndarray,
    horizon: int = 12,
) -> dict:
    """
    Bygger en PyDTMC-kedja på 16 embedded states och returnerar analysobjekt.

    Notera: ergodicitet och kommunikationsklasser måste tolkas på embedded space.
    Steady state är 16-dimensionellt och marginaliseras separat.
    """
    mc = MarkovChain(P_embedded, EMBEDDED_STATES)

    analysis = {
        "mc": mc,
        "is_ergodic": mc.is_ergodic,
        "is_irreducible": mc.is_irreducible,
        "recurrent_states": mc.recurrent_states,
        "transient_states": mc.transient_states,
        "communicating_classes": mc.communicating_classes,
        "steady_states": mc.steady_states,  # lista av stationära fördelningar (16-dim)
        "entropy_rate": mc.entropy_rate,
        "expected_transitions_horizon": pd.DataFrame(
            mc.expected_transitions(horizon),
            index=EMBEDDED_STATES,
            columns=EMBEDDED_STATES,
        ),
    }
    return analysis


def marginalize_steady_state(pi_emb: np.ndarray, states=STATE_ORDER) -> pd.Series:
    """
    Marginaliserar 16-dim stationär fördelning på embedded space till 4-dim
    fördelning på grundrymden:
        pi_S(s) = sum_a pi_emb(a, s)

    Detta är direkt jämförbart med första ordningens steady state.
    """
    n = len(states)
    pi_emb = np.asarray(pi_emb)
    pi_S = np.zeros(n)
    for i in range(n):       # föregående a
        for j in range(n):   # nuvarande s = b
            pi_S[j] += pi_emb[i * n + j]
    return pd.Series(pi_S, index=states, name="Marginal steady state")


# ------------------------------------------------------------
# 7) MAIN-RUNNER
# ------------------------------------------------------------
def run_second_order_analysis(
    xlsx_path: str,
    alpha_smoothed: float = 0.5,
    horizon: int = 12,
):
    df = load_markov_series(xlsx_path)
    sequence = df["State"].tolist()

    # Trilling-counts på grundrymden
    counts3 = second_order_counts(sequence, states=STATE_ORDER)

    # Embedded transition matrices
    P_mle_raw = embedded_transition_matrix(counts3, alpha=0.0)
    P_mle = validate_and_repair_rows(P_mle_raw)   # ev. uniformerar tomma rader

    P_smoothed = embedded_transition_matrix(counts3, alpha=alpha_smoothed)
    P_smoothed = validate_and_repair_rows(P_smoothed)   # vanligen onödigt med alpha > 0

    # DataFrames
    P_mle_df = as_embedded_dataframe(P_mle)
    P_smoothed_df = as_embedded_dataframe(P_smoothed)

    # Durationsmått
    dur_mle = expected_duration_second_order(counts3, alpha=0.0)
    dur_smoothed = expected_duration_second_order(counts3, alpha=alpha_smoothed)

    # PyDTMC-analys
    analysis_mle = analyze_with_pydtmc_embedded(P_mle, horizon=horizon)
    analysis_smoothed = analyze_with_pydtmc_embedded(P_smoothed, horizon=horizon)

    # Marginaliserade steady states (jämförbara med 1:a ordningens)
    marg_mle = [marginalize_steady_state(ss) for ss in analysis_mle["steady_states"]]
    marg_smoothed = [marginalize_steady_state(ss) for ss in analysis_smoothed["steady_states"]]

    # ---- Utskrifter ----
    print("=" * 80)
    print("DATAÖVERSIKT (2:a ordningen)")
    print("=" * 80)
    print(f"Antal observationer: {len(df)}")
    print(f"Startdatum: {df['Date'].min().date()}")
    print(f"Slutdatum:  {df['Date'].max().date()}")
    print(f"Antal trillingar: {len(sequence) - 2}")
    print(f"Antal embedded states: {len(EMBEDDED_STATES)}  (4^2)")
    print(f"Antal tillåtna övergångar i embedded space: {len(STATE_ORDER) ** 3}  (4^3)")

    n_observed_contexts = int(np.sum(counts3.sum(axis=2) > 0))
    print(f"Antal observerade kontexter (a, b): {n_observed_contexts} av 16")

    print("\n" + "=" * 80)
    print("EMBEDDED TRANSITION MATRIX (MLE) – icke-triviala rader")
    print("=" * 80)
    nonzero_rows = P_mle_df.loc[(P_mle_df.sum(axis=1) > 0).values]
    print(nonzero_rows.round(4))

    print("\n" + "=" * 80)
    print(f"EMBEDDED TRANSITION MATRIX (Smoothed, alpha={alpha_smoothed}) – icke-triviala rader")
    print("=" * 80)
    nonzero_rows_sm = P_smoothed_df.loc[(P_smoothed_df.sum(axis=1) > 0).values]
    print(nonzero_rows_sm.round(4))

    print("\n" + "=" * 80)
    print("FÖRVÄNTAD DURATION I NUVARANDE TILLSTÅND, GIVET FÖREGÅNGARE (MLE)")
    print("=" * 80)
    print(dur_mle.round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("FÖRVÄNTAD DURATION (Smoothed)")
    print("=" * 80)
    print(dur_smoothed.round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("PYDTMC-ANALYS (MLE, embedded chain)")
    print("=" * 80)
    print("Ergodisk:        ", analysis_mle["is_ergodic"])
    print("Irreducibel:     ", analysis_mle["is_irreducible"])
    print("Recurrent states:", analysis_mle["recurrent_states"])
    print("Transient states:", analysis_mle["transient_states"])
    print("Antal kommunicerande klasser:", len(analysis_mle["communicating_classes"]))
    print("Entropy rate:    ", analysis_mle["entropy_rate"])
    print("\nMarginaliserade steady states (MLE):")
    for k, m in enumerate(marg_mle):
        print(f"  Steady state #{k}:")
        print(m.round(4))

    print("\n" + "=" * 80)
    print(f"PYDTMC-ANALYS (Smoothed, alpha={alpha_smoothed}, embedded chain)")
    print("=" * 80)
    print("Ergodisk:        ", analysis_smoothed["is_ergodic"])
    print("Irreducibel:     ", analysis_smoothed["is_irreducible"])
    print("Recurrent states:", analysis_smoothed["recurrent_states"])
    print("Transient states:", analysis_smoothed["transient_states"])
    print("Antal kommunicerande klasser:", len(analysis_smoothed["communicating_classes"]))
    print("Entropy rate:    ", analysis_smoothed["entropy_rate"])
    print("\nMarginaliserade steady states (Smoothed):")
    for k, m in enumerate(marg_smoothed):
        print(f"  Steady state #{k}:")
        print(m.round(4))

    return {
        "df": df,
        "counts3": counts3,
        "P_mle": P_mle,
        "P_mle_df": P_mle_df,
        "P_smoothed": P_smoothed,
        "P_smoothed_df": P_smoothed_df,
        "dur_mle": dur_mle,
        "dur_smoothed": dur_smoothed,
        "analysis_mle": analysis_mle,
        "analysis_smoothed": analysis_smoothed,
        "marginal_steady_mle": marg_mle,
        "marginal_steady_smoothed": marg_smoothed,
    }


# ------------------------------------------------------------
# 8) SPARA RESULTAT
# ------------------------------------------------------------
def save_results(results: dict, out_dir: str, alpha_smoothed: float = 0.5, horizon: int = 12):
    import os
    os.makedirs(out_dir, exist_ok=True)

    xlsx_out = os.path.join(out_dir, "second_order_results.xlsx")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        # Trilling-counts som "stacked" DataFrame (a, b, c) -> count
        rows = []
        for i, a in enumerate(STATE_ORDER):
            for j, b in enumerate(STATE_ORDER):
                for k, c in enumerate(STATE_ORDER):
                    rows.append({"Prev (a)": a, "Curr (b)": b, "Next (c)": c,
                                 "Count": results["counts3"][i, j, k]})
        pd.DataFrame(rows).to_excel(writer, sheet_name="Trilling_counts", index=False)

        results["P_mle_df"].round(6).to_excel(writer, sheet_name="P_MLE_embedded")
        results["P_smoothed_df"].round(6).to_excel(
            writer, sheet_name=f"P_Smoothed_a{alpha_smoothed}_emb"
        )

        results["dur_mle"].to_excel(writer, sheet_name="Duration_MLE", index=False)
        results["dur_smoothed"].to_excel(writer, sheet_name="Duration_Smoothed", index=False)

        # Marginaliserade steady states
        if results["marginal_steady_mle"]:
            pd.concat(
                {f"SS_{k}": s for k, s in enumerate(results["marginal_steady_mle"])},
                axis=1
            ).to_excel(writer, sheet_name="MargSS_MLE")
        if results["marginal_steady_smoothed"]:
            pd.concat(
                {f"SS_{k}": s for k, s in enumerate(results["marginal_steady_smoothed"])},
                axis=1
            ).to_excel(writer, sheet_name="MargSS_Smoothed")

        # Embedded steady states (full 16-dim) - bara MLE för att hålla filen lagom
        ss_mle = results["analysis_mle"]["steady_states"]
        if ss_mle is not None and len(ss_mle) > 0:
            pd.DataFrame(
                np.array(ss_mle), columns=EMBEDDED_STATES
            ).to_excel(writer, sheet_name="EmbSS_MLE", index=False)

        ss_sm = results["analysis_smoothed"]["steady_states"]
        if ss_sm is not None and len(ss_sm) > 0:
            pd.DataFrame(
                np.array(ss_sm), columns=EMBEDDED_STATES
            ).to_excel(writer, sheet_name="EmbSS_Smoothed", index=False)

    print(f"Excel sparat: {xlsx_out}")

    # Textrapport
    txt_out = os.path.join(out_dir, "second_order_summary.txt")
    df = results["df"]
    a_mle = results["analysis_mle"]
    a_sm = results["analysis_smoothed"]
    lines = [
        "=" * 70,
        "SECOND-ORDER MARKOV via STATE EMBEDDING – SAMMANFATTNING",
        "=" * 70,
        f"Observationer       : {len(df)}",
        f"Startdatum          : {df['Date'].min().date()}",
        f"Slutdatum           : {df['Date'].max().date()}",
        f"Antal trillingar    : {len(df) - 2}",
        f"Embedded states     : {len(EMBEDDED_STATES)}",
        "",
        "--- MLE ---",
        f"Ergodisk            : {a_mle['is_ergodic']}",
        f"Irreducibel         : {a_mle['is_irreducible']}",
        f"# Recurrent         : {len(a_mle['recurrent_states'])}",
        f"# Transient         : {len(a_mle['transient_states'])}",
        f"# Komm. klasser     : {len(a_mle['communicating_classes'])}",
        f"Entropy rate        : {a_mle['entropy_rate']}",
        "Marginaliserade SS  :",
    ]
    for k, m in enumerate(results["marginal_steady_mle"]):
        lines.append(f"  #{k}: {m.round(4).to_dict()}")
    lines += [
        "",
        f"--- Smoothed (alpha={alpha_smoothed}) ---",
        f"Ergodisk            : {a_sm['is_ergodic']}",
        f"Irreducibel         : {a_sm['is_irreducible']}",
        f"# Recurrent         : {len(a_sm['recurrent_states'])}",
        f"# Transient         : {len(a_sm['transient_states'])}",
        f"# Komm. klasser     : {len(a_sm['communicating_classes'])}",
        f"Entropy rate        : {a_sm['entropy_rate']}",
        "Marginaliserade SS  :",
    ]
    for k, m in enumerate(results["marginal_steady_smoothed"]):
        lines.append(f"  #{k}: {m.round(4).to_dict()}")

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Textrapport sparad: {txt_out}")


# ------------------------------------------------------------
# 9) VISUALISERINGAR
# ------------------------------------------------------------
def make_visualizations(results: dict, out_dir: str, alpha_smoothed: float = 0.5, horizon: int = 12):
    import os
    import matplotlib.pyplot as plt
    import networkx as nx
    os.makedirs(out_dir, exist_ok=True)

    # -- 1. Heatmap av embedded transition matrix (16x16) --
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, P_df, label in [
        (axes[0], results["P_mle_df"], "MLE"),
        (axes[1], results["P_smoothed_df"], f"Smoothed α={alpha_smoothed}"),
    ]:
        im = ax.imshow(P_df.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(EMBEDDED_STATES)))
        ax.set_yticks(range(len(EMBEDDED_STATES)))
        ax.set_xticklabels(EMBEDDED_STATES, rotation=75, ha="right", fontsize=7)
        ax.set_yticklabels(EMBEDDED_STATES, fontsize=7)
        ax.set_title(f"Embedded Transition Matrix (16x16) – {label}")
        ax.set_xlabel("To (b, c)")
        ax.set_ylabel("From (a, b)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "embedded_transition_heatmap.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Sparad: embedded_transition_heatmap.png")

    # -- 2. Marginaliserade steady states (jämförbara med 1:a ordningen) --
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, marg_list, label in [
        (axes[0], results["marginal_steady_mle"], "MLE"),
        (axes[1], results["marginal_steady_smoothed"], f"Smoothed α={alpha_smoothed}"),
    ]:
        if marg_list:
            ss = marg_list[0]
            bars = ax.bar(STATE_ORDER, ss.values, color="#4C72B0", alpha=0.85)
            for bar, v in zip(bars, ss.values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title(f"Marginal Steady State – {label}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "marginal_steady_states.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Sparad: marginal_steady_states.png")

    # -- 3. Duration-jämförelse: hur påverkar föregångare a persistensen i b? --
    fig, ax = plt.subplots(figsize=(10, 5))
    dur_df = results["dur_smoothed"].copy()
    pivot = dur_df.pivot(index="Curr (b)", columns="Prev (a)", values="Expected duration")
    pivot = pivot.reindex(index=STATE_ORDER, columns=STATE_ORDER)
    pivot.plot(kind="bar", ax=ax, alpha=0.85)
    ax.set_ylabel("Expected duration (months)")
    ax.set_xlabel("Current state b")
    ax.set_title(f"Förväntad duration i b givet föregångare a (Smoothed α={alpha_smoothed})")
    ax.legend(title="Prev (a)", fontsize=8)
    ax.set_xticklabels(STATE_ORDER, rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "duration_by_predecessor.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Sparad: duration_by_predecessor.png")

    # -- 4. Embedded transition graph (16 noder, lite tätt men illustrativt) --
    mc_sm = results["analysis_smoothed"]["mc"]
    G = mc_sm.to_graph()
    # Layout: gruppera noder efter "current" b i fyra kluster
    pos = {}
    cluster_centers = {
        "Expansion":   (0, 0),
        "Slowdown":    (4, 0),
        "Contraction": (4, -4),
        "Recovery":    (0, -4),
    }
    for a, b in EMBEDDED_PAIRS:
        node_name = f"{a}|{b}"
        cx, cy = cluster_centers[b]
        # placera de fyra "from"-noderna kring centret
        ai = STATE_ORDER.index(a)
        angle = 2 * np.pi * ai / 4
        pos[node_name] = (cx + 0.7 * np.cos(angle), cy + 0.7 * np.sin(angle))

    fig, ax = plt.subplots(figsize=(12, 10))
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color="#4C72B0", alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=6, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[max(0.2, w * 3) for w in edge_weights],
        alpha=0.4,
        edge_color="gray",
        connectionstyle="arc3,rad=0.08",
        arrows=True, arrowsize=10,
    )
    ax.set_title(f"Embedded Transition Graph (16 noder) – Smoothed α={alpha_smoothed}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "embedded_graph.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Sparad: embedded_graph.png")


# ------------------------------------------------------------
# 10) KÖRNING
# ------------------------------------------------------------
if __name__ == "__main__":
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _alpha = 0.5
    _horizon = 12
    _out = _os.path.join(_here, "output_2nd_order")

    results = run_second_order_analysis(
        xlsx_path=_os.path.join(_here, "Markovseries clean.xlsx"),
        alpha_smoothed=_alpha,
        horizon=_horizon,
    )
    save_results(results, out_dir=_out, alpha_smoothed=_alpha, horizon=_horizon)
    make_visualizations(results, out_dir=_out, alpha_smoothed=_alpha, horizon=_horizon)
