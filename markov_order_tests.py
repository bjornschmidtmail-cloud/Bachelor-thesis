"""
Statistiska tester för Markov-ordningsval:
  1. Anderson-Goodman Likelihood-Ratio Test (LRT)
  2. BIC-jämförelse

Nollhypotes H0: processen är första ordningens Markov.
Alternativ   H1: processen är andra ordningens Markov.

Designval (Alt A för båda):
  - Oobserverade kontexter (N_ab = 0) exkluderas helt från LRT-summan,
    och df reduceras med (|S| - 1) per sådan kontext.
  - Inom observerade kontexter: termer där n_abc = 0 exkluderas från
    LRT-summan, och df reduceras med 1 per sådan term.
  - BIC använder log-likelihood beräknad på observerade counts utan
    modifikation av nollceller (log(0)-termer bidrar 0 till ell).
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

STATE_ORDER = ["Expansion", "Slowdown", "Contraction", "Recovery"]


# ------------------------------------------------------------
# 1) LÄS IN COUNTS FRÅN EXCEL
# ------------------------------------------------------------
def load_counts(
    first_order_xlsx: str,
    second_order_xlsx: str,
    states: list = STATE_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Läser råa counts från de två Excel-filerna.

    Returnerar:
        counts2 : (n, n)    first-order bigram counts
        counts3 : (n, n, n) second-order trigram counts
    """
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    # --- First-order counts (n x n DataFrame) ---
    df2 = pd.read_excel(first_order_xlsx, sheet_name="Counts", index_col=0)
    # Säkerställ rätt ordning
    df2 = df2.reindex(index=states, columns=states)
    counts2 = df2.values.astype(float)

    # --- Second-order counts (long format: Prev, Curr, Next, Count) ---
    df3 = pd.read_excel(second_order_xlsx, sheet_name="Trilling_counts")
    counts3 = np.zeros((n, n, n), dtype=float)
    for _, row in df3.iterrows():
        a = row["Prev (a)"]
        b = row["Curr (b)"]
        c = row["Next (c)"]
        if a in idx and b in idx and c in idx:
            counts3[idx[a], idx[b], idx[c]] = float(row["Count"])

    return counts2, counts3


# ------------------------------------------------------------
# 2) ANDERSON-GOODMAN LRT
# ------------------------------------------------------------
def anderson_goodman_lrt(
    counts2: np.ndarray,
    counts3: np.ndarray,
    states: list = STATE_ORDER,
) -> dict:
    """
    Beräknar Anderson-Goodman likelihood-ratio test för H0: första ordningen
    mot H1: andra ordningen.

    Teststatistika:
        Lambda = 2 * sum_{a,b,c} n_abc * ln( P2(c|a,b) / P1(c|b) )

    där summan körs endast över termer där:
      (i)  N_ab > 0  (kontexten är observerad)
      (ii) n_abc > 0 (trillingen är observerad)

    Frihetsgrader justeras:
      - Baslinjen är df_full = (|S|-1) * |S|^2 = 3 * 16 = 48
        minus (|S|-1) * |S| = 3 * 4 = 12  [1:a ordningens parametrar]
        = 36
      - För varje oobserverad kontext (a,b): df -= (|S| - 1)
      - För varje oobserverad cell (a,b,c) inom observerad kontext: df -= 1

    Returnerar dict med Lambda, df, p-värde, och diagnostik.
    """
    n = len(states)

    # First-order MLE: P1(c|b) = n_bc / N_b
    N_b = counts2.sum(axis=1)          # rad-summor, shape (n,)
    P1 = np.zeros((n, n))
    for b in range(n):
        if N_b[b] > 0:
            P1[b, :] = counts2[b, :] / N_b[b]

    # Second-order MLE: P2(c|a,b) = n_abc / N_ab
    N_ab = counts3.sum(axis=2)         # shape (n, n)
    P2 = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            if N_ab[a, b] > 0:
                P2[a, b, :] = counts3[a, b, :] / N_ab[a, b]

    # df-beräkning
    df_base = (n - 1) * (n ** 2 - n)   # = (|S|-1) * |S| * (|S|-1) = 36 för |S|=4
    df = df_base
    n_unobserved_contexts = 0
    n_zero_cells = 0

    Lambda = 0.0
    term_details = []   # för diagnostik

    for a in range(n):
        for b in range(n):
            if N_ab[a, b] == 0:
                # Oobserverad kontext: exkludera och justera df
                df -= (n - 1)
                n_unobserved_contexts += 1
                continue

            for c in range(n):
                n_abc = counts3[a, b, c]
                p1_cb = P1[b, c]
                p2_cab = P2[a, b, c]

                if n_abc == 0:
                    # Oobserverad cell inom observerad kontext: exkludera
                    df -= 1
                    n_zero_cells += 1
                    continue

                if p1_cb == 0:
                    # P1(c|b) = 0 men n_abc > 0: detta är en inkonsistens
                    # (kan inte hända om counts2 är marginalen av counts3)
                    # Flagga men hoppa över
                    term_details.append({
                        "a": states[a], "b": states[b], "c": states[c],
                        "n_abc": n_abc, "P2": p2_cab, "P1": p1_cb,
                        "contribution": None, "flag": "P1=0 trots n_abc>0"
                    })
                    continue

                contrib = 2.0 * n_abc * np.log(p2_cab / p1_cb)
                Lambda += contrib
                term_details.append({
                    "a": states[a], "b": states[b], "c": states[c],
                    "n_abc": n_abc, "P2": round(p2_cab, 4), "P1": round(p1_cb, 4),
                    "contribution": round(contrib, 4), "flag": ""
                })

    p_value = 1.0 - chi2.cdf(Lambda, df=df) if df > 0 else np.nan

    return {
        "Lambda": Lambda,
        "df": df,
        "df_base": df_base,
        "df_reduction_contexts": (n - 1) * n_unobserved_contexts,
        "df_reduction_cells": n_zero_cells,
        "n_unobserved_contexts": n_unobserved_contexts,
        "n_zero_cells_excluded": n_zero_cells,
        "p_value": p_value,
        "reject_H0_005": p_value < 0.05 if not np.isnan(p_value) else None,
        "reject_H0_001": p_value < 0.01 if not np.isnan(p_value) else None,
        "term_details": pd.DataFrame(term_details),
    }


# ------------------------------------------------------------
# 3) LOG-LIKELIHOOD-FUNKTIONER
# ------------------------------------------------------------
def loglik_first_order(counts2: np.ndarray) -> float:
    """
    Log-likelihood för första ordningens modell:
        ell_1 = sum_{b,c} n_bc * ln(P1(c|b))
    Nolltermer bidrar 0 (0 * ln(0) = 0 per konvention).
    """
    N_b = counts2.sum(axis=1)
    ell = 0.0
    for b in range(counts2.shape[0]):
        if N_b[b] == 0:
            continue
        for c in range(counts2.shape[1]):
            n_bc = counts2[b, c]
            if n_bc == 0:
                continue
            p1 = n_bc / N_b[b]
            ell += n_bc * np.log(p1)
    return ell


def loglik_second_order(counts3: np.ndarray) -> float:
    """
    Log-likelihood för andra ordningens modell:
        ell_2 = sum_{a,b,c} n_abc * ln(P2(c|a,b))
    Nolltermer bidrar 0.
    """
    N_ab = counts3.sum(axis=2)
    ell = 0.0
    n = counts3.shape[0]
    for a in range(n):
        for b in range(n):
            if N_ab[a, b] == 0:
                continue
            for c in range(n):
                n_abc = counts3[a, b, c]
                if n_abc == 0:
                    continue
                p2 = n_abc / N_ab[a, b]
                ell += n_abc * np.log(p2)
    return ell


# ------------------------------------------------------------
# 4) BIC-JÄMFÖRELSE
# ------------------------------------------------------------
def bic_comparison(
    counts2: np.ndarray,
    counts3: np.ndarray,
    states: list = STATE_ORDER,
) -> dict:
    """
    Beräknar BIC för första och andra ordningens modell.

    Antal fria parametrar:
        k1 = |S| * (|S| - 1) = 4 * 3 = 12   [rader i P1, minus normaliseringsvillkor]
        k2 = |S|^2 * (|S| - 1) = 16 * 3 = 48 [rader i P2, minus normaliseringsvillkor]

    OBS: k2 räknar alla möjliga kontexter, inklusive oobserverade.
    Vi rapporterar också k2_observed = antal kontexter med N_ab > 0 * (|S| - 1),
    som är ett mer konservativt mått på antalet faktiskt skattade parametrar.

    n = antal observationer som bidrar till likelihood:
        n1 = antal bigram = sum(counts2)
        n2 = antal trillingar = sum(counts3)

    BIC = -2 * ell + k * ln(n)
    Delta_BIC = BIC_1 - BIC_2  (positivt = stöd för 2:a ordningen)
    """
    n = len(states)

    ell1 = loglik_first_order(counts2)
    ell2 = loglik_second_order(counts3)

    n1 = counts2.sum()    # antal bigram
    n2 = counts3.sum()    # antal trillingar

    k1 = n * (n - 1)
    k2_full = n ** 2 * (n - 1)

    N_ab = counts3.sum(axis=2)
    n_observed_contexts = int((N_ab > 0).sum())
    k2_observed = n_observed_contexts * (n - 1)

    bic1 = -2 * ell1 + k1 * np.log(n1)
    bic2_full = -2 * ell2 + k2_full * np.log(n2)
    bic2_obs = -2 * ell2 + k2_observed * np.log(n2)

    # AIC för referens (ej primärt mått)
    aic1 = -2 * ell1 + 2 * k1
    aic2 = -2 * ell2 + 2 * k2_full

    return {
        "ell1": ell1,
        "ell2": ell2,
        "k1": k1,
        "k2_full": k2_full,
        "k2_observed": k2_observed,
        "n_observed_contexts": n_observed_contexts,
        "n1": n1,
        "n2": n2,
        "bic1": bic1,
        "bic2_full": bic2_full,
        "bic2_observed": bic2_obs,
        "delta_bic_full": bic1 - bic2_full,       # positivt = stöd för 2:a ordningen
        "delta_bic_observed": bic1 - bic2_obs,
        "aic1": aic1,
        "aic2": aic2,
        "delta_aic": aic1 - aic2,
    }


# ------------------------------------------------------------
# 5) PRESENTATION
# ------------------------------------------------------------
def print_results(lrt: dict, bic: dict, states: list = STATE_ORDER):
    n = len(states)

    print("=" * 70)
    print("ANDERSON-GOODMAN LIKELIHOOD-RATIO TEST")
    print("H0: första ordningens Markov")
    print("H1: andra ordningens Markov")
    print("=" * 70)
    print(f"Teststatistika  Lambda : {lrt['Lambda']:.4f}")
    print(f"Frihetsgrader (justerade):")
    print(f"  df_bas (full modell)   : {lrt['df_base']}")
    print(f"  - df för {lrt['n_unobserved_contexts']:2d} oobs. kontexter : "
          f"-{lrt['df_reduction_contexts']}")
    print(f"  - df för {lrt['n_zero_cells_excluded']:2d} nollceller       : "
          f"-{lrt['df_reduction_cells']}")
    print(f"  df (justerat)          : {lrt['df']}")
    print(f"p-värde                  : {lrt['p_value']:.6f}")
    print(f"Förkasta H0 (α=0.05)    : {lrt['reject_H0_005']}")
    print(f"Förkasta H0 (α=0.01)    : {lrt['reject_H0_001']}")

    print(f"\nChi2-kritiskt värde (α=0.05, df={lrt['df']}): "
          f"{chi2.ppf(0.95, lrt['df']):.4f}")
    print(f"Chi2-kritiskt värde (α=0.01, df={lrt['df']}): "
          f"{chi2.ppf(0.99, lrt['df']):.4f}")

    # Visa de 10 största bidragen till Lambda
    details = lrt["term_details"].copy()
    details_nonzero = details[details["contribution"].notna()].sort_values(
        "contribution", ascending=False
    )
    print(f"\nTop 10 bidrag till Lambda (av {len(details_nonzero)} termer):")
    print(details_nonzero.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("BIC-JÄMFÖRELSE")
    print("=" * 70)
    print(f"Log-likelihood 1:a ordningen : {bic['ell1']:.4f}  (n={int(bic['n1'])} bigram)")
    print(f"Log-likelihood 2:a ordningen : {bic['ell2']:.4f}  (n={int(bic['n2'])} trillingar)")
    print(f"Log-likelihood förbättring   : {bic['ell2'] - bic['ell1']:.4f}")
    print()
    print(f"Parametrar k1 (1:a ordningen)        : {bic['k1']}")
    print(f"Parametrar k2 full (2:a ordningen)   : {bic['k2_full']}")
    print(f"Parametrar k2 observerade kontexter  : {bic['k2_observed']}  "
          f"({bic['n_observed_contexts']} obs. kontexter × {n-1})")
    print()
    print(f"BIC 1:a ordningen          : {bic['bic1']:.4f}")
    print(f"BIC 2:a ordningen (full)   : {bic['bic2_full']:.4f}")
    print(f"BIC 2:a ordningen (obs.)   : {bic['bic2_observed']:.4f}")
    print()
    print(f"ΔBIC (full,  BIC1 - BIC2)  : {bic['delta_bic_full']:.4f}")
    print(f"ΔBIC (obs.,  BIC1 - BIC2)  : {bic['delta_bic_observed']:.4f}")
    print("  (positivt = stöd för 2:a ordningen)")
    print()
    # Tumregler för ΔBIC (Kass & Raftery 1995)
    for label, delta in [("full", bic["delta_bic_full"]),
                         ("obs.", bic["delta_bic_observed"])]:
        if delta > 10:
            strength = "Mycket starkt stöd för 2:a ordningen"
        elif delta > 6:
            strength = "Starkt stöd för 2:a ordningen"
        elif delta > 2:
            strength = "Positivt stöd för 2:a ordningen"
        elif delta > -2:
            strength = "Inget tydligt stöd åt något håll"
        else:
            strength = "Stöd för 1:a ordningen"
        print(f"  ΔBIC ({label}): {strength}")

    print()
    print(f"ΔAIC (för referens, ej primärt mått) : {bic['delta_aic']:.4f}")
    print("  (AIC penaliserar komplexitet svagare än BIC; se metoddiskussion)")


# ------------------------------------------------------------
# 6) SPARA RESULTAT
# ------------------------------------------------------------
def save_results(lrt: dict, bic: dict, out_dir: str):
    import os
    os.makedirs(out_dir, exist_ok=True)

    xlsx_out = os.path.join(out_dir, "order_test_results.xlsx")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        # LRT-sammanfattning
        summary_lrt = pd.DataFrame([{
            "Lambda": lrt["Lambda"],
            "df_base": lrt["df_base"],
            "df_justerat": lrt["df"],
            "n_oobs_kontexter": lrt["n_unobserved_contexts"],
            "n_nollceller_exkl": lrt["n_zero_cells_excluded"],
            "p_value": lrt["p_value"],
            "reject_H0_005": lrt["reject_H0_005"],
            "reject_H0_001": lrt["reject_H0_001"],
        }])
        summary_lrt.to_excel(writer, sheet_name="LRT_summary", index=False)

        # LRT-termdetaljer
        lrt["term_details"].to_excel(writer, sheet_name="LRT_terms", index=False)

        # BIC-sammanfattning
        summary_bic = pd.DataFrame([{
            "ell1": bic["ell1"],
            "ell2": bic["ell2"],
            "k1": bic["k1"],
            "k2_full": bic["k2_full"],
            "k2_observed": bic["k2_observed"],
            "n_observed_contexts": bic["n_observed_contexts"],
            "BIC1": bic["bic1"],
            "BIC2_full": bic["bic2_full"],
            "BIC2_observed": bic["bic2_observed"],
            "delta_BIC_full": bic["delta_bic_full"],
            "delta_BIC_observed": bic["delta_bic_observed"],
            "AIC1": bic["aic1"],
            "AIC2": bic["aic2"],
            "delta_AIC": bic["delta_aic"],
        }])
        summary_bic.to_excel(writer, sheet_name="BIC_summary", index=False)

    print(f"\nResultat sparat: {xlsx_out}")


# ------------------------------------------------------------
# 7) KÖRNING
# ------------------------------------------------------------
def run_order_tests(
    first_order_xlsx: str,
    second_order_xlsx: str,
    out_dir: str,
):
    print("Läser counts från Excel...")
    counts2, counts3 = load_counts(first_order_xlsx, second_order_xlsx)

    print(f"  First-order bigram counts:   {int(counts2.sum())} totalt")
    print(f"  Second-order trilling counts: {int(counts3.sum())} totalt")
    print(f"  Observerade kontexter (a,b): "
          f"{int((counts3.sum(axis=2) > 0).sum())} av {len(STATE_ORDER)**2}\n")

    lrt = anderson_goodman_lrt(counts2, counts3)
    bic = bic_comparison(counts2, counts3)

    print_results(lrt, bic)
    save_results(lrt, bic, out_dir)

    return {"lrt": lrt, "bic": bic}


if __name__ == "__main__":
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))

    results = run_order_tests(
        first_order_xlsx=_os.path.join(_here, "first_order_results.xlsx"),
        second_order_xlsx=_os.path.join(_here, "second_order_results.xlsx"),
        out_dir=_os.path.join(_here, "output_order_tests"),
    )
