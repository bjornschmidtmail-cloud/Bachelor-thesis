
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd


STATE_ORDER = ["Expansion", "Slowdown", "Contraction", "Recovery"]
DEFAULT_SEQUENCE_CANDIDATES = [
    "Markovseries clean.xlsx",
    "Markovseries_clean.xlsx",
    "Markovseries.xlsx",
]


def resolve_sequence_path(input_dir: str | Path | None, sequence: str | Path | None) -> Path:
    """Resolve sequence file from --sequence and/or --input-dir."""
    if input_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(input_dir).expanduser().resolve()

    if sequence:
        seq_path = Path(sequence).expanduser()
        if not seq_path.is_absolute():
            seq_path = base_dir / seq_path
        seq_path = seq_path.resolve()
        if not seq_path.is_file():
            raise FileNotFoundError(f"Could not find sequence file: {seq_path}")
        return seq_path

    for name in DEFAULT_SEQUENCE_CANDIDATES:
        candidate = base_dir / name
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(base_dir / name) for name in DEFAULT_SEQUENCE_CANDIDATES)
    raise FileNotFoundError(
        "No sequence file was specified and none of the default names were found.\n"
        f"Looked for:\n  {tried}\n\n"
        "Pass --sequence explicitly, for example:\n"
        "  --sequence \"C:/path/to/Markovseries clean.xlsx\""
    )


def resolve_output_dir(input_dir: str | Path | None, out_dir: str | Path | None) -> Path:
    """Resolve output directory. If omitted, create outputs inside input-dir or cwd."""
    base_dir = Path(input_dir).expanduser().resolve() if input_dir else Path.cwd().resolve()

    if out_dir:
        out_path = Path(out_dir).expanduser()
        if not out_path.is_absolute():
            out_path = base_dir / out_path
        return out_path.resolve()

    return (base_dir / "output_train_estimated_prediction_test").resolve()


def load_sequence(path: str | Path, header_row: int = 4) -> pd.DataFrame:
    """
    Load the chronological Markov state sequence.

    Assumes the Excel file has two relevant columns: date and state. By default,
    headers are read from row 4, corresponding to pandas header=3.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, header=header_row - 1)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported sequence file type: {path.suffix}. Use .xlsx, .xls, or .csv")

    if df.shape[1] < 2:
        raise ValueError("The sequence file must contain at least two columns: Date and State.")

    # Prefer columns whose names look like date/state; otherwise use the first two columns.
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    date_candidates = [c for c in cols if "date" in c.lower()]
    state_candidates = [c for c in cols if "state" in c.lower()]

    date_col = date_candidates[0] if date_candidates else cols[0]
    state_col = state_candidates[0] if state_candidates else cols[1]

    out = df[[date_col, state_col]].copy()
    out.columns = ["Date", "State"]
    out = out.dropna(subset=["Date", "State"]).copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["State"] = out["State"].astype(str).str.strip()
    out = out.dropna(subset=["Date"])

    invalid = sorted(set(out["State"]) - set(STATE_ORDER))
    if invalid:
        raise ValueError(
            f"Unknown states in sequence: {invalid}\n"
            f"Expected exactly these state labels: {STATE_ORDER}"
        )

    if out.empty:
        raise ValueError("No valid observations were loaded from the sequence file.")

    return out.sort_values("Date").reset_index(drop=True)


def first_order_counts(states: list[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(STATE_ORDER)}
    n = len(STATE_ORDER)
    counts = np.zeros((n, n), dtype=float)
    for current_state, next_state in zip(states[:-1], states[1:]):
        counts[idx[current_state], idx[next_state]] += 1.0
    return counts


def first_order_matrix(counts: np.ndarray, alpha: float) -> np.ndarray:
    counts_s = counts + alpha
    row_sums = counts_s.sum(axis=1, keepdims=True)
    return counts_s / row_sums


def second_order_counts(states: list[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(STATE_ORDER)}
    n = len(STATE_ORDER)
    counts = np.zeros((n, n, n), dtype=float)
    for prev_state, current_state, next_state in zip(states[:-2], states[1:-1], states[2:]):
        counts[idx[prev_state], idx[current_state], idx[next_state]] += 1.0
    return counts


def second_order_tensor(counts: np.ndarray, alpha: float) -> np.ndarray:
    counts_s = counts + alpha
    context_sums = counts_s.sum(axis=2, keepdims=True)
    return counts_s / context_sums


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    f1s = []
    for state in STATE_ORDER:
        tp = sum((yt == state and yp == state) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != state and yp == state) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == state and yp != state) for yt, yp in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def confusion(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    mat = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=int)
    idx = {s: i for i, s in enumerate(STATE_ORDER)}
    for yt, yp in zip(y_true, y_pred):
        mat[idx[yt], idx[yp]] += 1
    return pd.DataFrame(
        mat,
        index=[f"true_{s}" for s in STATE_ORDER],
        columns=[f"pred_{s}" for s in STATE_ORDER],
    )


def baseline_predictions(pred_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return majority and persistence baseline predictions."""
    majority_pred = ["Expansion"] * len(pred_df)
    persistence_pred = pred_df["current_state"].tolist()
    return majority_pred, persistence_pred


def run_prediction_comparison(
    sequence_path: str | Path,
    out_dir: str | Path,
    train_end: str = "2005-12-01",
    test_start: str = "2006-01-01",
    alpha: float = 0.5,
    header_row: int = 4,
) -> dict:
    df = load_sequence(sequence_path, header_row=header_row)

    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    train_df = df[df["Date"] <= train_end_ts].copy()
    if len(train_df) < 3:
        raise ValueError("Training period must contain at least three observations.")

    train_states = train_df["State"].tolist()
    idx = {s: i for i, s in enumerate(STATE_ORDER)}

    counts1 = first_order_counts(train_states)
    P1 = first_order_matrix(counts1, alpha=alpha)

    counts2 = second_order_counts(train_states)
    P2 = second_order_tensor(counts2, alpha=alpha)

    rows = []
    all_states = df["State"].tolist()
    all_dates = df["Date"].tolist()

    # t starts at 1 because second-order prediction needs X_{t-1}, X_t.
    # Test months are current months in the test period; target is next month.
    for t in range(1, len(df) - 1):
        if all_dates[t] < test_start_ts:
            continue

        prev_state = all_states[t - 1]
        current_state = all_states[t]
        true_next = all_states[t + 1]

        first_pred = STATE_ORDER[int(np.argmax(P1[idx[current_state], :]))]
        second_pred = STATE_ORDER[int(np.argmax(P2[idx[prev_state], idx[current_state], :]))]

        rows.append({
            "current_date": all_dates[t],
            "target_date": all_dates[t + 1],
            "prev_state": prev_state,
            "current_state": current_state,
            "true_next_state": true_next,
            "first_order_pred": first_pred,
            "second_order_pred": second_pred,
            "same_prediction": first_pred == second_pred,
            "first_correct": first_pred == true_next,
            "second_correct": second_pred == true_next,
        })

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise ValueError("No test predictions were produced. Check --test-start and the sequence date range.")

    y_true = pred_df["true_next_state"].tolist()
    y_first = pred_df["first_order_pred"].tolist()
    y_second = pred_df["second_order_pred"].tolist()
    y_majority, y_persistence = baseline_predictions(pred_df)

    total = len(pred_df)
    same = int(pred_df["same_prediction"].sum())
    disagree = total - same
    disagreement_df = pred_df[~pred_df["same_prediction"]].copy()

    first_only = int((disagreement_df["first_correct"] & ~disagreement_df["second_correct"]).sum())
    second_only = int((~disagreement_df["first_correct"] & disagreement_df["second_correct"]).sum())
    both_wrong = int((~disagreement_df["first_correct"] & ~disagreement_df["second_correct"]).sum())

    majority_correct = [yt == yp for yt, yp in zip(y_true, y_majority)]
    persistence_correct = [yt == yp for yt, yp in zip(y_true, y_persistence)]

    summary = {
        "sequence_file": str(Path(sequence_path).resolve()),
        "train_start": train_df["Date"].min().date().isoformat(),
        "train_end": train_df["Date"].max().date().isoformat(),
        "test_current_start": pred_df["current_date"].min().date().isoformat(),
        "test_current_end": pred_df["current_date"].max().date().isoformat(),
        "test_target_start": pred_df["target_date"].min().date().isoformat(),
        "test_target_end": pred_df["target_date"].max().date().isoformat(),
        "alpha": alpha,
        "n_train_observations": int(len(train_df)),
        "n_test_predictions": int(total),
        "majority_baseline_accuracy": float(np.mean(majority_correct)),
        "persistence_baseline_accuracy": float(np.mean(persistence_correct)),
        "first_order_accuracy": float(np.mean(pred_df["first_correct"])),
        "second_order_accuracy": float(np.mean(pred_df["second_correct"])),
        "first_order_macro_f1": macro_f1(y_true, y_first),
        "second_order_macro_f1": macro_f1(y_true, y_second),
        "prediction_overlap_count": int(same),
        "prediction_overlap_rate": float(same / total),
        "disagreement_count": int(disagree),
        "disagreement_rate": float(disagree / total),
        "first_order_correct_when_disagree": int(first_only),
        "second_order_correct_when_disagree": int(second_only),
        "both_wrong_when_disagree": int(both_wrong),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(out_dir / "train_estimated_modal_prediction_rows.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "train_estimated_modal_prediction_summary.csv", index=False)
    pd.DataFrame(P1, index=STATE_ORDER, columns=STATE_ORDER).to_csv(
        out_dir / "train_estimated_first_order_matrix.csv"
    )

    tensor_rows = []
    for i, prev_state in enumerate(STATE_ORDER):
        for j, current_state in enumerate(STATE_ORDER):
            for k, next_state in enumerate(STATE_ORDER):
                tensor_rows.append({
                    "prev_state": prev_state,
                    "current_state": current_state,
                    "next_state": next_state,
                    "probability": P2[i, j, k],
                    "train_count": counts2[i, j, k],
                })
    pd.DataFrame(tensor_rows).to_csv(out_dir / "train_estimated_second_order_tensor.csv", index=False)

    confusion(y_true, y_first).to_csv(out_dir / "train_estimated_first_order_confusion.csv")
    confusion(y_true, y_second).to_csv(out_dir / "train_estimated_second_order_confusion.csv")
    confusion(y_true, y_majority).to_csv(out_dir / "majority_baseline_confusion.csv")
    confusion(y_true, y_persistence).to_csv(out_dir / "persistence_baseline_confusion.csv")

    return {
        "summary": summary,
        "rows": pred_df,
        "P1": P1,
        "P2": P2,
        "confusion_first": confusion(y_true, y_first),
        "confusion_second": confusion(y_true, y_second),
        "confusion_majority": confusion(y_true, y_majority),
        "confusion_persistence": confusion(y_true, y_persistence),
        "out_dir": out_dir,
    }


def print_summary(result: dict) -> None:
    summary = result["summary"]
    print("\nSummary")
    print("=======")
    for key, value in summary.items():
        if isinstance(value, float):
            if "accuracy" in key or "rate" in key:
                print(f"{key}: {value:.4f} ({value:.2%})")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nFirst-order confusion matrix")
    print(result["confusion_first"])

    print("\nSecond-order confusion matrix")
    print(result["confusion_second"])

    print(f"\nOutputs saved to: {result['out_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate first- and second-order Markov chains on a train period and compare modal next-state predictions on a test period."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing the input sequence file. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help=(
            "Sequence filename or full path. If omitted, the script searches input-dir for: "
            + ", ".join(DEFAULT_SEQUENCE_CANDIDATES)
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. If omitted, creates output_train_estimated_prediction_test inside input-dir/current directory.",
    )
    parser.add_argument("--train-end", default="2005-12-01", help="Last date included in training period.")
    parser.add_argument("--test-start", default="2006-01-01", help="First current-month date included in test predictions.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Jeffreys/additive smoothing parameter.")
    parser.add_argument(
        "--header-row",
        type=int,
        default=4,
        help="Excel row number containing the headers. Default is 4, corresponding to pandas header=3.",
    )
    args = parser.parse_args()

    try:
        sequence_path = resolve_sequence_path(args.input_dir, args.sequence)
        out_dir = resolve_output_dir(args.input_dir, args.out_dir)
        result = run_prediction_comparison(
            sequence_path=sequence_path,
            out_dir=out_dir,
            train_end=args.train_end,
            test_start=args.test_start,
            alpha=args.alpha,
            header_row=args.header_row,
        )
        print_summary(result)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
