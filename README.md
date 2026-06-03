# README

# US Business Cycle Regime Modeling with First- and Second-Order Markov Chains

This repository contains the full implementation used in the bachelor thesis:

> *A Markov Approach to Modelling Business Cycles and Their Implications for Returns on the S&P 500*
>
> KTH Royal Institute of Technology, Spring 2026.

The project constructs a deterministic four-state U.S. business-cycle chronology using industrial production and inflation, estimates first- and second-order Markov chains, performs statistical order-selection tests, validates against NBER recessions, and evaluates regime-based S&P 500 allocation strategies.

---

# Quick Start — Exact Reproduction Steps

> This section was verified end-to-end on a clean checkout. Follow it in order to reproduce the thesis results. It documents three small setup steps that are easy to miss; the rest of the README gives per-script detail.

## 0. Environment

```bash
python -m pip install pandas numpy scipy matplotlib openpyxl pydtmc
```

Python 3.11+ is recommended (3.10 also works).

## 1. One-time data setup (required before gridsearch / walkforward)

`gridsearch.py` and `walkforward.py` read their macro data from an `input/` folder that is **not** part of the repository. The data they need is the `MoM` sheet inside `Markov.xlsx`, so create the folder and copy `Markov.xlsx` into it:

```bash
mkdir input
cp Markov.xlsx input/            # Windows PowerShell/CMD: copy Markov.xlsx input\
```

Without this you get `FileNotFoundError: ...\input`.

## 2. Where each script writes its output

Scripts create their own output **subfolders** (not the project root):

| Script | Output folder |
|---|---|
| `gridsearch.py` | `output/` |
| `walkforward.py` | `output/` |
| `markov_first_order_pydtmc.py` | `output/` |
| `markov_second_order_pydtmc.py` | `output_2nd_order/` |
| `markov_order_tests.py` | `output_order_tests/` |
| `markov_sp500_integration.py` | `output_sp500/` |
| `prediction test.py` | `output_train_estimated_prediction_test/` |

## 3. Run order (copy-paste)

```bash
# Step 1 — Calibration against NBER
python gridsearch.py

# Step 2 — Walk-forward out-of-sample validation
#   (re-optimizes the full 4-D grid per fold; allow several minutes to finish)
python walkforward.py

# Step 3 — First-order Markov chain
python markov_first_order_pydtmc.py

# Step 4 — Second-order Markov chain
python markov_second_order_pydtmc.py

# Step 5 — Order-selection tests
#   This script reads the two results files from the PROJECT ROOT, but
#   steps 3 and 4 wrote them into subfolders. Copy them up first:
cp output/first_order_results.xlsx .
cp output_2nd_order/second_order_results.xlsx .
python markov_order_tests.py

# Step 6 — S&P 500 integration
#   This script expects Markovseries_clean.xlsx (underscore); the repo ships the
#   space-named file. Create the underscore copy once:
cp "Markovseries clean.xlsx" Markovseries_clean.xlsx
python markov_sp500_integration.py

# Step 7 — Out-of-sample prediction comparison
python "prediction test.py"
```

(On Windows, replace `cp` with `copy` and adjust slashes.)

## 4. Expected reproduced values (verification checklist)

| Result | Expected value |
|---|---|
| Calibration best-F1 parameters | θ_i = −0.26, c_b = 0.32, τ = 1 (in-sample F1 ≈ 0.71) |
| First-order: Expansion persistence / π(Expansion) / duration | ≈ 0.92 / ≈ 0.785 / ≈ 12.5 months; chain ergodic |
| Second-order: Exp-after-Exp / Exp-after-Contraction duration | ≈ 13.4 months / ≈ 2 months (Jeffreys-smoothed) |
| Order selection | LRT rejects first-order null; ΔBIC ≈ −207 favors first-order |
| Walk-forward pooled OOS | F1 ≈ 0.64, recall ≈ 0.74, precision ≈ 0.56, accuracy ≈ 0.90 |
| S&P by-regime Sharpe | Recovery highest (≈ 1.68) |
| S&P strategies | Markov Weighted max drawdown ≈ −0.37 vs buy-and-hold ≈ −0.70 |
| Prediction test | first-order accuracy ≈ 80%, second-order accuracy ≈ 78% |

> **Note on θ_c.** The calibration objective (NBER recession F1) does **not** depend on the CPI threshold θ_c, because the recession signal fires whenever industrial production turns down (Slowdown ∪ Contraction). Every value of θ_c yields the identical F1, so a raw grid search may report a θ_c other than the thesis value while achieving the same F1. The thesis fixes θ_c = 0.0 as a neutral default; this choice does not affect any downstream result.

---

# Repository Structure

All files are expected to be placed in the project root directory.

```text
Project root/
│
├── README.md
├── CPIAUCSL.csv
├── TB3MS.csv
├── sp500_inflationadj.xlsx
├── Markov.xlsx
├── Markovseries clean.xlsx
│
├── input/                     # create this; place Markov.xlsx inside (see Quick Start §1)
│   └── Markov.xlsx
│
├── gridsearch.py
├── walkforward.py
├── markov_first_order_pydtmc.py
├── markov_second_order_pydtmc.py
├── markov_order_tests.py
├── markov_sp500_integration.py
└── prediction test.py
```

---

# Python Version

Recommended:

```bash
Python 3.11+
```

---

# Required Packages

Install all dependencies before running any scripts:

```bash
pip install pandas numpy scipy matplotlib openpyxl pydtmc
```

Main libraries used:

- pandas
- numpy
- scipy
- matplotlib
- openpyxl
- pydtmc

---

# Input Files

## 1. Regime Sequence

### `Markovseries clean.xlsx`

Contains the finalized monthly business-cycle state chronology.

Expected structure:

- Header row on row 4
- Columns:
  - `Dates`
  - `States`

Observed states:

- Expansion
- Slowdown
- Contraction
- Recovery

Important:

`markov_sp500_integration.py` looks for the underscore filename:

```text
Markovseries_clean.xlsx
```

but the repository ships the space-named file:

```text
Markovseries clean.xlsx
```

Create the underscore copy once before running the S&P 500 integration (see Quick Start §3, Step 6).

---

## 2. S&P 500 Data

### `sp500_inflationadj.xlsx`

Monthly inflation-adjusted S&P 500 price levels.

Used for:

- log returns
- regime return statistics
- allocation strategies
- Sharpe ratios
- drawdown analysis

---

## 3. Risk-Free Rate Inputs

### `TB3MS.csv`

FRED 3-Month Treasury Bill rate.

### `CPIAUCSL.csv`

FRED CPI All Urban Consumers.

Used together to compute the real monthly risk-free rate:

```text
r_real = (1 + r_nominal) / (1 + inflation) - 1
```

---

## 4. Classification Output / Macro Source

### `Markov.xlsx`

Contains the deterministic business-cycle classification generated from the macroeconomic preprocessing and parameter calibration stages. Its `MoM` sheet (monthly INDPRO/CPI changes and the NBER recession flag) is also the macro source consumed by `gridsearch.py` and `walkforward.py` via the `input/` folder (see Quick Start §1).

---

# Script Overview

## 1. `markov_first_order_pydtmc.py`

Estimates and analyzes the first-order Markov chain.

Main functionality:

- transition counts
- MLE transition matrix
- Jeffreys-smoothed transition matrix
- stationary distribution
- expected durations
- ergodicity analysis
- expected transitions
- Excel export
- summary export

Run:

```bash
python markov_first_order_pydtmc.py
```

Expected output (written to `output/`):

```text
output/first_order_results.xlsx
output/first_order_summary.txt
```

---

## 2. `markov_second_order_pydtmc.py`

Implements the second-order Markov model using state-space embedding.

Main functionality:

- trigram counts
- embedded 16×16 transition matrix
- structural zeros
- Jeffreys smoothing
- conditional durations
- embedded-chain stationary distributions
- marginalization back to base states
- PyDTMC analysis

Run:

```bash
python markov_second_order_pydtmc.py
```

Expected output (written to `output_2nd_order/`):

```text
output_2nd_order/second_order_results.xlsx
output_2nd_order/second_order_summary.txt
```

---

## 3. `markov_order_tests.py`

Performs formal model-order comparison between first- and second-order chains.

Implements:

- Anderson–Goodman likelihood ratio test
- BIC
- AIC
- adjusted degrees of freedom

**Prerequisite:** reads `first_order_results.xlsx` and `second_order_results.xlsx` from the **project root**. Because steps 3 and 4 write to subfolders, copy them up first (see Quick Start §3, Step 5).

Run:

```bash
python markov_order_tests.py
```

Expected output:

- likelihood-ratio statistics
- p-values
- BIC comparison
- preferred order
- `output_order_tests/order_test_results.xlsx`

---

## 4. `markov_sp500_integration.py`

Integrates the first-order Markov model with real S&P 500 returns.

Implements:

- by-regime return statistics
- forward-looking regime probabilities
- conditional expected excess returns
- allocation strategies
- Sharpe analysis
- drawdown analysis

Strategies evaluated:

1. Buy and Hold
2. Naive State Persistence
3. Markov Expected Return
4. Markov Weighted
5. Markov Multi-step

**Prerequisite:** requires `Markovseries_clean.xlsx` (underscore). See Quick Start §3, Step 6.

Run:

```bash
python markov_sp500_integration.py
```

Expected output (written to `output_sp500/`):

- strategy metrics
- cumulative return series
- Sharpe ratios
- regime statistics

---

## 5. `gridsearch.py`

Performs in-sample parameter optimization against the NBER recession chronology.

**Prerequisite:** requires the `input/` folder containing `Markov.xlsx` (see Quick Start §1).

Optimized parameters:

- industrial production threshold
- inflation threshold
- hysteresis buffer
- lag parameter

Metrics:

- precision
- recall
- F1 score

Run:

```bash
python gridsearch.py
```

Expected output (written to `output/`):

- best parameter combinations
- in-sample metrics
- exported result tables

---

## 6. `walkforward.py`

Runs expanding-window walk-forward out-of-sample evaluation.

**Prerequisite:** requires the `input/` folder containing `Markov.xlsx` (see Quick Start §1).

Implements:

- rolling retraining
- parameter re-optimization
- pooled OOS evaluation
- recession classification metrics

> **Runtime:** the full 4-D grid (≈48,000 combinations) is re-optimized for every walk-forward fold, so the script can take several minutes to complete. This is expected; let it run to the end.

Run:

```bash
python walkforward.py
```

Expected output (written to `output/`):

- pooled OOS metrics
- recession timing evaluation
- walk-forward result tables

---

## 7. `prediction test.py`

Out-of-sample prediction comparison between first- and second-order models.

Implements:

- one-step-ahead forecasting
- confusion matrices
- macro-F1
- persistence baseline
- majority baseline

Run:

```bash
python "prediction test.py"
```

Important:

The filename contains a space and therefore must be wrapped in quotation marks in many terminals.

Expected output (written to `output_train_estimated_prediction_test/`):

- prediction comparison tables
- accuracy metrics
- confusion matrices

---

# Recommended Execution Order

For full reproducibility of the thesis workflow, the calibration stage should be performed first. The copy-paste block in **Quick Start §3** runs this entire sequence; the steps below give the rationale.

## Step 1 — Run Gridsearch Against NBER

```bash
python gridsearch.py
```

Purpose:

- calibrates the deterministic classification rule
- optimizes thresholds against NBER recession chronology
- determines the final parameter configuration used throughout the thesis

Main calibrated parameters:

- industrial production threshold
- inflation threshold
- hysteresis buffer
- lag parameter

The finalized parameter values used in the thesis are:

```text
θ_i = -0.26
c_b = 0.32
τ = 1
θ_c = 0.0
```

(See the θ_c note in Quick Start §4: θ_c is not identified by the recession-F1 objective, so a raw grid search may report a different θ_c at the same F1.)

## Step 2 — Run Walk-Forward Validation

```bash
python walkforward.py
```

Purpose:

- validates parameter robustness out-of-sample
- evaluates recession classification stability
- compares against NBER chronology in a rolling expanding-window framework

## Step 3 — Estimate First-Order Chain

```bash
python markov_first_order_pydtmc.py
```

## Step 4 — Estimate Second-Order Chain

```bash
python markov_second_order_pydtmc.py
```

## Step 5 — Perform Order Selection Tests

```bash
cp output/first_order_results.xlsx .
cp output_2nd_order/second_order_results.xlsx .
python markov_order_tests.py
```

## Step 6 — Run S&P 500 Integration

```bash
cp "Markovseries clean.xlsx" Markovseries_clean.xlsx
python markov_sp500_integration.py
```

## Step 7 — Run OOS Prediction Comparison

```bash
python "prediction test.py"
```

---

# Main Empirical Findings

## First-Order Results

- Expansion persistence ≈ 92%
- Long-run Expansion probability ≈ 78%
- Expansion expected duration ≈ 12–13 months
- Chain is ergodic

## Second-Order Results

- Significant path-dependence in Expansion duration
- Expansion following Expansion:
  - ≈ 13 months expected duration
- Expansion following Contraction:
  - ≈ 2 months expected duration

## Order Selection

- LRT rejects first-order null
- BIC strongly favors first-order specification

Interpretation:

The second-order model captures real local path-dependence, but not enough to justify the added complexity as the primary specification.

## S&P 500 Integration

- Recovery regime has the highest Sharpe ratio
- Markov Weighted strategy produces substantially lower drawdowns than buy-and-hold
- Binary sign-based strategies collapse close to buy-and-hold because expected excess returns remain positive across regimes

---

# Common Errors

## `FileNotFoundError: ...\input` (gridsearch / walkforward)

These two scripts read from an `input/` folder that you must create. Run:

```bash
mkdir input
cp Markov.xlsx input/
```

See Quick Start §1.

## `FileNotFoundError: first_order_results.xlsx` (order tests)

`markov_order_tests.py` reads the two results files from the project root, but the first/second-order scripts write them into subfolders. Copy them up first:

```bash
cp output/first_order_results.xlsx .
cp output_2nd_order/second_order_results.xlsx .
```

## `FEL: Hittade inte 'Markovseries_clean.xlsx'` (S&P integration)

Create the underscore-named copy:

```bash
cp "Markovseries clean.xlsx" Markovseries_clean.xlsx
```

## Missing Excel Engine

If you get:

```text
ImportError: Missing optional dependency 'openpyxl'
```

install:

```bash
pip install openpyxl
```

## Missing PyDTMC

If you get:

```text
ModuleNotFoundError: No module named 'pydtmc'
```

install:

```bash
pip install pydtmc
```

## File Not Found Errors (general)

Most scripts expect the input files to exist directly in the project root.

Verify that:

```text
Markovseries clean.xlsx
sp500_inflationadj.xlsx
TB3MS.csv
CPIAUCSL.csv
```

are located in the same folder as the Python scripts, and that the `input/` folder has been created (see Quick Start §1).

---

# Thesis Reference

Hardell, M. & Schmidt, B. (2026).

*A Markov Approach to Modelling Business Cycles and Their Implications for Returns on the S&P 500.*

Bachelor Thesis in Mathematical Statistics (SF100X),
KTH Royal Institute of Technology.

---

# License

This repository is intended for academic and research purposes.
