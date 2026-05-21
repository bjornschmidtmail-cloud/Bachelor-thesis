# README

# US Business Cycle Regime Modeling with First- and Second-Order Markov Chains

This repository contains the full implementation used in the bachelor thesis:

> *A Markov Approach to Modelling Business Cycles and Their Implications for Returns on the S&P 500*
>
> KTH Royal Institute of Technology, Spring 2026.

The project constructs a deterministic four-state U.S. business-cycle chronology using industrial production and inflation, estimates first- and second-order Markov chains, performs statistical order-selection tests, validates against NBER recessions, and evaluates regime-based S&P 500 allocation strategies.

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

Several scripts also support:

```text
Markovseries_clean.xlsx
```

but the repository currently uses:

```text
Markovseries clean.xlsx
```

(with a space).

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

## 4. Classification Output

### `Markov.xlsx`

Contains the deterministic business-cycle classification generated from the macroeconomic preprocessing and parameter calibration stages.

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

Expected output:

```text
first_order_results.xlsx
first_order_summary.txt
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

Expected output:

```text
second_order_results.xlsx
second_order_summary.txt
```

---

## 3. `markov_order_tests.py`

Performs formal model-order comparison between first- and second-order chains.

Implements:

- Anderson–Goodman likelihood ratio test
- BIC
- AIC
- adjusted degrees of freedom

Run:

```bash
python markov_order_tests.py
```

Expected output:

- likelihood-ratio statistics
- p-values
- BIC comparison
- preferred order

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

Run:

```bash
python markov_sp500_integration.py
```

Expected output:

- strategy metrics
- cumulative return series
- Sharpe ratios
- regime statistics

---

## 5. `gridsearch.py`

Performs in-sample parameter optimization against the NBER recession chronology.

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

Expected output:

- best parameter combinations
- in-sample metrics
- exported result tables

---

## 6. `walkforward.py`

Runs expanding-window walk-forward out-of-sample evaluation.

Implements:

- rolling retraining
- parameter re-optimization
- pooled OOS evaluation
- recession classification metrics

Run:

```bash
python walkforward.py
```

Expected output:

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

Expected output:

- prediction comparison tables
- accuracy metrics
- confusion matrices

---

# Recommended Execution Order

For full reproducibility of the thesis workflow, the calibration stage should be performed first.

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
python markov_order_tests.py
```

## Step 6 — Run S&P 500 Integration

```bash
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

## Missing Excel Engine

If you get:

```text
ImportError: Missing optional dependency 'openpyxl'
```

install:

```bash
pip install openpyxl
```

---

## Missing PyDTMC

If you get:

```text
ModuleNotFoundError: No module named 'pydtmc'
```

install:

```bash
pip install pydtmc
```

---

## File Not Found Errors

Most scripts expect the input files to exist directly in the project root.

Verify that:

```text
Markovseries clean.xlsx
sp500_inflationadj.xlsx
TB3MS.csv
CPIAUCSL.csv
```

are located in the same folder as the Python scripts.

---

# Thesis Reference

Hardell, M. & Schmidt, B. (2026).

*A Markov Approach to Modelling Business Cycles and Their Implications for Returns on the S&P 500.*

Bachelor Thesis in Mathematical Statistics (SF100X),
KTH Royal Institute of Technology.

---

# License

This repository is intended for academic and research purposes.

