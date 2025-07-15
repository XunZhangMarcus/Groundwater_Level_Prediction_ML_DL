# Groundwater_Level_Prediction_ML_DL

*A Comparative Repository of Statistical, Machine‑Learning and Deep‑Learning Forecasters*

---

## 1 · Scope & Motivation

Reliable groundwater‑level (GWL) prediction underpins sustainable aquifer management and hydro‑decision support.  Despite the proliferation of specialised models, reproducible **cross‑paradigm benchmarks** remain scarce.  This repository therefore consolidates three canonical forecasting families—

* **Statistical Auto‑Regressive models** (ARIMA),
* **Tree‑based Gradient‑Boosting regressors** (XGBoost & LightGBM), and
* **Sequence‑to‑Sequence Neural Networks** (GRU, LSTM, Transformer),

into a single, data‑agnostic pipeline tailored to hydro‑temporal series.  All scripts ingest the same *sliding‑window* features and emit harmonised evaluation metrics (RMSE, MAE, MAPE), facilitating apples‑to‑apples comparison and ablation analysis.

---

## 2 · Repository Structure

| Script                          | Paradigm             | Highlights                                                                                                         | Typical Output                                   |
| ------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| **`train_arima_timeseries.py`** | Classical statistics | *Automatic differencing* via ADF & Ljung‑Box; grid search over <i>p, q</i>; dual‑view plots (raw vs. d‑order diff) | PNG plots, best ARIMA order, RMSE/MAE            |
| **`train_ml_timeseries.py`**    | GBDT (XGB / LGBM)    | Multi‑step forecasting through `MultiOutputRegressor`; rich hyper‑parameter defaults; joblib model dump            | Tr/Ts metrics, model .pkl, horizon‑1 plot        |
| **`train_timeseries.py`**       | Deep learning        | GRU / LSTM / Transformer back‑ends; derivative‑aware loss *(MSE + α·MSE<sub>Δ</sub>)*; gradient‑clipping           | Loss curves, prediction plot, best‑epoch weights |

All artefacts are stored under **`results/`** with deterministic filenames, enabling scripted post‑processing.

---

## 3 · Methodological Synopsis

### 3.1  ARIMA (<i>p,d,q</i>)

The script first tests stationarity via the Augmented Dickey–Fuller (ADF) statistic and **white‑noise behaviour** via the Ljung–Box test.  Differencing continues until *ADF ≤ 0.05 ∧ LB ≤ 0.05* or a user‑defined maximum <i>d</i>.  AIC‑driven grid search then selects the optimal autoregressive (<i>p</i>) and moving‑average (<i>q</i>) orders.  Forecasts are performed on the **original scale**, preserving physical interpretability.

### 3.2  Gradient‑Boosting Decision Trees (GBDT)

Given a normalised sliding‑window $\mathbf{x}_{t-w:t-1}$ and prediction horizon $H$, the framework wraps a base regressor (*XGBRegressor* or *LGBMRegressor*) inside `MultiOutputRegressor`, thereby learning $H$ coupled functions $f_h(\cdot)$.  Default hyper‑parameters favour **moderate depth** (≤6), **learning‑rate annealing** (0.03), and **column/row subsampling** (0.8) to curb over‑fitting on typical hydro‑datasets.

### 3.3  Sequence‑to‑Sequence Neural Nets

Each neural generator treats the input window as a univariate sequence after a singleton channel expansion.  Architectures include:

* **GRU** – gated recurrent unit with lightweight gating; O(<i>T·d²</i>) complexity.
* **LSTM** – long short‑term memory with explicit cell state; improved long‑lag retention.
* **Transformer** – self‑attention encoder–decoder (no recurrence) offering O(<i>T²</i>) global context.

A composite objective

$$
\mathcal{L}=\text{MSE}(y,\hat y) + \alpha\;\text{MSE}(\nabla y,\nabla \hat y)
$$

penalises both level and first‑difference errors, promoting **smoother hydrologically plausible trajectories** when $H>1$.  Optimisation uses AdamW with gradient clipping (‖g‖<sub>2</sub> ≤ 1) for numerical stability.

---

## 4 · Data Interface

* **Format**: Excel/CSV where each column represents a monitoring well; optional `日期` column is parsed as a DateTime index.
* **Pre‑processing**: Min–Max scaling to \[0, 1]; inverse transform automatically applied before metric computation.
* **Sliding Window Parameters**: window size *w*, horizon *H*, and train/test ratio are CLI flags shared across scripts.

> **Note**  Scripts assume evenly spaced records.  For irregular sampling or gap‑filled series, interpolate beforehand (e.g., linear or spline) to avoid temporal leakage.

---

## 5 · Quick Start

```bash
# ❶ Classical ARIMA (automatic d)
python train_arima_timeseries.py \
  --data_path database/ZoupingCounty_gwl_filled.xlsx \
  --well_col 4 --max_p 4 --max_q 4 --max_d 2

# ❷ Gradient‑Boosting (LightGBM, 4‑step forecast)
python train_ml_timeseries.py \
  --model lgbm --pred_len 4 --window_size 24

# ❸ Transformer (α=0.5 derivative term)
python train_timeseries.py \
  --model transformer --epochs 200 --alpha 0.5
```

All scripts log to stdout and persist visualisations in `results/`.

---

## 6 · Requirements

* Python ≥ 3.9
* NumPy, pandas, scikit‑learn
* matplotlib, joblib
* **PyTorch ≥ 2.0** (for deep nets)
* **XGBoost ≥ 1.7** & **LightGBM ≥ 4.0**
* statsmodels (for ARIMA)

A conda recipe is provided:

```bash
conda env create -f environment.yml
conda activate gwl_timeseries
```

---

## 7 · Extensibility

* **Multivariate Drivers**: Extend `create_sliding_window` to accept multi‑channel arrays (e.g., precipitation, pumping rates).
* **Probabilistic Forecasts**: Replace point‑wise losses with quantile (pinball) or Bayesian ensembles.
* **Cross‑Validation**: Plug‐in block bootstrap or time‑series split to replace the fixed hold‑out.

Contributions via pull requests are welcome; please follow the existing linting and unit‑test conventions.

---

## 8 · Citation

If you use this benchmark in academic work, please cite as:

> Zhang X., 2025. *A Unified Benchmark Suite for Groundwater‑Level Time‑Series Forecasting*.  GitHub repository.  DOI: to‑be‑assigned.

---

## 9 · License

Distributed under the MIT License.  See `LICENSE` for details.
