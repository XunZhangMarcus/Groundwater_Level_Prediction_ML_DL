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

### 2.1 Core Training Scripts

| Script                              | Paradigm             | Scope        | Highlights                                                                                                         |
| ----------------------------------- | -------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------ |
| **`train_arima_single_well.py`**   | Classical statistics | Single well  | *Automatic differencing* via ADF & Ljung‑Box; grid search over <i>p, q</i>; detailed logging                     |
| **`train_arima_multi_wells.py`**   | Classical statistics | Multi wells  | *Parallel processing*; batch ARIMA modeling; comprehensive visualization                                           |
| **`train_ml_single_well.py`**      | GBDT (XGB / LGBM)    | Single well  | Multi‑step forecasting through `MultiOutputRegressor`; joblib model persistence                                   |
| **`train_ml_multi_wells.py`**      | GBDT (XGB / LGBM)    | Multi wells  | *Parallel XGB/LGBM training*; automated result aggregation                                                        |
| **`train_3DL_single_well.py`**     | Deep learning        | Single well  | GRU / LSTM / Transformer; derivative‑aware loss *(MSE + α·MSE<sub>Δ</sub>)*; gradient‑clipping                   |
| **`train_3DL_multi_wells.py`**     | Deep learning        | Multi wells  | *Parallel neural training*; selective model training; advanced visualization control                              |

### 2.2 Supporting Infrastructure

| Component                    | Purpose                                                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **`DataCleaning.py`**        | Data preprocessing pipeline: Excel parsing, date handling, interpolation (linear/spline), dual‑plot visualization        |
| **`model/model.py`**         | Neural architecture definitions: GeneratorGRU, GeneratorLSTM, GeneratorTransformer with positional encoding             |
| **`run_all_wells_*.sh`**     | Legacy batch processing scripts for cluster environments                                                                  |
| **`database/`**              | Raw and processed groundwater level data; interpolation plots                                                             |
| **`results/`**               | Structured output directories: `multi_wells_[method]/`, `single_well_[method]/`                                           |

All artefacts follow **deterministic naming conventions** with comprehensive JSON metadata, Excel summaries, and publication‑ready visualizations.

---

## 3 · Methodological Synopsis

### 3.1  ARIMA (<i>p,d,q</i>)

The framework provides both **single‑well** and **multi‑well parallel** ARIMA implementations. Each script first tests stationarity via the Augmented Dickey–Fuller (ADF) statistic and **white‑noise behaviour** via the Ljung–Box test.  Differencing continues until *ADF ≤ 0.05 ∧ LB ≤ 0.05* or a user‑defined maximum <i>d</i>.  AIC‑driven grid search then selects the optimal autoregressive (<i>p</i>) and moving‑average (<i>q</i>) orders.  Forecasts are performed on the **original scale**, preserving physical interpretability.

**Multi‑well enhancements**: ProcessPoolExecutor‑based parallelization, comprehensive error handling, and automated batch visualization generation.

### 3.2  Gradient‑Boosting Decision Trees (GBDT)

Given a normalised sliding‑window $\mathbf{x}_{t-w:t-1}$ and prediction horizon $H$, the framework wraps a base regressor (*XGBRegressor* or *LGBMRegressor*) inside `MultiOutputRegressor`, thereby learning $H$ coupled functions $f_h(\cdot)$.  Default hyper‑parameters favour **moderate depth** (≤6), **learning‑rate annealing** (0.03), and **column/row subsampling** (0.8) to curb over‑fitting on typical hydro‑datasets.

**Parallel processing**: Both single‑well and multi‑well variants support concurrent training across wells and models, with automatic result aggregation and comprehensive performance reporting.

### 3.3  Sequence‑to‑Sequence Neural Networks

Each neural generator treats the input window as a univariate sequence after a singleton channel expansion.  Architectures include:

* **GRU** – gated recurrent unit with lightweight gating; O(<i>T·d²</i>) complexity.
* **LSTM** – enhanced with depthwise‑separable convolution for local feature extraction; improved long‑lag retention.
* **Transformer** – self‑attention encoder with causal masking (no recurrence) offering O(<i>T²</i>) global context.

A composite objective

$
\mathcal{L}=\text{MSE}(y,\hat y) + \alpha\;\text{MSE}(\nabla y,\nabla \hat y)
$

penalises both level and first‑difference errors, promoting **smoother hydrologically plausible trajectories** when $H>1$.  Optimisation uses AdamW with gradient clipping (‖g‖<sub>2</sub> ≤ 1) for numerical stability.

**Advanced features**: Selective model training (`--models transformer,lstm`), flexible visualization control (`--show_wells 1,3,5`), and comprehensive parallel processing with detailed progress tracking.

---

## 4 · Data Interface

* **Format**: Excel/CSV where each column represents a monitoring well; optional `日期` column is parsed as a DateTime index.
* **Pre‑processing**: Min–Max scaling to \[0, 1]; inverse transform automatically applied before metric computation.
* **Sliding Window Parameters**: window size *w*, horizon *H*, and train/test ratio are CLI flags shared across scripts.
* **Data Cleaning**: Automated preprocessing pipeline (`DataCleaning.py`) handles date parsing, interpolation, and generates before/after comparison plots.

> **Note**  Scripts assume evenly spaced records.  For irregular sampling or gap‑filled series, use the provided interpolation pipeline to avoid temporal leakage.

---

## 5 · Quick Start

### 5.1 Data Preprocessing
```bash
# Clean and interpolate raw groundwater data
python DataCleaning.py
# Output: database/ZoupingCounty_gwl_filled.xlsx + visualization plots
```

### 5.2 Single‑Well Modeling
```bash
# ❶ Classical ARIMA (automatic differencing)
python train_arima_single_well.py \
  --data_path database/ZoupingCounty_gwl_filled.xlsx \
  --well_col 4 --max_p 4 --max_q 4 --max_d 2

# ❷ Gradient‑Boosting (LightGBM, 4‑step forecast)
python train_ml_single_well.py \
  --data_path database/ZoupingCounty_gwl_filled.xlsx \
  --well_col 4 --model lgbm --pred_len 4 --window_size 24

# ❃ Transformer (α=0.5 derivative term)
python train_3DL_single_well.py \
  --data_path database/ZoupingCounty_gwl_filled.xlsx \
  --well_col 4 --model transformer --epochs 100 --alpha 0.5
```

### 5.3 Multi‑Well Parallel Processing
```bash
# ❶ Parallel ARIMA for wells 1-5 using 4 processes
python train_arima_multi_wells.py \
  --data_path database/ZoupingCounty_gwl_filled.xlsx \
  --start_col 1 --end_col 5 --parallel --n_jobs 4

# ❷ Parallel ML training (both XGBoost and LightGBM)
python train_ml_multi_wells.py \
  --start_col 1 --end_col 3 --parallel --n_jobs 4 \
  --window_size 24 --pred_len 4

# ❃ Selective deep learning (Transformer only, specific wells)
python train_3DL_multi_wells.py \
  --start_col 1 --end_col 5 --models "transformer" \
  --parallel --n_jobs 2 --show_wells "1,3,5" \
  --epochs 100 --alpha 0.5
```

All scripts log to stdout and persist comprehensive results in structured `results/` directories.

---

## 6 · Requirements

* Python ≥ 3.9
* NumPy, pandas, scikit‑learn
* matplotlib, openpyxl, joblib
* **PyTorch ≥ 2.0** (for deep nets)
* **XGBoost ≥ 1.7** & **LightGBM ≥ 4.0**
* statsmodels (for ARIMA)

Install dependencies:

```bash
pip install torch pandas openpyxl numpy scikit-learn matplotlib joblib xgboost lightgbm statsmodels
```

---

## 7 · Output Structure

Each training run generates comprehensive results:

```
results/
├── multi_wells_[method]/              # Multi-well results
│   ├── multi_wells_[method]_results.json     # Detailed JSON results
│   ├── multi_wells_[method]_summary.xlsx     # Excel summary table
│   ├── pred_vs_true_[model]_[well].png       # Individual prediction plots
│   ├── loss_curve_[model]_[well].png         # Training curves (deep learning)
│   └── multi_wells_[method]_overview.png     # Multi-well overview
└── single_well_[method]/              # Single-well results
    ├── [method]_results_[well].json           # Detailed results
    ├── [method]_summary_[well].xlsx           # Summary table
    └── pred_vs_true_[method]_[well].png       # Prediction plot
```

---

## 8 · Advanced Features

### 8.1 Parallel Processing Control
```bash
--parallel              # Enable parallel processing
--n_jobs 4             # Number of parallel processes
--show_wells "2,4,6"   # Generate plots only for specified wells
--models "transformer" # Train only specified models (deep learning)
```

### 8.2 Flexible Data Range Selection
```bash
--start_col 1          # Starting well column (1-indexed)
--end_col 5            # Ending well column (-1 for all)
--train_ratio 0.8      # Training set proportion
```

### 8.3 Model‑Specific Parameters
```bash
# ARIMA
--max_p 4 --max_q 4 --max_d 2    # Grid search ranges

# Machine Learning
--window_size 24 --pred_len 4     # Sliding window configuration

# Deep Learning
--epochs 100 --lr 1e-3 --alpha 0.5 --batch_size 16
```

---

## 9 · Extensibility

* **Multivariate Drivers**: Extend `create_sliding_window` to accept multi‑channel arrays (e.g., precipitation, pumping rates).
* **Probabilistic Forecasts**: Replace point‑wise losses with quantile (pinball) or Bayesian ensembles.
* **Cross‑Validation**: Plug‐in block bootstrap or time‑series split to replace the fixed hold‑out.
* **Custom Architectures**: Add new models to `model/model.py` and integrate via the factory pattern.

Contributions via pull requests are welcome; please follow the existing linting and unit‑test conventions.

---

## 10 · Citation

If you use this benchmark in academic work, please cite as:

> Zhang X., 2025. *A Unified Benchmark Suite for Groundwater‑Level Time‑Series Forecasting*.  GitHub repository.  DOI: to‑be‑assigned.

---

## 11 · License

Distributed under the MIT License.  See `LICENSE` for details.