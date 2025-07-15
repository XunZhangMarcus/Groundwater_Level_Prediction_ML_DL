#!/usr/bin/env python
# train_arima_timeseries.py  ——  ARIMA 自动差分 + 双视图
import argparse, os, logging, warnings, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 0. 工具函数
# ----------------------------------------------------------------------
def adf_pvalue(series):
    """ADF 检验 p 值（越小越稳）"""
    return adfuller(series.dropna())[1]


def ljung_box_pvalue(series, lags=10):
    """Ljung-Box p 值（>0.05 说明近白噪声，建模意义不大）"""
    return acorr_ljungbox(series.dropna(), lags=[lags],
                          return_df=True)["lb_pvalue"].iloc[0]


def make_stationary(series, max_d=2, lags=10, log=None):
    """
    迭代差分，直到：平稳(ADF≤0.05) 且 非白噪声(Ljung-Box≤0.05)
    返回差分阶 d；若失败返回 None
    """
    tmp = series.copy()
    for d in range(max_d + 1):        # d = 0,1,2,...
        adf_p = adf_pvalue(tmp)
        lb_p  = ljung_box_pvalue(tmp, lags=lags)
        if log:
            log.info(f"d={d} → ADF p={adf_p:.4f} , Ljung-Box p={lb_p:.4f}")
        if (adf_p <= .05) and (lb_p <= .05):
            return d
        if d == max_d:
            break
        tmp = tmp.diff().dropna()
    return None


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return np.sqrt(mse), mae


# ----------------------------------------------------------------------
# 1. 主流程
# ----------------------------------------------------------------------
def main(cfg):
    os.makedirs("results", exist_ok=True)
    log = logging.getLogger(__name__)

    # —— 数据读取 —— #
    df = pd.read_excel(cfg.data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    ser_raw = df.iloc[:, cfg.well_col - 1].astype("float32")

    # —— 自动判定差分阶 d —— #
    d = make_stationary(ser_raw, max_d=cfg.max_d, lags=10, log=log)
    if d is None:
        log.error("❌ 序列经最多差分仍不平稳或呈白噪声，ARIMA 不适用，建模终止。")
        sys.exit(1)
    log.info(f"✅ 差分阶 d = {d} ，继续 ARIMA 建模。")

    # —— 划分训练 / 测试（原始序列） —— #
    train_size = int(len(ser_raw) * cfg.train_ratio)
    train_raw, test_raw = ser_raw.iloc[:train_size], ser_raw.iloc[train_size:]

    # —— 网格搜索 (p,q) —— #
    best_aic, best_order = np.inf, None
    for p in range(cfg.max_p + 1):
        for q in range(cfg.max_q + 1):
            try:
                model = sm.tsa.ARIMA(train_raw, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic, best_order = model.aic, (p, d, q)
            except Exception:
                continue
    if best_order is None:
        log.error("未找到可行的 ARIMA(p,d,q) 模型，请扩大搜索范围或检查数据。")
        sys.exit(1)

    log.info(f"最佳阶次 ARIMA{best_order},  AIC = {best_aic:.2f}")

    # —— 最终拟合 & 预测（原始尺度） —— #
    best_model = sm.tsa.ARIMA(train_raw, order=best_order).fit()
    forecast_raw = best_model.get_forecast(steps=len(test_raw)).predicted_mean
    forecast_raw.index = test_raw.index

    # —— 评估指标 —— #
    rmse, mae = metrics(test_raw, forecast_raw)
    log.info(f"[Test] RMSE = {rmse:.4f}  MAE = {mae:.4f}")

    # —— 差分序列真实值 / 预测值 —— #
    diff_true = test_raw.diff(d).dropna()
    diff_pred = forecast_raw.diff(d).dropna()

    # ========= 可视化：分别输出两幅图 ========= #
    # (1) 原始序列
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(train_raw.index, train_raw, label="Train True", lw=1)
    ax1.plot(test_raw.index, test_raw, label="Test  True", lw=1)
    ax1.plot(forecast_raw.index, forecast_raw, label="Test  Pred", lw=1, ls="--")
    ax1.set_ylabel("GWL (original)")
    ax1.set_xlabel("Date")
    ax1.legend()
    orig_path = f"results/pred_vs_true_ARIMA_col{cfg.well_col}_orig.png"
    fig1.tight_layout()
    fig1.savefig(orig_path, dpi=300)
    plt.close(fig1)
    log.info(f"原始序列图已保存至 {orig_path}")

    # (2) d 阶差分序列
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(diff_true.index, diff_true, label="Diff True", lw=1)
    ax2.plot(diff_pred.index, diff_pred, label="Diff Pred", lw=1, ls="--")
    ax2.set_ylabel(f"GWL diff(d={d})")
    ax2.set_xlabel("Date")
    ax2.legend()
    diff_path = f"results/pred_vs_true_ARIMA_col{cfg.well_col}_diff.png"
    fig2.tight_layout()
    fig2.savefig(diff_path, dpi=300)
    plt.close(fig2)
    log.info(f"差分序列图已保存至 {diff_path}")


# ----------------------------------------------------------------------
# 2. 命令行入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="database/ZoupingCounty_gwl_filled.xlsx",
                        help="数据文件路径（Excel/CSV 都可）")
    parser.add_argument("--well_col", type=int, default=4,
                        help="井所在列序号（从 1 开始）")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--max_p", type=int, default=4, help="AR 最大阶")
    parser.add_argument("--max_q", type=int, default=4, help="MA 最大阶")
    parser.add_argument("--max_d", type=int, default=0, help="自动差分最大阶")
    cfg = parser.parse_args()

    main(cfg)

