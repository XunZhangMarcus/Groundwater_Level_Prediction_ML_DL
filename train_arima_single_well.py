#!/usr/bin/env python
# train_arima_timeseries.py  ——  单井位ARIMA自动差分建模
import argparse, os, logging, warnings, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import pickle

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 0. 工具函数
# ----------------------------------------------------------------------
def adf_pvalue(series):
    """ADF 检验 p 值（越小越稳）"""
    try:
        return adfuller(series.dropna())[1]
    except:
        return 1.0  # 如果检验失败，返回不显著


def ljung_box_pvalue(series, lags=10):
    """Ljung-Box p 值（>0.05 说明近白噪声，建模意义不大）"""
    try:
        clean_series = series.dropna()
        if len(clean_series) <= lags:
            return 1.0
        return acorr_ljungbox(clean_series, lags=[min(lags, len(clean_series)//4)],
                              return_df=True)["lb_pvalue"].iloc[0]
    except:
        return 1.0


def make_stationary(series, max_d=2, lags=10, log=None):
    """
    迭代差分，直到：平稳(ADF≤0.05) 且 非白噪声(Ljung-Box≤0.05)
    返回差分阶 d；若失败返回 None
    """
    tmp = series.copy()
    for d in range(max_d + 1):        # d = 0,1,2,...
        if len(tmp.dropna()) < 10:    # 数据太少
            return None
        adf_p = adf_pvalue(tmp)
        lb_p  = ljung_box_pvalue(tmp, lags=lags)
        if log:
            log.info(f"  差分阶数 d={d} → ADF p={adf_p:.4f}, Ljung-Box p={lb_p:.4f}")
        if (adf_p <= .05) and (lb_p <= .05):
            return d
        if d == max_d:
            break
        tmp = tmp.diff().dropna()
    return None


def metrics(y_true, y_pred):
    """计算评估指标"""
    # 过滤掉NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + 1e-8))) * 100
    return np.sqrt(mse), mae, mape


# ----------------------------------------------------------------------
# 1. 主流程
# ----------------------------------------------------------------------
def main(cfg):
    os.makedirs("results", exist_ok=True)
    log = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = f"results/single_well_arima"
    os.makedirs(output_dir, exist_ok=True)

    # —— 数据读取 —— #
    df = pd.read_excel(cfg.data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    # 获取井位名称和数据
    well_name = df.columns[cfg.well_col - 1]
    ser_raw = df.iloc[:, cfg.well_col - 1].astype("float32")
    
    log.info(f"开始处理井位: {well_name}")
    log.info(f"数据长度: {len(ser_raw)}, 有效数据: {ser_raw.notna().sum()}")

    # 检查数据有效性
    if ser_raw.isna().all() or len(ser_raw.dropna()) < 20:
        log.error(f"❌ 井位 {well_name} 数据不足或全为空值，建模终止")
        sys.exit(1)

    # —— 自动判定差分阶 d —— #
    log.info("开始自动差分检验...")
    d = make_stationary(ser_raw, max_d=cfg.max_d, lags=10, log=log)
    if d is None:
        log.error(f"❌ 井位 {well_name} 序列经最多差分仍不平稳或呈白噪声，ARIMA 不适用，建模终止")
        sys.exit(1)
    log.info(f"✅ 确定差分阶数 d = {d}，继续 ARIMA 建模")

    # —— 划分训练 / 测试（原始序列） —— #
    train_size = int(len(ser_raw) * cfg.train_ratio)
    train_raw = ser_raw.iloc[:train_size]
    test_raw = ser_raw.iloc[train_size:]
    log.info(f"数据划分: 训练集 {len(train_raw)} 样本, 测试集 {len(test_raw)} 样本")

    # —— 网格搜索 (p,q) —— #
    log.info(f"开始网格搜索 ARIMA(p,{d},q), p∈[0,{cfg.max_p}], q∈[0,{cfg.max_q}]")
    best_aic, best_order = np.inf, None
    search_count = 0
    total_combinations = (cfg.max_p + 1) * (cfg.max_q + 1)
    
    for p in range(cfg.max_p + 1):
        for q in range(cfg.max_q + 1):
            search_count += 1
            try:
                model = sm.tsa.ARIMA(train_raw, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic, best_order = model.aic, (p, d, q)
                    log.info(f"  新最优模型: ARIMA{best_order}, AIC={best_aic:.2f} ({search_count}/{total_combinations})")
            except Exception as e:
                continue
    
    if best_order is None:
        log.error(f"❌ 井位 {well_name} 未找到可行的 ARIMA(p,d,q) 模型，请扩大搜索范围或检查数据")
        sys.exit(1)

    log.info(f"✅ 网格搜索完成，最佳模型: ARIMA{best_order}, AIC = {best_aic:.2f}")

    # —— 最终拟合 & 预测（原始尺度） —— #
    log.info("开始最终模型拟合和预测...")
    best_model = sm.tsa.ARIMA(train_raw, order=best_order).fit()
    
    # 训练集预测（用于评估拟合效果）
    train_pred = best_model.fittedvalues
    
    # 测试集预测
    forecast_raw = best_model.get_forecast(steps=len(test_raw)).predicted_mean
    forecast_raw.index = test_raw.index

    # —— 评估指标 —— #
    train_rmse, train_mae, train_mape = metrics(train_raw.values, train_pred.values)
    test_rmse, test_mae, test_mape = metrics(test_raw.values, forecast_raw.values)
    
    log.info(f"=== 模型性能评估 ===")
    log.info(f"[训练集] RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, MAPE={train_mape:.2f}%")
    log.info(f"[测试集] RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")

    # —— 保存模型 —— #
    model_path = f"{output_dir}/arima_model_{well_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    log.info(f"模型已保存到: {model_path}")

    # —— 保存结果到JSON —— #
    results = {
        'well_name': well_name,
        'data_info': {
            'total_length': len(ser_raw),
            'valid_count': int(ser_raw.notna().sum()),
            'train_size': len(train_raw),
            'test_size': len(test_raw)
        },
        'model_info': {
            'best_order': best_order,
            'best_aic': float(best_aic),
            'diff_order': d,
            'model_path': model_path
        },
        'train_metrics': {
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'mape': float(train_mape)
        },
        'test_metrics': {
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'mape': float(test_mape)
        }
    }
    
    with open(f"{output_dir}/arima_results_{well_name}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"详细结果已保存到: {output_dir}/arima_results_{well_name}.json")

    # ========= 可视化：只输出原始序列预测图 ========= #
    log.info("生成可视化图表...")
    
    # 原始序列预测图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_raw.index, train_raw.values, label="Train True", color='blue', alpha=0.7, lw=1)
    ax.plot(train_pred.index, train_pred.values, label="Train Pred", color='blue', alpha=0.7, lw=1, ls='--')
    ax.plot(test_raw.index, test_raw.values, label="Test True", color='red', alpha=0.7, lw=1)
    ax.plot(forecast_raw.index, forecast_raw.values, label="Test Pred", color='red', alpha=0.7, lw=1, ls='--')
    
    ax.set_title(f"{well_name} - ARIMA{best_order} 时序预测\nTest RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    ax.set_ylabel("地下水位 (GWL)")
    ax.set_xlabel("日期")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    pred_path = f"{output_dir}/pred_vs_true_ARIMA_{well_name}.png"
    fig.tight_layout()
    fig.savefig(pred_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f"预测结果图已保存: {pred_path}")
    
    # —— 最终总结 —— #
    log.info(f"\n=== 单井位ARIMA建模完成 ===")
    log.info(f"井位名称: {well_name}")
    log.info(f"最优模型: ARIMA{best_order}")
    log.info(f"测试集性能: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    log.info(f"结果保存目录: {output_dir}/")
    
    return results


# ----------------------------------------------------------------------
# 2. 命令行入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="单井位ARIMA时序预测建模")
    parser.add_argument("--data_path", type=str,
                        default="database/ZoupingCounty_gwl_filled.xlsx",
                        help="数据文件路径（Excel/CSV 都可）")
    parser.add_argument("--well_col", type=int, default=4,
                        help="井位列序号（从 1 开始，跳过日期列）")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--max_p", type=int, default=4, help="AR 最大阶")
    parser.add_argument("--max_q", type=int, default=4, help="MA 最大阶")
    parser.add_argument("--max_d", type=int, default=2, help="自动差分最大阶")
    
    cfg = parser.parse_args()
    main(cfg)

