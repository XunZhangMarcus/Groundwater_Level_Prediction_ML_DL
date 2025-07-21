# train_perdict_ml_timeseries.py  ——  机器学习/统计版
import argparse, os, random, logging, joblib, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import json

# —————————————————— 机器学习器 —————————————————— #
from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 0. 通用工具
# ----------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def create_sliding_window(series, window, pred_len=1):
    X, y = [], []
    for i in range(len(series) - window - pred_len + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + pred_len])
    return np.asarray(X), np.asarray(y)


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return mse, mae, rmse, mape


# ----------------------------------------------------------------------
# 1. 建模与评估
# ----------------------------------------------------------------------
def fit_predict_ml(cfg, X_tr, y_tr, X_te, y_te, scaler):
    """XGBoost / LightGBM：多步预测 → MultiOutputRegressor 包装"""
    if cfg.model == "xgb":
        base = XGBRegressor(
            n_estimators=400,  # 弱学习器棵树数。值越大，模型容量越高，但训练时间与过拟合风险也升高。可与 learning_rate 联调：步长小 → 棵树多。
            max_depth=6, # 单棵树最大深度（LGBM 中 -1 代表不设上限，实际用 num_leaves 控制）。深度越大，单棵树越复杂，能拟合更细的非线性；但也更易过拟合、速度慢。
            learning_rate=0.03, # 每棵树学习步长（缩小权重）。较小值通常需要更多 n_estimators 来达到同样训练误差，却能提升泛化。XGBoost 默认 0.3；实际常用 0.01–0.1。
            subsample=0.8, # 行采样比例。每棵树训练前，随机抽取这部分样本，起到 Bagging 稳定效果，也可加速。0.5–0.9 常用。
            colsample_bytree=0.8, # 列采样比例。对宽特征矩阵尤其有用，能打散特征相关性。
            objective="reg:squarederror", # 损失函数。做回归一般用平方误差（RMSE）；也可以换成 reg:absoluteerror / l1 以提高对离群点鲁棒性。
            tree_method="hist", # 构树算法。hist 是直方图近似算法，比 exact 内存小、速度快。GPU 可用 gpu_hist。
            random_state=42,
        )
    else:  # lgbm
        base = LGBMRegressor(
            n_estimators=800,
            max_depth=-1,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="l2",
            random_state=42,
        )
    model = MultiOutputRegressor(base)
    model.fit(X_tr, y_tr)

    # —— 预测 —— #
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    # —— 反归一化 —— #
    pred_tr_inv = scaler.inverse_transform(pred_tr.reshape(-1, 1)).reshape(
        -1, cfg.pred_len
    )
    true_tr_inv = scaler.inverse_transform(y_tr.reshape(-1, 1)).reshape(
        -1, cfg.pred_len
    )
    pred_te_inv = scaler.inverse_transform(pred_te.reshape(-1, 1)).reshape(
        -1, cfg.pred_len
    )
    true_te_inv = scaler.inverse_transform(y_te.reshape(-1, 1)).reshape(
        -1, cfg.pred_len
    )

    return model, (pred_tr_inv, true_tr_inv, pred_te_inv, true_te_inv)


# ----------------------------------------------------------------------
# 2. 主流程
# ----------------------------------------------------------------------
def main(cfg):
    seed_everything(42)
    os.makedirs("results", exist_ok=True)
    log = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = f"results/single_well_{cfg.model}"
    os.makedirs(output_dir, exist_ok=True)

    # —— 数据读取 —— #
    df = pd.read_excel(cfg.data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    # 获取井位名称和数据
    well_name = df.columns[cfg.well_col - 1]  # 修正索引
    ser = df.iloc[:, cfg.well_col - 1].values.astype("float32").reshape(-1, 1)
    
    log.info(f"开始处理井位: {well_name}")
    log.info(f"使用模型: {cfg.model.upper()}")
    log.info(f"数据长度: {len(ser)}, 有效数据: {(~np.isnan(ser)).sum()}")

    # —— 归一化 —— #
    scaler = MinMaxScaler()
    ser_s = scaler.fit_transform(ser).flatten()

    # —— 滑动窗口 & 划分 —— #
    X, y = create_sliding_window(ser_s, cfg.window_size, cfg.pred_len)
    split_idx = int(len(X) * cfg.train_ratio)
    X_tr, y_tr = X[:split_idx], y[:split_idx]
    X_te, y_te = X[split_idx:], y[split_idx:]
    
    log.info(f"数据划分: 训练集 {len(X_tr)} 样本, 测试集 {len(X_te)} 样本")

    # ========= 训练 & 预测 ========= #
    log.info("开始模型训练和预测...")
    model, (p_tr_inv, t_tr_inv, p_te_inv, t_te_inv) = fit_predict_ml(
        cfg, X_tr, y_tr, X_te, y_te, scaler
    )
    
    # 保存模型
    model_path = f"{output_dir}/{cfg.model}_{well_name}.joblib"
    joblib.dump(model, model_path)
    log.info(f"模型已保存: {model_path}")

    # ========= 计算指标 ========= #
    train_mse, train_mae, train_rmse, train_mape = metrics(t_tr_inv, p_tr_inv)
    test_mse, test_mae, test_rmse, test_mape = metrics(t_te_inv, p_te_inv)
    
    log.info(f"=== 模型性能评估 ===")
    log.info(f"[训练集] RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, MAPE={train_mape:.2f}%")
    log.info(f"[测试集] RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")

    # ========= 保存详细结果到JSON ========= #
    results = {
        'well_name': well_name,
        'model_type': cfg.model.upper(),
        'data_info': {
            'total_length': len(ser),
            'valid_count': int((~np.isnan(ser)).sum()),
            'train_size': len(X_tr),
            'test_size': len(X_te),
            'window_size': cfg.window_size,
            'pred_len': cfg.pred_len
        },
        'model_info': {
            'model_type': cfg.model.upper(),
            'train_ratio': cfg.train_ratio,
            'model_path': model_path
        },
        'train_metrics': {
            'mse': float(train_mse),
            'mae': float(train_mae),
            'rmse': float(train_rmse),
            'mape': float(train_mape)
        },
        'test_metrics': {
            'mse': float(test_mse),
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'mape': float(test_mape)
        }
    }
    
    with open(f"{output_dir}/{cfg.model}_results_{well_name}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"详细结果已保存到: {output_dir}/{cfg.model}_results_{well_name}.json")

    # ========= 生成汇总表格 ========= #
    summary_data = {
        '井位': well_name,
        '模型类型': cfg.model.upper(),
        '训练样本数': len(X_tr),
        '测试样本数': len(X_te),
        '窗口大小': cfg.window_size,
        '预测步长': cfg.pred_len,
        'Train_MSE': f"{train_mse:.6f}",
        'Train_MAE': f"{train_mae:.4f}",
        'Train_RMSE': f"{train_rmse:.4f}",
        'Train_MAPE(%)': f"{train_mape:.2f}",
        'Test_MSE': f"{test_mse:.6f}",
        'Test_MAE': f"{test_mae:.4f}",
        'Test_RMSE': f"{test_rmse:.4f}",
        'Test_MAPE(%)': f"{test_mape:.2f}"
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(f"{output_dir}/{cfg.model}_summary_{well_name}.csv", index=False, encoding='utf-8-sig')
    summary_df.to_excel(f"{output_dir}/{cfg.model}_summary_{well_name}.xlsx", index=False)
    
    log.info(f"汇总结果已保存到: {output_dir}/{cfg.model}_summary_{well_name}.xlsx")

    # ========= 可视化（只画 horizon-1） ========= #
    log.info("生成可视化图表...")
    
    def first_step(arr):
        return arr[:, 0]

    pred_tr_1 = first_step(p_tr_inv)
    true_tr_1 = first_step(t_tr_inv)
    pred_te_1 = first_step(p_te_inv)
    true_te_1 = first_step(t_te_inv)

    # 生成时间索引
    if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
        # 使用原始时间索引
        train_start_idx = cfg.window_size
        train_end_idx = cfg.window_size + len(pred_tr_1)
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + len(pred_te_1)
        
        if len(df.index) > test_end_idx:
            idx_tr = df.index[train_start_idx:train_end_idx]
            idx_te = df.index[test_start_idx:test_end_idx]
        else:
            # 如果索引长度不够，使用数值索引
            idx_tr = np.arange(cfg.window_size, cfg.window_size + len(pred_tr_1))
            idx_te = np.arange(cfg.window_size + len(pred_tr_1), 
                             cfg.window_size + len(pred_tr_1) + len(pred_te_1))
    else:
        # 使用数值索引
        idx_tr = np.arange(cfg.window_size, cfg.window_size + len(pred_tr_1))
        idx_te = np.arange(cfg.window_size + len(pred_tr_1), 
                         cfg.window_size + len(pred_tr_1) + len(pred_te_1))

    # 原始序列预测图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx_tr, true_tr_1, label="Train True", color='blue', alpha=0.7, lw=1)
    ax.plot(idx_tr, pred_tr_1, label="Train Pred", color='blue', alpha=0.7, lw=1, ls='--')
    ax.plot(idx_te, true_te_1, label="Test True", color='red', alpha=0.7, lw=1)
    ax.plot(idx_te, pred_te_1, label="Test Pred", color='red', alpha=0.7, lw=1, ls='--')
    
    ax.set_title(f"{well_name} - {cfg.model.upper()} 时序预测\nTest RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    ax.set_ylabel("地下水位 (GWL)")
    ax.set_xlabel("日期" if hasattr(idx_tr, 'dtype') and 'datetime' in str(idx_tr.dtype) else "时间步")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 如果是时间索引，旋转x轴标签
    if hasattr(idx_tr, 'dtype') and 'datetime' in str(idx_tr.dtype):
        ax.tick_params(axis='x', rotation=45)
    
    pred_path = f"{output_dir}/pred_vs_true_{cfg.model}_{well_name}.png"
    fig.tight_layout()
    fig.savefig(pred_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"预测结果图已保存: {pred_path}")
    
    # —— 最终总结 —— #
    log.info(f"\n=== 单井位{cfg.model.upper()}建模完成 ===")
    log.info(f"井位名称: {well_name}")
    log.info(f"模型类型: {cfg.model.upper()}")
    log.info(f"测试集性能: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    log.info(f"结果保存目录: {output_dir}/")
    
    return results


# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path", type=str, default="database/ZoupingCounty_gwl_filled.xlsx"
    )
    p.add_argument("--well_col", type=int, default=4)
    p.add_argument("--window_size", type=int, default=24)
    p.add_argument("--pred_len", type=int, default=4)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--model", choices=["xgb", "lgbm"], default="lgbm")

    cfg = p.parse_args()
    main(cfg)
