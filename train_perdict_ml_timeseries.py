# train_ml_timeseries.py  ——  机器学习/统计版（无 ARIMA）
import argparse, os, random, logging, joblib, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

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

    # —— 数据读取 —— #
    df = pd.read_excel(cfg.data_path)
    col = df.columns[cfg.well_col]
    ser = df[col].values.astype("float32").reshape(-1, 1)

    # —— 归一化 —— #
    scaler = MinMaxScaler()
    ser_s = scaler.fit_transform(ser).flatten()

    # —— 滑动窗口 & 划分 —— #
    X, y = create_sliding_window(ser_s, cfg.window_size, cfg.pred_len)
    split_idx = int(len(X) * cfg.train_ratio)
    X_tr, y_tr = X[:split_idx], y[:split_idx]
    X_te, y_te = X[split_idx:], y[split_idx:]

    # ========= 训练 & 预测 ========= #
    model, (p_tr_inv, t_tr_inv, p_te_inv, t_te_inv) = fit_predict_ml(
        cfg, X_tr, y_tr, X_te, y_te, scaler
    )
    joblib.dump(model, f"results/{cfg.model}_{col}.joblib")

    # ========= 计算指标 ========= #
    for tag, (p, t) in {
        "Train": (p_tr_inv, t_tr_inv),
        "Test": (p_te_inv, t_te_inv),
    }.items():
        mse, mae, rmse, mape = metrics(t, p)
        log.info(f"[{tag}]  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%")

    # ========= 可视化（只画 horizon-1） ========= #
    def first_step(arr):
        return arr[:, 0]

    pred_tr_1 = first_step(p_tr_inv)
    true_tr_1 = first_step(t_tr_inv)
    pred_te_1 = first_step(p_te_inv)
    true_te_1 = first_step(t_te_inv)

    idx_tr = np.arange(cfg.window_size, cfg.window_size + len(pred_tr_1))
    idx_te = np.arange(
        cfg.window_size + len(pred_tr_1),
        cfg.window_size + len(pred_tr_1) + len(pred_te_1),
    )

    plt.figure(figsize=(10, 4))
    plt.plot(idx_tr, true_tr_1, label="Train True", lw=1)
    plt.plot(idx_tr, pred_tr_1, label="Train Pred", lw=1, ls="--")
    plt.plot(idx_te, true_te_1, label="Test True", lw=1)
    plt.plot(idx_te, pred_te_1, label="Test Pred", lw=1, ls="--")
    plt.xlabel("Time Step")
    plt.ylabel("Groundwater Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/pred_vs_true_{cfg.model}_{col}.png", dpi=300)
    plt.close()


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
