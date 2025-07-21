# train_predict_3DL.py  ——  深度学习时序预测版
import argparse, os, random, logging
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
from model.model import GeneratorGRU, GeneratorLSTM, GeneratorTransformer

# ----------------------------------------------------------------------
# 0. 工具函数
# ----------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_sliding_window(series, window, pred_len=1):
    X, y = [], []
    for i in range(len(series) - window - pred_len + 1):
        X.append(series[i:i+window])
        y.append(series[i+window:i+window+pred_len])
    return np.asarray(X), np.asarray(y)

class TimeSeriesDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (B,1,T)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):   return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def get_model(name,window_size, pred_len, device):
    if name == 'gru':
        return GeneratorGRU(input_size=window_size, out_size=pred_len).to(device)
    if name == 'lstm':
        return GeneratorLSTM(input_size=window_size, out_size=pred_len).to(device)
    if name == 'transformer':
        return GeneratorTransformer(input_dim=window_size, output_len=pred_len).to(device)
    raise ValueError(f'Unknown model {name}')

def metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return mse, mae, rmse, mape

# ------------------- Derivative Loss ------------------- #
### <<< NEW
def loss_with_derivative(pred, true, alpha):
    """
    二项损失 = MSE(值) + alpha * MSE(一阶差分)
    对 pred_len==1 时自动退化为纯 MSE。
    """
    mse_val = F.mse_loss(pred, true)
    if pred.size(1) < 2:          # 单步预测，无差分
        return mse_val
    d_pred = pred[:, 1:] - pred[:, :-1]
    d_true = true[:, 1:] - true[:, :-1]
    mse_diff = F.mse_loss(d_pred, d_true)
    return mse_val + alpha * mse_diff
# ------------------------------------------------------- #

# ----------------------------------------------------------------------
# 1. 主流程
# ----------------------------------------------------------------------
def main(cfg):
    seed_everything(np.random.randint(1000))
    os.makedirs('results', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = f"results/single_well_{cfg.model}"
    os.makedirs(output_dir, exist_ok=True)

    # -------- 数据读取 & 归一化 --------
    df = pd.read_excel(cfg.data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    # 获取井位名称和数据
    well_name = df.columns[cfg.well_col - 1]  # 修正索引
    ser = df.iloc[:, cfg.well_col - 1].values.astype('float32').reshape(-1, 1)
    
    log.info(f"开始处理井位: {well_name}")
    log.info(f"使用模型: {cfg.model.upper()}")
    log.info(f"数据长度: {len(ser)}, 有效数据: {(~np.isnan(ser)).sum()}")
    log.info(f"设备: {device}")

    scaler = MinMaxScaler()
    ser_s  = scaler.fit_transform(ser).flatten()

    X, y = create_sliding_window(ser_s, cfg.window_size, cfg.pred_len)
    split = int(len(X) * cfg.train_ratio)
    
    log.info(f"数据划分: 训练集 {len(X[:split])} 样本, 测试集 {len(X[split:])} 样本")

    train_loader = DataLoader(TimeSeriesDS(X[:split], y[:split]),
                              batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(TimeSeriesDS(X[split:], y[split:]),
                              batch_size=cfg.batch_size, shuffle=False)

    model = get_model(cfg.model,cfg.window_size, cfg.pred_len, device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    crit  = nn.MSELoss()

    best_rmse   = float('inf')
    save_path   = f'{output_dir}/best_{cfg.model}_{well_name}.pt'

    train_loss_hist, test_loss_hist = [], []
    metric_file = open(f'{output_dir}/metrics_{cfg.model}_{well_name}.txt', 'w', encoding='utf-8')
    metric_file.write('#Epoch\tRMSE\tMAE\tMAPE(%)\n')

    # 评估与绘图时 只取每个窗口预测中的第 1 个时间步，这样时间对齐简单且不需要递归。是绝大多数论文的默认做法。
    def _predict(loader):
        """
        返回 two arrays:
            - pred_flat: 预测序列（只取 horizon-1）
            - true_flat: 对齐的真实序列
        """
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb.to(device)).cpu().numpy()  # (B, pred_len)

                preds.append(out[:, 0])  # 只取 t+1
                trues.append(yb[:, 0].numpy())  # 同样只要 t+1 真值

        p = np.concatenate(preds, 0).reshape(-1, 1)
        t = np.concatenate(trues, 0).reshape(-1, 1)
        p = scaler.inverse_transform(p).flatten()
        t = scaler.inverse_transform(t).flatten()
        return p, t

    # def _predict(loader):
    #     p, t = [], []
    #     with torch.no_grad():
    #         for xb, yb in loader:
    #             p.append(model(xb.to(device)).cpu().numpy())
    #             t.append(yb.numpy())
    #     p = np.concatenate(p).reshape(-1, cfg.pred_len)
    #     t = np.concatenate(t).reshape(-1, cfg.pred_len)
    #     p = scaler.inverse_transform(p.reshape(-1, 1)).reshape(-1, cfg.pred_len)
    #     t = scaler.inverse_transform(t.reshape(-1, 1)).reshape(-1, cfg.pred_len)
    #     return p.flatten(), t.flatten()

    # ------------------------------------------------------------------
    # 2. 训练
    # ------------------------------------------------------------------
    for ep in range(1, cfg.epochs + 1):
        # —— Train —— #
        model.train(); tr_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            # loss = crit(out, yb)
            loss = loss_with_derivative(out, yb, alpha=cfg.alpha)  ### <<< NEW
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        # —— Eval —— #
        model.eval(); preds, trues = [], []
        test_loss_sum, test_cnt = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                out = model(xb).cpu().numpy()
                preds.append(out); trues.append(yb.numpy())

                test_loss_sum += crit(torch.tensor(out), yb).item() * xb.size(0)
                test_cnt      += xb.size(0)

        preds = np.concatenate(preds, 0).reshape(-1, cfg.pred_len)
        trues = np.concatenate(trues, 0).reshape(-1, cfg.pred_len)

        # 反归一化
        preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)) \
                            .reshape(-1, cfg.pred_len)
        trues_inv = scaler.inverse_transform(trues.reshape(-1, 1)) \
                            .reshape(-1, cfg.pred_len)

        mse, mae, rmse, mape = metrics(trues_inv, preds_inv)
        test_loss = test_loss_sum / test_cnt

        train_loss_hist.append(tr_loss)
        test_loss_hist.append(test_loss)
        metric_file.write(f'{ep}\t{rmse:.4f}\t{mae:.4f}\t{mape:.2f}\n')

        log.info(f'Epoch {ep:03d} | TrainLoss {tr_loss:.4f} | '
                 f'RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.2f}%')

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), save_path)

    log.info(f'Finished. Best RMSE={best_rmse:.4f}  model→ {save_path}')
    metric_file.close()

    # ------------------------------------------------------------------
    # 3. 最终评估和结果保存
    # ------------------------------------------------------------------
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # 计算最终的训练集和测试集指标
    pred_tr, true_tr = _predict(train_loader)
    pred_te, true_te = _predict(test_loader)
    
    # 计算最终指标
    train_mse, train_mae, train_rmse, train_mape = metrics(true_tr, pred_tr)
    test_mse, test_mae, test_rmse, test_mape = metrics(true_te, pred_te)
    
    log.info(f"=== 最终模型性能评估 ===")
    log.info(f"[训练集] RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, MAPE={train_mape:.2f}%")
    log.info(f"[测试集] RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")

    # ========= 保存详细结果到JSON ========= #
    results = {
        'well_name': well_name,
        'model_type': cfg.model.upper(),
        'data_info': {
            'total_length': len(ser),
            'valid_count': int((~np.isnan(ser)).sum()),
            'train_size': len(X[:split]),
            'test_size': len(X[split:]),
            'window_size': cfg.window_size,
            'pred_len': cfg.pred_len
        },
        'model_info': {
            'model_type': cfg.model.upper(),
            'epochs': cfg.epochs,
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.lr,
            'alpha': cfg.alpha,
            'train_ratio': cfg.train_ratio,
            'best_rmse': float(best_rmse),
            'model_path': save_path
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
        '训练样本数': len(X[:split]),
        '测试样本数': len(X[split:]),
        '窗口大小': cfg.window_size,
        '预测步长': cfg.pred_len,
        '训练轮数': cfg.epochs,
        '学习率': cfg.lr,
        'Alpha': cfg.alpha,
        '最佳RMSE': f"{best_rmse:.4f}",
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

    # ------------------------------------------------------------------
    # 4. 画 LOSS 曲线
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(test_loss_hist,  label='Test Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{well_name} - {cfg.model.upper()} 训练损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_curve_path = f'{output_dir}/loss_curve_{cfg.model}_{well_name}.png'
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"损失曲线已保存: {loss_curve_path}")

    # ------------------------------------------------------------------
    # 5. 预测 vs. 真实（使用与其他脚本一致的格式）
    # ------------------------------------------------------------------
    # 生成时间索引
    if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
        # 使用原始时间索引
        train_start_idx = cfg.window_size
        train_end_idx = cfg.window_size + len(pred_tr)
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + len(pred_te)
        
        if len(df.index) > test_end_idx:
            idx_tr = df.index[train_start_idx:train_end_idx]
            idx_te = df.index[test_start_idx:test_end_idx]
        else:
            # 如果索引长度不够，使用数值索引
            idx_tr = np.arange(cfg.window_size, cfg.window_size + len(pred_tr))
            idx_te = np.arange(cfg.window_size + len(pred_tr), 
                             cfg.window_size + len(pred_tr) + len(pred_te))
    else:
        # 使用数值索引
        idx_tr = np.arange(cfg.window_size, cfg.window_size + len(pred_tr))
        idx_te = np.arange(cfg.window_size + len(pred_tr), 
                         cfg.window_size + len(pred_tr) + len(pred_te))

    # 原始序列预测图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx_tr, true_tr, label="Train True", color='blue', alpha=0.7, lw=1)
    ax.plot(idx_tr, pred_tr, label="Train Pred", color='blue', alpha=0.7, lw=1, ls='--')
    ax.plot(idx_te, true_te, label="Test True", color='red', alpha=0.7, lw=1)
    ax.plot(idx_te, pred_te, label="Test Pred", color='red', alpha=0.7, lw=1, ls='--')
    
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
    log.info(f"最佳RMSE: {best_rmse:.4f}")
    log.info(f"最终测试集性能: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    log.info(f"结果保存目录: {output_dir}/")
    
    return results

# ----------------------------------------------------------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str,
                   default='database/ZoupingCounty_gwl_filled.xlsx')
    p.add_argument('--well_col', type=int, default=4)
    p.add_argument('--window_size', type=int, default=24)
    p.add_argument('--pred_len', type=int, default=4)
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--model', choices=['gru', 'lstm', 'transformer'],
                   default='transformer')
    p.add_argument('--alpha',       type=float, default=0.5,      ### <<< NEW
                   help='weight of derivative loss term')
    cfg = p.parse_args()
    main(cfg)
