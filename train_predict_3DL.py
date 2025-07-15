# train_timeseries.py  ——  全量替换版
import argparse, os, random, logging
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn.functional as F
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

    # -------- 数据读取 & 归一化 --------
    df   = pd.read_excel(cfg.data_path)
    col  = df.columns[cfg.well_col]
    ser  = df[col].values.astype('float32').reshape(-1, 1)

    scaler = MinMaxScaler()
    ser_s  = scaler.fit_transform(ser).flatten()

    X, y = create_sliding_window(ser_s, cfg.window_size, cfg.pred_len)
    split = int(len(X) * cfg.train_ratio)

    train_loader = DataLoader(TimeSeriesDS(X[:split], y[:split]),
                              batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(TimeSeriesDS(X[split:], y[split:]),
                              batch_size=cfg.batch_size, shuffle=False)

    model = get_model(cfg.model,cfg.window_size, cfg.pred_len, device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    crit  = nn.MSELoss()

    best_rmse   = float('inf')
    save_path   = f'./results/best_{cfg.model}_{col}.pt'

    train_loss_hist, test_loss_hist = [], []
    metric_file = open('results/metrics.txt', 'w', encoding='utf-8')
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
    # 3. 画 LOSS 曲线
    # ------------------------------------------------------------------
    plt.figure()
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(test_loss_hist,  label='Test  Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.tight_layout()
    # 根据不同的model调整保存名称
    plt.savefig(f'results/loss_curve_{cfg.model}_{col}.png', dpi=300)
    plt.close()

    # ------------------------------------------------------------------
    # 4. 预测 vs. 真实（最后 epoch，Train + Test）
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    pred_tr, true_tr = _predict(train_loader)
    pred_te, true_te = _predict(test_loader)

    idx_tr = np.arange(cfg.window_size, cfg.window_size + len(pred_tr))
    idx_te = np.arange(cfg.window_size + len(pred_tr),
                       cfg.window_size + len(pred_tr) + len(pred_te))

    plt.figure(figsize=(10, 4))
    plt.plot(idx_tr, true_tr, label='Train True',  lw=1)
    plt.plot(idx_tr, pred_tr, label='Train Pred',  lw=1, ls='--')
    plt.plot(idx_te, true_te, label='Test True',   lw=1)
    plt.plot(idx_te, pred_te, label='Test Pred',   lw=1, ls='--')
    plt.xlabel('Time Step'); plt.ylabel('Groundwater Level')
    plt.legend(); plt.tight_layout()
    # 根据不同的model调整保存名称
    plt.savefig(f'results/pred_vs_true_{cfg.model}_{col}.png', dpi=300)
    plt.close()

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
