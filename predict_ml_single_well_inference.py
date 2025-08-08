# predict_ml_inference.py - 使用训练好的XGB/LGBM模型进行预测
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import joblib
import warnings

# —————————————————— 机器学习器 —————————————————— #
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: xgboost 未安装")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: lightgbm 未安装")

warnings.filterwarnings("ignore", category=UserWarning)

# 设置matplotlib
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

def create_sliding_window(series, window, pred_len=1):
    """创建滑动窗口数据 - 与训练脚本完全相同"""
    X, y = [], []
    for i in range(len(series) - window - pred_len + 1):
        X.append(series[i:i+window])
        y.append(series[i+window:i+window+pred_len])
    return np.asarray(X), np.asarray(y)

def metrics(y_true, y_pred):
    """计算评估指标 - 与训练脚本完全相同"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return mse, mae, rmse, mape

def predict_with_model(model_path, data_path, well_col=4, window_size=24, pred_len=4, train_ratio=0.8):
    """
    使用训练好的XGB/LGBM模型进行预测
    完全仿照train_ml_single_well.py的处理方式
    
    Args:
        model_path: 模型权重文件路径 (.joblib)
        data_path: 数据文件路径
        well_col: 井位列索引
        window_size: 窗口大小
        pred_len: 预测长度
        train_ratio: 训练集比例（用于划分显示）
    """
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return None
    
    # 从文件名推断模型类型
    filename = os.path.basename(model_path)
    if 'xgb' in filename.lower():
        model_type = 'XGB'
    elif 'lgbm' in filename.lower():
        model_type = 'LGBM'
    else:
        model_type = 'ML'
    
    print(f"使用模型类型: {model_type}")
    
    # 1. 导入数据 - 与训练脚本完全相同
    print("正在加载数据...")
    df = pd.read_excel(data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    # 获取井位名称和数据 - 与训练脚本完全相同
    well_name = df.columns[well_col - 1]  # 修正索引
    ser = df.iloc[:, well_col - 1].values.astype("float32").reshape(-1, 1)
    
    print(f"井位名称: {well_name}")
    print(f"数据长度: {len(ser)}, 有效数据: {(~np.isnan(ser)).sum()}")
    
    # 数据归一化 - 与训练脚本完全相同
    scaler = MinMaxScaler()
    ser_s = scaler.fit_transform(ser).flatten()
    
    # 创建滑动窗口 - 与训练脚本完全相同
    X, y = create_sliding_window(ser_s, window_size, pred_len)
    print(f"生成样本数: {len(X)}")
    
    # 2. 装载模型
    print(f"正在加载模型: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保已安装相应的库 (xgboost 或 lightgbm)")
        return None
    
    # 3. 进行预测
    print("正在进行预测...")
    predictions = model.predict(X)
    
    # 反归一化 - 与训练脚本完全相同的处理方式
    pred_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, pred_len)
    true_inv = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1, pred_len)
    
    # 只取第一步预测结果（与训练脚本的可视化方式一致）
    def first_step(arr):
        return arr[:, 0]
    
    pred_1 = first_step(pred_inv)
    true_1 = first_step(true_inv)
    
    # 模拟训练集/测试集划分（与训练脚本保持一致）
    split_idx = int(len(pred_1) * train_ratio)
    
    pred_tr_1 = pred_1[:split_idx]
    true_tr_1 = true_1[:split_idx]
    pred_te_1 = pred_1[split_idx:]
    true_te_1 = true_1[split_idx:]
    
    # 计算评估指标 - 与训练脚本完全相同
    train_mse, train_mae, train_rmse, train_mape = metrics(true_tr_1, pred_tr_1)
    test_mse, test_mae, test_rmse, test_mape = metrics(true_te_1, pred_te_1)
    
    print(f"\n=== 模型性能评估 ===")
    print(f"[训练集] RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, MAPE={train_mape:.2f}%")
    print(f"[测试集] RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    
    # 4. 绘制预测结果图 - 完全仿照训练脚本的处理方式
    print("正在生成预测结果图...")
    
    # 生成时间索引 - 与训练脚本完全相同的逻辑
    if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
        # 使用原始时间索引
        train_start_idx = window_size
        train_end_idx = window_size + len(pred_tr_1)
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + len(pred_te_1)
        
        if len(df.index) > test_end_idx:
            idx_tr = df.index[train_start_idx:train_end_idx]
            idx_te = df.index[test_start_idx:test_end_idx]
        else:
            # 如果索引长度不够，使用数值索引
            idx_tr = np.arange(window_size, window_size + len(pred_tr_1))
            idx_te = np.arange(window_size + len(pred_tr_1), 
                             window_size + len(pred_tr_1) + len(pred_te_1))
    else:
        # 使用数值索引
        idx_tr = np.arange(window_size, window_size + len(pred_tr_1))
        idx_te = np.arange(window_size + len(pred_tr_1), 
                         window_size + len(pred_tr_1) + len(pred_te_1))
    
    # 原始序列预测图 - 与训练脚本完全相同的绘图方式
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx_tr, true_tr_1, label="Train True", color='blue', alpha=0.7, lw=1)
    ax.plot(idx_tr, pred_tr_1, label="Train Pred", color='blue', alpha=0.7, lw=1, ls='--')
    ax.plot(idx_te, true_te_1, label="Test True", color='red', alpha=0.7, lw=1)
    ax.plot(idx_te, pred_te_1, label="Test Pred", color='red', alpha=0.7, lw=1, ls='--')
    
    ax.set_title(f"{well_name} - {model_type} 时序预测\nTest RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    ax.set_ylabel("地下水位 (GWL)")
    ax.set_xlabel("日期" if hasattr(idx_tr, 'dtype') and 'datetime' in str(idx_tr.dtype) else "时间步")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 如果是时间索引，旋转x轴标签
    if hasattr(idx_tr, 'dtype') and 'datetime' in str(idx_tr.dtype):
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = "results/ml_inference"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f"{output_dir}/pred_vs_true_{model_type.lower()}_{well_name}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"预测结果图已保存: {plot_path}")
    
    # 5. 保存预测结果到Excel
    print("正在保存预测结果...")
    
    # 合并训练集和测试集的结果
    all_true = np.concatenate([true_tr_1, true_te_1])
    all_pred = np.concatenate([pred_tr_1, pred_te_1])
    all_index = np.concatenate([idx_tr, idx_te])
    
    results_df = pd.DataFrame({
        'True_Values': all_true,
        'Predictions': all_pred,
        'Error': all_true - all_pred,
        'Absolute_Error': np.abs(all_true - all_pred)
    })
    
    # 如果有时间索引，添加时间列
    if hasattr(all_index, 'dtype') and 'datetime' in str(all_index.dtype):
        results_df.insert(0, 'Date', all_index)
    else:
        results_df.insert(0, 'Time_Step', all_index)
    
    excel_path = f"{output_dir}/prediction_results_{model_type.lower()}_{well_name}.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 预测结果工作表
        results_df.to_excel(writer, sheet_name='Prediction_Results', index=False)
        
        # 评估指标工作表 - 与训练脚本格式一致
        overall_mse, overall_mae, overall_rmse, overall_mape = metrics(all_true, all_pred)
        
        metrics_df = pd.DataFrame({
            'Dataset': ['Overall', 'Train', 'Test'],
            'MSE': [overall_mse, train_mse, test_mse],
            'MAE': [overall_mae, train_mae, test_mae],
            'RMSE': [overall_rmse, train_rmse, test_rmse],
            'MAPE(%)': [overall_mape, train_mape, test_mape]
        })
        metrics_df.to_excel(writer, sheet_name='Evaluation_Metrics', index=False)
        
        # 汇总表格 - 仿照训练脚本的格式
        summary_data = {
            '井位': well_name,
            '模型类型': model_type,
            '预测样本数': len(all_pred),
            '窗口大小': window_size,
            '预测步长': pred_len,
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
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"预测结果已保存: {excel_path}")
    
    print(f"\n=== 预测完成 ===")
    print(f"井位: {well_name}")
    print(f"模型: {model_type}")
    print(f"预测样本数: {len(all_pred)}")
    print(f"整体RMSE: {overall_rmse:.4f}, MAE: {overall_mae:.4f}, MAPE: {overall_mape:.2f}%")
    print(f"测试集RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%")
    print(f"结果保存目录: {output_dir}/")
    
    return all_pred, all_true, overall_rmse, overall_mae, overall_mape

if __name__ == '__main__':
    # 配置参数 - 与训练脚本保持一致
    DATA_PATH = "database/ZoupingCounty_gwl_filled.xlsx"
    WELL_COL = 4  # 井位列索引
    WINDOW_SIZE = 24  # 窗口大小
    PRED_LEN = 4  # 预测长度
    TRAIN_RATIO = 0.8  # 训练集比例
    
    # 检查可用的模型文件，优先使用LGBM
    possible_paths = [
        "results/single_well_lgbm/lgbm_井3.joblib",
        "results/single_well_xgb/xgb_井3.joblib",
        "results/single_well_lgbm/lgbm_井1.joblib",
        "results/single_well_xgb/xgb_井1.joblib",
        "results/single_well_lgbm/lgbm_井2.joblib",
        "results/single_well_xgb/xgb_井2.joblib"
    ]
    
    model_found = False
    MODEL_PATH = None
    
    for path in possible_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            model_found = True
            print(f"找到模型文件: {MODEL_PATH}")
            break
    
    if not model_found:
        print("错误: 未找到任何训练好的模型文件")
        print("请先运行以下命令之一来训练模型:")
        print("  python train_ml_single_well.py --model xgb --well_col 4")
        print("  python train_ml_single_well.py --model lgbm --well_col 4")
        print("\n可能的模型文件路径:")
        for path in possible_paths:
            print(f"  {path}")
    else:
        # 执行预测
        predict_with_model(
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
            well_col=WELL_COL,
            window_size=WINDOW_SIZE,
            pred_len=PRED_LEN,
            train_ratio=TRAIN_RATIO
        )