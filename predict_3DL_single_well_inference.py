# predict_3DL_inference.py - 使用训练好的3DL模型进行预测
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.model import GeneratorGRU, GeneratorLSTM, GeneratorTransformer

def create_sliding_window(series, window, pred_len=1):
    """创建滑动窗口数据"""
    X, y = [], []
    for i in range(len(series) - window - pred_len + 1):
        X.append(series[i:i+window])
        y.append(series[i+window:i+window+pred_len])
    return np.asarray(X), np.asarray(y)

def get_model(name, window_size, pred_len, device):
    """获取模型实例"""
    if name == 'gru':
        return GeneratorGRU(input_size=window_size, out_size=pred_len).to(device)
    elif name == 'lstm':
        return GeneratorLSTM(input_size=window_size, out_size=pred_len).to(device)
    elif name == 'transformer':
        return GeneratorTransformer(input_dim=window_size, output_len=pred_len).to(device)
    else:
        raise ValueError(f'Unknown model {name}')

def metrics(y_true, y_pred):
    """计算评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return mse, mae, rmse, mape

def predict_with_model(model_path, data_path, well_col=4, window_size=24, pred_len=4, model_type='transformer'):
    """
    使用训练好的模型进行预测
    
    Args:
        model_path: 模型权重文件路径
        data_path: 数据文件路径
        well_col: 井位列索引
        window_size: 窗口大小
        pred_len: 预测长度
        model_type: 模型类型 ('gru', 'lstm', 'transformer')
    """
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 导入数据
    print("正在加载数据...")
    df = pd.read_excel(data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    # 获取井位名称和数据
    well_name = df.columns[well_col - 1]
    ser = df.iloc[:, well_col - 1].values.astype('float32').reshape(-1, 1)
    
    print(f"井位名称: {well_name}")
    print(f"数据长度: {len(ser)}, 有效数据: {(~np.isnan(ser)).sum()}")
    
    # 数据归一化
    scaler = MinMaxScaler()
    ser_s = scaler.fit_transform(ser).flatten()
    
    # 创建滑动窗口
    X, y = create_sliding_window(ser_s, window_size, pred_len)
    print(f"生成样本数: {len(X)}")
    
    # 2. 装载模型
    print(f"正在加载模型: {model_path}")
    model = get_model(model_type, window_size, pred_len, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. 进行预测
    print("正在进行预测...")
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for i in range(len(X)):
            # 准备输入数据
            x_input = torch.tensor(X[i:i+1], dtype=torch.float32).unsqueeze(1).to(device)  # (1,1,window_size)
            
            # 预测
            pred = model(x_input).cpu().numpy()  # (1, pred_len)
            predictions.append(pred[0, 0])  # 只取第一个预测值
            true_values.append(y[i, 0])  # 对应的真实值
    
    # 反归一化
    predictions = np.array(predictions).reshape(-1, 1)
    true_values = np.array(true_values).reshape(-1, 1)
    
    predictions_inv = scaler.inverse_transform(predictions).flatten()
    true_values_inv = scaler.inverse_transform(true_values).flatten()
    
    # 计算评估指标
    mse, mae, rmse, mape = metrics(true_values_inv, predictions_inv)
    print(f"\n=== 预测结果评估 ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 4. 绘制预测结果图
    print("正在生成预测结果图...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 生成时间索引
    if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
        start_idx = window_size
        end_idx = start_idx + len(predictions_inv)
        if len(df.index) > end_idx:
            time_index = df.index[start_idx:end_idx]
        else:
            time_index = np.arange(start_idx, end_idx)
    else:
        time_index = np.arange(window_size, window_size + len(predictions_inv))
    
    # 绘制折线图
    plt.figure(figsize=(15, 6))
    plt.plot(time_index, true_values_inv, label='True Values', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(time_index, predictions_inv, label='Predictions', color='red', alpha=0.8, linewidth=1.5, linestyle='--')
    
    plt.title(f'{well_name} - {model_type.upper()} Model Prediction Results\nRMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%', 
              fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Groundwater Level (GWL)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 如果是时间索引，旋转x轴标签
    if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = f"results/single_well_inference_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f"{output_dir}/prediction_{model_type}_{well_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"预测结果图已保存: {plot_path}")
    
    # 5. 保存预测结果到Excel
    print("正在保存预测结果...")
    results_df = pd.DataFrame({
        '真实值': true_values_inv,
        '预测值': predictions_inv,
        '误差': true_values_inv - predictions_inv,
        '绝对误差': np.abs(true_values_inv - predictions_inv)
    })
    
    # 如果有时间索引，添加时间列
    if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
        results_df.insert(0, '时间', time_index)
    else:
        results_df.insert(0, '时间步', time_index)
    
    excel_path = f"{output_dir}/single_well_inference_{model_type}_{well_name}.xlsx"
    results_df.to_excel(excel_path, index=False)
    print(f"预测结果已保存: {excel_path}")
    
    # 保存评估指标
    metrics_df = pd.DataFrame({
        '指标': ['MSE', 'MAE', 'RMSE', 'MAPE(%)'],
        '数值': [mse, mae, rmse, mape]
    })
    
    with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='评估指标', index=False)
    
    print(f"\n=== 预测完成 ===")
    print(f"井位: {well_name}")
    print(f"模型: {model_type.upper()}")
    print(f"预测样本数: {len(predictions_inv)}")
    print(f"结果保存目录: {output_dir}/")
    
    return predictions_inv, true_values_inv, rmse, mae, mape

if __name__ == '__main__':
    # 配置参数
    MODEL_PATH = "results/single_well_transformer/best_transformer_井3.pt"  # 模型权重文件路径
    DATA_PATH = "database/ZoupingCounty_gwl_filled.xlsx"  # 数据文件路径
    WELL_COL = 4  # 井位列索引 (井3对应第4列)
    WINDOW_SIZE = 24  # 窗口大小
    PRED_LEN = 4  # 预测长度
    MODEL_TYPE = 'transformer'  # 模型类型
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        print("请先运行训练脚本生成模型文件，或修改MODEL_PATH指向正确的模型文件")
    else:
        # 执行预测
        predict_with_model(
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
            well_col=WELL_COL,
            window_size=WINDOW_SIZE,
            pred_len=PRED_LEN,
            model_type=MODEL_TYPE
        )