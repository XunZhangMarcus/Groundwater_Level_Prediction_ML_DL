#!/usr/bin/env python
# predict_ml_multi_wells.py - 多井位机器学习模型预测脚本

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


def create_sliding_window(series, window, pred_len=1):
    """创建滑动窗口数据"""
    X, y = [], []
    for i in range(len(series) - window - pred_len + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + pred_len])
    return np.asarray(X), np.asarray(y)


def metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + 1e-8))) * 100
    return mse, mae, rmse, mape


def predict_single_well(data, model_path, well_name, window_size=24, pred_len=4):
    """单井位预测函数"""
    print(f"正在预测井位: {well_name}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    
    # 加载模型
    model = joblib.load(model_path)
    
    # 数据预处理
    if "日期" in data.index.names or hasattr(data.index, 'name'):
        ser = data.values.astype("float32").reshape(-1, 1)
        original_index = data.index
    else:
        ser = data.astype("float32").reshape(-1, 1)
        original_index = None
    
    # 检查数据有效性
    if np.isnan(ser).all() or len(ser[~np.isnan(ser)]) < window_size + pred_len + 10:
        print(f"井位 {well_name} 数据不足或全为空值")
        return None
    
    # 归一化
    scaler = MinMaxScaler()
    ser_s = scaler.fit_transform(ser).flatten()
    
    # 创建滑动窗口
    X, y = create_sliding_window(ser_s, window_size, pred_len)
    
    # 预测
    pred = model.predict(X)
    
    # 反归一化
    pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1, pred_len)
    true_inv = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1, pred_len)
    
    # 只取第一步预测结果用于可视化
    pred_1step = pred_inv[:, 0]
    true_1step = true_inv[:, 0]
    
    # 生成时间索引
    if original_index is not None:
        start_idx = window_size
        end_idx = start_idx + len(pred_1step)
        if len(original_index) > end_idx:
            pred_index = original_index[start_idx:end_idx]
        else:
            pred_index = np.arange(start_idx, end_idx)
    else:
        pred_index = np.arange(window_size, window_size + len(pred_1step))
    
    # 计算评估指标
    mse, mae, rmse, mape = metrics(true_1step, pred_1step)
    
    return {
        'well_name': well_name,
        'predictions': pred_1step,
        'true_values': true_1step,
        'index': pred_index,
        'metrics': {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape},
        'original_data': ser.flatten(),
        'original_index': original_index
    }


def plot_predictions(results, output_dir):
    """绘制预测结果图"""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_type in ['xgb', 'lgbm']:
        model_results = [r for r in results if r and model_type in r['well_name'].lower()]
        
        if not model_results:
            continue
        
        # 为每个井位单独绘图
        for result in model_results:
            well_name = result['well_name']
            predictions = result['predictions']
            true_values = result['true_values']
            index = result['index']
            metrics_dict = result['metrics']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制真实值和预测值
            ax.plot(index, true_values, label='真实值', color='blue', linewidth=2)
            ax.plot(index, predictions, label='预测值', color='red', linewidth=2, linestyle='--')
            
            # 设置标题和标签
            ax.set_title(f'{well_name} - {model_type.upper()} 预测结果\n'
                        f'RMSE={metrics_dict["rmse"]:.4f}, MAE={metrics_dict["mae"]:.4f}, MAPE={metrics_dict["mape"]:.2f}%',
                        fontsize=14)
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel('地下水位 (GWL)', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 如果是时间索引，旋转x轴标签
            if hasattr(index, 'dtype') and 'datetime' in str(index.dtype):
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = f"{output_dir}/prediction_{model_type}_{well_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"预测图已保存: {plot_path}")


def save_results_to_excel(results, output_path):
    """将预测结果保存到Excel文件"""
    all_data = []
    
    for result in results:
        if result is None:
            continue
            
        well_name = result['well_name']
        predictions = result['predictions']
        true_values = result['true_values']
        index = result['index']
        
        # 创建数据框
        for i, (idx, pred, true) in enumerate(zip(index, predictions, true_values)):
            all_data.append({
                '井位': well_name,
                '时间': idx,
                '真实值': true,
                '预测值': pred,
                '误差': abs(pred - true)
            })
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_excel(output_path, index=False)
        print(f"预测结果已保存到Excel: {output_path}")
        return df
    else:
        print("没有有效的预测结果可保存")
        return None


def main():
    # 配置参数
    data_path = "database/ZoupingCounty_gwl_filled.xlsx"
    model_dir = "results/multi_wells_ml"
    output_dir = "results/ml_inference"
    window_size = 24
    pred_len = 4
    
    print("=== 多井位机器学习模型预测 ===")
    print(f"数据文件: {data_path}")
    print(f"模型目录: {model_dir}")
    print(f"输出目录: {output_dir}")
    
    # 1. 导入数据
    print("\n1. 导入数据...")
    df = pd.read_excel(data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    print(f"数据形状: {df.shape}")
    print(f"井位列: {list(df.columns)}")
    
    # 2. 装载模型并进行预测
    print("\n2. 装载模型并进行预测...")
    results = []
    
    # 获取所有模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    print(f"找到模型文件: {model_files}")
    
    for model_file in model_files:
        # 解析模型文件名 (格式: model_type_well_name.joblib)
        parts = model_file.replace('.joblib', '').split('_', 1)
        if len(parts) != 2:
            continue
            
        model_type, well_name = parts
        model_path = os.path.join(model_dir, model_file)
        
        # 检查对应的井位数据是否存在
        if well_name not in df.columns:
            print(f"警告: 数据中未找到井位 {well_name}")
            continue
        
        # 进行预测
        result = predict_single_well(
            df[well_name], model_path, f"{model_type}_{well_name}", 
            window_size, pred_len
        )
        
        if result:
            results.append(result)
            metrics_dict = result['metrics']
            print(f"✅ {model_type}_{well_name}: RMSE={metrics_dict['rmse']:.4f}, "
                  f"MAE={metrics_dict['mae']:.4f}, MAPE={metrics_dict['mape']:.2f}%")
        else:
            print(f"❌ {model_type}_{well_name}: 预测失败")
    
    if not results:
        print("没有成功的预测结果")
        return
    
    # 3. 绘制预测结果图
    print("\n3. 绘制预测结果图...")
    plot_predictions(results, output_dir)
    
    # 4. 保存预测结果到Excel
    print("\n4. 保存预测结果...")
    excel_path = f"{output_dir}/prediction_results.xlsx"
    df_results = save_results_to_excel(results, excel_path)
    
    # 5. 生成汇总统计
    print("\n5. 预测结果汇总:")
    print("-" * 60)
    print(f"{'模型':<15} {'井位':<10} {'RMSE':<10} {'MAE':<10} {'MAPE(%)':<10}")
    print("-" * 60)
    
    for result in results:
        well_name = result['well_name']
        metrics_dict = result['metrics']
        print(f"{well_name:<15} {'':<10} {metrics_dict['rmse']:<10.4f} "
              f"{metrics_dict['mae']:<10.4f} {metrics_dict['mape']:<10.2f}")
    
    print("-" * 60)
    
    # 计算平均指标
    avg_rmse = np.mean([r['metrics']['rmse'] for r in results if not np.isnan(r['metrics']['rmse'])])
    avg_mae = np.mean([r['metrics']['mae'] for r in results if not np.isnan(r['metrics']['mae'])])
    avg_mape = np.mean([r['metrics']['mape'] for r in results if not np.isnan(r['metrics']['mape'])])
    
    print(f"{'平均值':<15} {'':<10} {avg_rmse:<10.4f} {avg_mae:<10.4f} {avg_mape:<10.2f}")
    print("\n预测完成！")


if __name__ == "__main__":
    main()