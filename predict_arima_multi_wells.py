#!/usr/bin/env python
# predict_arima_multi_wells.py - 多井位ARIMA模型预测脚本

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


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


def predict_single_well(data, model_path, well_name, train_ratio=0.8):
    """单井位ARIMA预测函数"""
    print(f"正在预测井位: {well_name}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 数据预处理
    ser_raw = data.astype("float32")
    
    # 检查数据有效性
    if ser_raw.isna().all() or len(ser_raw.dropna()) < 20:
        print(f"井位 {well_name} 数据不足或全为空值")
        return None
    
    # 划分训练/测试集（与训练时保持一致）
    train_size = int(len(ser_raw) * train_ratio)
    train_raw = ser_raw.iloc[:train_size]
    test_raw = ser_raw.iloc[train_size:]
    
    # 训练集预测（拟合值）
    train_pred = model.fittedvalues
    
    # 测试集预测
    forecast_raw = model.get_forecast(steps=len(test_raw)).predicted_mean
    forecast_raw.index = test_raw.index
    
    # 计算评估指标
    train_mse, train_mae, train_rmse, train_mape = metrics(train_raw.values, train_pred.values)
    test_mse, test_mae, test_rmse, test_mape = metrics(test_raw.values, forecast_raw.values)
    
    return {
        'well_name': well_name,
        'train_true': train_raw,
        'train_pred': train_pred,
        'test_true': test_raw,
        'test_pred': forecast_raw,
        'train_metrics': {'mse': train_mse, 'mae': train_mae, 'rmse': train_rmse, 'mape': train_mape},
        'test_metrics': {'mse': test_mse, 'mae': test_mae, 'rmse': test_rmse, 'mape': test_mape},
        'model_order': model.model.order,
        'original_data': ser_raw
    }


def plot_predictions(results, output_dir):
    """绘制预测结果图"""
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        if result is None:
            continue
            
        well_name = result['well_name']
        train_true = result['train_true']
        train_pred = result['train_pred']
        test_true = result['test_true']
        test_pred = result['test_pred']
        test_metrics = result['test_metrics']
        model_order = result['model_order']
        
        # 创建预测图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制训练集
        ax.plot(train_true.index, train_true.values, label='训练集真实值', color='blue', linewidth=2)
        ax.plot(train_pred.index, train_pred.values, label='训练集预测值', color='blue', 
                linewidth=2, linestyle='--', alpha=0.8)
        
        # 绘制测试集
        ax.plot(test_true.index, test_true.values, label='测试集真实值', color='red', linewidth=2)
        ax.plot(test_pred.index, test_pred.values, label='测试集预测值', color='red', 
                linewidth=2, linestyle='--', alpha=0.8)
        
        # 设置标题和标签
        ax.set_title(f'{well_name} - ARIMA{model_order} 预测结果\n'
                    f'Test RMSE={test_metrics["rmse"]:.4f}, MAE={test_metrics["mae"]:.4f}, MAPE={test_metrics["mape"]:.2f}%',
                    fontsize=14)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('地下水位 (GWL)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = f"{output_dir}/prediction_ARIMA_{well_name}.png"
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
        
        # 训练集数据
        train_true = result['train_true']
        train_pred = result['train_pred']
        for i, (idx, true_val, pred_val) in enumerate(zip(train_true.index, train_true.values, train_pred.values)):
            all_data.append({
                '井位': well_name,
                '时间': idx,
                '数据类型': '训练集',
                '真实值': true_val,
                '预测值': pred_val,
                '误差': abs(pred_val - true_val)
            })
        
        # 测试集数据
        test_true = result['test_true']
        test_pred = result['test_pred']
        for i, (idx, true_val, pred_val) in enumerate(zip(test_true.index, test_true.values, test_pred.values)):
            all_data.append({
                '井位': well_name,
                '时间': idx,
                '数据类型': '测试集',
                '真实值': true_val,
                '预测值': pred_val,
                '误差': abs(pred_val - true_val)
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
    model_dir_single = "results/single_well_arima"
    model_dir_multi = "results/multi_wells_arima"
    output_dir = "results/arima_inference"
    train_ratio = 0.8
    
    print("=== ARIMA模型预测 ===")
    print(f"数据文件: {data_path}")
    print(f"单井位模型目录: {model_dir_single}")
    print(f"多井位模型目录: {model_dir_multi}")
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
    
    # 检查两个模型目录
    model_dirs = []
    if os.path.exists(model_dir_single):
        model_dirs.append(model_dir_single)
    if os.path.exists(model_dir_multi):
        model_dirs.append(model_dir_multi)
    
    if not model_dirs:
        print("未找到ARIMA模型目录")
        return
    
    # 获取所有模型文件（避免重复）
    model_files = {}  # 使用字典避免重复井位
    for model_dir in model_dirs:
        files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'arima_model' in f]
        for f in files:
            well_name = f.replace('arima_model_', '').replace('.pkl', '')
            # 优先使用多井位模型目录的模型
            if well_name not in model_files or model_dir == model_dir_multi:
                model_files[well_name] = (model_dir, f)
    
    print(f"找到模型文件: {list(model_files.keys())}")
    
    for well_name, (model_dir, model_file) in model_files.items():
        # 解析模型文件名 (格式: arima_model_well_name.pkl)
        well_name = model_file.replace('arima_model_', '').replace('.pkl', '')
        model_path = os.path.join(model_dir, model_file)
        
        # 检查对应的井位数据是否存在
        if well_name not in df.columns:
            print(f"警告: 数据中未找到井位 {well_name}")
            continue
        
        # 进行预测
        result = predict_single_well(df[well_name], model_path, well_name, train_ratio)
        
        if result:
            results.append(result)
            test_metrics = result['test_metrics']
            print(f"✅ {well_name}: ARIMA{result['model_order']}, "
                  f"Test RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, MAPE={test_metrics['mape']:.2f}%")
        else:
            print(f"❌ {well_name}: 预测失败")
    
    if not results:
        print("没有成功的预测结果")
        return
    
    # 3. 绘制预测结果图
    print("\n3. 绘制预测结果图...")
    plot_predictions(results, output_dir)
    
    # 4. 保存预测结果到Excel
    print("\n4. 保存预测结果...")
    excel_path = f"{output_dir}/arima_prediction_results.xlsx"
    df_results = save_results_to_excel(results, excel_path)
    
    # 5. 生成汇总统计
    print("\n5. 预测结果汇总:")
    print("-" * 80)
    print(f"{'井位':<15} {'ARIMA阶次':<15} {'RMSE':<10} {'MAE':<10} {'MAPE(%)':<10}")
    print("-" * 80)
    
    for result in results:
        well_name = result['well_name']
        model_order = result['model_order']
        test_metrics = result['test_metrics']
        print(f"{well_name:<15} {str(model_order):<15} {test_metrics['rmse']:<10.4f} "
              f"{test_metrics['mae']:<10.4f} {test_metrics['mape']:<10.2f}")
    
    print("-" * 80)
    
    # 计算平均指标
    valid_results = [r for r in results if not np.isnan(r['test_metrics']['rmse'])]
    if valid_results:
        avg_rmse = np.mean([r['test_metrics']['rmse'] for r in valid_results])
        avg_mae = np.mean([r['test_metrics']['mae'] for r in valid_results])
        avg_mape = np.mean([r['test_metrics']['mape'] for r in valid_results])
        
        print(f"{'平均值':<15} {'':<15} {avg_rmse:<10.4f} {avg_mae:<10.4f} {avg_mape:<10.2f}")
    
    print("\nARIMA预测完成！")


if __name__ == "__main__":
    main()