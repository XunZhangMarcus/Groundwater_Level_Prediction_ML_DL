#!/usr/bin/env python
# predict_arima_single_well.py - 单井位ARIMA模型预测脚本

import argparse
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


def main(cfg):
    print("=== 单井位ARIMA模型预测 ===")
    print(f"数据文件: {cfg.data_path}")
    print(f"井位列序号: {cfg.well_col}")
    print(f"模型目录: {cfg.model_dir}")
    
    # 1. 导入数据
    print("\n1. 导入数据...")
    df = pd.read_excel(cfg.data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    # 获取井位名称和数据
    well_name = df.columns[cfg.well_col - 1]
    ser_raw = df.iloc[:, cfg.well_col - 1].astype("float32")
    
    print(f"井位名称: {well_name}")
    print(f"数据长度: {len(ser_raw)}, 有效数据: {ser_raw.notna().sum()}")
    
    # 检查数据有效性
    if ser_raw.isna().all() or len(ser_raw.dropna()) < 20:
        print(f"❌ 井位 {well_name} 数据不足或全为空值")
        return
    
    # 2. 装载模型
    print("\n2. 装载模型...")
    model_path = f"{cfg.model_dir}/arima_model_{well_name}.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return
    
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ 模型加载成功: ARIMA{model.model.order}")
    
    # 3. 进行预测
    print("\n3. 进行预测...")
    
    # 划分训练/测试集（与训练时保持一致）
    train_size = int(len(ser_raw) * cfg.train_ratio)
    train_raw = ser_raw.iloc[:train_size]
    test_raw = ser_raw.iloc[train_size:]
    
    print(f"数据划分: 训练集 {len(train_raw)} 样本, 测试集 {len(test_raw)} 样本")
    
    # 训练集预测（拟合值）
    train_pred = model.fittedvalues
    
    # 测试集预测
    forecast_raw = model.get_forecast(steps=len(test_raw)).predicted_mean
    forecast_raw.index = test_raw.index
    
    # 计算评估指标
    train_rmse, train_mae, train_mape = metrics(train_raw.values, train_pred.values)
    test_rmse, test_mae, test_mape = metrics(test_raw.values, forecast_raw.values)
    
    print(f"=== 预测性能评估 ===")
    print(f"[训练集] RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, MAPE={train_mape:.2f}%")
    print(f"[测试集] RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    
    # 4. 绘制折线图
    print("\n4. 绘制预测结果图...")
    
    # 创建输出目录
    output_dir = f"results/single_well_arima_inference"
    os.makedirs(output_dir, exist_ok=True)
    
    # 仿照训练脚本的可视化方式
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_raw.index, train_raw.values, label="Train True", color='blue', alpha=0.7, lw=1)
    ax.plot(train_pred.index, train_pred.values, label="Train Pred", color='blue', alpha=0.7, lw=1, ls='--')
    ax.plot(test_raw.index, test_raw.values, label="Test True", color='red', alpha=0.7, lw=1)
    ax.plot(forecast_raw.index, forecast_raw.values, label="Test Pred", color='red', alpha=0.7, lw=1, ls='--')
    
    ax.set_title(f"{well_name} - ARIMA{model.model.order} 时序预测\nTest RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    ax.set_ylabel("地下水位 (GWL)")
    ax.set_xlabel("日期")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存图片
    plot_path = f"{output_dir}/prediction_{well_name}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ 预测图已保存: {plot_path}")
    
    # 5. 保存预测结果到Excel
    print("\n5. 保存预测结果到Excel...")
    
    # 准备数据
    results_data = []
    
    # 训练集数据
    for i, (idx, true_val, pred_val) in enumerate(zip(train_raw.index, train_raw.values, train_pred.values)):
        results_data.append({
            '时间': idx,
            '数据类型': '训练集',
            '真实值': true_val,
            '预测值': pred_val,
            '误差': abs(pred_val - true_val)
        })
    
    # 测试集数据
    for i, (idx, true_val, pred_val) in enumerate(zip(test_raw.index, test_raw.values, forecast_raw.values)):
        results_data.append({
            '时间': idx,
            '数据类型': '测试集',
            '真实值': true_val,
            '预测值': pred_val,
            '误差': abs(pred_val - true_val)
        })
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(results_data)
    excel_path = f"{output_dir}/prediction_results_{well_name}.xlsx"
    results_df.to_excel(excel_path, index=False)
    
    print(f"✅ 预测结果已保存到Excel: {excel_path}")
    print(f"   - 数据行数: {len(results_df)}")
    print(f"   - 列名: {list(results_df.columns)}")
    
    # 6. 最终总结
    print(f"\n=== 单井位ARIMA预测完成 ===")
    print(f"井位名称: {well_name}")
    print(f"模型阶次: ARIMA{model.model.order}")
    print(f"测试集性能: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    print(f"结果保存目录: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单井位ARIMA模型预测")
    parser.add_argument("--data_path", type=str,
                        default="database/ZoupingCounty_gwl_filled.xlsx",
                        help="数据文件路径")
    parser.add_argument("--well_col", type=int, default=2,
                        help="井位列序号（从 1 开始，跳过日期列）")
    parser.add_argument("--model_dir", type=str,
                        default="results/single_well_arima",
                        help="模型文件目录")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例（需与训练时保持一致）")
    
    cfg = parser.parse_args()
    main(cfg)