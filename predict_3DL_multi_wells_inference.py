# predict_3DL_multi_wells_inference.py - 使用训练好的多井3DL模型进行预测
import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.model import GeneratorGRU, GeneratorLSTM, GeneratorTransformer

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    # 过滤掉NaN值
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

def predict_single_well(model_path, well_data, well_name, window_size=24, pred_len=4, model_type='transformer', device='cpu'):
    """
    单井位预测函数
    
    Args:
        model_path: 模型权重文件路径
        well_data: 井位数据 (pandas Series)
        well_name: 井位名称
        window_size: 窗口大小
        pred_len: 预测长度
        model_type: 模型类型
        device: 计算设备
    
    Returns:
        dict: 包含预测结果和评估指标的字典
    """
    result = {
        'well_name': well_name,
        'model_type': model_type.upper(),
        'success': False,
        'error_msg': None,
        'predictions': None,
        'true_values': None,
        'time_index': None,
        'metrics': {}
    }
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            result['error_msg'] = f"模型文件不存在: {model_path}"
            return result
        
        # 数据预处理
        if hasattr(well_data.index, 'dtype') and 'datetime' in str(well_data.index.dtype):
            ser = well_data.values.astype('float32').reshape(-1, 1)
            original_index = well_data.index
        else:
            ser = well_data.values.astype('float32').reshape(-1, 1)
            original_index = None
        
        # 检查数据有效性
        if np.isnan(ser).all() or len(ser[~np.isnan(ser)]) < window_size + pred_len + 10:
            result['error_msg'] = f"井位 {well_name} 数据不足或全为空值"
            return result
        
        # 数据归一化
        scaler = MinMaxScaler()
        ser_s = scaler.fit_transform(ser).flatten()
        
        # 创建滑动窗口
        X, y = create_sliding_window(ser_s, window_size, pred_len)
        
        # 加载模型
        model = get_model(model_type, window_size, pred_len, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # 进行预测
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
        
        # 生成时间索引
        if original_index is not None:
            start_idx = window_size
            end_idx = start_idx + len(predictions_inv)
            if len(original_index) > end_idx:
                time_index = original_index[start_idx:end_idx]
            else:
                time_index = np.arange(start_idx, end_idx)
        else:
            time_index = np.arange(window_size, window_size + len(predictions_inv))
        
        # 计算评估指标
        mse, mae, rmse, mape = metrics(true_values_inv, predictions_inv)
        
        # 保存结果
        result['predictions'] = predictions_inv
        result['true_values'] = true_values_inv
        result['time_index'] = time_index
        result['metrics'] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
        result['success'] = True
        
    except Exception as e:
        result['error_msg'] = f"井位 {well_name} 预测过程出错: {str(e)}"
    
    return result

def predict_multi_wells(models_dir, data_path, window_size=24, pred_len=4, model_types=['transformer']):
    """
    多井位预测主函数
    
    Args:
        models_dir: 模型文件目录
        data_path: 数据文件路径
        window_size: 窗口大小
        pred_len: 预测长度
        model_types: 模型类型列表
    """
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 导入数据
    print("正在加载数据...")
    df = pd.read_excel(data_path)
    if "时间戳" in df.columns:
        df["时间戳"] = pd.to_datetime(df["时间戳"])
        df = df.set_index("时间戳")
    elif "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    print(f"数据形状: {df.shape}")
    print(f"井位列: {df.columns.tolist()}")
    
    # 2. 查找可用的模型文件
    print("正在查找模型文件...")
    available_models = {}
    
    for model_type in model_types:
        model_files = glob.glob(os.path.join(models_dir, f"{model_type}_*.pt"))
        for model_file in model_files:
            # 从文件名提取井位名称
            filename = os.path.basename(model_file)
            well_name = filename.replace(f"{model_type}_", "").replace(".pt", "")
            
            if well_name not in available_models:
                available_models[well_name] = {}
            available_models[well_name][model_type] = model_file
    
    print(f"找到模型文件: {len(available_models)} 个井位")
    for well_name, models in available_models.items():
        print(f"  {well_name}: {list(models.keys())}")
    
    # 3. 进行预测
    print("\\n开始预测...")
    results = {}
    
    for well_name, models in available_models.items():
        if well_name not in df.columns:
            print(f"警告: 数据中未找到井位 {well_name}")
            continue
        
        results[well_name] = {}
        well_data = df[well_name]
        
        for model_type, model_path in models.items():
            print(f"正在预测: {well_name} - {model_type.upper()}")
            
            result = predict_single_well(
                model_path=model_path,
                well_data=well_data,
                well_name=well_name,
                window_size=window_size,
                pred_len=pred_len,
                model_type=model_type,
                device=device
            )
            
            results[well_name][model_type] = result
            
            if result['success']:
                metrics_info = result['metrics']
                print(f"  ✅ {well_name} - {model_type.upper()}: "
                      f"RMSE={metrics_info['rmse']:.4f}, "
                      f"MAE={metrics_info['mae']:.4f}, "
                      f"MAPE={metrics_info['mape']:.2f}%")
            else:
                print(f"  ❌ {well_name} - {model_type.upper()}: {result['error_msg']}")
    
    # 4. 生成预测结果图
    print("\\n正在生成预测结果图...")
    output_dir = "results/multi_wells_inference"
    os.makedirs(output_dir, exist_ok=True)
    
    successful_results = []
    for well_name, models in results.items():
        for model_type, result in models.items():
            if result['success']:
                successful_results.append((well_name, model_type, result))
    
    # 为每个成功的预测生成单独的图表
    for well_name, model_type, result in successful_results:
        # 预测结果图
        plt.figure(figsize=(15, 6))
        plt.plot(result['time_index'], result['true_values'], 
                label='True Values', color='blue', alpha=0.8, linewidth=1.5)
        plt.plot(result['time_index'], result['predictions'], 
                label='Predictions', color='red', alpha=0.8, linewidth=1.5, linestyle='--')
        
        metrics_info = result['metrics']
        plt.title(f'{well_name} - {model_type.upper()} Model Prediction Results\\n'
                 f'RMSE={metrics_info["rmse"]:.4f}, MAE={metrics_info["mae"]:.4f}, MAPE={metrics_info["mape"]:.2f}%', 
                 fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Groundwater Level (GWL)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 如果是时间索引，旋转x轴标签
        if hasattr(result['time_index'], 'dtype') and 'datetime' in str(result['time_index'].dtype):
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = f"{output_dir}/prediction_{model_type}_{well_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  预测结果图已保存: {plot_path}")
    
    # 5. 保存预测结果到Excel
    print("\\n正在保存预测结果...")
    
    # 为每个成功的预测保存Excel文件
    for well_name, model_type, result in successful_results:
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'True_Values': result['true_values'],
            'Predictions': result['predictions'],
            'Error': result['true_values'] - result['predictions'],
            'Absolute_Error': np.abs(result['true_values'] - result['predictions'])
        })
        
        # 如果有时间索引，添加时间列
        if hasattr(result['time_index'], 'dtype') and 'datetime' in str(result['time_index'].dtype):
            results_df.insert(0, 'Time', result['time_index'])
        else:
            results_df.insert(0, 'Time_Step', result['time_index'])
        
        # 保存到Excel
        excel_path = f"{output_dir}/prediction_results_{model_type}_{well_name}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 预测结果工作表
            results_df.to_excel(writer, sheet_name='Prediction_Results', index=False)
            
            # 评估指标工作表
            metrics_df = pd.DataFrame({
                'Metric': ['MSE', 'MAE', 'RMSE', 'MAPE(%)'],
                'Value': [result['metrics']['mse'], result['metrics']['mae'], 
                         result['metrics']['rmse'], result['metrics']['mape']]
            })
            metrics_df.to_excel(writer, sheet_name='Evaluation_Metrics', index=False)
        
        print(f"  预测结果已保存: {excel_path}")
    
    # 6. 生成汇总报告
    print("\\n正在生成汇总报告...")
    summary_data = []
    
    for well_name, models in results.items():
        for model_type, result in models.items():
            if result['success']:
                summary_data.append({
                    'Well_Name': well_name,
                    'Model_Type': result['model_type'],
                    'Sample_Count': len(result['predictions']),
                    'MSE': f"{result['metrics']['mse']:.6f}",
                    'MAE': f"{result['metrics']['mae']:.4f}",
                    'RMSE': f"{result['metrics']['rmse']:.4f}",
                    'MAPE(%)': f"{result['metrics']['mape']:.2f}",
                    'Status': 'Success'
                })
            else:
                summary_data.append({
                    'Well_Name': well_name,
                    'Model_Type': result['model_type'],
                    'Sample_Count': 0,
                    'MSE': 'FAILED',
                    'MAE': 'FAILED',
                    'RMSE': 'FAILED',
                    'MAPE(%)': 'FAILED',
                    'Status': result['error_msg']
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{output_dir}/multi_wells_prediction_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)
    print(f"汇总报告已保存: {summary_path}")
    
    # 7. 打印最终统计
    successful_count = len(successful_results)
    total_count = sum(len(models) for models in results.values())
    
    print(f"\\n=== 预测完成 ===")
    print(f"成功预测: {successful_count}/{total_count}")
    print(f"结果保存目录: {output_dir}/")
    
    if successful_count > 0:
        # 按模型类型统计
        for model_type in model_types:
            model_results = [result for _, mt, result in successful_results if mt == model_type]
            
            if len(model_results) > 0:
                avg_rmse = np.mean([res['metrics']['rmse'] for res in model_results])
                avg_mae = np.mean([res['metrics']['mae'] for res in model_results])
                avg_mape = np.mean([res['metrics']['mape'] for res in model_results])
                
                print(f"\\n=== {model_type.upper()} 模型统计 ===")
                print(f"成功预测井位数: {len(model_results)}")
                print(f"平均RMSE: {avg_rmse:.4f}")
                print(f"平均MAE: {avg_mae:.4f}")
                print(f"平均MAPE: {avg_mape:.2f}%")
    
    return results

if __name__ == '__main__':
    # 配置参数
    MODELS_DIR = "results/multi_wells_transformer"  # 模型文件目录
    DATA_PATH = "database/ZoupingCounty_gwl_filled.xlsx"  # 数据文件路径
    WINDOW_SIZE = 24  # 窗口大小
    PRED_LEN = 4  # 预测长度
    MODEL_TYPES = ['transformer']  # 要使用的模型类型，可以是 ['gru', 'lstm', 'transformer'] 或其子集
    
    # 检查模型目录是否存在
    if not os.path.exists(MODELS_DIR):
        print(f"错误: 模型目录不存在 {MODELS_DIR}")
        print("请先运行多井训练脚本生成模型文件")
    else:
        # 执行多井预测
        results = predict_multi_wells(
            models_dir=MODELS_DIR,
            data_path=DATA_PATH,
            window_size=WINDOW_SIZE,
            pred_len=PRED_LEN,
            model_types=MODEL_TYPES
        )