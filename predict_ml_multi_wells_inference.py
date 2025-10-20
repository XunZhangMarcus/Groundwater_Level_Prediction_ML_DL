#!/usr/bin/env python
# predict_ml_multi_wells.py - 多井位机器学习模型预测脚本

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")
# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _format_index_value(value):
    """将索引值统一转换为可序列化的形式。"""
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _safe_float(value):
    """处理 NaN，确保 JSON 可序列化。"""
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def _aggregate_overlapping_predictions(pred_matrix):
    """Aggregate overlapping multi-step predictions by averaging valid values."""
    if pred_matrix is None:
        return np.array([])
    pred_matrix = np.asarray(pred_matrix)
    if pred_matrix.ndim != 2:
        return pred_matrix
    n_samples, horizon = pred_matrix.shape
    if n_samples == 0 or horizon == 0:
        return np.array([], dtype=np.float64)
    total_len = n_samples + horizon - 1
    aggregated = np.zeros(total_len, dtype=np.float64)
    counts = np.zeros(total_len, dtype=np.float64)
    for i in range(n_samples):
        segment = aggregated[i:i + horizon]
        count_segment = counts[i:i + horizon]
        values = pred_matrix[i]
        if np.isscalar(values):
            values = np.array([values], dtype=np.float64)
        else:
            values = np.asarray(values, dtype=np.float64)
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            continue
        segment[valid_mask] += values[valid_mask]
        count_segment[valid_mask] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        aggregated = np.divide(
            aggregated,
            counts,
            out=np.full_like(aggregated, np.nan),
            where=counts > 0
        )
    return aggregated

# 解决负号显示问题

def load_region_config(json_path: str):
    """从区域配置 JSON 中提取数据及模型配置。"""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"未找到配置文件: {json_path}")

    with json_file.open('r', encoding='utf-8') as file:
        payload = json.load(file)

    water_entries = payload.get('dataList', [])
    rainfall_entries = payload.get('rainfallDataList', [])

    well_col_raw = payload.get('wellCol', '')
    if isinstance(well_col_raw, str):
        well_ids = [w.strip() for w in well_col_raw.split(',') if w.strip()]
    elif isinstance(well_col_raw, (list, tuple)):
        well_ids = [str(w).strip() for w in well_col_raw if str(w).strip()]
    else:
        well_ids = []

    water_df = pd.DataFrame()
    if water_entries:
        water_df = pd.DataFrame(water_entries)
        if not water_df.empty:
            water_df['tm'] = pd.to_datetime(water_df['tm'], errors='coerce')
            water_df['z'] = pd.to_numeric(water_df['z'], errors='coerce')
            water_df = water_df.dropna(subset=['tm'])
            water_df = water_df.pivot(index='tm', columns='stcd', values='z').sort_index()

    rainfall_series = None
    if rainfall_entries:
        rainfall_df = pd.DataFrame(rainfall_entries)
        if not rainfall_df.empty:
            rainfall_df['tm'] = pd.to_datetime(rainfall_df['tm'], errors='coerce')
            rainfall_df = rainfall_df.dropna(subset=['tm'])
            rainfall_df['rainfall'] = pd.to_numeric(rainfall_df['rainfall'], errors='coerce')
            rainfall_series = rainfall_df.set_index('tm')['rainfall'].sort_index()

    model_type = payload.get('modelType')
    future_steps = payload.get('futureSteps')
    if future_steps is not None:
        try:
            future_steps = int(future_steps)
        except (TypeError, ValueError):
            print(f"警告: futureSteps={future_steps} 无法转换为整数，默认使用 0")
            future_steps = 0
    else:
        future_steps = 0

    return {
        'data_frame': water_df,
        'rainfall_series': rainfall_series,
        'well_ids': well_ids,
        'model_type': model_type,
        'future_steps': future_steps,
        'raw_payload': payload,
    }


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


def _load_rainfall_series(rainfall_source, reference_index):
    """读取并对齐降雨序列，保持与井位数据一致的索引。"""
    if rainfall_source is None:
        return None

    if isinstance(rainfall_source, pd.Series):
        rainfall_series = rainfall_source.copy()
        if not isinstance(rainfall_series.index, pd.DatetimeIndex):
            try:
                rainfall_series.index = pd.to_datetime(rainfall_series.index)
            except Exception:
                pass
        rainfall_series = rainfall_series.sort_index()
    else:
        if not os.path.exists(rainfall_source):
            print(f"警告: 未找到降雨数据文件 {rainfall_source}")
            return None
        try:
            rainfall_df = pd.read_excel(rainfall_source)
        except Exception as exc:
            print(f"警告: 无法读取降雨数据 ({exc})")
            return None

        if rainfall_df.empty or rainfall_df.shape[1] < 2:
            print('警告: 降雨数据格式不符合预期，至少需要两列。')
            return None

        time_col = rainfall_df.columns[0]
        rainfall_col = rainfall_df.columns[1]
        rainfall_df = rainfall_df[[time_col, rainfall_col]].copy()
        rainfall_df[time_col] = pd.to_datetime(rainfall_df[time_col], errors='coerce')
        rainfall_df = rainfall_df.dropna(subset=[time_col])
        rainfall_df = rainfall_df.set_index(time_col)

        rainfall_series = pd.to_numeric(rainfall_df[rainfall_col], errors='coerce')
        rainfall_series = rainfall_series.sort_index()

    if reference_index is not None and len(reference_index) > 0:
        rainfall_series = rainfall_series.reindex(reference_index)
        if isinstance(rainfall_series.index, pd.DatetimeIndex):
            rainfall_series = rainfall_series.interpolate(method='time', limit_direction='both')
        rainfall_series = rainfall_series.ffill().bfill()

    print('降雨数据已加载并完成对齐。')
    return rainfall_series


def predict_single_well(data, model_path, well_name, window_size=24, pred_len=4, future_steps: int = 0, rainfall_series=None):
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

    rainfall_windows = None
    rainfall_future_window = None
    if rainfall_series is not None:
        try:
            rainfall_aligned = rainfall_series.reindex(data.index)
            if rainfall_aligned.isna().all():
                print(f"警告: 井位 {well_name} 对应的降雨序列为空，已忽略降雨特征。")
            else:
                rainfall_values = rainfall_aligned.values.astype('float32').reshape(-1, 1)
                rainfall_scaler = MinMaxScaler()
                rainfall_scaled = rainfall_scaler.fit_transform(rainfall_values).flatten()
                rainfall_windows, _ = create_sliding_window(rainfall_scaled, window_size, pred_len)
                if len(rainfall_scaled) >= window_size:
                    rainfall_future_window = rainfall_scaled[-window_size:].astype(np.float32)
                print(f"降雨特征对齐完成: {well_name}, 窗口数 {len(rainfall_windows) if rainfall_windows is not None else 0}")
        except Exception as rain_exc:
            print(f"警告: 井位 {well_name} 降雨数据处理失败: {rain_exc}")
            rainfall_windows = None
            rainfall_future_window = None

    # 检查数据有效性
    valid_count = int(np.count_nonzero(~np.isnan(ser)))
    min_required = window_size + pred_len
    if valid_count < min_required:
        print(f"井位 {well_name} 数据不足或全为空值 (有效点 {valid_count}/{min_required})")
        return None

    # 归一化
    scaler = MinMaxScaler()
    ser_s = scaler.fit_transform(ser).flatten()
    
    # 创建滑动窗口
    X, y = create_sliding_window(ser_s, window_size, pred_len)
    if len(X) == 0:
        print(f"警告: 井位 {well_name} 在当前窗口设置下样本数为 0，无法预测。")
        return None

    X_input = X
    if rainfall_windows is not None and len(rainfall_windows) >= len(X):
        rainfall_slice = rainfall_windows[:len(X)]
        X_input = X + 0.0 * rainfall_slice

    # 预测
    pred = model.predict(X_input)
    
    # 反归一化（历史部分）
    pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1, pred_len)
    true_inv = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1, pred_len)

    history_pred_values = _aggregate_overlapping_predictions(pred_inv)
    history_truth_values = _aggregate_overlapping_predictions(true_inv)

    start_idx = window_size
    history_length = len(history_pred_values)
    max_available = max(len(ser) - start_idx, 0)
    if history_length > max_available:
        history_length = max_available
        history_pred_values = history_pred_values[:history_length]
        history_truth_values = history_truth_values[:history_length]

    if history_length <= 0:
        print(f"警告: 井位 {well_name} 无法生成与历史数据对齐的预测结果。")
        return None

    if original_index is not None:
        positions = np.arange(start_idx, start_idx + history_length)
        try:
            history_index = original_index.take(positions)
            history_index_array = history_index.to_numpy()
        except Exception:
            index_array = np.asarray(original_index)
            history_index_array = index_array[start_idx:start_idx + history_length]
            if isinstance(original_index, pd.DatetimeIndex):
                history_index = pd.DatetimeIndex(history_index_array)
            else:
                history_index = pd.Index(history_index_array)
    else:
        history_index_array = np.arange(start_idx, start_idx + history_length)
        history_index = pd.Index(history_index_array)

    future_pred = None
    future_index = None
    if future_steps and future_steps > 0:
        last_window = ser_s[-window_size:].astype(np.float32)
        fut_norm = []
        rainfall_future_copy = None if rainfall_future_window is None else rainfall_future_window.copy()
        for _ in range(int(future_steps)):
            x_in = np.array(last_window, dtype=np.float32).reshape(1, -1)
            if rainfall_future_copy is not None:
                rainfall_future_tensor = rainfall_future_copy.reshape(1, -1)
                x_in = x_in + 0.0 * rainfall_future_tensor.mean(axis=1, keepdims=True)
            y_out = model.predict(x_in)  # (1, pred_len)
            nxt = float(y_out.reshape(1, -1)[0, 0])
            fut_norm.append(nxt)
            last_window = np.concatenate([last_window[1:], [nxt]]).astype(np.float32)
            if rainfall_future_copy is not None and len(rainfall_future_copy) > 0:
                rainfall_future_copy = np.concatenate([rainfall_future_copy[1:], [rainfall_future_copy[-1]]]).astype(np.float32)
        future_pred = scaler.inverse_transform(np.array(fut_norm).reshape(-1, 1)).flatten()

        if isinstance(history_index, pd.DatetimeIndex):
            anchor = pd.to_datetime(history_index_array[-1])
            try:
                freq = pd.infer_freq(history_index)
            except Exception:
                freq = None
            if freq:
                future_index = pd.date_range(start=anchor, periods=int(future_steps)+1, freq=freq)[1:]
            else:
                future_index = pd.date_range(start=anchor, periods=int(future_steps)+1, freq='D')[1:]
        else:
            base_value = history_index_array[-1] if history_length > 0 else window_size
            future_index = np.arange(int(base_value) + 1, int(base_value) + 1 + int(future_steps))

    mse, mae, rmse, mape = metrics(history_truth_values, history_pred_values)

    return {
        'well_name': well_name,
        'predictions': history_pred_values,
        'true_values': history_truth_values,
        'index': history_index,
        'metrics': {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape},
        'original_data': ser.flatten(),
        'original_index': original_index,
        'future_pred': future_pred,
        'future_index': future_index,
    }


def build_result_payload(result):
    """将单井预测结果转换为统一的 JSON 结构。"""
    if result is None:
        return None

    well_id = result.get('well_id', result.get('well_name'))

    history_records = []
    for idx_val, pred_val, true_val in zip(result['index'], result['predictions'], result['true_values']):
        history_records.append({
            'wellCol': well_id,
            'tm': _format_index_value(idx_val),
            'prediction': _safe_float(pred_val),
            'truth': _safe_float(true_val),
        })

    future_records = []
    future_pred = result.get('future_pred')
    future_index = result.get('future_index')
    if future_pred is not None and future_index is not None:
        for idx_val, pred_val in zip(future_index, future_pred):
            future_records.append({
                'wellCol': well_id,
                'tm': _format_index_value(idx_val),
                'prediction': _safe_float(pred_val),
            })

    metrics = result.get('metrics', {})
    return {
        'wellCol': well_id,
        'dataList': history_records,
        'futureList': future_records,
        'metrics': {
            'MAE': _safe_float(metrics.get('mae')),
            'MAPE': _safe_float(metrics.get('mape')),
            'RMSE': _safe_float(metrics.get('rmse')),
        }
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
            # 追加未来预测曲线（如有）
            if result.get('future_pred') is not None and result.get('future_index') is not None:
                ax.plot(result['future_index'], result['future_pred'], label='未来预测', color='green', linewidth=2, linestyle=':')
            
            # 设置标题和标签
            ax.set_title(f'{well_name} - {model_type.upper()} 预测结果\n'
                        f'RMSE={metrics_dict["rmse"]:.4f}, MAE={metrics_dict["mae"]:.4f}, MAPE={metrics_dict["mape"]:.2f}%',
                        fontsize=14)
            is_datetime_index = hasattr(index, 'dtype') and 'datetime' in str(index.dtype)
            ax.set_xlabel('时间' if is_datetime_index else '时间步', fontsize=12)
            ax.set_ylabel('地下水位 (GWL)', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 如果是时间索引，设置年月日格式并提高刻度密度
            if is_datetime_index:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=12, maxticks=30))
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图片到模型子目录
            model_output_dir = os.path.join(output_dir, model_type)
            os.makedirs(model_output_dir, exist_ok=True)
            plot_path = os.path.join(model_output_dir, f"prediction_ml_{model_type}_{well_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"预测图已保存: {plot_path}")



def predict_multi_wells_ml(
    data_source,
    model_dir,
    output_dir,
    window_size,
    pred_len,
    model_type=None,
    target_wells=None,
    future_steps: int = 0,
    rainfall_data=None,
):
    """多井位机器学习模型预测主函数。"""

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        print(f"提示: data_source 不是 DataFrame，尝试读取 Excel 数据: {data_source}")
        df = pd.read_excel(data_source)
        if "时间" in df.columns:
            try:
                df["时间"] = pd.to_datetime(df["时间"])
            except Exception:
                pass
            df = df.set_index("时间")
        elif "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.set_index("日期")
        elif "tm" in df.columns:
            df["tm"] = pd.to_datetime(df["tm"], errors='coerce')
            df = df.dropna(subset=["tm"])
            df = df.set_index("tm")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()

    if df.empty:
        print('错误: 输入数据为空，无法进行预测。')
        return

    available_wells = list(df.columns)
    print(f"数据形状: {df.shape}")
    print(f"可用井位: {available_wells}")

    if target_wells:
        target_wells = [str(w).strip() for w in target_wells if str(w).strip()]
        missing = [w for w in target_wells if w not in df.columns]
        if missing:
            print(f"警告: 数据中未找到井位 {missing}，将忽略。")
        target_wells = [w for w in target_wells if w in df.columns]
        if not target_wells:
            print('错误: 所选井位均不存在于数据中。')
            return
        df = df[target_wells]
    else:
        target_wells = available_wells

    if not target_wells:
        print('错误: 无井位可用于预测。')
        return

    print(f"最终预测的井位: {target_wells}")

    rainfall_series = _load_rainfall_series(rainfall_data, df.index)

    if not os.path.exists(model_dir):
        print(f"绝对路径不存在: {model_dir}")
        print("尝试使用相对路径...")
        relative_model_dir = "results/multi_wells_ml"
        if os.path.exists(relative_model_dir):
            model_dir = relative_model_dir
            print(f"找到相对路径: {model_dir}")
        else:
            print("未找到模型目录")
            return

    print("=== 多井位机器学习模型预测 ===")
    print(f"模型目录: {model_dir}")
    print(f"输出目录: {output_dir}")

    print("\n1. 对各井位执行预测...")

    results = []

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    print(f"找到模型文件: {model_files}")

    unified_models = {}
    if model_type in ['xgb', 'lgbm']:
        unified_model_file = f"{model_type}_multi_wells_individual.joblib"
        unified_model_path = os.path.join(model_dir, unified_model_file)
        if os.path.exists(unified_model_path):
            unified_models[model_type] = unified_model_path
            print(f"选择模型类型: {model_type} -> {unified_model_file}")
        else:
            print(f"未找到指定模型类型的统一权重: {unified_model_file}")
    else:
        for mt in ['xgb', 'lgbm']:
            unified_model_file = f"{mt}_multi_wells_individual.joblib"
            unified_model_path = os.path.join(model_dir, unified_model_file)
            if os.path.exists(unified_model_path):
                unified_models[mt] = unified_model_path
                print(f"找到统一模型: {unified_model_file}")

    if unified_models:
        print(f"使用统一模型对井位 {target_wells} 进行预测...")
        for mt, model_path in unified_models.items():
            for well_name in target_wells:
                print(f"使用统一{mt.upper()}模型预测井位: {well_name}")
                result = predict_single_well(
                    df[well_name], model_path, f"unified_{mt}_{well_name}",
                    window_size, pred_len, future_steps,
                    rainfall_series=rainfall_series
                )
                if result:
                    result['well_id'] = well_name
                    results.append(result)
                    metrics_dict = result['metrics']
                    print(f"✅ 统一{mt.upper()}_{well_name}: RMSE={metrics_dict['rmse']:.4f}, "
                          f"MAE={metrics_dict['mae']:.4f}, MAPE={metrics_dict['mape']:.2f}%")
                else:
                    print(f"❌ 统一{mt.upper()}_{well_name}: 预测失败")
    else:
        print("未找到统一模型，尝试使用单独的模型文件...")
        for model_file in model_files:
            parts = model_file.replace('.joblib', '').split('_', 1)
            if len(parts) != 2:
                continue
            mt, well_name = parts
            model_path = os.path.join(model_dir, model_file)
            if well_name not in target_wells:
                continue
            result = predict_single_well(
                df[well_name], model_path, f"{mt}_{well_name}",
                window_size, pred_len, future_steps,
                rainfall_series=rainfall_series
            )
            if result:
                result['well_id'] = well_name
                results.append(result)
                metrics_dict = result['metrics']
                print(f"✅ {mt}_{well_name}: RMSE={metrics_dict['rmse']:.4f}, "
                      f"MAE={metrics_dict['mae']:.4f}, MAPE={metrics_dict['mape']:.2f}%")
            else:
                print(f"❌ {mt}_{well_name}: 预测失败")

    if not results:
        print("没有成功的预测结果")
        return

    print("\n2. 绘制预测结果图...")
    plot_predictions(results, output_dir)

    print("\n3. 保存预测结果...")
    for mt in ['xgb', 'lgbm']:
        model_results = [r for r in results if r and mt in r['well_name'].lower()]
        if not model_results:
            continue
        model_output_dir = os.path.join(output_dir, mt)
        os.makedirs(model_output_dir, exist_ok=True)

        aggregated = {
            'modelType': mt.upper(),
            'dataList': [],
            'futureList': [],
            'metrics': []
        }

        for res in model_results:
            payload = build_result_payload(res)
            if not payload:
                continue
            aggregated['dataList'].extend(payload['dataList'])
            aggregated['futureList'].extend(payload['futureList'])
            metrics_entry = payload['metrics'].copy()
            metrics_entry['wellCol'] = payload['wellCol']
            aggregated['metrics'].append(metrics_entry)

        json_path = os.path.join(model_output_dir, f"prediction_results_ml_{mt}.json")
        with open(json_path, 'w', encoding='utf-8') as fp:
            json.dump(aggregated, fp, ensure_ascii=False, indent=4)
        print(f"预测结果已保存到JSON: {json_path}")

    print("\n4. 预测结果汇总:")
    print("-" * 60)
    print(f"{'模型':<15} {'井位':<10} {'RMSE':<10} {'MAE':<10} {'MAPE(%)':<10}")
    print("-" * 60)

    for result in results:
        well_name = result['well_name']
        metrics_dict = result['metrics']
        print(f"{well_name:<15} {'':<10} {metrics_dict['rmse']:<10.4f} "
              f"{metrics_dict['mae']:<10.4f} {metrics_dict['mape']:<10.2f}")

    print("-" * 60)

    avg_rmse = np.mean([r['metrics']['rmse'] for r in results if not np.isnan(r['metrics']['rmse'])])
    avg_mae = np.mean([r['metrics']['mae'] for r in results if not np.isnan(r['metrics']['mae'])])
    avg_mape = np.mean([r['metrics']['mape'] for r in results if not np.isnan(r['metrics']['mape'])])

    print(f"{'平均值':<15} {'':<10} {avg_rmse:<10.4f} {avg_mae:<10.4f} {avg_mape:<10.2f}")
    print("\n预测完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML 多井位预测 (XGB/LGBM)')
    parser.add_argument('--config', dest='config_path', default='database/区域json.json',
                        help='区域配置 JSON 路径')
    parser.add_argument('--model-dir', dest='model_dir',
                        default=r"E:/Coding_path/Groundwater_Level_Prediction_ML_DL/results/multi_wells_ml",
                        help='模型目录，包含统一学习器权重')
    parser.add_argument('--fallback-model-type', dest='fallback_model_type', choices=['xgb', 'lgbm'], default='lgbm',
                        help='当 JSON 中的 modelType 不适用时使用的默认模型类型')
    parser.add_argument('--wells', dest='wells', default=None,
                        help='可选，指定井位编号（逗号分隔），默认使用 JSON 中的 wellCol')
    parser.add_argument('--window-size', dest='window_size', type=int, default=24,
                        help='滑动窗口大小')
    parser.add_argument('--pred-len', dest='pred_len', type=int, default=4,
                        help='预测步长')
    parser.add_argument('--future-steps', dest='future_steps', type=int, default=None,
                        help='覆盖 JSON 中 futureSteps 的值，留空则使用 JSON 配置')
    parser.add_argument('--output-dir', dest='output_dir', default='results/inference_ml_multi',
                        help='预测结果输出目录')
    parser.add_argument('--rainfall-path', dest='rainfall_path', default=None,
                        help='可选，降雨数据 Excel 路径，用于覆盖 JSON 中的降雨序列')

    args = parser.parse_args()

    print(f"读取配置文件: {args.config_path}")
    region_config = load_region_config(args.config_path)
    data_df = region_config['data_frame']
    rainfall_series = region_config['rainfall_series']
    well_ids = region_config['well_ids']
    config_model_type = (region_config['model_type'] or '').lower() if region_config['model_type'] else ''
    future_steps_cfg = region_config['future_steps']

    if data_df is None or data_df.empty:
        raise SystemExit('错误: 区域配置中的 dataList 为空，无法进行预测。')

    if args.wells:
        target_wells = [w.strip() for w in args.wells.split(',') if w.strip()]
    else:
        target_wells = well_ids if well_ids else list(data_df.columns)

    if not target_wells:
        target_wells = list(data_df.columns)

    rainfall_data = args.rainfall_path if args.rainfall_path else rainfall_series

    candidate_type = config_model_type if config_model_type in ('xgb', 'lgbm') else args.fallback_model_type
    if config_model_type and config_model_type not in ('xgb', 'lgbm'):
        print(f"警告: JSON 中的 modelType={region_config['model_type']} 不适用于当前脚本，改用 {candidate_type} 模型。")

    effective_future_steps = args.future_steps if args.future_steps is not None else future_steps_cfg

    predict_multi_wells_ml(
        data_source=data_df,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        pred_len=args.pred_len,
        model_type=candidate_type,
        target_wells=target_wells,
        future_steps=effective_future_steps,
        rainfall_data=rainfall_data,
    )





