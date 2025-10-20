# predict_ml_inference.py - 使用训练好的XGB/LGBM模型进行预测
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from pathlib import Path
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

# 设置matplotlib中文字体
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

def _load_rainfall_series(rainfall_source, reference_index):
    """读取并对齐降雨序列，保持与井位索引一致。"""
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


def predict_with_model(model_path, data_path, well_col, window_size, pred_len, future_steps: int = 0, rainfall_data=None):
    """
    使用训练好的XGB/LGBM模型进行预测
    完全仿照train_ml_single_well.py的处理方式
    
    Args:
        model_path: 模型权重文件路径 (.joblib)
        data_path: 数据文件路径或已整理好的 DataFrame
        well_col: 井位列索引
        window_size: 窗口大小
        pred_len: 预测长度
        rainfall_data: 降雨数据（文件路径或 Series ）
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
    if isinstance(data_path, pd.DataFrame):
        df = data_path.copy()
    else:
        df = pd.read_excel(data_path)
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
        raise ValueError('输入数据为空，无法进行预测。')

    column_alias = {str(col): col for col in df.columns}
    if isinstance(well_col, str):
        if well_col not in column_alias:
            raise ValueError(f"井位 {well_col} 不存在于数据列中")
        well_column = column_alias[well_col]
    else:
        try:
            well_idx = int(well_col) - 1
        except Exception as exc:
            raise ValueError(f"井位标识 {well_col} 无法解析为索引") from exc
        if well_idx < 0 or well_idx >= len(df.columns):
            raise IndexError(f"井位索引 {well_col} 超出范围")
        well_column = df.columns[well_idx]

    well_name = str(well_column)
    well_series = pd.to_numeric(df[well_column], errors='coerce')
    ser = well_series.values.astype("float32").reshape(-1, 1)

    rainfall_series = _load_rainfall_series(rainfall_data, df.index)
    print(f"井位名称: {well_name}")
    print(f"数据长度: {len(ser)}, 有效数据: {(~np.isnan(ser)).sum()}")

    rainfall_windows = None
    rainfall_future_window = None
    if rainfall_series is not None:
        try:
            rainfall_aligned = rainfall_series.reindex(well_series.index)
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

    # 数据归一化 - 与训练脚本完全相同
    scaler = MinMaxScaler()
    ser_s = scaler.fit_transform(ser).flatten()
    
    # 创建滑动窗口 - 与训练脚本完全相同
    X, y = create_sliding_window(ser_s, window_size, pred_len)
    print(f"生成样本数: {len(X)}")

    if X.size == 0:
        print('警告: 样本数量不足，无法执行模型预测。')
        return None

    if X.ndim == 1:
        X = X.reshape(-1, window_size)

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
    X_input = X
    if rainfall_windows is not None and len(rainfall_windows) >= len(X):
        rainfall_slice = rainfall_windows[:len(X)]
        X_input = X + 0.0 * rainfall_slice
    predictions = model.predict(X_input)

    # 反归一化
    pred_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, pred_len)
    true_inv = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1, pred_len)

    history_pred_values = _aggregate_overlapping_predictions(pred_inv)
    history_truth_values = _aggregate_overlapping_predictions(true_inv)

    start_idx = window_size
    history_length = len(history_pred_values)
    max_available = max(len(df.index) - start_idx, 0)
    if history_length > max_available:
        history_length = max_available
        history_pred_values = history_pred_values[:history_length]
        history_truth_values = history_truth_values[:history_length]

    if history_length == 0:
        print('警告: 无法生成与历史数据对齐的预测结果。')
        return None

    index_array = df.index.to_numpy()
    history_index_array = index_array[start_idx:start_idx + history_length]
    if isinstance(df.index, pd.DatetimeIndex):
        history_values = pd.DatetimeIndex(history_index_array)
        use_datetime_history = True
    else:
        history_values = pd.Index(history_index_array)
        use_datetime_history = isinstance(history_values, pd.DatetimeIndex) or 'datetime' in str(history_values.dtype)

    valid_mask = (~np.isnan(history_truth_values)) & (~np.isnan(history_pred_values))
    if np.any(valid_mask):
        overall_mse, overall_mae, overall_rmse, overall_mape = metrics(
            history_truth_values[valid_mask],
            history_pred_values[valid_mask]
        )
    else:
        overall_mse = overall_mae = overall_rmse = overall_mape = float('nan')

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
            y_out = model.predict(x_in)
            nxt = float(y_out.reshape(1, -1)[0, 0])
            fut_norm.append(nxt)
            last_window = np.concatenate([last_window[1:], [nxt]]).astype(np.float32)
            if rainfall_future_copy is not None and len(rainfall_future_copy) > 0:
                rainfall_future_copy = np.concatenate([rainfall_future_copy[1:], [rainfall_future_copy[-1]]]).astype(np.float32)
        future_pred = scaler.inverse_transform(np.array(fut_norm).reshape(-1, 1)).flatten()

        if isinstance(df.index, pd.DatetimeIndex):
            anchor = df.index[-1]
            try:
                freq = pd.infer_freq(df.index)
            except Exception:
                freq = None
            if freq:
                future_index = pd.date_range(start=anchor, periods=int(future_steps)+1, freq=freq)[1:]
            else:
                future_index = pd.date_range(start=anchor, periods=int(future_steps)+1, freq='D')[1:]
        else:
            base_value = history_index_array[-1] if history_length > 0 else window_size
            future_index = np.arange(int(base_value)+1, int(base_value)+1+int(future_steps))

    print("正在生成预测结果图...")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history_values, history_truth_values, label='真实值', color='blue', linewidth=2)
    ax.plot(history_values, history_pred_values, label='预测值', color='red', linewidth=2, linestyle='--')
    if future_pred is not None and future_index is not None:
        ax.plot(future_index, future_pred, label='未来预测', color='green', linewidth=2, linestyle=':')

    ax.set_title(f"{well_name} - {model_type.upper()} 模型预测结果\nRMSE={overall_rmse:.4f}, MAE={overall_mae:.4f}, MAPE={overall_mape:.2f}%")
    ax.set_ylabel('地下水位 (GWL)')
    ax.set_xlabel('时间' if use_datetime_history else '时间步')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if use_datetime_history:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=12, maxticks=30))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_dir = "results/inference_ml_single"
    os.makedirs(output_dir, exist_ok=True)
    model_output_dir = os.path.join(output_dir, model_type.lower())
    os.makedirs(model_output_dir, exist_ok=True)
    plot_path = f"{model_output_dir}/prediction_ml_{model_type.lower()}_{well_name}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"预测结果图已保存: {plot_path}")


    print("正在保存预测结果...")

    data_list = []
    for idx_val, true_val, pred_val in zip(history_values, history_truth_values, history_pred_values):
        data_list.append({
            'wellCol': well_name,
            'tm': _format_index_value(idx_val),
            'prediction': _safe_float(pred_val),
            'truth': _safe_float(true_val),
        })

    future_list = []
    if future_pred is not None and future_index is not None:
        for idx_val, pred_val in zip(future_index, future_pred):
            future_list.append({
                'wellCol': well_name,
                'tm': _format_index_value(idx_val),
                'prediction': _safe_float(pred_val),
            })

    output_payload = {
        'wellCol': well_name,
        'dataList': data_list,
        'futureList': future_list,
        'metrics': {
            'MAE': _safe_float(overall_mae),
            'MAPE': _safe_float(overall_mape),
            'RMSE': _safe_float(overall_rmse),
        },
        'modelType': model_type,
    }
    json_path = f"{model_output_dir}/prediction_results_ml_{model_type.lower()}_{well_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=4)

    print(f"预测结果已保存: {json_path}")

    print("\n=== 预测完成 ===")
    print(f"井位: {well_name}")
    print(f"模型: {model_type}")
    print(f"历史样本数: {len(history_pred_values)}")
    print(f"整体RMSE: {overall_rmse:.4f}, MAE: {overall_mae:.4f}, MAPE: {overall_mape:.2f}%")
    if future_pred is not None:
        print(f"未来预测步数: {len(future_pred)}")
    print(f"结果保存目录: {output_dir}/")

    return history_pred_values, history_truth_values, overall_rmse, overall_mae, overall_mape

# 统一的命令行入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML 单井位预测 (XGB/LGBM)')
    parser.add_argument('--config', dest='config_path', default='database/单站lgbm_json.json',
                        help='区域配置 JSON 路径')
    parser.add_argument('--model', dest='model_path', default=None,
                        help='模型文件(.joblib)路径；如果未提供则结合配置选择默认模型')
    parser.add_argument('--model-dir', dest='model_dir',
                        default=r"E:/Coding_path/Groundwater_Level_Prediction_ML_DL/results/multi_wells_ml",
                        help='统一模型目录，包含 xgb_multi_wells_individual.joblib / lgbm_multi_wells_individual.joblib')
    parser.add_argument('--fallback-model-type', dest='fallback_model_type', choices=['xgb', 'lgbm'], default='lgbm',
                        help='当 JSON 中的 modelType 不适用于当前脚本时使用的默认模型类型')
    parser.add_argument('--window-size', dest='window_size', type=int, default=24,
                        help='滑动窗口大小')
    parser.add_argument('--pred-len', dest='pred_len', type=int, default=4,
                        help='预测步长')

    args = parser.parse_args()

    print(f"读取配置文件: {args.config_path}")
    region_config = load_region_config(args.config_path)
    data_df = region_config['data_frame']
    rainfall_series = region_config['rainfall_series']
    well_ids = region_config['well_ids']
    config_model_type = region_config['model_type']
    future_steps = region_config['future_steps']

    if data_df is None or data_df.empty:
        raise SystemExit('错误: 区域配置中的 dataList 为空，无法进行预测。')

    if not well_ids:
        print('警告: JSON 未提供 wellCol 信息，将使用数据列名称作为井位标识。')
        well_ids = [str(col) for col in data_df.columns]

    unique_wells = [w for w in well_ids if w]
    if len(unique_wells) != 1:
        print(f"错误: 该脚本仅支持单井位预测，检测到 {len(unique_wells)} 个井位: {unique_wells}")
        raise SystemExit(1)

    target_well = unique_wells[0]

    model_path = args.model_path
    model_type_key = (config_model_type or '').lower() if config_model_type else ''
    if not model_path:
        candidate_type = model_type_key if model_type_key in ('xgb', 'lgbm') else args.fallback_model_type
        if model_type_key and model_type_key not in ('xgb', 'lgbm'):
            print(f"警告: JSON 中的 modelType={config_model_type} 不适用于当前脚本，改用 {candidate_type} 模型。")

        unified_name = f"{candidate_type}_multi_wells_individual.joblib"
        candidate = os.path.join(args.model_dir, unified_name)
        if os.path.exists(candidate):
            model_path = candidate
            print(f"使用模型: {candidate_type} -> {model_path}")
        else:
            rel_dir = 'results/multi_wells_ml'
            candidate_rel = os.path.join(rel_dir, unified_name)
            if os.path.exists(candidate_rel):
                model_path = candidate_rel
                print(f"使用相对路径模型: {candidate_type} -> {model_path}")

    if not model_path:
        print('错误: 未找到任何训练好的模型文件。请通过 --model 指定，或检查模型目录。')
    else:
        print(f"\n>>> 正在处理井位 {target_well}")
        predict_with_model(
            model_path=model_path,
            data_path=data_df,
            well_col=target_well,
            window_size=args.window_size,
            pred_len=args.pred_len,
            future_steps=future_steps,
            rainfall_data=rainfall_series,
        )
