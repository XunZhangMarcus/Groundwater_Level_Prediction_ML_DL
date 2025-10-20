#!/usr/bin/env python
# predict_arima_single_well_inference.py - 单井位 ARIMA 模型预测

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _format_index_value(value):
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
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def load_region_config(json_path: str):
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"未找到配置文件: {json_path}")

    with json_file.open('r', encoding='utf-8') as fp:
        payload = json.load(fp)

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
        'future_steps': future_steps,
        'raw_payload': payload,
    }


def metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + 1e-8))) * 100
    return rmse, mae, mape


def predict_with_arima(
    model_path,
    data_df,
    well_col,
    train_ratio,
    future_steps,
    output_dir,
):
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return None

    if isinstance(well_col, str):
        if well_col not in data_df.columns:
            raise ValueError(f"井位 {well_col} 不存在于数据列中")
        well_name = well_col
    else:
        idx = int(well_col) - 1
        if idx < 0 or idx >= len(data_df.columns):
            raise IndexError(f"井位索引 {well_col} 超出范围")
        well_name = data_df.columns[idx]

    print("=== 单井位 ARIMA 模型预测 ===")
    print(f"井位: {well_name}")
    print(f"模型文件: {model_path}")

    well_series = pd.to_numeric(data_df[well_name], errors='coerce')
    well_series = well_series.dropna()

    if well_series.empty:
        print(f"❌ 井位 {well_name} 数据为空，无法预测")
        return None

    if len(well_series) < 20:
        print(f"❌ 井位 {well_name} 数据量不足 (有效点 {len(well_series)})")
        return None

    with open(model_path, 'rb') as fp:
        model = pickle.load(fp)

    print("模型加载成功")

    train_size = int(len(well_series) * train_ratio)
    train_series = well_series.iloc[:train_size]
    test_series = well_series.iloc[train_size:]

    print(f"训练集样本数: {len(train_series)}, 测试集样本数: {len(test_series)}")

    try:
        order = model.model.order
    except Exception:
        order = None

    if order is None:
        local_model = model
    else:
        local_model = sm.tsa.ARIMA(train_series, order=order).fit()

    train_pred = local_model.fittedvalues
    if hasattr(train_pred, 'reindex'):
        train_pred = train_pred.reindex(train_series.index)

    total_steps = len(test_series) + max(int(future_steps or 0), 0)
    forecast = local_model.get_forecast(steps=total_steps)
    forecast_values = forecast.predicted_mean

    forecast_test = forecast_values.iloc[:len(test_series)].copy()
    if len(test_series) > 0:
        forecast_test.index = test_series.index

    history_truth = pd.concat([train_series, test_series])
    history_pred = pd.concat([train_pred, forecast_test])
    history_pred = history_pred.reindex(history_truth.index)

    rmse, mae, mape = metrics(history_truth.values, history_pred.values)

    future_pred = None
    future_index = None
    if future_steps and future_steps > 0:
        future_pred = forecast_values.iloc[len(test_series):len(test_series) + future_steps].to_numpy()
        if len(history_truth.index) > 0:
            anchor = history_truth.index[-1]
        else:
            anchor = None
        if isinstance(history_truth.index, pd.DatetimeIndex):
            try:
                freq = pd.infer_freq(history_truth.index)
            except Exception:
                freq = None
            if freq:
                future_index = pd.date_range(start=anchor, periods=future_steps + 1, freq=freq)[1:]
            else:
                future_index = pd.date_range(start=anchor, periods=future_steps + 1, freq='D')[1:]
        else:
            base_value = history_truth.index[-1] if len(history_truth.index) > 0 else 0
            future_index = np.arange(int(base_value) + 1, int(base_value) + 1 + int(future_steps))

    print("正在生成预测结果图...")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history_truth.index, history_truth.values, label='真实值', color='blue', linewidth=2)
    ax.plot(history_pred.index, history_pred.values, label='预测值', color='red', linewidth=2, linestyle='--')
    if future_pred is not None and future_index is not None:
        ax.plot(future_index, future_pred, label='未来预测', color='green', linewidth=2, linestyle=':')

    ax.set_title(f"{well_name} - ARIMA 模型预测结果\nRMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
    ax.set_ylabel('地下水位 (GWL)')
    ax.set_xlabel('时间' if isinstance(history_truth.index, pd.DatetimeIndex) else '时间步')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if isinstance(history_truth.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=12, maxticks=30))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"prediction_arima_{well_name}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"预测结果图已保存: {plot_path}")

    print("正在保存预测结果...")
    data_list = []
    for idx in history_truth.index:
        data_list.append({
            'wellCol': str(well_name),
            'tm': _format_index_value(idx),
            'prediction': _safe_float(history_pred.loc[idx]),
            'truth': _safe_float(history_truth.loc[idx]),
        })

    future_list = []
    if future_pred is not None and future_index is not None:
        for idx, pred_val in zip(future_index, future_pred):
            future_list.append({
                'wellCol': str(well_name),
                'tm': _format_index_value(idx),
                'prediction': _safe_float(pred_val),
            })

    payload = {
        'wellCol': str(well_name),
        'dataList': data_list,
        'futureList': future_list,
        'metrics': {
            'MAE': _safe_float(mae),
            'MAPE': _safe_float(mape),
            'RMSE': _safe_float(rmse),
        },
        'modelType': 'arima',
    }

    json_path = os.path.join(output_dir, f"prediction_results_arima_{well_name}.json")
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=4)
    print(f"预测结果已保存: {json_path}")

    print("\n=== 预测完成 ===")
    print(f"井位: {well_name}")
    print(f"历史样本数: {len(history_truth)}")
    print(f"整体RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    if future_pred is not None:
        print(f"未来预测步数: {len(future_pred)}")
    print(f"结果保存目录: {output_dir}/")

    return history_pred.values, history_truth.values, rmse, mae, mape


def locate_model(model_path, model_dir, well_name):
    if model_path:
        if os.path.exists(model_path):
            return model_path
        print(f"警告: 指定的模型文件不存在 {model_path}")

    candidates = []
    if model_dir:
        if os.path.isdir(model_dir):
            candidates.append(Path(model_dir) / 'arima_model_unified_arima_model.pkl')
            candidates.append(Path(model_dir) / f'arima_model_{well_name}.pkl')
        elif os.path.isfile(model_dir):
            candidates.append(Path(model_dir))

    fallback_dir = Path('results/multi_wells_arima')
    candidates.append(fallback_dir / 'arima_model_unified_arima_model.pkl')
    candidates.append(fallback_dir / f'arima_model_{well_name}.pkl')

    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARIMA 单井位预测')
    parser.add_argument('--config', dest='config_path', default='database/单站arima_json.json',
                        help='区域配置 JSON 路径')
    parser.add_argument('--model', dest='model_path', default=None,
                        help='模型文件(.pkl)路径；如果未提供则结合目录自动查找')
    parser.add_argument('--model-dir', dest='model_dir',
                        default=r"E:/Coding_path/Groundwater_Level_Prediction_ML_DL/results/multi_wells_arima",
                        help='ARIMA 模型目录')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float, default=0.8,
                        help='训练集比例 (0-1)')
    parser.add_argument('--future-steps', dest='future_steps', type=int, default=None,
                        help='覆盖 JSON 中 futureSteps 的值，留空则使用 JSON 配置')
    parser.add_argument('--output-dir', dest='output_dir', default='results/inference_arima_single',
                        help='预测结果输出目录')

    args = parser.parse_args()

    print(f"读取配置文件: {args.config_path}")
    region_config = load_region_config(args.config_path)
    data_df = region_config['data_frame']
    well_ids = region_config['well_ids']
    future_steps_cfg = region_config['future_steps']

    if data_df is None or data_df.empty:
        raise SystemExit('错误: 区域配置中的 dataList 为空，无法进行预测。')

    if not well_ids:
        well_ids = [str(col) for col in data_df.columns]

    unique_wells = [w for w in well_ids if w]
    if len(unique_wells) != 1:
        print(f"错误: 该脚本仅支持单井位预测，检测到 {len(unique_wells)} 个井位: {unique_wells}")
        raise SystemExit(1)

    target_well = unique_wells[0]

    model_file = locate_model(args.model_path, args.model_dir, target_well)
    if not model_file:
        print('错误: 未找到任何训练好的 ARIMA 模型文件。')
        raise SystemExit(1)
    print(f"使用模型: {model_file}")

    effective_future_steps = args.future_steps if args.future_steps is not None else future_steps_cfg

    os.makedirs(args.output_dir, exist_ok=True)

    predict_with_arima(
        model_path=model_file,
        data_df=data_df,
        well_col=target_well,
        train_ratio=args.train_ratio,
        future_steps=effective_future_steps,
        output_dir=args.output_dir,
    )
