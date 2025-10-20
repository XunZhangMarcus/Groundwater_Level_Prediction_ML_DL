#!/usr/bin/env python
# predict_arima_multi_wells_inference.py - 多井位 ARIMA 模型预测

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

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


def _aggregate_overlapping(pred_matrix):
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
            where=counts > 0,
        )
    return aggregated


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
        return np.nan, np.nan, np.nan, np.nan
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + 1e-8))) * 100
    return mse, mae, rmse, mape


def locate_model(model_dir, well_name):
    candidates = []
    if model_dir:
        model_dir = Path(model_dir)
        if model_dir.is_file():
            candidates.append(model_dir)
        elif model_dir.is_dir():
            candidates.append(model_dir / 'arima_model_unified_arima_model.pkl')
            candidates.append(model_dir / f'arima_model_{well_name}.pkl')
    fallback_dir = Path('results/multi_wells_arima')
    candidates.append(fallback_dir / 'arima_model_unified_arima_model.pkl')
    candidates.append(fallback_dir / f'arima_model_{well_name}.pkl')

    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return None


def predict_single_well(
    model_path,
    well_series,
    well_name,
    train_ratio,
    future_steps,
):
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None

    well_series = pd.to_numeric(well_series, errors='coerce')
    well_series = well_series.dropna()

    if well_series.empty or len(well_series) < 20:
        print(f"井位 {well_name} 数据不足或全为空值")
        return None

    with open(model_path, 'rb') as fp:
        model = pickle.load(fp)

    train_size = int(len(well_series) * train_ratio)
    train_series = well_series.iloc[:train_size]
    test_series = well_series.iloc[train_size:]

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

    mse, mae, rmse, mape = metrics(history_truth.values, history_pred.values)

    future_pred = None
    future_index = None
    if future_steps and future_steps > 0:
        future_pred = forecast_values.iloc[len(test_series):len(test_series) + future_steps].to_numpy()
        if isinstance(history_truth.index, pd.DatetimeIndex):
            anchor = history_truth.index[-1]
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

    return {
        'well_name': str(well_name),
        'predictions': history_pred.values,
        'true_values': history_truth.values,
        'index': history_truth.index,
        'metrics': {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape},
        'future_pred': future_pred,
        'future_index': future_index,
    }


def build_result_payload(result):
    if result is None:
        return None

    well_id = result.get('well_name')

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
    os.makedirs(output_dir, exist_ok=True)

    for result in results:
        if result is None:
            continue
        well_name = result['well_name']
        predictions = result['predictions']
        true_values = result['true_values']
        index = result['index']
        metrics_dict = result['metrics']

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(index, true_values, label='真实值', color='blue', linewidth=2)
        ax.plot(index, predictions, label='预测值', color='red', linewidth=2, linestyle='--')
        if result.get('future_pred') is not None and result.get('future_index') is not None:
            ax.plot(result['future_index'], result['future_pred'], label='未来预测', color='green', linewidth=2, linestyle=':')

        is_datetime_index = hasattr(index, 'dtype') and 'datetime' in str(index.dtype)
        ax.set_title(
            f'{well_name} - ARIMA 预测结果\n'
            f'RMSE={metrics_dict["rmse"]:.4f}, MAE={metrics_dict["mae"]:.4f}, MAPE={metrics_dict["mape"]:.2f}%'
        )
        ax.set_xlabel('时间' if is_datetime_index else '时间步', fontsize=12)
        ax.set_ylabel('地下水位 (GWL)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        if is_datetime_index:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=12, maxticks=30))
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"prediction_arima_{well_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"预测图已保存: {plot_path}")


def predict_multi_wells_arima(
    data_source,
    model_path,
    output_dir,
    train_ratio,
    future_steps,
):
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
        return []

    results = []
    for well in df.columns:
        print(f"\n>>> 正在处理井位 {well}")
        well_series = df[well]
        result = predict_single_well(
            model_path=model_path,
            well_series=well_series,
            well_name=str(well),
            train_ratio=train_ratio,
            future_steps=future_steps,
        )
        if result:
            results.append(result)
        else:
            print(f"❌ ARIMA_{well}: 预测失败")

    if not results:
        print("没有成功的预测结果")
        return []

    print("\n2. 绘制预测结果图...")
    plot_predictions(results, output_dir)

    print("\n3. 保存预测结果...")
    aggregated = {
        'modelType': 'arima',
        'dataList': [],
        'futureList': [],
        'metrics': []
    }

    for res in results:
        payload = build_result_payload(res)
        if payload:
            aggregated['dataList'].extend(payload['dataList'])
            aggregated['futureList'].extend(payload['futureList'])
            metrics_entry = payload['metrics'].copy()
            metrics_entry['wellCol'] = payload['wellCol']
            aggregated['metrics'].append(metrics_entry)

    json_path = os.path.join(output_dir, "prediction_results_arima.json")
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(aggregated, fp, ensure_ascii=False, indent=4)
    print(f"预测结果已保存到JSON: {json_path}")

    print("\n4. 预测结果汇总:")
    print("-" * 60)
    print(f"{'井位':<15} {'RMSE':<12} {'MAE':<12} {'MAPE(%)':<12}")
    print("-" * 60)
    for res in results:
        print(f"{res['well_name']:<15} {res['metrics']['rmse']:<12.4f} {res['metrics']['mae']:<12.4f} {res['metrics']['mape']:<12.2f}")
    print("-" * 60)

    avg_rmse = np.nanmean([res['metrics']['rmse'] for res in results])
    avg_mae = np.nanmean([res['metrics']['mae'] for res in results])
    avg_mape = np.nanmean([res['metrics']['mape'] for res in results])
    print(f"{'平均值':<15} {'':<12} {avg_rmse:<12.4f} {avg_mae:<12.4f} {avg_mape:<12.2f}")
    print("\nARIMA 预测完成！")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARIMA 多井位预测')
    parser.add_argument('--config', dest='config_path', default='database/区域arima_json.json',
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
    parser.add_argument('--output-dir', dest='output_dir', default='results/inference_arima_multi',
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
        target_wells = [str(col) for col in data_df.columns]
    else:
        target_wells = well_ids

    model_file = args.model_path
    if not model_file and target_wells:
        model_file = locate_model(args.model_dir, target_wells[0])

    if not model_file:
        print('错误: 未找到任何训练好的 ARIMA 模型文件。')
        raise SystemExit(1)
    print(f"使用模型: {model_file}")

    effective_future_steps = args.future_steps if args.future_steps is not None else future_steps_cfg

    predict_multi_wells_arima(
        data_source=data_df,
        model_path=model_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        future_steps=effective_future_steps,
    )
