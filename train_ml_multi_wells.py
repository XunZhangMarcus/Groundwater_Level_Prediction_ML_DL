#!/usr/bin/env python
# train_ml_multi_wells.py  ——  多井位机器学习同时建模预测
import argparse, os, random, logging, joblib, warnings, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json

# —————————————————— 机器学习器 —————————————————— #
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 0. 通用工具
# ----------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def create_sliding_window(series, window, pred_len=1):
    X, y = [], []
    for i in range(len(series) - window - pred_len + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + pred_len])
    return np.asarray(X), np.asarray(y)


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


# ----------------------------------------------------------------------
# 1. 单井位机器学习建模函数
# ----------------------------------------------------------------------
def fit_single_well_ml(well_data, well_name, cfg, well_index, model_type):
    """
    单井位机器学习建模函数，用于并行处理
    返回: (well_name, model_type, result_dict)
    """
    result = {
        'well_name': well_name,
        'well_index': well_index,
        'model_type': model_type.upper(),
        'success': False,
        'error_msg': None,
        'train_metrics': {},
        'test_metrics': {},
        'model_path': None,
        'data_info': {}
    }
    
    try:
        # 检查数据有效性
        if "日期" in well_data.index.names or hasattr(well_data.index, 'name'):
            ser = well_data.values.astype("float32").reshape(-1, 1)
            original_index = well_data.index
        else:
            ser = well_data.astype("float32").reshape(-1, 1)
            original_index = None
            
        if np.isnan(ser).all() or len(ser[~np.isnan(ser)]) < cfg.window_size + cfg.pred_len + 10:
            result['error_msg'] = f"井位 {well_name} 数据不足或全为空值"
            return well_name, model_type, result
        
        # 归一化
        scaler = MinMaxScaler()
        ser_s = scaler.fit_transform(ser).flatten()
        
        # 滑动窗口 & 划分
        X, y = create_sliding_window(ser_s, cfg.window_size, cfg.pred_len)
        split_idx = int(len(X) * cfg.train_ratio)
        X_tr, y_tr = X[:split_idx], y[:split_idx]
        X_te, y_te = X[split_idx:], y[split_idx:]
        
        # 创建模型配置对象
        class ModelConfig:
            def __init__(self, model_type, pred_len):
                self.model = model_type
                self.pred_len = pred_len
        
        model_cfg = ModelConfig(model_type, cfg.pred_len)
        
        # 训练 & 预测
        model, (p_tr_inv, t_tr_inv, p_te_inv, t_te_inv) = fit_predict_ml(
            model_cfg, X_tr, y_tr, X_te, y_te, scaler
        )
        
        # 保存模型
        # 创建模型保存目录
        model_save_dir = f'results/multi_wells_ml'
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = f"{model_save_dir}/{model_type}_{well_name}.joblib"
        joblib.dump(model, model_path)
        result['model_path'] = model_path
        
        # 计算指标
        train_mse, train_mae, train_rmse, train_mape = metrics(t_tr_inv, p_tr_inv)
        test_mse, test_mae, test_rmse, test_mape = metrics(t_te_inv, p_te_inv)
        
        result['train_metrics'] = {
            'mse': float(train_mse),
            'mae': float(train_mae),
            'rmse': float(train_rmse),
            'mape': float(train_mape)
        }
        result['test_metrics'] = {
            'mse': float(test_mse),
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'mape': float(test_mape)
        }
        
        result['data_info'] = {
            'total_length': len(ser),
            'valid_count': int((~np.isnan(ser)).sum()),
            'train_size': len(X_tr),
            'test_size': len(X_te),
            'window_size': cfg.window_size,
            'pred_len': cfg.pred_len
        }
        
        # 保存预测数据用于可视化（只保存第一步预测）
        def first_step(arr):
            return arr[:, 0]
        
        pred_tr_1 = first_step(p_tr_inv)
        true_tr_1 = first_step(t_tr_inv)
        pred_te_1 = first_step(p_te_inv)
        true_te_1 = first_step(t_te_inv)
        
        # 生成时间索引
        if original_index is not None:
            # 使用原始时间索引
            train_start_idx = cfg.window_size
            train_end_idx = cfg.window_size + len(pred_tr_1)
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + len(pred_te_1)
            
            if len(original_index) > test_end_idx:
                train_index = original_index[train_start_idx:train_end_idx]
                test_index = original_index[test_start_idx:test_end_idx]
            else:
                # 如果索引长度不够，使用数值索引
                train_index = np.arange(train_start_idx, train_end_idx)
                test_index = np.arange(test_start_idx, test_end_idx)
        else:
            # 使用数值索引
            train_index = np.arange(cfg.window_size, cfg.window_size + len(pred_tr_1))
            test_index = np.arange(cfg.window_size + len(pred_tr_1), 
                                 cfg.window_size + len(pred_tr_1) + len(pred_te_1))
        
        result['pred_data'] = {
            'train_true': true_tr_1,
            'train_pred': pred_tr_1,
            'test_true': true_te_1,
            'test_pred': pred_te_1,
            'train_index': train_index,
            'test_index': test_index
        }
        
        result['success'] = True
        
    except Exception as e:
        result['error_msg'] = f"井位 {well_name} 建模过程出错: {str(e)}"
    
    return well_name, model_type, result


def fit_predict_ml(cfg, X_tr, y_tr, X_te, y_te, scaler):
    """XGBoost / LightGBM：多步预测 → MultiOutputRegressor 包装"""
    if cfg.model == "xgb":
        base = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )
    else:  # lgbm
        base = LGBMRegressor(
            n_estimators=800,
            max_depth=-1,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="l2",
            random_state=42,
        )
    model = MultiOutputRegressor(base)
    model.fit(X_tr, y_tr)

    # —— 预测 —— #
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    # —— 反归一化 —— #
    pred_tr_inv = scaler.inverse_transform(pred_tr.reshape(-1, 1)).reshape(
        -1, cfg.pred_len
    )
    true_tr_inv = scaler.inverse_transform(y_tr.reshape(-1, 1)).reshape(
        -1, cfg.pred_len
    )
    pred_te_inv = scaler.inverse_transform(pred_te.reshape(-1, 1)).reshape(
        -1, cfg.pred_len
    )
    true_te_inv = scaler.inverse_transform(y_te.reshape(-1, 1)).reshape(
        -1, cfg.pred_len
    )

    return model, (pred_tr_inv, true_tr_inv, pred_te_inv, true_te_inv)


# ----------------------------------------------------------------------
# 2. 多井位可视化函数
# ----------------------------------------------------------------------
def plot_individual_well_results(results, output_dir, show_wells=None, log=None):
    """为每口井的每个模型生成单独的预测结果图"""
    successful_results = [(name, model, res) for name, models in results.items() 
                         for model, res in models.items() if res['success']]
    
    if len(successful_results) == 0:
        if log:
            log.warning("❌ 没有成功建模的井位，跳过单井位图表生成")
        return
    
    # 如果指定了展示井位序号，只为指定井位生成图表
    if show_wells:
        # 根据井位序号过滤
        results_to_plot = []
        for well_name, model_type, result in successful_results:
            if result['well_index'] in show_wells:
                results_to_plot.append((well_name, model_type, result))
        
        if log:
            log.info(f"只为指定井位序号生成图表: {show_wells}")
            log.info(f"匹配到的结果: {len(results_to_plot)} 个")
    else:
        results_to_plot = successful_results
        if log:
            log.info(f"为所有成功结果生成图表，共 {len(results_to_plot)} 个")
    
    if len(results_to_plot) == 0:
        if log:
            log.warning("❌ 没有匹配的结果需要生成图表")
        return
    
    for well_name, model_type, result in results_to_plot:
        pred_data = result['pred_data']
        test_rmse = result['test_metrics']['rmse']
        test_mae = result['test_metrics']['mae']
        test_mape = result['test_metrics']['mape']
        
        # 原始序列预测图
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(pred_data['train_index'], pred_data['train_true'], 
                label="Train True", color='blue', alpha=0.7, lw=1)
        ax.plot(pred_data['train_index'], pred_data['train_pred'], 
                label="Train Pred", color='blue', alpha=0.7, lw=1, ls='--')
        ax.plot(pred_data['test_index'], pred_data['test_true'], 
                label="Test True", color='red', alpha=0.7, lw=1)
        ax.plot(pred_data['test_index'], pred_data['test_pred'], 
                label="Test Pred", color='red', alpha=0.7, lw=1, ls='--')
        
        ax.set_title(f"{well_name} - {model_type} 时序预测\nTest RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
        ax.set_ylabel("地下水位 (GWL)")
        ax.set_xlabel("日期" if hasattr(pred_data['train_index'], 'dtype') and 'datetime' in str(pred_data['train_index'].dtype) else "时间步")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 如果是时间索引，旋转x轴标签
        if hasattr(pred_data['train_index'], 'dtype') and 'datetime' in str(pred_data['train_index'].dtype):
            ax.tick_params(axis='x', rotation=45)
        
        # 保存图表
        pred_path = f"{output_dir}/pred_vs_true_{model_type}_{well_name}.png"
        fig.tight_layout()
        fig.savefig(pred_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if log:
            log.info(f"预测结果图已保存: {pred_path}")
    
    if log:
        log.info(f"单井位预测图表生成完成，共 {len(results_to_plot)} 个文件")


def plot_multi_wells_results(results, cfg, output_dir, log=None):
    """生成多井位预测结果的可视化图表"""
    
    # 解析要展示的井位序号
    show_wells = None
    if cfg.show_wells:
        show_wells = [int(x.strip()) for x in cfg.show_wells.split(',')]
    
    # 1. 为每口井的每个模型生成单独的预测图
    plot_individual_well_results(results, output_dir, show_wells, log)
    
    # 2. 为每个模型生成成功建模的井位概览图
    for model_type in ['xgb', 'lgbm']:
        successful_wells = [name for name, models in results.items() 
                           if model_type in models and models[model_type]['success']]
        
        if len(successful_wells) == 0:
            continue
        
        # 计算子图布局
        n_wells = len(successful_wells)
        n_cols = min(4, n_wells)
        n_rows = (n_wells + n_cols - 1) // n_cols
        
        # 创建多子图
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # 统一处理axes为二维数组格式
        if n_wells == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, well_name in enumerate(successful_wells):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            result = results[well_name][model_type]
            pred_data = result['pred_data']
            
            # 绘制训练集和测试集
            ax.plot(pred_data['train_index'], pred_data['train_true'], 
                    label='Train True', color='blue', alpha=0.7, linewidth=1)
            ax.plot(pred_data['train_index'], pred_data['train_pred'],
                    label='Train Pred', color='blue', alpha=0.7, linewidth=1, linestyle='--')
            ax.plot(pred_data['test_index'], pred_data['test_true'],
                    label='Test True', color='red', alpha=0.7, linewidth=1)
            ax.plot(pred_data['test_index'], pred_data['test_pred'],
                    label='Test Pred', color='red', alpha=0.7, linewidth=1, linestyle='--')
            
            # 设置标题和标签
            rmse = result['test_metrics']['rmse']
            ax.set_title(f"{well_name}\n{model_type.upper()}, RMSE={rmse:.3f}", fontsize=10)
            ax.set_xlabel('Date', fontsize=8)
            ax.set_ylabel('GWL', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
        
        # 隐藏多余的子图
        for idx in range(n_wells, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/multi_wells_{model_type}_overview.png", dpi=300, bbox_inches='tight')
        plt.close()


# ----------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------
def main(cfg):
    seed_everything(42)
    os.makedirs("results", exist_ok=True)
    log = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = f"results/multi_wells_ml"
    os.makedirs(output_dir, exist_ok=True)

    # —— 数据读取 —— #
    df = pd.read_excel(cfg.data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    # 选择井位列（跳过日期列）
    well_columns = df.columns[cfg.start_col-1:cfg.end_col] if cfg.end_col > 0 else df.columns[cfg.start_col-1:]
    
    log.info(f"开始处理 {len(well_columns)} 个井位: {list(well_columns)}")
    log.info(f"使用模型: XGBoost 和 LightGBM")
    
    # —— 并行处理多个井位和模型 —— #
    results = {}
    
    # 准备任务列表：每个井位 × 每个模型
    tasks = []
    for idx, col in enumerate(well_columns):
        well_index = cfg.start_col + idx
        for model_type in ['xgb', 'lgbm']:
            tasks.append((df[col], col, cfg, well_index, model_type))
    
    if cfg.parallel:
        # 并行处理
        log.info(f"使用 {cfg.n_jobs} 个进程并行处理 {len(tasks)} 个任务")
        
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as executor:
            # 提交任务
            future_to_task = {
                executor.submit(fit_single_well_ml, *task): task 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                well_name, model_type, result = future.result()
                
                if well_name not in results:
                    results[well_name] = {}
                results[well_name][model_type] = result
                
                if result['success']:
                    log.info(f"✅ {well_name} - {model_type.upper()}: "
                           f"Test RMSE={result['test_metrics']['rmse']:.4f}")
                else:
                    log.warning(f"❌ {well_name} - {model_type.upper()}: {result['error_msg']}")
    else:
        # 串行处理
        log.info(f"串行处理模式，处理 {len(tasks)} 个任务")
        for task in tasks:
            well_name, model_type, result = fit_single_well_ml(*task)
            
            if well_name not in results:
                results[well_name] = {}
            results[well_name][model_type] = result
            
            if result['success']:
                log.info(f"✅ {well_name} - {model_type.upper()}: "
                       f"Test RMSE={result['test_metrics']['rmse']:.4f}")
            else:
                log.warning(f"❌ {well_name} - {model_type.upper()}: {result['error_msg']}")
    
    # —— 结果汇总 —— #
    total_tasks = len(well_columns) * 2  # 每个井位 × 2个模型
    successful_count = sum(1 for models in results.values() 
                          for res in models.values() if res['success'])
    log.info(f"\n建模完成: {successful_count}/{total_tasks} 个任务成功")
    
    # 保存详细结果到JSON
    summary_results = {}
    for well_name, models in results.items():
        summary_results[well_name] = {}
        for model_type, result in models.items():
            summary_results[well_name][model_type] = {
                'success': result['success'],
                'error_msg': result['error_msg'],
                'model_type': result['model_type'],
                'well_index': result['well_index'],
                'train_metrics': result['train_metrics'],
                'test_metrics': result['test_metrics'],
                'data_info': result['data_info'],
                'model_path': result['model_path']
            }
    
    with open(f"{output_dir}/multi_wells_ml_results.json", 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False, default=str)
    
    # —— 生成汇总表格 —— #
    summary_df = []
    for well_name, models in results.items():
        for model_type, result in models.items():
            if result['success']:
                summary_df.append({
                    '井位': well_name,
                    '井位序号': result['well_index'],
                    '模型类型': result['model_type'],
                    '训练样本数': result['data_info']['train_size'],
                    '测试样本数': result['data_info']['test_size'],
                    'Train_RMSE': f"{result['train_metrics']['rmse']:.4f}",
                    'Train_MAE': f"{result['train_metrics']['mae']:.4f}",
                    'Train_MAPE(%)': f"{result['train_metrics']['mape']:.2f}",
                    'Test_RMSE': f"{result['test_metrics']['rmse']:.4f}",
                    'Test_MAE': f"{result['test_metrics']['mae']:.4f}",
                    'Test_MAPE(%)': f"{result['test_metrics']['mape']:.2f}"
                })
            else:
                summary_df.append({
                    '井位': well_name,
                    '井位序号': result['well_index'],
                    '模型类型': result['model_type'],
                    '训练样本数': 'FAILED',
                    '测试样本数': 'FAILED',
                    'Train_RMSE': '-',
                    'Train_MAE': '-',
                    'Train_MAPE(%)': '-',
                    'Test_RMSE': '-',
                    'Test_MAE': '-',
                    'Test_MAPE(%)': '-'
                })
    
    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(f"{output_dir}/multi_wells_ml_summary.csv", index=False, encoding='utf-8-sig')
    summary_df.to_excel(f"{output_dir}/multi_wells_ml_summary.xlsx", index=False)
    
    log.info(f"汇总结果已保存到: {output_dir}/multi_wells_ml_summary.xlsx")
    
    # —— 可视化 —— #
    log.info("生成可视化图表...")
    plot_multi_wells_results(results, cfg, output_dir, log)
    log.info(f"可视化图表已保存到: {output_dir}/")
    
    # 打印最终统计
    if successful_count > 0:
        # 按模型类型统计
        for model_type in ['xgb', 'lgbm']:
            model_results = [res for models in results.values() 
                           for mt, res in models.items() 
                           if mt == model_type and res['success']]
            
            if len(model_results) > 0:
                avg_rmse = np.mean([res['test_metrics']['rmse'] for res in model_results])
                avg_mae = np.mean([res['test_metrics']['mae'] for res in model_results])
                avg_mape = np.mean([res['test_metrics']['mape'] for res in model_results])
                
                log.info(f"\n=== {model_type.upper()} 模型统计 ===")
                log.info(f"成功建模井位数: {len(model_results)}/{len(well_columns)}")
                log.info(f"平均测试RMSE: {avg_rmse:.4f}")
                log.info(f"平均测试MAE: {avg_mae:.4f}")
                log.info(f"平均测试MAPE: {avg_mape:.2f}%")


# ----------------------------------------------------------------------
# 4. 命令行入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="多井位机器学习时序预测建模")
    parser.add_argument("--data_path", type=str,
                        default="database/ZoupingCounty_gwl_filled.xlsx",
                        help="数据文件路径（Excel/CSV 都可）")
    parser.add_argument("--start_col", type=int, default=2,
                        help="起始井位列序号（从 2 开始，跳过日期列）")
    parser.add_argument("--end_col", type=int, default=4,
                        help="结束井位列序号（-1表示到最后一列）")
    parser.add_argument("--window_size", type=int, default=24,
                        help="滑动窗口大小")
    parser.add_argument("--pred_len", type=int, default=4,
                        help="预测步长")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--parallel", action="store_true",
                        help="是否使用并行处理")
    parser.add_argument("--n_jobs", type=int, default=mp.cpu_count()//2,
                        help="并行处理的进程数")
    parser.add_argument("--show_wells", type=str, default=None,
                        help="指定要展示预测图的井位序号，用逗号分隔，如 '2,4,6'；不指定则为所有井位生成图表")
    
    cfg = parser.parse_args()
    main(cfg)