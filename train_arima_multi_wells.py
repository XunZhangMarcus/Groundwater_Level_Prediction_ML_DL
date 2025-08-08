#!/usr/bin/env python
# train_arima_multi_wells.py  ——  多井位ARIMA同时建模预测
import argparse, os, logging, warnings, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import json
import pickle

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 0. 工具函数
# ----------------------------------------------------------------------
def adf_pvalue(series):
    """ADF 检验 p 值（越小越稳）"""
    try:
        return adfuller(series.dropna())[1]
    except:
        return 1.0  # 如果检验失败，返回不显著


def ljung_box_pvalue(series, lags=10):
    """Ljung-Box p 值（>0.05 说明近白噪声，建模意义不大）"""
    try:
        clean_series = series.dropna()
        if len(clean_series) <= lags:
            return 1.0
        return acorr_ljungbox(clean_series, lags=[min(lags, len(clean_series)//4)],
                              return_df=True)["lb_pvalue"].iloc[0]
    except:
        return 1.0


def make_stationary(series, max_d=2, lags=10):
    """
    迭代差分，直到：平稳(ADF≤0.05) 且 非白噪声(Ljung-Box≤0.05)
    返回差分阶 d；若失败返回 None
    """
    tmp = series.copy()
    for d in range(max_d + 1):        # d = 0,1,2,...
        if len(tmp.dropna()) < 10:    # 数据太少
            return None
        adf_p = adf_pvalue(tmp)
        lb_p  = ljung_box_pvalue(tmp, lags=lags)
        
        if (adf_p <= .05) and (lb_p <= .05):
            return d
        if d == max_d:
            break
        tmp = tmp.diff().dropna()
    return None


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


# ----------------------------------------------------------------------
# 1. 单井位ARIMA建模函数
# ----------------------------------------------------------------------
def fit_single_well_arima(well_data, well_name, cfg, well_index):
    """
    单井位ARIMA建模函数，用于并行处理
    返回: (well_name, result_dict)
    """
    result = {
        'well_name': well_name,
        'well_index': well_index,
        'success': False,
        'error_msg': None,
        'best_order': None,
        'best_aic': None,
        'diff_order': None,
        'train_metrics': {},
        'test_metrics': {},
        'forecast': None,
        'train_data': None,
        'test_data': None
    }
    
    try:
        # 检查数据有效性
        ser_raw = well_data.astype("float32")
        if ser_raw.isna().all() or len(ser_raw.dropna()) < 20:
            result['error_msg'] = f"井位 {well_name} 数据不足或全为空值"
            return well_name, result
        
        # —— 自动判定差分阶 d —— #
        d = make_stationary(ser_raw, max_d=cfg.max_d, lags=10)
        if d is None:
            result['error_msg'] = f"井位 {well_name} 序列经最多差分仍不平稳或呈白噪声"
            return well_name, result
        
        result['diff_order'] = d
        
        # —— 划分训练 / 测试（原始序列） —— #
        train_size = int(len(ser_raw) * cfg.train_ratio)
        train_raw = ser_raw.iloc[:train_size]
        test_raw = ser_raw.iloc[train_size:]
        
        # —— 网格搜索 (p,q) —— #
        best_aic, best_order = np.inf, None
        for p in range(cfg.max_p + 1):
            for q in range(cfg.max_q + 1):
                try:
                    model = sm.tsa.ARIMA(train_raw, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic, best_order = model.aic, (p, d, q)
                except Exception:
                    continue
        
        if best_order is None:
            result['error_msg'] = f"井位 {well_name} 未找到可行的 ARIMA(p,d,q) 模型"
            return well_name, result
        
        result['best_order'] = best_order
        result['best_aic'] = best_aic
        
        # —— 最终拟合 & 预测（原始尺度） —— #
        best_model = sm.tsa.ARIMA(train_raw, order=best_order).fit()
        
        # 保存模型
        output_dir = f"results/multi_wells_arima"
        os.makedirs(output_dir, exist_ok=True)
        model_path = f"{output_dir}/arima_model_{well_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        result['model_path'] = model_path
        
        # 训练集预测（用于评估拟合效果）
        train_pred = best_model.fittedvalues
        
        # 测试集预测
        forecast_raw = best_model.get_forecast(steps=len(test_raw)).predicted_mean
        forecast_raw.index = test_raw.index
        
        # —— 评估指标 —— #
        train_rmse, train_mae, train_mape = metrics(train_raw.values, train_pred.values)
        test_rmse, test_mae, test_mape = metrics(test_raw.values, forecast_raw.values)
        
        result['train_metrics'] = {
            'rmse': train_rmse,
            'mae': train_mae, 
            'mape': train_mape
        }
        result['test_metrics'] = {
            'rmse': test_rmse,
            'mae': test_mae,
            'mape': test_mape
        }
        
        # 保存预测结果用于可视化
        result['forecast'] = forecast_raw
        result['train_data'] = train_raw
        result['test_data'] = test_raw
        result['train_pred'] = train_pred
        result['success'] = True
        
    except Exception as e:
        result['error_msg'] = f"井位 {well_name} 建模过程出错: {str(e)}"
    
    return well_name, result


# ----------------------------------------------------------------------
# 2. 多井位可视化函数
# ----------------------------------------------------------------------
def plot_individual_well_results(results, output_dir, show_wells=None, log=None):
    """为每口井生成单独的预测结果图"""
    successful_wells = [name for name, res in results.items() if res['success']]
    
    if len(successful_wells) == 0:
        if log:
            log.warning("❌ 没有成功建模的井位，跳过单井位图表生成")
        return
    
    # 如果指定了展示井位序号，只为指定井位生成图表
    if show_wells:
        # 根据井位序号过滤
        wells_to_plot = []
        for well_name, result in results.items():
            if result['success'] and result['well_index'] in show_wells:
                wells_to_plot.append(well_name)
        
        if log:
            log.info(f"只为指定井位序号生成图表: {show_wells}")
            log.info(f"匹配到的井位: {wells_to_plot}")
    else:
        wells_to_plot = successful_wells
        if log:
            log.info(f"为所有成功井位生成图表，共 {len(wells_to_plot)} 个")
    
    if len(wells_to_plot) == 0:
        if log:
            log.warning("❌ 没有匹配的井位需要生成图表")
        return
    
    for well_name in wells_to_plot:
        result = results[well_name]
        train_raw = result['train_data']
        test_raw = result['test_data']
        train_pred = result['train_pred']
        forecast_raw = result['forecast']
        best_order = result['best_order']
        test_rmse = result['test_metrics']['rmse']
        test_mae = result['test_metrics']['mae']
        test_mape = result['test_metrics']['mape']
        
        # 原始序列预测图 - 使用你提供的代码模板
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(train_raw.index, train_raw.values, label="Train True", color='blue', alpha=0.7, lw=1)
        ax.plot(train_pred.index, train_pred.values, label="Train Pred", color='blue', alpha=0.7, lw=1, ls='--')
        ax.plot(test_raw.index, test_raw.values, label="Test True", color='red', alpha=0.7, lw=1)
        ax.plot(forecast_raw.index, forecast_raw.values, label="Test Pred", color='red', alpha=0.7, lw=1, ls='--')
        ax.set_title(f"{well_name} - ARIMA{best_order} 时序预测\nTest RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
        ax.set_ylabel("地下水位 (GWL)")
        ax.set_xlabel("日期")
        ax.legend()
        ax.grid(True, alpha=0.3)
        pred_path = f"{output_dir}/pred_vs_true_ARIMA_{well_name}.png"
        fig.tight_layout()
        fig.savefig(pred_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if log:
            log.info(f"预测结果图已保存: {pred_path}")
    
    if log:
        log.info(f"单井位预测图表生成完成，共 {len(wells_to_plot)} 个文件")


def plot_multi_wells_results(results, cfg, output_dir, log=None):
    """生成多井位预测结果的可视化图表"""
    
    # 解析要展示的井位序号
    show_wells = None
    if cfg.show_wells:
        show_wells = [int(x.strip()) for x in cfg.show_wells.split(',')]
    
    # 1. 为每口井生成单独的预测图
    plot_individual_well_results(results, output_dir, show_wells, log)
    
    # 2. 成功建模的井位概览图
    successful_wells = [name for name, res in results.items() if res['success']]
    
    if len(successful_wells) == 0:
        print("❌ 没有成功建模的井位，跳过可视化")
        return
    
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
        
        result = results[well_name]
        train_data = result['train_data']
        test_data = result['test_data']
        train_pred = result['train_pred']
        forecast = result['forecast']
        
        # 绘制训练集
        ax.plot(train_data.index, train_data.values, 
                label='Train True', color='blue', alpha=0.7, linewidth=1)
        ax.plot(train_pred.index, train_pred.values,
                label='Train Pred', color='blue', alpha=0.7, linewidth=1, linestyle='--')
        
        # 绘制测试集
        ax.plot(test_data.index, test_data.values,
                label='Test True', color='red', alpha=0.7, linewidth=1)
        ax.plot(forecast.index, forecast.values,
                label='Test Pred', color='red', alpha=0.7, linewidth=1, linestyle='--')
        
        # 设置标题和标签
        order_str = f"ARIMA{result['best_order']}"
        rmse = result['test_metrics']['rmse']
        ax.set_title(f"{well_name}\n{order_str}, RMSE={rmse:.3f}", fontsize=10)
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
    plt.savefig(f"{output_dir}/multi_wells_arima_overview.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_heatmap(results, output_dir):
    """生成指标汇总热力图"""
    successful_wells = [name for name, res in results.items() if res['success']]
    
    if len(successful_wells) == 0:
        return
    
    # 准备数据
    metrics_data = []
    well_names = []
    
    for well_name in successful_wells:
        result = results[well_name]
        metrics_data.append([
            result['test_metrics']['rmse'],
            result['test_metrics']['mae'], 
            result['test_metrics']['mape'],
            result['best_aic']
        ])
        well_names.append(well_name)
    
    metrics_df = pd.DataFrame(metrics_data, 
                             index=well_names,
                             columns=['RMSE', 'MAE', 'MAPE(%)', 'AIC'])
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(8, max(6, len(successful_wells)*0.4)))
    
    # 标准化数据用于颜色映射（每列独立标准化）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(metrics_df.values)
    
    im = ax.imshow(normalized_data, cmap='RdYlBu_r', aspect='auto')
    
    # 设置刻度和标签
    ax.set_xticks(range(len(metrics_df.columns)))
    ax.set_xticklabels(metrics_df.columns)
    ax.set_yticks(range(len(metrics_df.index)))
    ax.set_yticklabels(metrics_df.index)
    
    # 添加数值标注
    for i in range(len(metrics_df.index)):
        for j in range(len(metrics_df.columns)):
            value = metrics_df.iloc[i, j]
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                   color='white' if abs(normalized_data[i, j]) > 1 else 'black',
                   fontsize=8)
    
    ax.set_title('多井位ARIMA模型性能指标热力图')
    plt.colorbar(im, ax=ax, label='标准化值')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_wells_metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


# ----------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------
def main(cfg):
    os.makedirs("results", exist_ok=True)
    log = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = f"results/multi_wells_arima"
    os.makedirs(output_dir, exist_ok=True)
    
    # —— 数据读取 —— #
    df = pd.read_excel(cfg.data_path)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
    
    # 选择井位列（跳过日期列）
    well_columns = df.columns[cfg.start_col-1:cfg.end_col] if cfg.end_col > 0 else df.columns[cfg.start_col-1:]
    
    log.info(f"开始处理 {len(well_columns)} 个井位: {list(well_columns)}")
    
    # —— 并行处理多个井位 —— #
    results = {}
    
    if cfg.parallel:
        # 并行处理
        log.info(f"使用 {cfg.n_jobs} 个进程并行处理")
        
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as executor:
            # 提交任务
            future_to_well = {
                executor.submit(fit_single_well_arima, df[col], col, cfg, cfg.start_col + idx): (col, idx)
                for idx, col in enumerate(well_columns)
            }
            
            # 收集结果
            for future in as_completed(future_to_well):
                well_name, result = future.result()
                results[well_name] = result
                
                if result['success']:
                    log.info(f"✅ {well_name}: ARIMA{result['best_order']}, "
                           f"Test RMSE={result['test_metrics']['rmse']:.4f}")
                else:
                    log.warning(f"❌ {well_name}: {result['error_msg']}")
    else:
        # 串行处理
        log.info("串行处理模式")
        for idx, col in enumerate(well_columns):
            well_index = cfg.start_col + idx
            well_name, result = fit_single_well_arima(df[col], col, cfg, well_index)
            results[well_name] = result
            
            if result['success']:
                log.info(f"✅ {well_name}: ARIMA{result['best_order']}, "
                       f"Test RMSE={result['test_metrics']['rmse']:.4f}")
            else:
                log.warning(f"❌ {well_name}: {result['error_msg']}")
    
    # —— 结果汇总 —— #
    successful_count = sum(1 for res in results.values() if res['success'])
    log.info(f"\n建模完成: {successful_count}/{len(well_columns)} 个井位成功")
    
    # 保存详细结果到JSON
    summary_results = {}
    for well_name, result in results.items():
        summary_results[well_name] = {
            'success': result['success'],
            'error_msg': result['error_msg'],
            'best_order': result['best_order'],
            'best_aic': result['best_aic'],
            'diff_order': result['diff_order'],
            'model_path': result.get('model_path', None),
            'train_metrics': result['train_metrics'],
            'test_metrics': result['test_metrics']
        }
    
    with open(f"{output_dir}/multi_wells_arima_results.json", 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 生成汇总表格
    summary_df = []
    for well_name, result in results.items():
        if result['success']:
            summary_df.append({
                '井位': well_name,
                'ARIMA阶次': str(result['best_order']),
                '差分阶数': result['diff_order'],
                'AIC': f"{result['best_aic']:.2f}",
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
                'ARIMA阶次': 'FAILED',
                '差分阶数': '-',
                'AIC': '-',
                'Train_RMSE': '-',
                'Train_MAE': '-', 
                'Train_MAPE(%)': '-',
                'Test_RMSE': '-',
                'Test_MAE': '-',
                'Test_MAPE(%)': '-'
            })
    
    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(f"{output_dir}/multi_wells_arima_summary.csv", index=False, encoding='utf-8-sig')
    summary_df.to_excel(f"{output_dir}/multi_wells_arima_summary.xlsx", index=False)
    
    log.info(f"汇总结果已保存到: {output_dir}/multi_wells_arima_summary.xlsx")
    
    # —— 可视化 —— #
    log.info("生成可视化图表...")
    plot_multi_wells_results(results, cfg, output_dir, log)
    log.info(f"可视化图表已保存到: {output_dir}/")
    
    # 打印最终统计
    if successful_count > 0:
        successful_results = [res for res in results.values() if res['success']]
        avg_rmse = np.mean([res['test_metrics']['rmse'] for res in successful_results])
        avg_mae = np.mean([res['test_metrics']['mae'] for res in successful_results])
        avg_mape = np.mean([res['test_metrics']['mape'] for res in successful_results])
        
        log.info(f"\n=== 多井位ARIMA建模汇总统计 ===")
        log.info(f"成功建模井位数: {successful_count}/{len(well_columns)}")
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
    
    parser = argparse.ArgumentParser(description="多井位ARIMA时序预测建模")
    parser.add_argument("--data_path", type=str,
                        default="database/ZoupingCounty_gwl_filled.xlsx",
                        help="数据文件路径（Excel/CSV 都可）")
    parser.add_argument("--start_col", type=int, default=2,
                        help="起始井位列序号（从 2 开始，跳过日期列）")
    parser.add_argument("--end_col", type=int, default=3,
                        help="结束井位列序号（-1表示到最后一列）")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--max_p", type=int, default=4, help="AR 最大阶")
    parser.add_argument("--max_q", type=int, default=4, help="MA 最大阶")
    parser.add_argument("--max_d", type=int, default=2, help="自动差分最大阶")
    parser.add_argument("--parallel", action="store_true",
                        help="是否使用并行处理")
    parser.add_argument("--n_jobs", type=int, default=mp.cpu_count()//2,
                        help="并行处理的进程数")
    parser.add_argument("--show_wells", type=str, default=None,
                        help="指定要展示预测图的井位序号，用逗号分隔，如 '2,4,6'；不指定则为所有井位生成图表")
    
    cfg = parser.parse_args()
    main(cfg)