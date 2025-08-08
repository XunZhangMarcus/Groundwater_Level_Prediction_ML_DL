"""
GWL_data.xlsx 数据处理脚本
将不同井号的水位时序数据处理成单独的Excel文件，每个井位一个文件
每个井位的数据只包含该井位有数据的时间段，保留水位空值
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

def process_gwl_data():
    """
    处理GWL_data.xlsx文件，将每个井位的数据单独保存到不同的Excel文件中
    每个井位的数据只包含该井位有数据的时间段，保留水位空值
    数据格式：井号 | 地址 | 水位 | 时间
    """
    
    # 文件路径
    input_file = "./database/GWL_data.xlsx"
    output_dir = Path("./database/individual_wells")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("开始处理GWL_data.xlsx文件...")
    print(f"输出目录: {output_dir}")
    
    # 读取Excel文件
    try:
        df_raw = pd.read_excel(input_file, header=0)  # 第一行作为表头
        print(f"✅ 成功读取文件: {input_file}")
        print(f"原始数据形状: {df_raw.shape}")
        print(f"列名: {df_raw.columns.tolist()}")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 分析数据结构
    print("\n📊 数据结构分析:")
    print("前10行数据:")
    print(df_raw.head(10))
    
    # 检查列名，根据实际情况调整
    expected_columns = ['井号', '地址', '水位', '时间']
    if list(df_raw.columns) != expected_columns:
        print(f"⚠️  列名不匹配，实际列名: {df_raw.columns.tolist()}")
        # 重命名列
        if len(df_raw.columns) == 4:
            df_raw.columns = expected_columns
            print(f"✅ 已重命名列为: {expected_columns}")
    
    # 处理时间列
    print("\n🕐 处理时间数据...")
    
    # 尝试解析时间
    def parse_datetime_flexible(time_series):
        """灵活解析时间格式"""
        parsed_times = []
        for time_val in time_series:
            if pd.isna(time_val):
                parsed_times.append(pd.NaT)
                continue
                
            time_str = str(time_val).strip()
            
            # 尝试多种格式
            formats_to_try = [
                "%d/%m/%Y %H:%M:%S",  # 30/8/2017 00:00:00
                "%Y-%m-%d %H:%M:%S",  # 2017-08-30 00:00:00
                "%d/%m/%Y",           # 30/8/2017
                "%Y-%m-%d",           # 2017-08-30
            ]
            
            parsed = None
            for fmt in formats_to_try:
                try:
                    parsed = pd.to_datetime(time_str, format=fmt)
                    break
                except:
                    continue
            
            if parsed is None:
                try:
                    parsed = pd.to_datetime(time_str, errors='coerce')
                except:
                    parsed = pd.NaT
            
            parsed_times.append(parsed)
        
        return pd.Series(parsed_times)
    
    df_raw['时间'] = parse_datetime_flexible(df_raw['时间'])
    
    # 移除时间解析失败的行
    valid_time_mask = df_raw['时间'].notna()
    df_clean = df_raw[valid_time_mask].copy()
    
    print(f"时间解析成功率: {valid_time_mask.sum()}/{len(df_raw)} ({valid_time_mask.mean()*100:.1f}%)")
    
    # 转换水位为数值（保留空值）
    df_clean['水位'] = pd.to_numeric(df_clean['水位'], errors='coerce')
    
    print(f"水位数据转换完成，保留空值: {df_clean['水位'].isna().sum()} 个空值")
    
    # 按时间排序
    df_clean = df_clean.sort_values('时间').reset_index(drop=True)
    
    print(f"\n📈 清洗后数据形状: {df_clean.shape}")
    print(f"时间范围: {df_clean['时间'].min()} 到 {df_clean['时间'].max()}")
    
    # 创建井位标识符（井号-地址）
    df_clean['井位标识'] = df_clean['井号'].astype(str) + '-' + df_clean['地址'].astype(str)
    
    # 分析井位信息
    print("\n🏗️ 分析井位信息...")
    well_info = df_clean.groupby(['井号', '地址', '井位标识']).agg({
        '水位': ['count', 'min', 'max', 'mean'],
        '时间': ['min', 'max']
    }).round(2)
    
    well_info.columns = ['记录数', '最小水位', '最大水位', '平均水位', '开始时间', '结束时间']
    well_info = well_info.reset_index()
    
    print(f"发现 {len(well_info)} 个不同的井位")
    
    # 为每个井位单独保存数据
    print("\n💾 为每个井位单独保存数据...")
    
    saved_files = []
    well_summary = []
    
    for idx, (well_id, well_data) in enumerate(df_clean.groupby('井位标识')):
        # 获取井位信息
        well_row = well_info[well_info['井位标识'] == well_id].iloc[0]
        well_code = well_row['井号']
        well_address = well_row['地址']
        
        # 按时间排序
        well_data = well_data.sort_values('时间').reset_index(drop=True)
        
        # 创建该井位的时序数据
        well_timeseries = well_data[['时间', '水位']].copy()
        well_timeseries.rename(columns={'时间': '时间戳', '水位': f'井位_{well_code}_水位'}, inplace=True)
        
        # 生成安全的文件名（移除特殊字符）
        safe_filename = well_id.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        filename = f"well_{safe_filename}.xlsx"
        filepath = output_dir / filename
        
        # 保存文件为Excel格式
        try:
            well_timeseries.to_excel(filepath, index=False, engine='openpyxl')
            saved_files.append(filename)
            
            # 记录摘要信息
            well_summary.append({
                '文件名': filename,
                '井号': well_code,
                '地址': well_address,
                '记录数': len(well_timeseries),
                '开始时间': well_timeseries['时间戳'].min(),
                '结束时间': well_timeseries['时间戳'].max(),
                '最小水位': well_timeseries[f'井位_{well_code}_水位'].min(),
                '最大水位': well_timeseries[f'井位_{well_code}_水位'].max(),
                '平均水位': well_timeseries[f'井位_{well_code}_水位'].mean().round(2),
                '时间跨度_天': (well_timeseries['时间戳'].max() - well_timeseries['时间戳'].min()).days
            })
            
            if (idx + 1) % 20 == 0:  # 每处理20个井位显示一次进度
                print(f"已处理 {idx + 1}/{len(df_clean.groupby('井位标识'))} 个井位...")
                
        except Exception as e:
            print(f"❌ 保存井位 {well_id} 数据失败: {e}")
            continue
    
    print(f"\n✅ 成功保存 {len(saved_files)} 个井位的数据文件")
    
    # 创建总体摘要
    summary_df = pd.DataFrame(well_summary)
    summary_file = output_dir / "wells_summary.xlsx"
    summary_df.to_excel(summary_file, index=False, engine='openpyxl')
    
    # 保存井位详细信息
    detail_file = output_dir / "wells_detailed_info.xlsx"
    well_info.to_excel(detail_file, index=False, engine='openpyxl')
    
    print(f"✅ 井位摘要已保存到: {summary_file}")
    print(f"✅ 井位详细信息已保存到: {detail_file}")
    
    # 显示处理结果统计
    print("\n📊 处理结果统计:")
    print(f"- 总井位数: {len(saved_files)}")
    print(f"- 平均记录数: {summary_df['记录数'].mean():.0f}")
    print(f"- 最长时间跨度: {summary_df['时间跨度_天'].max()} 天")
    print(f"- 最短时间跨度: {summary_df['时间跨度_天'].min()} 天")
    
    # 显示前10个井位的摘要
    print("\n📋 前10个井位摘要:")
    print(summary_df.head(10)[['文件名', '井号', '记录数', '开始时间', '结束时间', '时间跨度_天']])
    
    return summary_df, well_info

if __name__ == "__main__":
    # 执行数据处理
    summary_data, detail_data = process_gwl_data()
    
    print("\n🎉 数据处理完成！")
    print("输出文件:")
    print("- database/individual_wells/: 包含所有井位的单独Excel文件")
    print("- wells_summary.xlsx: 井位摘要信息")
    print("- wells_detailed_info.xlsx: 井位详细信息")