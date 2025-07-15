#!/bin/bash
# 批量运行所有井位的ARIMA模型训练

DATA_PATH="database/ZoupingCounty_gwl_filled.xlsx"
RESULTS_DIR="results"
TRAIN_RATIO=0.8
MAX_P=4
MAX_Q=4
MAX_D=2

# 获取井名列表（假设用python读取列名）
well_names=( $(python -c "import pandas as pd; df = pd.read_excel('$DATA_PATH'); print(' '.join([str(c) for c in df.columns[1:24]]))") )

for ((i=1; i<=23; i++)); do
  well_name=${well_names[$((i-1))]}
  OUTDIR="$RESULTS_DIR/ARIMA_${well_name}"
  mkdir -p "$OUTDIR"
  python train_arima_timeseries.py \
    --data_path "$DATA_PATH" \
    --well_col $((i+1)) \
    --train_ratio $TRAIN_RATIO \
    --max_p $MAX_P \
    --max_q $MAX_Q \
    --max_d $MAX_D
  # 移动结果到对应文件夹
  mv results/pred_vs_true_ARIMA_col$((i+1))_orig.png "$OUTDIR/" 2>/dev/null
  mv results/pred_vs_true_ARIMA_col$((i+1))_diff.png "$OUTDIR/" 2>/dev/null
done

echo "所有井位ARIMA模型训练完成，结果已分类保存。" 