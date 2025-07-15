#!/bin/bash
# 批量运行所有井位和所有ML模型的训练

DATA_PATH="database/ZoupingCounty_gwl_filled.xlsx"
RESULTS_DIR="results"
WINDOW_SIZE=24
PRED_LEN=4
TRAIN_RATIO=0.8

# 获取井名列表（假设用python读取列名）
well_names=( $(python -c "import pandas as pd; df = pd.read_excel('$DATA_PATH'); print(' '.join([str(c) for c in df.columns[1:24]]))") )

for ((i=1; i<=23; i++)); do
  well_name=${well_names[$((i-1))]}
  for model in xgb lgbm; do
    OUTDIR="$RESULTS_DIR/${model}_${well_name}"
    mkdir -p "$OUTDIR"
    python train_perdict_ml_timeseries.py \
      --data_path "$DATA_PATH" \
      --well_col $i \
      --window_size $WINDOW_SIZE \
      --pred_len $PRED_LEN \
      --train_ratio $TRAIN_RATIO \
      --model $model
    # 移动结果到对应文件夹
    mv results/${model}_${well_name}.joblib "$OUTDIR/" 2>/dev/null
    mv results/pred_vs_true_${model}_*.png "$OUTDIR/" 2>/dev/null
  done
done

echo "所有井位和ML模型训练完成，结果已分类保存。"
