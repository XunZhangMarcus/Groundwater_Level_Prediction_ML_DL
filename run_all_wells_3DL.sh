#!/bin/bash
# 批量运行所有井位和所有模型的训练

DATA_PATH="database/ZoupingCounty_gwl_filled.xlsx"
RESULTS_DIR="results"
WINDOW_SIZE=24
PRED_LEN=4
TRAIN_RATIO=0.8
BATCH_SIZE=16
EPOCHS=200
LR=1e-3
ALPHA=0.5

# 获取井名列表（假设用python读取列名）
well_names=( $(python -c "import pandas as pd; df = pd.read_excel('$DATA_PATH'); print(' '.join([str(c) for c in df.columns[1:24]]))") )

for ((i=1; i<=23; i++)); do
  well_name=${well_names[$((i-1))]}
  for model in gru lstm transformer; do
    OUTDIR="$RESULTS_DIR/${model}_${well_name}"
    mkdir -p "$OUTDIR"
    python train_predict_3DL.py \
      --data_path "$DATA_PATH" \
      --well_col $i \
      --window_size $WINDOW_SIZE \
      --pred_len $PRED_LEN \
      --train_ratio $TRAIN_RATIO \
      --batch_size $BATCH_SIZE \
      --epochs $EPOCHS \
      --lr $LR \
      --model $model \
      --alpha $ALPHA
    # 移动结果到对应文件夹
    mv results/best_${model}_*.pt "$OUTDIR/" 2>/dev/null
    mv results/loss_curve_${model}_*.png "$OUTDIR/" 2>/dev/null
    mv results/pred_vs_true_${model}_*.png "$OUTDIR/" 2>/dev/null
    mv results/metrics.txt "$OUTDIR/metrics_${model}_${well_name}.txt" 2>/dev/null
  done
done

echo "所有井位和模型训练完成，结果已分类保存。"

