#!/bin/bash

export USE_CAUSAL=false


LR_LIST=(5e-4 1e-3)
BS_LIST=(256 512)

GPU=7

for LR in "${LR_LIST[@]}"
do
  for BS in "${BS_LIST[@]}"
  do
    echo "Running lr=$LR bs=$BS"

    export WANDB_NAME="qwen_v3_synth_u_lr${LR}_bs${BS}"

    CUDA_VISIBLE_DEVICES=$GPU python run_qwen_v3.py \
        --cond_modal text \
        --training_stage finetune \
        --save_folder ./sweep/synth_u_qwen_v3/lr_${LR}_bs_${BS} \
        --model_diff_config_path configs/synth_u_qwen_v3/diff/model_text2ts_dep.yaml \
        --model_cond_config_path configs/synth_u_qwen_v3/cond/text_msmdiffmv.yaml \
        --train_config_path configs/synth_u_qwen_v3/train.yaml \
        --evaluate_config_path configs/synth_u_qwen_v3/evaluate.yaml \
        --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
        --multipatch_num 3 \
        --L_patch_len 2 \
        --base_patch 4 \
        --epochs 2500 \
        --batch_size $BS \
        --lr $LR \
        --samples_name "real_text_samples.pt" \
        --model_ckpt_name "model_best_loss.pth" \
        --n_runs 1
  done
done