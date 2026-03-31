#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# =========================
# 路径
# =========================

# 👉 必须是真实数据（GT）
REAL_PATH="/playpen-shared/haochenz/sweep_text2ts_v6/synth_u/verbalts_v6/0/real_text_samples_model_epoch_199.pt"
SAMPLE_DIR="/playpen-shared/haochenz/sweep_text2ts_v6/synth_u/verbalts_v6/0"

SAVE_FILE="./fid_results/verbalts_v6.txt"

mkdir -p ./fid_results

echo "FID Results" > ${SAVE_FILE}
echo "==========================" >> ${SAVE_FILE}

# =========================
# 遍历所有 ckpt samples
# =========================

for sample_path in ${SAMPLE_DIR}/real_text_samples_model_epoch_*.pt
do
    sample_name=$(basename ${sample_path})

    echo "======================================"
    echo "Evaluating: ${sample_name}"
    echo "======================================"

    result=$(python calculate_fid.py \
        --real_path ${REAL_PATH} \
        --fake_path ${sample_path} \
        --ckpt_path "../fid_vae_ckpts/vae_synth_u/best.pt" \
        --batch_size 128 \
        --hidden_size 128 \
        --num_layers 2 \
        --num_heads 8 \
        --latent_dim 64 \
        --save_path "tmp.txt" \
        --num_samples 2850
    )

    echo "${sample_name}" >> ${SAVE_FILE}
    echo "${result}" >> ${SAVE_FILE}
    echo "--------------------------" >> ${SAVE_FILE}

done

echo "All FID computed! Saved to ${SAVE_FILE}"