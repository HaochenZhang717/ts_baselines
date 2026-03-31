#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# =========================
# 路径
# =========================

# 👉 必须是真实数据（GT）
REAL_PATH="/playpen/haochenz/oldsweep/synth_u_qwen_v1/lr_1e-3_bs_128-L10C128H8D128-dropout0.1-cfg0.1/0/real_text_samples_model_epoch_2499.pt"
SAMPLE_DIR='/playpen-shared/haochenz/flowts_neurips_baseline/synthu/LR0.0001-BS$128'


SAVE_FILE="./fid_results/synth_u_flowts.txt"

mkdir -p ./fid_results

echo "FID Results" > ${SAVE_FILE}
echo "==========================" >> ${SAVE_FILE}

# =========================
# 遍历所有 ckpt samples
# =========================

for sample_path in ${SAMPLE_DIR}/flowts_samples_*.npy
do
    sample_name=$(basename ${sample_path})

    echo "======================================"
    echo "Evaluating: ${sample_name}"
    echo "======================================"

    result=$(python calculate_fid_diffusionTS.py \
        --real_path ${REAL_PATH} \
        --fake_path ${sample_path} \
        --ckpt_path "../fid_vae_ckpts/vae_synth_u/best.pt" \
        --batch_size 128 \
        --hidden_size 128 \
        --num_layers 2 \
        --num_heads 8 \
        --latent_dim 64 \
        --num_samples 2850 \
        --save_path "tmp.txt"
    )

    echo "${sample_name}" >> ${SAVE_FILE}
    echo "${result}" >> ${SAVE_FILE}
    echo "--------------------------" >> ${SAVE_FILE}

done

echo "All FID computed! Saved to ${SAVE_FILE}"