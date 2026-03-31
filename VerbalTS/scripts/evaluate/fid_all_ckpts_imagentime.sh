#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# =========================
# 路径
# =========================

# 👉 必须是真实数据（GT）
REAL_PATH="/playpen-shared/haochenz/ts_baselines/ImagenTime/logs/synth_u/conditional-bs=128-lr=0.0001-ch_mult=1-2-attn_res=16-8-4-unet_ch=64-delay=4-32/samples_epoch_1500.pt"

SAMPLE_DIR="/playpen-shared/haochenz/ts_baselines/ImagenTime/logs/synth_u/conditional-bs=128-lr=0.0001-ch_mult=1-2-attn_res=16-8-4-unet_ch=64-delay=4-32"


SAVE_FILE="./fid_results/synth_u_imagen_time.txt"

mkdir -p ./fid_results

echo "FID Results" > ${SAVE_FILE}
echo "==========================" >> ${SAVE_FILE}

# =========================
# 遍历所有 ckpt samples
# =========================

for sample_path in ${SAMPLE_DIR}/samples_epoch_*.pt
do
    sample_name=$(basename ${sample_path})

    echo "======================================"
    echo "Evaluating: ${sample_name}"
    echo "======================================"

    result=$(python calculate_fid_imagentime.py \
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