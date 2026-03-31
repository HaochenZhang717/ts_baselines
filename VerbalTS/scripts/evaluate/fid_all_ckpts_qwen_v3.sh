#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# =========================
# 路径
# =========================

# 👉 必须是真实数据（GT）
REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen_v3/text2ts_msmdiffmv/1/real_text_samples_model_best_loss.pt"

SAMPLE_DIR="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen_v3/text2ts_msmdiffmv/1"

SAVE_FILE="./fid_results/synth_u_qwen_v3_generation_run1.txt"

mkdir -p ./fid_results

echo "FID Results" > ${SAVE_FILE}
echo "==========================" >> ${SAVE_FILE}

# =========================
# 遍历所有 ckpt samples
# =========================

for sample_path in ${SAMPLE_DIR}/real_text_samples_model_*.pt
do
    sample_name=$(basename ${sample_path})

    echo "======================================"
    echo "Evaluating: ${sample_name}"
    echo "======================================"

    result=$(python calculate_fid.py \
        --real_path ${REAL_PATH} \
        --fake_path ${sample_path} \
        --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
        --batch_size 128 \
        --hidden_size 128 \
        --num_layers 2 \
        --num_heads 8 \
        --latent_dim 64 \
        --save_path "tmp.txt"
    )

    echo "${sample_name}" >> ${SAVE_FILE}
    echo "${result}" >> ${SAVE_FILE}
    echo "--------------------------" >> ${SAVE_FILE}

done

echo "All FID computed! Saved to ${SAVE_FILE}"