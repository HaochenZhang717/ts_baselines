#!/bin/bash

# =========================
# GPU
# =========================
export CUDA_VISIBLE_DEVICES=3

# =========================
# 基础配置
# =========================
EXP_NAME="vae_synth_m"

DATA_ROOT="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_m"
CKPT_PATH="./vae_ckpts/${EXP_NAME}/best.pt"

SAVE_ROOT="./vae_embeddings/synth_m"
mkdir -p ${SAVE_ROOT}

# =========================
# 模型参数（必须一致）
# =========================
BATCH_SIZE=128

HIDDEN_SIZE=128
NUM_LAYERS=2
NUM_HEADS=8
LATENT_DIM=64

# =========================
# Split 循环
# =========================

for SPLIT in train valid test
do
    DATA_PATH="${DATA_ROOT}/${SPLIT}_ts.npy"
    SAVE_PATH="${SAVE_ROOT}/${SPLIT}_vae.npy"

    echo "======================================"
    echo "Processing ${SPLIT} split"
    echo "======================================"

    python extract_vae_embeds.py \
        --data_path ${DATA_PATH} \
        --ckpt_path ${CKPT_PATH} \
        --batch_size ${BATCH_SIZE} \
        --hidden_size ${HIDDEN_SIZE} \
        --num_layers ${NUM_LAYERS} \
        --num_heads ${NUM_HEADS} \
        --latent_dim ${LATENT_DIM} \
        --save_path ${SAVE_PATH}

    echo "Saved ${SPLIT} embeddings to ${SAVE_PATH}"
done

echo "All splits done!"