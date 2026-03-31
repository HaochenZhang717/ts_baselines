#!/bin/bash

# =========================
# CONFIG
# =========================
export CUDA_VISIBLE_DEVICES=5

CAPS_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1"
SAVE_ROOT="/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1"

SCRIPT_PATH="precompute_long_clip_embeds_from_npy.py"
NPY_NAME="caps"

BATCH_SIZE=32
DEVICE="cuda"

mkdir -p $SAVE_ROOT

echo "Start precomputing LongClip embeddings..."

# =========================
# TRAIN
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/train_embeds_long_clip_seq_0324.pt \
    --npy_name ${NPY_NAME} \
    --split train \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# =========================
# VAL
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/valid_embeds_long_clip_seq_0324.pt \
    --npy_name ${NPY_NAME} \
    --split valid \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# =========================
# TEST
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/test_embeds_long_clip_seq_0324.pt \
    --npy_name ${NPY_NAME} \
    --split test \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

echo "Done!"