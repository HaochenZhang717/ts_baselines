#!/bin/bash

# =========================
# CONFIG
# =========================
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/playpen/haochenz/hf_cache

#CAPS_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_caps_one_per_channel_0309/synth_u"
#SAVE_ROOT="/playpen-shared/haochenz/LitsDatasets/128_len_caps_one_per_channel_0309/synth_u"

CAPS_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u"
SAVE_ROOT="/playpen-shared/haochenz/LitsDatasets/128_len_ts/synth_u"

SCRIPT_PATH="precompute_qwen_embeds.py"

BATCH_SIZE=128
DEVICE="cuda"

mkdir -p $SAVE_ROOT

echo "Start precomputing Qwen embeddings..."

# =========================
# TRAIN
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/train_embeds_caps_0324.pt \
    --split train \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# =========================
# VAL
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/valid_embeds_caps_0324.pt \
    --split valid \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# =========================
# TEST
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/test_embeds_caps_0324.pt \
    --split test \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

#echo "Done!"