export USE_CAUSAL=false


LR_LIST=(5e-4)
BS_LIST=(128)

LAYERS=4
CHANNELS=128
NHEADS=8
DIFFUSION_EMBEDDING_DIM=128

for LR in "${LR_LIST[@]}"
do
  for BS in "${BS_LIST[@]}"
  do
    echo "Running lr=$LR bs=$BS"

    export WANDB_NAME="qwen_v5_synth_u_cosine-lr${LR}_bs${BS}-L${LAYERS}C${CHANNELS}H${NHEADS}D${DIFFUSION_EMBEDDING_DIM}-dropout0.1"

    CUDA_VISIBLE_DEVICES=4 python run_qwen_v5.py \
        --cond_modal text \
        --training_stage finetune \
        --save_folder ./sweep/synth_u_qwen_v1/lr_${LR}_bs_${BS}-L${LAYERS}C${CHANNELS}H${NHEADS}D${DIFFUSION_EMBEDDING_DIM}-dropout0.1 \
        --model_diff_config_path configs/synth_u_qwen/diff/model_text2ts_dep.yaml \
        --model_cond_config_path configs/synth_u_qwen/cond/text_msmdiffmv.yaml \
        --train_config_path configs/synth_u_qwen/train.yaml \
        --evaluate_config_path configs/synth_u_qwen/evaluate.yaml \
        --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
        --clip_folder "" \
        --multipatch_num 4 \
        --L_patch_len 2 \
        --base_patch 4 \
        --epochs 2500 \
        --layers ${LAYERS} \
        --channels ${CHANNELS} \
        --nheads ${NHEADS} \
        --diffusion_embedding_dim ${DIFFUSION_EMBEDDING_DIM} \
        --batch_size ${BS} \
        --lr ${LR} \
        --clip_cache_path "" \
        --samples_name "real_text_samples.pt" \
        --model_ckpt_name "model_best_loss.pth"
done
done