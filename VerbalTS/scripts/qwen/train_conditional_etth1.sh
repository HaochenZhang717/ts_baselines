export USE_CAUSAL=false


export SCHEDULER=cosine


LR_LIST=(1e-3)
BS_LIST=(128)

LAYERS=10
CHANNELS=128
NHEADS=8
DIFFUSION_EMBEDDING_DIM=128

for LR in "${LR_LIST[@]}"
do
  for BS in "${BS_LIST[@]}"
  do
    echo "Running lr=$LR bs=$BS"

    export WANDB_PROJECT="Qwen_Embed_VerbalTS_Arch"
    export WANDB_NAME="Qwen_Embed_VerbalTS_Arch"
    export CONFIG_NAME="etth1_qwen"
    CUDA_VISIBLE_DEVICES=7 python run_qwen.py \
        --cond_modal text \
        --training_stage finetune \
        --save_folder ./${CONFIG_NAME}/${WANDB_NAME}/ \
        --model_diff_config_path configs/${CONFIG_NAME}/diff/model_text2ts_dep.yaml \
        --model_cond_config_path configs/${CONFIG_NAME}/cond/text_msmdiffmv.yaml \
        --train_config_path configs/${CONFIG_NAME}/train.yaml \
        --evaluate_config_path configs/${CONFIG_NAME}/evaluate.yaml \
        --clip_folder "" \
        --multipatch_num 3 \
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