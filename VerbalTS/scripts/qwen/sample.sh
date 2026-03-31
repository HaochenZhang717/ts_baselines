export CUDA_VISIBLE_DEVICES=4
export USE_CAUSAL=false

CKPT_DIR="/playpen/haochenz/VerbalTS_reimplement/sweep/synth_u_qwen_v1/lr_1e-3_bs_128/0/ckpts"
SAVE_ROOT="/playpen/haochenz/VerbalTS_reimplement/sweep/synth_u_qwen_v1/lr_1e-3_bs_128"

LAYERS=10
CHANNELS=128
NHEADS=8
DIFFUSION_EMBEDDING_DIM=128
LR=1e-3
BS=128

for ckpt_path in ${CKPT_DIR}/model_*.pth
do
    ckpt_name=$(basename ${ckpt_path})
    ckpt_name_noext=${ckpt_name%.pth}

    echo "======================================"
    echo "Running checkpoint: ${ckpt_name}"
    echo "======================================"

    python run_qwen.py \
        --cond_modal text \
        --training_stage finetune \
        --save_folder ${SAVE_ROOT} \
        --model_diff_config_path configs/synth_u_qwen/diff/model_text2ts_dep.yaml \
        --model_cond_config_path configs/synth_u_qwen/cond/text_msmdiffmv.yaml \
        --train_config_path configs/synth_u_qwen/train.yaml \
        --evaluate_config_path configs/synth_u_qwen/evaluate.yaml \
        --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
        --clip_folder "" \
        --multipatch_num 3 \
        --L_patch_len 2 \
        --base_patch 4 \
        --epochs 2500 \
        --clip_cache_path "" \
        --samples_name "real_text_samples_${ckpt_name_noext}.pt" \
        --model_ckpt_name ${ckpt_name} \
        --only_evaluate True \
        --n_runs 1 \
        --layers ${LAYERS} \
        --channels ${CHANNELS} \
        --nheads ${NHEADS} \
        --diffusion_embedding_dim ${DIFFUSION_EMBEDDING_DIM} \
        --batch_size ${BS} \
        --lr ${LR} \

done

echo "All checkpoints finished!"