export CUDA_VISIBLE_DEVICES=6
export USE_CAUSAL=false


#LAYERS=3
#CHANNELS=128
#NHEADS=8
#DIFFUSION_EMBEDDING_DIM=128
LR=5e-4
BS=512

CKPT_DIR="/playpen-shared/haochenz/sweep_text2ts_v6/synth_u/verbalts_v6/0/ckpts"
SAVE_ROOT="/playpen-shared/haochenz/sweep_text2ts_v6/synth_u/verbalts_v6"


for ckpt_path in ${CKPT_DIR}/model_*.pth
do
    ckpt_name=$(basename ${ckpt_path})
    ckpt_name_noext=${ckpt_name%.pth}

    echo "======================================"
    echo "Running checkpoint: ${ckpt_name}"
    echo "======================================"

    python run_v6.py \
        --cond_modal text \
        --training_stage finetune \
        --save_folder ${SAVE_ROOT} \
        --model_diff_config_path configs/synth_u/diff/model_text2ts_dep.yaml \
        --model_cond_config_path configs/synth_u/cond/text_msmdiffmv.yaml \
        --train_config_path configs/synth_u/train.yaml \
        --evaluate_config_path configs/synth_u/evaluate.yaml \
        --data_folder /playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u \
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
        --batch_size ${BS} \
        --lr ${LR}
done

echo "All checkpoints finished!"