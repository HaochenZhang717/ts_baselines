export USE_CAUSAL=false

#CUDA_VISIBLE_DEVICES=7 python run_qwen_v3.py \
#    --cond_modal text \
#    --training_stage finetune \
#    --save_folder ./verbalts_orig_save/synth_u_qwen_v3/text2ts_msmdiffmv \
#    --model_diff_config_path configs/synth_u_qwen_v3/diff/model_text2ts_dep.yaml \
#    --model_cond_config_path configs/synth_u_qwen_v3/cond/text_msmdiffmv.yaml \
#    --train_config_path configs/synth_u_qwen_v3/train.yaml \
#    --evaluate_config_path configs/synth_u_qwen_v3/evaluate.yaml \
#    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
#    --clip_folder "" \
#    --multipatch_num 3 \
#    --L_patch_len 2 \
#    --base_patch 4 \
#    --epochs 2500 \
#    --batch_size 512 \
#    --clip_cache_path "" \
#    --samples_name "real_text_samples.pt" \
#    --model_ckpt_name "model_best_loss.pth" \
#    --only_evaluate True


export CUDA_VISIBLE_DEVICES=6

CKPT_DIR="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen_v3/text2ts_msmdiffmv/0/ckpts"
SAVE_ROOT="./verbalts_orig_save/synth_u_qwen_v3/text2ts_msmdiffmv"

for ckpt_path in ${CKPT_DIR}/model_*.pth
do
    ckpt_name=$(basename ${ckpt_path})
    ckpt_name_noext=${ckpt_name%.pth}

    echo "======================================"
    echo "Running checkpoint: ${ckpt_name}"
    echo "======================================"

    python run_qwen_v3.py \
        --cond_modal text \
        --training_stage finetune \
        --save_folder ${SAVE_ROOT} \
        --model_diff_config_path configs/synth_u_qwen_v3/diff/model_text2ts_dep.yaml \
        --model_cond_config_path configs/synth_u_qwen_v3/cond/text_msmdiffmv.yaml \
        --train_config_path configs/synth_u_qwen_v3/train.yaml \
        --evaluate_config_path configs/synth_u_qwen_v3/evaluate.yaml \
        --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
        --clip_folder "" \
        --multipatch_num 3 \
        --L_patch_len 2 \
        --base_patch 4 \
        --epochs 2500 \
        --batch_size 512 \
        --clip_cache_path "" \
        --samples_name "real_text_samples_${ckpt_name_noext}.pt" \
        --model_ckpt_name ${ckpt_name} \
        --only_evaluate True \
        --n_runs 2

done

echo "All checkpoints finished!"