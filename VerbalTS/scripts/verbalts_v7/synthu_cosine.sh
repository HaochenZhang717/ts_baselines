LR_LIST=(1e-3 5e-4 2e-4)
BS_LIST=(64 128 256 512)

export SCHEDULER=cosine

for LR in "${LR_LIST[@]}"
do
  for BS in "${BS_LIST[@]}"
  do
    echo "Running lr=$LR bs=$BS"

    export WANDB_NAME="orig_verbalts_my_cap-lr${LR}_bs${BS}_cosine"

    CUDA_VISIBLE_DEVICES=6 python run_v7.py \
    --cond_modal text \
    --training_stage finetune \
    --save_folder ../sweep_text2ts/synth_u/orig_verbalts_my_cap-lr${LR}_bs${BS}_cosine \
    --model_diff_config_path configs/synth_u_v7/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_u_v7/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_u_v7/train.yaml \
    --evaluate_config_path configs/synth_u_v7/evaluate.yaml \
    --data_folder /playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 700 \
    --n_runs 1 \
    --batch_size ${BS} \
    --lr ${LR} \
    --clip_cache_path "" \
    --samples_name "samples.pt"
done
done