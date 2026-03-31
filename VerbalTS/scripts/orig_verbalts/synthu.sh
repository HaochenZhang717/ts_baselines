LR_LIST=(1e-3)
BS_LIST=(512)

export SCHEDULER=MULTISTEP

for LR in "${LR_LIST[@]}"
do
  for BS in "${BS_LIST[@]}"
  do
    echo "Running lr=$LR bs=$BS"

#    export WANDB_NAME="orig_verbalts_orig_cap-lr${LR}_bs${BS}"
    export WANDB_NAME="orig_verbalts_cap0324-lr${LR}_bs${BS}"

    CUDA_VISIBLE_DEVICES=0 python run.py \
    --cond_modal text \
    --training_stage finetune \
    --save_folder ../sweep_text2ts/synth_u/orig_verbalts_orig_cap \
    --model_diff_config_path configs/synth_u/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_u/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_u/train.yaml \
    --evaluate_config_path configs/synth_u/evaluate.yaml \
    --data_folder /playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 700 \
    --batch_size ${BS} \
    --lr ${LR} \
    --n_runs 3 \
    --clip_cache_path "" \
    --samples_name "samples.pt" \
    --only_evaluate True
done
done