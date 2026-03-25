export CUDA_VISIBLE_DEVICES=4
#python run_unconditional.py \
#  --config ./configs/neurips_baseline/synth_u.yaml

export WANDB_PROJECT="unconditional_baselines"
export WANDB_NAME="VerbalTS_uncond_EDM"
python run_unconditional_my_model.py \
  --config ./configs/neurips_baseline/synth_u.yaml \
  --log_dir ../edm_results/verbal_ts
