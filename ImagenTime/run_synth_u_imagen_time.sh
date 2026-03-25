export CUDA_VISIBLE_DEVICES=4

export WANDB_PROJECT="unconditional_baselines"
export WANDB_NAME="ImagenTime_uncond_EDM"

python run_unconditional.py \
  --config ./configs/neurips_baseline/synth_u.yaml \
  --log_dir ../edm_results/imagen_time



#python run_unconditional_my_model.py \
#  --config ./configs/neurips_baseline/synth_u.yaml