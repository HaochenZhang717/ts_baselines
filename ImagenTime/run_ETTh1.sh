export CUDA_VISIBLE_DEVICES=0
#python run_unconditional.py \
#  --config ./configs/neurips_baseline/synth_u.yaml

export WANDB_PROJECT="ETTh1"
export WANDB_NAME="VerbalTS_uncond_EDM"
python run_unconditional.py \
  --config ./configs/neurips_baseline/etth1.yaml \
  --log_dir ../ETTh1_results/ImagenTime
