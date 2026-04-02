export CUDA_VISIBLE_DEVICES=1
#python run_unconditional.py \
#  --config ./configs/neurips_baseline/synth_u.yaml

export WANDB_PROJECT="ETTh1"
export WANDB_NAME="VerbalTS_uncond_EDM_0402"
python run_unconditional.py \
  --config ./configs/neurips_baseline/etth1.yaml \
  --log_dir ../ETTh1_results/ImagenTime_0402 \
  --fid_vae_ckpt_path "/playpen-shared/haochenz/fid_vae_ckpts/vae_etth1/best.pt"

