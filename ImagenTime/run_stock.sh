export CUDA_VISIBLE_DEVICES=2

export WANDB_PROJECT="Stock"
export WANDB_NAME="ImagenTime_0402"
python run_unconditional.py \
  --config ./configs/neurips_baseline/stock.yaml \
  --log_dir ../stock_results/ImagenTime_0402 \
  --fid_vae_ckpt_path "/playpen-shared/haochenz/fid_vae_ckpts/vae_etth1/best.pt"

