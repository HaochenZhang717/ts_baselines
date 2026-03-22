#!/bin/bash
set -e

## environment setting
#export PATH="$HOME/.conda/envs/tsgen1/bin:$PATH"   ## change this to your conda env path
#conda activate vlm

export CUDA_VISIBLE_DEVICES=0

echo "Running FlowTS on mydataset..."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for hucfg_num_steps in 800
do
for hucfg_Kscale in 0.03
do
export hucfg_attention_rope_use=-1
export hucfg_lr=3e-4
export hucfg_num_steps=${hucfg_num_steps}
export hucfg_Kscale=${hucfg_Kscale}
export hucfg_t_sampling=logitnorm
export results_folder=./Checkpoints_mydataset_flows

echo "========================================"
echo "hucfg_attention_rope_use=${hucfg_attention_rope_use}"
echo "hucfg_lr=${hucfg_lr}"
echo "hucfg_num_steps=${hucfg_num_steps}"
echo "hucfg_Kscale=${hucfg_Kscale}"
echo "hucfg_t_sampling=${hucfg_t_sampling}"
echo "results_folder=${results_folder}"
echo "========================================"

python mydataset_nips.py
done
done