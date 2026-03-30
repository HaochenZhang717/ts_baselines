import numpy as np
import torch
from metrics import compute_fid
import matplotlib.pyplot as plt


def get_all_fid_scores(data_folder):
    epoch_list = np.arange(0, 2000, 100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid_list = []
    for epoch in epoch_list:
        data_path = f"{data_folder}/samples_epoch_{epoch}.pt"
        data = torch.load(data_path)
        x_real = data["real_ts"]
        gens = data["gen_ts"]
        fid_score = compute_fid(
            x_real, gens,
            "/playpen-shared/haochenz/fid_vae_ckpts/vae_synth_u/best.pt")
        fid_score = fid_score['fid']
        fid_list.append(fid_score)
    return fid_list


uncond_fids = get_all_fid_scores(
    "/playpen-shared/haochenz/ts_baselines/edm_results/imagen_time/synth_u/conditional-bs=128-lr=0.0001-ch_mult=1-2-attn_res=16-8-4-unet_ch=64-delay=4-32"
)

text_cond_fids = get_all_fid_scores(
    "/playpen-shared/haochenz/ts_baselines/edm_results_text_conditional/imagen_time/synth_u_text/conditional-bs=128-lr=0.0001-ch_mult=1-2-attn_res=16-8-4-unet_ch=64-delay=4-32"
)

cross_attn_cond_fids = get_all_fid_scores(
    "/playpen-shared/haochenz/ts_baselines/edm_results_text_conditional/LDM_debug/synth_u_text_ldm/conditional-bs=128-lr=0.0001-ch_mult=1-2-attn_res=2-4-8-unet_ch=64-delay=4-32"
)


epoch_list = np.arange(0, 2000, 100)

plt.figure()

plt.plot(epoch_list, uncond_fids, label="Unconditional")
# plt.plot(epoch_list, text_cond_fids, label="Text Conditional")
plt.plot(epoch_list, cross_attn_cond_fids, label="Cross Attention Text Conditional")

print(f"Unconditional minimum FID: {min(uncond_fids)}")
print(f"Cross attention minimum FID: {min(cross_attn_cond_fids)}")
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID vs Epoch")
plt.legend()

plt.grid(True)

plt.savefig("fid_curve.png", dpi=600)
plt.show()