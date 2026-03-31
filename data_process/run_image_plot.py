import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 关键
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter1d


def plot_one(task):
    i, channel_i, segment, save_dir = task

    num_segments = 1
    segment_len = len(segment)
    t = np.arange(segment_len)

    fig, axes = plt.subplots(
        1,
        num_segments,
        figsize=(4, 4),
        dpi=100
    )

    if num_segments == 1:
        axes = [axes]

    for j, ax in enumerate(axes):

        ax.plot(t, segment, linewidth=1)

        ax.set_xticks([])
        # ax.set_yticks([])

        # 黑色边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")

    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=0,
        hspace=0
    )

    plt.savefig(
        f"{save_dir}/image{i}_ch{channel_i}.png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


def save_simple(save_dir, ts_path):

    os.makedirs(save_dir, exist_ok=True)

    train_ts = np.load(ts_path, allow_pickle=True)
    print(f"Train shape: {train_ts.shape}")

    num_channels = train_ts.shape[-1]

    tasks = []

    for i in range(train_ts.shape[0]):
        for channel_i in range(num_channels):
            seg = train_ts[i, :, channel_i]
            seg = gaussian_filter1d(seg, sigma=10, mode="nearest")
            tasks.append((i, channel_i, seg, save_dir))

    print(f"Total images: {len(tasks)}")

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(plot_one, tasks), total=len(tasks)))


if __name__ == "__main__":
    data_path = "/playpen-shared/haochenz/LitsDatasets/128_len_ts/ETTh1"
    save_path = "/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs/ETTh1"
    os.makedirs(f"{save_path}/train", exist_ok=True)
    os.makedirs(f"{save_path}/valid", exist_ok=True)
    os.makedirs(f"{save_path}/test", exist_ok=True)

    save_simple(
        save_dir=f"{save_path}/train",
        ts_path=f"{data_path}/train_ts.npy")

    save_simple(
        save_dir=f"{save_path}/valid",
        ts_path=f"{data_path}/valid_ts.npy")

    save_simple(
        save_dir=f"{save_path}/test",
        ts_path=f"{data_path}/test_ts.npy")
