import json
import numpy as np
from collections import defaultdict
import re


def parse_image_name(name):
    """
    image0_ch3.png → (0, 3)
    """
    m = re.match(r"image(\d+)_ch(\d+)\.png", name)
    if m is None:
        raise ValueError(f"Invalid image name: {name}")
    return int(m.group(1)), int(m.group(2))


def jsonl_to_npy(jsonl_path, save_path):
    data = defaultdict(dict)  # {image_id: {channel_id: caption}}

    max_image_id = -1
    max_channel_id = -1

    # -------- read jsonl --------
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)

            image_name = item["image"]
            caption = item["caption"]

            image_id, ch_id = parse_image_name(image_name)

            data[image_id][ch_id] = caption

            max_image_id = max(max_image_id, image_id)
            max_channel_id = max(max_channel_id, ch_id)

    N = max_image_id + 1
    C = max_channel_id + 1

    print(f"N={N}, C={C}")

    # -------- build array --------
    arr = np.empty((N, C), dtype=object)

    for i in range(N):
        for c in range(C):
            if c in data[i]:
                arr[i, c] = data[i][c]
            else:
                arr[i, c] = ""  # 或者 None

    # -------- save --------
    np.save(save_path, arr)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    # -----------------------------
    # usage
    # -----------------------------
    jsonl_to_npy(
        "/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1/train_caps.jsonl",
        "/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1/train_caps.npy"
    )

    jsonl_to_npy(
        "/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1/valid_caps.jsonl",
        "/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1/valid_caps.npy"
    )

    jsonl_to_npy(
        "/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1/test_caps.jsonl",
        "/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1/test_caps.npy"
    )

    # train_caps = np.load("/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1/train_caps.npy", allow_pickle=True)
    # print("train_caps.shape =", train_caps.shape)

    # test_caps = np.load("/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1/test_caps.npy", allow_pickle=True)
    # print("test_caps.shape =", test_caps.shape)

    # val_caps = np.load("/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1/valid_caps.npy", allow_pickle=True)
    # print("val_caps.shape =", val_caps.shape)