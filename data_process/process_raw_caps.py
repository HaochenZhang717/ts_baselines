import json
import glob
import os
import re

# 你的jsonl目录
# folder = "./synthetic_u_caps/valid"
# output_file = os.path.join(folder, "time_series_caps_merged_sorted.jsonl")
# npy_save_name = os.path.join(folder, "valid_text_my_caps.npy")

# folder = "./synthetic_u_caps/train"
# output_file = os.path.join(folder, "time_series_caps_merged_sorted.jsonl")
# npy_save_name = os.path.join(folder, "train_text_my_caps.npy")


# folder = "./synthetic_u_caps_version2/test"
# output_file = os.path.join(folder, "time_series_caps_merged_sorted.jsonl")
# npy_save_name = os.path.join(folder, "test_text_my_caps_v2.npy")
# expected_set = set(range(4000))
# name_format =  "test_time_series_caps_*.jsonl"


# folder = "./synthetic_u_caps_version2/train"
# output_file = os.path.join(folder, "time_series_caps_merged_sorted.jsonl")
# npy_save_name = os.path.join(folder, "train_text_my_caps_v2.npy")
# expected_set = set(range(24000))
# name_format =  "train_time_series_caps_*.jsonl"


# folder = "./synthetic_u_caps_version2/valid"
# output_file = os.path.join(folder, "time_series_caps_merged_sorted.jsonl")
# npy_save_name = os.path.join(folder, "valid_text_my_caps_v2.npy")
# expected_set = set(range(4000))
# name_format =  "valid_time_series_caps_*.jsonl"

# folder = "/Users/zhc/Documents/LitsDatasets/128_len_caps/synth_u"
# output_file = os.path.join(folder, "train_caps.jsonl")
# expected_set = set(range(24000))
# name_format =  "train_caps_*.jsonl"




# 找到前8个文件（排除 toy 和 3072）
# files = sorted([
#     f for f in glob.glob(os.path.join(folder, "time_series_caps_*.jsonl"))
#     if "toy" not in f and "3072" not in f
# ])

# files = sorted([f for f in glob.glob(os.path.join(folder,name_format))])
#
# print("Files to merge:")
# for f in files:
#     print(f)
#
# all_data = []
#
# # 读取所有jsonl
# for file in files:
#     with open(file, "r") as f:
#         for line in f:
#             all_data.append(json.loads(line))
#
# print("Total samples before sort:", len(all_data))
#
# # 提取 image_XXXX 里的数字
# def extract_number(item):
#     name = item["image"]
#
#     match = re.search(r"image(\d+)_seg(\d+)_ch(\d+)", name)
#
#     if match:
#         image_id = int(match.group(1))
#         seg_id = int(match.group(2))
#         ch_id = int(match.group(3))
#         return (image_id, seg_id, ch_id)
#
#     return (10**12, 10**12, 10**12)
#
#
# # 排序
# all_data.sort(key=extract_number)
#
# print("Sorting done.")
#
# # 输出文件
#
# with open(output_file, "w") as f:
#     for item in all_data:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")
#
# print("Saved to:", output_file)
#
#
#
#
# import json
# import re
#
# file_path = output_file
#
# ids = []
#
# # 读取所有image编号
# with open(file_path, "r") as f:
#     for line in f:
#         item = json.loads(line)
#         match = re.search(r"image_(\d+)", item["image"])
#         if match:
#             ids.append(int(match.group(1)))
#
# print("Total samples:", len(ids))
#
# ids_set = set(ids)
#
#
#
# missing = sorted(expected_set - ids_set)
# extra = sorted(ids_set - expected_set)
#
# # 检查重复
# from collections import Counter
# counter = Counter(ids)
# duplicates = [k for k, v in counter.items() if v > 1]
#
# print("Missing count:", len(missing))
# print("Duplicate count:", len(duplicates))
# print("Extra count:", len(extra))
#
# if missing:
#     print("First 20 missing:", missing[:20])
#
# if duplicates:
#     print("First 20 duplicates:", duplicates[:20])
#
# if extra:
#     print("Extra ids:", extra[:20])
#
#
#
# import json
# import numpy as np
#
#
# captions = []
#
# with open(output_file, "r", encoding="utf-8") as f:
#     for line in f:
#         data = json.loads(line)
#         captions.append(data["caption"])
#
# # 转成 (N, 1) numpy array
# captions_array = np.array(captions, dtype=object).reshape(-1, 1)
#
# print(captions_array.shape)  # (N, 1)
# print(captions_array.dtype)  # object
#
# np.save(npy_save_name, captions_array)








import argparse
import json
import glob
import os
import re
from collections import Counter
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing jsonl caption files")

    parser.add_argument("--name_format", type=str, required=True,
                        help="Glob format of jsonl files (e.g. train_caps_*.jsonl)")

    parser.add_argument("--output_file", type=str, default=None,
                        help="Merged jsonl output")

    parser.add_argument("--npy_save_name", type=str, default=None,
                        help="Output npy captions file")

    return parser.parse_args()


# ------------------------------------------------------------
# sorting key
# ------------------------------------------------------------
def extract_key(item):

    name = item["image"].replace(".png", "")

    try:
        image_part, ch_part = name.split("_")
        image_id = int(image_part.replace("image", ""))
        ch_id = int(ch_part.replace("ch", ""))
        return (image_id, ch_id)
    except:
        return (10**12, 10**12)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():

    args = parse_args()

    folder = args.folder
    name_format = args.name_format

    if args.output_file is None:
        output_file = os.path.join(folder, "merged_caps.jsonl")
    else:
        output_file = args.output_file

    # expected_set = set(range(args.expected_num))

    print("\nSearching files...")

    files = sorted(glob.glob(os.path.join(folder, name_format)))

    if len(files) == 0:
        raise RuntimeError("No files found.")

    print("Files to merge:")
    for f in files:
        print("  ", f)

    # --------------------------------------------------------
    # load jsonl
    # --------------------------------------------------------

    all_data = []

    print("\nLoading jsonl...")

    for file in files:

        with open(file, "r") as f:
            for line in f:
                all_data.append(json.loads(line))

    print("Total samples before sort:", len(all_data))

    # --------------------------------------------------------
    # sort
    # --------------------------------------------------------

    print("\nSorting by image, seg, ch...")

    all_data.sort(key=extract_key)

    print("Sorting done.")

    # --------------------------------------------------------
    # save jsonl
    # --------------------------------------------------------

    print("\nSaving merged jsonl...")
    output_dir = os.path.dirname(output_file)
    print("Output dir:", output_dir)
    os.makedirs(os.path.join(output_dir), exist_ok=True)

    with open(output_file, "w") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved to:", output_file)

    # --------------------------------------------------------
    # dataset consistency check
    # --------------------------------------------------------

    print("\nChecking dataset consistency...")

    ids = []

    for item in all_data:
        match = re.search(r"image(\d+)", item["image"])
        if match:
            ids.append(int(match.group(1)))

    print("Total samples:", len(ids))

    # --------------------------------------------------------
    # export npy captions
    # --------------------------------------------------------

    if args.npy_save_name is not None:

        print("\nSaving npy captions...")

        captions = [item["caption"] for item in all_data]

        captions_array = np.array(captions, dtype=object).reshape(-1, 1)

        print("Captions shape:", captions_array.shape)

        np.save(args.npy_save_name, captions_array)

        print("Saved npy:", args.npy_save_name)


if __name__ == "__main__":
    main()