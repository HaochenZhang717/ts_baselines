import re
import matplotlib.pyplot as plt

def parse_fid_file(path):
    with open(path, "r") as f:
        text = f.read()

    blocks = text.split("--------------------------")

    data = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) == 0:
            continue

        # 第一行：文件名
        name = lines[0]

        # 修复：匹配 Epoch1099
        match_epoch = re.search(r"[Ee]poch[_]?(\d+)", name)

        if "best_loss" in name:
            epoch = -1
        elif match_epoch:
            epoch = int(match_epoch.group(1))
        else:
            continue

        # 找 FID
        fid = None
        for line in lines:
            if "FID" in line:
                match_fid = re.search(r"FID:\s*\$([0-9.]+)", line)
                if match_fid:
                    fid = float(match_fid.group(1))
                    break

        if fid is not None:
            data.append((epoch, fid))

    data.sort(key=lambda x: x[0])
    return data

def plot_fid(data, save_name):
    epochs = [x[0] for x in data]
    fids = [x[1] for x in data]

    plt.figure()
    plt.plot(epochs, fids, marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("FID")
    plt.title("FID vs Epoch")
    plt.grid()

    plt.savefig(save_name)


if __name__ == "__main__":
    path = "../fid_results/synth_u_qwen_v1_generation_L10D128H8_lr_1e-3_bs_128.txt"  # 改成你的路径
    data = parse_fid_file(path)

    print("Parsed data:")
    for d in data:
        print(d)

    plot_fid(data, save_name="synth_u_qwen_v1_L10D128H8_lr_1e-3_bs_128.png")