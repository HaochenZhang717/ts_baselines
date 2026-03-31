import re
import matplotlib.pyplot as plt

def parse_fid_file(path, score_name="FID"):
    with open(path, "r") as f:
        text = f.read()
    # breakpoint()
    blocks = text.split("--------------------------")

    data = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) == 0:
            continue

        # 找 epoch
        name = lines[0]
        match_epoch = re.search(r"epoch_(\d+)", name)

        if "best_loss" in name:
            epoch = -1  # 用 -1 表示 best
        elif match_epoch:
            epoch = int(match_epoch.group(1))
        else:
            continue


        # 找 FID
        fid = None
        for line in lines:
            if score_name in line:
                match_fid = re.search(r"\$([0-9.]+)", line)
                if match_fid:
                    fid = float(match_fid.group(1))
                    break

        if fid is not None:
            data.append((epoch, fid))

    # 排序（best放最前）
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
    path = "../fid_results/synth_u_imagen_time.txt"  # 改成你的路径
    data_fid = parse_fid_file(path, score_name="FID")
    data_kid = parse_fid_file(path, score_name="KID")

    print("Parsed data:")
    for d in data_fid:
        print(d)

    plot_fid(data_fid, save_name="synth_u_imagen_time_fid.png")
    plot_fid(data_kid, save_name="synth_u_imagen_time_kid.png")

    # path = "../fid_results/synth_u_qwen_v1_generation_run1.txt"  # 改成你的路径
    # data_fid = parse_fid_file(path, score_name="FID")
    # data_kid = parse_fid_file(path, score_name="KID")
    #
    # print("Parsed data:")
    # for d in data_fid:
    #     print(d)
    #
    # plot_fid(data_fid, save_name="synth_u_qwen_v1_generation_run1_fid.png")
    # plot_fid(data_kid, save_name="synth_u_qwen_v1_generation_run1_kid.png")