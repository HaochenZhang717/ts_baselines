import numpy as np
import torch

caps = np.load("/Users/zhc/Documents/LitsDatasets/128_len_ts/synthetic_u/train_text_caps.npy")
for cap in caps:
    print(cap[0])
