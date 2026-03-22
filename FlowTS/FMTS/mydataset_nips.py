import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from engine.solver import Trainer
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one
)

# =========================
# 1. 数据路径
# =========================
DATA_ROOT = '../../data/synthetic_u'
TRAIN_PATH = os.path.join(DATA_ROOT, 'train_ts.npy')
VALID_PATH = os.path.join(DATA_ROOT, 'valid_ts.npy')
TEST_PATH  = os.path.join(DATA_ROOT, 'test_ts.npy')

train_raw = np.load(TRAIN_PATH).astype(np.float32)   # (24000,128,1)
valid_raw = np.load(VALID_PATH).astype(np.float32)
test_raw  = np.load(TEST_PATH).astype(np.float32)

print("train_raw shape:", train_raw.shape)
print("valid_raw shape:", valid_raw.shape)
print("test_raw  shape:", test_raw.shape)

assert train_raw.ndim == 3 and valid_raw.ndim == 3 and test_raw.ndim == 3

# =========================
# 2. 用 train 拟合 scaler，再把数据映射到 [-1,1]
# =========================
feat_num = train_raw.shape[-1]
scaler = MinMaxScaler()
scaler.fit(train_raw.reshape(-1, feat_num))

def to_neg_one_one(x):
    x01 = scaler.transform(x.reshape(-1, feat_num)).reshape(x.shape)
    x11 = normalize_to_neg_one_to_one(x01)
    return x11.astype(np.float32)

def from_neg_one_one(x):
    x01 = unnormalize_to_zero_to_one(x.reshape(-1, feat_num)).reshape(x.shape)
    raw = scaler.inverse_transform(x01.reshape(-1, feat_num)).reshape(x.shape)
    return raw

train = to_neg_one_one(train_raw)
valid = to_neg_one_one(valid_raw)
test  = to_neg_one_one(test_raw)

# =========================
# 3. Dataset
# =========================
class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.samples = data
        self.sample_num = data.shape[0]

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num



# =========================
# 5. 配置与模型
# =========================
class Args_Example:
    def __init__(self) -> None:
        self.config_path = './Config/neurips_baselines/mydataset.yaml'
        self.gpu = 0

args = Args_Example()
configs = load_yaml_config(args.config_path)


# =========================
# 4. 训练 dataloader
# =========================
train_dataset = MyDataset(train)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=configs['solver']['batch_size'],
    shuffle=True,
    num_workers=0,
    drop_last=False,
    pin_memory=True
)


valid_dataset = MyDataset(valid)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=configs['solver']['batch_size'],
    shuffle=False,
    num_workers=0,
    drop_last=False,
    pin_memory=True
)

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

model = instantiate_from_config(configs['model']).to(device)
trainer = Trainer(
    config=configs, args=args, model=model,
    dataloader={'train_dataloader': train_dataloader,
                'valid_dataloader': valid_dataloader},
)

# =========================
# 6. 训练
# =========================
trainer.train()
# trainer.load(1000)

# =========================
# 7. forecasting 测试
# =========================
_, seq_length, feat_num = test.shape
pred_length = 24

test_dataset = MyDataset(test)
real = test_raw.copy()

test_dataloader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
save_dir = configs['solver']['results_folder']

trainer.load("best")
window = configs["model"]["params"]["seq_length"]
var_num = configs["model"]["params"]["feature_size"]
samples = trainer.sample(num=len(test_dataset), size_every=2001, shape=[var_num, window])
sample = from_neg_one_one(samples)
np.save(os.path.join(save_dir, f'flowts_samples_best.npy'), samples)
