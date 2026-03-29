import numpy as np
import torchaudio.transforms as transforms
import os
import sys
import torch
import torch.utils.data as Data
import pandas as pd


try:
    from data.data_provider.data_factory import data_provider
except Exception:
    data_provider = None

try:
    from data.long_range import parse_datasets
except Exception:
    parse_datasets = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PreSplitNPYDataset(torch.utils.data.Dataset):
    def __init__(self, arr):
        self.arr = torch.from_numpy(arr).float()

    def __getitem__(self, idx):
        return self.arr[idx]

    def __len__(self):
        return len(self.arr)

def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world data.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    assert data_name in ['stock', 'energy', 'metro']

    if data_name == 'stock':
        ori_data = np.loadtxt('./data/short_range/stock_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('./data/short_range/energy_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'metro':
        ori_data = np.loadtxt('./data/short_range/metro_data.csv', delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the data
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the data (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


def gen_dataloader(args):
    if args.dataset == 'sine':
        args.dataset_size = 10000
        ori_data = sine_data_generation(args.dataset_size, args.seq_len, args.input_channels)
        ori_data = torch.Tensor(np.array(ori_data))
        train_set = Data.TensorDataset(ori_data)

    elif args.dataset in ['stock', 'energy']:
        ori_data = real_data_loading(args.dataset, args.seq_len)
        ori_data = torch.Tensor(np.array(ori_data))
        train_set = Data.TensorDataset(ori_data)

    elif args.dataset in ['mujoco']:
        train_set = MujocoDataset(args.seq_len, args.dataset, args.path, 0.0)

    elif args.dataset in ['solar_weekly', 'fred_md', 'nn5_daily', 'temperature_rain', 'traffic_hourly', 'kdd_cup']:
        ori_data = parse_datasets(args.dataset, args.batch_size, args.device, args)
        ori_data = torch.stack(ori_data)
        args.seq_len = ori_data.shape[1]  # update seq_len to match the dataset
        full_len = ori_data.shape[0]
        randperm = torch.randperm(full_len)
        train_data = ori_data[randperm[:int(full_len * 0.8)]]
        test_data = ori_data[randperm[int(full_len * 0.8):]]
        train_set = Data.TensorDataset(train_data)
        test_set = Data.TensorDataset(test_data)
        train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers)
        test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        return train_loader, test_loader

    elif args.dataset in ['physionet', 'climate']:
        train_loader, test_loader = parse_datasets(args.dataset, args.batch_size, args.device, args)
        return train_loader, test_loader

    elif args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        train_data, train_loader = data_provider(args, flag='train')
        test_data, test_loader = data_provider(args, flag='test')
        return train_loader, test_loader

    elif args.dataset in ['synth_u']:
        train_path = '../data/synthetic_u/train_ts.npy'
        valid_path = '../data/synthetic_u/valid_ts.npy'
        test_path  = '../data/synthetic_u/test_ts.npy'

        train_data = np.load(train_path).astype(np.float32)
        valid_data = np.load(valid_path).astype(np.float32)
        test_data  = np.load(test_path).astype(np.float32)

        # 你的数据已经是 (N, L, D)
        assert train_data.ndim == 3, f"Expected train_data shape (N,L,D), got {train_data.shape}"
        assert valid_data.ndim == 3, f"Expected valid_data shape (N,L,D), got {valid_data.shape}"
        assert test_data.ndim == 3, f"Expected test_data shape (N,L,D), got {test_data.shape}"

        # 若 config 里 seq_len 和真实不一致，强制同步
        args.seq_len = train_data.shape[1]
        args.input_channels = train_data.shape[2]
        args.input_size = args.input_channels

        train_set = Data.TensorDataset(torch.Tensor(train_data))
        test_set = Data.TensorDataset(torch.Tensor(test_data))

        train_loader = Data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_loader = Data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        return train_loader, test_loader

    elif args.dataset in ['synth_m']:
        train_path = '../data/synthetic_m/train_ts.npy'
        valid_path = '../data/synthetic_m/valid_ts.npy'
        test_path  = '../data/synthetic_m/test_ts.npy'

        train_data = np.load(train_path).astype(np.float32)
        valid_data = np.load(valid_path).astype(np.float32)
        test_data  = np.load(test_path).astype(np.float32)
        # print(f"train shape: {train_data.shape}")
        # print(f"valid shape: {valid_data.shape}")
        # print(f"test shape: {test_data.shape}")
        # breakpoint()
        # 你的数据已经是 (N, L, D)
        assert train_data.ndim == 3, f"Expected train_data shape (N,L,D), got {train_data.shape}"
        assert valid_data.ndim == 3, f"Expected valid_data shape (N,L,D), got {valid_data.shape}"
        assert test_data.ndim == 3, f"Expected test_data shape (N,L,D), got {test_data.shape}"

        # 若 config 里 seq_len 和真实不一致，强制同步
        args.seq_len = train_data.shape[1]
        args.input_channels = train_data.shape[2]
        args.input_size = args.input_channels

        train_set = Data.TensorDataset(torch.Tensor(train_data))
        test_set = Data.TensorDataset(torch.Tensor(test_data))

        train_loader = Data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_loader = Data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        return train_loader, test_loader

    elif args.dataset in ['istanbul_traffic']:
        train_path = '../data/istanbul_traffic/train_ts.npy'
        valid_path = '../data/istanbul_traffic/valid_ts.npy'
        test_path  = '../data/istanbul_traffic/test_ts.npy'

        train_data = np.load(train_path).astype(np.float32)
        valid_data = np.load(valid_path).astype(np.float32)
        test_data  = np.load(test_path).astype(np.float32)

        # 你的数据已经是 (N, L, D)
        assert train_data.ndim == 3, f"Expected train_data shape (N,L,D), got {train_data.shape}"
        assert valid_data.ndim == 3, f"Expected valid_data shape (N,L,D), got {valid_data.shape}"
        assert test_data.ndim == 3, f"Expected test_data shape (N,L,D), got {test_data.shape}"

        # 若 config 里 seq_len 和真实不一致，强制同步
        args.seq_len = train_data.shape[1]
        args.input_channels = train_data.shape[2]
        args.input_size = args.input_channels

        train_set = Data.TensorDataset(torch.Tensor(train_data))
        test_set = Data.TensorDataset(torch.Tensor(test_data))

        train_loader = Data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_loader = Data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        return train_loader, test_loader

    elif args.dataset in ['synth_u_text']:
        train_path = '../data/synthetic_u/train_ts.npy'
        valid_path = '../data/synthetic_u/valid_ts.npy'
        test_path  = '../data/synthetic_u/test_ts.npy'

        train_data = np.load(train_path).astype(np.float32)
        valid_data = np.load(valid_path).astype(np.float32)
        test_data  = np.load(test_path).astype(np.float32)

        train_text_embed = torch.load("../data/synthetic_u/train_embeds_caps_0324.pt", map_location="cpu")
        valid_text_embed = torch.load("../data/synthetic_u/valid_embeds_caps_0324.pt", map_location="cpu")
        test_text_embed = torch.load("../data/synthetic_u/test_embeds_caps_0324.pt", map_location="cpu")

        # 你的数据已经是 (N, L, D)
        assert train_data.ndim == 3, f"Expected train_data shape (N,L,D), got {train_data.shape}"
        assert valid_data.ndim == 3, f"Expected valid_data shape (N,L,D), got {valid_data.shape}"
        assert test_data.ndim == 3, f"Expected test_data shape (N,L,D), got {test_data.shape}"

        # 若 config 里 seq_len 和真实不一致，强制同步
        args.seq_len = train_data.shape[1]
        args.input_channels = train_data.shape[2]
        args.input_size = args.input_channels

        train_set = Data.TensorDataset(torch.Tensor(train_data), train_text_embed)
        test_set = Data.TensorDataset(torch.Tensor(test_data), test_text_embed)

        train_loader = Data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_loader = Data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        return train_loader, test_loader

    elif args.dataset in ['synth_u_text_ldm']:
        train_path = '../data/synthetic_u/train_ts.npy'
        valid_path = '../data/synthetic_u/valid_ts.npy'
        test_path = '../data/synthetic_u/test_ts.npy'

        train_data = np.load(train_path).astype(np.float32)
        valid_data = np.load(valid_path).astype(np.float32)
        test_data = np.load(test_path).astype(np.float32)

        # 你的数据已经是 (N, L, D)
        assert train_data.ndim == 3, f"Expected train_data shape (N,L,D), got {train_data.shape}"
        assert valid_data.ndim == 3, f"Expected valid_data shape (N,L,D), got {valid_data.shape}"
        assert test_data.ndim == 3, f"Expected test_data shape (N,L,D), got {test_data.shape}"

        # 若 config 里 seq_len 和真实不一致，强制同步
        args.seq_len = train_data.shape[1]
        args.input_channels = train_data.shape[2]
        args.input_size = args.input_channels

        train_set = DatasetForPrecomputedEmbed(train_data)
        test_set = DatasetForPrecomputedEmbed(test_data)

        train_loader = Data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_loader = Data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        return train_loader, test_loader

    elif args.dataset in ['aireadi']:
        train_set = AIREADIDataset(
            calorie_path="/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed/calorie_train.parquet",
            glucose_path="/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed/glucose_train.parquet",
            window_size=512,
            predict_length=128,
            stride=100,
        )

        test_set = AIREADIDataset(
            calorie_path="/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed/calorie_valid.parquet",
            glucose_path="/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed/glucose_valid.parquet",
            window_size=512,
            predict_length=128,
            stride=100,
        )
        print(f"Train set size: {len(train_set)}")
        print(f"Test set size: {len(test_set)}")
        train_datum = train_set[0][0]
        args.seq_len = train_datum.shape[0]
        args.input_channels = train_datum.shape[1]
        args.input_size = args.input_channels

        train_loader = Data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_loader = Data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        return train_loader, test_loader



    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)

    # for the short-term time series benchmark, the entire dataset for both training and testing
    return train_loader, train_loader


def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def stft_transform(data, args):
    data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
    n_fft = args.n_fft
    hop_length = args.hop_length
    spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True, power=None)
    transformed_data = spec(data)
    real, min_real, max_real = MinMaxScaler(transformed_data.real.numpy(), True)
    real = (real - 0.5) * 2
    imag, min_imag, max_imag = MinMaxScaler(transformed_data.imag.numpy(), True)
    imag = (imag - 0.5) * 2
    # saving min and max values, we will need them for inverse transform
    args.min_real, args.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
    args.min_imag, args.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)
    return torch.Tensor(real), torch.tensor(imag)


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')



class AIREADIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        calorie_path,
        glucose_path,
        window_size=128,
        predict_length=64,
        stride=64,
    ):
        self.window_size = window_size
        self.pred_length = predict_length

        self.calorie_df = pd.read_parquet(calorie_path)
        self.glucose_df = pd.read_parquet(glucose_path)


        def extract_pid(x):
            if x is None:
                return None

            if isinstance(x, (list, np.ndarray)):
                if len(x) == 0:
                    return None
                return x[0]

            return x

        self.calorie_df = self.calorie_df[~self.calorie_df["is_missing"]].reset_index(drop=True)


        self.calorie_df["pid"] = self.calorie_df["patient_id"].apply(extract_pid)
        self.glucose_df["pid"] = self.glucose_df["patient_id"].apply(extract_pid)

        # 分别找到各自有效 pid
        cal_ids = set(self.calorie_df["pid"].dropna())
        glu_ids = set(self.glucose_df["pid"].dropna())

        self.common_ids = list(cal_ids & glu_ids)

        # 过滤 dataframe
        self.calorie_df = self.calorie_df[
            self.calorie_df["pid"].apply(lambda x: x in self.common_ids)
        ].reset_index(drop=True)

        self.glucose_df = self.glucose_df[
            self.glucose_df["pid"].apply(lambda x: x in self.common_ids)
        ].reset_index(drop=True)

        # 建 index
        self.calorie_map = {
            row['pid']: row
            for _, row in self.calorie_df.iterrows()
        }

        self.glucose_map = {
            row['pid']: row
            for _, row in self.glucose_df.iterrows()
        }

        # ========= 新增：calorie 插值 =========
        self.aligned_data = {}

        for pid in self.common_ids:

            cal = self.calorie_map[pid]
            glu = self.glucose_map[pid]

            # ===== 1️⃣ 取数据 =====
            t_cal = np.array(cal["time_local"], dtype="datetime64[ns]")
            x_cal = np.array(cal["calorie"], dtype=np.float32)

            t_glu = np.array(glu["time_local"], dtype="datetime64[ns]")
            x_glu = np.array(glu["glucose"], dtype=np.float32)

            # ===== 2️⃣ 转时间为 float（秒）=====
            t_cal_float = t_cal.astype("int64") / 1e9
            t_glu_float = t_glu.astype("int64") / 1e9

            # ===== 3️⃣ 插值 =====
            if len(t_cal_float) > 1:
                interp_cal = np.interp(
                    t_glu_float,
                    t_cal_float,
                    x_cal,
                    left=0.0,
                    right=0.0
                )

                # mask：哪些是有效插值范围
                mask = (t_glu_float >= t_cal_float.min()) & (t_glu_float <= t_cal_float.max())

            else:
                interp_cal = np.zeros_like(x_glu, dtype=np.float32)
                mask = np.zeros_like(x_glu, dtype=bool)

            # ===== 4️⃣ 存起来 =====
            self.aligned_data[pid] = {
                "glucose": x_glu,
                "calorie_interp": interp_cal,
                "calorie_mask": mask.astype(np.float32),
                "time": t_glu
            }

        # ========= sliding window =========
        self.samples = []

        for pid in self.common_ids:

            data = self.aligned_data[pid]

            glu = data["glucose"]
            cal = data["calorie_interp"]
            mask = data["calorie_mask"]

            L = len(glu)

            # ❗防止长度不够
            if L < window_size:
                continue

            for start in range(0, L - window_size + 1, stride):
                end = start + window_size

                self.samples.append({
                    "glucose": glu[start:end],
                    "calorie": cal[start:end],
                    "mask": mask[start:end],
                    "pid": pid
                })

        # ========= compute normalization stats =========
        # all_glu = []
        # all_cal = []
        #
        # for s in self.samples:
        #     all_glu.append(s["glucose"])
        #     all_cal.append(s["calorie"])
        #
        # all_glu = np.concatenate(all_glu)
        # all_cal = np.concatenate(all_cal)
        #
        # self.glu_min = all_glu.min()
        # self.glu_max = all_glu.max()
        #
        # self.cal_min = all_cal.min()
        # self.cal_max = all_cal.max()
        #
        # print("Glucose min/max:", self.glu_min, self.glu_max)
        # print("Calorie min/max:", self.cal_min, self.cal_max)

        self.glu_min = 40.0
        self.glu_max = 401.0
        self.cal_min = 0.0
        self.cal_max = 4841.48

    def normalize(self, x, x_min, x_max):
        return 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        glucose = torch.from_numpy(s["glucose"]).unsqueeze(-1).float()
        glucose = self.normalize(glucose, self.cal_min, self.cal_max)

        mask_ts = torch.ones_like(glucose)
        mask_ts[-self.pred_length:] = 0

        calorie = torch.from_numpy(s["calorie"]).unsqueeze(-1).float()
        calorie = self.normalize(calorie, self.cal_min, self.cal_max)

        context = calorie[: self.window_size - self.pred_length]


        return glucose, mask_ts, context



class DatasetForPrecomputedEmbed(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return index, self.data[index]


class MujocoDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, data_name, path, missing_rate=0.0):
        # import pdb;pdb.set_trace()
        import pathlib
        here = pathlib.Path(__file__).resolve().parent.parent
        base_loc = here / 'data'
        loc = pathlib.Path(path)
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.samples = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)
        else:
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)
            if not os.path.exists(loc):
                os.mkdir(loc)
            loc = here / 'data' / data_name
            tensors = load_data(loc)
            time = tensors['train_X'][:, :, :1].cpu().numpy()
            data = tensors['train_X'][:, :, 1:].reshape(-1, 14).cpu().numpy()

            self.original_sample = []
            norm_data = normalize(data)
            norm_data = norm_data.reshape(4620, seq_len, 14)
            idx = torch.randperm(len(norm_data))

            for i in range(len(norm_data)):
                self.original_sample.append(norm_data[idx[i]].copy())
            self.X_mean = np.mean(np.array(self.original_sample), axis=0).reshape(1,
                                                                                  np.array(self.original_sample).shape[
                                                                                      1],
                                                                                  np.array(self.original_sample).shape[
                                                                                      2])
            generator = torch.Generator().manual_seed(56789)
            for i in range(len(norm_data)):
                removed_points = torch.randperm(norm_data[i].shape[0], generator=generator)[
                                 :int(norm_data[i].shape[0] * missing_rate)].sort().values
                norm_data[i][removed_points] = float('nan')
            norm_data = np.concatenate((norm_data, time), axis=2)
            self.samples = []
            for i in range(len(norm_data)):
                self.samples.append(norm_data[idx[i]])

            self.samples = np.array(self.samples)

            norm_data_tensor = torch.Tensor(self.samples[:, :, :-1]).float().cuda()

            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
            self.last = torch.Tensor(self.samples[:, :, -1][:, -1]).float()
            self.original_sample = torch.tensor(self.original_sample)
            self.samples = torch.tensor(self.samples)
            loc = here / 'data' / (data_name + str(missing_rate))
            save_data(loc, data=self.samples,
                      original_data=self.original_sample
                      )
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

    def __getitem__(self, index):
        return self.original_sample[index], self.samples[index]

    def __len__(self):
        return len(self.samples)
