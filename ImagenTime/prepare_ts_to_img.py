import os, sys
import torch
import numpy as np
import torch.multiprocessing
import logging
from tqdm import tqdm
from metrics import evaluate_model_uncond
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_uncond
import wandb

import torch.utils.data as Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    # set-up data and device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = gen_dataloader(args)

    model = ImagenTime(args=args, device=args.device).to(args.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")
    if args.use_stft:
        model.init_stft_embedder(train_loader)

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


    all_ts_img = []
    for i, data in enumerate(train_loader, 1):
        x_ts = data[0].to(args.device)
        x_img = model.ts_to_img(x_ts)
        all_ts_img.append(x_img)
    all_ts_img = torch.stack(all_ts_img)
    print(f"Image shape: {all_ts_img.shape}")




if __name__ == '__main__':
    args = parse_args_uncond()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
