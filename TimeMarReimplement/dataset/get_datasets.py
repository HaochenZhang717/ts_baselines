import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import importlib

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    # print(cls)
    # breakpoint()
    return cls(**config.get("params", dict()))


def build_dataset(config):
    #  Config 
    dataloader_config = config.dataloader  #  dataloader  Struct 

    batch_size = config.batch_size
    #  params  output_dir
    dataloader_config.params.output_dir = config.output_dir

    dataset = instantiate_from_config(dataloader_config)

    total_size = len(dataset)
    train_size = total_size-batch_size
    val_size = batch_size

    # train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))

    val_dataset = Subset(dataset, val_indices)

    return dataset, val_dataset

def build_dataloader(config, args=None):
    #  Config 
    dataloader_config = config.dataloader

    batch_size = config.batch_size
    try:
        dim=config.dataloader.params.dim
    except:
        print("no dim in config")
        dim=0

    if dim==5:
        if config.dataloader.params.window==24:
            file_path=f'../output/samples/Sines_ground_truth_24_train.npy'
            train_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
            # val_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
            total_size = len(train_dataset)
            train_size = total_size-batch_size
            val_size = batch_size
            val_indices = list(range(train_size, train_size + val_size))
            val_dataset = Subset(train_dataset, val_indices)
            print("data load from:", file_path)
    elif dim==14:
        file_path=f'../output/samples/Mujoco_norm_truth_24_train.npy'
        train_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
        val_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
        print("data load from:", file_path)
    else:
        print("load data from config")
        train_dataset, val_dataset = build_dataset(config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        # pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        # pin_memory=True,
        drop_last=True
    )
    return train_loader, val_loader

from torch.utils.data import ConcatDataset

def build_dataloader_var(config, data, args=None, window=24):
    if data=="Sines":
        file_path=f'../output/samples/{data}_ground_truth_{window}_train.npy'
    elif data=="Mujoco":
        file_path=f'../output/samples/{data}_norm_truth_{window}_train.npy'
    else:
        window = config['dataloader']['params']['window']
        file_path=f'../output/samples/{data}_norm_truth_{window}_train.npy'

    train_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
    ori_dataset = normalize_to_neg_one_to_one(np.load(file_path).astype(np.float32))
    repeat_times=10
    from torch.utils.data import ConcatDataset
    train_dataset = [train_dataset for _ in range(repeat_times)]
    train_dataset = ConcatDataset(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              config.batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              drop_last = False)
    val_loader = torch.utils.data.DataLoader(
        ori_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader

