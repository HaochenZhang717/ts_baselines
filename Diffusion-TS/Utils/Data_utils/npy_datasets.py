import os
import torch
import numpy as np
from torch.utils.data import Dataset
from Utils.masking_utils import noise_mask


class PreSplitNPYDataset(Dataset):
    def __init__(
        self,
        name,
        data_root,
        window,
        period='train',
        split='train',
        seed=123,
        predict_length=None,
        missing_ratio=None,
        style='separate',
        distribution='geometric',
        mean_mask_length=3,
        output_dir=None,
        **kwargs
    ):
        super().__init__()

        assert split in ['train', 'valid', 'test']
        assert period in ['train', 'test']

        self.name = name
        self.data_root = data_root
        self.window = window
        self.period = period
        self.split = split
        self.seed = seed
        self.pred_len = predict_length
        self.missing_ratio = missing_ratio
        self.style = style
        self.distribution = distribution
        self.mean_mask_length = mean_mask_length
        self.output_dir = output_dir

        file_map = {
            'train': 'train_ts.npy',
            'valid': 'valid_ts.npy',
            'test': 'test_ts.npy'
        }

        path = os.path.join(data_root, file_map[split])
        self.samples = np.load(path).astype(np.float32)

        assert self.samples.ndim == 3, f"Expected (N,L,D), got {self.samples.shape}"
        assert self.samples.shape[1] == window, (
            f"Window mismatch: got {self.samples.shape[1]}, expected {window}"
        )

        self.sample_num = self.samples.shape[0]

        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape, dtype=bool)
                masks[:, -predict_length:, :] = 0
                self.masking = masks
            else:
                raise NotImplementedError(
                    "For period='test', either missing_ratio or predict_length must be set."
                )

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples, dtype=bool)

        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx]
            mask = noise_mask(
                x,
                self.missing_ratio,
                self.mean_mask_length,
                self.style,
                self.distribution
            )
            masks[idx] = mask

        np.random.set_state(st0)
        return masks

    def __getitem__(self, ind):
        if self.period == 'test':
            return (
                torch.from_numpy(self.samples[ind]).float(),
                torch.from_numpy(self.masking[ind])
            )
        return torch.from_numpy(self.samples[ind]).float()

    def __len__(self):
        return self.sample_num