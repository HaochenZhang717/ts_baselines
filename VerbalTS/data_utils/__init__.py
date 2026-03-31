from torch.utils.data import DataLoader

from .data import CustomDataset, MyDataset, QwenV3Dataset, V7Dataset

datasets = {
    "custom": CustomDataset,
    "mine": MyDataset,
    "qwen_v3": QwenV3Dataset,
    "v7": V7Dataset,
}

# class GenerationDataset:
#     def __init__(self, configs):
#         self.configs = configs
#         self.dataset = datasets[configs["name"]](**configs)
#
#     def get_loader(self, split, batch_size, shuffle=True, num_workers=1, include_self=False):
#         loader = DataLoader(
#             dataset=self.dataset.get_split(split, include_self),
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers)
#         return loader
#
#     @property
#     def num_attr_ops(self):
#         return self.dataset.attr_n_ops
#



class GenerationDataset:
    def __init__(self, configs):
        self.configs = configs

        dataset_name = configs["name"]
        if dataset_name not in datasets:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        self.dataset = datasets[dataset_name](**configs)

    def get_loader(
        self,
        split,
        batch_size,
        shuffle=True,
        num_workers=16,
        include_self=False,
        pin_memory=True,
        drop_last=False,
    ):
        split_dataset = self.dataset.get_split(split, include_self)

        collate_fn = getattr(split_dataset, "collate_fn", None)

        loader = DataLoader(
            dataset=split_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return loader

    @property
    def num_attr_ops(self):
        return getattr(self.dataset, "attr_n_ops", None)
