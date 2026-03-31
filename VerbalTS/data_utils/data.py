import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
import time
import torch

def merge_single_image(embed_dict):
    """
    输入:
        embed_dict = self.embed_0323['image23999']

    输出:
        tensor shape = (4, 3, D)
    """

    segments = sorted(embed_dict.keys())  # seg1, seg2, seg3, seg4

    merged = []

    for seg in segments:
        stage_dict = embed_dict[seg]

        # 固定顺序！
        stages = ["early", "middle", "late"]

        stage_embeds = [stage_dict[s] for s in stages]  # list of (D,)
        stage_embeds = torch.stack(stage_embeds, dim=0)  # (3, D)

        merged.append(stage_embeds)

    merged = torch.stack(merged, dim=0)  # (4, 3, D)

    return merged


class CustomDataset:
    def __init__(self, folder, **kwargs):
        super().__init__()
        self.folder = folder
        self._load_meta()

    def _load_meta(self):
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.attr_list = self.meta["attr_list"]
        n_attr = len(self.attr_list)
        self.attr_ids = np.arange(n_attr)
        self.attr_n_ops = np.array(self.meta["attr_n_ops"])

    def get_split(self, split, *args):
        return CustomSplit(self.folder, split)


class CustomSplit(Dataset):
    def __init__(self, folder, split="train"):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."
        self.split = split            
        self.folder = folder
        self._load_data()

        print(f"Split: {self.split}, total samples {self.n_samples}.")

    def _load_data(self):
        ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, n_steps]
        # caps = np.load(os.path.join(self.folder, self.split+fr"_text_caps.npy"), allow_pickle=True)
        # caps = np.load(os.path.join(self.folder, self.split+fr"_caps_0324.npy"), allow_pickle=True)

        self.ts = ts

        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[1]
        self.time_point = np.arange(self.n_steps)

    def __getitem__(self, idx):
        # cap_id = random.randint(0, len(self.caps[idx])-1)
        tmp_ts = self.ts[idx]
        if len(tmp_ts.shape) == 1:
            tmp_ts = tmp_ts[...,np.newaxis]

        return {"ts": tmp_ts,
                "ts_len": tmp_ts.shape[0],
                "tp": self.time_point}

    def __len__(self):
        return self.n_samples


class MyDataset:
    """
    Wrapper class so that the block-causal dataset fits
    the GenerationDataset interface.
    """

    def __init__(
        self,
        ts_path,
        caps_path,
        vae_embed_path,
        text_embed_path,
        seq_len,
        num_channels,
        num_segments,
        **kwargs
    ):
        self.ts_path = ts_path
        self.caps_path = caps_path
        self.seq_len = seq_len
        self.text_embed_path = text_embed_path
        self.vae_embed_path = vae_embed_path
        self.num_segments = num_segments
        self.num_channels = num_channels

        self.attr_n_ops = None

    def get_split(self, split, text_type=None, *args):
        return MySplit(
            ts_path=self.ts_path,
            caps_path=self.caps_path,
            vae_embed_path=self.vae_embed_path,
            seq_len=self.seq_len,
            text_embed_path=self.text_embed_path,
            num_channels=self.num_channels,
            num_segments=self.num_segments,
            split=split,
        )


class MySplit(Dataset):
    def __init__(
        self,
        ts_path,
        caps_path,
        vae_embed_path,
        text_embed_path,
        seq_len,
        num_channels,
        num_segments,
        split="train",
    ):
        super().__init__()

        self.split = split
        self.num_segments = num_segments
        self.num_channels = num_channels

        self.caps_path = caps_path
        # ------------------------
        # load data
        # ------------------------
        self.ts = None
        self.moment_embed = None
        if ts_path != "none":
            self.ts = np.load(f"{ts_path}/{split}_ts.npy", allow_pickle=True)  # (N,T,C)
            self.N, self.T, self.C = self.ts.shape
            self.my_caps_qwen_embed = torch.load(os.path.join(self.caps_path, self.split + fr"_embeds_qwen_seq.pt"), weights_only=False)

            # self.moment_embed = np.load(f"{ts_path}/{split}_moment_embeds.npy", allow_pickle=True)
        else:
            self.N = -1
            self.T = seq_len
            self.C = num_channels

        if not text_embed_path.endswith('.pt'):
            # self.text_embed = torch.load(f"{text_embed_path}/{split}_embeds.pt", map_location="cpu")
            self.text_embed = torch.load(f"{text_embed_path}/{split}_embed_one_per_segment_0323.pt", map_location="cpu")
        else:
            self.text_embed = torch.load(text_embed_path, map_location="cpu")


        self.vae_embed = None
        if vae_embed_path != "none":
            self.vae_embed = np.load(f"{vae_embed_path}/{split}_vae.npy", allow_pickle=True)
            self.vae_embed = torch.from_numpy(self.vae_embed)


        self.caps = None
        if self.caps_path != "none":
            if not self.caps_path.endswith(".jsonl"):
                caps_dict = {}
                with open(f"{self.caps_path}/{split}_caps_ready.jsonl", "r") as f:
                    for line in f:
                        item = json.loads(line)
                        caps_dict[item["id"]] = item["captions"]
                self.caps = caps_dict
            else:
                caps_dict = {}
                with open(self.caps_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        caps_dict[item["id"]] = item["captions"]
                self.caps = caps_dict

        assert self.T % self.num_segments == 0

        self.segment_length = self.T // self.num_segments

        self.ids = sorted(
            self.text_embed.keys(),
            key=lambda x: int(x.replace("image", "")),
        )

        self.block_ids = list(range(self.num_segments))
        self.num_block_choices = len(self.block_ids)

        print(
            f"[CausalSplit:{self.split}] "
            f"N={self.N}, T={self.T}, C={self.C}, segments={self.num_segments}"
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        ts_id = int(image_id.replace("image", ""))

        if self.ts is not None:
            ts = self.ts[ts_id]  # (T,C)
            ts = torch.from_numpy(ts).float().transpose(0, 1)  # (C,T)
        else:
            ts = torch.zeros((self.C, self.T)).float()

        if self.caps is not None:
            caps = self.caps[image_id]
        else:
            caps = "caps not loaded."

        if self.vae_embed is not None:
            vae_embed = self.vae_embed[ts_id]
        else:
            vae_embed = None

        # ------------------------
        # text embedding
        # ------------------------
        # print("ts shape in getitem ", ts.shape)
        text_embed_all_segments = []
        for target_block in self.block_ids:
            channel_embeds = []
            for c in range(self.C):
                key = f"seg{target_block+1}_channel{c}"
                emb = self.text_embed[image_id][key]
                channel_embeds.append(emb)
            text_embed = torch.stack(channel_embeds, dim=0)  # (C,D)
            text_embed_all_segments.append(text_embed)
        text_embed_all_segments = torch.stack(text_embed_all_segments, dim=0) # (num_segments,C,D)

        caps_qwen = self.my_caps_qwen_embed[image_id]

        example_key = f"seg1_channel0"
        example_tensor = caps_qwen[example_key]
        if not isinstance(example_tensor, torch.Tensor):
            example_tensor = torch.tensor(example_tensor)

        L, D = example_tensor.shape
        my_caps_qwen_embeds = torch.zeros(
            self.C, self.num_segments, L, D,
            dtype=example_tensor.dtype
        )

        for s in range(self.num_segments):
            for c in range(self.C):
                key = f"seg{s + 1}_channel{c}"
                emb = caps_qwen[key]
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)
                my_caps_qwen_embeds[c, s] = emb


        item = {
            "ts": ts,
            "ts_len": self.T,
            "text_embedding_all_segments": text_embed_all_segments,
            "my_caps_qwen_embeds": my_caps_qwen_embeds,
            "image_id": image_id,
            "ts_id": ts_id,
            "caps": caps,
            "vae_embeds": vae_embed,
            "moment_embed": torch.from_numpy(self.moment_embed[idx]).float() if self.moment_embed is not None else None,
            # 'attn_mask': build_block_causal_mask(self.T, text_embed_all_segments.shape[0])
        }

        return item

    @staticmethod
    def collate_fn(batch):
        out = {}
        out["ts"] = torch.stack([b["ts"] for b in batch])
        out["my_caps_qwen_embeds"] = torch.stack([b["my_caps_qwen_embeds"] for b in batch])
        out["vae_embeds"] = torch.stack([b["vae_embeds"] for b in batch]) if batch[0]["vae_embeds"] is not None else None
        out["ts_len"] = torch.tensor([b["ts_len"] for b in batch])
        out["text_embedding_all_segments"] = torch.stack([b["text_embedding_all_segments"] for b in batch])
        out["image_id"] = [b["image_id"] for b in batch]
        out["ts_id"] = torch.tensor([b["ts_id"] for b in batch])
        out["caps"] = [b["caps"] for b in batch]
        out["moment_embed"] = torch.stack([b["moment_embed"] for b in batch]) if batch[0]["moment_embed"] is not None else None

        # out["attn_mask"] = torch.stack([b["attn_mask"] for b in batch])

        return out


class QwenV3Dataset:
    """
    Wrapper class so that the block-causal dataset fits
    the GenerationDataset interface.
    """

    def __init__(
        self,
        ts_path,
        caps_path,
        vae_embed_path,
        text_embed_path,
        seq_len,
        num_channels,
        num_segments,
        **kwargs
    ):
        self.ts_path = ts_path
        self.caps_path = caps_path
        self.seq_len = seq_len
        self.text_embed_path = text_embed_path
        self.vae_embed_path = vae_embed_path
        self.num_segments = num_segments
        self.num_channels = num_channels

        self.attr_n_ops = None

    def get_split(self, split, text_type=None, *args):
        return QwenV3Split(
            ts_path=self.ts_path,
            caps_path=self.caps_path,
            vae_embed_path=self.vae_embed_path,
            seq_len=self.seq_len,
            text_embed_path=self.text_embed_path,
            num_channels=self.num_channels,
            num_segments=self.num_segments,
            split=split,
        )


class QwenV3Split(Dataset):
    def __init__(
        self,
        ts_path,
        caps_path,
        vae_embed_path,
        text_embed_path,
        seq_len,
        num_channels,
        num_segments,
        split="train",
    ):
        super().__init__()

        self.split = split
        self.num_segments = num_segments
        self.num_channels = num_channels

        self.caps_path = caps_path
        # ------------------------
        # load data
        # ------------------------
        self.ts = None
        self.moment_embed = None
        if ts_path != "none":
            self.ts = np.load(f"{ts_path}/{split}_ts.npy", allow_pickle=True)  # (N,T,C)
            self.N, self.T, self.C = self.ts.shape
        else:
            self.N = -1
            self.T = seq_len
            self.C = num_channels

        # if not text_embed_path.endswith('.pt'):
            # self.text_embed = torch.load(f"{text_embed_path}/{split}_embeds.pt", map_location="cpu")
            # self.text_embed = torch.load(f"{text_embed_path}/{split}_caps_embeds.pt", map_location="cpu")
        # else:
        #     self.text_embed = torch.load(text_embed_path, map_location="cpu")


        # self.vae_embed = None
        # if vae_embed_path != "none":
        #     self.vae_embed = np.load(f"{vae_embed_path}/{split}_vae.npy", allow_pickle=True)
        #     self.vae_embed = torch.from_numpy(self.vae_embed)


        self.caps = None
        # if self.caps_path != "none":
        #     if not self.caps_path.endswith(".jsonl"):
        #         caps_dict = {}
        #         with open(f"{self.caps_path}/{split}_caps_ready.jsonl", "r") as f:
        #             for line in f:
        #                 item = json.loads(line)
        #                 caps_dict[item["id"]] = item["captions"]
        #         self.caps = caps_dict
        #     else:
        #         caps_dict = {}
        #         with open(self.caps_path, "r") as f:
        #             for line in f:
        #                 item = json.loads(line)
        #                 caps_dict[item["id"]] = item["captions"]
        #         self.caps = caps_dict

        assert self.T % self.num_segments == 0

        self.segment_length = self.T // self.num_segments

        self.ids = sorted(
            self.text_embed.keys(),
            key=lambda x: int(x.replace("image", "")),
        )

        self.block_ids = list(range(self.num_segments))
        self.num_block_choices = len(self.block_ids)

        print(
            f"[CausalSplit:{self.split}] "
            f"N={self.N}, T={self.T}, C={self.C}, segments={self.num_segments}"
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        ts_id = int(image_id.replace("image", ""))

        if self.ts is not None:
            ts = self.ts[ts_id]  # (T,C)
            ts = torch.from_numpy(ts).float().transpose(0, 1)  # (C,T)
        else:
            ts = torch.zeros((self.C, self.T)).float()

        if self.caps is not None:
            caps = self.caps[image_id]
        else:
            caps = "caps not loaded."

        if self.vae_embed is not None:
            vae_embed = self.vae_embed[ts_id]
        else:
            vae_embed = None

        # ------------------------
        # text embedding
        # ------------------------
        # print("ts shape in getitem ", ts.shape)
        text_embed_all_segments = []
        for target_block in self.block_ids:
            channel_embeds = []
            for c in range(self.C):
                key = f"seg{target_block+1}_channel{c}"
                emb = self.text_embed[image_id][key]
                channel_embeds.append(emb)
            text_embed = torch.stack(channel_embeds, dim=0)  # (C,D)
            text_embed_all_segments.append(text_embed)
        text_embed_all_segments = torch.stack(text_embed_all_segments, dim=0) # (num_segments,C,D)

        item = {
            "ts": ts,
            "ts_len": self.T,
            "text_embedding_all_segments": text_embed_all_segments,
            "image_id": image_id,
            "ts_id": ts_id,
            "caps": caps,
            "vae_embeds": vae_embed,
            "moment_embed": torch.from_numpy(self.moment_embed[idx]).float() if self.moment_embed is not None else None,
        }
        return item

    @staticmethod
    def collate_fn(batch):
        out = {}
        out["ts"] = torch.stack([b["ts"] for b in batch])
        out["vae_embeds"] = torch.stack([b["vae_embeds"] for b in batch]) if batch[0]["vae_embeds"] is not None else None
        out["ts_len"] = torch.tensor([b["ts_len"] for b in batch])
        out["text_embedding_all_segments"] = torch.stack([b["text_embedding_all_segments"] for b in batch])
        out["image_id"] = [b["image_id"] for b in batch]
        out["ts_id"] = torch.tensor([b["ts_id"] for b in batch])
        out["caps"] = [b["caps"] for b in batch]
        out["moment_embed"] = torch.stack([b["moment_embed"] for b in batch]) if batch[0]["moment_embed"] is not None else None
        return out


class V7Dataset:
    def __init__(self, folder, my_caps_path, **kwargs):
        super().__init__()
        self.folder = folder
        self.my_caps_path = my_caps_path
        self._load_meta()

    def _load_meta(self):
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.attr_list = self.meta["attr_list"]
        n_attr = len(self.attr_list)
        self.attr_ids = np.arange(n_attr)
        self.attr_n_ops = np.array(self.meta["attr_n_ops"])

    def get_split(self, split, *args):
        return V7Split(self.folder, self.my_caps_path, split)


class V7Split(Dataset):
    def __init__(self, folder, my_caps_path, split="train"):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."
        self.split = split
        self.folder = folder
        self.my_caps_path = my_caps_path
        self._load_data()

        print(f"Split: {self.split}, total samples {self.n_samples}.")

    def _load_data(self):
        # ===== load ts =====
        ts = np.load(os.path.join(self.folder, self.split + "_ts.npy"))  # [N, T]

        self.ts = ts
        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[1]
        self.time_point = np.arange(self.n_steps)


    def __getitem__(self, idx):
        # ===== time series =====
        tmp_ts = self.ts[idx]
        if len(tmp_ts.shape) == 1:
            tmp_ts = tmp_ts[..., np.newaxis]  # (T, 1)

        # ===== my_cap（处理成 string）=====
        return {
            "ts": tmp_ts.astype(np.float32),
            "ts_len": tmp_ts.shape[0],
            "tp": self.time_point.astype(np.float32),
            "idx": idx
        }

    def __len__(self):
        return self.n_samples

    @staticmethod
    def collate_fn(batch):
        out = {}

        # ===== tensor部分 =====
        out["ts"] = torch.stack([
            torch.from_numpy(b["ts"]) for b in batch
        ])  # (B, T, C)

        out["tp"] = torch.stack([
            torch.from_numpy(b["tp"]) for b in batch
        ])  # (B, T)

        # ===== string部分（保持list）=====
        out["indices"] = [b["idx"] for b in batch]
        return out