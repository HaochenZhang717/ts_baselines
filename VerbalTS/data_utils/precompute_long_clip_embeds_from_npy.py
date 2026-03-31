import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import numpy as np


# =========================
# CLIP Text Encoder
# =========================
class ClipTextEncoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        model_path = "/playpen-shared/haochenz/long_clip"

        self.model = CLIPTextModelWithProjection.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 正确 max length
        self.max_length = min(
            self.model.config.max_position_embeddings,
            self.tokenizer.model_max_length
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, text_list):
        inputs = self.tokenizer(
            text_list,
            padding="max_length",  # ✅ 关键！
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_emb = outputs.last_hidden_state  # (B, L, D)

        return text_emb, attention_mask


# =========================
# merge captions
# =========================
def merge_caps(caps_list):
    """
    caps_list:
    [
        {'seg1_channel0': '...'},
        {'seg2_channel0': '...'},
        ...
    ]

    → single string
    """
    merged = []
    for d in caps_list:
        for k, v in d.items():
            merged.append(f"{k}: {v}")

    return "\n".join(merged)


# =========================
# 主逻辑
# =========================
def precompute_from_jsonl(
    caps_path,
    save_path,
    split="train",
    batch_size=64,
    device="cuda"
):

    print("Loading captions...")

    caps_dict = {}
    with open(f"{caps_path}/{split}_caps_ready.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            caps_dict[item["id"]] = item["captions"]

    print(f"Loaded {len(caps_dict)} samples")

    encoder = ClipTextEncoder(device).to(device)

    all_texts = []
    image_ids = []

    # =========================
    # 先把所有 caption merge
    # =========================
    for image_id in caps_dict:
        caps_list = caps_dict[image_id]
        merged_text = merge_caps(caps_list)

        all_texts.append(merged_text)
        image_ids.append(image_id)

    print("Start encoding...")

    all_embeds = []

    # =========================
    # batch encode
    # =========================
    for i in tqdm(range(0, len(all_texts), batch_size)):
        batch_text = all_texts[i:i + batch_size]
        embeds = encoder(batch_text)  # (B, L, D)
        all_embeds.append(embeds.cpu())

    # =========================
    # 拼成 (N, L, D)
    # =========================
    all_embeds = torch.cat(all_embeds, dim=0)  # (N, L, D)

    # =========================
    # 保存
    # =========================
    torch.save({
        "embeddings": all_embeds,   # (N, L, D)
        "ids": image_ids
    }, save_path)

    print(f"Saved to {save_path}")
    print("Shape:", all_embeds.shape)


def precompute_from_npy(
    caps_path,
    save_path,
    npy_name,
    split="train",
    batch_size=64,
    device="cuda"
):

    print("Loading captions...")
    caps = np.load(f"{caps_path}/{split}_{npy_name}.npy", allow_pickle=True)
    encoder = ClipTextEncoder(device).to(device)

    all_embeds = []
    all_masks = []
    # =========================
    # batch encode
    # =========================

    for i in tqdm(range(0, len(caps), batch_size)):
        batch_caps = caps[i:i + batch_size]  # (B, C)

        B, C = batch_caps.shape

        # flatten
        flat_text = []
        for b in range(B):
            for c in range(C):
                flat_text.append(batch_caps[b][c])

        embeds, attn_masks = encoder(flat_text)  # (B*C, L, D)

        # reshape
        embeds = embeds.view(B, C, embeds.shape[1], embeds.shape[2])
        attn_masks = attn_masks.view(B, C, attn_masks.shape[1])

        all_embeds.append(embeds.cpu())
        all_masks.append(attn_masks.cpu())


    # for i in tqdm(range(0, len(caps), batch_size)):
        # batch_text = [cap[0] for cap in caps[i:i + batch_size]]
        # # batch_text = caps[i:i + batch_size]
        # embeds, attn_masks = encoder(batch_text)  # (B, L, D)
        # all_embeds.append(embeds.cpu())
        # all_masks.append(attn_masks.cpu())

    # =========================
    # 拼成 (N, L, D)
    # =========================
    all_embeds = torch.cat(all_embeds, dim=0)  # (N, L, D)
    all_masks = torch.cat(all_masks, dim=0)
    # =========================
    # 保存
    # =========================
    # torch.save(all_embeds, save_path)

    torch.save({
        "embeddings": all_embeds,  # (N, L, D)
        "all_masks": all_masks
    }, save_path)
    print(f"Saved to {save_path}")
    print("Shape:", all_embeds.shape)


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--caps_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--npy_name", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    precompute_from_npy(
        caps_path=args.caps_path,
        save_path=args.save_path,
        npy_name=args.npy_name,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )