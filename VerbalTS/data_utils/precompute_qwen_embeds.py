import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np



def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# =========================
# Qwen encoder（无 projector）
# =========================
class QwenTextEncoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B",
            padding_side="left"
        )
        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B"
        ).to(device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts):
        batch = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(**batch)  # (B, L, 1024)
        embeddings = last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


# =========================
# 主逻辑
# =========================
def precompute(
    caps_path,
    save_path,
    split="train",
    batch_size=64,
    device="cuda"
):

    print("Loading captions...")

    caps_dict = {}
    all_caps = np.load(f"{caps_path}/{split}_caps_0324.npy", allow_pickle=True)

    print(f"Loaded {len(all_caps)} samples")

    encoder = QwenTextEncoder(device).to(device)
    embeds_all = []
    for i in tqdm(range(0, len(all_caps), batch_size)):
        batch_text = [cap[0] for cap in all_caps[i:i + batch_size]]

        embeds = encoder(batch_text)  # (b, L, 1024)
        embeds_all.append(embeds.cpu())

    embeds_all = torch.cat(embeds_all, dim=0)
    breakpoint()
    # =========================
    # 保存
    # =========================
    torch.save(embeds_all, save_path)

    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--caps_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    precompute(
        caps_path=args.caps_path,
        save_path=args.save_path,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )