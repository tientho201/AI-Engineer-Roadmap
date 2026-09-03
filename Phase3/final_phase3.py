"""
train_minigpt.py — Dự án tổng hợp Phase 3 (nâng cao)
- BPE tokenizer (tokenizers)
- RoPE thay positional embedding
- Validation + early stopping
- TensorBoard logging
- Attention map visualization
- So sánh temperature khi sinh văn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mini_gpt_v2 import MiniGPTRoPE

SCRIPT_DIR = Path(__file__).parent
TEXT_PATH = SCRIPT_DIR / "kieu.txt"
TOKENIZER_PATH = SCRIPT_DIR / "bpe_tokenizer.json"
RUNS_DIR = SCRIPT_DIR / "runs"
ATTN_DIR = SCRIPT_DIR / "attention_maps"


def train_bpe_tokenizer(text_path: Path, vocab_size: int = 800) -> Tokenizer:
    if TOKENIZER_PATH.exists():
        return Tokenizer.from_file(str(TOKENIZER_PATH))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>"],
        show_progress=True,
    )
    tokenizer.train([str(text_path)], trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    return tokenizer


class BPEDataset(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, block_size: int = 128):
        self.tokenizer = tokenizer
        self.data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
        self.block_size = block_size

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(s).ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.tolist())

    def token_strings(self, ids: torch.Tensor) -> list[str]:
        return [self.tokenizer.id_to_token(int(i)) for i in ids.tolist()]

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, i: int):
        chunk = self.data[i : i + self.block_size + 1]
        return chunk[:-1], chunk[1:]


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        total += loss.item() * len(x)
        n += len(x)
    return total / n


def plot_attention_maps(attn_maps, token_labels, save_dir: Path, sample_name: str = "layer0"):
    """attn_maps[0]: (B, h, L, L) — vẽ 8 head của layer đầu."""
    attn = attn_maps[0][0].detach().cpu().float()  # (h, L, L)
    n_heads = attn.size(0)
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n_heads > 1 else [axes]

    labels = token_labels[: attn.size(-1)]
    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attn[h], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Head {h}")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(f"Attention maps — {sample_name}", fontsize=12)
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{sample_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Đã lưu attention map: {out}")


def compare_temperatures(model, ds: BPEDataset, device, prompt: str, temperatures=(0.3, 0.8, 1.5)):
    model.eval()
    ctx = ds.encode(prompt)[None].to(device)
    print("\n=== So sánh temperature ===")
    for temp in temperatures:
        out = model.generate(ctx, max_new_tokens=120, temperature=temp, top_k=40)
        text = ds.decode(out[0])
        print(f"\n[T={temp}]")
        print(text[:300] + ("..." if len(text) > 300 else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    text = TEXT_PATH.read_text(encoding="utf-8")
    tokenizer = train_bpe_tokenizer(TEXT_PATH, vocab_size=800)
    ds = BPEDataset(text, tokenizer, block_size=args.block_size)
    if len(ds) < 10:
        raise ValueError(
            f"Corpus quá ngắn ({len(tokenizer.encode(text).ids)} token) "
            f"cho block_size={args.block_size}. Giảm block_size hoặc thêm text vào kieu.txt."
        )

    n_val = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPTRoPE(
        ds.vocab_size, d_model=256, n_heads=8,
        n_layers=6, max_len=args.block_size, dropout=0.1,
    ).to(device)

    if device == "cuda":
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
    )

    writer = SummaryWriter(log_dir=str(RUNS_DIR / "minigpt_bpe_rope"))
    best_val, wait, global_step = float("inf"), 0, 0

    print(f"vocab BPE: {ds.vocab_size} | train: {n_train} | val: {n_val} | device: {device}")

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_n = 0.0, 0
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            train_loss += loss.item() * len(x)
            train_n += len(x)
            writer.add_scalar("loss/train_step", loss.item(), global_step)
            global_step += 1

            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        avg_train = train_loss / train_n
        avg_val = evaluate(model, val_dl, device)
        writer.add_scalar("loss/train_epoch", avg_train, epoch)
        writer.add_scalar("loss/val", avg_val, epoch)
        writer.add_scalar("lr", sched.get_last_lr()[0], epoch)
        print(f"epoch {epoch} | train {avg_train:.4f} | val {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            wait = 0
            torch.save(model.state_dict(), SCRIPT_DIR / "best_minigpt.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping tại epoch {epoch}")
                break

    # Load best weights
    ckpt = SCRIPT_DIR / "best_minigpt.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    # Attention map — 1 câu ngắn, layer 0 (8 heads)
    model.eval()
    sample_text = "Trăm năm trong cõi"
    sample_ids = ds.encode(sample_text)[None].to(device)
    with torch.no_grad():
        _, _, attn_maps = model(sample_ids, return_attn=True)
    labels = ds.token_strings(sample_ids[0])
    plot_attention_maps(attn_maps, labels, ATTN_DIR, sample_name="layer0_first_block")

    compare_temperatures(model, ds, device, prompt="Trăm năm")
    writer.close()
    print(f"\nTensorBoard: tensorboard --logdir {RUNS_DIR}")


if __name__ == "__main__":
    main()
