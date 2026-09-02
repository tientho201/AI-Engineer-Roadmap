from contextlib import nullcontext
import os
from pathlib import Path

import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

@dataclass
class DataConfig:
    batch_size: int = 32
    learning_rate: float = 0.01
    epochs: int = 100
    weigh_decay: float = 0.01
    grad_clip: float = 1.0
    patience: int = 5
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True


def train(model: nn.Module, train_ds: Dataset, val_ds: Dataset, data_config: DataConfig, device: str = "cuda"):
    train_loader = DataLoader(train_ds, batch_size=data_config.batch_size, shuffle=True, num_workers=data_config.num_workers, pin_memory=data_config.pin_memory, drop_last=data_config.drop_last)
    val_loader = DataLoader(
        val_ds,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=data_config.learning_rate,
        weight_decay=data_config.weigh_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=data_config.learning_rate, total_steps=data_config.epochs * len(train_loader)
    )
    
    scaler = torch.amp.GradScaler(device=device) # mixed precision -> nhanh hơn ~2x
    best_val, wait = float("inf"), 0
    use_amp = device != "cpu"
    non_blocking = data_config.pin_memory and use_amp
    
    for epoch in range(data_config.epochs):
        # ---------- TRAIN ----------
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            # non_blocking=True cho phép copy tensor từ CPU -> GPU không đồng bộ (asynchronous)
            xb, yb = xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
            amp_ctx = (
                torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                y_hat = model(xb)
                loss = criterion(y_hat, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), data_config.grad_clip) 
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        total_loss = total_loss / len(train_loader)
        
        
        # ---------- VALIDATION ----------
        model.eval()
        total , correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_hat = model(xb)
                loss = criterion(y_hat, yb)
                total += loss.item() * len(xb)
                correct += (y_hat.argmax(dim=1) == yb).sum().item()
        val_loss =  total / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        print(f"epoch {epoch:3d} | train {total_loss:.4f} | "
              f"val {val_loss:.4f} | acc {val_acc:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")
        
        # ---------- SAVE BEST MODEL ----------
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}, "best.pt")
        else:
            wait += 1
            if wait >= data_config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    checkpoint = torch.load("best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model, best_val

