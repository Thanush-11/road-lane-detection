import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TuSimpleSegmentationDataset
from src.models import UNetSmall


def iou_score(pred, target, eps=1e-6):
    # pred/target: (B,1,H,W) binary 0/1
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred*target).sum(dim=(1,2,3))
    return ((inter + eps) / (union + eps)).mean().item()


def train(
    processed_root="data/processed/tusimple",
    out_dir="models/checkpoints",
    epochs=10,
    batch_size=8,
    lr=1e-3,
    img_size=(360, 640),
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    train_ds = TuSimpleSegmentationDataset(processed_root, "train", img_size=img_size)
    val_ds = TuSimpleSegmentationDataset(processed_root, "val", img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNetSmall().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        train_loss = 0.0

        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            loss = loss_fn(logits, masks)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= max(1, len(train_loader))

        # Validate
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                val_iou += iou_score(pred, masks)

        val_loss /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))

        ckpt_path = Path(out_dir) / f"unet_tusimple_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_iou": val_iou}, ckpt_path)

        if val_iou > best_iou:
            best_iou = val_iou
            best_path = Path(out_dir) / "unet_tusimple_best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_iou": val_iou}, best_path)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f} best_iou={best_iou:.4f}")

    print(f"Training done. Best checkpoint: {Path(out_dir) / 'unet_tusimple_best.pt'}")


if __name__ == "__main__":
    train()
