"""UNet++ oil spill detector training script.

Trains UNet++ with SCSE attention on Zenodo/DARTIS SAR oil spill dataset.
Features:
    - ResNet34 encoder with ImageNet pretrained weights
    - BCE + Dice combined loss
    - AdamW optimizer with cosine annealing LR schedule
    - Mixed precision (AMP) training
    - Validation IoU tracking with model checkpointing
    - TensorBoard logging

Usage:
    python ml/training/train_detector.py --data-dir ./data/dartis --epochs 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "detect"))
from app.models.unetpp_scse import UNetPlusPlusSCSE
from augmentation import get_train_transforms, get_val_transforms
from dataset import SAROilSpillDataset

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds_binary = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds_binary * targets).sum()
    union = preds_binary.sum() + targets.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_f1(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds_binary = (torch.sigmoid(preds) > threshold).float()
    tp = (preds_binary * targets).sum()
    fp = (preds_binary * (1 - targets)).sum()
    fn = ((1 - preds_binary) * targets).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1).float()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(logits.detach(), masks)
        n_batches += 1

        if batch_idx % 20 == 0:
            print(
                f"  [Batch {batch_idx}/{len(loader)}] loss={loss.item():.4f} iou={compute_iou(logits.detach(), masks):.4f}"
            )

    return {"loss": total_loss / n_batches, "iou": total_iou / n_batches}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1).float()

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_iou += compute_iou(logits, masks)
        total_f1 += compute_f1(logits, masks)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "iou": total_iou / n_batches,
        "f1": total_f1 / n_batches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNet++ oil spill detector")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to DARTIS dataset root")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/detector",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--in-channels", type=int, default=6)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--log-dir", type=str, default="./runs/detector")
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[train] Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_transforms = get_train_transforms(args.image_size)
    val_transforms = get_val_transforms(args.image_size)

    full_dataset = SAROilSpillDataset(args.data_dir, transform=None)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))

    train_dataset = SAROilSpillDataset(
        args.data_dir, indices=train_indices, transform=train_transforms
    )
    val_dataset = SAROilSpillDataset(args.data_dir, indices=val_indices, transform=val_transforms)

    print(f"[train] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNetPlusPlusSCSE(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=args.in_channels,
        classes=1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model params: {total_params:,} total, {trainable_params:,} trainable")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    scaler = (
        torch.amp.GradScaler(device.type)
        if args.mixed_precision and device.type == "cuda"
        else None
    )

    writer = SummaryWriter(log_dir=args.log_dir)

    start_epoch = 0
    best_iou = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_iou = ckpt.get("val_iou", 0.0)
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"[train] Resumed from epoch {start_epoch}, best IoU={best_iou:.4f}")

    print(f"[train] Starting training: {args.epochs} epochs, lr={args.lr}")
    print(f"[train] Output: {args.output_dir}, Logs: {args.log_dir}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 60}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        scheduler.step()

        val_metrics = validate(model, val_loader, criterion, device)

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/iou", train_metrics["iou"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/iou", val_metrics["iou"], epoch)
        writer.add_scalar("val/f1", val_metrics["f1"], epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        print(f"[train] Train Loss={train_metrics['loss']:.4f} IoU={train_metrics['iou']:.4f}")
        print(
            f"[train] Val   Loss={val_metrics['loss']:.4f} IoU={val_metrics['iou']:.4f} F1={val_metrics['f1']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_iou": val_metrics["iou"],
            "val_f1": val_metrics["f1"],
            "args": vars(args),
        }
        if scaler:
            checkpoint["scaler"] = scaler.state_dict()

        torch.save(checkpoint, os.path.join(args.output_dir, "last.pth"))

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(checkpoint, os.path.join(args.output_dir, "unetpp_scse_best.pth"))
            print(f"[train] New best IoU: {best_iou:.4f} -> saved")

    writer.close()
    print(f"\n[train] Training complete. Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
