"""Train UNet++ SCSE model on synthetic SAR oil spill data.

Run this inside the sentinel-detect Docker container after data is loaded.
"""

from __future__ import annotations

import os
import sys

import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from app.models.unetpp_scse import UNetPlusPlusSCSE
from augmentation import get_train_transforms, get_val_transforms
from dataset import SAROilSpillDataset


def main():
    # Configuration
    data_dir = os.getenv("SYNTHETIC_DATA_DIR", "data/synthetic")
    epochs = int(os.getenv("TRAIN_EPOCHS", "5"))
    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "4"))
    image_size = int(os.getenv("IMAGE_SIZE", "512"))
    lr = float(os.getenv("LR", "1e-4"))
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info(f"Training on device: {device}")

    # Load datasets
    train_dataset = SAROilSpillDataset(
        os.path.join(data_dir, "train"),
        transform=get_train_transforms(image_size),
        image_size=image_size,
    )
    val_dataset = SAROilSpillDataset(
        os.path.join(data_dir, "val"),
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize model: UNet++ with SCSE attention, ResNet34 encoder, ImageNet pretrained
    model = UNetPlusPlusSCSE(
        encoder_name="resnet34",
        encoder_weights="imagenet",  # ImageNet pretrained encoder
        in_channels=3,
        classes=1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} params, {trainable_params:,} trainable")

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_iou = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            loss.backward()
            torch.nn.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        # Validation
        model.eval()
        val_iou = 0.0
        val_count = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(images)
                probs = torch.sigmoid(logits)

                # Compute IoU
                pred_mask = (probs > 0.5).float()
                intersection = (pred_mask * masks).sum()
                union = pred_mask.sum() + masks.sum() - intersection
                iou = (intersection + 1e-7) / (union + 1e-7)
                val_iou += iou.item()
                val_count += 1

        val_iou /= max(val_count, 1)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} -> Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}"
        )

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt_path = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pth")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_iou": val_iou,
                    "arch": "UNetPlusPlusSCSE",
                },
                ckpt_path,
            )
            logger.info(f"Saved best model with Val IoU: {best_val_iou:.4f} -> {ckpt_path}")

    logger.info(f"Training complete. Best Val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
