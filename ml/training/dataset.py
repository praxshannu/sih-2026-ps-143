"""Dataset loader for Zenodo/DARTIS SAR oil spill datasets.

Loads paired SAR image + mask files with train/val/test splits.
Supports both Zenodo oil spill dataset and DARTIS format.

Expected directory structure:
    data_dir/
        images/
            0001.tif
            0002.tif
        masks/
            0001.tif
            0002.tif

Usage:
    from dataset import SAROilSpillDataset
    ds = SAROilSpillDataset("./data/dartis", transform=get_train_transforms(512))
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SAROilSpillDataset(Dataset):
    """SAR oil spill segmentation dataset.

    Args:
        root: Root directory containing images/ and masks/ subdirectories.
        indices: Optional subset of indices to use (for train/val splits).
        transform: Optional albumentations-style transform applied to
            both image and mask jointly.
        image_size: Resize images to this size if specified.
    """

    def __init__(
        self,
        root: str,
        indices: list[int] | None = None,
        transform: Callable | None = None,
        image_size: int = 512,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.image_size = image_size

        image_dir = self.root / "images"
        mask_dir = self.root / "masks"

        if not image_dir.exists():
            image_dir = self.root / "image"
        if not mask_dir.exists():
            mask_dir = self.root / "mask"

        self.image_paths = sorted(
            list(image_dir.glob("*.tif"))
            + list(image_dir.glob("*.tiff"))
            + list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.jpg"))
        )
        self.mask_paths = sorted(
            list(mask_dir.glob("*.tif"))
            + list(mask_dir.glob("*.tiff"))
            + list(mask_dir.glob("*.png"))
            + list(mask_dir.glob("*.jpg"))
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No SAR images found under {image_dir}. "
                "Prepare a real dataset before training; synthetic fixtures are "
                "available only through scripts/synthetic_train.py."
            )
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image/mask count mismatch: {len(self.image_paths)} images, "
                f"{len(self.mask_paths)} masks"
            )

        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices if i < len(self.image_paths)]
            self.mask_paths = [self.mask_paths[i] for i in indices if i < len(self.mask_paths)]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        if img_path.suffix == ".npy":
            image = np.load(img_path)
        else:
            try:
                import rasterio

                with rasterio.open(img_path) as src:
                    image = src.read().astype(np.float32)
            except ImportError:
                from PIL import Image

                img_pil = Image.open(img_path).convert("RGB")
                image = np.array(img_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0

        if mask_path.suffix == ".npy":
            mask = np.load(mask_path)
        else:
            try:
                import rasterio

                with rasterio.open(mask_path) as src:
                    mask = src.read(1).astype(np.float32)
            except ImportError:
                from PIL import Image

                mask_pil = Image.open(mask_path).convert("L")
                mask = np.array(mask_pil, dtype=np.float32) / 255.0

        if image.ndim == 2:
            image = image[np.newaxis, ...]
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]
        elif mask.ndim == 3:
            mask = mask[0]

        if image.shape[0] > 3:
            image = image[:3]

        if self.transform is not None:
            try:
                img_t = image.transpose(1, 2, 0) if image.ndim == 3 else image
                mask_t = mask.transpose(1, 2, 0) if mask.ndim == 3 else mask
                transformed = self.transform(image=img_t, mask=mask_t)
                image = transformed["image"]
                if image.ndim == 3:
                    image = image.transpose(2, 0)
                mask = transformed["mask"]
            except ImportError:
                pass

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        mask = (mask > 0.5).float()

        # Multimodal contract: production consumes 6 channels
        # (VV, VH, U10, V10, CU, CV). Real Zenodo images have two SAR bands;
        # auxiliary forcing is absent from the image archive, so pad only the
        # missing channels with zeros. This keeps the data contract explicit.
        if image.ndim == 3 and image.shape[0] < 6:
            zeros = torch.zeros(
                (6 - image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype
            )
            image = torch.cat([image, zeros], dim=0)
        elif image.ndim == 3 and image.shape[0] > 6:
            image = image[:6]

        return image, mask


def get_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    image_size: int = 512,
    val_split: float = 0.15,
    test_split: float = 0.1,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None,
) -> tuple:
    """Create train/val/test dataloaders with proper splits."""
    from torch.utils.data import DataLoader

    full_dataset = SAROilSpillDataset(data_dir, transform=None, image_size=image_size)
    n = len(full_dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    indices = list(range(n))
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    from augmentation import get_train_transforms, get_val_transforms

    train_tf = train_transform or get_train_transforms(image_size)
    val_tf = val_transform or get_val_transforms(image_size)

    train_ds = SAROilSpillDataset(
        data_dir, indices=train_idx, transform=train_tf, image_size=image_size
    )
    val_ds = SAROilSpillDataset(data_dir, indices=val_idx, transform=val_tf, image_size=image_size)
    test_ds = SAROilSpillDataset(
        data_dir, indices=test_idx, transform=val_tf, image_size=image_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"[dataset] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
