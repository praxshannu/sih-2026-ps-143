"""SAR-specific augmentations for oil spill detection training.

Provides albumentations-based transforms optimized for SAR imagery:
- Geometric: rotation, flip, elastic deformation
- Radiometric: speckle noise, brightness/contrast, gamma
- SAR-specific: adaptive histogram equalization, dB scaling artifacts

Usage:
    from augmentation import get_train_transforms, get_val_transforms
    train_tf = get_train_transforms(512)
    val_tf = get_val_transforms(512)
"""

from __future__ import annotations

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class SpeckleNoise:
    """Multiplicative speckle noise common in SAR imagery."""

    def __init__(self, p: float = 0.5, noise_range: tuple[float, float] = (0.02, 0.15)) -> None:
        self.p = p
        self.noise_range = noise_range

    def __call__(self, image, mask=None, **kwargs):
        import numpy as np

        if np.random.random() < self.p:
            sigma = np.random.uniform(*self.noise_range)
            noise = np.random.normal(1.0, sigma, image.shape).astype(image.dtype)
            image = image * noise
            image = np.clip(image, 0, 255) if image.max() > 1 else np.clip(image, 0, 1)
        return {"image": image, "mask": mask}

    def get_params(self):
        return {}


class SARBrightnessContrast:
    """Brightness and contrast variation simulating different SAR acquisition conditions."""

    def __init__(self, p: float = 0.5, brightness: float = 0.2, contrast: float = 0.2) -> None:
        self.p = p
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, mask=None, **kwargs):
        import numpy as np

        if np.random.random() < self.p:
            b = np.random.uniform(-self.brightness, self.brightness)
            c = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            image = (
                np.clip(image * c + b, 0, 1)
                if image.max() <= 1
                else np.clip(image * c + b * 255, 0, 255)
            )
        return {"image": image, "mask": mask}


def get_train_transforms(
    image_size: int = 512,
    p: float = 0.5,
) -> A.Compose | None:
    """Training augmentations with SAR-specific transforms."""
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose(
        [
            A.Resize(image_size, image_size, p=1.0),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p * 0.5),
            A.RandomRotate90(p=p * 0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=0,
                p=p,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(0.001, 0.01), p=0.5),
                    A.ISONoise(p=0.5),
                ],
                p=p * 0.6,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.CLAHE(clip_limit=2.0, p=0.3),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                ],
                p=p * 0.7,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=50, sigma=5, p=0.3),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
                ],
                p=p * 0.3,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 512) -> A.Compose | None:
    """Validation augmentations: resize + normalize only."""
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose(
        [
            A.Resize(image_size, image_size, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )


def get_test_transforms(image_size: int = 512) -> A.Compose | None:
    """Test/inference transforms: same as validation."""
    return get_val_transforms(image_size)


def get_sar_db_transform(db_min: float = -25.0, db_max: float = 0.0):
    """Convert SAR amplitude to dB scale and normalize to [0, 1].

    Args:
        db_min: Minimum dB value (clip below this).
        db_max: Maximum dB value (clip above this).

    Returns:
        Callable transform that converts amplitude -> dB -> normalized.
    """
    import numpy as np

    def transform(image, mask=None, **kwargs):
        amplitude = np.clip(image, 1e-10, None)
        db = 10.0 * np.log10(amplitude)
        normalized = (db - db_min) / (db_max - db_min)
        normalized = np.clip(normalized, 0, 1)
        return {"image": normalized, "mask": mask}

    return transform
