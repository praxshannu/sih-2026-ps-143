"""UNet++ encoder with SCSE (Channel + Spatial Squeeze-Excitation) attention.

Uses a timm pretrained backbone and wraps decoder blocks with SCSE modules
that apply both channel-wise and spatial attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SCSE Attention Module
# ---------------------------------------------------------------------------


class ChannelSE(nn.Module):
    """Channel Squeeze-Excitation."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialSE(nn.Module):
    """Spatial Squeeze-Excitation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.conv(x))
        return x * w


class SCSEBlock(nn.Module):
    """Parallel Channel-SE + Spatial-SE attention block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_se = ChannelSE(channels, reduction)
        self.spatial_se = SpatialSE(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_se(x) + self.spatial_se(x)


# ---------------------------------------------------------------------------
# Attention Decoder Block
# ---------------------------------------------------------------------------


class AttentionDecoderBlock(nn.Module):
    """Decoder block with SCSE attention + residual skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCSEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels + skip_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        # Upsample to the skip resolution (×2 per decoder stage) BEFORE fusion.
        # Without this the head emits 1/32-scale logits unusable for masking.
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            diff_h = x.size(2) - skip.size(2)
            diff_w = x.size(3) - skip.size(3)
            skip = F.pad(
                skip, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
            )
            x = torch.cat([x, skip], dim=1)

        residual = self.skip_proj(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.scse(x)
        return x + residual


# ---------------------------------------------------------------------------
# UNet++ with SCSE
# ---------------------------------------------------------------------------


class UNetPlusPlusSCSE(nn.Module):
    """UNet++ with SCSE attention using a timm pretrained encoder.

    Args:
        encoder_name: timm model name for the encoder (e.g. "resnet34").
        encoder_weights: pretrained weights identifier ("imagenet" or None).
        in_channels: number of input channels (default 3 for RGB-like,
                     can be >3 for multi-modal SAR + ERA5).
        classes: number of output segmentation classes.
        decoder_channels: decoder channel progression.
        decoder_attention_reduction: SE reduction ratio.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
        decoder_attention_reduction: int = 16,
    ) -> None:
        super().__init__()

        # timm is imported lazily: it is a heavy optional dependency that is
        # only needed to *build* an encoder. Importing it at module scope made
        # the whole package unimportable (and every test uncollectable) on
        # machines where torch is present but timm is not.
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "timm is required to build a UNet++ encoder. Install it with "
                "`pip install timm`, or use the deterministic Tier-A detector "
                "which has no torch dependency at all."
            ) from exc

        # Build encoder from timm
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=encoder_weights is not None,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3, 4),
        )

        encoder_channels = self.encoder.feature_info.channels()
        # ResNet bottleneck: channels = [64, 128, 256, 512, 1024] (for resnet50+)
        # ResNet basic:     channels = [64, 64, 128, 256, 512]
        decoder_channels_full = list(decoder_channels) + [decoder_channels[-1]]

        self.decoder_blocks = nn.ModuleList()
        prev_ch = encoder_channels[-1]
        for i, out_ch in enumerate(decoder_channels_full):
            skip_ch = encoder_channels[-(i + 2)] if i < len(encoder_channels) - 1 else 0
            self.decoder_blocks.append(AttentionDecoderBlock(prev_ch, skip_ch, out_ch))
            prev_ch = out_ch

        # Final 1x1 convolution for class prediction
        self.segmentation_head = nn.Conv2d(decoder_channels_full[-1], classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        x = features[-1]
        for i, block in enumerate(self.decoder_blocks):
            skip = features[-(i + 2)] if i < len(features) - 1 else None
            x = block(x, skip)

        return self.segmentation_head(x)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        classes: int = 1,
        device: str | torch.device = "cpu",
    ) -> UNetPlusPlusSCSE:
        model = cls(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        return model
