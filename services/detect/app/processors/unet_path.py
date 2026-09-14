"""Tier-B (UNet++) status and, only when it is real, Grad-CAM.

The rule this module enforces is simple and non-negotiable: **a network that
has never converged must not emit numbers that look authoritative.**

There are no trained UNet++ weights on this machine (see ``docs/PLAN.md`` §3:
no labelled oil-spill masks, and 100 epochs on an M2 CPU is not a thing that
happens during a build). When no checkpoint is present the API therefore
reports ``model_status: "untrained"`` and emits **no probabilities at all** —
not zeros, not a soft-max of an ImageNet encoder, not a random-init forward
pass. ``probabilities`` is ``None`` and the response says why.

If a checkpoint *is* present, :func:`gradcam_heatmap` runs a genuine Grad-CAM
(Selvaraju et al., 2017): forward hook to capture the last convolutional
activations, backward pass on the spill-class score, channel weights from the
global-average-pooled gradients, ReLU, bilinear upsample to input resolution.
That path has never executed on this host — there is no checkpoint and no
torch — and the response says so rather than implying otherwise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_STATUS_UNTRAINED = "untrained"
MODEL_STATUS_LOADED = "loaded"
MODEL_STATUS_UNAVAILABLE = "unavailable"

DEFAULT_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pth")
MODEL_NAME = os.getenv("MODEL_NAME", "unetpp_scse_resnet34")


@dataclass(frozen=True)
class UNetStatus:
    """Machine-readable state of the Tier-B network."""

    model_status: str
    model_name: str = MODEL_NAME
    checkpoint: str | None = None
    checkpoint_found: bool = False
    torch_available: bool = False
    reason: str = ""
    probabilities: None = None  # always None here; never a guessed array

    @property
    def trained(self) -> bool:
        return self.model_status == MODEL_STATUS_LOADED

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "status": self.model_status,
            "checkpoint": self.checkpoint,
            "checkpoint_found": self.checkpoint_found,
            "torch_available": self.torch_available,
            "reason": self.reason,
            "probabilities": self.probabilities,
            "trained": self.trained,
        }


def torch_available() -> bool:
    """Whether torch can be imported here (cheap, no import side effects)."""
    import importlib.util

    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:  # noqa: BLE001
        return False


def resolve_unet_status(checkpoint: str | Path | None = None) -> UNetStatus:
    """Decide, without loading anything heavy, what the UNet++ can honestly claim."""
    ckpt = str(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
    has_torch = torch_available()

    if not has_torch:
        return UNetStatus(
            model_status=MODEL_STATUS_UNAVAILABLE,
            checkpoint=ckpt,
            torch_available=False,
            reason=(
                "torch is not importable on this host, so the UNet++ cannot run at all. "
                "The deterministic Tier-A detector is the active detector."
            ),
        )

    path = Path(ckpt)
    if not path.is_file():
        return UNetStatus(
            model_status=MODEL_STATUS_UNTRAINED,
            checkpoint=ckpt,
            checkpoint_found=False,
            torch_available=True,
            reason=(
                f"no trained checkpoint at {ckpt}. A randomly-initialised UNet++ would emit "
                "probabilities that look authoritative and mean nothing, so this service "
                "emits none. Train one (docs/PLAN.md §3) or use the Tier-A detector."
            ),
        )
    return UNetStatus(
        model_status=MODEL_STATUS_LOADED,
        checkpoint=str(path.resolve()),
        checkpoint_found=True,
        torch_available=True,
        reason="trained checkpoint present; UNet++ logits are usable",
    )


class UntrainedModelError(RuntimeError):
    """Raised when inference is attempted with no trained checkpoint."""


def load_unetpp(
    checkpoint: str | Path,
    *,
    encoder_name: str = "resnet34",
    in_channels: int = 6,
    classes: int = 1,
    device: str = "cpu",
) -> Any:
    """Load a *trained* UNet++. Raises if torch or the checkpoint is missing."""
    status = resolve_unet_status(checkpoint)
    if not status.trained:
        raise UntrainedModelError(status.reason)

    import torch  # noqa: PLC0415 - heavy, and only needed on the trained path

    from app.models.unetpp_scse import UNetPlusPlusSCSE  # noqa: PLC0415

    model = UNetPlusPlusSCSE.from_pretrained(
        str(checkpoint),
        encoder_name=encoder_name,
        in_channels=in_channels,
        classes=classes,
        device=device,
    )
    model.to(torch.device(device))
    model.eval()
    return model


def gradcam_heatmap(
    model: Any,
    input_tensor: Any,
    *,
    target_layer: Any | None = None,
    output_size: tuple[int, int] | None = None,
) -> Any:
    """Genuine Grad-CAM for a trained UNet++ spill classifier.

    Args:
        model: a trained ``UNetPlusPlusSCSE`` (or any convnet emitting one
            logit map per pixel) in eval mode.
        input_tensor: ``(1, C, H, W)`` torch tensor with ``requires_grad``
            handling done by torch (the input itself need not require grad;
            the *activations* do).
        target_layer: the convolutional layer to explain. Defaults to the last
            ``nn.Conv2d`` found in the model, which is the standard choice.
        output_size: ``(H, W)`` to upsample the heatmap to; defaults to the
            spatial size of ``input_tensor``.

    Returns:
        ``(H, W)`` float32 numpy array in ``[0, 1]`` — the ReLU'd, normalised
        class-activation map.

    Raises:
        UntrainedModelError: if torch is unavailable (nothing to differentiate).
    """
    if not torch_available():
        raise UntrainedModelError(
            "Grad-CAM needs torch and a trained network; neither is present on this host"
        )

    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    layer = target_layer or _last_conv2d(model)
    if layer is None:
        raise ValueError("no nn.Conv2d layer found to explain")

    captured: dict[str, Any] = {}

    def _forward(_module: Any, _inp: Any, out: Any) -> None:
        captured["activations"] = out
        out.register_hook(lambda grad: captured.__setitem__("gradients", grad))

    handle = layer.register_forward_hook(_forward)
    try:
        model.eval()
        logits = model(input_tensor)
        # One class: use the mean logit over the map as the differentiable score,
        # which is the segmentation analogue of the classification class score.
        score = logits.mean()
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        activations = captured.get("activations")
        gradients = captured.get("gradients")
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks captured nothing — check the target layer")

        weights = gradients.mean(dim=tuple(range(2, gradients.dim())), keepdim=True)
        cam_tensor = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        h, w = output_size or tuple(input_tensor.shape[-2:])
        cam_tensor = F.interpolate(cam_tensor, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam_tensor.squeeze().detach().cpu().numpy().astype("float32")
    finally:
        handle.remove()

    lo, hi = float(cam.min()), float(cam.max())
    cam = (cam - lo) / (hi - lo) if hi - lo > 1e-12 else cam * 0.0
    return cam


def _last_conv2d(model: Any) -> Any | None:
    import torch  # noqa: PLC0415

    found = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            found = module
    return found
