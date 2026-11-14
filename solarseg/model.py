"""
Neural network architecture for solar filament segmentation.

WHY THIS EXISTS:
Filament segmentation requires preserving fine topological spatial boundaries (high-resolution
skip connections) while learning contextual quiet-Sun vs. active region patterns (deep receptive
field). A U-Net with a ResNet34 encoder provides proven representational capacity while fitting
comfortably within the memory limits of a free Google Colab T4 (16 GB) or an RTX 3050 (4 GB).
Accepts 2 input channels:
  Channel 0: Flattened H-alpha intensity
  Channel 1: Radial distance in solar radii
Outputs 1 raw logit per pixel for binary cross-entropy and soft Dice loss computation.
"""

from __future__ import annotations

from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from solarseg.config import ModelConfig


class SolarFilamentUNet(nn.Module):
    """
    U-Net wrapper for 2-channel or 3-channel input with configurable encoder and optional auxiliary heads.

    WHY THIS EXISTS:
    Encapsulates SMP U-Net construction, handles multi-channel adaptation cleanly,
    and exposes modular forward passes supporting both single-head inference (default)
    and multi-task auxiliary heads (centerline and boundary) during training.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 2,
        classes: int = 1,
        aux_centerline: bool = False,
        aux_boundary: bool = False,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        self.aux_centerline = aux_centerline
        self.aux_boundary = aux_boundary

        # Construct SMP U-Net
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # Output raw logits
        )

        # Optional auxiliary heads attached to decoder features (channels = 16)
        decoder_channels = 16
        if self.aux_centerline:
            self.centerline_head = nn.Sequential(
                nn.Conv2d(decoder_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
            )
        else:
            self.centerline_head = None

        if self.aux_boundary:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(decoder_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
            )
        else:
            self.boundary_head = None

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Logits tensor of shape (B, 1, H, W) if no auxiliary heads are enabled.
            Dictionary {"mask": mask_logits, ...} if auxiliary heads are enabled.
        """
        if not (self.aux_centerline or self.aux_boundary):
            return self.model(x)

        feats = self.model.encoder(x)
        dec = self.model.decoder(feats)
        mask_logits = self.model.segmentation_head(dec)

        outputs = {"mask": mask_logits}
        if self.aux_centerline and self.centerline_head is not None:
            outputs["centerline"] = self.centerline_head(dec)
        if self.aux_boundary and self.boundary_head is not None:
            outputs["boundary"] = self.boundary_head(dec)
        return outputs

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        """
        Support loading state dicts formatted for the inner SMP Unet directly
        (e.g., keys starting with 'encoder.' or 'decoder.') as well as full wrapper keys.
        """
        if any(k.startswith("encoder.") or k.startswith("decoder.") or k.startswith("segmentation_head.") for k in state_dict.keys()):
            return self.model.load_state_dict(state_dict, strict=strict, assign=assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def build_model(config: Optional[ModelConfig] = None) -> SolarFilamentUNet:
    """
    Model factory instantiated from ModelConfig.

    WHY THIS EXISTS:
    Standardizes model instantiation across training, inference, and testing scripts.
    """
    if config is None:
        config = ModelConfig()

    has_centerline = getattr(config, "aux_centerline", False) or getattr(config, "aux_skeleton", False)
    has_boundary = getattr(config, "aux_boundary", False) or getattr(config, "aux_distance_transform", False)

    return SolarFilamentUNet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        classes=config.classes,
        aux_centerline=has_centerline,
        aux_boundary=has_boundary,
    )
