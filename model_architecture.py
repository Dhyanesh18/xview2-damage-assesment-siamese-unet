"""
Siamese U-Net Model for Building Damage Assessment
--------------------------------------------------

This module defines a Siamese U-Net architecture that takes two aligned images:
    - Pre-disaster image
    - Post-disaster image
and predicts a per-pixel damage classification mask.

Architecture:
    - Shared-weight encoder (based on segmentation_models_pytorch Unet encoder).
    - Feature-level difference fusion using absolute difference.
    - Standard U-Net decoder and segmentation head for pixel-wise prediction.

Key idea:
    By passing both pre- and post-disaster images through the same encoder,
    we ensure that features are comparable. The absolute difference between
    the features captures the changes between the two images, which are then
    decoded into a segmentation map.

Dependencies:
    - segmentation_models_pytorch (SMP)
    - PyTorch
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SiameseUNet(nn.Module):
    """
    Siamese U-Net with shared encoder for change detection.

    Args:
        encoder_name (str): Name of the encoder backbone (e.g., 'resnet34').
        encoder_weights (str or None): Pretrained weights for the encoder
                                       (e.g., 'imagenet' or None).
        classes (int): Number of output segmentation classes.

    Attributes:
        encoder (nn.Module): Shared encoder network.
        decoder (nn.Module): U-Net decoder network.
        segmentation_head (nn.Module): Final prediction head producing class logits.
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=5):
        super(SiameseUNet, self).__init__()

        # Create a standard U-Net from SMP to extract components
        base_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes
        )

        # Use the same encoder for both images (shared weights)
        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        self.segmentation_head = base_unet.segmentation_head

    def forward(self, pre_image, post_image):
        """
        Forward pass of the Siamese U-Net.

        Args:
            pre_image (torch.Tensor): Tensor of shape (B, 3, H, W) for pre-disaster image.
            post_image (torch.Tensor): Tensor of shape (B, 3, H, W) for post-disaster image.

        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W).
        """
        # Extract multi-scale features from both images
        features_pre = self.encoder(pre_image)
        features_post = self.encoder(post_image)

        # Fuse features by taking absolute difference at each scale
        fused_features = [
            torch.abs(f_pre - f_post)
            for f_pre, f_post in zip(features_pre, features_post)
        ]

        # Decode fused features into segmentation map
        decoder_output = self.decoder(fused_features)

        # Apply segmentation head to get per-class logits
        out = self.segmentation_head(decoder_output)
        return out
