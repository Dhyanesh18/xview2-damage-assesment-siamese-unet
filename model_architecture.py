import torch 
import torch.nn as nn
import segmentation_models_pytorch as smp

class SiameseUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=5):
        super(SiameseUNet, self).__init__()
        base_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights = encoder_weights,
            in_channels=3,
            classes=classes
        )
        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        self.segmentation_head = base_unet.segmentation_head

    def forward(self, pre_image, post_image):
        features_pre = self.encoder(pre_image)
        features_post = self.encoder(post_image)

        fused_features = []
        for f_pre, f_post in zip(features_pre, features_post):
            fused_features.append(torch.abs(f_pre - f_post))

        # Decode fused features
        decoder_output = self.decoder(fused_features)

        out = self.segmentation_head(decoder_output)
        return out
