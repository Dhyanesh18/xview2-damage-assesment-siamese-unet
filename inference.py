"""
Standalone Inference Script for Siamese U-Net Damage Assessment
---------------------------------------------------------------

This script loads a trained Siamese U-Net model from a checkpoint, 
processes a pre-disaster and a post-disaster image, 
performs damage segmentation, and saves the predicted mask to disk.

Requirements:
    - PyTorch
    - segmentation_models_pytorch
    - Pillow
    - NumPy
    - model_architecture.py (custom Siamese U-Net definition)
"""

import torch
import numpy as np
from PIL import Image
from model_architecture import SiameseUNet  # Custom model
import segmentation_models_pytorch as smp   # Required for encoder backbone

# -----------------------
# Configuration
# -----------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "siamese_unet_foc_epoch_20.pth"  # Model weights file
ENCODER_NAME = "resnet34"                     # Backbone encoder
ENCODER_WEIGHTS = "imagenet"                  # Pretrained encoder weights
NUM_CLASSES = 5                               # Number of damage classes

# Normalization parameters (ImageNet statistics)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# -----------------------
# Load the model
# -----------------------

model = SiameseUNet(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    classes=NUM_CLASSES
)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------
# Image Preprocessing
# -----------------------

def load_image(image_path):
    """
    Loads and preprocesses an image for model input.

    Steps:
        - Open image and convert to RGB
        - Scale pixel values to [0, 1]
        - Convert to tensor with shape (3, H, W)
        - Normalize using ImageNet mean and std
        - Add batch dimension

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Normalized image tensor of shape (1, 3, H, W).
    """
    img = Image.open(image_path).convert("RGB")
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = (img - mean) / std
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# -----------------------
# Run Inference
# -----------------------

pre_image = load_image("test/images_pre/guatemala-volcano_00000003_pre_disaster.png").to(DEVICE)
post_image = load_image("test/images_post/guatemala-volcano_00000003_post_disaster.png").to(DEVICE)

with torch.no_grad():
    output = model(pre_image, post_image)  # Model outputs logits
    preds = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Class labels per pixel

# -----------------------
# Save Predicted Mask
# -----------------------

mask_img = Image.fromarray(preds.astype(np.uint8))
mask_img.save("predicted_mask.png")

print("Saved mask to: predicted_mask.png")
