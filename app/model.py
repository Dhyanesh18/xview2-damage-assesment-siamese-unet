"""
app/model.py
------------
Model loading, preprocessing, and inference utilities for the Siamese U-Net
damage assessment system.

Responsibilities:
    - Load a pretrained Siamese U-Net model from checkpoint.
    - Preprocess input images for inference.
    - Run prediction on pre/post-disaster image pairs to produce
      segmentation masks.

The model expects two aligned RGB images: pre-disaster and post-disaster.
It outputs a pixel-wise classification mask indicating damage categories.
"""

import torch
import numpy as np
from PIL import Image
from model_architecture import SiameseUNet  # Custom architecture module

# -----------------------
# Configuration constants
# -----------------------

# Device selection: use GPU if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to trained model checkpoint
CHECKPOINT = "siamese_unet_focal_epoch_8.pth"

# Encoder configuration for the Siamese U-Net
ENCODER_NAME = "resnet34"       # Backbone architecture
ENCODER_WEIGHTS = "imagenet"    # Pretrained weights for encoder
NUM_CLASSES = 5                 # Number of segmentation classes

# Normalization parameters (ImageNet statistics)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# -----------------------
# Model Loading
# -----------------------

def load_model():
    """
    Loads the Siamese U-Net model with the specified encoder and weights.

    Returns:
        model (torch.nn.Module): Trained Siamese U-Net model in eval mode.
    """
    model = SiameseUNet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        classes=NUM_CLASSES
    )
    # Load trained weights from checkpoint
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode (disable dropout, etc.)
    return model


# -----------------------
# Preprocessing
# -----------------------

def preprocess(img):
    """
    Converts a PIL image to a normalized tensor suitable for model input.

    Steps:
        - Convert image to NumPy array and scale to [0, 1]
        - Convert to Torch tensor and rearrange dimensions to (C, H, W)
        - Normalize using ImageNet mean and std
        - Add batch dimension

    Args:
        img (PIL.Image.Image): RGB input image.

    Returns:
        torch.Tensor: Normalized image tensor of shape (1, 3, H, W).
    """
    img = np.array(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = (img - mean) / std
    img = img.unsqueeze(0)  # Add batch dimension
    return img


# -----------------------
# Prediction
# -----------------------

def predict(model, pre_img, post_img):
    """
    Runs damage segmentation prediction on a pre/post image pair.

    Args:
        model (torch.nn.Module): Trained Siamese U-Net model.
        pre_img (PIL.Image.Image): Pre-disaster RGB image.
        post_img (PIL.Image.Image): Post-disaster RGB image.

    Returns:
        np.ndarray: Predicted segmentation mask (H, W) with class IDs.
    """
    # Preprocess both images
    pre_tensor = preprocess(pre_img).to(DEVICE)
    post_tensor = preprocess(post_img).to(DEVICE)

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Model expects both images as input
        output = model(pre_tensor, post_tensor)
        # Take argmax across channel dimension to get class labels
        preds = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return preds
