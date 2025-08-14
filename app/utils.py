"""
app/utils.py
------------
Utility functions for visualizing damage segmentation results.

Responsibilities:
    - Convert class index masks to color-coded images.
    - Overlay color-coded masks on top of original images.

These utilities are used in the FastAPI prediction pipeline to produce
human-readable output for the damage assessment model.
"""

import numpy as np
from PIL import Image

# Color mapping for segmentation classes
# Each class index is mapped to an RGB color for visualization.
COLORS = {
    0: (0, 0, 0),        # Background (black)
    1: (255, 0, 0),      # Minor Damage (red)
    2: (0, 255, 0),      # Major Damage (green)
    3: (0, 0, 255),      # Destroyed (blue)
    4: (255, 255, 0)     # Other (yellow)
}


def mask_to_color(mask):
    """
    Convert a class-index mask to a color-coded PIL Image.

    Args:
        mask (np.ndarray): 2D array of shape (H, W) with integer class labels.

    Returns:
        PIL.Image.Image: RGB image where each class is represented by a color.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign a specific color to each class
    for cls, color in COLORS.items():
        color_mask[mask == cls] = color

    return Image.fromarray(color_mask)


def overlay_mask(original, mask, alpha=0.5):
    """
    Overlay a color-coded mask on top of the original image.

    Args:
        original (PIL.Image.Image): The base image (usually post-disaster).
        mask (PIL.Image.Image): Color-coded mask image.
        alpha (float): Transparency level for blending (0 = only original,
                       1 = only mask).

    Returns:
        PIL.Image.Image: Blended RGBA image.
    """
    # Ensure both images are in RGBA mode and match in size
    original = original.convert("RGBA").resize(mask.size)
    mask = mask.convert("RGBA")

    # Blend the original and mask images
    return Image.blend(original, mask, alpha)
