# app/utils.py
import numpy as np
from PIL import Image

COLORS = {
    0: (0, 0, 0),        # Background
    1: (255, 0, 0),      # Minor Damage
    2: (0, 255, 0),      # Major Damage
    3: (0, 0, 255),      # Destroyed
    4: (255, 255, 0)     # Other
}

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        color_mask[mask == cls] = color
    return Image.fromarray(color_mask)

def overlay_mask(original, mask, alpha=0.5):
    original = original.convert("RGBA").resize(mask.size)
    mask = mask.convert("RGBA")
    return Image.blend(original, mask, alpha)
