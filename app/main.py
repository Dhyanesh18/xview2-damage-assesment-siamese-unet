"""
app/main.py
-----------
FastAPI inference service for a Siamese U-Net building damage assessment model.

This service accepts two images — a pre-disaster image and a post-disaster image —
runs them through a trained Siamese U-Net model, generates a damage segmentation mask,
overlays the mask onto the post-disaster image, and returns the result as a PNG.

Endpoints:
    POST /predict
        Request:
            Form-data with two files:
                - pre_disaster: Pre-disaster RGB image
                - post_disaster: Post-disaster RGB image
        Response:
            PNG image with damage mask overlaid on the post-disaster image.
"""

import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse 

# Import custom model loading and prediction functions
from .model import load_model, predict
# Import utility functions for mask coloring and overlay creation
from .utils import mask_to_color, overlay_mask

# Create FastAPI application instance
app = FastAPI()

# Load the trained model once at startup to avoid reloading on every request
model = load_model()

@app.post("/predict")
async def predict_damage(
    pre_disaster: UploadFile = File(...),
    post_disaster: UploadFile = File(...)
):
    """
    Accepts pre- and post-disaster images, predicts damage segmentation,
    and returns an overlaid PNG image.

    Args:
        pre_disaster (UploadFile): Pre-disaster RGB image file.
        post_disaster (UploadFile): Post-disaster RGB image file.

    Returns:
        StreamingResponse: PNG image with damage segmentation overlay.
    """

    # Load uploaded images into PIL format (ensuring RGB mode)
    pre = Image.open(io.BytesIO(await pre_disaster.read())).convert("RGB")
    post = Image.open(io.BytesIO(await post_disaster.read())).convert("RGB")

    # Run model prediction to get the damage class mask (numpy array)
    mask = predict(model, pre, post)

    # Convert mask to a color-coded image
    color_mask = mask_to_color(mask)

    # Overlay the mask on top of the post-disaster image
    overlay = overlay_mask(post, color_mask)

    # Optionally save to disk for debugging or logging
    overlay.save("predicted_mask.png")

    # Prepare image for HTTP streaming as PNG
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)

    # Return image directly as HTTP response
    return StreamingResponse(buf, media_type="image/png")
