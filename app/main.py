import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse 

from .model import load_model, predict
from .utils import mask_to_color, overlay_mask

app = FastAPI()

model = load_model()

@app.post("/predict")
async def predict_damage(
    pre_disaster: UploadFile = File(...),
    post_disaster: UploadFile = File(...)
):
    pre = Image.open(io.BytesIO(await pre_disaster.read())).convert("RGB")
    post = Image.open(io.BytesIO(await post_disaster.read())).convert("RGB")

    mask = predict(model, pre, post)

    color_mask = mask_to_color(mask)
    overlay = overlay_mask(post, color_mask)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
