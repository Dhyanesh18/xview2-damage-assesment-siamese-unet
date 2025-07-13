import torch
import numpy as np
from PIL import Image
from model_architecture import SiameseUNet
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "siamese_unet_foc_epoch_20.pth"
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 5

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

model = SiameseUNet(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    classes=NUM_CLASSES
)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img) / 255.0  # [0-1]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = (img - mean) / std
    img = img.unsqueeze(0)
    return img

pre_image = load_image("test/images/guatemala-volcano_00000003_pre_disaster.png").to(DEVICE)
post_image = load_image("test/images/guatemala-volcano_00000003_post_disaster.png").to(DEVICE)

with torch.no_grad():
    output = model(pre_image, post_image)
    preds = torch.argmax(output, dim=1).squeeze().cpu().numpy()

mask_img = Image.fromarray(preds.astype(np.uint8))
mask_img.save("predicted_mask.png")
print("Saved mask to: predicted_mask.png")
