import torch
import numpy as np
from PIL import Image
from model_architecture import SiameseUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "siamese_unet_focal_epoch_8.pth"
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 5


mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)


def load_model():
    model = SiameseUNet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess(img):
    img = np.array(img)/255.0
    img = torch.from_numpy(img).permute(2,0,1).float()
    img = (img - mean)/std
    img = img.unsqueeze(0)
    return img

def predict(model, pre_img, post_img):
    pre_tensor = preprocess(pre_img).to(DEVICE)
    post_tensor = preprocess(post_img).to(DEVICE)

    with torch.no_grad():
        output = model(pre_tensor, post_tensor)
        preds = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return preds