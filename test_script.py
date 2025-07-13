import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv

from model_architecture import SiameseUNet

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "siamese_unet_foc_epoch_20.pth"
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 5

TEST_PRE_DIR = "./test/images_pre/"
TEST_POST_DIR = "./test/images_post/"
GT_MASKS_DIR = "./test/masks/"
OUT_DIR = "./test/predicted_masks/"
CSV_PATH = "./test/metrics_report.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# ImageNet normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Load model
model = SiameseUNet(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    classes=NUM_CLASSES
)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Helper: preprocess
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img) / 255.0  # Scale [0,1]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = (img - mean) / std
    img = img.unsqueeze(0)
    return img

# IoU + Dice + PixelAcc helper
def compute_metrics(preds, mask, num_classes=5):
    class_ious = []
    class_dices = []
    class_accs = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (mask == cls)

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        pred_sum = pred_inds.sum()
        target_sum = target_inds.sum()

        # IoU
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        class_ious.append(iou)

        # Dice
        if pred_sum + target_sum == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / (pred_sum + target_sum)
        class_dices.append(dice)

        # Pixel Acc for this class
        total_pixels = target_inds.size
        correct_pixels = intersection + ((~pred_inds & ~target_inds).sum())
        class_acc = correct_pixels / total_pixels
        class_accs.append(class_acc)

    overall_acc = (preds == mask).sum() / mask.size

    return class_ious, class_dices, class_accs, overall_acc


# Inference loop
pre_files = sorted(os.listdir(TEST_PRE_DIR))
post_files = sorted(os.listdir(TEST_POST_DIR))
gt_files = sorted(os.listdir(GT_MASKS_DIR))

assert len(pre_files) == len(post_files) == len(gt_files), "Mismatch in test file counts!"

iou_accum = []
dice_accum = []
pixel_accum = []
overall_accum = []

for pre_file, post_file, gt_file in tqdm(zip(pre_files, post_files, gt_files), total=len(pre_files), desc="Evaluating"):
    pre_path = os.path.join(TEST_PRE_DIR, pre_file)
    post_path = os.path.join(TEST_POST_DIR, post_file)
    gt_path = os.path.join(GT_MASKS_DIR, gt_file)

    pre_img = load_image(pre_path).to(DEVICE)
    post_img = load_image(post_path).to(DEVICE)

    with torch.no_grad():
        output = model(pre_img, post_img)
        preds = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Save prediction
    out_path = os.path.join(OUT_DIR, pre_file.replace("pre", "mask"))
    mask_img = Image.fromarray(preds)
    mask_img.save(out_path)

    # GT mask
    gt_mask = np.array(Image.open(gt_path))

    # Compute metrics
    ious, dices, class_accs, overall_acc = compute_metrics(preds, gt_mask, num_classes=NUM_CLASSES)
    iou_accum.append(ious)
    dice_accum.append(dices)
    pixel_accum.append(class_accs)
    overall_accum.append(overall_acc)

# Aggregate metrics
iou_accum = np.array(iou_accum)
dice_accum = np.array(dice_accum)
pixel_accum = np.array(pixel_accum)
overall_accum = np.array(overall_accum)

mean_ious = np.mean(iou_accum, axis=0)
mean_dices = np.mean(dice_accum, axis=0)
mean_pixel_accs = np.mean(pixel_accum, axis=0)
mean_overall_acc = np.mean(overall_accum)

print("\nEvaluation done!")
print(f"Mean IoU:  {np.mean(mean_ious):.4f}")
print(f"Mean Dice: {np.mean(mean_dices):.4f}")
print(f"Overall Pixel Acc: {mean_overall_acc:.4f}")
print("\nPer-class metrics:")
for cls_idx in range(NUM_CLASSES):
    print(f"Class {cls_idx}: IoU: {mean_ious[cls_idx]:.4f}, Dice: {mean_dices[cls_idx]:.4f}, Pixel Acc: {mean_pixel_accs[cls_idx]:.4f}")

# Save to CSV
with open(CSV_PATH, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Class", "IoU", "Dice", "Pixel Accuracy"])

    for cls_idx in range(NUM_CLASSES):
        writer.writerow([cls_idx, mean_ious[cls_idx], mean_dices[cls_idx], mean_pixel_accs[cls_idx]])

    writer.writerow([])
    writer.writerow(["Mean IoU", np.mean(mean_ious)])
    writer.writerow(["Mean Dice", np.mean(mean_dices)])
    writer.writerow(["Overall Pixel Accuracy", mean_overall_acc])

print(f"\nCSV report saved to: {CSV_PATH}")
