import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from dataset_class import XView2Dataset
from model_architecture import SiameseUNet
import segmentation_models_pytorch as smp

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = "train"
BATCH_SIZE = 1
EPOCHS = 20
LR = 1e-4

# Data
train_dataset = XView2Dataset(ROOT_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = SiameseUNet()
model = model.to(DEVICE)

# Losses - Using Focal Loss for class imbalance
dice_loss_fn = smp.losses.DiceLoss(mode="multiclass")
focal_loss_fn = smp.losses.FocalLoss(mode="multiclass", alpha=0.25, gamma=2.0)

# Alternative: You can also use class weights if you know the exact distribution
# class_weights = torch.tensor([0.1, 2.0, 10.0, 10.0, 10.0]).to(DEVICE)  # Adjust based on your data
# ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Metrics
def pixel_accuracy(output, mask):
    _, preds = torch.max(output, 1)
    correct = (preds == mask).float()
    return correct.sum() / correct.numel()

def mean_iou(output, mask, num_classes=5):
    _, preds = torch.max(output, 1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (mask == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(torch.tensor(1.0, device=output.device))
        else:
            ious.append(intersection / union)
    return torch.mean(torch.stack(ious))

def per_class_iou(output, mask, num_classes=5):
    """Calculate IoU for each class (excluding background for better insight)"""
    _, preds = torch.max(output, 1)
    class_ious = []
    for cls in range(1, num_classes):  # Skip background (class 0)
        pred_inds = (preds == cls)
        target_inds = (mask == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            class_ious.append(0.0)  # No ground truth for this class
        else:
            class_ious.append((intersection / union).item())
    return class_ious

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss, epoch_acc, epoch_iou = 0.0, 0.0, 0.0
    epoch_class_ious = [[] for _ in range(4)]  # For classes 1-4
    
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    
    for batch in loop:
        pre_img = batch["pre-image"].to(DEVICE)
        post_img = batch["post-image"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(pre_img, post_img)
        
        # Combined loss: Dice + Focal Loss
        dice_loss = dice_loss_fn(outputs, mask)
        focal_loss = focal_loss_fn(outputs, mask)
        loss = dice_loss + focal_loss
        
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = pixel_accuracy(outputs, mask)
        iou = mean_iou(outputs, mask)
        class_ious = per_class_iou(outputs, mask)
        
        # Accumulate metrics
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_iou += iou.item()
        
        for i, class_iou in enumerate(class_ious):
            epoch_class_ious[i].append(class_iou)
        
        loop.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc.item():.4f}",
            "IoU": f"{iou.item():.4f}"
        })
    
    # Calculate averages
    avg_loss = epoch_loss / len(train_loader)
    avg_acc = epoch_acc / len(train_loader)
    avg_iou = epoch_iou / len(train_loader)
    
    # Calculate average per-class IoU
    avg_class_ious = []
    for i in range(4):
        if epoch_class_ious[i]:
            avg_class_ious.append(sum(epoch_class_ious[i]) / len(epoch_class_ious[i]))
        else:
            avg_class_ious.append(0.0)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | mIoU: {avg_iou:.4f}")
    print(f"Per-class IoU - Class1: {avg_class_ious[0]:.4f}, Class2: {avg_class_ious[1]:.4f}, Class3: {avg_class_ious[2]:.4f}, Class4: {avg_class_ious[3]:.4f}")
    
    # Save model
    torch.save(model.state_dict(), f"siamese_unet_foc_epoch_{epoch+1}.pth")

print("Training completed!")