import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T

class XView2Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): If None, uses ImageNet normalization.
        """
        self.root_dir = root_dir
        self.pre_dir = os.path.join(root_dir, "images_pre")
        self.post_dir = os.path.join(root_dir, "images_post")
        self.mask_dir = os.path.join(root_dir, "masks")

        # Default transform if not passed
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.sample_ids = sorted([
            f.replace("_pre_disaster.png", "")
            for f in os.listdir(self.pre_dir)
            if f.endswith("_pre_disaster.png")
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        pre_path = os.path.join(self.pre_dir, f"{sample_id}_pre_disaster.png")
        post_path = os.path.join(self.post_dir, f"{sample_id}_post_disaster.png")
        mask_path = os.path.join(self.mask_dir, f"{sample_id}_mask.png")

        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")
        mask = Image.open(mask_path)

        pre_img = self.transform(pre_img)
        post_img = self.transform(post_img)
        mask = torch.from_numpy(np.array(mask)).long()

        return {
            "pre-image": pre_img,
            "post-image": post_img,
            "mask": mask,
        }
