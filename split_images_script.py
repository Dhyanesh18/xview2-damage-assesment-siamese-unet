import os
import shutil
from pathlib import Path

# Set base paths
train_dir = Path("test")
images_dir = train_dir / "images"
pre_dir = train_dir / "images_pre"
post_dir = train_dir / "images_post"

# Create output folders if they don't exist
pre_dir.mkdir(exist_ok=True)
post_dir.mkdir(exist_ok=True)

# Move files into pre and post folders
for img_path in images_dir.glob("*.png"):
    fname = img_path.name
    if "_pre_disaster" in fname:
        shutil.move(str(img_path), pre_dir / fname)
    elif "_post_disaster" in fname:
        shutil.move(str(img_path), post_dir / fname)
    else:
        print(f"Skipping unrecognized image: {fname}")

print("Images split into 'images_pre' and 'images_post' folders.")
