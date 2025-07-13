import os
import json
import numpy as np
from shapely import wkt
from rasterio.features import rasterize
from PIL import Image
from tqdm import tqdm

LABEL_DIR = "train_images_labels_targets/train/labels/"
OUTPUT_MASK_DIR = "train_images_labels_targets/train/masks/"
WIDTH, HEIGHT = 1024, 1024

LABEL_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

def generate_mask_from_json(json_file, output_path):
    with open(json_file, "r") as f:
        data = json.load(f)

    features = data.get("features", {}).get("xy", [])
    shapes = []

    for feature in features:
        props = feature.get("properties", {})
        damage = props.get("subtype")
        wkt_str = feature.get("wkt")

        if damage not in LABEL_MAP or not wkt_str:
            continue

        try:
            polygon = wkt.loads(wkt_str)
            label = LABEL_MAP[damage]
            shapes.append((polygon, label))
        except Exception as e:
            print(f"[!] Error parsing WKT in {json_file}: {e}")

    if not shapes:
        print(f"[!] No valid shapes in: {json_file}")
        return

    mask = rasterize(
        shapes=shapes,
        out_shape=(HEIGHT, WIDTH),
        fill=255,  # 255 = background/no building
        dtype=np.uint8
    )

    Image.fromarray(mask).save(output_path)

def generate_all_masks():
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".json")]
    print(f"[i] Found {len(files)} label files.")

    for name in tqdm(files, desc="Generating damage masks"):
        json_path = os.path.join(LABEL_DIR, name)
        output_path = os.path.join(OUTPUT_MASK_DIR, name.replace('.json', '_mask.png'))
        generate_mask_from_json(json_path, output_path)

if __name__ == "__main__":
    generate_all_masks()
