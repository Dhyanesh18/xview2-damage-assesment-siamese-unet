import json
import numpy as np
import cv2
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path
from PIL import Image

from glob import glob
from tqdm import tqdm

post_json_paths = glob("test/labels/*_post_disaster.json")
output_dir = Path("test/masks")
output_dir.mkdir(parents=True, exist_ok=True)

# Class mapping for xView2 damage subtypes to integer labels
DAMAGE_CLASSES = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}

def create_damage_mask_from_post_json(post_json_path, image_size=(1024, 1024)):
    """
    Generates a 5-class damage classification mask from a post-disaster GeoJSON.
    0 = background
    1 = no-damage
    2 = minor-damage
    3 = major-damage
    4 = destroyed
    """
    with open(post_json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(image_size, dtype=np.uint8)

    for feature in data["features"]["xy"]:
        props = feature["properties"]
        subtype = props.get("subtype", "no-damage")  # Default to no-damage if missing
        class_id = DAMAGE_CLASSES.get(subtype, 1)

        poly = wkt.loads(feature["wkt"])
        if isinstance(poly, Polygon):
            polygons = [poly]
        elif isinstance(poly, MultiPolygon):
            polygons = list(poly.geoms)
        else:
            continue

        for p in polygons:
            coords = np.array(p.exterior.coords).round().astype(np.int32)
            cv2.fillPoly(mask, [coords], class_id)

    return mask

# Example usage
# post_json_path = Path("test/labels/guatemala-volcano_00000003_post_disaster.json")
# mask = create_damage_mask_from_post_json(post_json_path)
# Image.fromarray(mask).save("guatemala-volcano_00000003_mask.png")

for post_json_path in tqdm(post_json_paths):
    post_json_path = Path(post_json_path)
    mask = create_damage_mask_from_post_json(post_json_path)

    base_name = post_json_path.stem.replace("_post_disaster", "") + "_mask.png"
    mask_path = output_dir / base_name
    Image.fromarray(mask).save(mask_path)