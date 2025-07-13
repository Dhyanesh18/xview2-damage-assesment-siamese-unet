import json
from shapely import wkt

with open("train_images_labels_targets/train/targets/hurricane-florence_00000103_post_disaster_target.png") as f:
    data = json.load(f)

count = 0
for feature in data["features"]["xy"]:
    if feature["properties"].get("feature_type") == "building":
        poly = wkt.loads(feature["wkt"])
        count += 1

print(f"Found {count} building polygons.")
