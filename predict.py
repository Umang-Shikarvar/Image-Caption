import os
import json
from PIL import Image
from tqdm import tqdm

from inference import generate_caption  # your existing function

# ==== CONFIG ====
IMAGE_DIR = "/Users/umangshikarvar/Desktop/Project/coco/val2017"     # e.g., /path/to/coco/val2017
RESULTS_PATH = "predictions.json"

# ==== Collect all image files ====
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

results = []

for filename in tqdm(image_files, desc="Generating captions"):
    image_path = os.path.join(IMAGE_DIR, filename)
    try:
        image = Image.open(image_path).convert("RGB")
        caption = generate_caption(image)

        # Extract image ID from filename (e.g., 000000391895.jpg → 391895)
        image_id = int(os.path.splitext(filename)[0].split("_")[-1])

        results.append({
            "image_id": image_id,
            "caption": caption.strip()
        })

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# ==== Save to results.json ====
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f)

print(f"Saved {len(results)} captions to {RESULTS_PATH}")