
import os
from collections import defaultdict

# === Config ===
SPLIT_DIR = "data/split"
SPLITS = ["train", "val", "test"]

# === Init trackers ===
plant_disease_map = defaultdict(set)
split_counts = defaultdict(int)
total_classes = set()

# === Helper to parse plant name from class folder ===
def parse_plant_and_disease(class_name):
    if '___' in class_name:
        return class_name.split('___', 1)
    else:
        return class_name, "unknown_disease"

# === Walk through split folders ===
for split in SPLITS:
    split_path = os.path.join(SPLIT_DIR, split)
    for class_name in os.listdir(split_path):
        class_dir = os.path.join(split_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        plant, disease = parse_plant_and_disease(class_name)
        plant_disease_map[plant].add(disease)
        total_classes.add(class_name)

        image_count = len([
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        ])
        split_counts[split] += image_count

# === Display Results ===
print("\n📊 Dataset Summary")
print("-" * 40)
print(f"Total unique classes: {len(total_classes)}")
print(f"Plant types: {len(plant_disease_map)}\n")

for plant, diseases in plant_disease_map.items():
    print(f"{plant}: {len(diseases)} disease types")

print("\n🧪 Split Counts")
print("-" * 40)
for split in SPLITS:
    print(f"{split.capitalize():<6}: {split_counts[split]} images")

total_images = sum(split_counts.values())
print(f"Total: {total_images} images\n")
