import os
import shutil

# === Configure paths ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "PlantVillage-Dataset", "raw", "color")
DEST_DIR = os.path.join(BASE_DIR, "data", "processed")

# === Create dest dir if not exists ===
os.makedirs(DEST_DIR, exist_ok=True)

# === Iterate through each folder in the source ===
for class_dir in os.listdir(SOURCE_DIR):
    src_path = os.path.join(SOURCE_DIR, class_dir)

    if os.path.isdir(src_path):
        dest_path = os.path.join(DEST_DIR, class_dir)

        # Skip if already exists
        if os.path.exists(dest_path):
            print(f"Skipping existing class: {class_dir}")
            continue

        # Copy without renaming
        print(f"Copying: {class_dir}")
        shutil.copytree(src_path, dest_path)

print("\n✅ All folders copied into data/processed/ without renaming.")
