
import os
import shutil
import random
from sklearn.model_selection import train_test_split

# === Config ===
INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/split"
SPLITS = ["train", "val", "test"]
RATIOS = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test
SEED = 42

random.seed(SEED)

# === Create output folders ===
for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# === Process each class folder ===
for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    trainval_imgs, test_imgs = train_test_split(images, test_size=RATIOS[2], random_state=SEED)
    train_imgs, val_imgs = train_test_split(trainval_imgs, test_size=RATIOS[1]/(RATIOS[0]+RATIOS[1]), random_state=SEED)

    for split, split_imgs in zip(SPLITS, [train_imgs, val_imgs, test_imgs]):
        split_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for img in split_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy2(src, dst)

print("✅ Done: Split into train/val/test with stratified class folders.")
