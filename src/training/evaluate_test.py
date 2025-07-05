
import os
import torch
import mlflow
import json
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda, Compose, Resize
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# === Config ===
MODEL_DIR = "models/vit-finetuned"
DATA_DIR = "data/split/test"
BATCH_SIZE = 16
EXPERIMENT_NAME = "plant-disease-classifier"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device)
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)

transform_fn = Compose([
    Resize((224, 224)),
    Lambda(lambda img: img.convert("RGB")),
    Lambda(lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze())
])

with open(os.path.join(MODEL_DIR, "class_mapping.json")) as f:
    class_mapping = json.load(f)

test_dataset = ImageFolder(DATA_DIR, transform=transform_fn)
test_dataset.class_to_idx = class_mapping
test_dataset.classes = [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
index_remap = dict((i, class_mapping[class_name]) for i, class_name in enumerate(test_dataset.classes))
test_dataset.targets = [index_remap[label] for label in test_dataset.targets]

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate(model, loader):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    preds, labels = [], []
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(pixel_values=x)
            loss = loss_fn(out.logits, y)
            total_loss += loss.item()
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    print("✅ True label distribution:", Counter(labels))
    print("🔍 Predicted label distribution:", Counter(preds))
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return total_loss / len(loader), acc, f1

with mlflow.start_run(run_name="test_evaluation_fixed_resize"):
    test_loss, test_acc, test_f1 = evaluate(model, test_loader)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1", test_f1)
    print(f"\n🧪 Test Evaluation → Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
