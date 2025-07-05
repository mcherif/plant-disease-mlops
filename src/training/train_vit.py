import os
import torch
import mlflow
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda, Compose, Resize
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score

# === Config ===
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MODEL_NAME = "google/vit-base-patch16-224"
DATA_DIR = "data/split"
EXPERIMENT_NAME = "plant-disease-classifier"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("⚠️ WARNING: CUDA not available. Training may be slow.")

# === MLflow setup ===
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# === Processor and transform ===
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

transform_fn = Compose([
    Resize((224, 224)),
    Lambda(lambda img: img.convert("RGB")),
    Lambda(lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze())
])

# === Load datasets ===
train_dataset = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_fn)
val_dataset = ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform_fn)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Model ===
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=len(train_dataset.classes), ignore_mismatched_sizes=True
).to(device)

# === Training setup ===
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# === Evaluation function ===
def evaluate(model, loader):
    model.eval()
    total_loss, preds, labels = 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(pixel_values=x)
            loss = criterion(out.logits, y)
            total_loss += loss.item()
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return total_loss / len(loader), acc, f1

# === Training ===
with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("model", MODEL_NAME)

    mlflow.log_dict(train_dataset.class_to_idx, "class_mapping.json")

    for epoch in range(EPOCHS):
        model.train()
        total_loss, preds, labels = 0, [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(pixel_values=x)
            loss = criterion(out.logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        val_loss, val_acc, val_f1 = evaluate(model, val_loader)

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_accuracy", acc, step=epoch)
        mlflow.log_metric("train_f1", f1, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)

    model.save_pretrained("models/vit-finetuned")
    processor.save_pretrained("models/vit-finetuned")
    mlflow.log_artifacts("models/vit-finetuned", artifact_path="model")
