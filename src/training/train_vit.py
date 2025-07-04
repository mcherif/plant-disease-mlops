import os
import torch
import mlflow
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score
from torchvision.transforms import ToTensor
from torchvision.transforms import Lambda
from torchvision.transforms import Compose
import numpy as np

# === Config ===
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MODEL_NAME = "google/vit-base-patch16-224"
DATA_DIR = "data/processed"
EXPERIMENT_NAME = "plant-disease-classifier"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("⚠️ WARNING: CUDA not available. Training may be very slow.")

# === Set up MLflow ===
mlflow.set_tracking_uri("file:../../mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

# === Load processor ===
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# === Transform function ===


def transform(example):
    return processor(images=example, return_tensors="pt")["pixel_values"].squeeze()


transform_fn = Compose([
    Lambda(lambda img: img.convert("RGB")),
    Lambda(transform),
])

# === Load dataset ===
train_dataset = ImageFolder(DATA_DIR, transform=transform_fn)
class_names = train_dataset.classes
NUM_CLASSES = len(class_names)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Load model ===
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
model.to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
with mlflow.start_run():
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    # Log class names
    class_mapping = train_dataset.class_to_idx
    mlflow.log_dict(class_mapping, "class_mapping.json")
    mlflow.log_param("class mapping", class_mapping)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        preds, labels = [], []

        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(pixel_values=inputs)
            loss = criterion(outputs.logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")

        mlflow.log_metric("train_loss", epoch_loss /
                          len(train_loader), step=epoch)
        mlflow.log_metric("train_accuracy", acc, step=epoch)
        mlflow.log_metric("train_f1", f1, step=epoch)
        print(
            f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}  Acc: {acc:.4f}  F1: {f1:.4f}")

    # Save final model
    output_dir = "models/vit-finetuned"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    mlflow.log_artifacts(output_dir, artifact_path="model")
