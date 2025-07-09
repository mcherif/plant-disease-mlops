from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from prefect import flow, task
import torch
import os
import mlflow
import json
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda, Compose, Resize
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch import nn
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt


# === Config ===
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MODEL_NAME = "google/vit-base-patch16-224"
MODEL_DIR = "models/vit-finetuned"
DATA_DIR = "data/split"
EXPERIMENT_NAME = "plant-disease-classifier"
PATIENCE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@task
def load_data():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    transform_fn = Compose([
        Resize((224, 224)),
        Lambda(lambda img: img.convert("RGB")),
        Lambda(lambda img: processor(images=img, return_tensors="pt")
               ["pixel_values"].squeeze())
    ])

    train_ds = ImageFolder(os.path.join(
        DATA_DIR, "train"), transform=transform_fn)
    val_ds = ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform_fn)
    test_ds = ImageFolder(os.path.join(DATA_DIR, "test"),
                          transform=transform_fn)

    class_mapping = train_ds.class_to_idx
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f)

    return train_ds, val_ds, test_ds, processor, class_mapping


@task
def train_model(train_ds, val_ds, processor, class_mapping):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=len(class_mapping), ignore_mismatched_sizes=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name="prefect_train"):
        mlflow.set_tag("stage", "prefect-pipeline")
        mlflow.set_tag("triggered_by", "full-run")
        try:
            commit = os.popen("git rev-parse HEAD").read().strip()
            mlflow.set_tag("git_commit", commit)
        except:
            pass
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")

        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_dict(class_mapping, "class_mapping.json")

        best_val_acc = 0
        epochs_no_improve = 0

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

            # validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(pixel_values=x)
                    val_preds.extend(torch.argmax(
                        out.logits, dim=1).cpu().numpy())
                    val_labels.extend(y.cpu().numpy())
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="weighted")

            mlflow.log_metric("train_loss", total_loss /
                              len(train_loader), step=epoch)
            mlflow.log_metric("train_accuracy", acc, step=epoch)
            mlflow.log_metric("train_f1", f1, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            print(
                f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                    break

        model.save_pretrained(MODEL_DIR, safe_serialization=True)
        processor.save_pretrained(MODEL_DIR)
        mlflow.log_artifacts(MODEL_DIR, artifact_path="model")
        mlflow.log_artifact("artifacts/best_model.safetensors")

    return MODEL_DIR


@task
def evaluate_model(model_dir, test_ds):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with open(os.path.join(model_dir, "class_mapping.json")) as f:
        class_mapping = json.load(f)

    test_ds.class_to_idx = class_mapping
    test_ds.classes = [k for k, _ in sorted(
        class_mapping.items(), key=lambda x: x[1])]
    test_ds.targets = [class_mapping[test_ds.classes[label]]
                       for label in test_ds.targets]

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = AutoModelForImageClassification.from_pretrained(
        model_dir).to(device)


def evaluate(model, loader, class_names, debug=False):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
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

    # Optional debug output
    if debug:
        report = classification_report(
            labels, preds, target_names=class_names, output_dict=True)
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact("artifacts/classification_report.json")

    # Save confusion matrix
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, values_format='d', cmap='viridis')
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    return total_loss / len(loader), acc, f1


with mlflow.start_run(run_name="prefect_test_eval"):
    test_loss, test_acc, test_f1 = evaluate(
        model, test_loader, class_names, debug=True)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1", test_f1)

    # ✅ Log everything in the artifacts folder
    mlflow.log_artifacts("artifacts", artifact_path="eval_artifacts")

    print(
        f"\n🧪 Test Evaluation → Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")


@flow(name="plant-disease-pipeline")
def plant_disease_pipeline():
    train_ds, val_ds, test_ds, processor, class_mapping = load_data()
    model_dir = train_model(train_ds, val_ds, processor, class_mapping)
    evaluate_model(model_dir, test_ds)


if __name__ == "__main__":
    plant_disease_pipeline()
