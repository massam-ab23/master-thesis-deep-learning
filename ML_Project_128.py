"""
NOTE:
This file is automatically converted from a Jupyter Notebook (.ipynb).

Some notebook-specific commands (e.g., !pip install, Google Colab code)
are still present and may not run as a standard Python script.

For full functionality and original workflow, please refer to the
Jupyter Notebook version included in this repository.

This script is provided for easier code readability on GitHub.
"""


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade timm')
import importlib
importlib.reload(timm)


# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#STEP 2: Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128"
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


import os

# Path to the main directory (change this to your local path)
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128"

# List of class folders
class_folders = ["Class1_CarTouch", "Class2_UnTouch", "Class3_AllTouch"]

# Initialize total counter
total_images = 0

# Dictionary to store individual class image counts
image_counts = {}

# Count images in each class folder
for class_name in class_folders:
    class_path = os.path.join(base_dir, class_name)
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(images)
        image_counts[class_name] = count
        total_images += count
    else:
        image_counts[class_name] = 0
        print(f"Warning: Folder '{class_name}' not found.")

# Print results
print("Image counts per class:")
for class_name, count in image_counts.items():
    print(f"{class_name}: {count} images")

print(f"\nTotal images: {total_images}")


# In[ ]:


import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

# === DATA CONFIGURATION ===
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128/split_data_128"
batch_size = 32

# Training transforms with data augmentation
train_transforms_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # Normalize to mean=0.5, std=0.5 (grayscale-style)
])

# Validation/test transforms — no augmentation
val_test_transforms_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Create Datasets using ImageFolder (folder name = class label) ===
train_dataset_128 = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms_128)
val_dataset_128 = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transforms_128)
test_dataset_128 = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transforms_128)

#the size of train_dataset_128, val_dataset_128, test_dataset_128

# === Wrap datasets into DataLoaders (for EfficientNet & LeViT) ===
train_loader_128 = DataLoader(train_dataset_128, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader_128 = DataLoader(val_dataset_128, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader_128 = DataLoader(test_dataset_128, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Number of training images: {len(train_dataset_128)}")
print(f"Number of validation images: {len(val_dataset_128)}")
print(f"Number of test images: {len(test_dataset_128)}")


# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#STEP 2: Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128"
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


#STEP 4: Split the Dataset
# Split the images into train (70%), val (15%), test (15%) in separate folders.
import os
import shutil
import random

def split_dataset(base_dir, output_dir, class_names, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    for class_name in class_names:
        class_path = os.path.join(base_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        train_cutoff = int(train_ratio * len(images))
        val_cutoff = int((train_ratio + val_ratio) * len(images))

        train_files = images[:train_cutoff]
        val_files = images[train_cutoff:val_cutoff]
        test_files = images[val_cutoff:]

        for fname in train_files:
            shutil.copy(os.path.join(class_path, fname), os.path.join(output_dir, 'train', class_name, fname))
        for fname in val_files:
            shutil.copy(os.path.join(class_path, fname), os.path.join(output_dir, 'val', class_name, fname))
        for fname in test_files:
            shutil.copy(os.path.join(class_path, fname), os.path.join(output_dir, 'test', class_name, fname))

# Run it
# New (save inside Google Drive):
output_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128/split_data"
split_dataset(base_dir, output_dir, class_names)


# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128/split_data"
batch_size = 32

shared_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

shared_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=shared_train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=shared_eval_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=shared_eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(" Data Summary:")
print(f"Number of training images:   {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images:       {len(test_dataset)}")

# ============================
# STEP 3: Model Setup
# ============================
num_classes = 3

# EfficientNet-B0
model = timm.create_model('efficientnet_b0', pretrained=True)
model.classifier = nn.Linear(model.get_classifier().in_features, num_classes)
model = model.to(device)

# Vision Transformer
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = nn.Linear(vit_model.head.in_features, num_classes)
vit_model = vit_model.to(device)

# Swin Transformer
swin_model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
swin_model = swin_model.to(device)

# ============================
# STEP 4: Training Function
# ============================
import time

def train_one_model(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    print(f"\nTraining {model_name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            # Save the best model
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_model.pth")

    model.load_state_dict(best_model_state)

    # =========================
    # Inference & Timing
    # =========================
    y_true, y_pred = [], []
    total_inference_time = 0.0
    total_pred_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_infer = time.perf_counter()
            outputs = model(images)
            end_infer = time.perf_counter()

            start_pred = time.perf_counter()
            _, predicted = torch.max(outputs, 1)
            end_pred = time.perf_counter()

            batch_size_now = labels.size(0)
            total_samples += batch_size_now
            total_inference_time += (end_infer - start_infer)
            total_pred_time += (end_pred - start_pred)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{model_name}] Average Inference Time per Batch: {total_inference_time:.4f} sec")
    print(f"[{model_name}] Average Prediction Time per Batch: {total_pred_time:.4f} sec")

    return {
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "test_accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "avg_infer_time_per_batch": total_inference_time / len(test_loader),
        "avg_pred_time_per_batch": total_pred_time / len(test_loader)
    }


# ============================
# STEP 5: Train All Models
# ============================
results = {}

results["EfficientNet-B0"] = train_one_model(model, "EfficientNet-B0", train_loader, val_loader, test_loader)
results["ViT"] = train_one_model(vit_model, "ViT", train_loader, val_loader, test_loader)
results["Swin Transformer"] = train_one_model(swin_model, "Swin Transformer", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Plot Accuracy and Loss
# ============================
plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_acc'], label=f"{model_name} Train")
    plt.plot(data['val_acc'], label=f"{model_name} Val")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_loss'], label=f"{model_name} Train")
    plt.plot(data['val_loss'], label=f"{model_name} Val")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# STEP 7: Confusion Matrices & Summary
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

summary_df = pd.DataFrame([
    {
        "Model": model_name,
        "Test Accuracy": round(data["test_accuracy"], 4),
        "Precision": round(data["precision"], 4),
        "Recall": round(data["recall"], 4),
        "F1 Score": round(data["f1_score"], 4),
        "Avg Inference Time (s)": round(data.get("avg_infer_time_per_batch", 0.0), 6),
        "Avg Prediction Time (s)": round(data.get("avg_pred_time_per_batch", 0.0), 6)
    }
    for model_name, data in results.items()
])

from IPython.display import display
display(summary_df)


for model_name, model_data in results.items():
    cm = model_data["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_confusion_matrix.png")
    plt.show()


# In[ ]:


"The last one"


# 128x128

# without splitting the data, using randomely

# In[ ]:


get_ipython().system('pip uninstall torch torchvision torchaudio timm -y')
get_ipython().system('pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118')
get_ipython().system('pip install timm==0.9.12')
get_ipython().system('pip install numpy==1.26.4 --force-reinstall')


# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#STEP 2: Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128"
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Random Split Dataset Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128"
batch_size = 32
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Transforms
shared_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

shared_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load all images
from PIL import Image
from torchvision.datasets import ImageFolder

class RGBImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")  # Ensure it's RGB
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# Use this safer dataset loader instead
full_dataset = RGBImageFolder(data_dir, transform=shared_train_transform)

# Split sizes
total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # Ensures total matches

# Optional: for reproducibility
g = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=g)

# Apply correct transforms per subset
train_dataset.dataset.transform = shared_train_transform
val_dataset.dataset.transform = shared_eval_transform
test_dataset.dataset.transform = shared_eval_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(" Data Summary:")
print(f"Number of training images:   {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images:       {len(test_dataset)}")

# ============================
# STEP 3: Model Setup
# ============================
num_classes = 3

# EfficientNet-B0 (SAFE way)
model = timm.create_model('efficientnet_b0', pretrained=True)
n_features = model.get_classifier().in_features
model.classifier = nn.Linear(n_features, num_classes)
model = model.to(device)

# Vision Transformer
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_features = vit_model.head.in_features
vit_model.head = nn.Linear(vit_features, num_classes)
vit_model = vit_model.to(device)

# Swin Transformer
# Swin Transformer (Final Correct Setup)
swin_model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=num_classes,  # Enable head here
    global_pool='avg'         # Force global average pooling
)
swin_model = swin_model.to(device)

dummy = torch.randn(2, 3, 224, 224).to(device)
print("Swin output shape:", swin_model(dummy).shape)


# ============================
# STEP 4: Training Function
# ============================
def train_one_model(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    print(f"\nTraining {model_name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_model.pth")

    model.load_state_dict(best_model_state)

    # =========================
    # Inference & Timing
    # =========================
    y_true, y_pred = [], []
    total_inference_time = 0.0
    total_pred_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_infer = time.perf_counter()
            outputs = model(images)
            end_infer = time.perf_counter()

            start_pred = time.perf_counter()
            _, predicted = torch.max(outputs, 1)
            end_pred = time.perf_counter()

            batch_size_now = labels.size(0)
            total_samples += batch_size_now
            total_inference_time += (end_infer - start_infer)
            total_pred_time += (end_pred - start_pred)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{model_name}] Average Inference Time per Batch: {total_inference_time:.4f} sec")
    print(f"[{model_name}] Average Prediction Time per Batch: {total_pred_time:.4f} sec")

    return {
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "test_accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "avg_infer_time_per_batch": total_inference_time / len(test_loader),
        "avg_pred_time_per_batch": total_pred_time / len(test_loader)
    }

# ============================
# STEP 5: Train All Models
# ============================
results = {}
results["EfficientNet-B0"] = train_one_model(model, "EfficientNet-B0", train_loader, val_loader, test_loader)
results["ViT"] = train_one_model(vit_model, "ViT", train_loader, val_loader, test_loader)
results["Swin Transformer"] = train_one_model(swin_model, "Swin Transformer", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Plot Accuracy and Loss
# ============================
plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_acc'], label=f"{model_name} Train")
    plt.plot(data['val_acc'], label=f"{model_name} Val")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_loss'], label=f"{model_name} Train")
    plt.plot(data['val_loss'], label=f"{model_name} Val")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# STEP 7: Confusion Matrices & Summary
# ============================
summary_df = pd.DataFrame([
    {
        "Model": model_name,
        "Test Accuracy": round(data["test_accuracy"], 4),
        "Precision": round(data["precision"], 4),
        "Recall": round(data["recall"], 4),
        "F1 Score": round(data["f1_score"], 4),
        "Avg Inference Time (s)": round(data.get("avg_infer_time_per_batch", 0.0), 6),
        "Avg Prediction Time (s)": round(data.get("avg_pred_time_per_batch", 0.0), 6)
    }
    for model_name, data in results.items()
])

from IPython.display import display
display(summary_df)

for model_name, model_data in results.items():
    cm = model_data["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_confusion_matrix.png")
    plt.show()


# In[ ]:





# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


#STEP 4: Split the Dataset
# Split the images into train (70%), val (15%), test (15%) in separate folders.
import os
import shutil
import random

def split_dataset(base_dir, output_dir, class_names, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    for class_name in class_names:
        class_path = os.path.join(base_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        train_cutoff = int(train_ratio * len(images))
        val_cutoff = int((train_ratio + val_ratio) * len(images))

        train_files = images[:train_cutoff]
        val_files = images[train_cutoff:val_cutoff]
        test_files = images[val_cutoff:]

        for fname in train_files:
            shutil.copy(os.path.join(class_path, fname), os.path.join(output_dir, 'train', class_name, fname))
        for fname in val_files:
            shutil.copy(os.path.join(class_path, fname), os.path.join(output_dir, 'val', class_name, fname))
        for fname in test_files:
            shutil.copy(os.path.join(class_path, fname), os.path.join(output_dir, 'test', class_name, fname))

# Run it
# New (save inside Google Drive):
output_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256/split_data"
split_dataset(base_dir, output_dir, class_names)


# In[ ]:


data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256/split_data"
print(data_dir)


# In[ ]:


# ============================
# STEP 1: Imports & Setup
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import timm  # PyTorch Image Models

# === Check device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================
# STEP 2: Dataset (Resize to 256x256)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256/split_data"
print(data_dir)

batch_size = 32

# === Unified Transform (224x224 for all models including ViT) ===

# === Unified Transform (224x224 and RGB) ===

shared_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

shared_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=shared_train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=shared_eval_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=shared_eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(" Data Summary:")
print(f"Number of training images:   {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images:       {len(test_dataset)}")

# ============================
# STEP 3: Model Setup
# ============================
num_classes = 3

# EfficientNet-B0
model = timm.create_model('efficientnet_b0', pretrained=True)
model.classifier = nn.Linear(model.get_classifier().in_features, num_classes)
model = model.to(device)

# Vision Transformer
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = nn.Linear(vit_model.head.in_features, num_classes)
vit_model = vit_model.to(device)

# Swin Transformer
swin_model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
swin_model = swin_model.to(device)

# ============================
# STEP 4: Training Function
# ============================
import time

def train_one_model(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    print(f"\nTraining {model_name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            # Save the best model
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_model.pth")

    model.load_state_dict(best_model_state)

    # =========================
    # Inference & Timing
    # =========================
    y_true, y_pred = [], []
    total_inference_time = 0.0
    total_pred_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_infer = time.perf_counter()
            outputs = model(images)
            end_infer = time.perf_counter()

            start_pred = time.perf_counter()
            _, predicted = torch.max(outputs, 1)
            end_pred = time.perf_counter()

            batch_size_now = labels.size(0)
            total_samples += batch_size_now
            total_inference_time += (end_infer - start_infer)
            total_pred_time += (end_pred - start_pred)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{model_name}] Average Inference Time per Batch: {total_inference_time:.4f} sec")
    print(f"[{model_name}] Average Prediction Time per Batch: {total_pred_time:.4f} sec")

    return {
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "test_accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "avg_infer_time_per_batch": total_inference_time / len(test_loader),
        "avg_pred_time_per_batch": total_pred_time / len(test_loader)
    }


# ============================
# STEP 5: Train All Models
# ============================
results = {}

results["EfficientNet-B0"] = train_one_model(model, "EfficientNet-B0", train_loader, val_loader, test_loader)
results["ViT"] = train_one_model(vit_model, "ViT", train_loader, val_loader, test_loader)
results["Swin Transformer"] = train_one_model(swin_model, "Swin Transformer", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Plot Accuracy and Loss
# ============================
plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_acc'], label=f"{model_name} Train")
    plt.plot(data['val_acc'], label=f"{model_name} Val")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_loss'], label=f"{model_name} Train")
    plt.plot(data['val_loss'], label=f"{model_name} Val")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# STEP 7: Confusion Matrices & Summary
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

summary_df = pd.DataFrame([
    {
        "Model": model_name,
        "Test Accuracy": round(data["test_accuracy"], 4),
        "Precision": round(data["precision"], 4),
        "Recall": round(data["recall"], 4),
        "F1 Score": round(data["f1_score"], 4),
        "Avg Inference Time (s)": round(data.get("avg_infer_time_per_batch", 0.0), 6),
        "Avg Prediction Time (s)": round(data.get("avg_pred_time_per_batch", 0.0), 6)
    }
    for model_name, data in results.items()
])

from IPython.display import display
display(summary_df)


for model_name, model_data in results.items():
    cm = model_data["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_confusion_matrix.png")
    plt.show()


# In[ ]:


get_ipython().system('pip uninstall torch torchvision torchaudio timm -y')
get_ipython().system('pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118')
get_ipython().system('pip install timm==0.9.12')
get_ipython().system('pip install numpy==1.26.4 --force-reinstall')


# without splitting the data, using randomely
# 

# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Random Split Dataset Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256"
batch_size = 32
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Transforms
shared_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

shared_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load all images
from PIL import Image
from torchvision.datasets import ImageFolder

class RGBImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")  # Ensure it's RGB
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# Use this safer dataset loader instead
full_dataset = RGBImageFolder(data_dir, transform=shared_train_transform)
label_set = set([label for _, label in full_dataset])
print("Labels in dataset:", label_set)

# Split sizes
total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # Ensures total matches

# Optional: for reproducibility
g = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=g)

# Apply correct transforms per subset
train_dataset.dataset.transform = shared_train_transform
val_dataset.dataset.transform = shared_eval_transform
test_dataset.dataset.transform = shared_eval_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(" Data Summary:")
print(f"Number of training images:   {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images:       {len(test_dataset)}")

# ============================
# STEP 3: Model Setup
# ============================
num_classes = 3

# EfficientNet-B0 (SAFE way)
model = timm.create_model('efficientnet_b0', pretrained=True)
n_features = model.get_classifier().in_features
model.classifier = nn.Linear(n_features, num_classes)
model = model.to(device)

# Vision Transformer
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_features = vit_model.head.in_features
vit_model.head = nn.Linear(vit_features, num_classes)
vit_model = vit_model.to(device)

# Swin Transformer
# Swin Transformer (Final Correct Setup)
swin_model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=num_classes,  # Enable head here
    global_pool='avg'         # Force global average pooling
)
swin_model = swin_model.to(device)

dummy = torch.randn(2, 3, 224, 224).to(device)
print("Swin output shape:", swin_model(dummy).shape)


# ============================
# STEP 4: Training Function
# ============================
def train_one_model(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    print(f"\nTraining {model_name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_model.pth")

    model.load_state_dict(best_model_state)

    # =========================
    # Inference & Timing
    # =========================
    y_true, y_pred = [], []
    total_inference_time = 0.0
    total_pred_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_infer = time.perf_counter()
            outputs = model(images)
            end_infer = time.perf_counter()

            start_pred = time.perf_counter()
            _, predicted = torch.max(outputs, 1)
            end_pred = time.perf_counter()

            batch_size_now = labels.size(0)
            total_samples += batch_size_now
            total_inference_time += (end_infer - start_infer)
            total_pred_time += (end_pred - start_pred)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{model_name}] Average Inference Time per Batch: {total_inference_time:.4f} sec")
    print(f"[{model_name}] Average Prediction Time per Batch: {total_pred_time:.4f} sec")

    return {
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "test_accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "avg_infer_time_per_batch": total_inference_time / len(test_loader),
        "avg_pred_time_per_batch": total_pred_time / len(test_loader)
    }

# ============================
# STEP 5: Train All Models
# ============================
results = {}
results["EfficientNet-B0"] = train_one_model(model, "EfficientNet-B0", train_loader, val_loader, test_loader)
results["ViT"] = train_one_model(vit_model, "ViT", train_loader, val_loader, test_loader)
results["Swin Transformer"] = train_one_model(swin_model, "Swin Transformer", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Plot Accuracy and Loss
# ============================
plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_acc'], label=f"{model_name} Train")
    plt.plot(data['val_acc'], label=f"{model_name} Val")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_loss'], label=f"{model_name} Train")
    plt.plot(data['val_loss'], label=f"{model_name} Val")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# STEP 7: Confusion Matrices & Summary
# ============================
summary_df = pd.DataFrame([
    {
        "Model": model_name,
        "Test Accuracy": round(data["test_accuracy"], 4),
        "Precision": round(data["precision"], 4),
        "Recall": round(data["recall"], 4),
        "F1 Score": round(data["f1_score"], 4),
        "Avg Inference Time (s)": round(data.get("avg_infer_time_per_batch", 0.0), 6),
        "Avg Prediction Time (s)": round(data.get("avg_pred_time_per_batch", 0.0), 6)
    }
    for model_name, data in results.items()
])

from IPython.display import display
display(summary_df)

for model_name, model_data in results.items():
    cm = model_data["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_confusion_matrix.png")
    plt.show()


# 512x512

# 

# In[ ]:





# In[ ]:


swin_model.eval()
images, labels = next(iter(train_loader))
images = images.to(device)

with torch.no_grad():
    outputs = swin_model(images)

print("Swin Transformer output shape:", outputs.shape)


# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


#STEP 4: Split the Dataset
# Split the images into train (70%), val (15%), test (15%) in separate folders.
import os
import shutil
import random

def split_dataset(base_dir, output_dir, class_names, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    for class_name in class_names:
        class_path = os.path.join(base_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        train_cutoff = int(train_ratio * len(images))
        val_cutoff = int((train_ratio + val_ratio) * len(images))

        train_files = images[:train_cutoff]
        val_files = images[train_cutoff:val_cutoff]
        test_files = images[val_cutoff:]

        for fname in train_files:
            shutil.copy(os.path.join(class_path, fname), os.path.join(output_dir, 'train', class_name, fname))
        for fname in val_files:
            shutil.copy(os.path.join(class_path, fname), os.path.join(output_dir, 'val', class_name, fname))
        for fname in test_files:
            shutil.copy(os.path.join(class_path, fname), os.path.join(output_dir, 'test', class_name, fname))

# Run it
# New (save inside Google Drive):
output_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512/split_data"
split_dataset(base_dir, output_dir, class_names)


# In[ ]:


# ============================
# STEP 5: Define Transforms & Dataloaders (512x512 Version)
# ============================
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

# === DATA CONFIGURATION ===
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512/split_data"
batch_size = 32

# -------------------------------------
# Transforms for EfficientNet & LeViT (512x512 input)
# -------------------------------------
# ============================
# STEP 2: Data Setup
# ============================
shared_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

shared_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=shared_train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=shared_eval_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=shared_eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(" Data Summary:")
print(f"Number of training images:   {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images:       {len(test_dataset)}")

# ============================
# STEP 3: Model Setup
# ============================
num_classes = 3

# EfficientNet-B0
model = timm.create_model('efficientnet_b0', pretrained=True)
model.classifier = nn.Linear(model.get_classifier().in_features, num_classes)
model = model.to(device)

# Vision Transformer
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = nn.Linear(vit_model.head.in_features, num_classes)
vit_model = vit_model.to(device)

# Swin Transformer
swin_model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
swin_model = swin_model.to(device)

# ============================
# STEP 4: Training Function
# ============================
import time

def train_one_model(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    print(f"\nTraining {model_name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            # Save the best model
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_model.pth")

    model.load_state_dict(best_model_state)

    # =========================
    # Inference & Timing
    # =========================
    y_true, y_pred = [], []
    total_inference_time = 0.0
    total_pred_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_infer = time.perf_counter()
            outputs = model(images)
            end_infer = time.perf_counter()

            start_pred = time.perf_counter()
            _, predicted = torch.max(outputs, 1)
            end_pred = time.perf_counter()

            batch_size_now = labels.size(0)
            total_samples += batch_size_now
            total_inference_time += (end_infer - start_infer)
            total_pred_time += (end_pred - start_pred)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{model_name}] Average Inference Time per Batch: {total_inference_time:.4f} sec")
    print(f"[{model_name}] Average Prediction Time per Batch: {total_pred_time:.4f} sec")

    return {
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "test_accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "avg_infer_time_per_batch": total_inference_time / len(test_loader),
        "avg_pred_time_per_batch": total_pred_time / len(test_loader)
    }


# ============================
# STEP 5: Train All Models
# ============================
results = {}

results["EfficientNet-B0"] = train_one_model(model, "EfficientNet-B0", train_loader, val_loader, test_loader)
results["ViT"] = train_one_model(vit_model, "ViT", train_loader, val_loader, test_loader)
results["Swin Transformer"] = train_one_model(swin_model, "Swin Transformer", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Plot Accuracy and Loss
# ============================
plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_acc'], label=f"{model_name} Train")
    plt.plot(data['val_acc'], label=f"{model_name} Val")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_loss'], label=f"{model_name} Train")
    plt.plot(data['val_loss'], label=f"{model_name} Val")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# STEP 7: Confusion Matrices & Summary
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

summary_df = pd.DataFrame([
    {
        "Model": model_name,
        "Test Accuracy": round(data["test_accuracy"], 4),
        "Precision": round(data["precision"], 4),
        "Recall": round(data["recall"], 4),
        "F1 Score": round(data["f1_score"], 4),
        "Avg Inference Time (s)": round(data.get("avg_infer_time_per_batch", 0.0), 6),
        "Avg Prediction Time (s)": round(data.get("avg_pred_time_per_batch", 0.0), 6)
    }
    for model_name, data in results.items()
])

from IPython.display import display
display(summary_df)


for model_name, model_data in results.items():
    cm = model_data["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_confusion_matrix.png")
    plt.show()




# In[ ]:


get_ipython().system('pip uninstall torch torchvision torchaudio timm -y')
get_ipython().system('pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118')
get_ipython().system('pip install timm==0.9.12')
get_ipython().system('pip install numpy==1.26.4 --force-reinstall')


# without splitting the data, using randomely

# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Random Split Dataset Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512"
batch_size = 32
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Transforms
shared_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

shared_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load all images
from PIL import Image
from torchvision.datasets import ImageFolder

class RGBImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")  # Ensure it's RGB
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# Use this safer dataset loader instead
full_dataset = RGBImageFolder(data_dir, transform=shared_train_transform)
label_set = set([label for _, label in full_dataset])
print("Labels in dataset:", label_set)

# Split sizes
total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # Ensures total matches

# Optional: for reproducibility
g = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=g)

# Apply correct transforms per subset
train_dataset.dataset.transform = shared_train_transform
val_dataset.dataset.transform = shared_eval_transform
test_dataset.dataset.transform = shared_eval_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(" Data Summary:")
print(f"Number of training images:   {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images:       {len(test_dataset)}")

# ============================
# STEP 3: Model Setup
# ============================
num_classes = 3

# EfficientNet-B0 (SAFE way)
model = timm.create_model('efficientnet_b0', pretrained=True)
n_features = model.get_classifier().in_features
model.classifier = nn.Linear(n_features, num_classes)
model = model.to(device)

# Vision Transformer
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_features = vit_model.head.in_features
vit_model.head = nn.Linear(vit_features, num_classes)
vit_model = vit_model.to(device)

# Swin Transformer
# Swin Transformer (Final Correct Setup)
swin_model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=num_classes,  # Enable head here
    global_pool='avg'         # Force global average pooling
)
swin_model = swin_model.to(device)

dummy = torch.randn(2, 3, 224, 224).to(device)
print("Swin output shape:", swin_model(dummy).shape)


# ============================
# STEP 4: Training Function
# ============================
def train_one_model(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    print(f"\nTraining {model_name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_model.pth")

    model.load_state_dict(best_model_state)

    # =========================
    # Inference & Timing
    # =========================
    y_true, y_pred = [], []
    total_inference_time = 0.0
    total_pred_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_infer = time.perf_counter()
            outputs = model(images)
            end_infer = time.perf_counter()

            start_pred = time.perf_counter()
            _, predicted = torch.max(outputs, 1)
            end_pred = time.perf_counter()

            batch_size_now = labels.size(0)
            total_samples += batch_size_now
            total_inference_time += (end_infer - start_infer)
            total_pred_time += (end_pred - start_pred)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{model_name}] Average Inference Time per Batch: {total_inference_time:.4f} sec")
    print(f"[{model_name}] Average Prediction Time per Batch: {total_pred_time:.4f} sec")

    return {
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "test_accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "avg_infer_time_per_batch": total_inference_time / len(test_loader),
        "avg_pred_time_per_batch": total_pred_time / len(test_loader)
    }

# ============================
# STEP 5: Train All Models
# ============================
results = {}
results["EfficientNet-B0"] = train_one_model(model, "EfficientNet-B0", train_loader, val_loader, test_loader)
results["ViT"] = train_one_model(vit_model, "ViT", train_loader, val_loader, test_loader)
results["Swin Transformer"] = train_one_model(swin_model, "Swin Transformer", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Plot Accuracy and Loss
# ============================
plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_acc'], label=f"{model_name} Train")
    plt.plot(data['val_acc'], label=f"{model_name} Val")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    plt.plot(data['train_loss'], label=f"{model_name} Train")
    plt.plot(data['val_loss'], label=f"{model_name} Val")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# STEP 7: Confusion Matrices & Summary
# ============================
summary_df = pd.DataFrame([
    {
        "Model": model_name,
        "Test Accuracy": round(data["test_accuracy"], 4),
        "Precision": round(data["precision"], 4),
        "Recall": round(data["recall"], 4),
        "F1 Score": round(data["f1_score"], 4),
        "Avg Inference Time (s)": round(data.get("avg_infer_time_per_batch", 0.0), 6),
        "Avg Prediction Time (s)": round(data.get("avg_pred_time_per_batch", 0.0), 6)
    }
    for model_name, data in results.items()
])

from IPython.display import display
display(summary_df)

for model_name, model_data in results.items():
    cm = model_data["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_confusion_matrix.png")
    plt.show()


# In[ ]:


Code to Visualize Flip, Rotation, Jitter


# In[ ]:


import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# === Augmentations (same as training, but removed Normalize + ToTensor for visibility)
preview_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])

# === Base image folder (already defined)
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512/split_data/train"
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

# === Visualize one image per class with 3 augmentations
for class_name in class_names:
    class_path = os.path.join(base_dir, class_name)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]

    if not image_files:
        print(f"No images found in {class_name}. Skipping.")
        continue

    img_path = os.path.join(class_path, random.choice(image_files))
    img = Image.open(img_path).convert("RGB")

    # Plot original + 3 augmented versions
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title(f"{class_name}\nOriginal")
    plt.axis("off")

    for i in range(3):
        aug_img = preview_transform(img)
        plt.subplot(1, 4, i+2)
        plt.imshow(aug_img)
        plt.title(f"Augmented {i+1}")
        plt.axis("off")

    plt.suptitle(f"Augmentation Examples - {class_name}", fontsize=14)
    plt.tight_layout()
    plt.show()


# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# FULL CLEAN PRUNING + QUANTIZATION CODE FOR TEST SET ONLY
# 
#  pruning + quantization + evaluation only on the test set based on my trained Swin Transformer model

# the simple ONNX export again:

# In[ ]:


get_ipython().system('pip install onnx')


# Here is your simplified and modified code ready to use for Original ViT

# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128/split_data_128"
batch_size = 32

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_eval)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Images: {len(train_dataset)} | Val Images: {len(val_dataset)} | Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Model (ViT)
# ============================
num_classes = 3

vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = nn.Linear(vit_model.head.in_features, num_classes)
vit_model = vit_model.to(device)

# ============================
# STEP 4: Train and Evaluate Function
# ============================
def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_model.pth")

    model.load_state_dict(best_model_state)

    # ============================
    # Inference & Timing
    # ============================
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    avg_infer_time = total_infer_time / total_samples

    print(f"\nTest Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

# ============================
# STEP 5: Train & Evaluate ViT
# ============================
vit_results = train_and_evaluate(vit_model, "ViT", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(vit_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/ViT_confusion_matrix.png")
plt.show()


# Pruning Code

# In[ ]:


import torch.nn.utils.prune as prune
import copy

# ============================
# STEP 1: Load Best Model
# ============================
pruned_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
pruned_vit.head = nn.Linear(pruned_vit.head.in_features, num_classes)
pruned_vit.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model.pth"))
pruned_vit = pruned_vit.to(device)

# ============================
# STEP 2: Apply Pruning (20% on Linear layers)
# ============================
def prune_model(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model

pruned_vit = prune_model(pruned_vit, amount=0.2)

# OPTIONAL: Check sparsity
def check_sparsity(model):
    total_zeros = 0
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_zeros += torch.sum(module.weight == 0).item()
            total_elements += module.weight.nelement()
    sparsity = 100.0 * total_zeros / total_elements
    print(f"Pruned Model Sparsity: {sparsity:.2f}%")

check_sparsity(pruned_vit)

# ============================
# STEP 3: Evaluate Pruned Model
# ============================
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    avg_infer_time = total_infer_time / total_samples

    print(f"\n[Pruned ViT] Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time: {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

pruned_results = evaluate_model(pruned_vit, test_loader)

# ============================
# STEP 4: Save Confusion Matrix
# ============================
plt.figure(figsize=(6, 5))
sns.heatmap(pruned_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Pruned ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/Pruned_ViT_confusion_matrix.png")
plt.show()


# Quantization Code

# In[ ]:


import copy

# ============================
# STEP 1: Load Best Model
# ============================
quant_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
quant_vit.head = nn.Linear(quant_vit.head.in_features, num_classes)
quant_vit.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model.pth"))
quant_vit = quant_vit.to(device)

# Put model to CPU for quantization
quant_vit.cpu()

# ============================
# STEP 2: Apply Dynamic Quantization
# ============================
quantized_vit = torch.quantization.quantize_dynamic(
    quant_vit,
    {nn.Linear},  # Quantize only Linear layers
    dtype=torch.qint8
)

print("Quantization done!")

# ============================
# STEP 3: Evaluate Quantized Model
# ============================
def evaluate_model_cpu(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cpu(), labels.cpu()

            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    avg_infer_time = total_infer_time / total_samples

    print(f"\n[Quantized ViT] Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time: {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

quantized_results = evaluate_model_cpu(quantized_vit, test_loader)

# ============================
# STEP 4: Save Confusion Matrix
# ============================
plt.figure(figsize=(6, 5))
sns.heatmap(quantized_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Quantized ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/Quantized_ViT_confusion_matrix.png")
plt.show()


# Pruning + Quantization → Evaluate

# In[ ]:


# STEP 1: Load Best ViT model
combo_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
combo_vit.head = nn.Linear(combo_vit.head.in_features, num_classes)
combo_vit.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model.pth"))
combo_vit = combo_vit.to(device)

# STEP 2: Apply Pruning (20% on Linear layers)
def prune_model(model, amount=0.2):
    pruned_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            pruned_modules.append(module)
    return model, pruned_modules

combo_vit, pruned_modules = prune_model(combo_vit, amount=0.2)

# Optional: Check sparsity
check_sparsity(combo_vit)

# STEP 3 (IMPORTANT): Make pruning permanent → REMOVE pruning reparametrization
for module in pruned_modules:
    prune.remove(module, 'weight')

print("Pruning made permanent!")

# Move pruned model to CPU for quantization
combo_vit.cpu()

# STEP 4: Apply Dynamic Quantization
combo_quantized_vit = torch.quantization.quantize_dynamic(
    combo_vit,
    {nn.Linear},
    dtype=torch.qint8
)

print("Pruning + Quantization done!")

# STEP 5: Evaluate Quantized Pruned Model
combo_results = evaluate_model_cpu(combo_quantized_vit, test_loader)

# STEP 6: Save Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(combo_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Pruned + Quantized ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/Pruned_Quantized_ViT_confusion_matrix.png")
plt.show()


# to export to ONNX the ViT_best_model.pth

# In[ ]:


get_ipython().system('mv /content/ViT_best_model.onnx /content/drive/MyDrive/Master_Thesis_Project/')


# In[ ]:


import torch
import timm
import torch.nn as nn

# Load model
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = nn.Linear(vit_model.head.in_features, 3)
vit_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model.pth"))
vit_model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX with higher opset version
torch.onnx.export(vit_model, dummy_input, "ViT_best_model.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=16)

print("Export to ONNX completed with opset 16.")


# In[ ]:


get_ipython().system('pip install onnxruntime')


# In[ ]:


import onnx
import onnxruntime as ort
import numpy as np

# Load ONNX model
onnx_model_path = "/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model.onnx"
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Run test inference
ort_session = ort.InferenceSession(onnx_model_path)

# Dummy input (batch_size=1, 3 channels, 224x224)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

outputs = ort_session.run(None, {'input': dummy_input})
print("ONNX model inference successful! Output shape:", outputs[0].shape)


# In[ ]:


get_ipython().system('pip install tensorflow==2.15')
get_ipython().system('pip install keras==2.15')
get_ipython().system('pip install tensorflow-addons==0.23')
get_ipython().system('pip install onnx onnx-tf --upgrade')


# In[ ]:


get_ipython().system('pip install keras==2.15')


# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


import torch
import timm
import torch.nn as nn

# Define model
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.head = nn.Linear(model.head.in_features, 3)  # You have 3 classes

# Load weights
model_path = "/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

print("Model loaded and ready!")


# In[ ]:


# Install keras2onnx (very simple ONNX -> TF SavedModel)
get_ipython().system('pip install keras2onnx')

# Install onnx (already installed but just in case)
get_ipython().system('pip install onnx')


# In[ ]:


get_ipython().system('pip uninstall tensorflow tensorflow-text tensorflow-addons tensorflow-decision-forests -y')
get_ipython().system('pip install tensorflow==2.18.0 --upgrade --quiet')


# In[ ]:


get_ipython().system('pip install tensorflow-addons==0.22.0')


# In[ ]:


get_ipython().system('pip uninstall -y tensorflow keras tensorflow-addons')
get_ipython().system('pip install tensorflow==2.14 tensorflow-addons==0.22 onnx onnx-tf')



# In[ ]:


get_ipython().system('pip install onnx onnx-tf tensorflow')


# In[ ]:


get_ipython().system('pip uninstall -y tensorflow keras tensorflow-addons onnx-tf')
get_ipython().system('pip install tensorflow==2.14 tensorflow-addons==0.22 onnx-tf==1.10')


# In[ ]:


get_ipython().system('pip install numpy==1.24.4 --force-reinstall')


# In[ ]:


import numpy as np
print(np.__version__)


# In[ ]:


get_ipython().system('pip uninstall onnx-tf -y')
get_ipython().system('pip install onnx-tf==1.12.0')


# In[ ]:


get_ipython().system('pip install numpy==1.24.4')
get_ipython().system('pip install tensorflow==2.14.0')
get_ipython().system('pip install tf2onnx==1.16.1')
get_ipython().system('pip install onnx==1.17.0')


# In[ ]:


get_ipython().system('pip install onnx==1.17.0')
get_ipython().system('pip install tensorflow==2.14.0')
get_ipython().system('pip install tf2onnx==1.16.1')


# In[ ]:


get_ipython().system('python -m tf2onnx.convert --opset 14 --onnx-file /content/ViT_best_model.onnx --saved-model /content/vit_saved_model')


# HOW TO CONVERT ViT_best_model.pth → TFLite

# In[ ]:


# Install required tools
get_ipython().system('pip install -q onnx tensorflow tf2onnx')


# In[ ]:


get_ipython().system('pip install onnx tf2onnx tensorflow')


# In[ ]:


get_ipython().system('pip install numpy==1.24.4 tensorflow==2.14.0 tf2onnx==1.16.1')


# In[ ]:


get_ipython().system('pip install numpy==1.23.5')


# In[ ]:


get_ipython().system('pip install numpy==1.24.3 tensorflow==2.14 tf2onnx')


# In[ ]:


get_ipython().system('pip install numpy==1.23.5 --force-reinstall')


# In[ ]:


get_ipython().system('sudo apt-get update -y')
get_ipython().system('sudo apt-get install python3.10 python3.10-dev python3.10-venv python3.10-distutils -y')
get_ipython().system('sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1')

# check
get_ipython().system('python3 --version')


# In[ ]:


get_ipython().system('python3.10 -m venv vit_env')
get_ipython().system('vit_env/bin/python --version   # Check version → should show Python 3.10')
get_ipython().system('vit_env/bin/python -m pip install --upgrade pip')
get_ipython().system('vit_env/bin/python -m pip install tensorflow==2.14.0 numpy==1.24.4 tf2onnx==1.16.1 onnx')

# Convert to TFLite (inside the correct Python)
get_ipython().system('vit_env/bin/python -c')

get_ipython().system('python3 --version')


# In[ ]:


import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("/content/vit_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("/content/ViT_best_model.tflite", "wb") as f:
    f.write(tflite_model)

print(" TFLite model exported successfully!")


# In[ ]:


import os
os.kill(os.getpid(), 9)


# Raspberry Pi Python Inference (using tflite_runtime)

# In[ ]:


import time
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Load TFLite model
model_path = "ViT_best_model.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare a dummy input (or load your test image here)
dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], dummy_input)

# Run inference
start = time.time()
interpreter.invoke()
end = time.time()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output_data)
print("Inference time: {:.3f} ms".format((end - start) * 1000))


# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# Orginal 256

# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256/split_data_256"
batch_size = 32

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_eval)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Images: {len(train_dataset)} | Val Images: {len(val_dataset)} | Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Model (ViT)
# ============================
num_classes = 3

vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = nn.Linear(vit_model.head.in_features, num_classes)
vit_model = vit_model.to(device)

# ============================
# STEP 4: Train and Evaluate Function
# ============================
def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_model_2.pth")

    model.load_state_dict(best_model_state)

    # ============================
    # Inference & Timing
    # ============================
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    avg_infer_time = total_infer_time / total_samples

    print(f"\nTest Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

# ============================
# STEP 5: Train & Evaluate ViT
# ============================
vit_results = train_and_evaluate(vit_model, "ViT", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(vit_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/ViT_confusion_matrix.png")
plt.show()


# pruning 256

# In[ ]:


import torch.nn.utils.prune as prune
import copy

# ============================
# STEP 1: Load Best Model
# ============================
pruned_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
pruned_vit.head = nn.Linear(pruned_vit.head.in_features, num_classes)
pruned_vit.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model_2.pth"))
pruned_vit = pruned_vit.to(device)

# ============================
# STEP 2: Apply Pruning (20% on Linear layers)
# ============================
def prune_model(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model

pruned_vit = prune_model(pruned_vit, amount=0.2)

# OPTIONAL: Check sparsity
def check_sparsity(model):
    total_zeros = 0
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_zeros += torch.sum(module.weight == 0).item()
            total_elements += module.weight.nelement()
    sparsity = 100.0 * total_zeros / total_elements
    print(f"Pruned Model Sparsity: {sparsity:.2f}%")

check_sparsity(pruned_vit)

# ============================
# STEP 3: Evaluate Pruned Model
# ============================
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    avg_infer_time = total_infer_time / total_samples

    print(f"\n[Pruned ViT] Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time: {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

pruned_results = evaluate_model(pruned_vit, test_loader)

# ============================
# STEP 4: Save Confusion Matrix
# ============================
plt.figure(figsize=(6, 5))
sns.heatmap(pruned_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Pruned ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/Pruned_ViT_confusion_matrix.png")
plt.show()


# Quantization 256

# In[ ]:


import copy

# ============================
# STEP 1: Load Best Model
# ============================
quant_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
quant_vit.head = nn.Linear(quant_vit.head.in_features, num_classes)
quant_vit.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model_2.pth"))
quant_vit = quant_vit.to(device)

# Put model to CPU for quantization
quant_vit.cpu()

# ============================
# STEP 2: Apply Dynamic Quantization
# ============================
quantized_vit = torch.quantization.quantize_dynamic(
    quant_vit,
    {nn.Linear},  # Quantize only Linear layers
    dtype=torch.qint8
)

print("Quantization done!")

# ============================
# STEP 3: Evaluate Quantized Model
# ============================
def evaluate_model_cpu(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cpu(), labels.cpu()

            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    avg_infer_time = total_infer_time / total_samples

    print(f"\n[Quantized ViT] Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time: {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

quantized_results = evaluate_model_cpu(quantized_vit, test_loader)

# ============================
# STEP 4: Save Confusion Matrix
# ============================
plt.figure(figsize=(6, 5))
sns.heatmap(quantized_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Quantized ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/Quantized_ViT_confusion_matrix.png")
plt.show()


# Pruning + Quantization 256

# In[ ]:


# STEP 1: Load Best ViT model
combo_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
combo_vit.head = nn.Linear(combo_vit.head.in_features, num_classes)
combo_vit.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model_2.pth"
))
combo_vit = combo_vit.to(device)

# STEP 2: Apply Pruning (20% on Linear layers)
def prune_model(model, amount=0.2):
    pruned_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            pruned_modules.append(module)
    return model, pruned_modules

combo_vit, pruned_modules = prune_model(combo_vit, amount=0.2)

# Optional: Check sparsity
check_sparsity(combo_vit)

# STEP 3 (IMPORTANT): Make pruning permanent → REMOVE pruning reparametrization
for module in pruned_modules:
    prune.remove(module, 'weight')

print("Pruning made permanent!")

# Move pruned model to CPU for quantization
combo_vit.cpu()

# STEP 4: Apply Dynamic Quantization
combo_quantized_vit = torch.quantization.quantize_dynamic(
    combo_vit,
    {nn.Linear},
    dtype=torch.qint8
)

print("Pruning + Quantization done!")

# STEP 5: Evaluate Quantized Pruned Model
combo_results = evaluate_model_cpu(combo_quantized_vit, test_loader)

# STEP 6: Save Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(combo_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Pruned + Quantized ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/Pruned_Quantized_ViT_confusion_matrix.png")
plt.show()


# In[ ]:


get_ipython().system('pip install onnx')


# In[ ]:


import torch
import timm
import torch.nn as nn

# Load model
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = nn.Linear(vit_model.head.in_features, 3)
vit_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model_2.pth"))
vit_model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX with higher opset version
torch.onnx.export(vit_model, dummy_input, "ViT_best_model_2.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=16)

print("Export to ONNX completed with opset 16.")


# In[ ]:


get_ipython().system('mv /content/ViT_best_model_2.onnx /content/drive/MyDrive/Master_Thesis_Project/')


# In[ ]:


get_ipython().system('pip install onnxruntime')


# Before uploading, it is good practice to make sure my ONNX model runs fine.

# In[ ]:


import onnx
import onnxruntime as ort
import numpy as np

# Load ONNX model
onnx_model_path = "/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model_2.onnx"
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Run test inference
ort_session = ort.InferenceSession(onnx_model_path)

# Dummy input (batch_size=1, 3 channels, 224x224)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

outputs = ort_session.run(None, {'input': dummy_input})
print("ONNX model inference successful! Output shape:", outputs[0].shape)


# Orginal 512

# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512/split_data_512"
batch_size = 32

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_eval)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Images: {len(train_dataset)} | Val Images: {len(val_dataset)} | Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Model (ViT)
# ============================
num_classes = 3

vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = nn.Linear(vit_model.head.in_features, num_classes)
vit_model = vit_model.to(device)

# ============================
# STEP 4: Train and Evaluate Function
# ============================
def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_model_3.pth")

    model.load_state_dict(best_model_state)

    # ============================
    # Inference & Timing
    # ============================
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    avg_infer_time = total_infer_time / total_samples

    print(f"\nTest Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

# ============================
# STEP 5: Train & Evaluate ViT
# ============================
vit_results = train_and_evaluate(vit_model, "ViT", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(vit_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/ViT_confusion_matrix.png")
plt.show()


# pruning 512

# In[ ]:


import torch.nn.utils.prune as prune
import copy

# ============================
# STEP 1: Load Best Model
# ============================
pruned_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
pruned_vit.head = nn.Linear(pruned_vit.head.in_features, num_classes)
pruned_vit.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model_3.pth"))
pruned_vit = pruned_vit.to(device)

# ============================
# STEP 2: Apply Pruning (20% on Linear layers)
# ============================
def prune_model(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model

pruned_vit = prune_model(pruned_vit, amount=0.2)

# OPTIONAL: Check sparsity
def check_sparsity(model):
    total_zeros = 0
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_zeros += torch.sum(module.weight == 0).item()
            total_elements += module.weight.nelement()
    sparsity = 100.0 * total_zeros / total_elements
    print(f"Pruned Model Sparsity: {sparsity:.2f}%")

check_sparsity(pruned_vit)

# ============================
# STEP 3: Evaluate Pruned Model
# ============================
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    avg_infer_time = total_infer_time / total_samples

    print(f"\n[Pruned ViT] Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time: {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

pruned_results = evaluate_model(pruned_vit, test_loader)

# ============================
# STEP 4: Save Confusion Matrix
# ============================
plt.figure(figsize=(6, 5))
sns.heatmap(pruned_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Pruned ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/Pruned_ViT_confusion_matrix.png")
plt.show()


# Quantization 512

# In[ ]:


import copy

# ============================
# STEP 1: Load Best Model
# ============================
quant_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
quant_vit.head = nn.Linear(quant_vit.head.in_features, num_classes)
quant_vit.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model_3.pth"))
quant_vit = quant_vit.to(device)

# Put model to CPU for quantization
quant_vit.cpu()

# ============================
# STEP 2: Apply Dynamic Quantization
# ============================
quantized_vit = torch.quantization.quantize_dynamic(
    quant_vit,
    {nn.Linear},  # Quantize only Linear layers
    dtype=torch.qint8
)

print("Quantization done!")

# ============================
# STEP 3: Evaluate Quantized Model
# ============================
def evaluate_model_cpu(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cpu(), labels.cpu()

            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    avg_infer_time = total_infer_time / total_samples

    print(f"\n[Quantized ViT] Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time: {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

quantized_results = evaluate_model_cpu(quantized_vit, test_loader)

# ============================
# STEP 4: Save Confusion Matrix
# ============================
plt.figure(figsize=(6, 5))
sns.heatmap(quantized_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Quantized ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/Quantized_ViT_confusion_matrix.png")
plt.show()


# Pruning + Quantization 512

# In[ ]:


# STEP 1: Load Best ViT model
combo_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
combo_vit.head = nn.Linear(combo_vit.head.in_features, num_classes)
combo_vit.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model_3.pth"
))
combo_vit = combo_vit.to(device)

# STEP 2: Apply Pruning (20% on Linear layers)
def prune_model(model, amount=0.2):
    pruned_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            pruned_modules.append(module)
    return model, pruned_modules

combo_vit, pruned_modules = prune_model(combo_vit, amount=0.2)

# Optional: Check sparsity
check_sparsity(combo_vit)

# STEP 3 (IMPORTANT): Make pruning permanent → REMOVE pruning reparametrization
for module in pruned_modules:
    prune.remove(module, 'weight')

print("Pruning made permanent!")

# Move pruned model to CPU for quantization
combo_vit.cpu()

# STEP 4: Apply Dynamic Quantization
combo_quantized_vit = torch.quantization.quantize_dynamic(
    combo_vit,
    {nn.Linear},
    dtype=torch.qint8
)

print("Pruning + Quantization done!")

# STEP 5: Evaluate Quantized Pruned Model
combo_results = evaluate_model_cpu(combo_quantized_vit, test_loader)

# STEP 6: Save Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(combo_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Pruned + Quantized ViT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/Pruned_Quantized_ViT_confusion_matrix.png")
plt.show()


# In[ ]:


import torch
import timm
import torch.nn as nn

# Load model
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = nn.Linear(vit_model.head.in_features, 3)
vit_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/ViT_best_model_3.pth"))
vit_model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX with higher opset version
torch.onnx.export(vit_model, dummy_input, "ViT_best_model_3.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=16)

print("Export to ONNX completed with opset 16.")


# In[ ]:


get_ipython().system('mv /content/ViT_best_model_3.onnx /content/drive/MyDrive/Master_Thesis_Project/')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pip install onnx-tf tensorflow')


# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# for the efficientNet-B0

# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128/split_data_128"
batch_size = 32

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_eval)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Images: {len(train_dataset)} | Val Images: {len(val_dataset)} | Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Model (EfficientNet-B0)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)  # Load EfficientNet-B0 with ImageNet pretrained weights
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)  # Replace classifier with 3-class version
effnet_model = effnet_model.to(device)  # Send model to GPU or CPU


# ============================
# STEP 4: Train and Evaluate Function
# ============================
def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_modelEff_128.pth")

    model.load_state_dict(best_model_state)

    # ============================
    # Inference & Timing
    # ============================
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    avg_infer_time = total_infer_time / total_samples

    print(f"\nTest Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

# ============================
# STEP 5: Train & Evaluate EfficientNet
# ============================
effnet_results = train_and_evaluate(effnet_model, "EfficientNet", train_loader, val_loader, test_loader)


# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(effnet_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_confusion_matrix.png")
plt.show()


# ONNX Export Code for Orginal

# In[ ]:


import torch
import timm
import torch.nn as nn

# ============================
# STEP 1: Load Best Trained Model
# ============================
num_classes = 3

# Create model and load the best trained weights
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_128.pth"))
model.eval()

print("Loaded trained model.")

# ============================
# STEP 2: Export to ONNX
# ============================

# Define dummy input → important → use correct shape → EfficientNet B0 expects (1, 3, 224, 224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
onnx_export_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_128.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_export_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

print(f"Model exported to ONNX -> {onnx_export_path}")


# Efficient net Pruning

# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import timm
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128/split_data_128"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Trained Model
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_128.pth"))
effnet_model = effnet_model.to(device)
effnet_model.eval()

print("Loaded best trained model.")

# ============================
# STEP 4: Apply Pruning
# ============================
parameters_to_prune = []
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

# Prune 30% of weights globally
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print("Pruning applied.")

# Remove pruning hooks
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning finalized and permanent.")

# ============================
# STEP 5: Evaluate Pruned Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

effnet_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        start_time = time.perf_counter()
        outputs = effnet_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nPruned Model - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Pruned - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_confusion_matrix.png")
plt.show()


# ONNX Export Code for pruning

# In[ ]:


import torch
import timm
import torch.nn as nn
import torch.nn.utils.prune as prune

# ============================
# STEP 1: Load Best Trained Model and Apply Pruning (again → needed to export with pruning)
# ============================

num_classes = 3

# Create model
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_128.pth"))
model = model.to('cpu')
model.eval()

print("Loaded trained model.")

# Apply pruning again
parameters_to_prune = []
for module in model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print("Pruning applied.")

# Remove pruning re-parametrization → important for ONNX export
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning finalized and ready for export.")

# ============================
# STEP 2: Export to ONNX
# ============================

dummy_input = torch.randn(1, 3, 224, 224)

onnx_export_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_modelEff_128.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_export_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

print(f"Pruned model exported to ONNX -> {onnx_export_path}")


# STRUCTURED PRUNING EXAMPLE

# In[ ]:


import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import timm
import torch.nn.utils.prune as prune
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import onnx
import onnxruntime as ort

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ============================
# Load Best Trained Model
# ============================
num_classes = 3
effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_128.pth"))
effnet_model = effnet_model.to(device)
effnet_model.train()

print("Loaded trained model.")

# ============================
# Apply Structured Pruning (Moderate → safer)
# ============================
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=0)  # 20% only now
    elif isinstance(module, nn.Linear):
        prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=0)

print("Structured pruning applied (20%).")

# ============================
# Fine-tuning (Recover accuracy)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128/split_data_128"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_eval)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(effnet_model.parameters(), lr=0.0005, momentum=0.9)

effnet_model.train()
for epoch in range(20):  # Small fine-tune → 20 epochs is enough
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = effnet_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Fine-tuning Epoch {epoch+1}/5 | Loss: {running_loss/len(train_loader):.4f}")

print("Fine-tuning done.")

# Remove pruning re-parametrization (make permanent)
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.remove(module, 'weight')

print("Pruning finalized and permanent.")

# ============================
# Export to ONNX (with dynamic axes)
# ============================
effnet_model = effnet_model.to("cpu")
effnet_model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_finetuned.onnx"

torch.onnx.export(
    effnet_model, dummy_input, onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)

print("Model exported to ONNX with dynamic batch size!")

# ============================
# Load Test Dataset
# ============================
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Test Images: {len(test_dataset)}")

# ============================
# Inference with ONNX (GPU)
# ============================
ort_session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

for images, labels in test_loader:
    images = images.numpy()
    labels = labels.numpy()

    start_time = time.perf_counter()
    ort_outputs = ort_session.run(None, {"input": images})
    end_time = time.perf_counter()

    outputs = torch.tensor(ort_outputs[0])
    _, predicted = torch.max(outputs, 1)

    total_infer_time += (end_time - start_time)
    total_samples += labels.shape[0]

    y_true.extend(labels)
    y_pred.extend(predicted.numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\n[ONNX Pruned + Fine-tuned + GPU] Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet (Structured Pruned + Fine-tuned + ONNX + GPU) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_structured_pruned_finetuned_onnx_confusion_matrix.png")
plt.show()

print("Done and saved confusion matrix.")


# QUANTIZATION CODE EFFICIENTnet

# In[ ]:


import torch
import timm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 1: Load best trained model
# ============================
num_classes = 3
effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_128.pth"))
effnet_model = effnet_model.to('cpu')   # Quantization only works on CPU
effnet_model.eval()

print("Loaded model and switched to CPU for quantization.")

# ============================
# STEP 2: Apply Quantization (Dynamic Quantization)
# ============================
quantized_model = torch.quantization.quantize_dynamic(
    effnet_model,
    {nn.Linear},  # only linear layers → safe and effective
    dtype=torch.qint8
)

print("Quantization applied.")

# ============================
# STEP 3: Data Preparation (test set)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128/split_data_128"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 4: Evaluate Quantized Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

quantized_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cpu'), labels.to('cpu')
        start_time = time.perf_counter()
        outputs = quantized_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nQuantized Model - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 5: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Quantized - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quantized_confusion_matrix.png")
plt.show()

# ============================
# STEP 6: Save Quantized Model
# ============================
torch.save(quantized_model.state_dict(), "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quantized_model.pth")
print("Quantized model saved.")


# In[ ]:


from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_128.onnx",
    model_output="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quant_128.onnx",
    weight_type=QuantType.QInt8
)

print("Quantized model saved.")


# ONNX Export Code → for Quantized Model (128)

# In[ ]:


import torch
import timm
import torch.nn as nn

# ============================
# STEP 1: Load best trained model and quantize it again (needed for ONNX export)
# ============================

num_classes = 3
effnet_model = timm.create_model("efficientnet_b0", pretrained=False)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_128.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded model and prepared for quantization.")

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    effnet_model,
    {nn.Linear},
    dtype=torch.qint8
)

print("Quantization applied.")

# ============================
# STEP 2: Export to ONNX
# ============================

dummy_input = torch.randn(1, 3, 224, 224)

onnx_export_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quantized_modelEff_128.onnx"

torch.onnx.export(
    quantized_model,
    dummy_input,
    onnx_export_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

print(f"Quantized model exported to ONNX -> {onnx_export_path}")


# PRUNING + QUANTIZATION + EVALUATION

# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import timm
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup (Test Only)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/128/split_data_128"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Trained Model
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_128.pth"))
effnet_model = effnet_model.to('cpu')  # pruning and quantization for CPU only
effnet_model.eval()

print("Loaded best trained model.")

# ============================
# STEP 4: Apply Pruning
# ============================
parameters_to_prune = []
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

# Apply 30% global unstructured pruning
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print("Pruning applied.")

# Remove pruning re-parametrization
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning finalized and permanent.")

# ============================
# STEP 5: Apply Quantization
# ============================
quantized_model = torch.quantization.quantize_dynamic(
    effnet_model,
    {nn.Linear},  # Quantize only linear layers
    dtype=torch.qint8
)

print("Quantization applied.")

# ============================
# STEP 6: Evaluate Pruned + Quantized Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

quantized_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cpu'), labels.to('cpu')
        start_time = time.perf_counter()
        outputs = quantized_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nPruned + Quantized Model - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 7: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Pruned + Quantized - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_quantized_confusion_matrix.png")
plt.show()

# ============================
# STEP 8: Save Final Model
# ============================
torch.save(quantized_model.state_dict(), "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_quantized_model.pth")
print("Pruned + Quantized model saved.")


# In[ ]:


from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_modelEff_128.onnx",
    model_output="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_quant_128.onnx",
    weight_type=QuantType.QInt8
)

print("Quantized model saved.")


# 256x256

# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


# ============================
# STEP 1: Imports & Configuration
# ============================
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256/split_data_256"
batch_size = 32

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_eval)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Images: {len(train_dataset)} | Val Images: {len(val_dataset)} | Test Images: {len(test_dataset)}")


# ============================
# STEP 3: Load Best Model (EfficientNet-B0)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)  # Load EfficientNet-B0 with ImageNet pretrained weights
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)  # Replace classifier with 3-class version
effnet_model = effnet_model.to(device)  # Send model to GPU or CPU


# ============================
# STEP 4: Train and Evaluate Function
# ============================
def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_modelEff_256.pth")


    model.load_state_dict(best_model_state)

    # ============================
    # Inference & Timing
    # ============================
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    avg_infer_time = total_infer_time / total_samples

    print(f"\nTest Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

# ============================
# STEP 5: Train & Evaluate EfficientNet
# ============================
effnet_results = train_and_evaluate(effnet_model, "EfficientNet", train_loader, val_loader, test_loader)


# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(effnet_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet - Confusion Matrix (256 Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_confusion_matrix_256.png")
plt.show()


# Export Trained 256 Model to ONNX

# In[ ]:


import torch
import timm
import torch.nn as nn

# ============================
# STEP 1: Load trained model (256 version)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=False)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)

# Load trained weights
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_256.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded trained model (256 version).")

# ============================
# STEP 2: Export to ONNX
# ============================
dummy_input = torch.randn(1, 3, 224, 224)

onnx_export_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_256.onnx"

torch.onnx.export(
    effnet_model,
    dummy_input,
    onnx_export_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11,   # good compatibility
    do_constant_folding=True
)

print(f"Model exported to ONNX at: {onnx_export_path}")


# pruning 256x256 efficint

# In[ ]:


import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import timm
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup (256 version)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256/split_data_256"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Trained Model (256 version)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_256.pth"))
effnet_model = effnet_model.to(device)
effnet_model.eval()

print("Loaded best trained model.")

# ============================
# STEP 4: Apply Pruning
# ============================
parameters_to_prune = []
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print("Pruning applied.")

for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning finalized and permanent.")

# ============================
# STEP 5: Evaluate Pruned Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

effnet_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        start_time = time.perf_counter()
        outputs = effnet_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nPruned Model (256) - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Pruned (256) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_confusion_matrix_256.png")
plt.show()


# ONNX Export Code for Pruned (256 version)

# In[ ]:


import torch
import timm
import torch.nn as nn
import torch.nn.utils.prune as prune

# ============================
# STEP 1: Load Pruned Model (256 version)
# ============================
num_classes = 3

# Load model
effnet_model = timm.create_model("efficientnet_b0", pretrained=False)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)

# Load trained weights (pruned version already in memory)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_256.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded model for pruning.")

# Apply pruning (same as during inference)
parameters_to_prune = []
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print("Pruning applied.")

# Remove pruning hooks → must do before ONNX export
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning finalized (permanent), ready to export.")

# ============================
# STEP 2: Export to ONNX
# ============================
dummy_input = torch.randn(1, 3, 224, 224)

onnx_export_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_modelEff_256.onnx"

torch.onnx.export(
    effnet_model,
    dummy_input,
    onnx_export_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

print(f"Pruned model exported to ONNX at: {onnx_export_path}")


# quantization 256x256

# In[ ]:


import torch
import timm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 1: Load best trained model (256 version)
# ============================
num_classes = 3
effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_256.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded model and switched to CPU for quantization.")

# ============================
# STEP 2: Apply Quantization (Dynamic Quantization)
# ============================
quantized_model = torch.quantization.quantize_dynamic(
    effnet_model,
    {nn.Linear},
    dtype=torch.qint8
)

print("Quantization applied.")

# ============================
# STEP 3: Data Preparation (256 test set)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256/split_data_256"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 4: Evaluate Quantized Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

quantized_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cpu'), labels.to('cpu')
        start_time = time.perf_counter()
        outputs = quantized_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nQuantized Model (256) - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 5: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Quantized (256) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quantized_confusion_matrix_256.png")
plt.show()

# ============================
# STEP 6: Save Quantized Model
# ============================
torch.save(quantized_model.state_dict(), "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quantized_model_256.pth")
print("Quantized model saved.")


# In[ ]:


from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_256.onnx",
    model_output="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quant_256.onnx",
    weight_type=QuantType.QInt8
)

print("Quantized model saved.")


# FULL CODE → FX Graph Mode Quantization + Export to TFLite (for EfficientNet model)

# In[ ]:


get_ipython().system('pip uninstall torch -y')
get_ipython().system('pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu')


# In[ ]:


import torch
print(torch.__version__)


# In[ ]:


get_ipython().system('pip install numpy==1.24.4')


# In[ ]:


get_ipython().system('pip install pytorch2keras tensorflow')


# pruning + quantization 256x256

# In[ ]:


import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import timm
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup (Test Only - 256 version)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/256/split_data_256"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Trained Model (256 version)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_256.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded best trained model.")

# ============================
# STEP 4: Apply Pruning
# ============================
parameters_to_prune = []
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print("Pruning applied.")

for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning finalized and permanent.")

# ============================
# STEP 5: Apply Quantization
# ============================
quantized_model = torch.quantization.quantize_dynamic(
    effnet_model,
    {nn.Linear},
    dtype=torch.qint8
)

print("Quantization applied.")

# ============================
# STEP 6: Evaluate Pruned + Quantized Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

quantized_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cpu'), labels.to('cpu')
        start_time = time.perf_counter()
        outputs = quantized_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nPruned + Quantized Model (256) - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 7: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Pruned + Quantized (256) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_quantized_confusion_matrix_256.png")
plt.show()

# ============================
# STEP 8: Save Final Model
# ============================
torch.save(quantized_model.state_dict(), "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_quantized_model_256.pth")
print("Pruned + Quantized model saved.")


# In[ ]:


from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_modelEff_256.onnx",
    model_output="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_quant_256.onnx",
    weight_type=QuantType.QInt8
)

print(" Quantized model saved.")


# 512x512

# In[ ]:


#STEP 1: Setup in Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define Paths
import os
base_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512"

class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']


# In[ ]:


import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup (512 version)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512/split_data_512"
batch_size = 32

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_eval)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Images: {len(train_dataset)} | Val Images: {len(val_dataset)} | Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load EfficientNet-B0
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model = effnet_model.to(device)

# ============================
# STEP 4: Train and Evaluate Function
# ============================
def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"/content/drive/MyDrive/Master_Thesis_Project/{model_name}_best_modelEff_512.pth")

    model.load_state_dict(best_model_state)

    # ============================
    # Inference & Timing
    # ============================
    y_true, y_pred = [], []
    total_infer_time = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            _, predicted = torch.max(outputs, 1)

            total_infer_time += (end_time - start_time)
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    avg_infer_time = total_infer_time / total_samples

    print(f"\nTest Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "avg_infer_time_per_sample": avg_infer_time
    }

# ============================
# STEP 5: Train & Evaluate EfficientNet
# ============================
effnet_results = train_and_evaluate(effnet_model, "EfficientNet", train_loader, val_loader, test_loader)

# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(effnet_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet - Confusion Matrix (512 Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_confusion_matrix_512.png")
plt.show()


# ONNX Export Code for EfficientNet-B0 Original (Trained on 512 Dataset)

# In[ ]:


import torch
import timm
import torch.nn as nn

# ============================
# STEP 1: Load Best Trained Model (512 version)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=False)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)

# Load the best model trained on 512 dataset
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_512.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded trained model for 512 dataset.")

# ============================
# STEP 2: Export to ONNX
# ============================

# Create dummy input → batch size 1 and input shape 224x224 (EfficientNet default)
dummy_input = torch.randn(1, 3, 224, 224)

# Define export path
onnx_export_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_modelEff_512.onnx"

# Export to ONNX
torch.onnx.export(
    effnet_model,
    dummy_input,
    onnx_export_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11,
    do_constant_folding=True  # optimize constant expressions
)

print(f"Model exported successfully to: {onnx_export_path}")


# In[ ]:





# pruning 512x512

# In[ ]:


import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import timm
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup (512 version)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512/split_data_512"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Trained Model (512 version)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_512.pth"))
effnet_model = effnet_model.to(device)
effnet_model.eval()

print("Loaded best trained model.")

# ============================
# STEP 4: Apply Pruning
# ============================
parameters_to_prune = []
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

#prune.L1Unstructured: This means that PyTorch is removing individual weights (the smallest magnitude ones), regardless of their location — across all selected layers.
#prune.ln_structured: Structured pruning removes entire channels or neurons. This causes a sudden drop in representational capacity, especially if done blindly without retraining.
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print("Pruning applied.")

for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning finalized and permanent.")

# ============================
# STEP 5: Evaluate Pruned Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

effnet_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        start_time = time.perf_counter()
        outputs = effnet_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nPruned Model (512) - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Pruned (512) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_confusion_matrix_512.png")
plt.show()


# ONNX Export Code for Pruned EfficientNet-B0 (512 version)

# In[ ]:


import torch
import timm
import torch.nn as nn

# ============================
# STEP 1: Load Pruned Model (512 version)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=False)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)

# Load pruned model (after pruning and removing hooks → permanent)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_512.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded pruned model (512 version).")

# ============================
# STEP 2: Export to ONNX
# ============================

# Create dummy input → batch size 1 and input shape 224x224 (EfficientNet default)
dummy_input = torch.randn(1, 3, 224, 224)

# Define export path
onnx_export_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_modelEff_512.onnx"

# Export to ONNX
torch.onnx.export(
    effnet_model,
    dummy_input,
    onnx_export_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

print(f"Pruned model exported successfully to: {onnx_export_path}")


# In[ ]:


import onnx


# In[ ]:


#Unstructured pruning is easier, safer, and works out of the box in PyTorch.
#Structured pruning can lead to massive drops in accuracy and even slower speeds, unless carefully retrained and exported to optimized inference engines.
# maybe we can improve by retraining or fine-tuning. But is that important!!
import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import timm
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup (512 version)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512/split_data_512"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Trained Model (512 version)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_512.pth"))
effnet_model = effnet_model.to(device)
effnet_model.eval()

print("Loaded best trained model.")

# ============================
# STEP 4: Apply Pruning
# ============================
parameters_to_prune = []
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

# STRUCTURED PRUNING
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=0.3, n=1, dim=0)
    elif isinstance(module, nn.Linear):
        prune.ln_structured(module, name='weight', amount=0.3, n=1, dim=1)

print("Pruning applied.")

for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning finalized and permanent.")

# ============================
# STEP 5: Evaluate Pruned Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

effnet_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        start_time = time.perf_counter()
        outputs = effnet_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nPruned Model (512) - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 6: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Pruned (512) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_confusion_matrix_512.png")
plt.show()


# quantization 512x512

# In[ ]:


import torch
import timm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 1: Load best trained model (512 version)
# ============================
num_classes = 3
effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_512.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded model and switched to CPU for quantization.")

# ============================
# STEP 2: Apply Quantization (Dynamic Quantization)
# ============================
quantized_model = torch.quantization.quantize_dynamic(
    effnet_model,
    {nn.Linear},
    dtype=torch.qint8
)

print("Quantization applied.")

# ============================
# STEP 3: Data Preparation (512 test set)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512/split_data_512"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 4: Evaluate Quantized Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

quantized_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cpu'), labels.to('cpu')
        start_time = time.perf_counter()
        outputs = quantized_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nQuantized Model (512) - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 5: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Quantized (512) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quantized_confusion_matrix_512.png")
plt.show()

# ============================
# STEP 6: Save Quantized Model
# ============================
torch.save(quantized_model.state_dict(), "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quantized_model_512.pth")
print("Quantized model saved.")


# In[ ]:


from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_modelEff_512.onnx",
    model_output="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quant.onnx",
    weight_type=QuantType.QInt8
)

print("Quantized model saved.")


# In[ ]:


import torch
import timm
import torch.nn as nn

# ============================
# STEP 1: Load Pruned Model (512 version)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=False)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)

# Load pruned model (after pruning and removing hooks → permanent)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_512.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded pruned model (512 version).")

# ============================
# STEP 2: Export to ONNX
# ============================

# Create dummy input → batch size 1 and input shape 224x224 (EfficientNet default)
dummy_input = torch.randn(1, 3, 224, 224)

# Define export path
onnx_export_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_quantized_model_512.pth"

# Export to ONNX
torch.onnx.export(
    effnet_model,
    dummy_input,
    onnx_export_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

print(f"Pruned model exported successfully to: {onnx_export_path}")


# In[ ]:


get_ipython().system('pip install onnxruntime onnx onnxruntime-tools')


# In[ ]:


from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_modelEff_512.onnx",
    model_output="/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_quant.onnx",
    weight_type=QuantType.QInt8
)

print(" Quantized model saved.")


# In[ ]:


# Optionally install
get_ipython().system('pip install onnxruntime onnxruntime-tools')

# Use ONNX Runtime's quantization CLI or Python API


# In[ ]:


import timm
import torch
import torch.nn as nn

num_classes = 3
model = timm.create_model('efficientnet_b0', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Load original (non-quantized) weights
state_dict = torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_512.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_512_original.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(" Exported original model to ONNX.")


# In[ ]:


from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths
float_onnx_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_512_original.onnx"
quantized_onnx_path = "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_512_quantized.onnx"

# Quantize the ONNX model weights to int8
quantize_dynamic(
    model_input=float_onnx_path,
    model_output=quantized_onnx_path,
    weight_type=QuantType.QInt8  # Try QUInt8 if needed
)

print(f" Quantized ONNX model saved to:\n{quantized_onnx_path}")


# Pruned + Quantized 512x512

# In[ ]:


import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import timm
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# STEP 2: Data Setup (Test Only - 512 version)
# ============================
data_dir = "/content/drive/MyDrive/Master_Thesis_Project/SnowRoadCropped/512/split_data_512"
batch_size = 32

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test Images: {len(test_dataset)}")

# ============================
# STEP 3: Load Best Trained Model (512 version)
# ============================
num_classes = 3

effnet_model = timm.create_model("efficientnet_b0", pretrained=True)
effnet_model.classifier = nn.Linear(effnet_model.classifier.in_features, num_classes)
effnet_model.load_state_dict(torch.load("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_best_modelEff_512.pth"))
effnet_model = effnet_model.to('cpu')
effnet_model.eval()

print("Loaded best trained model.")

# ============================
# STEP 4: Apply Pruning
# ============================
parameters_to_prune = []
for module in effnet_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print("Pruning applied.")

for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning finalized and permanent.")

# ============================
# STEP 5: Apply Quantization
# ============================
quantized_model = torch.quantization.quantize_dynamic(
    effnet_model,
    {nn.Linear},
    dtype=torch.qint8
)

print("Quantization applied.")

# ============================
# STEP 6: Evaluate Pruned + Quantized Model
# ============================
y_true, y_pred = [], []
total_infer_time = 0.0
total_samples = 0

quantized_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cpu'), labels.to('cpu')
        start_time = time.perf_counter()
        outputs = quantized_model(images)
        end_time = time.perf_counter()

        _, predicted = torch.max(outputs, 1)

        total_infer_time += (end_time - start_time)
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

avg_infer_time = total_infer_time / total_samples

print(f"\nPruned + Quantized Model (512) - Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Avg Inference Time (s): {avg_infer_time:.6f}")

# ============================
# STEP 7: Save Confusion Matrix
# ============================
class_names = ['Class1_CarTouch', 'Class2_UnTouch', 'Class3_AllTouch']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("EfficientNet Pruned + Quantized (512) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_quantized_confusion_matrix_512.png")
plt.show()

# ============================
# STEP 8: Save Final Model
# ============================
torch.save(quantized_model.state_dict(), "/content/drive/MyDrive/Master_Thesis_Project/EfficientNet_pruned_quantized_model_512.pth")
print("Pruned + Quantized model saved.")

