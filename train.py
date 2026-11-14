import os
import time
import copy
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ============ НАСТРОЙКИ ============
DATA_DIR = Path("data")
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}")

# ============ СКАЧИВАНИЕ ДАТАСЕТА ============
# Берём уменьшенную версию Kaggle Cats vs Dogs (~540 МБ, 25к картинок)
URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
ZIP_PATH = DATA_DIR / "cats_dogs.zip"
RAW_DIR = DATA_DIR / "raw"

DATA_DIR.mkdir(exist_ok=True)

if not RAW_DIR.exists():
    if not ZIP_PATH.exists():
        print("Скачиваем датасет (~540 МБ)...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("Скачано.")
    print("Распаковываем...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    # внутри будет PetImages/Cat и PetImages/Dog
    RAW_DIR.mkdir(exist_ok=True)
    os.rename(DATA_DIR / "PetImages", RAW_DIR / "PetImages")

# ============ ОЧИСТКА БИТЫХ ФАЙЛОВ ============
# В датасете есть несколько битых jpg — убираем, иначе DataLoader упадёт
def clean_broken_images(folder):
    from PIL import Image
    removed = 0
    for p in Path(folder).rglob("*.jpg"):
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            p.unlink()
            removed += 1
    return removed

if not (DATA_DIR / ".cleaned").exists():
    n = clean_broken_images(RAW_DIR / "PetImages")
    print(f"Удалено битых картинок: {n}")
    (DATA_DIR / ".cleaned").touch()

# ============ ТРАНСФОРМАЦИИ ============
# ResNet ждёт 224x224 и нормализацию ImageNet
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ============ ДАТАСЕТ ============
full_ds = datasets.ImageFolder(RAW_DIR / "PetImages", transform=train_tf)
print(f"Классы: {full_ds.classes}")
print(f"Всего картинок: {len(full_ds)}")

# 80/20 split
n_total = len(full_ds)
n_val = int(n_total * 0.2)
n_train = n_total - n_val
train_ds, val_ds = torch.utils.data.random_split(
    full_ds, [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)
# val без аугментаций
val_ds.dataset = copy.deepcopy(full_ds)
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============ МОДЕЛЬ ============
# Transfer learning: берём ResNet18, обученный на ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Замораживаем всё, кроме последнего слоя
for p in model.parameters():
    p.requires_grad = False
# Меняем последний слой под 2 класса
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ============ ОБУЧЕНИЕ ============
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\nНачинаем обучение...")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
    va_loss, va_acc = run_epoch(model, val_loader, criterion)
    dt = time.time() - t0
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)
    print(f"Эпоха {epoch}/{EPOCHS} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
          f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
          f"{dt:.1f}s")

# ============ СОХРАНЕНИЕ ============
torch.save({
    "model_state": model.state_dict(),
    "classes": full_ds.classes,
}, "cat_dog_model.pth")
print("\nМодель сохранена: cat_dog_model.pth")

# ============ ГРАФИКИ ============
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="train")
plt.plot(history["val_acc"], label="val")
plt.title("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png")
print("График сохранён: training_curves.png")