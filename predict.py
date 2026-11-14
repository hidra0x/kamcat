import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

print("=" * 50)
print("🐱 KamCat v1.0 — кот или собака? 🐶")
print("=" * 50)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("kamcat.pth", map_location=DEVICE)
classes = ckpt["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(ckpt["model_state"])
model = model.to(DEVICE).eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    cat_p, dog_p = probs[0].item(), probs[1].item()
    print(f"\nФото: {image_path}")
    print(f"  🐱 Кот:    {cat_p*100:5.1f}%")
    print(f"  🐶 Собака: {dog_p*100:5.1f}%")
    if cat_p > dog_p:
        print(f"  → Это КОТ! 🐱 (уверенность {cat_p*100:.1f}%)")
    else:
        print(f"  → Это СОБАКА! 🐶 (уверенность {dog_p*100:.1f}%)")

if len(sys.argv) < 2:
    print("\nИспользование: python predict.py <путь_к_фото>")
    sys.exit(1)

for p in sys.argv[1:]:
    if Path(p).exists():
        predict(p)
    else:
        print(f"❌ Файл не найден: {p}")