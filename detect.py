import sys
from pathlib import Path
from ultralytics import YOLO

print("=" * 50)
print("🐱🐶 KamCat YOLO v2.3 — детектор животных 🎯")
print("=" * 50)

# yolov8x — максимальная точность, 130 МБ
# Скачается при первом запуске (~30-60 сек)
model = YOLO("yolov8x.pt")

# Классы COCO: 15=cat, 16=dog
TARGET_CLASSES = {15: "🐱 Кот", 16: "🐶 Собака"}

def detect(image_path, conf_threshold=0.2):
    print(f"\n📷 Обработка: {image_path}")

    # conf=0.2 — самый низкий разумный порог, чтобы поймать мелких/сложных
    # iou=0.5  — сливаем пересекающиеся рамки
    results = model(image_path, conf=conf_threshold, iou=0.5, verbose=False)
    r = results[0]

    # Собираем найденных котов/собак, сортируем по уверенности
    found = []
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls in TARGET_CLASSES:
            found.append((TARGET_CLASSES[cls], conf))

    found.sort(key=lambda x: x[1], reverse=True)

    # Сохраняем картинку с рамками
    out_path = Path(image_path).stem + "_detected.jpg"
    r.save(filename=out_path)

    if found:
        print(f"  Найдено животных (кот/собака): {len(found)}")
        for name, conf in found:
            print(f"    {name} — уверенность {conf*100:.1f}%")
    else:
        print("  Животных (кот/собака) не найдено")

    # Диагностика: показываем ВСЁ, что YOLO увидел
    if len(r.boxes) > 0:
        print(f"  📋 Все объекты на фото ({len(r.boxes)}):")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = r.names[cls]
            marker = "✅" if cls in TARGET_CLASSES else "⚠️"
            print(f"    {marker} {name} — {conf*100:.1f}%")

    print(f"  💾 Результат сохранён: {out_path}")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nИспользование: python detect.py <путь_к_фото> [ещё_фото...]")
        sys.exit(1)

    for p in sys.argv[1:]:
        if Path(p).exists():
            detect(p)
        else:
            print(f"❌ Файл не найден: {p}")

    print("\n✅ Готово!")