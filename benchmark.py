import matplotlib.pyplot as plt

# Классификаторы кот/собака (честное сравнение)
models = [
    ("Ребёнок (3 года)", 90.0, 3*365*24*60, "gray"),
    ("KamCat v1.0\n(ResNet18)", 97.5, 9, "#f5c518"),
    ("EfficientNet-B0", 98.0, 20, "#3b82f6"),
    ("ResNet50", 98.5, 30, "#22c55e"),
    ("Человек (взрослый)", 99.0, 20*365*24*60, "#ef4444"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

# === График 1: точность ===
names = [m[0] for m in models]
accs = [m[1] for m in models]
colors = [m[3] for m in models]

bars = ax1.barh(names, accs, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_xlim(85, 100)
ax1.set_xlabel("Точность (%)", fontsize=11)
ax1.set_title("Точность классификации: кот или собака", fontsize=13, fontweight="bold")
ax1.grid(axis="x", alpha=0.3)
for bar, acc in zip(bars, accs):
    ax1.text(acc + 0.2, bar.get_y() + bar.get_height()/2,
             f"{acc}%", va="center", fontsize=10, fontweight="bold")

# === График 2: время vs точность ===
times = [m[2] for m in models]
ax2.scatter(times, accs, c=colors, s=250, zorder=3,
            edgecolor="black", linewidth=1.2)
for name, acc, t, c in models:
    ax2.annotate(name.replace("\n", " "), (t, acc),
                 textcoords="offset points", xytext=(10, 6),
                 fontsize=9, fontweight="bold")
ax2.set_xscale("log")
ax2.set_xlabel("Время обучения (минут, лог-шкала)", fontsize=11)
ax2.set_ylabel("Точность (%)", fontsize=11)
ax2.set_title("Эффективность: KamCat — оптимальный баланс", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3, which="both")
ax2.set_ylim(88, 100)

# Подпись внизу
fig.text(0.5, -0.02,
         "KamCat: 97.5% за 9 минут на ноутбуке  |  "
         "ResNet50: +1% за 3x больше времени  |  "
         "Человек: 99% за 20 лет",
         ha="center", fontsize=9, style="italic", color="#666")

plt.tight_layout()
plt.savefig("benchmark.png", dpi=150, bbox_inches="tight")
print("✅ Сохранено: benchmark.png")