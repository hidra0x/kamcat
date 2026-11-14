"""
KamCat v1.0 — AI Toolkit
CLI-интерфейс: классификатор, детектор, генератор картинок, генератор текста.
"""

import sys
from pathlib import Path

# ============ ЦВЕТА (ANSI) ============
ORANGE = "\033[38;5;214m"
GREEN  = "\033[92m"
RED    = "\033[91m"
GRAY   = "\033[90m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ============ ГЛОБАЛЬНОЕ СОСТОЯНИЕ ============
_state = {
    "classifier": None,
    "yolo_model": None,
    "yolo_model_name": "yolov8x",
    "sd_pipe": None,
    "text_model": None,
    "device": None,
}

# ============ УТИЛИТЫ ============

def clear():
    import os
    os.system("cls" if os.name == "nt" else "clear")


def hr(char="─", width=54):
    print(f"{GRAY}{char * width}{RESET}")


def print_header():
    print(f"{ORANGE}╔{'═' * 52}╗{RESET}")
    print(f"{ORANGE}║{BOLD}        🐱🐶  KamCat v1.0 — AI Toolkit           {RESET}{ORANGE}║{RESET}")
    print(f"{ORANGE}║{GRAY}              by Артём • 2026                    {RESET}{ORANGE}║{RESET}")
    print(f"{ORANGE}╚{'═' * 52}╝{RESET}")


def ask(prompt, default=None):
    try:
        val = input(f"\n  {ORANGE}→{RESET} {prompt}: ").strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        return None


def ask_path(prompt):
    while True:
        p = ask(prompt)
        if p is None:
            return None
        if p.lower() in ("q", "quit", "exit"):
            return None
        path = Path(p.strip('"').strip("'"))
        if path.exists():
            return path
        print(f"  {RED}❌ Файл не найден: {p}{RESET}")


def get_device():
    if _state["device"] is None:
        import torch
        _state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _state["device"]


# ============ КЛАССИФИКАТОР ============

def load_classifier():
    if _state["classifier"] is not None:
        return _state["classifier"]

    import torch
    import torch.nn as nn
    from torchvision import transforms, models

    model_path = Path("kamcat.pth")
    if not model_path.exists():
        print(f"  {RED}❌ Файл kamcat.pth не найден!{RESET}")
        return None

    print(f"  {GRAY}Загружаем классификатор...{RESET}")
    device = get_device()
    ckpt = torch.load(model_path, map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    _state["classifier"] = {"model": model, "tf": tf, "device": device, "classes": ckpt["classes"]}
    print(f"  {GREEN}✅ Классификатор загружен{RESET}")
    return _state["classifier"]


def classify_image(image_path):
    import torch
    from PIL import Image

    clf = load_classifier()
    if clf is None:
        return

    img = Image.open(image_path).convert("RGB")
    x = clf["tf"](img).unsqueeze(0).to(clf["device"])
    with torch.no_grad():
        probs = torch.softmax(clf["model"](x), dim=1)[0]

    cat_p, dog_p = probs[0].item(), probs[1].item()
    print(f"\n  {BOLD}📷 {image_path}{RESET}")
    print(f"    🐱 Кот:    {cat_p*100:5.1f}%")
    print(f"    🐶 Собака: {dog_p*100:5.1f}%")
    if cat_p > dog_p:
        print(f"    {GREEN}→ Это КОТ! 🐱{RESET}")
    else:
        print(f"    {GREEN}→ Это СОБАКА! 🐶{RESET}")


def classifier_menu():
    while True:
        print(f"\n  {BOLD}🐱🐶 KamCat v1.0 — классификатор{RESET}")
        hr()
        print(f"  {ORANGE}1.{RESET} Классифицировать фото")
        print(f"  {ORANGE}0.{RESET} Назад")

        choice = ask("Твой выбор")
        if choice is None or choice == "0":
            return
        elif choice == "1":
            p = ask_path("Путь к фото (или q)")
            if p:
                classify_image(p)


# ============ YOLO ============

def load_yolo(model_name=None):
    from ultralytics import YOLO

    name = model_name or _state["yolo_model_name"]
    if _state["yolo_model"] is not None and _state["yolo_model_name"] == name:
        return _state["yolo_model"]

    print(f"  {GRAY}Загружаем {name}...{RESET}")
    _state["yolo_model"] = YOLO(f"{name}.pt")
    _state["yolo_model_name"] = name
    print(f"  {GREEN}✅ {name} загружена{RESET}")
    return _state["yolo_model"]


def detect_image(image_path):
    model = load_yolo()
    print(f"\n  {BOLD}📷 {image_path}{RESET}")

    results = model(str(image_path), conf=0.25, iou=0.5, verbose=False)
    r = results[0]

    TARGET = {15: "🐱 Кот", 16: "🐶 Собака"}
    found = [(TARGET[int(b.cls[0])], float(b.conf[0]))
             for b in r.boxes if int(b.cls[0]) in TARGET]
    found.sort(key=lambda x: x[1], reverse=True)

    out_path = Path(image_path).stem + "_detected.jpg"
    r.save(filename=out_path)

    if found:
        print(f"    Найдено: {len(found)}")
        for name, conf in found:
            print(f"      {name} — {conf*100:.1f}%")
    else:
        print(f"    {GRAY}Животных не найдено{RESET}")

    print(f"    💾 {out_path}")


def yolo_menu():
    while True:
        print(f"\n  {BOLD}🎯 KamCat YOLO v2 — детектор{RESET}")
        print(f"  {GRAY}Текущая модель: {_state['yolo_model_name']}{RESET}")
        hr()
        print(f"  {ORANGE}1.{RESET} Детекция на фото")
        print(f"  {ORANGE}2.{RESET} Сменить модель (nano/small/medium/x)")
        print(f"  {ORANGE}0.{RESET} Назад")

        choice = ask("Твой выбор")
        if choice is None or choice == "0":
            return
        elif choice == "1":
            p = ask_path("Путь к фото (или q)")
            if p:
                detect_image(p)
        elif choice == "2":
            print(f"\n  {BOLD}Выбери модель YOLO:{RESET}")
            print(f"  {ORANGE}1.{RESET} nano   (6 МБ)")
            print(f"  {ORANGE}2.{RESET} small  (22 МБ)")
            print(f"  {ORANGE}3.{RESET} medium (50 МБ)")
            print(f"  {ORANGE}4.{RESET} x      (130 МБ)")
            m = ask("Модель")
            mapping = {"1": "yolov8n", "2": "yolov8s", "3": "yolov8m", "4": "yolov8x"}
            if m in mapping:
                _state["yolo_model"] = None
                _state["yolo_model_name"] = mapping[m]
                print(f"  {GREEN}✅ Модель: {mapping[m]}{RESET}")


# ============ ГЕНЕРАТОР КАРТИНОК ============

def load_sd():
    if _state["sd_pipe"] is not None:
        return _state["sd_pipe"]

    import torch
    from diffusers import StableDiffusionPipeline

    print(f"  {GRAY}Загружаем SD 1.5...{RESET}")
    device = get_device()
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if str(device) == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)
    if str(device) == "cuda":
        pipe.enable_attention_slicing()

    _state["sd_pipe"] = pipe
    print(f"  {GREEN}✅ SD 1.5 загружена{RESET}")
    return pipe


def generate_image(prompt, negative="blurry, low quality, distorted",
                   steps=25, guidance=7.5, seed=None):
    import torch
    from datetime import datetime

    pipe = load_sd()
    device = get_device()

    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"\n  🎨 Промпт: {prompt}")
    print(f"  {GRAY}   steps={steps}, guidance={guidance}, seed={seed}{RESET}")
    print(f"  {GRAY}   Генерация...{RESET}")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"generated_{ts}_seed{seed}.png"
    image.save(out)
    print(f"  {GREEN}✅ Сохранено: {out}{RESET}")


def gen_menu():
    steps, guidance = 25, 7.5
    while True:
        print(f"\n  {BOLD}🎨 KamCat Gen v1.0 — SD 1.5{RESET}")
        print(f"  {GRAY}steps={steps}, guidance={guidance}{RESET}")
        hr()
        print(f"  {ORANGE}1.{RESET} Сгенерировать по промпту")
        print(f"  {ORANGE}2.{RESET} Настройки (steps / guidance)")
        print(f"  {ORANGE}0.{RESET} Назад")

        choice = ask("Твой выбор")
        if choice is None or choice == "0":
            return
        elif choice == "1":
            prompt = ask("Промпт (или q)")
            if prompt and prompt.lower() != "q":
                generate_image(prompt, steps=steps, guidance=guidance)
        elif choice == "2":
            s = ask(f"steps [{steps}]", default=str(steps))
            g = ask(f"guidance [{guidance}]", default=str(guidance))
            if s and s.isdigit():
                steps = int(s)
            if g:
                try:
                    guidance = float(g)
                except ValueError:
                    pass
            print(f"  {GREEN}✅ steps={steps}, guidance={guidance}{RESET}")


# ============ ГЕНЕРАТОР ТЕКСТА (KamCat-Micro) ============

def load_text_model():
    if _state["text_model"] is not None:
        return _state["text_model"]

    import torch

    micro_dir = Path("kamcat_micro")
    if not micro_dir.exists():
        print(f"  {RED}❌ Папка kamcat_micro не найдена!{RESET}")
        return None

    sys.path.insert(0, str(micro_dir.resolve()))

    try:
        from model import KamCatMicro
        from tokenizer import BPETokenizer

        print(f"  {GRAY}Загружаем KamCat-Micro...{RESET}")
        device = get_device()
        tok = BPETokenizer.load(micro_dir / "data" / "vocab.json")
        vocab_size = len(tok.token2id)

        ckpt = torch.load(micro_dir / "kamcat_micro.pth", map_location=device)
        config = ckpt["config"]
        model = KamCatMicro(vocab_size=vocab_size, **config).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        _state["text_model"] = {
            "model": model, "tok": tok, "device": device,
            "vocab_size": vocab_size, "loss": ckpt.get("loss", 0),
        }
        print(f"  {GREEN}✅ KamCat-Micro загружена (loss {ckpt.get('loss', 0):.2f}){RESET}")
        return _state["text_model"]
    except Exception as e:
        print(f"  {RED}❌ Ошибка загрузки: {e}{RESET}")
        return None


def generate_text(prompt, max_tokens=50, temperature=0.8, top_k=50):
    import torch

    state = load_text_model()
    if state is None:
        return

    model = state["model"]
    tok = state["tok"]
    device = state["device"]

    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_tokens,
                             temperature=temperature, top_k=top_k)

    text = tok.decode(out[0].tolist())
    print(f"\n  {BOLD}📝 {prompt}{RESET}")
    print(f"  {CYAN}{text}{RESET}")


def text_gen_menu():
    while True:
        print(f"\n  {BOLD}📝 KamCat-Micro — генератор текста{RESET}")
        print(f"  {GRAY}Своя модель, обученная с нуля (loss 1.58){RESET}")
        hr()
        print(f"  {ORANGE}1.{RESET} Сгенерировать по промпту")
        print(f"  {ORANGE}2.{RESET} Готовые промпты")
        print(f"  {ORANGE}0.{RESET} Назад")

        choice = ask("Твой выбор")
        if choice is None or choice == "0":
            return
        elif choice == "1":
            prompt = ask("Промпт (или q)")
            if prompt and prompt.lower() != "q":
                generate_text(prompt)
        elif choice == "2":
            print(f"\n  {BOLD}Готовые промпты:{RESET}")
            prompts = ["Привет, я", "Коты — это", "Собаки любят", "KamCat — это",
                       "Нейросеть", "Сегодня", "Мой кот", "Я создал"]
            for i, p in enumerate(prompts, 1):
                print(f"  {ORANGE}{i}.{RESET} {p}")
            num = ask("Номер промпта (или q)")
            if num and num.isdigit() and 1 <= int(num) <= len(prompts):
                generate_text(prompts[int(num) - 1])


# ============ ГЛАВНОЕ МЕНЮ ============

def main_menu():
    while True:
        print_header()
        print(f"\n  {BOLD}Выбери модель:{RESET}")
        print(f"  {ORANGE}1.{RESET} 🐱🐶 KamCat v1.0   — классификатор")
        print(f"  {ORANGE}2.{RESET} 🎯 KamCat YOLO v2  — детектор (рамки)")
        print(f"  {ORANGE}3.{RESET} 🎨 KamCat Gen v1.0 — генератор (SD 1.5)")
        print(f"  {ORANGE}4.{RESET} 📝 KamCat-Micro    — генератор текста")
        print(f"  {ORANGE}0.{RESET} ❌ Выход")

        choice = ask("Твой выбор")
        if choice is None or choice == "0":
            print(f"\n  {GREEN}До встречи! 🐱🐶{RESET}\n")
            return
        elif choice == "1":
            classifier_menu()
        elif choice == "2":
            yolo_menu()
        elif choice == "3":
            gen_menu()
        elif choice == "4":
            text_gen_menu()
        else:
            print(f"  {RED}⚠️ Неверный выбор{RESET}")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print(f"\n\n  {GRAY}Прервано. До встречи!{RESET}\n")