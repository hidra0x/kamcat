"""
Генерация текста из обученной KamCat-Micro.
"""

import torch
from model import KamCatMicro
from tokenizer import BPETokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # Загружаем токенизатор
    tok = BPETokenizer.load("data/vocab.json")
    vocab_size = len(tok.token2id)
    print(f"Vocab: {vocab_size}")

    # Загружаем модель
    ckpt = torch.load("kamcat_micro.pth", map_location=device)
    config = ckpt["config"]
    model = KamCatMicro(vocab_size=vocab_size, **config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loss_val = ckpt.get("loss", 0)
    print(f"Модель загружена (loss {loss_val:.4f})")
    print("=" * 60)

    # Тест генерации
    prompts = [
        "Привет, я",
        "Коты — это",
        "Собаки любят",
        "KamCat — это",
        "Нейросеть",
        "Сегодня",
        "Мой кот",
        "Я создал",
    ]

    for prompt in prompts:
        ids = tok.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=50)

        text = tok.decode(out[0].tolist())
        print(f"\n> {prompt}")
        print(f"  {text}")

    print("\n" + "=" * 60)
    print("Готово!")

if __name__ == "__main__":
    main()