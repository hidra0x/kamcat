"""
Обучение KamCat-Micro (v5).
Vocab=2000, ~800K токенов, 1 эпоха (~1 час).
"""

import time
import math
import torch
from model import KamCatMicro
from dataset import make_dataloader
from tokenizer import BPETokenizer

# ============ ГИПЕРПАРАМЕТРЫ ============
SEQ_LEN = 128
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EPOCHS = 2          
WARMUP_STEPS = 500
LOG_EVERY = 1000
SAVE_EVERY = 5000
FAST_TEST = False

MODEL_CONFIG = {
    "d_model": 64,
    "n_heads": 2,
    "n_layers": 1,
    "d_ff": 256,
    "max_seq_len": SEQ_LEN,
}
# =======================================


def get_lr(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}с"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}ч {m}м"
    return f"{m}м {s}с"


def save_checkpoint(model, vocab_size, step, loss, path="kamcat_micro.pth"):
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "config": MODEL_CONFIG,
        "step": step,
        "loss": loss,
    }, path)


def train():
    print("=" * 60)
    print("🚀 Обучение KamCat-Micro (v5) — vocab=2000")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    loader = make_dataloader(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    vocab_size = loader.dataset.vocab_size
    print(f"Vocab: {vocab_size}")
    print(f"Токенов: {len(loader.dataset.tokens):,}")

    model = KamCatMicro(vocab_size=vocab_size, **MODEL_CONFIG).to(device)
    print(f"Параметров: {model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    steps_per_epoch = len(loader)
    if FAST_TEST:
        steps_per_epoch = min(steps_per_epoch, 100)
        epochs_to_run = 1
    else:
        epochs_to_run = EPOCHS

    total_steps = steps_per_epoch * epochs_to_run
    print(f"Шагов: {total_steps:,} ({steps_per_epoch:,} × {epochs_to_run})")
    print(f"Ожидаемое время: ~{format_time(total_steps * 0.02)}")
    print("=" * 60)

    model.train()
    global_step = 0
    start_time = time.time()
    best_loss = float("inf")
    last_avg = float("inf")

    for epoch in range(epochs_to_run):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for x, y in loader:
            if FAST_TEST and epoch_steps >= steps_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            lr = LR * get_lr(global_step, WARMUP_STEPS, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg = epoch_loss / epoch_steps
                last_avg = avg
                elapsed = time.time() - start_time
                remaining = (elapsed / global_step) * (total_steps - global_step)
                print(f"  step {global_step:7d}/{total_steps:,} | "
                      f"loss {avg:.4f} | lr {lr:.2e} | "
                      f"прошло {format_time(elapsed)} | осталось ~{format_time(remaining)}")

            if global_step % SAVE_EVERY == 0:
                avg = epoch_loss / epoch_steps
                if avg < best_loss:
                    best_loss = avg
                save_checkpoint(model, vocab_size, global_step, avg)
                print(f"  💾 Чекпоинт на шаге {global_step} (loss {avg:.4f})")

        avg = epoch_loss / epoch_steps
        epoch_time = time.time() - epoch_start
        print(f"  📊 Эпоха {epoch+1}/{epochs_to_run} | loss {avg:.4f} | {format_time(epoch_time)}")
        save_checkpoint(model, vocab_size, global_step, avg)

    save_checkpoint(model, vocab_size, global_step, last_avg)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"✅ Обучение завершено за {format_time(elapsed)}")
    print(f"💾 Модель: kamcat_micro.pth")
    print("=" * 60)

    # Тест генерации
    print("\n🎨 Тест генерации:")
    tok = BPETokenizer.load("data/vocab.json")
    model.eval()

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
            out = model.generate(x, max_new_tokens=40, temperature=0.8, top_k=50)
        text = tok.decode(out[0].tolist())
        print(f"  > {prompt}")
        print(f"    {text}")
        print()


if __name__ == "__main__":
    train()