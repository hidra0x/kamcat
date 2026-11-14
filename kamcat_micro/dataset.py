"""
Датасет для обучения KamCat-Micro.
Загружает корпус, токенизирует, нарезает на последовательности.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizer import BPETokenizer


class TextDataset(Dataset):
    """Датасет: случайные окна по seq_len из корпуса."""

    def __init__(self, corpus_path, vocab_path, seq_len=128):
        self.seq_len = seq_len

        # Загружаем токенизатор
        print("Загружаем токенизатор...")
        self.tokenizer = BPETokenizer.load(vocab_path)
        self.vocab_size = len(self.tokenizer.token2id)
        print(f"  Vocab: {self.vocab_size}")

        # Токенизируем весь корпус
        print("Токенизируем корпус...")
        text = Path(corpus_path).read_text(encoding="utf-8")
        self.tokens = self.tokenizer.encode(text)
        print(f"  Токенов: {len(self.tokens):,}")

        # Сколько окон помещается
        self.n_windows = len(self.tokens) - seq_len
        print(f"  Окон по {seq_len}: {self.n_windows:,}")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        # Окно: input = tokens[idx:idx+seq_len]
        #        target = tokens[idx+1:idx+seq_len+1]
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def make_dataloader(corpus_path="data/corpus.txt",
                    vocab_path="data/vocab.json",
                    seq_len=128,
                    batch_size=16,
                    shuffle=True,
                    num_workers=0):
    """Создаёт DataLoader"""
    ds = TextDataset(corpus_path, vocab_path, seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


if __name__ == "__main__":
    print("=" * 50)
    print("📊 Тест датасета")
    print("=" * 50)

    loader = make_dataloader(seq_len=128, batch_size=4)

    x, y = next(iter(loader))
    print(f"\nБатч x: {x.shape}")
    print(f"Батч y: {y.shape}")
    print(f"x[0][:10]: {x[0][:10].tolist()}")
    print(f"y[0][:10]: {y[0][:10].tolist()}")

    # Проверка: y = x со сдвигом на 1
    shift_ok = (x[0][1:] == y[0][:-1]).all().item()
    print(f"y = x со сдвигом: {shift_ok}")

    print(f"\nБатчей в эпохе: {len(loader)}")
    print("✅ Датасет работает!")