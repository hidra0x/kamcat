"""
BPE-токенизатор с нуля (v4).
min_pair_count=200, max_token_length=8 — короткие токены.
"""

import json
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"


class BPETokenizer:
    def __init__(self, vocab_size=2000, min_pair_count=200, max_token_length=8):
        self.vocab_size = vocab_size
        self.min_pair_count = min_pair_count
        self.max_token_length = max_token_length
        self.token2id = {}
        self.id2token = {}
        self.merges = {}

    def _init_vocab(self, text):
        chars = sorted(set(text))
        tokens = [PAD, UNK, BOS, EOS] + chars
        self.token2id = {t: i for i, t in enumerate(tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    def _get_pairs(self, tokens):
        return Counter(zip(tokens, tokens[1:]))

    def _merge(self, tokens, pair, new_token):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _unique_token(self, base):
        token = base
        suffix = 0
        while token in self.token2id:
            suffix += 1
            token = f"{base}#{suffix}"
        return token

    def train(self, text):
        print(f"Обучаем BPE: vocab_size={self.vocab_size}, "
              f"min_pair={self.min_pair_count}, max_len={self.max_token_length}")
        self._init_vocab(text)

        tokens = list(text)
        print(f"  Начальных токенов: {len(self.token2id)}")

        skipped = 0
        while len(self.token2id) < self.vocab_size:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break

            # Ищем лучшую пару, которая НЕ превышает max_token_length
            best_pair = None
            best_count = 0
            for pair, count in pairs.most_common():
                if count < self.min_pair_count:
                    break  # дальше только реже
                combined = "".join(pair)
                if len(combined) > self.max_token_length:
                    skipped += 1
                    continue  # пропускаем слишком длинные
                best_pair = pair
                best_count = count
                break

            if best_pair is None:
                print(f"  Остановка: нет пар с count>={self.min_pair_count} "
                      f"и длиной<={self.max_token_length}")
                break

            base = "".join(best_pair)
            new_token = self._unique_token(base)

            self.merges[best_pair] = new_token
            self.token2id[new_token] = len(self.token2id)
            self.id2token[len(self.token2id) - 1] = new_token
            tokens = self._merge(tokens, best_pair, new_token)

            if len(self.token2id) % 200 == 0:
                print(f"  Токенов: {len(self.token2id)}, пара: {best_pair} (×{best_count})")

        print(f"  Итог: {len(self.token2id)} токенов")
        if skipped:
            print(f"  Пропущено длинных пар: {skipped}")
        return self

    def encode(self, text):
        tokens = list(text)
        for pair, new_token in self.merges.items():
            tokens = self._merge(tokens, pair, new_token)
        return [self.token2id.get(t, self.token2id[UNK]) for t in tokens]

    def decode(self, ids):
        tokens = [self.id2token.get(i, UNK) for i in ids]
        tokens = [t.split("#")[0] for t in tokens]
        return "".join(tokens)

    def save(self, path="data/vocab.json"):
        Path(path).parent.mkdir(exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "token2id": self.token2id,
            "merges": {f"{k[0]}||{k[1]}": v for k, v in self.merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Словарь сохранён: {path}")

    @classmethod
    def load(cls, path="data/vocab.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.token2id = data["token2id"]
        tok.id2token = {i: t for t, i in tok.token2id.items()}
        tok.merges = {tuple(k.split("||")): v for k, v in data["merges"].items()}
        return tok


def _ensure_chars(corpus):
    """Добавляет все нужные символы в корпус."""
    extra = "!?—–-:;\"'()[]0123456789ЭЮЯЁэюяё"
    missing = [c for c in extra if c not in corpus]
    if missing:
        print(f"  ⚠️ Добавляем символы: {''.join(missing)}")
        corpus = corpus + "\n" + " ".join(missing) + "\n"
    return corpus


if __name__ == "__main__":
    corpus_path = Path("data/corpus.txt")
    corpus = corpus_path.read_text(encoding="utf-8")
    corpus = _ensure_chars(corpus)

    print(f"Корпус: {len(corpus)} символов")

    # ВАЖНО: min_pair_count=200, max_token_length=8
    tok = BPETokenizer(vocab_size=2000, min_pair_count=200, max_token_length=8)
    tok.train(corpus)
    tok.save()

    print("\n=== Тест encode/decode ===")
    tests = [
        "Привет, KamCat! Я создал тебя с нуля.",
        "Кот спит? Да!",
        "Это - тест.",
        "Собака: друг человека.",
        "Я создал KamCat с нуля, чтобы понять, как работают нейросети.",
        "Нейросеть — это математика, а не магия.",
    ]
    all_ok = True
    for test in tests:
        ids = tok.encode(test)
        decoded = tok.decode(ids)
        ok = decoded == test
        all_ok = all_ok and ok
        print(f"{'✅' if ok else '⚠️'} {test}")
        print(f"   Токенов: {len(ids)} → {ids[:10]}{'...' if len(ids) > 10 else ''}")
        if not ok:
            print(f"   Декод:   {decoded}")

    # Проверка на 100 строках
    lines = [l for l in corpus.split("\n") if l.strip()][:100]
    ok_count = sum(1 for line in lines if tok.decode(tok.encode(line)) == line)
    print(f"\nВосстановлено без потерь: {ok_count}/{len(lines)} строк")

    # Средняя длина токена
    total_tokens = sum(len(tok.encode(l)) for l in lines)
    total_chars = sum(len(l) for l in lines)
    avg_len = total_chars / total_tokens if total_tokens else 0
    print(f"Средняя длина токена: {avg_len:.2f} символа")

    # Проверка длины токенов
    long_tokens = [t for t in tok.token2id if len(t) > 8]
    if long_tokens:
        print(f"⚠️ Токены длиннее 8: {len(long_tokens)} (макс: {max(len(t) for t in long_tokens)})")
    else:
        print(f"✅ Все токены <= 8 символов!")

    print(f"{'✅ Всё ок!' if all_ok and ok_count == len(lines) else '⚠️ Есть проблемы'}")