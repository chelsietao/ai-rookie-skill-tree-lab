"""A deterministic word tokenizer for fully local tiny Transformers."""

from __future__ import annotations

from collections import Counter

import torch

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


class LocalTokenizer:
    def __init__(self, texts: list[str], max_length: int = 16):
        words = Counter(word.lower().strip(";,.?!") for text in texts for word in text.split())
        self.tokens = SPECIAL + sorted(words)
        self.vocab = {token: index for index, token in enumerate(self.tokens)}
        self.max_length = max_length

    def encode(self, text: str) -> list[int]:
        words = [word.lower().strip(";,.?!") for word in text.split()]
        values = (
            [self.vocab["[CLS]"]] + [self.vocab.get(word, 1) for word in words] + [self.vocab["[SEP]"]]
        )
        return (values[: self.max_length] + [0] * self.max_length)[: self.max_length]

    def batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor([self.encode(text) for text in texts], dtype=torch.long)
        return ids, (ids != 0).long()

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.tokens[index] for index in ids if index < len(self.tokens) and index > 4)
