"""Train TF-IDF, Tiny BERT pre-training/fine-tuning, and Tiny GPT."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
)

from rookie_lab.nlp.tokenizer import LocalTokenizer


def _baselines(frame: pd.DataFrame, seed: int) -> dict[str, object]:
    train, valid = train_test_split(
        frame, test_size=0.25, stratify=frame["sentiment"], random_state=seed
    )
    sentiment = Pipeline(
        [("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("model", LogisticRegression(max_iter=300))]
    ).fit(train["text"], train["sentiment"])
    probability = sentiment.predict_proba(valid["text"])[:, 1]
    intent = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(max_iter=300))]).fit(
        train["text"], train["intent"]
    )
    return {
        "sentiment_roc_auc": float(roc_auc_score(valid["sentiment"], probability)),
        "intent_accuracy": float(accuracy_score(valid["intent"], intent.predict(valid["text"]))),
        "features": "TF-IDF word unigrams and bigrams",
    }


def _mask(ids: torch.Tensor, tokenizer: LocalTokenizer, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    eligible = (ids > 4) & (torch.rand(ids.shape, generator=generator) < 0.15)
    labels = ids.clone()
    labels[~eligible] = -100
    masked = ids.clone()
    masked[eligible] = tokenizer.vocab["[MASK]"]
    return masked, labels


def _train_classifier(
    model: BertForSequenceClassification,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    train_y: torch.Tensor,
    valid_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    valid_y: torch.Tensor,
    epochs: int,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(train_ids, train_mask, train_y), batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for ids, mask, labels in loader:
            optimizer.zero_grad()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probability = (
            model(input_ids=valid_ids, attention_mask=valid_mask).logits.softmax(1)[:, 1].numpy()
        )
    return {
        "accuracy": float(accuracy_score(valid_y.numpy(), probability >= 0.5)),
        "roc_auc": float(roc_auc_score(valid_y.numpy(), probability)),
    }


def _bert(frame: pd.DataFrame, profile: str, tokenizer: LocalTokenizer, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    ids, attention = tokenizer.batch(frame["text"].tolist())
    labels = torch.from_numpy(frame["sentiment"].to_numpy("int64", copy=True))
    indices = np.arange(len(frame))
    train_idx, valid_idx = train_test_split(indices, test_size=0.25, stratify=labels, random_state=seed)
    config = BertConfig(
        vocab_size=len(tokenizer.tokens),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        num_labels=2,
    )
    pretrainer = BertForMaskedLM(config)
    masked, mlm_labels = _mask(ids[train_idx], tokenizer, seed)
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=0.003)
    pretraining_loss: list[float] = []
    for _ in range(1 if profile == "ci" else 3):
        optimizer.zero_grad()
        loss = pretrainer(input_ids=masked, attention_mask=attention[train_idx], labels=mlm_labels).loss
        loss.backward()
        optimizer.step()
        pretraining_loss.append(float(loss.detach()))

    epochs = 2 if profile == "ci" else 5
    scratch = BertForSequenceClassification(config)
    scratch_metrics = _train_classifier(
        scratch,
        ids[train_idx],
        attention[train_idx],
        labels[train_idx],
        ids[valid_idx],
        attention[valid_idx],
        labels[valid_idx],
        epochs,
    )
    transferred = BertForSequenceClassification(config)
    transferred.bert.load_state_dict(pretrainer.bert.state_dict(), strict=False)
    transferred_metrics = _train_classifier(
        transferred,
        ids[train_idx],
        attention[train_idx],
        labels[train_idx],
        ids[valid_idx],
        attention[valid_idx],
        labels[valid_idx],
        epochs,
    )
    return {
        "architecture": "Hugging Face BERT with one encoder block and four attention heads",
        "masked_language_pretraining_loss": pretraining_loss,
        "from_scratch": scratch_metrics,
        "pretrained_then_finetuned": transferred_metrics,
    }


def _gpt(frame: pd.DataFrame, profile: str, tokenizer: LocalTokenizer, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    ids, attention = tokenizer.batch(frame["text"].tolist())
    config = GPT2Config(
        vocab_size=len(tokenizer.tokens),
        n_positions=32,
        n_embd=32,
        n_layer=1,
        n_head=4,
        bos_token_id=tokenizer.vocab["[CLS]"],
        eos_token_id=tokenizer.vocab["[SEP]"],
        pad_token_id=tokenizer.vocab["[PAD]"],
    )
    model = GPT2LMHeadModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(ids, attention), batch_size=64, shuffle=True)
    losses: list[float] = []
    for _ in range(1 if profile == "ci" else 4):
        for token_ids, mask in loader:
            optimizer.zero_grad()
            labels = token_ids.clone()
            labels[mask == 0] = -100
            loss = model(input_ids=token_ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    prompt = torch.tensor([[tokenizer.vocab["[CLS]"], tokenizer.vocab.get("helpful", 1)]])
    model.eval()
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=6, do_sample=False, pad_token_id=0)[0].tolist()
    return {
        "architecture": "Hugging Face GPT-2 style decoder with causal multi-head attention",
        "final_loss": losses[-1],
        "perplexity": float(math.exp(min(losses[-1], 10))),
        "greedy_generation": tokenizer.decode(generated),
    }


def run(data_dir: Path, profile: str, seed: int = 20260820) -> dict[str, object]:
    torch.set_num_threads(2)
    frame = pd.read_csv(data_dir / "support_texts.csv.gz")
    tokenizer = LocalTokenizer(frame["text"].tolist())
    return {
        "tasks": ["sentiment", "intent/NLU", "masked-language pre-training", "text generation/NLG"],
        "tfidf": _baselines(frame, seed),
        "bert": _bert(frame, profile, tokenizer, seed),
        "gpt": _gpt(frame, profile, tokenizer, seed),
        "vocabulary_size": len(tokenizer.tokens),
        "offline": True,
    }
