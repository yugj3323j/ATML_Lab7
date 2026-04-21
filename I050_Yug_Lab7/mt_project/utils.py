import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader, Dataset

from model import (
    AttentionDecoder,
    AttentionSeq2Seq,
    BahdanauAttention,
    Encoder,
)


SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

_DEVANAGARI_DETECT_RE = re.compile(r"[\u0900-\u097F]")
_DEVANAGARI_TOKEN_RE = re.compile(r"[\u0900-\u097F]+|[A-Za-z0-9_]+|[^\s]", flags=re.UNICODE)


def tokenize_text(text: str) -> List[str]:
    text = str(text).strip()
    if not text:
        return []

    # If we see any Devanagari codepoints, use a Unicode-property-aware tokenizer.
    # This keeps words like "नमस्ते" intact (instead of splitting matras/virama),
    # making training + decoding much more readable and learnable.
    if _DEVANAGARI_DETECT_RE.search(text):
        raw_tokens = _DEVANAGARI_TOKEN_RE.findall(text)
        # Split common punctuation that lives inside the Devanagari unicode block (e.g. danda).
        punctuation = {"।", "॥", "?", "!", ",", ".", ":", ";", "(", ")", "[", "]", "{", "}", "\"", "'"}
        tokens: List[str] = []
        for tok in raw_tokens:
            if not tok:
                continue
            if len(tok) > 1 and tok[-1] in punctuation:
                base = tok[:-1]
                if base:
                    tokens.append(base)
                tokens.append(tok[-1])
                continue
            tokens.append(tok)
        return tokens

    text = text.lower()
    return re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)


class Vocabulary:
    def __init__(self, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}

    def build(self, tokenized_texts: Sequence[Sequence[str]]) -> None:
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        self.token_to_idx = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.token_to_idx)

        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def numericalize(self, tokens: Sequence[str]) -> List[int]:
        unk_idx = self.token_to_idx["<unk>"]
        return [self.token_to_idx.get(token, unk_idx) for token in tokens]

    def denumericalize(self, indices: Sequence[int]) -> List[str]:
        return [self.idx_to_token.get(index, "<unk>") for index in indices]

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def to_dict(self) -> Dict[str, object]:
        return {
            "min_freq": self.min_freq,
            "token_to_idx": self.token_to_idx,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Vocabulary":
        vocab = cls(min_freq=int(data["min_freq"]))
        vocab.token_to_idx = {str(k): int(v) for k, v in dict(data["token_to_idx"]).items()}
        vocab.idx_to_token = {idx: token for token, idx in vocab.token_to_idx.items()}
        return vocab


def load_parallel_data(data_path: str, max_rows: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    extension = os.path.splitext(data_path)[1].lower()
    if extension == ".tsv":
        df = pd.read_csv(data_path, sep="\t", nrows=max_rows)
    else:
        df = pd.read_csv(data_path, nrows=max_rows)

    normalized_columns = {col.lower().strip(): col for col in df.columns}
    english_candidates = ["english", "en", "source", "src", "input"]
    hindi_candidates = ["hindi", "hi", "target", "tgt", "output"]

    english_column = next((normalized_columns[col] for col in english_candidates if col in normalized_columns), None)
    hindi_column = next((normalized_columns[col] for col in hindi_candidates if col in normalized_columns), None)

    if english_column is None or hindi_column is None:
        if df.shape[1] >= 2:
            english_column, hindi_column = df.columns[:2]
        else:
            raise ValueError(
                "Dataset must contain English and Hindi columns. Supported names include "
                "english/hindi, en/hi, source/target, src/tgt."
            )

    df = df[[english_column, hindi_column]].dropna().drop_duplicates()
    df.columns = ["english", "hindi"]
    df["english"] = df["english"].astype(str).str.strip()
    df["hindi"] = df["hindi"].astype(str).str.strip()
    df = df[(df["english"] != "") & (df["hindi"] != "")]
    df = df.reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def find_parallel_dataset_file(root_path: str) -> str:
    supported_extensions = {".csv", ".tsv"}
    candidate_files = []

    for current_root, _, files in os.walk(root_path):
        for file_name in files:
            extension = os.path.splitext(file_name)[1].lower()
            if extension in supported_extensions:
                candidate_files.append(os.path.join(current_root, file_name))

    if not candidate_files:
        raise FileNotFoundError(
            f"No CSV or TSV dataset file found inside: {root_path}"
        )

    priority_names = [
        "dataset_english_hindi.csv",
        "english_hindi.csv",
        "english-hindi.csv",
    ]
    lowered_map = {os.path.basename(path).lower(): path for path in candidate_files}
    for name in priority_names:
        if name in lowered_map:
            return lowered_map[name]

    candidate_files.sort()
    return candidate_files[0]


def train_test_split_df(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_index = int(len(shuffled) * (1 - test_size))
    train_df = shuffled.iloc[:split_index].reset_index(drop=True)
    test_df = shuffled.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df


def prepare_sequences(df: pd.DataFrame) -> Tuple[List[List[str]], List[List[str]]]:
    english_tokens = [tokenize_text(text) for text in df["english"].tolist()]
    hindi_tokens = [tokenize_text(text) for text in df["hindi"].tolist()]
    return english_tokens, hindi_tokens


@dataclass
class Sample:
    src_ids: List[int]
    trg_ids: List[int]


class TranslationDataset(Dataset):
    def __init__(
        self,
        english_tokens: Sequence[Sequence[str]],
        hindi_tokens: Sequence[Sequence[str]],
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
    ) -> None:
        self.samples: List[Sample] = []
        sos_idx = trg_vocab.token_to_idx["<sos>"]
        eos_idx = trg_vocab.token_to_idx["<eos>"]
        src_eos_idx = src_vocab.token_to_idx["<eos>"]
        src_sos_idx = src_vocab.token_to_idx["<sos>"]

        for src_tokens, trg_tokens in zip(english_tokens, hindi_tokens):
            src_ids = [src_sos_idx] + src_vocab.numericalize(src_tokens) + [src_eos_idx]
            trg_ids = [sos_idx] + trg_vocab.numericalize(trg_tokens) + [eos_idx]
            self.samples.append(Sample(src_ids=src_ids, trg_ids=trg_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def collate_batch(
    batch: Sequence[Sample],
    src_pad_idx: int,
    trg_pad_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_src_len = max(len(sample.src_ids) for sample in batch)
    max_trg_len = max(len(sample.trg_ids) for sample in batch)

    padded_src = []
    padded_trg = []

    for sample in batch:
        src = sample.src_ids + [src_pad_idx] * (max_src_len - len(sample.src_ids))
        trg = sample.trg_ids + [trg_pad_idx] * (max_trg_len - len(sample.trg_ids))
        padded_src.append(src)
        padded_trg.append(trg)

    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_trg, dtype=torch.long)


def build_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
    min_freq: int = 1,
) -> Tuple[DataLoader, DataLoader, Vocabulary, Vocabulary]:
    train_src_tokens, train_trg_tokens = prepare_sequences(train_df)
    test_src_tokens, test_trg_tokens = prepare_sequences(test_df)

    src_vocab = Vocabulary(min_freq=min_freq)
    trg_vocab = Vocabulary(min_freq=min_freq)
    src_vocab.build(train_src_tokens)
    trg_vocab.build(train_trg_tokens)

    train_dataset = TranslationDataset(train_src_tokens, train_trg_tokens, src_vocab, trg_vocab)
    test_dataset = TranslationDataset(test_src_tokens, test_trg_tokens, src_vocab, trg_vocab)

    src_pad_idx = src_vocab.token_to_idx["<pad>"]
    trg_pad_idx = trg_vocab.token_to_idx["<pad>"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, src_pad_idx, trg_pad_idx),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, src_pad_idx, trg_pad_idx),
    )

    return train_loader, test_loader, src_vocab, trg_vocab


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    elapsed = end_time - start_time
    minutes = int(elapsed / 60)
    seconds = int(elapsed - (minutes * 60))
    return minutes, seconds


def decode_indices(indices: Sequence[int], vocab: Vocabulary) -> List[str]:
    tokens = []
    for index in indices:
        token = vocab.idx_to_token.get(int(index), "<unk>")
        if token == "<eos>":
            break
        if token not in {"<sos>", "<pad>"}:
            tokens.append(token)
    return tokens


def ids_from_sentence(sentence: str, vocab: Vocabulary, device: torch.device) -> torch.Tensor:
    tokens = tokenize_text(sentence)
    numericalized = (
        [vocab.token_to_idx["<sos>"]]
        + vocab.numericalize(tokens)
        + [vocab.token_to_idx["<eos>"]]
    )
    return torch.tensor(numericalized, dtype=torch.long, device=device).unsqueeze(0)


def calculate_bleu(
    model,
    data_loader: DataLoader,
    src_vocab: Vocabulary,
    trg_vocab: Vocabulary,
    device: torch.device,
) -> float:
    references = []
    hypotheses = []
    model.eval()

    with torch.no_grad():
        for src, trg in data_loader:
            src = src.to(device)
            trg = trg.to(device)
            predictions, _ = model.greedy_decode(
                src,
                trg_vocab.token_to_idx["<sos>"],
                trg_vocab.token_to_idx["<eos>"],
                max_len=trg.size(1) + 10,
            )

            for pred_ids, trg_ids in zip(predictions.cpu().tolist(), trg.cpu().tolist()):
                predicted_tokens = decode_indices(pred_ids, trg_vocab)
                target_tokens = decode_indices(trg_ids, trg_vocab)
                if predicted_tokens:
                    hypotheses.append(predicted_tokens)
                else:
                    hypotheses.append(["<unk>"])
                references.append([target_tokens])

    smoothing = SmoothingFunction().method1
    return float(corpus_bleu(references, hypotheses, smoothing_function=smoothing))


def save_training_plot(loss_history: Dict[str, List[float]], output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    for model_name, losses in loss_history.items():
        plt.plot(range(1, len(losses) + 1), losses, marker="o", label=model_name)
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metrics(metrics: Dict[str, object], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def load_metrics(metrics_path: str) -> Dict[str, object]:
    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_vocabularies(src_vocab: Vocabulary, trg_vocab: Vocabulary, output_path: str) -> None:
    torch.save(
        {
            "src_vocab": src_vocab.to_dict(),
            "trg_vocab": trg_vocab.to_dict(),
        },
        output_path,
    )


def load_vocabularies(vocab_path: str) -> Tuple[Vocabulary, Vocabulary]:
    payload = torch.load(vocab_path, map_location="cpu")
    return (
        Vocabulary.from_dict(payload["src_vocab"]),
        Vocabulary.from_dict(payload["trg_vocab"]),
    )


def build_attention_model(
    src_vocab_size: int,
    trg_vocab_size: int,
    emb_dim: int,
    hidden_dim: int,
    src_pad_idx: int,
    trg_pad_idx: int,
    device: torch.device,
    num_layers: int = 1,
    dropout: float = 0.2,
):
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=src_pad_idx,
    )
    attention = BahdanauAttention(hidden_dim)
    decoder = AttentionDecoder(
        output_dim=trg_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        attention=attention,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=trg_pad_idx,
    )
    return AttentionSeq2Seq(encoder, decoder, src_pad_idx, device).to(device)


def load_model_for_inference(
    checkpoint_path: str,
    vocab_path: str,
    device: torch.device,
):
    src_vocab, trg_vocab = load_vocabularies(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = build_attention_model(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        emb_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        src_pad_idx=src_vocab.token_to_idx["<pad>"],
        trg_pad_idx=trg_vocab.token_to_idx["<pad>"],
        device=device,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, src_vocab, trg_vocab, checkpoint


def translate_sentence(
    sentence: str,
    models_dir: str = "saved_models",
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = os.path.join(models_dir, "vocab.pt")
    checkpoint_path = os.path.join(models_dir, "lstm_model.pt")
    if not os.path.exists(checkpoint_path):
        legacy_checkpoint = os.path.join(models_dir, "attention_model.pt")
        checkpoint_path = legacy_checkpoint

    model, src_vocab, trg_vocab, _ = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        device=device,
    )

    src_tensor = ids_from_sentence(sentence, src_vocab, device)
    with torch.no_grad():
        predicted_ids, attention = model.greedy_decode(
            src_tensor,
            trg_vocab.token_to_idx["<sos>"],
            trg_vocab.token_to_idx["<eos>"],
            max_len=50,
        )

    tokens = decode_indices(predicted_ids.squeeze(0).cpu().tolist(), trg_vocab)
    return {
        "tokens": tokens,
        "translation": " ".join(tokens),
        "attention": attention.squeeze(0).cpu().numpy().tolist()
        if attention is not None and attention.numel() > 0
        else None,
    }


_PRETRAINED_MT_CACHE: Dict[str, object] = {}


def translate_sentence_pretrained(
    sentence: str,
    model_name: str = "Helsinki-NLP/opus-mt-en-hi",
    device: Optional[torch.device] = None,
    num_beams: int = 4,
    max_new_tokens: int = 64,
) -> Dict[str, object]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = f"{model_name}:{device.type}"

    if cache_key not in _PRETRAINED_MT_CACHE:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _PRETRAINED_MT_CACHE[cache_key] = {"tokenizer": tokenizer, "model": model}

    bundle = _PRETRAINED_MT_CACHE[cache_key]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return {"translation": text, "model": model_name}


def prepare_sample_translations(
    model,
    samples_df: pd.DataFrame,
    src_vocab: Vocabulary,
    trg_vocab: Vocabulary,
    device: torch.device,
    limit: int = 5,
) -> List[Dict[str, str]]:
    collected = []
    model.eval()
    for _, row in samples_df.head(limit).iterrows():
        src_tensor = ids_from_sentence(row["english"], src_vocab, device)
        with torch.no_grad():
            predicted_ids, _ = model.greedy_decode(
                src_tensor,
                trg_vocab.token_to_idx["<sos>"],
                trg_vocab.token_to_idx["<eos>"],
                max_len=50,
            )
        predicted_tokens = decode_indices(predicted_ids.squeeze(0).cpu().tolist(), trg_vocab)
        collected.append(
            {
                "english": row["english"],
                "reference": row["hindi"],
                "prediction": " ".join(predicted_tokens),
            }
        )
    return collected


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)
