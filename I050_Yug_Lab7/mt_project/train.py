import argparse
import os
import random
import time
from typing import Dict, List, Tuple

import kagglehub
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    build_attention_model,
    build_dataloaders,
    calculate_bleu,
    ensure_directory,
    epoch_time,
    find_parallel_dataset_file,
    load_parallel_data,
    prepare_sample_translations,
    save_metrics,
    save_training_plot,
    save_vocabularies,
    train_test_split_df,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    clip: float,
    device: torch.device,
    teacher_forcing_ratio: float,
) -> float:
    model.train()
    epoch_loss = 0.0

    for src, trg in data_loader:
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output, _ = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(len(data_loader), 1)


def evaluate_epoch(
    model,
    data_loader,
    criterion,
    device: torch.device,
) -> float:
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, trg in data_loader:
            src = src.to(device)
            trg = trg.to(device)

            output, _ = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg_flat)
            epoch_loss += loss.item()

    return epoch_loss / max(len(data_loader), 1)


def save_checkpoint(
    model,
    path: str,
    config: Dict[str, float],
    train_losses: List[float],
    val_losses: List[float],
    bleu_score: float,
    completed_epochs: int,
) -> None:
    torch.save(
        {
            "model_name": "lstm_attention",
            "model_state_dict": model.state_dict(),
            "config": config,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "bleu_score": bleu_score,
            "completed_epochs": completed_epochs,
        },
        path,
    )


def load_checkpoint_if_available(
    model,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[int, List[float], List[float]]:
    if not os.path.exists(checkpoint_path):
        return 0, [], []

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_losses = list(checkpoint.get("train_losses", []))
    val_losses = list(checkpoint.get("val_losses", []))
    completed_epochs = int(checkpoint.get("completed_epochs", len(train_losses)))
    print(f"Resuming from checkpoint: {checkpoint_path}")
    print(f"Completed epochs found: {completed_epochs}")
    return completed_epochs, train_losses, val_losses


def train_and_evaluate(
    data_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    hidden_dim: int,
    learning_rate: float,
    teacher_forcing_ratio: float,
    dropout: float,
    num_layers: int,
    test_size: float,
    seed: int,
    min_freq: int,
    clip: float,
    max_rows: int,
    resume: bool,
) -> Dict[str, object]:
    set_seed(seed)
    ensure_directory(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_parallel_data(data_path, max_rows=max_rows, seed=seed)
    train_df, test_df = train_test_split_df(df, test_size=test_size, random_state=seed)
    train_loader, test_loader, src_vocab, trg_vocab = build_dataloaders(
        train_df,
        test_df,
        batch_size=batch_size,
        min_freq=min_freq,
    )

    src_pad_idx = src_vocab.token_to_idx["<pad>"]
    trg_pad_idx = trg_vocab.token_to_idx["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "teacher_forcing_ratio": teacher_forcing_ratio,
        "dropout": dropout,
        "num_layers": num_layers,
    }

    model = build_attention_model(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        emb_dim=embedding_dim,
        hidden_dim=hidden_dim,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        device=device,
        num_layers=num_layers,
        dropout=dropout,
    )

    metrics: Dict[str, object] = {
        "dataset": {
            "path": data_path,
            "total_samples": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        },
        "config": config,
        "lstm": {},
    }

    save_vocabularies(src_vocab, trg_vocab, os.path.join(output_dir, "vocab.pt"))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_path = os.path.join(output_dir, "lstm_model.pt")
    legacy_checkpoint_path = os.path.join(output_dir, "attention_model.pt")
    start_epoch = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")

    if resume:
        resume_path = checkpoint_path if os.path.exists(checkpoint_path) else legacy_checkpoint_path
        start_epoch, train_losses, val_losses = load_checkpoint_if_available(model, resume_path, device)
        if val_losses:
            best_val_loss = min(val_losses)

    print("\nTraining LSTM attention model...")
    if start_epoch >= epochs:
        print(f"Checkpoint already contains {start_epoch} epochs, which meets or exceeds the target of {epochs}.")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            clip,
            device,
            teacher_forcing_ratio,
        )
        val_loss = evaluate_epoch(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(
            f"LSTM Epoch {epoch + 1:02}/{epochs} | "
            f"Time: {epoch_mins}m {epoch_secs}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}",
            flush=True,
        )

        # Always save a checkpoint so we can inspect intermediate models, even if
        # validation loss is NaN or does not strictly improve.
        save_checkpoint(
            model=model,
            path=checkpoint_path,
            config=config,
            train_losses=train_losses,
            val_losses=val_losses,
            bleu_score=0.0,
            completed_epochs=epoch + 1,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # The file at `checkpoint_path` is already updated above; this keeps
            # `best_val_loss` tracking intact.
            pass

    bleu_score = calculate_bleu(model, test_loader, src_vocab, trg_vocab, device)
    sample_translations = prepare_sample_translations(
        model,
        test_df,
        src_vocab,
        trg_vocab,
        device,
        limit=5,
    )

    save_checkpoint(
        model=model,
        path=checkpoint_path,
        config=config,
        train_losses=train_losses,
        val_losses=val_losses,
        bleu_score=bleu_score,
        completed_epochs=epochs,
    )

    metrics["lstm"] = {
        "bleu_score": bleu_score,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "sample_translations": sample_translations,
    }

    print("\nSample translations for LSTM model:")
    for sample in sample_translations:
        print(f"English   : {sample['english']}")
        print(f"Reference : {sample['reference']}")
        print(f"Prediction: {sample['prediction']}")
        print("-" * 60)

    save_training_plot({"LSTM Model": train_losses}, os.path.join(output_dir, "training_loss.png"))
    save_metrics(metrics, os.path.join(output_dir, "metrics.json"))

    print("\nTraining complete.")
    print(f"LSTM BLEU: {metrics['lstm']['bleu_score']:.4f}")
    print(f"Artifacts saved to: {output_dir}")
    return metrics


def resolve_data_path(data_path: str, use_kagglehub: bool) -> str:
    if use_kagglehub:
        download_dir = kagglehub.dataset_download("preetviradiya/english-hindi-dataset")
        resolved_path = find_parallel_dataset_file(download_dir)
        print(f"KaggleHub dataset downloaded to: {download_dir}")
        print(f"Using dataset file: {resolved_path}")
        return resolved_path

    if not data_path:
        raise ValueError("Provide --data_path or use --use_kagglehub to download the dataset automatically.")

    if os.path.isdir(data_path):
        resolved_path = find_parallel_dataset_file(data_path)
        print(f"Using dataset file from directory: {resolved_path}")
        return resolved_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    return data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train English to Hindi LSTM translation model.")
    parser.add_argument("--data_path", type=str, default="", help="Path to English-Hindi CSV/TSV file or directory.")
    parser.add_argument("--output_dir", type=str, default="saved_models", help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=6, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding size.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden size for encoder and decoder LSTMs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help="Teacher forcing ratio.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min_freq", type=int, default=1, help="Minimum token frequency.")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=5000,
        help="Limit dataset rows for faster training.",
    )
    parser.add_argument(
        "--use_kagglehub",
        action="store_true",
        help="Download the preetviradiya English-Hindi dataset using kagglehub and train on it.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from saved_models/lstm_model.pt if it exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_data_path = resolve_data_path(args.data_path, args.use_kagglehub)
    train_and_evaluate(
        data_path=resolved_data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        dropout=args.dropout,
        num_layers=args.num_layers,
        test_size=args.test_size,
        seed=args.seed,
        min_freq=args.min_freq,
        clip=args.clip,
        max_rows=args.max_rows,
        resume=args.resume,
    )
