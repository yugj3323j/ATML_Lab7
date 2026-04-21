<div align="center">

# 🌐 English → Hindi Neural Machine Translator

**Encoder–Decoder with Bahdanau Attention (LSTM)**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A sequence-to-sequence neural machine translation system with Bahdanau attention, trained on an English–Hindi parallel corpus, featuring a dark glassmorphism Streamlit dashboard for interactive translation.

**Yug Nagda** · Roll No. **I050** · ATML Lab 7
<img width="1908" height="891" alt="Screenshot 2026-04-22 035037" src="https://github.com/user-attachments/assets/1666399e-bf11-4c57-a17e-24eaf4f287e1" />


</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Training](#-training)
- [Running the App](#-running-the-app)
- [App Features](#-app-features)
- [Model Performance](#-model-performance)
- [Notes](#-notes)

---

## 🔍 Overview

This project implements a complete **English → Hindi machine translation** pipeline using an LSTM-based encoder–decoder architecture with **Bahdanau (additive) attention**. The system includes:

- **Data preprocessing** — tokenization for English and Devanagari script with Unicode-aware splitting
- **Training pipeline** — configurable epochs, batch size, teacher forcing, gradient clipping, and checkpoint resume
- **BLEU evaluation** — corpus-level BLEU score with smoothing for quantitative assessment
- **Pretrained fallback** — Helsinki-NLP/opus-mt-en-hi via HuggingFace Transformers for comparison
- **Interactive UI** — a Streamlit dashboard with dark glassmorphism styling for real-time translation

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────┐
│                  Encoder (LSTM)                   │
│  Input → Embedding → Bidirectional LSTM → States  │
└────────────────────┬─────────────────────────────┘
                     │ hidden states
           ┌─────────▼─────────┐
           │ Bahdanau Attention │
           │  (Additive Attn)   │
           └─────────┬─────────┘
                     │ context vector
┌────────────────────▼─────────────────────────────┐
│              Decoder (LSTM + Attention)            │
│  <sos> → Embedding → LSTM → Attention → FC → Token│
│            (autoregressive generation)             │
└──────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
I050_Yug-Nagda_ATML_Lab-7/
├── README.md
├── screenshot.png
└── mt_project/
    ├── app.py                    # Streamlit UI (dark glassmorphism theme)
    ├── model.py                  # Encoder, Decoder, Attention, Seq2Seq
    ├── train.py                  # Training loop with checkpointing
    ├── utils.py                  # Tokenization, vocab, BLEU, data loading
    ├── requirements.txt          # Python dependencies
    ├── Dataset_English_Hindi.csv # Parallel corpus (~90k sentence pairs)
    └── saved_models/
        ├── lstm_model.pt         # Trained model checkpoint
        ├── vocab.pt              # Source & target vocabularies
        ├── metrics.json          # BLEU, losses, sample translations
        └── training_loss.png     # Loss curve visualization
```

---

## ⚙ Setup & Installation

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋 Training

Run training from inside the `mt_project/` folder:

```bash
# Full training (default: 6 epochs, 5000 rows)
python train.py --data_path Dataset_English_Hindi.csv

# Quick training run
python train.py --data_path Dataset_English_Hindi.csv --epochs 4 --batch_size 16 --max_rows 3000

# Resume from a previous checkpoint
python train.py --data_path Dataset_English_Hindi.csv --resume

# Download dataset automatically via KaggleHub
python train.py --use_kagglehub --epochs 8
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 6 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--embedding_dim` | 128 | Token embedding dimension |
| `--hidden_dim` | 256 | LSTM hidden state dimension |
| `--learning_rate` | 0.001 | Adam optimizer learning rate |
| `--teacher_forcing_ratio` | 0.5 | Teacher forcing probability |
| `--dropout` | 0.2 | Dropout rate |
| `--num_layers` | 1 | Number of LSTM layers |
| `--max_rows` | 5000 | Dataset row limit for fast experiments |

---

## 🚀 Running the App

```bash
cd mt_project
streamlit run app.py
```

Open the URL shown in the terminal (default: [http://localhost:8501](http://localhost:8501)).

---

## ✨ App Features

- **Dark Glassmorphism UI** — frosted glass cards, neon accent gradients, animated title
- **Two Translation Models** — switch between the locally trained LSTM model and the pretrained Helsinki-NLP model via sidebar
- **Real-time Translation** — enter any English sentence and get the Hindi translation instantly
- **Metric Dashboard** — BLEU score, training loss, and validation loss displayed as stylish metric cards
- **Training Loss Graph** — visual loss curve from the training run
- **Sample Translations** — side-by-side comparison table of reference vs. predicted translations

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| BLEU Score | 0.0077 |
| Training Loss | 4.7920 |
| Validation Loss | 6.9613 |
| Epochs Trained | 8 |
| Dataset Size | 7,947 pairs |

> **Note:** The BLEU score reflects early-stage training on a limited subset. Performance improves significantly with more epochs and a larger vocabulary.

---

## 📝 Notes

- If the local checkpoint is missing, train the model first using the commands above.
- Use `--resume` to continue training from the last saved checkpoint if interrupted.
- If GPU memory is tight, reduce `batch_size`, `max_rows`, or `hidden_dim`.
- The pretrained model (`Helsinki-NLP/opus-mt-en-hi`) downloads automatically on first use.

---

<div align="center">

**Yug Nagda** · I050 · ATML Lab 7

</div>
