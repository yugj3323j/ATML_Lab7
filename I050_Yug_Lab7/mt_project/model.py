import random
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: SimpleDecoder, device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)

        _, hidden, cell = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, src_len, hidden_dim = encoder_outputs.size()
        repeated_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(src_mask == 0, -1e10)
        return torch.softmax(attention, dim=1)


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        hidden_dim: int,
        attention: BahdanauAttention,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))

        decoder_hidden = hidden[-1]
        attention_weights = self.attention(decoder_hidden, encoder_outputs, src_mask)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.fc_out(
            torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        )
        return prediction, hidden, cell, attention_weights


class AttentionSeq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: AttentionDecoder,
        src_pad_idx: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.src_pad_idx).to(self.device)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)
        attention_scores = torch.zeros(batch_size, trg_len, src.size(1), device=self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        src_mask = self.create_mask(src)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell, attention_weights = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs,
                src_mask,
            )
            outputs[:, t] = output
            attention_scores[:, t] = attention_weights
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs, attention_scores

    def greedy_decode(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs, hidden, cell = self.encoder(src)
        src_mask = self.create_mask(src)

        input_token = torch.full((src.size(0),), sos_idx, dtype=torch.long, device=self.device)
        generated_tokens = [input_token]
        collected_attention = []

        for _ in range(max_len):
            output, hidden, cell, attention_weights = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs,
                src_mask,
            )
            input_token = output.argmax(dim=1)
            generated_tokens.append(input_token)
            collected_attention.append(attention_weights)

            if torch.all(input_token == eos_idx):
                break

        tokens = torch.stack(generated_tokens, dim=1)
        attention_matrix = (
            torch.stack(collected_attention, dim=1)
            if collected_attention
            else torch.empty(0, device=self.device)
        )
        return tokens, attention_matrix


class SimpleSeq2SeqInference(Seq2Seq):
    def greedy_decode(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 50,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, hidden, cell = self.encoder(src)

        input_token = torch.full((src.size(0),), sos_idx, dtype=torch.long, device=self.device)
        generated_tokens = [input_token]

        for _ in range(max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            input_token = output.argmax(dim=1)
            generated_tokens.append(input_token)

            if torch.all(input_token == eos_idx):
                break

        tokens = torch.stack(generated_tokens, dim=1)
        return tokens, None
