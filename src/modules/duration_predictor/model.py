from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, hidden_dim)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class DurationPredictor(nn.Module):
    def __init__(
        self,
        audio_vocab_size: int,
        text_vocab_size: int,
        audio_add_tokens: int = 2, # mask + padding tokens
        hidden_size: int = 256,
        n_text_layer: int = 4,
        n_cross_layer: int = 4,
        n_head: int = 8,
        output_dim: int = 1,
    ):
        super().__init__()

        self.audio_vocab_size = audio_vocab_size
        self.text_vocab_size = text_vocab_size

        # Text Encoder: Embedding + Transformer Layers
        self.text_embed = nn.Embedding(self.text_vocab_size+1, hidden_size)
        self.text_pe = PositionalEncoding(hidden_size)
        # Audio Encoder: Embedding
        self.audio_embed = nn.Embedding(self.audio_vocab_size + audio_add_tokens, hidden_size)
        self.audio_pe = PositionalEncoding(hidden_size)

        # Transformer Encoder for Text
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_head,
                dim_feedforward=hidden_size*2,
                batch_first=True,
            ),
            num_layers=n_text_layer
        )

        # Transformer Decoder Layers with Cross-Attention in Every Layer
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=n_head,
                dim_feedforward=hidden_size*2,
                batch_first=True,
            ),
            num_layers=n_cross_layer
        )
        # Final Classification Layer
        self.predictor = nn.Linear(hidden_size, output_dim)

    @staticmethod
    def _make_causal_mask(T: int, device: torch.device) -> Tensor:
        """Upper‑triangular mask (True = mask). Shape [T, T]."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        text_ids: Tensor,
        audio_ids: Tensor,
        text_mask: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B = text_ids.size(0)

        text_key_padding = None
        if text_mask is not None:
            # Transformer expects True for *padding* positions
            text_key_padding = ~text_mask.bool()
        # Encode text
        text_emb = self.text_pe(self.text_embed(text_ids))
        text_features = self.text_encoder(text_emb, src_key_padding_mask=text_key_padding) # (B, L_text, D)

        audio_emb = self.audio_pe(self.audio_embed(audio_ids))
        # Causal Masking for Decoder
        seq_len = audio_emb.size(1)

        tgt_key_padding = None
        if audio_mask is not None:
            tgt_key_padding = ~audio_mask.bool()

        causal_mask = self._make_causal_mask(seq_len, audio_ids.device)
        # Transformer Decoder with Cross-Attention in Each Layer
        decoder_out = self.decoder(
            tgt=audio_emb,
            memory=text_emb,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=text_key_padding,
        )
        # Length Prediction
        length_logits = self.predictor(decoder_out).squeeze(-1)
        return length_logits