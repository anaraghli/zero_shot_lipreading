import math
import torch
import torch.nn as nn

from model_baseline import ConvFeatureExtractor, NUM_CLASSES, BLANK_IDX

# You already defined VOCAB_CHARS / NUM_CLASSES / BLANK_IDX in model_baseline.py


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding, adapted for batch_first input:
    x: (B, T, D)
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return x


class LipReadingTransformerCTC(nn.Module):
    """
    Transformer-based lip-reading model:

      - CNN feature extractor per frame
      - Transformer encoder over time (with positional encoding)
      - Linear layer to CTC character logits

    Forward:
      video: (B, T, C, H, W)
      lengths: (B,) actual lengths before padding

    Returns:
      logits: (T_max, B, num_classes) for CTC
    """

    def __init__(
        self,
        cnn_out_dim: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        assert d_model == cnn_out_dim, "d_model must match cnn_out_dim for simplicity"

        self.cnn = ConvFeatureExtractor(out_dim=cnn_out_dim)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=500)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # we work with (B, T, D)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, video: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        video: (B, T, C, H, W)
        lengths: (B,)

        Returns:
          logits: (T_max, B, num_classes) for CTC
        """
        B, T, C, H, W = video.shape

        # CNN per frame
        x = video.view(B * T, C, H, W)          # (B*T, C, H, W)
        feats = self.cnn(x)                     # (B*T, F)
        feats = feats.view(B, T, -1)            # (B, T, F)

        # Positional encoding
        feats = self.pos_encoder(feats)         # (B, T, F)

        # Build padding mask: True = pad position to ignore
        device = lengths.device
        T_max = feats.size(1)
        # shape (B, T_max) with True where t >= length
        seq_range = torch.arange(T_max, device=device).unsqueeze(0)  # (1, T)
        lengths_exp = lengths.unsqueeze(1)                           # (B, 1)
        src_key_padding_mask = seq_range >= lengths_exp              # (B, T)

        # Transformer encoder
        # batch_first=True, so input is (B, T, F), mask is (B, T)
        out = self.transformer(
            feats,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, T, F)

        logits = self.fc(out)                # (B, T, num_classes)
        logits = logits.transpose(0, 1)      # (T, B, num_classes) for CTC

        return logits
