import torch
import torch.nn as nn
import torch.nn.functional as F

# should match dataset vocab
VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 '"
BLANK_IDX = 0
NUM_CLASSES = len(VOCAB_CHARS) + 1  # +1 for CTC blank


class ConvFeatureExtractor(nn.Module):
    """
    Simple CNN to turn each frame (3x64x64) into a feature vector.
    We'll apply it frame-wise.
    """

    def __init__(self, in_channels=3, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),         # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),    # out_dim x4x4
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # -> out_dim x1x1

    def forward(self, x):
        """
        x: (B*T, C, H, W)
        returns: (B*T, out_dim)
        """
        h = self.conv(x)
        h = self.pool(h)  # (B*T, out_dim, 1, 1)
        h = h.view(h.size(0), -1)
        return h


class LipReadingGRUCTC(nn.Module):
    """
    Baseline lip-reading model:
      - frame-wise CNN feature extractor
      - BiGRU over time
      - linear layer to CTC character logits
    """

    def __init__(
        self,
        cnn_out_dim=256,
        rnn_hidden=256,
        rnn_layers=2,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        self.cnn = ConvFeatureExtractor(out_dim=cnn_out_dim)
        self.rnn = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)  # bidirectional

    def forward(self, video, lengths):
        """
        video: (B, T, C, H, W)
        lengths: (B,) actual lengths before padding (for CTC)

        returns:
          logits: (T_max, B, num_classes)   [CTC expects (T,B,C)]
        """
        B, T, C, H, W = video.shape
        # merge batch and time to pass through CNN
        x = video.view(B * T, C, H, W)           # (B*T, C, H, W)
        feats = self.cnn(x)                      # (B*T, F)
        feats = feats.view(B, T, -1)             # (B, T, F)

        # pack sequence for RNN
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # (B, T_max, 2*hidden)

        logits = self.fc(out)                    # (B, T_max, num_classes)
        logits = logits.transpose(0, 1)          # (T_max, B, num_classes) for CTC
        return logits
