from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class IQCNN(nn.Module):
    """1D CNN encoder for IQ samples (2x128).

    Outputs:
      logits: (B, num_classes)
      emb:    (B, emb_dim)
    """
    def __init__(self, num_classes: int, emb_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        self.feat = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 128 -> 64

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 64 -> 32

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> (B, 256, 1)
        )

        self.emb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, emb_dim),
        )
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 2, 128)
        h = self.feat(x)
        e = self.emb(h)                 # (B, emb_dim)
        e = F.normalize(e, dim=-1)      # helpful for similarity later
        logits = self.cls(e)            # (B, C)
        return logits, e
