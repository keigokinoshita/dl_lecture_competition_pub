import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        n_layer: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        if n_layer == 5:
            self.blocks = nn.Sequential(
                ConvBlock(in_channels, hid_dim, p_drop=p_drop),
                ConvBlock(hid_dim, hid_dim, p_drop=p_drop),
                ConvBlock(hid_dim, hid_dim, p_drop=p_drop),
                ConvBlock(hid_dim, hid_dim, p_drop=p_drop),
                ConvBlock(hid_dim, hid_dim, p_drop=p_drop),
            )
        else:
            self.blocks = nn.Sequential(
                ConvBlock(in_channels, hid_dim, p_drop=p_drop),
                ConvBlock(hid_dim, hid_dim, p_drop=p_drop),
                ConvBlock(hid_dim, hid_dim, p_drop=p_drop),
            )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X) + X  # skip connection
        X = F.gelu(self.batchnorm2(X))

        return self.dropout(X)