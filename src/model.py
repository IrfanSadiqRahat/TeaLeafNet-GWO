"""
TeaLeafNet — CNN with residual blocks for tea leaf disease classification.
Published in Scientific Reports (Nature), 2025. DOI: 10.1038/s41598-025-02378-0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard residual block with skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class TeaLeafNet(nn.Module):
    """
    CNN for tea leaf disease classification.
    Input: (B, 3, 256, 256) RGB images
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        # Stem: ZeroPad + Conv + BN + ReLU + MaxPool
        self.stem = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Residual blocks
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.classifier(x)
