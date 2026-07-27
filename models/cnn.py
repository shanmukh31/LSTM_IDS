"""
CNN Feature Extractor
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(32),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.MaxPool1d(kernel_size=2)

        )

    def forward(self, x):

        x = self.feature_extractor(x)

        return x


if __name__ == "__main__":

    model = CNNFeatureExtractor()

    x = torch.randn(8, 1, 78)

    output = model(x)

    print(output.shape)