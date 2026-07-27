"""
LSTM Feature Extractor
"""

import torch
import torch.nn as nn


class LSTMFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):

        # CNN Output:
        # (Batch, Channels, Length)

        # Convert to:
        # (Batch, Sequence Length, Features)

        x = x.permute(0, 2, 1)

        output, (hidden, cell) = self.lstm(x)

        return hidden[-1]


if __name__ == "__main__":

    model = LSTMFeatureExtractor()

    x = torch.randn(8, 64, 19)

    output = model(x)

    print(output.shape)