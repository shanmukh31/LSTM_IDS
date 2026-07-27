"""
Hybrid CNN-LSTM Network
"""

import torch
import torch.nn as nn

from models.cnn import CNNFeatureExtractor
from models.lstm import LSTMFeatureExtractor


class CNNLSTM(nn.Module):

    def __init__(self, num_classes=14):

        super().__init__()

        self.cnn = CNNFeatureExtractor()

        self.lstm = LSTMFeatureExtractor()

        self.classifier = nn.Sequential(

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, num_classes)

        )

    def forward(self, x):

        # Input:
        # (Batch, 78)

        x = x.unsqueeze(1)

        # CNN

        x = self.cnn(x)

        # LSTM

        x = self.lstm(x)

        # Classification

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = CNNLSTM()

    x = torch.randn(16, 78)

    output = model(x)

    print(model)

    print()

    print(output.shape)