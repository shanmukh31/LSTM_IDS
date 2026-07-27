"""
PyTorch Dataset for CICIDS2017
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PROCESSED_PATH = Path("../datasets/processed")


class CICIDSDataset(Dataset):

    def __init__(self, train=True):

        if train:

            self.X = np.load(PROCESSED_PATH / "X_train.npy")
            self.y = np.load(PROCESSED_PATH / "y_train.npy")

        else:

            self.X = np.load(PROCESSED_PATH / "X_test.npy")
            self.y = np.load(PROCESSED_PATH / "y_test.npy")

        self.X = torch.tensor(self.X, dtype=torch.float32)

        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        return self.X[index], self.y[index]


if __name__ == "__main__":

    train_dataset = CICIDSDataset(train=True)

    print()

    print("Training Samples :", len(train_dataset))

    print("Feature Shape    :", train_dataset[0][0].shape)

    print("Label            :", train_dataset[0][1])