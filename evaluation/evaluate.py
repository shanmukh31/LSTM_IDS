import os

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing.dataset import CICIDSDataset
from models.cnn_lstm import CNNLSTM


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {DEVICE}\n")


# ==========================================================
# Create Results Directory
# ==========================================================

os.makedirs(
    "../results",
    exist_ok=True
)


# ==========================================================
# Load Dataset
# ==========================================================

test_dataset = CICIDSDataset(
    train=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)


# ==========================================================
# Load Model
# ==========================================================

model = CNNLSTM().to(DEVICE)

model.load_state_dict(
    torch.load(
        "../saved_models/cnn_lstm_best.pth",
        map_location=DEVICE
    )
)

model.eval()


# ==========================================================
# Prediction
# ==========================================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)

        y = y.to(DEVICE)


        outputs = model(X)


        _, predicted = torch.max(
            outputs,
            1
        )


        all_labels.extend(
            y.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# ==========================================================
# Calculate Metrics
# ==========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


cm = confusion_matrix(
    all_labels,
    all_predictions
)


report = classification_report(
    all_labels,
    all_predictions,
    zero_division=0
)


# ==========================================================
# Display Results
# ==========================================================

print("=" * 60)

print(
    f"Accuracy : {accuracy*100:.2f}%"
)

print(
    f"Precision: {precision*100:.2f}%"
)

print(
    f"Recall   : {recall*100:.2f}%"
)

print(
    f"F1 Score : {f1*100:.2f}%"
)

print("=" * 60)


print("\nConfusion Matrix\n")

print(cm)


print("\nClassification Report\n")

print(report)



# ==========================================================
# Save Classification Report
# ==========================================================

with open(
    "../results/classification_report.txt",
    "w"
) as file:

    file.write(report)



# ==========================================================
# Save Metrics
# ==========================================================

with open(
    "../results/metrics.txt",
    "w"
) as file:

    file.write(
        f"Accuracy : {accuracy*100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision*100:.2f}%\n"
    )

    file.write(
        f"Recall   : {recall*100:.2f}%\n"
    )

    file.write(
        f"F1 Score : {f1*100:.2f}%\n"
    )


# ==========================================================
# Confusion Matrix Heatmap
# ==========================================================

plt.figure(
    figsize=(12,10)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5
)


plt.title(
    "CNN-LSTM Confusion Matrix"
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "True Class"
)


plt.tight_layout()


plt.savefig(
    "../results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()



print("\nEvaluation Completed Successfully.")

print("Results saved in ../results/")