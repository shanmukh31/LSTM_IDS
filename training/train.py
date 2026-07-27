import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from preprocessing.dataset import CICIDSDataset
from models.cnn_lstm import CNNLSTM



# ==========================================================
# Device Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {DEVICE}\n")



# ==========================================================
# Hyperparameters
# ==========================================================

BATCH_SIZE = 256

EPOCHS = 10

LEARNING_RATE = 0.001



# ==========================================================
# Dataset
# ==========================================================

train_dataset = CICIDSDataset(
    train=True
)


test_dataset = CICIDSDataset(
    train=False
)



train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)



# ==========================================================
# Model
# ==========================================================

model = CNNLSTM().to(DEVICE)



criterion = nn.CrossEntropyLoss()



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)



# ==========================================================
# Directories
# ==========================================================

os.makedirs(
    "../saved_models",
    exist_ok=True
)


os.makedirs(
    "../results",
    exist_ok=True
)



# ==========================================================
# Training History
# ==========================================================

train_losses = []

test_losses = []

train_accuracies = []

test_accuracies = []



best_accuracy = 0.0



# ==========================================================
# Training Loop
# ==========================================================

for epoch in range(EPOCHS):


    # ------------------------------------------------------
    # Training Phase
    # ------------------------------------------------------

    model.train()


    train_loss = 0.0

    train_correct = 0

    train_total = 0



    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )



    for X, y in progress_bar:


        X = X.to(DEVICE)

        y = y.to(DEVICE)



        optimizer.zero_grad()



        outputs = model(X)



        loss = criterion(
            outputs,
            y
        )



        loss.backward()



        optimizer.step()



        train_loss += loss.item()



        _, predicted = torch.max(
            outputs,
            1
        )


        train_total += y.size(0)


        train_correct += (
            predicted == y
        ).sum().item()



        progress_bar.set_postfix(
            Loss=f"{loss.item():.4f}"
        )



    train_loss /= len(train_loader)


    train_accuracy = (
        100 * train_correct / train_total
    )



    # ------------------------------------------------------
    # Testing Phase
    # ------------------------------------------------------

    model.eval()


    test_loss = 0.0

    test_correct = 0

    test_total = 0



    with torch.no_grad():


        for X, y in test_loader:


            X = X.to(DEVICE)

            y = y.to(DEVICE)



            outputs = model(X)



            loss = criterion(
                outputs,
                y
            )



            test_loss += loss.item()



            _, predicted = torch.max(
                outputs,
                1
            )



            test_total += y.size(0)


            test_correct += (
                predicted == y
            ).sum().item()



    test_loss /= len(test_loader)


    test_accuracy = (
        100 * test_correct / test_total
    )



    # ------------------------------------------------------
    # Save History
    # ------------------------------------------------------

    train_losses.append(
        train_loss
    )

    test_losses.append(
        test_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    test_accuracies.append(
        test_accuracy
    )



    # ------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------

    scheduler.step(
        test_loss
    )



    # ------------------------------------------------------
    # Save Best Model
    # ------------------------------------------------------

    if test_accuracy > best_accuracy:


        best_accuracy = test_accuracy


        torch.save(
            model.state_dict(),
            "../saved_models/cnn_lstm_best.pth"
        )


        print("\nBest model updated.")



    # ------------------------------------------------------
    # Epoch Summary
    # ------------------------------------------------------

    print("-" * 60)


    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )


    print(
        f"Train Loss     : {train_loss:.4f}"
    )


    print(
        f"Train Accuracy : {train_accuracy:.2f}%"
    )


    print(
        f"Test Loss      : {test_loss:.4f}"
    )


    print(
        f"Test Accuracy  : {test_accuracy:.2f}%"
    )


    print(
        f"Best Accuracy  : {best_accuracy:.2f}%"
    )


    print("-" * 60)



# ==========================================================
# Save Final Model
# ==========================================================

torch.save(
    model.state_dict(),
    "../saved_models/cnn_lstm_final.pth"
)



# ==========================================================
# Generate Training Curves
# ==========================================================


# Loss Curve

plt.figure(
    figsize=(8,5)
)


plt.plot(
    train_losses,
    label="Train Loss"
)


plt.plot(
    test_losses,
    label="Test Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "CNN-LSTM Loss Curve"
)


plt.legend()


plt.grid()



plt.savefig(
    "../results/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)



plt.close()



# Accuracy Curve

plt.figure(
    figsize=(8,5)
)


plt.plot(
    train_accuracies,
    label="Train Accuracy"
)


plt.plot(
    test_accuracies,
    label="Test Accuracy"
)



plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy (%)"
)


plt.title(
    "CNN-LSTM Accuracy Curve"
)


plt.legend()


plt.grid()



plt.savefig(
    "../results/accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)



plt.close()



# ==========================================================
# Completion Message
# ==========================================================

print("\nTraining Completed Successfully.")

print(
    f"Best Test Accuracy : {best_accuracy:.2f}%"
)

print(
    "Final model saved as cnn_lstm_final.pth"
)

print(
    "Training graphs saved in results folder."
)