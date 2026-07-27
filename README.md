# CNN-LSTM Based Intrusion Detection System Using CICIDS2017

## Overview

This project implements a deep learning-based Intrusion Detection System (IDS) using a hybrid **Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM)** architecture.

The model learns network traffic patterns from the CICIDS2017 dataset and classifies traffic into multiple attack categories. CNN extracts important feature patterns, while LSTM captures sequential relationships between network features.

The complete implementation is developed using **PyTorch**.

---

## Objectives

- Detect malicious network traffic using deep learning.
- Perform multi-class intrusion classification.
- Analyze network traffic features automatically.
- Build an efficient CNN-LSTM based IDS pipeline.

---

## Dataset

**Dataset:** CICIDS2017

The dataset contains realistic network traffic samples including:

- BENIGN traffic
- Denial of Service attacks
- Brute Force attacks
- Bot attacks
- Port Scanning attacks
- Other malicious activities

### Dataset Features

- 78 network traffic features
- Multi-class classification problem
- Preprocessed using feature scaling and label encoding

---

# System Architecture



---

# Model Architecture

## CNN Component

Used for extracting important patterns from network traffic features.

Functions:
- Feature extraction
- Pattern recognition
- Dimensional representation learning


## LSTM Component

Used for learning relationships between extracted features.

Functions:
- Sequential feature learning
- Temporal dependency detection


## Classifier

Fully connected layers classify the input traffic into attack categories.

---

# Running the Project

## 1. Data Preprocessing

Run:

```bash
python preprocessing/preprocess.py
```

Generated files:

```text
datasets/processed/

├── X_train.npy
├── X_test.npy
├── y_train.npy
├── y_test.npy
├── scaler.pkl
└── label_encoder.pkl
```

---

## 2. Model Training

Run:

```bash
python training/train.py
```

Trained models are saved in:

```text
saved_models/

├── cnn_lstm_best.pth
└── cnn_lstm_final.pth
```

---

## 3. Model Evaluation

Run:

```bash
python evaluation/evaluate.py
```

Generated evaluation results:

```text
results/

├── confusion_matrix.png
├── classification_report.txt
└── metrics.txt
```

---

## 4. Intrusion Prediction

Run:

```bash
python prediction/predict.py
```

---

# Model Performance

## Final CNN-LSTM Model Performance

```text
Best Test Accuracy : 98.25%

Train Accuracy : 98.14%

Test Accuracy  : 98.18%
```

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix