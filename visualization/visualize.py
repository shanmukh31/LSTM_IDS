"""
CICIDS2017 Dataset Visualization
Creates graphs for analysis and reporting
"""


import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



# ==========================================================
# Configuration
# ==========================================================


PROCESSED_PATH = "../datasets/processed"

RESULT_PATH = "../results"


os.makedirs(
    RESULT_PATH,
    exist_ok=True
)



# ==========================================================
# Load Data
# ==========================================================


X_train = np.load(
    f"{PROCESSED_PATH}/X_train.npy"
)


X_test = np.load(
    f"{PROCESSED_PATH}/X_test.npy"
)


y_train = np.load(
    f"{PROCESSED_PATH}/y_train.npy"
)


y_test = np.load(
    f"{PROCESSED_PATH}/y_test.npy"
)



print("Data Loaded")

print(
    "Training:",
    X_train.shape
)

print(
    "Testing:",
    X_test.shape
)



# ==========================================================
# 1. Class Distribution
# ==========================================================


unique, counts = np.unique(
    y_train,
    return_counts=True
)


plt.figure(
    figsize=(10,6)
)


sns.barplot(
    x=unique,
    y=counts
)


plt.title(
    "CICIDS2017 Attack Class Distribution"
)


plt.xlabel(
    "Class Label"
)


plt.ylabel(
    "Number of Samples"
)


plt.xticks(
    rotation=45
)


plt.tight_layout()


plt.savefig(
    f"{RESULT_PATH}/class_distribution.png",
    dpi=300
)


plt.close()



# ==========================================================
# 2. Dataset Split Visualization
# ==========================================================


labels = [
    "Training",
    "Testing"
]


values = [
    len(X_train),
    len(X_test)
]


plt.figure(
    figsize=(6,5)
)


plt.bar(
    labels,
    values
)


plt.title(
    "Dataset Split"
)


plt.ylabel(
    "Number of Samples"
)


plt.tight_layout()


plt.savefig(
    f"{RESULT_PATH}/dataset_split.png",
    dpi=300
)


plt.close()



# ==========================================================
# 3. Feature Histogram
# ==========================================================


plt.figure(
    figsize=(10,6)
)


plt.hist(
    X_train[:,0],
    bins=50
)


plt.title(
    "Feature Distribution Example"
)


plt.xlabel(
    "Normalized Feature Value"
)


plt.ylabel(
    "Frequency"
)


plt.tight_layout()


plt.savefig(
    f"{RESULT_PATH}/feature_histogram.png",
    dpi=300
)


plt.close()



# ==========================================================
# 4. Correlation Heatmap
# ==========================================================


# Select first 20 features
# for readability

feature_df = pd.DataFrame(
    X_train[:, :20]
)



plt.figure(
    figsize=(12,8)
)


sns.heatmap(
    feature_df.corr(),
    cmap="coolwarm"
)



plt.title(
    "Feature Correlation Heatmap"
)


plt.tight_layout()



plt.savefig(
    f"{RESULT_PATH}/correlation_heatmap.png",
    dpi=300
)



plt.close()



print("\nVisualization Completed Successfully")

print(
    "Images saved inside results/"
)