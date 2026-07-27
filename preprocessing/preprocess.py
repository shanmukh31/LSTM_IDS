"""
Preprocess CICIDS2017 Dataset
Prototype Version (300k Samples)

Author: You
Framework: PyTorch
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ==========================================================
# Configuration
# ==========================================================

RAW_DATASET = Path("../datasets/raw/MachineLearningCSV")
OUTPUT = Path("../datasets/processed")

OUTPUT.mkdir(parents=True, exist_ok=True)

TOTAL_SAMPLE_SIZE = 300_000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ==========================================================
# Load + Sample
# ==========================================================

def load_and_sample():

    csv_files = sorted(RAW_DATASET.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found.")

    print("\nFound", len(csv_files), "CSV files\n")

    samples = []

    sample_per_file = TOTAL_SAMPLE_SIZE // len(csv_files)

    for file in csv_files:

        print(f"Reading {file.name}")

        df = pd.read_csv(file, low_memory=False)

        # Remove spaces in column names
        df.columns = df.columns.str.strip()

        # Replace Inf
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Remove NaN
        df.dropna(inplace=True)

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Sample from this file
        if len(df) > sample_per_file:

            df = df.sample(
                n=sample_per_file,
                random_state=RANDOM_STATE
            )

        samples.append(df)

        print(f"   Selected {len(df):,} rows")

    print("\nMerging sampled data...\n")

    return pd.concat(samples, ignore_index=True)


# ==========================================================
# Main
# ==========================================================

def preprocess():

    df = load_and_sample()

    print("Merged Shape :", df.shape)

    # ------------------------------------------
    # Features / Labels
    # ------------------------------------------
    print("\nClass Distribution:\n")
    print(df["Label"].value_counts())
    X = df.drop("Label", axis=1)

    y = df["Label"]

    # ------------------------------------------
    # Encode Labels
    # ------------------------------------------

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    joblib.dump(
        encoder,
        OUTPUT / "label_encoder.pkl"
    )

    # ------------------------------------------
    # Standardize Features
    # ------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    joblib.dump(
        scaler,
        OUTPUT / "scaler.pkl"
    )

    # ------------------------------------------
    # Train Test Split
    # ------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ------------------------------------------
    # Save
    # ------------------------------------------

    np.save(OUTPUT / "X_train.npy", X_train)
    np.save(OUTPUT / "X_test.npy", X_test)
    np.save(OUTPUT / "y_train.npy", y_train)
    np.save(OUTPUT / "y_test.npy", y_test)

    print("\n====================================")
    print("Preprocessing Complete")
    print("====================================")

    print(f"Training Samples : {len(X_train):,}")
    print(f"Testing Samples  : {len(X_test):,}")
    print(f"Classes          : {len(np.unique(y))}")

    print("\nFiles Saved To:")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    preprocess()