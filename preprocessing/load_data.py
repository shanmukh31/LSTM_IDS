"""
Load and merge all CICIDS2017 CSV files
"""

from pathlib import Path
import pandas as pd


# Dataset folder
DATASET_PATH = Path("../datasets/raw/MachineLearningCSV")


def load_dataset():

    csv_files = sorted(DATASET_PATH.glob("*.csv"))

    print(f"\nFound {len(csv_files)} CSV files\n")

    dataframes = []

    for file in csv_files:

        print(f"Loading: {file.name}")

        df = pd.read_csv(file, low_memory=False)

        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    print("\n====================================")
    print("Dataset Successfully Loaded")
    print("====================================")

    print(f"Rows    : {merged_df.shape[0]}")
    print(f"Columns : {merged_df.shape[1]}")

    print("\nAttack Classes:\n")
    print(merged_df[" Label"].value_counts())

    return merged_df


if __name__ == "__main__":

    dataset = load_dataset()