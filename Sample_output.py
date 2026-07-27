"""
CNN-LSTM IDS Prediction Demo
Generates sample input/output for README and results folder
"""

import os
import random
import numpy as np
import torch
import joblib

from models.cnn_lstm import CNNLSTM


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


MODEL_PATH = "../saved_models/cnn_lstm_best.pth"

X_TEST_PATH = "../datasets/processed/X_test.npy"

Y_TEST_PATH = "../datasets/processed/y_test.npy"

ENCODER_PATH = "../datasets/processed/label_encoder.pkl"


OUTPUT_PATH = "../results/sample_prediction.txt"



# ==========================================================
# Load Model
# ==========================================================

model = CNNLSTM().to(DEVICE)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)


model.eval()


print("\nModel Loaded Successfully")



# ==========================================================
# Load Data
# ==========================================================

X_test = np.load(
    X_TEST_PATH
)


y_test = np.load(
    Y_TEST_PATH
)


label_encoder = joblib.load(
    ENCODER_PATH
)



# ==========================================================
# Random Sample Selection
# ==========================================================

sample_index = random.randint(
    0,
    len(X_test)-1
)


sample = X_test[sample_index]


actual_label = y_test[sample_index]



# ==========================================================
# Select Number of Features to Display
# ==========================================================

# Change this number according to your need
# Actual model always uses all 78 features

display_features = random.choice(
    [5,10,15,20]
)



display_sample = sample[:display_features]



# ==========================================================
# Prediction
# ==========================================================

input_tensor = torch.tensor(
    sample,
    dtype=torch.float32
)


# Shape:
# (Features)
# becomes
# (Batch, Features)

input_tensor = input_tensor.unsqueeze(0)


input_tensor = input_tensor.to(DEVICE)



with torch.no_grad():

    output = model(
        input_tensor
    )


    probabilities = torch.softmax(
        output,
        dim=1
    )


    confidence, prediction = torch.max(
        probabilities,
        1
    )



predicted_class = prediction.item()


confidence = confidence.item()*100



predicted_label = label_encoder.inverse_transform(
    [predicted_class]
)[0]


actual_label_name = label_encoder.inverse_transform(
    [actual_label]
)[0]



# ==========================================================
# Display Result
# ==========================================================

result = f"""

==================================================
CNN-LSTM IDS Prediction Demo
==================================================


Device Used:
{DEVICE}


Random Sample Index:
{sample_index}


Number of Features Displayed:
{display_features}

(Note: Model internally uses all 78 features)


Sample Input Features:

{display_sample}



Actual Class:
{actual_label_name}



Predicted Class:
{predicted_label}



Prediction Confidence:
{confidence:.2f}%


==================================================

"""


print(result)



# ==========================================================
# Save Output
# ==========================================================

os.makedirs(
    "../results",
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w"
) as file:

    file.write(result)



print(
    f"\nSaved output to {OUTPUT_PATH}"
)



