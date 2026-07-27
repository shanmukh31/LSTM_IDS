import torch
import numpy as np
import joblib
import warnings

from models.cnn_lstm import CNNLSTM


# ==========================================================
# Ignore sklearn feature warning
# ==========================================================

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)



# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


MODEL_PATH = "../saved_models/cnn_lstm_best.pth"

SCALER_PATH = "../datasets/processed/scaler.pkl"

ENCODER_PATH = "../datasets/processed/label_encoder.pkl"



print(f"\nUsing Device: {DEVICE}\n")



# ==========================================================
# Load preprocessing objects
# ==========================================================

scaler = joblib.load(
    SCALER_PATH
)


label_encoder = joblib.load(
    ENCODER_PATH
)



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


print("Model Loaded Successfully")



# ==========================================================
# Prediction Function
# ==========================================================

def predict(sample):


    # Convert to numpy

    sample = np.asarray(sample)



    # Remove extra dimensions

    sample = np.squeeze(sample)



    # Convert:
    # (78,)
    #
    # to:
    # (1,78)

    sample = sample.reshape(
        1,
        -1
    )



    # Apply scaler

    sample = scaler.transform(
        sample
    )



    # Tensor conversion

    sample = torch.tensor(
        sample,
        dtype=torch.float32
    )


    sample = sample.to(
        DEVICE
    )



    # IMPORTANT:
    # CNNLSTM.forward()
    # already performs:
    #
    # x.unsqueeze(1)


    with torch.no_grad():

        output = model(sample)


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )



    predicted_class = prediction.item()


    confidence = confidence.item()



    attack_name = label_encoder.inverse_transform(
        [predicted_class]
    )[0]



    return attack_name, confidence




# ==========================================================
# Test Prediction
# ==========================================================

if __name__ == "__main__":


    X_test = np.load(
        "../datasets/processed/X_test.npy"
    )


    print(
        "Sample Shape:",
        X_test[0].shape
    )



    prediction, confidence = predict(
        X_test[0]
    )



    print("\n==============================")

    print(
        "Prediction:",
        prediction
    )


    print(
        f"Confidence: {confidence*100:.2f}%"
    )


    print("==============================")