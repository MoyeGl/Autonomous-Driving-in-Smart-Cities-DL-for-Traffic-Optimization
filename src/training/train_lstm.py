# =============================================================================
# train_lstm.py
# Purpose: Train LSTM model with GPU support, normalization, plotting, and evaluation
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.preprocessing.feature_extraction import extract_features
from src.preprocessing.data_loader import HighDLoader
from src.preprocessing.sequence_dataset import SpeedSequenceDataset
from src.models.lstm_model import SpeedLSTM

# =====================
# Hyperparameters
# =====================
SEQUENCE_LENGTH = 30
PREDICTION_HORIZON = 5
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8

# =====================
# Data Loading Function
# =====================
def load_and_prepare_data(dataset_path, recording_id=1):
    print("📥 Loading raw data...")
    loader = HighDLoader(dataset_path)
    tracks, _, _ = loader.load_data(recording_id)

    print("🧠 Extracting and scaling features...")
    features_df = extract_features(tracks)

    print("🔄 Creating sequences for LSTM...")
    dataset = SpeedSequenceDataset(
        tracks_df=features_df,
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON
    )

    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"✅ Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    return train_loader, test_loader

# =====================
# Evaluation & Plotting
# =====================
def evaluate_and_visualize(model, test_loader, device):
    print("\n🔍 Evaluating model on test data...")
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    # ✅ Convert lists of numpy arrays into one contiguous numpy array (fast & safe)
    predictions = np.array(predictions, dtype=np.float32).flatten()
    actuals = np.array(actuals, dtype=np.float32).flatten()

    # ✅ Metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("\n📊 Evaluation Metrics:")
    print(f"✅ Mean Squared Error (MSE): {mse:.6f}")
    print(f"✅ Mean Absolute Error (MAE): {mae:.6f}")
    print(f"✅ R² Score: {r2:.4f}  (Closer to 1 is better)")

    # ✅ Plot predictions vs actual for first 200 samples
    plt.figure(figsize=(10, 5))
    plt.plot(actuals[:200], label='Actual Speed')
    plt.plot(predictions[:200], label='Predicted Speed')
    plt.title('Actual vs Predicted Speed (First 200 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (normalized)')
    plt.legend()
    plt.grid(True)
    plt.savefig("prediction_vs_actual.png")
    plt.show()
    print("📈 Plot saved as prediction_vs_actual.png")

# =====================
# Training Function
# =====================
def train_model(train_loader, test_loader, input_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")

    model = SpeedLSTM(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if torch.isnan(loss):
                print("⚠ NaN loss detected, skipping this batch")
                continue

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"📉 Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

    # ✅ Save model
    torch.save(model.state_dict(), "lstm_speed_predictor.pth")
    print("✅ Model saved as lstm_speed_predictor.pth")

    # ✅ Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Testing Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve.png")
    plt.show()
    print("📊 Training curve saved as training_curve.png")

    # ✅ Evaluate & visualize predictions
    evaluate_and_visualize(model, test_loader, device)

# =====================
# Main Entry Point
# =====================
def main():
    dataset_path = "../../data/highD-dataset-v1.0"
    train_loader, test_loader = load_and_prepare_data(dataset_path)

    # Infer input size from one batch
    sample_input, _ = next(iter(train_loader))
    input_size = sample_input.shape[2]

    train_model(train_loader, test_loader, input_size)

if __name__ == "__main__":
    main()
