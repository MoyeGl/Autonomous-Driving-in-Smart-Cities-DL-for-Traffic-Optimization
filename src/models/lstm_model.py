# =============================================================================
# lstm_model.py
# Purpose: Define PyTorch LSTM model for speed prediction
# =============================================================================

import torch
import torch.nn as nn

class SpeedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):

        super(SpeedLSTM, self).__init__()

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)  # Predict a single value (speed)

    def forward(self, x):

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)

        # Take output of the last time step only
        last_output = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)

        # Fully connected layer
        prediction = self.fc(last_output)  # shape: (batch_size, 1)

        return prediction
