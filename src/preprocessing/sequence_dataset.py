# =============================================================================
# sequence_dataset.py
# Purpose: Prepare sequences for LSTM training
# =============================================================================
import numpy as np
from torch.utils.data import Dataset

class SpeedSequenceDataset(Dataset):
    def __init__(self, tracks_df, sequence_length, prediction_horizon):
        self.inputs = []
        self.targets = []
        print("Creating sequences for LSTM...")

        data_array = tracks_df.values  # All scaled features
        speed_index = list(tracks_df.columns).index('speed')

        for i in range(len(data_array) - sequence_length - prediction_horizon):
            input_seq = data_array[i:i + sequence_length]
            target = data_array[i + sequence_length + prediction_horizon - 1, speed_index]
            self.inputs.append(input_seq)
            self.targets.append(target)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32).reshape(-1, 1)

        print(f"Total sequences: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
