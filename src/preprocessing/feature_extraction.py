# =============================================================================
# feature_extraction.py
# Purpose: Extract and clean features for ML/RL models
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_basic_features(tracks_df):
    print("🔧 Computing basic features...")
    tracks_df = tracks_df.sort_values(by=["id", "frame"])

    # Rename velocity columns
    if 'xVelocity' in tracks_df.columns:
        tracks_df = tracks_df.rename(columns={'xVelocity': 'speed_longitudinal'})
    if 'yVelocity' in tracks_df.columns:
        tracks_df = tracks_df.rename(columns={'yVelocity': 'speed_lateral'})

    # Total speed magnitude
    tracks_df['speed'] = np.sqrt(
        tracks_df['speed_longitudinal'] ** 2 + tracks_df['speed_lateral'] ** 2
    )

    # Replace invalid speeds
    tracks_df['speed'] = tracks_df['speed'].fillna(0).clip(0, 60)

    # Compute acceleration
    tracks_df['acceleration'] = tracks_df.groupby('id')['speed'].diff().fillna(0)

    # Lane change detection
    tracks_df['lane_change'] = tracks_df.groupby('id')['laneId'].diff().fillna(0).abs()
    tracks_df['lane_change'] = tracks_df['lane_change'].apply(lambda x: 1 if x > 0 else 0)

    return tracks_df

def compute_relative_features(tracks_df):
    print(" Computing relative features...")
    possible_leader_columns = ['precedingId', 'precedingVehicleId', 'frontVehicleId']
    leader_col = next((col for col in possible_leader_columns if col in tracks_df.columns), None)

    if leader_col is None:
        print(" No preceding vehicle column. Setting default values.")
        tracks_df['leader_id'] = 0
        tracks_df['delta_x'] = 1000
        tracks_df['delta_v'] = 0
        tracks_df['time_gap'] = np.inf
        return tracks_df

    tracks_df['leader_id'] = tracks_df[leader_col]

    leader_data = tracks_df[['frame', 'id', 'x', 'speed']].rename(
        columns={'id': 'leader_id', 'x': 'leader_x', 'speed': 'leader_speed'}
    )
    tracks_df = tracks_df.merge(leader_data, on=['frame', 'leader_id'], how='left')

    # Compute relative features
    tracks_df['delta_x'] = (tracks_df['leader_x'] - tracks_df['x']).fillna(1000)
    tracks_df['delta_v'] = (tracks_df['leader_speed'] - tracks_df['speed']).fillna(0)
    tracks_df['time_gap'] = tracks_df.apply(
        lambda row: (row['delta_x'] / row['speed']) if row['speed'] > 0 else np.inf,
        axis=1
    )

    # Clip extreme values
    tracks_df['delta_x'] = tracks_df['delta_x'].clip(-1000, 1000)
    tracks_df['delta_v'] = tracks_df['delta_v'].clip(-50, 50)
    tracks_df['time_gap'] = tracks_df['time_gap'].replace([np.inf, -np.inf], 1000)

    # Drop temp columns
    tracks_df.drop(columns=['leader_x', 'leader_speed'], errors='ignore', inplace=True)
    return tracks_df

def extract_features(tracks_df):
    print("\nStarting full feature extraction pipeline...")
    tracks_df = compute_basic_features(tracks_df)
    tracks_df = compute_relative_features(tracks_df)

    # Select final features
    features = ['speed', 'acceleration', 'laneId', 'delta_x', 'delta_v', 'time_gap']
    tracks_df = tracks_df[features].replace([np.inf, -np.inf], 1000).fillna(0)

    # Normalize
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(tracks_df)
    scaled_df = pd.DataFrame(scaled_data, columns=features)

    print(" Feature extraction and scaling complete!\n")
    return scaled_df
