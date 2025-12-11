# =============================================================================
# main.py
# Author: Moye Nyuysoni Glein Perry
# Purpose: Entry point to  data loading and feature engineering
# =============================================================================

from preprocessing.data_loader import HighDLoader
from preprocessing.feature_extraction import extract_features

def main():
    # Set your dataset path relative to the project root
    dataset_path = "../data/highD-dataset-v1.0"  #

    print("Initializing data loader...")
    loader = HighDLoader(dataset_path=dataset_path)

    try:
        # Load recording 1 by default
        tracks, tracks_meta, recording_meta = loader.load_data(recording_id=1)
        loader.show_summary()
    except FileNotFoundError as e:
        print(e)
        return

    # Apply feature engineering
    print("Applying feature engineering...")
    features_df = extract_features(tracks)

    # Show preview of features
    print("\n===== Engineered Features Preview =====")
    print(features_df[['id', 'frame', 'speed', 'acceleration', 'laneId', 'delta_x', 'delta_v', 'time_gap']].head())
    print("=======================================\n")
    print("\nAll columns in tracks_df:\n", list(tracks.columns))
    print(" Feature engineering complete. Ready for model training or RL environment setup!")


if __name__ == "__main__":
    main()
