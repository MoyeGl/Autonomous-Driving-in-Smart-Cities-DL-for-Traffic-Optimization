# =============================================================================
# data_loader.py
# Author: Moye Nyuysoni
# Purpose: Load and validate highD dataset files
# =============================================================================

import os
import pandas as pd

class HighDLoader:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        # Initialize containers for data
        self.tracks = None
        self.tracks_meta = None
        self.recording_meta = None

    def _file_path(self, filename):

        return os.path.join(self.dataset_path, filename)

    def load_data(self, recording_id=1):

        rec_id = str(recording_id).zfill(2)

        tracks_file = f"{rec_id}_tracks.csv"
        tracks_meta_file = f"{rec_id}_tracksMeta.csv"
        recording_meta_file = f"{rec_id}_recordingMeta.csv"

        print(f"Loading data for recording {rec_id}...")

        # Validate and load each file
        self.tracks = self._load_csv(tracks_file)
        self.tracks_meta = self._load_csv(tracks_meta_file)
        self.recording_meta = self._load_csv(recording_meta_file)

        print("Data successfully loaded!")
        return self.tracks, self.tracks_meta, self.recording_meta

    def _load_csv(self, filename):

        file_path = self._file_path(filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ERROR: File not found -> {file_path}")

        print(f"Loading file: {file_path}")
        return pd.read_csv(file_path)

    def show_summary(self):

        print("\n===== DATA SUMMARY =====")
        if self.tracks is not None:
            print(f"Tracks: {len(self.tracks)} rows, {len(self.tracks.columns)} columns")
            print(self.tracks.head())
        if self.tracks_meta is not None:
            print(f"\nTracks Meta: {len(self.tracks_meta)} rows, {len(self.tracks_meta.columns)} columns")
            print(self.tracks_meta.head())
        if self.recording_meta is not None:
            print(f"\nRecording Meta Info:")
            print(self.recording_meta.head())
        print("========================\n")
