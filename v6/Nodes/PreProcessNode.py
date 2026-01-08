import sys
LOGGING_ENABLED = '--debug' in sys.argv
"""
Preprocess Node
This will be used to preprocess data before it is sent to the Judge Node.
It must be non-trainable and purely mathematical.
will be used for both training and inference purposes

"""

from .BaseNode import BaseNode
import numpy as np

class PreProcessNode(BaseNode):
    def __init__(self, position):
        if LOGGING_ENABLED:
            print(f'[DEBUG] PreProcessNode initialized at position {position}')
        self.position = position

        # Frozen preprocessing parameters
        self.feature_order = []
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.category_maps = {}

    def process(self, data):
        """
        Deterministic preprocessing pipeline.
        Used identically in training and inference.
        """
        data = self.handle_missing_values(data)
        data = self.encode_categorical_features(data)
        data = self.normalize_data(data)
        return self.vectorize_features(data)

    # ===== Core transforms =====

    def vectorize_features(self, data):
        if not self.feature_order:
            self.feature_order = list(data.keys())
        return np.array([data.get(f, 0.0) for f in self.feature_order], dtype=float)

    def normalize_data(self, data):
        for k, v in data.items():
            if k in self.mins and k in self.maxs:
                denom = self.maxs[k] - self.mins[k]
                data[k] = (v - self.mins[k]) / denom if denom != 0 else 0.0
        return data

    def standardize_data(self, data):
        for k, v in data.items():
            if k in self.means and k in self.stds:
                data[k] = (v - self.means[k]) / self.stds[k] if self.stds[k] != 0 else 0.0
        return data

    # ===== Dataset-level setup (called once) =====

    def dataset_standardize(self, dataset):
        for k in dataset[0]:
            values = [row[k] for row in dataset if row[k] is not None]
            self.means[k] = np.mean(values)
            self.stds[k] = np.std(values) + 1e-8

    def dataset_normalize(self, dataset):
        for k in dataset[0]:
            values = [row[k] for row in dataset if row[k] is not None]
            self.mins[k] = min(values)
            self.maxs[k] = max(values)

    def handle_missing_values(self, data):
        for k, v in data.items():
            if v is None:
                data[k] = self.means.get(k, 0.0)
        return data

    def encode_categorical_features(self, data):
        for k, v in list(data.items()):
            if isinstance(v, str):
                if k not in self.category_maps:
                    self.category_maps[k] = {}
                if v not in self.category_maps[k]:
                    self.category_maps[k][v] = len(self.category_maps[k])
                data[k] = float(self.category_maps[k][v])
        return data