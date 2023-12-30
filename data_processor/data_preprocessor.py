import pandas as pd
from data_processor.features.features import Features


class DataPreprocessor:
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data

    def preprocess_data(self):
        self.add_features()
        return self.data.copy()

    def add_features(self):

        features = Features(self.data)
        features.generate_features()
        self.data = features.get_data()
