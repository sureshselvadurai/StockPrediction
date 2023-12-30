import pandas as pd
from data_processor.features.features import Features
from data_processor.EDA.EDA import EDA


class DataPreprocessor:
    def __init__(self, symbol, data):
        self.eda = None
        self.symbol = symbol
        self.data = data

    def preprocess_data(self):
        self.add_features()
        return self.data.copy()

    def add_features(self):

        features = Features(self.data)
        features.generate_features()
        self.data = features.get_data()
        self.eda = EDA(self.data, self.symbol)
        self.eda.generateEDA()

    def get_model_features(self):
        return self.eda.get_features()
