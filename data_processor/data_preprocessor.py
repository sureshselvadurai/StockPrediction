import pandas as pd
from data_processor.features.date_features import DateFeatures


class DataPreprocessor:
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data

    def preprocess_data(self):
        self.add_features()
        return self.data.copy()

    def add_features(self):

        date_features = DateFeatures(self.data)
        date_features.generate_features()
        self.data = date_features.get_data()
        self.data['Constant'] = 1
