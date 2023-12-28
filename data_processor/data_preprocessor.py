model_target = 'Close'


class DataPreprocessor:
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data

    def preprocess_data(self):
        self.add_features()
        return self.data.copy()

    def add_features(self):
        self.data['Constant'] = 1
