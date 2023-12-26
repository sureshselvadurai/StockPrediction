model_target = 'Close'


def add_features():
    pass


class DataPreprocessor:
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data

    def preprocess_data(self):
        add_features()
        return self.data.copy()
