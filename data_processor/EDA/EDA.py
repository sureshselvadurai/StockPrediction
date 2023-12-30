from config import model_features, model_target, correlation_bar


class EDA:
    def __init__(self, data, symbol):
        self.high_correlation = None
        self.data = data.copy()
        self.report = {}
        self.symbol = symbol

    def generateEDA(self):
        self.data = self.data.dropna()
        self.correlation()
        self.saveEDA()

    def correlation(self):
        df = self.data[model_features]
        correlation_matrix = df.corr()
        correlation_with_target = correlation_matrix[model_target].abs().sort_values(ascending=False)
        self.report["correlation"] = correlation_with_target
        self.high_correlation = correlation_with_target[correlation_with_target > correlation_bar].index.tolist()

    def saveEDA(self):
        for analysis in self.report.keys():
            self.report[analysis].to_csv(f"data_processor/EDA/report/{self.symbol}_{analysis}.csv")

    def get_features(self):
        return self.high_correlation
