from config import model_features, model_target, correlation_bar
from utils.utils import round_psi
import pandas as pd


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
        is_stable, unstable_features = self.feature_stability(self.data)
        if is_stable:
            return self.high_correlation
        else:
            return [element for element in self.high_correlation if element not in unstable_features]

    def feature_stability(self, data, period=15, features=[]):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
        data['15_Day_Period'] = (data['Date'] - data['Date'].min()).dt.days // period

        # Calculate the CSI for each feature over 15-day periods
        # Calculate the CSI for each feature over 15-day periods
        csi_data = []
        features = self.high_correlation if len(features) == 0 else features
        for feature in features:
            csi_values = []
            for period in data['15_Day_Period'].unique():
                subset = data[data['15_Day_Period'] == period]
                csi = subset[feature].std() / subset[feature].mean()
                csi_values.append(csi)
            csi_data.append(csi_values)

        # Create a DataFrame with CSI values
        csi_df = pd.DataFrame(csi_data, index=features).T
        csi_df.map(round_psi).to_csv(f"data_processor/EDA/report/{self.symbol}_CSI.csv")
        unstable_features = csi_df.columns[csi_df.max() > 0.1].tolist()
        if len(unstable_features) > 0:
            return False, unstable_features
        else:
            return True, []

