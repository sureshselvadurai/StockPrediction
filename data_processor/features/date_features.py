import pandas as pd


class DateFeatures:
    def __init__(self, data):
        self.data = data
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)

    def get_data(self):
        self.data['Date'] = self.data['Date'].astype(str)
        return self.data.copy()

    def generate_features(self):
        self.data['DayOfWeek'] = self.data['Date'].apply(lambda x: x.weekday())


