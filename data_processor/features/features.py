import pandas as pd


class Features:
    def __init__(self, data):
        self.data = data
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)

    def get_data(self):
        self.data['Date'] = self.data['Date'].astype(str)
        return self.data.copy()

    def generate_features(self):
        self.data['DayOfWeek'] = self.data['Date'].apply(lambda x: x.weekday())
        self.data['Rolling_Mean'] = self.data['Close'].rolling(window=10).mean()
        self.data['Rolling_Std'] = self.data['Close'].rolling(window=10).std()
        self.data['Price_RoC'] = self.data['Close'].pct_change()
        self.generate_ema_features()

        self.data['Price_Delta'] = self.data['Close'].diff(1)
        self.data['Gain'] = self.data['Price_Delta'].apply(lambda x: x if x > 0 else 0)
        self.data['Loss'] = self.data['Price_Delta'].apply(lambda x: abs(x) if x < 0 else 0)
        average_gain = self.data['Gain'].rolling(window=14).mean()
        average_loss = self.data['Loss'].rolling(window=14).mean()

        # Calculate RSI
        self.data['RS'] = average_gain / average_loss
        self.data['RSI'] = 100 - (100 / (1 + self.data['RS']))


    def generate_ema_features(self):
        short_window = 12
        long_window = 26
        self.data['EMA'] = self.data['Close'].ewm(span=10, adjust=False).mean()
        self.data['Short_EMA'] = self.data['Close'].ewm(span=short_window, adjust=False).mean()
        self.data['Long_EMA'] = self.data['Close'].ewm(span=long_window, adjust=False).mean()

        self.data['MACD'] = self.data['Short_EMA'] - self.data['Long_EMA']
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()


