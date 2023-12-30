import pandas as pd
import numpy as np

class Features:
    def __init__(self, data):
        self.data = data
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)

    def get_data(self):
        self.data['Date'] = self.data['Date'].astype(str)
        return self.data.copy()

    def generate_features(self):
        self.data['DayOfWeek'] = self.data['Date'].apply(lambda x: x.weekday())

        # Mean
        self.data['Rolling_Mean'] = self.data['Close'].rolling(window=10).mean()
        self.data['Rolling_Std'] = self.data['Close'].rolling(window=10).std()
        self.data['Price_RoC'] = self.data['Close'].pct_change()
        self.generate_ema_features()

        # Change in price
        self.data['Price_Delta'] = self.data['Close'].diff(1)
        self.data['Gain'] = self.data['Price_Delta'].apply(lambda x: x if x > 0 else 0)
        self.data['Loss'] = self.data['Price_Delta'].apply(lambda x: abs(x) if x < 0 else 0)
        average_gain = self.data['Gain'].rolling(window=14).mean()
        average_loss = self.data['Loss'].rolling(window=14).mean()

        # Calculate RSI
        self.data['RS'] = average_gain / average_loss
        self.data['RSI'] = 100 - (100 / (1 + self.data['RS']))

        # Bollinger Bands
        self.data['Upper_Band'] = self.data['Rolling_Mean'] + 2 * self.data['Rolling_Std']
        self.data['Lower_Band'] = self.data['Rolling_Mean'] - 2 * self.data['Rolling_Std']

        # Z-Score
        self.data['Z_Score'] = (self.data['Close'] - self.data['Rolling_Mean']) / self.data['Rolling_Std']

        # Cumulative Returns
        self.data['Cumulative_Returns'] = (1 + self.data['Price_RoC']).cumprod()

        # Additional Features based on Closing Price (Window not greater than 20)
        self.data['Log_Returns'] = self.data['Close'].pct_change().apply(lambda x: np.log(1 + x))  # Log returns
        self.data['Moving_Average_5'] = self.data['Close'].rolling(window=5).mean()  # 5-day moving average
        self.data['Moving_Average_10'] = self.data['Close'].rolling(window=10).mean()  # 10-day moving average
        self.data['Moving_Average_15'] = self.data['Close'].rolling(window=15).mean()  # 15-day moving average
        self.data['Price_RoC_5'] = self.data['Close'].pct_change(periods=5)  # 5-day percentage return
        self.data['Price_RoC_10'] = self.data['Close'].pct_change(periods=10)  # 10-day percentage return
        self.data['Price_RoC_15'] = self.data['Close'].pct_change(periods=15)  # 15-day percentage return
        self.data['Price_RoC_30'] = self.data['Close'].pct_change(periods=30)  # 10-day percentage return
        self.data['Price_RoC_45'] = self.data['Close'].pct_change(periods=45)  # 15-day percentage return
        self.data['Price_RoC_60'] = self.data['Close'].pct_change(periods=60)  # 15-day percentage return

        # Additional Features
        self.data['Std_Dev_RoC_5'] = self.data['Price_RoC_5'].rolling(window=5).std()  # Standard deviation of 5-day RoC
        self.data['Std_Dev_RoC_10'] = self.data['Price_RoC_10'].rolling(window=10).std()  # Standard deviation of 10-day RoC
        self.data['Expanding_Mean'] = self.data['Close'].expanding().mean()  # Expanding mean
        self.data['Expanding_Std'] = self.data['Close'].expanding().std()  # Expanding standard deviation

        # More Features
        self.data['Price_RoC_3'] = self.data['Close'].pct_change(periods=3)  # 3-day percentage return
        self.data['Price_RoC_7'] = self.data['Close'].pct_change(periods=7)  # 7-day percentage return
        self.data['Price_RoC_12'] = self.data['Close'].pct_change(periods=12)  # 12-day percentage return
        self.data['Moving_Average_3'] = self.data['Close'].rolling(window=3).mean()  # 3-day moving average

        # Additional Mathematical Indicators
        self.data['Square_Root_Close'] = np.sqrt(self.data['Close'])  # Square root of Close price
        self.data['Log_Close'] = np.log1p(self.data['Close'])  # Natural logarithm of Close price
        self.data['Close_Squared'] = self.data['Close'] ** 2  # Square of Close price
        self.data['Close_Cubed'] = self.data['Close'] ** 3  # Cube of Close price
        self.data['Close_Inverse'] = 1 / self.data['Close']  # Inverse of Close price

        # Trigonometric Functions
        self.data['Sin_Close'] = np.sin(self.data['Close'])
        self.data['Cos_Close'] = np.cos(self.data['Close'])
        self.data['Tan_Close'] = np.tan(self.data['Close'])

    def generate_ema_features(self):
        short_window = 12
        long_window = 26
        self.data['EMA'] = self.data['Close'].ewm(span=10, adjust=False).mean()
        self.data['Short_EMA'] = self.data['Close'].ewm(span=short_window, adjust=False).mean()
        self.data['Long_EMA'] = self.data['Close'].ewm(span=long_window, adjust=False).mean()

        self.data['MACD'] = self.data['Short_EMA'] - self.data['Long_EMA']
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
