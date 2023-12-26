import pandas as pd
import yfinance as yf
import os

data_folder = 'stock_data'


def download_stock_data(stock_symbols, start_date, end_date):
    stock_data = {}
    for symbol in stock_symbols:

        file_path = f"{data_folder}/{symbol}_data.csv"

        if os.path.exists(file_path):
            data = pd.read_csv(file_path, parse_dates=True)
        else:
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            data.to_csv(file_path)

        stock_data[symbol] = data

    return stock_data


class DataLoader:

    def __init__(self, csv_file):
        self.csv_file = csv_file

    def load_data(self, start_date, end_date):
        # Load stock symbols from CSV file
        stock_symbols = self.load_stock_symbols()

        # Download stock data using yfinance
        stock_data = download_stock_data(stock_symbols, start_date, end_date)

        return stock_data

    def load_stock_symbols(self):
        df = pd.read_csv(self.csv_file)
        stock_symbols = df['Symbol'].tolist()
        return stock_symbols
