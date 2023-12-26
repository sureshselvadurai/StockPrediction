from data.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor
from model.lstm_model import LSTMModel
from predict_model.model_predictor import PredictModel


def main():
    csv_file = "data_files/stocks.csv"
    start_date = "2023-01-01"  # Start date
    end_date = "2023-12-31"  # End date

    data_loader = DataLoader(csv_file)
    stock_data = data_loader.load_data(start_date, end_date)

    for symbol, data in stock_data.items():
        # Data Preprocessing
        data_preprocessor = DataPreprocessor(symbol, data)
        data = data_preprocessor.preprocess_data()

        # Create model
        model = LSTMModel(symbol, data)
        model.create_model()

        predict = PredictModel(symbol, data, model)
        predict.predict()


if __name__ == "__main__":
    main()
