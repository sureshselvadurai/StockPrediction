from data_processor.data_loader import DataLoader
from data_processor.data_preprocessor import DataPreprocessor
from data_models.lstm_model import LSTMModel
from data_models.model_predictor import PredictModel


def main():
    csv_file = "data/stocks.csv"
    start_date = "2023-01-01"  # Start date
    end_date = "2023-12-10"  # End date

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
