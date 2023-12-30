from data_processor.data_loader import DataLoader
from data_processor.data_preprocessor import DataPreprocessor
from data_models.lstm_model import LSTMModel
from data_models.model_predictor import PredictModel
from config import csv_file, start_date, end_date, clearPrevious
from utils.utils import clear_folder


def main():
    data_loader = DataLoader(csv_file)
    stock_data = data_loader.load_data(start_date, end_date)
    clear_folder("data_output/") if clearPrevious else None

    for symbol, data in stock_data.items():
        # Data Preprocessing
        data_preprocessor = DataPreprocessor(symbol, data)
        data = data_preprocessor.preprocess_data()
        model_features = data_preprocessor.get_model_features()

        # Create model
        model = LSTMModel(symbol, data, model_features)
        model.create_model()

        predict = PredictModel(symbol, data, model, model_features)
        predict.predict()


if __name__ == "__main__":
    main()
