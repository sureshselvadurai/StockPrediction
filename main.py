from data_processor.data_loader import DataLoader
from data_processor.data_preprocessor import DataPreprocessor
from data_models.lstm_model import LSTMModel
from data_models.model_predictor import PredictModel
from config import csv_file, start_date, end_date, clearPrevious
from utils.utils import clear_folder, merge_csv_files, generate_report, save_error_log


def main():
    error_log = []
    clear_folder("data_output/") if clearPrevious else None
    clear_folder("data_processor/EDA/report/") if clearPrevious else None

    data_loader = DataLoader(csv_file)
    stock_data = data_loader.load_data(start_date, end_date)

    total_stocks = len(stock_data)
    for index, (symbol, data) in enumerate(stock_data.items(), start=1):
        try:
            # Data Preprocessing
            print(f"Processing stock - {symbol} ({index}/{total_stocks}) - {index / total_stocks * 100:.2f}% completed")
            data_preprocessor = DataPreprocessor(symbol, data)
            data = data_preprocessor.preprocess_data()
            model_features = data_preprocessor.get_model_features()

            # Create model
            model = LSTMModel(symbol, data, model_features)
            model.create_model()

            predict = PredictModel(symbol, data, model, model_features)
            predict.predict()

        except Exception as e:
            error_details = {
                'Symbol': symbol,
                'Error Message': str(e)
            }
            error_log.append(error_details)
            print(f"An error occurred for stock {symbol}: {str(e)}")

    merge_csv_files('data_output/report')
    merge_csv_files('data_output/model')
    generate_report()
    save_error_log(error_log)


if __name__ == "__main__":
    main()
