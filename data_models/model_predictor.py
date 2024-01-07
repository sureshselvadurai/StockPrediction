import numpy as np
from datetime import datetime, timedelta
from utils.utils import generate_close_timestamps
from data_processor.data_preprocessor import DataPreprocessor
import pandas as pd
from tensorflow.keras.models import load_model
from utils.utils import str_list
from utils.utils import move_column_to_last
from config import model_target, days_to_predict, look_back, constants


class PredictModel:
    def __init__(self, symbol, data, model, model_features):
        self.result_df = None
        self.dates_iterations = None
        self.scaled_data = None
        self.data = data
        self.symbol = symbol
        self.train_model = model
        self.model_features = model_features
        self.scaler, self.label_encoder = model.get_scales()
        self.model = self.model = load_model(f"data_output/keras/{self.symbol}.keras")

    def predict(self):
        self.scaled_data = self.scale_data(self.data.copy())
        self.init_dates()
        self.iter_predict()
        self.plot_predictions()

    def scale_data(self, data):
        for column in self.model_features:
            if data[column].dtype == 'O':
                data[column] = self.label_encoder.fit_transform(data[column])

        model_columns = np.concatenate((self.model_features, [model_target]))[
            np.unique(np.concatenate((self.model_features, [model_target])), return_index=True)[1]]
        data = data[model_columns]
        data = move_column_to_last(data, model_target)

        for feature in self.model_features:
            data[feature] = self.scaler[feature].fit_transform(data[feature].values.reshape(-1, 1))

        return data.dropna()

    def init_dates(self):

        start_date_str = self.data['Date'].iloc[-1]
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S%z')
        predict_to_date = start_date + timedelta(days=days_to_predict)
        print("Predicted Until  : " + str(predict_to_date))
        self.dates_iterations = generate_close_timestamps(start_date, predict_to_date)

    def plot_predictions(self):
        self.train_model.plot_predictions_train(self.result_df)

    def preprocess_scale_tail(self, predict_df):
        data_preprocessor = DataPreprocessor(self.symbol, predict_df)
        data = data_preprocessor.preprocess_data()
        data = self.scale_data(data)
        return np.array([data.tail(look_back).values])

    def iter_predict(self):
        predict_df = self.data.copy()
        features = []
        prediction = []

        for date in self.dates_iterations:
            model_data = self.preprocess_scale_tail(predict_df)

            next_prediction_scaled = self.model.predict(model_data)[:, -1]
            oot_predict = self.scaler[model_target].inverse_transform(next_prediction_scaled.reshape(-1, 1))

            new_data = pd.DataFrame(oot_predict.reshape(-1, 1), columns=[model_target])
            new_data['Date'] = date

            predict_df = pd.concat([predict_df, new_data], ignore_index=True)
            predict_df = predict_df[self.model_features+constants]
            features.append(model_data)
            prediction.append(oot_predict)

        self.save_predictions(features, prediction)

    def save_predictions(self, features, prediction):

        predictions = str_list(prediction, True)
        features = str_list(features)
        dates = self.dates_iterations

        # Create a DataFrame to store the final results
        result_df = pd.DataFrame({
            'Tag': 'Prediction',
            'Features': features,
            'Time': dates,
            'Prediction': predictions
        })
        self.result_df = result_df
