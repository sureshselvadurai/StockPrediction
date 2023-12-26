from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
from datetime import datetime, timedelta
from utils.utils import generate_close_timestamps
from preprocessing.data_preprocessor import DataPreprocessor
import pandas as pd
from utils.utils import create_dataset

model_features = ['Close']
model_target = 'Close'
days_to_predict = 10
look_back = 3
model_target = 'Close'


class PredictModel:
    def __init__(self, symbol, data, model):
        self.result_df = None
        self.dates_iterations = None
        self.scaled_data = None
        self.data = data
        self.symbol = symbol
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = model.get_model()

    def predict(self):
        self.scale_data()
        self.init_dates()
        self.iter_predict()
        self.plot_predictions()

    def scale_data(self):
        data = self.data.copy()
        for column in model_features:
            if data[column].dtype == 'O':
                data[column] = self.label_encoder.fit_transform(data[column])

        data[model_features] = self.scaler.fit_transform(data[model_features])
        model_columns = np.concatenate((model_features, [model_target]))[
            np.unique(np.concatenate((model_features, [model_target])), return_index=True)[1]]
        data = data[model_columns]
        self.scaled_data = data.dropna()

    def init_dates(self):

        start_date_str = self.data['Date'].iloc[-1]
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S%z')
        predict_to_date = start_date + timedelta(days=days_to_predict)
        self.dates_iterations = generate_close_timestamps(start_date, predict_to_date)

    def iter_predict(self):
        rolling_sequence = np.array([self.scaled_data[-look_back:].values])
        predict_df = self.data.copy()
        # add scale below

        for i in range(len(self.dates_iterations)):

            next_prediction_scaled = self.model.predict(rolling_sequence)[:, -1]
            next_prediction_scaled = next_prediction_scaled.reshape(-1, 1)
            next_prediction = self.scaler.inverse_transform(next_prediction_scaled)
            new_data = pd.DataFrame(next_prediction.reshape(-1, 1), columns=[model_target])
            new_data['Date'] = self.dates_iterations[i]

            predict_df = pd.concat([predict_df, new_data], ignore_index=True)
            predict_df = predict_df[model_features + [model_target]].dropna()

            data_preprocessor = DataPreprocessor(self.symbol, predict_df)
            predict_df = data_preprocessor.preprocess_data()

            rolling_sequence = np.array([predict_df.tail(look_back).values])

        df_predictions = predict_df.tail(len(self.dates_iterations))

        # Invert predictions to the original scale
        predictions = df_predictions.iloc[:, -1]
        features = df_predictions.iloc[:, :-1].values.tolist()
        dates = self.dates_iterations

        # Create a DataFrame to store the final results
        result_df = pd.DataFrame({
            'Tag': 'Prediction',
            'Prediction': predictions,
            'Features': features,
            'Time': dates
        })

        # Save the DataFrame to a CSV file
        result_df.to_csv(f"predict_data/{self.symbol}_predictions.csv", index=False)
        self.result_df = result_df

    def plot_predictions(self):
        plt = self.model.get_plot()
        plt.plot(self.result_df['Time'], self.result_df['Prediction'], label='Predicted (Rolling)', color='purple', linestyle='dashed')



