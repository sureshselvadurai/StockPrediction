import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
from utils.utils import create_dataset, move_column_to_last
from utils.plot_data import Plot
from utils.utils import get_stock_data_yf
from config import model_features, model_target, train_test_split, look_back, epochs, is_plot


class LSTMModel:

    def __init__(self, symbol, data):
        self.plot = None
        self.scaled_data = None
        self.test_y = None
        self.test_x = None
        self.train_y = None
        self.train_x = None
        self.model = None

        self.data = data
        self.symbol = symbol
        self.init_model()
        self.label_encoder = LabelEncoder()
        scale_key = {}
        for feature in model_features:
            scale_key[feature] = MinMaxScaler(feature_range=(0, 1))
        self.scaler = scale_key

    def get_model(self):
        return self.model

    def get_scales(self):
        return self.scaler, self.label_encoder

    def create_model(self):
        self.scale_data()
        self.generate_test_train()
        self.fit_model()
        self.generate_model_performance()

    def generate_test_train(self):
        train_size = int(len(self.scaled_data) * train_test_split)
        train, test = self.scaled_data[0:train_size], self.scaled_data[train_size:len(self.scaled_data)]
        train_x, train_y = create_dataset(train.values, look_back)
        test_x, test_y = create_dataset(test.values, look_back)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def fit_model(self):
        self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=1, verbose=2)
        self.model.save(f"data_models/{self.symbol}_lstm_model.keras")

    def init_model(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, len(model_features))))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model

    def generate_model_performance(self):
        train_predict = self.model.predict(self.train_x)
        test_predict = self.model.predict(self.test_x)

        # Invert predictions to original scale
        train_predict = self.scaler[model_target].inverse_transform(train_predict.reshape(-1, 1))
        train_y = self.scaler[model_target].inverse_transform(self.train_y.reshape(-1, 1))

        test_predict = self.scaler[model_target].inverse_transform(test_predict.reshape(-1, 1))
        test_y = self.scaler[model_target].inverse_transform(self.test_y.reshape(-1, 1))

        # Calculate RMSE for training data_processor
        train_score = np.sqrt(mean_squared_error(train_y[:, -1], train_predict[:, -1]))
        print("Train RMSE: {:.2f}".format(train_score))

        # Calculate RMSE for testing data_processor
        test_score = np.sqrt(mean_squared_error(test_y[:, -1], test_predict[:, -1]))
        print("Test RMSE: {:.2f}".format(test_score))

        # Create a DataFrame to store the results
        columns = ['Tag', 'Prediction', 'Features', 'Time']
        train_df = pd.DataFrame({'Tag': 'Train',
                                 'Prediction': train_predict[:, -1],
                                 'Actual': train_y[:, -1],
                                 'Features': self.train_x[:, :, :].tolist(),
                                 'Time': self.data['Date'][look_back - 1:look_back - 1 + len(train_y)]})

        test_df = pd.DataFrame({'Tag': 'Test',
                                'Prediction': test_predict[:, -1],
                                'Actual': test_y[:, -1],
                                'Features': self.test_x[:, :, :].tolist(),
                                'Time': self.data['Date'][
                                        look_back - 1 + len(train_y) + look_back - 1:look_back - 1 + len(
                                            train_y) + look_back - 1 + len(test_y)]})

        # Concatenate train and test DataFrames
        result_df = pd.concat([train_df, test_df])

        # Save the DataFrame to a CSV file
        result_df.to_csv(f"data_output/{self.symbol}_test_train_prediction.csv", index=False)
        self.plot_model(train_predict, test_predict)

    def scale_data(self):
        data = self.data.copy()
        for column in model_features:
            if data[column].dtype == 'O':
                data[column] = self.label_encoder.fit_transform(data[column])

        model_columns = np.concatenate((model_features, [model_target]))[
            np.unique(np.concatenate((model_features, [model_target])), return_index=True)[1]]
        data = data[model_columns]
        data = move_column_to_last(data, model_target)
        for feature in model_features:
            data[feature] = self.scaler[feature].fit_transform(data[feature].values.reshape(-1, 1))
        self.scaled_data = data.dropna()

    def plot_model(self, train_predict, test_predict):
        # Plotting training and testing data_processor on the same plot
        self.plot = Plot()

        train_dates = self.data['Date'][look_back:look_back + len(train_predict)]
        actual_train = self.data['Close'][look_back:look_back + len(train_predict)].values.reshape(-1, 1)
        predicted_train = train_predict[:, -1].reshape(-1, 1)

        self.plot.add_data_series(train_dates, actual_train, 'Actual (Train)', 'blue')
        self.plot.add_data_series(train_dates, predicted_train, 'Predicted (Train)', 'orange', 'dashed')

        test_dates = self.data['Date'][look_back + len(train_predict) + look_back - 1:look_back + len(
            train_predict) + look_back - 1 + len(test_predict)]
        actual_test = self.data['Close'][look_back + len(train_predict) + look_back - 1:look_back + len(
            train_predict) + look_back - 1 + len(test_predict)].values.reshape(-1, 1)
        predicted_test = test_predict[:, -1].reshape(-1, 1)

        self.plot.add_data_series(test_dates, actual_test, 'Actual (Test)', 'green')
        self.plot.add_data_series(test_dates, predicted_test, 'Predicted (Test)', 'red', 'dashed')

    def plot_predictions_train(self, data):

        data['Timestamp'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S%z')

        file_path = f"data_output/{self.symbol}_data_oot.csv"

        if os.path.exists(file_path):
            actual_data = pd.read_csv(file_path, parse_dates=True)
        else:
            actual_data = get_stock_data_yf(data['Timestamp'].max(), data['Timestamp'].min(), self.symbol)
            actual_data.to_csv(file_path)
            actual_data = actual_data.reset_index()
            actual_data = actual_data.rename(columns={'index': 'Date'})
            actual_data['Date'] = actual_data['Date'].astype(str)

        actual_data['Timestamp'] = pd.to_datetime(actual_data['Date'], format='%Y-%m-%d %H:%M:%S%z')
        data = pd.merge(data, actual_data[['Timestamp', 'Close']], how='inner', left_on='Timestamp',
                        right_on='Timestamp')
        data.rename(columns={'Close': 'Actual'}, inplace=True)
        data[['Time', 'Prediction', 'Actual']].to_csv(f"data_output/{self.symbol}_oot_prediction.csv")

        oot_score = np.sqrt(mean_squared_error(data['Actual'], data['Prediction']))
        print("OOT RMSE: {:.2f}".format(oot_score))

        self.plot.add_data_series(data['Time'], data['Prediction'], 'Predicted (Rolling)', 'purple', 'dashed')
        self.plot.add_data_series(data['Time'], data['Actual'], 'Actual (Rolling)', 'black')
        if is_plot:
            self.plot.show_plot(self.symbol)
