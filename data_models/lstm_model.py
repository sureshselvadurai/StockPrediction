import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
from data_processor.data_preprocessor import DataPreprocessor
import os
from utils.utils import create_dataset, move_column_to_last
from utils.plot_data import Plot
from utils.utils import get_stock_data_yf, write_dict_to_csv
from config import model_target, train_test_split, look_back, epochs, is_plot, to_predict, constants
from data_processor.EDA.model_EDA import modelEDA
from data_processor.EDA.EDA import EDA
from sklearn.model_selection import GridSearchCV


class LSTMModel:

    def __init__(self, symbol, data, model_features):
        self.save_report = None
        self.plot = None
        self.scaled_data = None
        self.test_y = None
        self.test_x = None
        self.train_y = None
        self.train_x = None
        self.model = None

        self.data = data
        self.symbol = symbol
        self.model_features = model_features
        self.report = {}

        self.init_model()
        self.label_encoder = LabelEncoder()
        scale_key = {}
        for feature in self.model_features:
            scale_key[feature] = MinMaxScaler(feature_range=(0, 1))
        self.scaler = scale_key

    def init_model(self):
        self.report["symbol"] = self.symbol
        self.report["model_features"] = " | ".join(self.model_features)

        model = Sequential()
        model.add(LSTM(100, input_shape=(look_back, len(self.model_features))))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model

    def create_model(self):
        self.scale_data()
        self.generate_test_train()
        self.fit_model()
        self.generate_model_performance()

    def scale_data(self):
        data = self.data.copy()
        for column in self.model_features:
            if data[column].dtype == 'O':
                data[column] = self.label_encoder.fit_transform(data[column])

        model_columns = np.concatenate((self.model_features, [model_target]))[
            np.unique(np.concatenate((self.model_features, [model_target])), return_index=True)[1]]
        # data = data[model_columns]
        data = move_column_to_last(data, model_target)
        for feature in self.model_features:
            data[feature] = self.scaler[feature].fit_transform(data[feature].values.reshape(-1, 1))
        self.scaled_data = data.dropna()

    def generate_test_train(self):
        train_size = int(len(self.scaled_data) * train_test_split)
        train, test = self.scaled_data[0:train_size][self.model_features], self.scaled_data[train_size-look_back:len(self.scaled_data)][self.model_features]
        train_x, train_y = create_dataset(train.values, look_back)
        test_x, test_y = create_dataset(test.values, look_back)
        self.scaled_data = self.scaled_data.iloc[look_back:]

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def fit_model(self):
        self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=1, verbose=2)
        self.model.save(f"data_output/keras/{self.symbol}.keras")

        self.report['train_size'] = len(self.train_x)
        self.report['feature_size'] = len(self.model_features)
        eda = modelEDA(self.model, self.train_x, self.symbol, self.model_features, self.data)
        eda.generate_report()

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
        self.report["train_score"] = train_score

        # Calculate RMSE for testing data_processor
        test_score = np.sqrt(mean_squared_error(test_y[:, -1], test_predict[:, -1]))
        print("Test RMSE: {:.2f}".format(test_score))
        self.report["test_score"] = test_score

        # Create a DataFrame to store the results
        columns = ['Tag', 'Prediction', 'Features', 'Time']
        train_df = pd.DataFrame({'Tag': 'Train',
                                 'Prediction': train_predict[:, -1],
                                 'Actual': train_y[:, -1],
                                 'Features': self.train_x[:, :, :].tolist()
                                 })

        test_df = pd.DataFrame({'Tag': 'Test',
                                'Prediction': test_predict[:, -1],
                                'Actual': test_y[:, -1],
                                'Features': self.test_x[:, :, :].tolist()
                                })

        # Concatenate train and test DataFrames
        result_df = pd.concat([train_df, test_df])
        result_df['Time'] = self.scaled_data['Date'].tolist()
        self.save_report = result_df[['Tag', 'Prediction', 'Actual', 'Time']]

        self.plot_model(train_predict, test_predict)

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
        data['Date'] = data['Timestamp'].dt.strftime('%Y-%m-%d')

        file_path = f"data_output/stock_data/{self.symbol}_OOT.csv"

        if os.path.exists(file_path):
            actual_data = pd.read_csv(file_path)
        else:
            actual_data = get_stock_data_yf(data['Timestamp'].max(), data['Timestamp'].min(), self.symbol)
            actual_data.to_csv(file_path)
            actual_data = actual_data.reset_index()
            actual_data = actual_data.rename(columns={'index': 'Date'})

        actual_data['Date'] = actual_data['Date'].astype(str)
        actual_data['Timestamp'] = pd.to_datetime(actual_data['Date'].str[:-6], format='%Y-%m-%d %H:%M:%S')
        actual_data['Date'] = actual_data['Timestamp'].dt.strftime('%Y-%m-%d')
        if not to_predict:
            data = pd.merge(data, actual_data[['Close', 'Date']], how='inner', left_on='Date',
                            right_on='Date')
            data.rename(columns={'Close': 'Actual'}, inplace=True)
            oot_score = np.sqrt(mean_squared_error(data['Actual'], data['Prediction']))
            print("OOT RMSE: {:.2f}".format(oot_score))
            self.report["oot_score"] = oot_score
        else:
            data = pd.merge(data, actual_data[['Close', 'Date']], how='left', left_on='Date',
                            right_on='Date')
            data.rename(columns={'Close': 'Actual'}, inplace=True)

        data['Tag'] = 'Prediction'
        self.save_report = pd.concat([data[['Tag', 'Prediction', 'Actual', 'Time']], self.save_report])
        self.save_report["Stock"] = self.symbol
        self.save_report['Time'] = pd.to_datetime(self.save_report['Time'])

        # Sort by "Stock" and then by "Time" in descending order
        sorted_df = self.save_report.sort_values(by=['Stock', 'Time'], ascending=[True, False])

        sorted_df.to_csv(f"data_output/model/{self.symbol}.csv", index=False)

        self.plot.add_data_series(data['Time'], data['Prediction'], 'Predicted (Rolling)', 'purple', 'dashed')
        self.plot.add_data_series(data['Time'], data['Actual'], 'Actual (Rolling)', 'black')

        self.perform_oot_eda(data)
        write_dict_to_csv(self.report, f"data_output/report/{self.symbol}.csv")

        if is_plot:
            self.plot.show_plot(self.symbol)
            self.plot.save_plot(f"data_output/plot/{self.symbol}.png")

    def perform_oot_eda(self, data):
        data['Close'] = data['Prediction']
        data['Date'] = data['Time']
        data['Type'] = "OOT"
        whole_data = pd.concat([self.data[['Close', 'Date']], data[['Close', 'Date']]])
        whole_data['Close'] = whole_data['Close'].astype(float)

        data_preprocessor = DataPreprocessor(self.symbol, whole_data[['Date', 'Close']])

        data = data_preprocessor.preprocess_data()
        eda = EDA(data, self.symbol)
        is_oot_stable, unstable_oot_features = eda.feature_stability(data, 15, self.model_features)
        self.report['is_oot_stable'], self.report['unstable_oot_features'] = is_oot_stable, unstable_oot_features

    def get_model(self):
        return self.model

    def get_scales(self):
        return self.scaler, self.label_encoder
