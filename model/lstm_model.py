import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt
from utils.utils import create_dataset

model_features = ['Close']
model_target = 'Close'
constants = ['Date']
train_test_split = 0.8
look_back = 3
epochs = 1


class LSTMModel:

    def __init__(self, symbol, data):
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
        self.scaler = MinMaxScaler(feature_range=(0, 1))

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
        train_x, train_y = create_dataset(train.values)
        test_x, test_y = create_dataset(test.values)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def fit_model(self):
        self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=1, verbose=2)
        self.model.save(f"generated_models/{self.symbol}_lstm_model.keras")

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
        train_predict = self.scaler.inverse_transform(
            np.hstack((self.train_x[:, -1, :-1], train_predict.reshape(-1, 1))))
        train_y = self.scaler.inverse_transform(np.hstack((self.train_x[:, -1, :-1], self.train_y.reshape(-1, 1))))

        test_predict = self.scaler.inverse_transform(np.hstack((self.test_x[:, -1, :-1], test_predict.reshape(-1, 1))))
        test_y = self.scaler.inverse_transform(np.hstack((self.test_x[:, -1, :-1], self.test_y.reshape(-1, 1))))

        # Calculate RMSE for training data
        train_score = np.sqrt(mean_squared_error(train_y[:, -1], train_predict[:, -1]))
        print("Train RMSE: {:.2f}".format(train_score))

        # Calculate RMSE for testing data
        test_score = np.sqrt(mean_squared_error(test_y[:, -1], test_predict[:, -1]))
        print("Test RMSE: {:.2f}".format(test_score))

        # Create a DataFrame to store the results
        columns = ['Tag', 'Prediction', 'Features', 'Time']
        train_df = pd.DataFrame({'Tag': 'Train',
                                 'Prediction': train_predict[:, -1],
                                 'Actual': train_y[:, -1],
                                 'Features': self.train_x[:, -1, :].tolist(),
                                 'Time': self.data['Date'][look_back - 1:look_back - 1 + len(train_y)]})

        test_df = pd.DataFrame({'Tag': 'Test',
                                'Prediction': test_predict[:, -1],
                                'Actual': test_y[:, -1],
                                'Features': self.test_x[:, -1, :].tolist(),
                                'Time': self.data['Date'][
                                        look_back - 1 + len(train_y) + look_back - 1:look_back - 1 + len(
                                            train_y) + look_back - 1 + len(test_y)]})

        # Concatenate train and test DataFrames
        result_df = pd.concat([train_df, test_df])

        # Save the DataFrame to a CSV file
        result_df.to_csv(f"model_predictions/{self.symbol}_prediction.csv", index=False)
        # self.plot_model(train_predict, test_predict)

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

    def plot_model(self, train_predict, test_predict):
        # Plotting training and testing data on the same plot

        plt.figure(figsize=(15, 9))
        train_dates = self.data['Date'][look_back:look_back + len(train_predict)]
        actual_train = self.data['Close'][look_back:look_back + len(train_predict)].values.reshape(-1, 1)
        predicted_train = train_predict[:, -1].reshape(-1, 1)
        plt.plot(train_dates, actual_train, label='Actual (Train)', color='blue')
        plt.plot(train_dates, predicted_train, label='Predicted (Train)', linestyle='dashed', color='orange')

        test_dates = self.data['Date'][look_back + len(train_predict) + look_back - 1:look_back + len(
            train_predict) + look_back - 1 + len(test_predict)]
        actual_test = self.data['Close'][look_back + len(train_predict) + look_back - 1:look_back + len(
            train_predict) + look_back - 1 + len(test_predict)].values.reshape(-1, 1)
        predicted_test = test_predict[:, -1].reshape(-1, 1)
        plt.plot(test_dates, actual_test, label='Actual (Test)', color='green')
        plt.plot(test_dates, predicted_test, label='Predicted (Test)', linestyle='dashed', color='red')

        plt.title(f'{self.symbol} LSTM Model Performance')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()

        plt.tight_layout()
        plt.show()
