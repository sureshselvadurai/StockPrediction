import numpy as np
import pandas as pd
import datetime
from utilities.utils import generate_close_timestamps
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv(f"data/raw/apple_processed.csv")

# Use only relevant columns for modeling
features = ['lag_1', 'DayOfWeek']
target = 'target_scaled'
df = df[['Timestamp'] + features + [target]]

# Drop rows with missing values
df = df.dropna()

# Convert categorical columns to integer categories
label_encoder = LabelEncoder()
for column in features:
    if df[column].dtype == 'O':  # 'O' stands for object (string)
        df[column] = label_encoder.fit_transform(df[column])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[features + [target]])

# Create lagged feature 'lag_1'
df_scaled = pd.DataFrame(df_scaled, columns=features + [target])
df_scaled['lag_1'] = df_scaled[target].shift(1)
df_scaled = df_scaled.dropna()

# Split the data into training and testing sets
train_size = int(len(df_scaled) * 0.8)
train, test = df_scaled[0:train_size], df_scaled[train_size:len(df_scaled)]


# Convert the data to the format expected by LSTM (numpy arrays)
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY)


look_back = 3  # Number of time steps to look back
trainX, trainY = create_dataset(train.values, look_back)
testX, testY = create_dataset(test.values, look_back)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, len(features) + 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

# Save the trained model
model.save(f"models/apple_lstm_model.keras")

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions to original scale
trainPredict = scaler.inverse_transform(np.hstack((trainX[:, -1, :-1], trainPredict.reshape(-1, 1))))
trainY = scaler.inverse_transform(np.hstack((trainX[:, -1, :-1], trainY.reshape(-1, 1))))

testPredict = scaler.inverse_transform(np.hstack((testX[:, -1, :-1], testPredict.reshape(-1, 1))))
testY = scaler.inverse_transform(np.hstack((testX[:, -1, :-1], testY.reshape(-1, 1))))

# Calculate RMSE
trainScore = np.sqrt(mean_squared_error(trainY[:, -1], trainPredict[:, -1]))
testScore = np.sqrt(mean_squared_error(testY[:, -1], testPredict[:, -1]))

# Create a DataFrame to store the results
columns = ['Tag', 'Prediction', 'Features', 'Time']
train_df = pd.DataFrame({'Tag': 'Train',
                         'Prediction': trainPredict[:, -1],
                         'Features': trainX[:, -1, :-1].tolist(),
                         'Time': df['Timestamp'][look_back - 1:look_back - 1 + len(trainY)]})

test_df = pd.DataFrame({'Tag': 'Test',
                        'Prediction': testPredict[:, -1],
                        'Features': testX[:, -1, :-1].tolist(),
                        'Time': df['Timestamp'][look_back - 1 + len(trainY) + look_back - 1:look_back - 1 + len(
                            trainY) + look_back - 1 + len(testY)]})

# Concatenate train and test DataFrames
result_df = pd.concat([train_df, test_df])

# Save the DataFrame to a CSV file
result_df.to_csv(f"data/raw/apple_prediction.csv", index=False)

# Plot the results
# plt.figure(figsize=(14, 6))
#
# # Plot training data
# plt.plot(df['Timestamp'][look_back - 1:look_back - 1 + len(trainY)], trainY[:, -1], label='Actual (Train)', color='blue')
# plt.plot(df['Timestamp'][look_back - 1:look_back - 1 + len(trainY)], trainPredict[:, -1], label='Predicted (Train)', color='red')
#
# # Plot testing data
# plt.plot(df['Timestamp'][look_back - 1 + len(trainY) + look_back - 1:look_back - 1 + len(trainY) + look_back - 1 + len(testY)], testY[:, -1], label='Actual (Test)', color='green')
# plt.plot(df['Timestamp'][look_back - 1 + len(trainY) + look_back - 1:look_back - 1 + len(trainY) + look_back - 1 + len(testY)], testPredict[:, -1], label='Predicted (Test)', color='orange')
#
# plt.title('LSTM Model for Stock Price Prediction')
# plt.xlabel('Timestamp')
# plt.ylabel('Scaled Stock Price')
# plt.legend()
# plt.show()

# separate code



# Extract the last timestamp from the DataFrame
last_timestamp = df['Timestamp'].iloc[-1]

# Convert the timestamp string to a datetime object
last_date = datetime.datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S')

# Calculate the number of days to predict
predict_to_date = datetime.datetime(2023, 3, 25)
dates_iterations = generate_close_timestamps(last_timestamp, predict_to_date.strftime("%Y-%m-%d %H:%M:%S"))
# Perform rolling predictions
# Perform rolling predictions

rolling_sequence = np.array([df_scaled[-look_back:].values])
predict_df = df_scaled.copy()

for i in range(len(dates_iterations)):
    # Predict the next value based on the rolling sequence
    next_prediction = model.predict(rolling_sequence)[:, -1]

    new_data = pd.DataFrame(next_prediction.reshape(-1, 1), columns=[target])

    new_data['Timestamp'] = dates_iterations[i]
    new_data['DayOfWeek'] = pd.to_datetime(new_data['Timestamp']).dt.dayofweek

    # # Convert categorical columns to integer categories
    # for column in features:
    #     if new_data[column].dtype == 'O':  # 'O' stands for object (string)
    #         new_data[column] = label_encoder.fit_transform(new_data[column])

    predict_df = pd.concat([predict_df, new_data], ignore_index=True)
    predict_df['lag_1'] = predict_df[target].shift(1)
    predict_df = predict_df[features + [target]].dropna()

    rolling_sequence = np.array([predict_df[features + [target]].tail(look_back).values])

# Concatenate DataFrames into a single DataFrame
df_predictions = predict_df.tail(len(dates_iterations))
# Invert normalization for rolling predictions

# Invert predictions to the original scale
predictions = df_predictions.iloc[:, -1]
features = df_predictions.iloc[:, :-1].values.tolist()
dates = dates_iterations
# Create a DataFrame to store the final results
result_df = pd.DataFrame({
    'Tag': 'Prediction',
    'Prediction': predictions,
    'Features': df_predictions.iloc[:, :-1].values.tolist(),  # Adjust here
    'Time': dates
})

# Save the DataFrame to a CSV file
result_df.to_csv(f"predict/apple_predictions.csv", index=False)

# Plot the results
plt.figure(figsize=(14, 6))

# Plot training data
plt.plot(df['Timestamp'][look_back - 1:look_back - 1 + len(trainY)], trainY[:, -1], label='Actual (Train)',
         color='blue')
plt.plot(df['Timestamp'][look_back - 1:look_back - 1 + len(trainY)], trainPredict[:, -1], label='Predicted (Train)',
         color='red')

# Plot testing data
plt.plot(df['Timestamp'][
         look_back - 1 + len(trainY) + look_back - 1:look_back - 1 + len(trainY) + look_back - 1 + len(testY)],
         testY[:, -1], label='Actual (Test)', color='green')
plt.plot(df['Timestamp'][
         look_back - 1 + len(trainY) + look_back - 1:look_back - 1 + len(trainY) + look_back - 1 + len(testY)],
         testPredict[:, -1], label='Predicted (Test)', color='orange')

# Plot rolling predictions
plt.plot(result_df['Time'], result_df['Prediction'], label='Predicted (Rolling)', color='purple', linestyle='dashed')

plt.title('LSTM Model for Stock Price Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Scaled Stock Price')
plt.legend()
plt.show()
