model_features = ['Close', 'DayOfWeek', 'Rolling_Std', 'EMA', 'Long_EMA', 'MACD', 'Gain', 'Loss']
model_target = 'Close'
constants = ['Date']

look_back = 3
epochs = 5
train_test_split = 0.8
days_to_predict = 30
csv_file = "data/stocks.csv"
start_date = "2023-01-01"
end_date = "2023-10-10"
is_plot = True
clearPrevious = False
