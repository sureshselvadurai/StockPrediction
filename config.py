model_features = ['Loss', 'Gain', 'Close', 'DayOfWeek', 'Rolling_Mean', 'Rolling_Std', 'Price_RoC', 'EMA', 'Short_EMA', 'Long_EMA', 'MACD', 'Signal_Line']
model_target = 'Close'
constants = ['Date']

correlation_bar = 0.7
look_back = 3
epochs = 5
train_test_split = 0.8

days_to_predict = 15
start_date = "2023-01-01"
end_date = "2023-12-01"
clearPrevious = True

is_plot = True
csv_file = "data/stocks.csv"
